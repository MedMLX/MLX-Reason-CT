"""FP32 Metal Primus, multimodal projection and Qwen3.5 hybrid decoding.

Source tensor layouts and the qualified FP32 arithmetic are preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from nv_reason_ct_mlx import gated_delta
from nv_reason_ct_mlx.chunk_delta import chunk_delta
from nv_reason_ct_mlx.errors import InvalidInputError
from nv_reason_ct_mlx.mlx_weights import SHARDS
from nv_reason_ct_mlx.runtime import import_mlx

mx: Any = import_mlx()
nn: Any = import_module("mlx.nn")


@dataclass
class DecoderCache:
    """One request's full-attention KV and linear-attention convolution/state."""

    layers: list[dict[str, Any]] = field(default_factory=lambda: [{} for _ in range(32)])
    offset: int = 0

    @property
    def nbytes(self) -> int:
        return sum(int(array.nbytes) for layer in self.layers for array in layer.values())

    def arrays(self) -> list[Any]:
        return [array for layer in self.layers for array in layer.values()]


class NativeModel:
    """Pinned 3D network with bounded attention and feed-forward workspaces."""

    def __init__(self, model_dir: Path) -> None:
        self.dtype = mx.float32
        self.config = json.loads((model_dir / "config.json").read_text())
        self.weights: dict[str, Any] = {}
        for name in SHARDS:
            self.weights.update(mx.load(str(model_dir / "mlx" / name)))
        for name, value in self.weights.items():
            if value.dtype != mx.float32:
                raise InvalidInputError(f"Native cache must store FP32 tensors: {name}")
            self.weights[name] = value.astype(self.dtype)
        mx.eval(*self.weights.values())
        self.delta = gated_delta
        self.eos_ids = tuple(
            json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"]
        )
        coords = np.indices((24, 24, 24), dtype=np.float32).reshape(3, -1).T
        inv = 1 / np.power(np.float32(10000), np.arange(12, dtype=np.float32) / 12)
        phase = (coords[..., None] * inv).reshape(13824, 36).repeat(2, axis=-1)
        self.vision_sin = mx.array(np.sin(phase))
        self.vision_cos = mx.array(np.cos(phase))
        # These constants are built on CPU: Metal pow differs at six frequencies.
        self.text_inv_freq = mx.array(
            1 / np.power(np.float32(10000000), np.arange(0, 64, 2, dtype=np.float32) / 64)
        )

    @staticmethod
    def silu(x: Any) -> Any:
        fp = x.astype(mx.float32)
        # Direct division matches ATen SiLU; multiplying sigmoid rounds twice.
        return (fp / (1 + mx.exp(-fp))).astype(x.dtype)

    @staticmethod
    def sigmoid(x: Any) -> Any:
        return (1 / (1 + mx.exp(-x.astype(mx.float32)))).astype(x.dtype)

    @staticmethod
    def gelu(x: Any) -> Any:
        return nn.gelu(x.astype(mx.float32)).astype(x.dtype)

    def linear(self, x: Any, name: str) -> Any:
        weight = self.weights[name + ".weight"]
        bias = self.weights.get(name + ".bias")
        return x @ weight.T if bias is None else mx.addmm(bias, x, weight.T)

    def norm(self, x: Any, name: str, eps: float = 1e-05) -> Any:
        dtype = x.dtype
        return mx.fast.layer_norm(
            x.astype(mx.float32),
            self.weights[name + ".weight"].astype(mx.float32),
            self.weights[name + ".bias"].astype(mx.float32),
            eps,
        ).astype(dtype)

    def rms(self, x: Any, name: str) -> Any:
        fp = x.astype(mx.float32)
        value = fp * mx.rsqrt(mx.mean(fp * fp, axis=-1, keepdims=True) + 1e-06)
        dtype = x.dtype
        return (value * (1 + self.weights[name + ".weight"].astype(mx.float32))).astype(dtype)

    @staticmethod
    def attention(q: Any, k: Any, v: Any, *, causal: bool, offset: int = 0) -> Any:
        outputs: list[Any] = []
        for start in range(0, q.shape[2], 256):
            end = min(start + 256, q.shape[2])
            mask = None
            if causal:
                mask = (
                    mx.arange(k.shape[2])[None, :]
                    <= mx.arange(offset + start, offset + end)[:, None]
                )
            result = mx.fast.scaled_dot_product_attention(
                q[:, :, start:end], k, v, scale=q.shape[-1] ** (-0.5), mask=mask
            )
            mx.eval(result)
            outputs.append(result)
        return mx.concatenate(outputs, axis=2)

    def vision(self, pixels: Any) -> tuple[Any, Any]:
        if pixels.shape != (1, 1, 192, 192, 192):
            raise InvalidInputError("NV-Reason-CT vision expects a single 192-cubed CZYX CT")
        prefix = "model.vision3d.sub_vision"
        x = pixels.astype(self.dtype).reshape(1, 1, 24, 8, 24, 8, 24, 8)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(1, 13824, 512)
        weight = self.weights[prefix + ".down_projection.proj.weight"].reshape(864, 512)
        x = x @ weight.T + self.weights[prefix + ".down_projection.proj.bias"]
        for i in range(16):
            p = f"{prefix}.eva.blocks.{i}"
            h = self.norm(x, p + ".norm1")
            q, k, v = [
                self.linear(h, p + f".attn.{s}_proj").reshape(1, -1, 12, 72).transpose(0, 2, 1, 3)
                for s in ("q", "k", "v")
            ]

            def rotate(t: Any) -> Any:
                dtype = t.dtype
                t = t.astype(mx.float32)
                pairs = t.reshape(*t.shape[:-1], 36, 2)
                rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(t.shape)
                return (t * self.vision_cos + rotated * self.vision_sin).astype(dtype)

            q, k = (rotate(q), rotate(k))
            h = self.attention(q, k, v, causal=False).transpose(0, 2, 1, 3).reshape(1, -1, 864)
            h = self.linear(self.norm(h, p + ".attn.norm"), p + ".attn.proj")
            scale = self.weights[p + ".gamma_1"]
            x = x + scale * h
            h = self.norm(x, p + ".norm2")
            chunks: list[Any] = []
            for start in range(0, h.shape[1], 512):
                part = h[:, start : start + 512]
                part = self.silu(self.linear(part, p + ".mlp.fc1_g")) * self.linear(
                    part, p + ".mlp.fc1_x"
                )
                part = self.linear(self.norm(part, p + ".mlp.norm"), p + ".mlp.fc2")
                mx.eval(part)
                chunks.append(part)
            h = mx.concatenate(chunks, axis=1)
            scale = self.weights[p + ".gamma_2"]
            x = x + scale * h
            mx.eval(x)
        features = self.norm(x, prefix + ".eva.norm")
        p = "model.vision3d.merger"
        embeddings = self.linear(
            self.gelu(self.linear(self.norm(features, p + ".norm", 1e-06), p + ".linear_fc1")),
            p + ".linear_fc2",
        )
        mx.eval(features, embeddings)
        return (features, embeddings)

    def rotary(self, x: Any, positions: Any) -> Any:
        phase = positions.astype(mx.float32)[..., None] * self.text_inv_freq
        channels = [0 if j % 3 == 0 or (j >= 30 and j % 3 == 2) else j % 3 for j in range(32)]
        freqs = mx.stack([phase[axis, ..., j] for j, axis in enumerate(channels)], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = (mx.cos(freqs).astype(x.dtype)[:, None], mx.sin(freqs).astype(x.dtype)[:, None])
        head, tail = (x[..., :64], x[..., 64:])
        rotated = mx.concatenate([-head[..., 32:], head[..., :32]], axis=-1)
        return mx.concatenate([head * cos + rotated * sin, tail], axis=-1)

    def full_attention(self, x: Any, p: str, positions: Any, state: dict[str, Any]) -> Any:
        batch, length, _ = x.shape
        q, gate = mx.split(
            self.linear(x, p + ".q_proj").reshape(batch, length, 16, 512), 2, axis=-1
        )
        q = self.rms(q, p + ".q_norm").transpose(0, 2, 1, 3)
        k = self.rms(
            self.linear(x, p + ".k_proj").reshape(batch, length, 4, 256), p + ".k_norm"
        ).transpose(0, 2, 1, 3)
        v = self.linear(x, p + ".v_proj").reshape(batch, length, 4, 256).transpose(0, 2, 1, 3)
        q, k = (self.rotary(q, positions), self.rotary(k, positions))
        offset = 0
        if "k" in state:
            offset = state["k"].shape[2]
            k, v = (
                mx.concatenate([state["k"], k], axis=2),
                mx.concatenate([state["v"], v], axis=2),
            )
        state.update(k=k, v=v)
        h = (
            self.attention(q, k, v.astype(q.dtype), causal=True, offset=offset)
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, -1)
        )
        return self.linear(h * self.sigmoid(gate.reshape(batch, length, -1)), p + ".o_proj")

    def linear_attention(self, x: Any, p: str, state: dict[str, Any]) -> Any:
        batch, length, _ = x.shape
        exponential = mx.exp
        qkv = self.linear(x, p + ".in_proj_qkv")
        previous = state.get("conv", mx.zeros((batch, 3, 8192), dtype=qkv.dtype))
        conv_input = mx.concatenate([previous, qkv], axis=1)
        state["conv"] = mx.contiguous(conv_input[:, -3:])
        qkv = self.silu(
            mx.conv1d(
                conv_input,
                self.weights[p + ".conv1d.weight"].astype(qkv.dtype).transpose(0, 2, 1),
                groups=8192,
            )
        )
        q, k, v = mx.split(qkv, [2048, 4096], axis=-1)
        q, k = (q.reshape(batch, length, 16, 128), k.reshape(batch, length, 16, 128))
        v = v.reshape(batch, length, 32, 128)
        normalized: list[Any] = []
        for value in (q, k):
            squared = value * value
            norm = mx.sum(squared, axis=-1, keepdims=True).astype(value.dtype)
            normalized.append((value * mx.rsqrt(norm + 1e-06)).astype(mx.float32))
        q, k = normalized
        q = q * 128 ** (-0.5)
        a = self.linear(x, p + ".in_proj_a").astype(mx.float32)
        beta = self.sigmoid(self.linear(x, p + ".in_proj_b")).astype(mx.float32)
        a = a + self.weights[p + ".dt_bias"].astype(mx.float32)
        logarithm = mx.log1p
        softplus = mx.where(a > 20, a, logarithm(exponential(a)))
        log_decay = -exponential(self.weights[p + ".A_log"].astype(mx.float32)) * softplus
        recurrent = state.get(
            "recurrent", mx.zeros((batch, 32, 128, 128), dtype=mx.float32)
        ).astype(mx.float32)
        if length > 1:
            h, recurrent = chunk_delta(q, k, v, log_decay, beta, recurrent)
        else:
            recurrence = self.delta.gated_delta_kernel
            h, recurrent = recurrence(
                q, k, v.astype(mx.float32), exponential(log_decay), beta, recurrent
            )
        state["recurrent"] = recurrent
        h = h.astype(x.dtype)
        fp = h.astype(mx.float32)
        inverse = mx.rsqrt(mx.mean(fp * fp, axis=-1, keepdims=True) + 1e-06)
        h = (fp * inverse).astype(x.dtype)
        h = h * self.weights[p + ".norm.weight"]
        z = self.linear(x, p + ".in_proj_z").reshape(batch, length, 32, 128)
        activation = self.silu
        h = (h.astype(mx.float32) * activation(z.astype(mx.float32))).astype(x.dtype)
        return self.linear(h.reshape(batch, length, -1), p + ".out_proj")

    def embed(self, ids: Any) -> Any:
        value = self.weights["model.language_model.embed_tokens.weight"][ids]
        return value

    def decode(
        self, embeddings: Any, positions: Any, cache: DecoderCache | None = None
    ) -> tuple[Any, DecoderCache]:
        """Return all final hidden states; callers project only consumed tokens."""
        if cache is None:
            cache = DecoderCache()
        x = embeddings.astype(self.dtype)
        for i in range(32):
            p = f"model.language_model.layers.{i}"
            h = self.rms(x, p + ".input_layernorm")
            if (i + 1) % 4:
                h = self.linear_attention(h, p + ".linear_attn", cache.layers[i])
            else:
                h = self.full_attention(h, p + ".self_attn", positions, cache.layers[i])
            x = x + h
            h = self.rms(x, p + ".post_attention_layernorm")
            chunks: list[Any] = []
            for start in range(0, h.shape[1], 256):
                part = h[:, start : start + 256]
                part = self.linear(
                    self.silu(self.linear(part, p + ".mlp.gate_proj"))
                    * self.linear(part, p + ".mlp.up_proj"),
                    p + ".mlp.down_proj",
                )
                mx.eval(part)
                chunks.append(part)
            x = x + mx.concatenate(chunks, axis=1)
            mx.eval(x, *cache.layers[i].values())
        cache.offset += embeddings.shape[1]
        x = self.rms(x, "model.language_model.norm")
        mx.eval(x)
        return (x, cache)

    def logits(self, hidden: Any) -> Any:
        weight = self.weights["model.language_model.embed_tokens.weight"]
        return hidden @ weight.T
