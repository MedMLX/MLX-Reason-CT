"""Native Metal Primus, multimodal projection and Qwen3.5 hybrid decoding.

All learned arithmetic runs on Metal. Internal arithmetic identities retain
the pinned receipt names: source_bfloat16 preserves NVIDIA's BF16 cast/cache
boundaries; bfloat16 stores BF16 weights with FP32 accumulation and decoding.
Public bfloat16 selects source_bfloat16; public bfloat16_fp32 selects bfloat16.
Source layouts remain visible in the converted cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from mlx_reason_ct.chunk_delta import chunk_delta
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.mlx_weights import SHARDS
from mlx_reason_ct.runtime import import_mlx

mx: Any = import_mlx()
nn: Any = import_module("mlx.nn")
# Admit Metal before loading the adapted recurrence kernels.
gated_delta: Any = import_module("mlx_reason_ct.gated_delta")
source_arithmetic: Any = import_module("mlx_reason_ct.source_arithmetic")


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

    def __init__(self, model_dir: Path, *, precision: str = "float32") -> None:
        if precision not in {"float32", "bfloat16", "source_bfloat16"}:
            raise InvalidInputError("Unsupported NV-Reason-CT precision profile")
        self.dtype: Any = mx.float32 if precision == "float32" else mx.bfloat16
        self.precision: str = precision
        self.accumulate_float32: bool = precision == "bfloat16"
        self.config: dict[str, Any] = json.loads((model_dir / "config.json").read_text())
        self.weights: dict[str, Any] = {}
        for name in SHARDS:
            self.weights.update(mx.load(str(model_dir / "mlx" / name)))
        for name, value in self.weights.items():
            if value.dtype != mx.float32:
                raise InvalidInputError(f"Native cache must store FP32 tensors: {name}")
            self.weights[name] = value.astype(self.dtype)
        mx.eval(*self.weights.values())
        self.delta = gated_delta
        self.eos_ids: tuple[int, ...] = tuple(
            json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"]
        )
        # Primus uses 12 frequencies on each of three axes, interleaved pairs.
        coords = np.indices((24, 24, 24), dtype=np.float32).reshape(3, -1).T
        inv = 1 / np.power(np.float32(10000), np.arange(12, dtype=np.float32) / 12)
        phase = (coords[..., None] * inv).reshape(13824, 36).repeat(2, axis=-1)
        # Primus builds FP32 rotary constants even with BF16 learned weights.
        self.vision_sin: Any = mx.array(np.sin(phase))
        self.vision_cos: Any = mx.array(np.cos(phase))
        if self.precision == "source_bfloat16":
            self.vision_sin, self.vision_cos = source_arithmetic.vision_constants()
        # The source constructs this non-learned buffer on the host before transfer.
        # Metal pow differs by one ULP at six frequencies, magnified by positions.
        self.text_inv_freq: Any = mx.array(
            1 / np.power(np.float32(10000000), np.arange(0, 64, 2, dtype=np.float32) / 64)
        )

    @staticmethod
    def silu(x: Any) -> Any:
        if x.dtype == mx.bfloat16:
            return source_arithmetic.silu(x)
        # ATen's SiLU divides directly; x * sigmoid(x) rounds twice in FP32.
        fp = x.astype(mx.float32)
        return (fp / (1 + mx.exp(-fp))).astype(x.dtype)

    @staticmethod
    def sigmoid(x: Any) -> Any:
        if x.dtype == mx.bfloat16:
            return source_arithmetic.sigmoid(x)
        # ATen computes BF16 activations in opmath FP32, then casts once.
        return (1 / (1 + mx.exp(-x.astype(mx.float32)))).astype(x.dtype)

    @staticmethod
    def gelu(x: Any) -> Any:
        return nn.gelu(x.astype(mx.float32)).astype(x.dtype)

    def linear(self, x: Any, name: str, *, source_length: int | None = None) -> Any:
        # BF16 vision operands accumulate into FP32 without an output downcast.
        # Decoder inputs stay FP32 to preserve cached/uncached consistency.
        if self.accumulate_float32 and name.startswith("model.vision3d."):
            x = x.astype(mx.bfloat16).astype(mx.float32)
        weight = self.weights[name + ".weight"]
        bias = self.weights.get(name + ".bias")
        rows = x.shape[-2] if source_length is None else source_length
        if self.precision == "source_bfloat16":
            if (
                x.shape[:-1] == (1, 1)
                and rows == 1
                and tuple(reversed(weight.shape)) in source_arithmetic.OBSERVED_LANES
            ):
                assert bias is None
                return source_arithmetic.cached_projection(x, weight)
            decoder_projection = name.endswith(
                (
                    ".in_proj_qkv",
                    ".in_proj_a",
                    ".in_proj_b",
                    ".in_proj_z",
                    ".mlp.gate_proj",
                    ".mlp.up_proj",
                    ".mlp.down_proj",
                    ".linear_attn.out_proj",
                    ".self_attn.q_proj",
                    ".self_attn.k_proj",
                    ".self_attn.v_proj",
                    ".self_attn.o_proj",
                )
            )
            if name.startswith("model.vision3d.") or decoder_projection:
                slices = (
                    2
                    if rows >= 13824 and name.endswith((".in_proj_a", ".in_proj_b"))
                    else 3
                    if rows >= 13824 and name.endswith(".mlp.down_proj")
                    else 1
                )
                sliced = name.endswith(
                    (
                        ".mlp.down_proj",
                        ".linear_attn.out_proj",
                        ".self_attn.k_proj",
                        ".self_attn.v_proj",
                        ".self_attn.o_proj",
                    )
                )
                return source_arithmetic.projection(
                    x,
                    weight,
                    bias=bias,
                    slices=slices,
                    sliced=sliced,
                )
        if self.accumulate_float32:
            weight = weight.astype(mx.float32)
            if bias is not None:
                bias = bias.astype(mx.float32)
        return x @ weight.T if bias is None else mx.addmm(bias, x, weight.T)

    def norm(self, x: Any, name: str, eps: float = 1e-5) -> Any:
        if x.dtype == mx.bfloat16 and not self.accumulate_float32 and x.shape[-1] in (864, 2304):
            return source_arithmetic.layernorm(
                x,
                self.weights[name + ".weight"],
                self.weights[name + ".bias"],
                eps,
            )
        # CUDA LayerNorm keeps statistics and affine arithmetic in opmath FP32.
        # MLX's BF16 affine path rounds before the source's single output cast.
        dtype = mx.float32 if self.accumulate_float32 else x.dtype
        return mx.fast.layer_norm(
            x.astype(mx.float32),
            self.weights[name + ".weight"].astype(mx.float32),
            self.weights[name + ".bias"].astype(mx.float32),
            eps,
        ).astype(dtype)

    def rms(self, x: Any, name: str) -> Any:
        # Qwen's learned weight is a zero-centered offset; addition is FP32.
        fp = x.astype(mx.float32)
        if self.precision == "source_bfloat16" and x.shape[-1] in (128, 256, 2560):
            value = fp * source_arithmetic.rsqrt(source_arithmetic.mean_square(fp * fp) + 1e-6)
        else:
            value = fp * mx.rsqrt(mx.mean(fp * fp, axis=-1, keepdims=True) + 1e-6)
        dtype = mx.float32 if self.accumulate_float32 else x.dtype
        return (value * (1 + self.weights[name + ".weight"].astype(mx.float32))).astype(dtype)

    @staticmethod
    def attention(q: Any, k: Any, v: Any, *, causal: bool, offset: int = 0) -> Any:
        if q.dtype == mx.bfloat16:
            return source_arithmetic.attention(q, k, v, causal=causal, offset=offset)
        # Bound the materialized score workspace even when a backend cannot fuse SDPA.
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
                q[:, :, start:end],
                k,
                v,
                scale=q.shape[-1] ** -0.5,
                mask=mask,
            )
            mx.eval(result)
            outputs.append(result)
        return mx.concatenate(outputs, axis=2)

    def vision(self, pixels: Any) -> tuple[Any, Any]:
        if pixels.shape != (1, 1, 192, 192, 192):
            raise InvalidInputError("NV-Reason-CT vision expects a single 192-cubed CZYX CT")
        prefix = "model.vision3d.sub_vision"
        # Non-overlapping Conv3d is exactly a flattened patch projection.
        x = pixels.astype(self.dtype).reshape(1, 1, 24, 8, 24, 8, 24, 8)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(1, 13824, 512)
        weight = self.weights[prefix + ".down_projection.proj.weight"].reshape(864, 512)
        # CUDA Conv3d rounds its convolution before adding the channel bias.
        x = (
            source_arithmetic.convolution(x, weight)
            if self.precision == "source_bfloat16"
            else x @ weight.T
        ) + self.weights[prefix + ".down_projection.proj.bias"]
        if self.accumulate_float32:
            x = x.astype(mx.float32)
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

            q, k = rotate(q), rotate(k)
            if self.accumulate_float32:
                q, k, v = (t.astype(mx.float32) for t in (q, k, v))
            h = self.attention(q, k, v, causal=False).transpose(0, 2, 1, 3).reshape(1, -1, 864)
            h = self.linear(self.norm(h, p + ".attn.norm"), p + ".attn.proj")
            scale = self.weights[p + ".gamma_1"]
            if self.accumulate_float32:
                scale, h = scale.astype(mx.float32), h.astype(mx.float32)
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
            if self.accumulate_float32:
                scale, h = scale.astype(mx.float32), h.astype(mx.float32)
            x = x + scale * h
            mx.eval(x)
        features = self.norm(x, prefix + ".eva.norm")
        p = "model.vision3d.merger"
        embeddings = self.linear(
            self.gelu(self.linear(self.norm(features, p + ".norm", 1e-6), p + ".linear_fc1")),
            p + ".linear_fc2",
        )
        mx.eval(features, embeddings)
        return features, embeddings

    def rotary(self, x: Any, positions: Any) -> Any:
        phase = positions.astype(mx.float32)[..., None] * self.text_inv_freq
        # Interleaved temporal/height/width sections 11/11/10, exactly as Qwen3.5.
        channels = [0 if j % 3 == 0 or j >= 30 and j % 3 == 2 else j % 3 for j in range(32)]
        freqs = mx.stack([phase[axis, ..., j] for j, axis in enumerate(channels)], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(freqs).astype(x.dtype)[:, None], mx.sin(freqs).astype(x.dtype)[:, None]
        head, tail = x[..., :64], x[..., 64:]
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
        q, k = self.rotary(q, positions), self.rotary(k, positions)
        offset = 0
        if "k" in state:
            offset = state["k"].shape[2]
            k, v = mx.concatenate([state["k"], k], axis=2), mx.concatenate([state["v"], v], axis=2)
        state.update(k=k, v=v)
        h = (
            self.attention(q, k, v.astype(q.dtype), causal=True, offset=offset)
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, -1)
        )
        return self.linear(h * self.sigmoid(gate.reshape(batch, length, -1)), p + ".o_proj")

    def linear_attention(self, x: Any, p: str, state: dict[str, Any]) -> Any:
        batch, length, _ = x.shape
        source_bf16 = self.precision == "source_bfloat16"
        exponential = source_arithmetic.exp if source_bf16 else mx.exp
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
        q, k = q.reshape(batch, length, 16, 128), k.reshape(batch, length, 16, 128)
        v = v.reshape(batch, length, 32, 128)
        if self.accumulate_float32:
            q, k = q.astype(mx.float32), k.astype(mx.float32)
        normalized: list[Any] = []
        for value in (q, k):
            squared = value * value
            # ATen sums BF16 squares in FP32 opmath, then returns BF16. MLX's
            # BF16 reduction accumulates differently; preserve both cast points.
            if self.precision == "source_bfloat16":
                squared = squared.astype(mx.float32)
            norm = (
                source_arithmetic.sum_square(squared)
                if source_bf16
                else mx.sum(squared, axis=-1, keepdims=True)
            ).astype(value.dtype)
            normalized.append((value * mx.rsqrt(norm + 1e-6)).astype(mx.float32))
        q, k = normalized
        q = q * (128**-0.5)
        a = self.linear(x, p + ".in_proj_a").astype(mx.float32)
        beta = self.sigmoid(self.linear(x, p + ".in_proj_b")).astype(mx.float32)
        a = a + self.weights[p + ".dt_bias"].astype(mx.float32)
        # ATen Softplus uses log1p(exp(x)) below its explicit threshold of 20.
        logarithm = source_arithmetic.log1p if source_bf16 else mx.log1p
        softplus = mx.where(a > 20, a, logarithm(exponential(a)))
        log_decay = -exponential(self.weights[p + ".A_log"].astype(mx.float32)) * softplus
        recurrent = state.get(
            "recurrent", mx.zeros((batch, 32, 128, 128), dtype=mx.float32)
        ).astype(mx.float32)
        # Both prefill and single-token recurrence execute on Metal.
        if length > 1:
            h, recurrent = chunk_delta(
                q,
                k,
                v,
                log_decay,
                beta,
                recurrent,
                source_bfloat16=source_bf16,
            )
        else:
            recurrence = (
                source_arithmetic.gated_delta if source_bf16 else self.delta.gated_delta_kernel
            )
            h, recurrent = recurrence(
                q,
                k,
                v.astype(mx.float32),
                exponential(log_decay),
                beta,
                recurrent,
            )
        # The pinned cache inherits convolution dtype when storing FP32 recurrence.
        state["recurrent"] = (
            recurrent.astype(self.dtype) if self.precision == "source_bfloat16" else recurrent
        )
        h = h.astype(x.dtype)
        fp = h.astype(mx.float32)
        inverse = (
            source_arithmetic.rsqrt(source_arithmetic.mean_square(fp * fp) + 1e-6)
            if source_bf16
            else mx.rsqrt(mx.mean(fp * fp, axis=-1, keepdims=True) + 1e-6)
        )
        h = (fp * inverse).astype(x.dtype)
        h = h * self.weights[p + ".norm.weight"]
        z = self.linear(x, p + ".in_proj_z").reshape(batch, length, 32, 128)
        activation = source_arithmetic.silu if source_bf16 else self.silu
        h = (h.astype(mx.float32) * activation(z.astype(mx.float32))).astype(x.dtype)
        return self.linear(h.reshape(batch, length, -1), p + ".out_proj")

    def embed(self, ids: Any) -> Any:
        value = self.weights["model.language_model.embed_tokens.weight"][ids]
        return value.astype(mx.float32) if self.accumulate_float32 else value

    def decode(
        self, embeddings: Any, positions: Any, cache: DecoderCache | None = None
    ) -> tuple[Any, DecoderCache]:
        """Return all final hidden states; callers project only consumed tokens."""
        if cache is None:
            cache = DecoderCache()
        x = embeddings.astype(mx.float32 if self.accumulate_float32 else self.dtype)
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
                    self.silu(
                        self.linear(
                            part,
                            p + ".mlp.gate_proj",
                            source_length=h.shape[1],
                        )
                    )
                    * self.linear(part, p + ".mlp.up_proj", source_length=h.shape[1]),
                    p + ".mlp.down_proj",
                    source_length=h.shape[1],
                )
                mx.eval(part)
                chunks.append(part)
            x = x + mx.concatenate(chunks, axis=1)
            mx.eval(x, *cache.layers[i].values())
        cache.offset += embeddings.shape[1]
        x = self.rms(x, "model.language_model.norm")
        mx.eval(x)
        return x, cache

    def logits(self, hidden: Any) -> Any:
        weight = self.weights["model.language_model.embed_tokens.weight"]
        if self.accumulate_float32:
            weight = weight.astype(mx.float32)
        return hidden @ weight.T
