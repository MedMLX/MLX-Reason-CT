"""Metal implementation of Transformers 5.10.4's 64-token gated-delta prefill.

The chunk algebra and FP32 accumulation boundaries follow torch_chunk_gated_delta_rule.
Sequential recurrence can produce materially different hidden states over
long CT token sequences.
"""

from __future__ import annotations

import math
from typing import Any

from mlx_reason_ct.runtime import import_mlx

mx: Any = import_mlx()


def source_cumsum(g: Any) -> Any:
    """ATen CUDA's Sklansky order, including its row-count-dependent block size.

    Decay uses differences of these sums inside exp; reassociation at long CT
    contexts measurably changes decoder states. No higher/lower precision cast.
    """
    rows = math.prod(g.shape[:-1])
    threads_log = min(9, max(4, (9 + 6 - (rows - 1).bit_length()) // 2))
    block_size = min(64, 2 << threads_log)
    output: list[Any] = []
    previous = mx.zeros(g.shape[:-1], dtype=mx.float32)
    index = mx.arange(block_size)
    for start in range(0, 64, block_size):
        part = g[..., start : start + block_size]
        part = mx.concatenate([part[..., :1] + previous[..., None], part[..., 1:]], axis=-1)
        step = 1
        while step < block_size:
            source = (index // (2 * step)) * 2 * step + step - 1
            part = mx.where(index % (2 * step) >= step, part + part[..., source], part)
            step *= 2
        previous = part[..., -1]
        output.append(part)
    return mx.concatenate(output, axis=-1)


def chunk_delta(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state: Any,
    *,
    source_bfloat16: bool = False,
) -> tuple[Any, Any]:
    """Consume normalized/scaled Q/K and log decay; state uses MLX-LM's V,K order."""
    from mlx_reason_ct import source_arithmetic

    exponential = source_arithmetic.exp if source_bfloat16 else mx.exp
    batch, length, heads, _ = v.shape
    count = (length + 63) // 64
    padding = count * 64 - length
    repeat = heads // q.shape[2]
    q, k = mx.repeat(q, repeat, axis=2), mx.repeat(k, repeat, axis=2)

    def chunks(x: Any) -> Any:
        x = x.transpose(0, 2, 1, 3).astype(mx.float32)
        x = mx.pad(x, [(0, 0), (0, 0), (0, padding), (0, 0)])
        return x.reshape(batch, heads, count, 64, x.shape[-1])

    q, k, v = chunks(q), chunks(k), chunks(v)
    beta = chunks(beta[..., None])
    g = source_cumsum(chunks(g[..., None])[..., 0])
    lower = mx.arange(64)[:, None] >= mx.arange(64)[None, :]
    decay = mx.where(lower, exponential(mx.minimum(g[..., :, None] - g[..., None, :], 0)), 0)
    kb, vb = k * beta, v * beta
    attn = mx.where(mx.eye(64, dtype=mx.bool_) | ~lower, 0, -(kb @ k.swapaxes(-1, -2)) * decay)
    # This triangular recurrence intentionally matches the source reduction order.
    for i in range(1, 64):
        row = attn[..., i, :i]
        product = row[..., :, None] * attn[..., :i, :i]
        update = row + (
            source_arithmetic.sum_product(product) if source_bfloat16 else mx.sum(product, axis=-2)
        )
        attn[..., i, :i] = update
        mx.eval(attn)
    attn = attn + mx.eye(64)
    values = attn @ vb
    keys = attn @ (kb * exponential(g)[..., None])
    state = state.swapaxes(-1, -2)
    output: list[Any] = []
    mx.eval(values, keys, decay)
    for i in range(count):
        qi, ki = q[:, :, i], k[:, :, i]
        gi = g[:, :, i]
        new = values[:, :, i] - keys[:, :, i] @ state
        attention = (qi @ ki.swapaxes(-1, -2)) * decay[:, :, i]
        part = (qi * exponential(gi)[..., None]) @ state + attention @ new
        state = (
            state * exponential(gi[..., -1, None, None])
            + (ki * exponential(gi[..., -1, None] - gi)[..., None]).swapaxes(-1, -2) @ new
        )
        mx.eval(part, state)
        output.append(part)
    result = mx.concatenate(output, axis=2)[:, :, :length].transpose(0, 2, 1, 3)
    return result, mx.contiguous(state.swapaxes(-1, -2))
