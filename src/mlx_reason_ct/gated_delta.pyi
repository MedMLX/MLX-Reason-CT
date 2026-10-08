"""Types for the verbatim pinned MLX-LM recurrence module."""

import mlx.core as mx

def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array: ...
def compute_lower_bound_g(
    A_log: mx.array, a: mx.array, dt_bias: mx.array, lower_bound: float
) -> mx.array: ...
def normalize_qk(
    q: mx.array, k: mx.array, *, inv_scale: float, eps: float
) -> tuple[mx.array, mx.array]: ...
def gated_delta_kernel_xtree(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_kernel_unpacked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    use_kernel: bool = True,
    lower_bound: float | None = None,
    allow_neg_eigval: bool = False,
) -> tuple[mx.array, mx.array]: ...
