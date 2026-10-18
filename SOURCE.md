# Native NV-Reason-CT source provenance

The package supports macOS arm64 with MLX on Metal only. Shared runtime admission
raises when Metal is unavailable; learned operations have no host fallback.
NIfTI IO, geometry, cropping, tokenization and offline weight conversion are host
stages, not inference backends.

## Pinned model and arithmetic

[NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT/tree/386b93e034983f6c1fc841a43833a1b6a0cd9c13)
(OpenMDW-1.1) supplies the model, processor, tokenizer, template and configurations.
The staged upstream LICENSE is required. BF16 source parameters expand losslessly
to FP32 in the converted bundle; there is no retraining.

Physical resampling follows MONAI 1.6.0 (Apache-2.0). Primus follows
dynamic-network-architectures 0.4.3 (Apache-2.0), with EVA attention, SwiGLU and
rotary semantics from timm 1.0.22 (Apache-2.0). The merger, hybrid decoder,
MRoPE, cast boundaries and cache storage follow Transformers 5.10.4 (Apache-2.0).
The disconnected 2D tower, inactive mask token and duplicate tied LM head are
excluded; the source processor rejects 2D images and videos.

Host CT resampling retains MONAI 1.6.0's `spatial_resample` and `to_norm_affine`
normalization, followed by the pinned PyTorch CPU float32 `affine_grid` and
border-clamped `grid_sample` arithmetic. This includes the separately rounded
axis scaling, four-term affine products and eight trilinear corner stores.
The implementation uses NumPy on the host and bounds coordinate storage to
eight output planes. Synthetic host-resampling fixtures come from the original
MONAI/PyTorch CPU operators on the Intel source host with one thread, isolating
the K4 arithmetic from threaded BLAS shard tails. They check resampling
independently of the model weights, vision outputs and generated reports.
Source files are
[`AffineGridGenerator.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/AffineGridGenerator.cpp)
and
[`GridSampler.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/GridSampler.cpp).

Public precision selection retains these arithmetic identities:

| Public profile | Internal profile | Learned storage and arithmetic |
| --- | --- | --- |
| `float32` (default) | `float32` | FP32 weights, activations and caches |
| `bfloat16` | `source_bfloat16` | BF16 weights and source cast/cache boundaries |
| `bfloat16_fp32` | `bfloat16` | BF16 weights with FP32 accumulation and decoder caches |

The alternate BF16-storage/FP32-accumulation profile changes the source numerical
contract. It is not original-source BF16 parity. Profile names select arithmetic
on the same Metal backend.

## Metal recurrence

The recurrent kernels derive from MLX-LM 0.32.0
`mlx_lm/models/gated_delta.py` (MIT, Copyright Apple).

MLX loads through the shared runtime before the recurrent kernels. The adapted
implementation retains the Metal kernel bodies and dispatch, with no host
inference path. The package initializer and MedMLX declaration remain lazy.

The multi-token prefill retains Transformers' 64-token chunk algebra. Its
Sklansky scan, activation formulas, normalization, projection stores and
single-token recurrence preserve the pinned source arithmetic. Source operator
provenance includes PyTorch 2.13.0 revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`, the Flash source submodule
`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, and MLX 0.32.2 layout helpers.
MIT/BSD notices remain in `src/mlx_reason_ct/licenses/`.

`source_arithmetic_assets` contains Metal kernels and losslessly
encoded, model-independent unary/rotary constants. Their original encoding and
source provenance are retained; they are not selectable execution backends or
platform-keyed test fixtures. They contain no learned parameters or case-specific
neural outputs. Their source operation order and rounding behavior are part of
the model's arithmetic contract.

The source-BF16 projection kernel keeps each output's K8 block order, carry
truncation, slice stores and BF16 epilogue. Its execution reads BF16 or FP32
operands in place and computes up to four rows per thread. With no inf, NaN or
subnormal operands, the K8 exponent maximum is computed from operand powers and
alignment is computed as `trunc(v * 2^-common)` inside a [2^-100, 2^125] window.
Blocks outside the window, and projections with such operands, use the
unchanged reference block. The fast path is bitwise identical to the previous
kernel on seeded wide-exponent, cancellation, inf/NaN/subnormal, ragged-row,
bias and sliced cases. On one M1 Max it was 1.6-3.8 times faster for 1024-row
decoder projections. This timing is not full-model qualification.

Source-BF16 attention uses the same fast K8 alignment with a local operand
check, including the probability tiles formed inside the kernel. Special
operands and blocks outside the alignment window use the original bit-shift
block. Query/key and probability/value accumulation order, softmax arithmetic,
and the 64-query vision and 128-query decoder launch limits are unchanged.
Query blocks are reused across score columns, and shuffled probability blocks
are reused across value channels. Each score and output retains its original
K8 accumulation order.
Projection launches group up to 512 rows to reduce dispatch overhead, retaining
the same per-output arithmetic and partition stores. See the
[verification scope](docs/verification.md) for measured evidence and limits.

## Current test references

Tests use one reference recorded on darwin-arm64 by
`scripts/record_arithmetic_reference.py`. It records seeded NumPy float64 formulas
and explicit BF16 rounding; it does not import production arithmetic or execute
Metal. Tests check numerical properties with rounding bounds, output dtypes,
cache reuse, generation completion and request limits. The independent recurrence
comparison retains its existing tolerances.

Metal tests also compare optimized attention K8 carries with the retained
bit-shift reference and analytic truncation/cancellation/overflow anchors.
Batched projections are checked against an unbatched reference across a ragged
batch boundary with bias and source partition stores.
Uniform-attention tests independently check masked value means across key tiles,
query batches, output channels and grouped heads.

See [verification scope](docs/verification.md). These component checks do not
establish full-model parity, clinical validation or report accuracy.
