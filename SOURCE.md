# Native NV-Reason-CT source provenance

The package supports macOS arm64 with MLX on Metal only. Shared runtime admission
raises when Metal is unavailable; learned operations have no host fallback.
NIfTI IO, geometry, cropping, tokenization and offline weight conversion are host
stages, not inference backends.

## Pinned model and arithmetic

[NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT/tree/386b93e034983f6c1fc841a43833a1b6a0cd9c13)
(OpenMDW-1.1) supplies the model, processor, tokenizer, template and configurations.
The converted manifest binds source weights and staged runtime-file hashes.
The staged upstream LICENSE is required. BF16 source parameters expand losslessly
to FP32 in the converted bundle; there is no retraining.

Physical resampling follows MONAI 1.6.0 (Apache-2.0). Primus follows
dynamic-network-architectures 0.4.3 (Apache-2.0), with EVA attention, SwiGLU and
rotary semantics from timm 1.0.22 (Apache-2.0). The merger, hybrid decoder,
MRoPE, cast boundaries and cache storage follow Transformers 5.10.4 (Apache-2.0).
The disconnected 2D tower, inactive mask token and duplicate tied LM head are
excluded; the source processor rejects 2D images and videos.

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
`mlx_lm/models/gated_delta.py` (MIT, Copyright Apple). The original file hash was
`f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428`.
The adapted file hash is
`3a4c89d96ef40829423cf0eb0eed37420c3bd5256b675e5d99bb2cd797e874d7`.

The Apple-only cutover removed `gated_delta_update`, `gated_delta_ops`, their
single-step operation helper, unused gate/normalization helpers, and redundant
kernel-factory probes that returned `None`. MLX now loads through the shared
runtime. Metal kernel bodies and their dispatch remain unchanged. Retaining the
unused host fallback was rejected because it contradicted the supported runtime.
The package initializer and MedMLX declaration remain lazy.

The multi-token prefill retains Transformers' 64-token chunk algebra. Its
Sklansky scan, activation formulas, normalization, projection stores and
single-token recurrence preserve the pinned source arithmetic. Source operator
provenance includes PyTorch 2.13.0 revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`, the Flash source submodule
`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, and MLX 0.32.2 layout helpers.
MIT/BSD notices remain in `src/mlx_reason_ct/licenses/`.

`source_arithmetic_assets` contains hash-bound Metal kernels and losslessly
encoded, model-independent unary/rotary constants. Their original encoding and
source provenance are retained; they are not selectable execution backends or
platform-keyed test fixtures. They contain no learned parameters or case-specific
neural outputs. Removing this Metal arithmetic to erase source-platform names
was rejected because it would change the model.

## Current test references

Historical foreign-platform bit-pattern fixtures and qualification receipts were
retired. Tests use one reference recorded on darwin-arm64 by
`scripts/record_arithmetic_reference.py`. It records seeded NumPy float64 formulas
and explicit BF16 rounding; it does not import production arithmetic or execute
Metal. Tests check numerical properties with rounding bounds, output dtypes,
cache reuse, generation completion and request limits. The independent recurrence
comparison retains its existing tolerances.

See [verification scope](docs/verification.md). These component checks do not
establish full-model parity, clinical validation or report accuracy.
