# Verification scope

The supported host is macOS arm64 with MLX on Metal. Learned execution fails
explicitly when Metal is unavailable. Host preprocessing and bundle conversion
are separate from learned execution.

## Seeded references

`tests/fixtures/nv_reason_ct/arithmetic.json` was recorded on darwin-arm64 with
NumPy 2.3.5 and Python 3.12. It contains analytic float64 expectations, explicit
BF16 storage rounding and seed/provenance metadata. It contains no learned
weights or medical data. Re-record it from the repository root:

```bash
python scripts/record_arithmetic_reference.py
```

Tests load this single reference directly. There is no platform/backend key or
alternative fixture directory, and the recording script is not imported at test
time. Recording analytic expectations does not execute Metal.

FP32 accumulation comparisons use `gamma(n) = n*u/(1-n*u)`, with
`u = 2**-24`, for the seeded, well-conditioned sums. Projection error scales
with the sum of absolute products. Normalization permits four accumulation bounds
for statistics, normalization and affine operations. BF16 comparisons additionally
permit one storage ULP (`2**-7` relative); finite activation formulas permit eight
FP32 epsilons for exponential/division approximations. Exact equality is retained
for discrete contracts, lossless conversion, and copies/casts within one MLX run.
The existing independent recurrent-state comparison keeps its original bounds.

These tests check mathematical and storage contracts using rounding bounds,
without requiring bitwise agreement with another platform's reduction order.

## Checks

```bash
make env
make qa
make build
```

Metal tests require GPU access. In a sandbox without it, run host-stage tests:

```bash
python -m pytest -q --ignore=tests/test_source_arithmetic.py
```

This subset covers MedMLX discovery, lazy imports, public API orchestration,
input/asset failures, overwrite guards, versioned output records, geometry,
tokenization and weight conversion. It does not exercise learned Metal arithmetic.
The Metal suite checks recurrence, precision profiles and cache behavior. Neither
subset establishes full-model parity, report accuracy or clinical validation.
All generated reports require human review.
