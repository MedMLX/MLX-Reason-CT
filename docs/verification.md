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

## Source-BF16 optimization measurement (2026-10-09)

One synthetic `small-question` case was measured on an Apple M1 Max with 32 GPU
cores and 64 GiB memory, macOS 27.0, Python 3.12.15 and MLX 0.32.3. It uses the
pinned checkpoint, a 192-cubed processed volume, 13,842 prompt tokens, greedy
decoding and the prompt `Answer only CT.`. The completed output is two tokens,
including EOS. This measures full model execution for a short response, not
long-generation throughput.

| Implementation | Vision (s) | Decoder prefill (s) | Cached generation total (s) |
| --- | ---: | ---: | ---: |
| Pre-`a798113` projection control | 595.0 | 1,003.5 | 1,602.4 |
| `eb2c10b`, before this change | 567.0 | 502.6 | 1,074.3 |
| Fast attention K8 and 512-row projection batches | 452.5 | 433.6 | 889.2 |

The control substitutes only the projection implementation from `a798113^`;
the rest of the model and runtime are identical. Each row is one instrumented,
synchronized run. Total time includes model construction, vision, decoding and
observer checks, but excludes host preprocessing and bundle verification.
The new total is 17.2% lower than `eb2c10b` and 1.80 times faster than the older
projection control. These are single-case measurements, not repeated full-model
statistics or timings of the published wheel.

All recorded vision features, projected embeddings, final eight prefill hidden
states and consumed cached logits are bitwise identical before and after this
change. The vision and hidden boundaries also match the CUDA source exactly;
cached source-logit comparisons pass the unchanged `atol=0.125`, `rtol=0.01`,
normalized RMSE `<=0.005` gates. Exact tokens and EOS termination are retained.
Cached peak MLX allocation is approximately 12.37 GiB in both native runs.

The initial fast-K8/batching uncached pass was interrupted by a system GPU reset;
macOS named WebKit in a `CDM Kill timeout` report and MLX received an
innocent-victim error.
A fresh-model retry reused the verified native vision features and first token,
then rebuilt the full prefix without a decoder cache. It reached EOS with
bitwise-unchanged native logits, zero source tolerance violations and normalized
RMSE 0.0000297. This is recovered continuation evidence, not an uninterrupted
full-model qualification run; both receipts are retained.

Isolated attention probes used one warmup and seven alternating synchronized
samples at the existing launch sizes: median speedups were 1.26 times for vision
and 1.36 times for decoder attention at full context. Projection batch probes
covered ten model shapes plus ragged, wide-exponent and partitioned cases; all
outputs matched bitwise. These probes support the execution changes, while the
full-model run checks their integration.

Local scripts, raw samples and hash-bound receipts are retained under
`outputs/bf16-20261009/` and are not packaged. This evidence does not complete
the four-case original-BF16 qualification matrix or establish clinical accuracy.

### Attention query and probability reuse

A subsequent attention-only change reuses each query block across score columns
and each probability block across output channels, retaining per-output K8 order.
On the same host, five warmups and 15 alternating synchronized samples measured
median attention-launch speedups of 1.20 times for vision and 1.25 times for
full-context decoder prefill. Every sampled output was bitwise unchanged.
Earlier probes had substantial timing variation; these figures use the stable
repeated probe. Larger feed-forward batches were also exact, but their roughly
3% improvement was borderline against the probe's noise threshold, so that
setting is unchanged.

The follow-up full-model cached run completed in 763.7 seconds, compared with
889.2 seconds for fast K8 and 512-row projection batches: 14.1% less time.
Vision took 378.0 seconds versus 452.5 seconds; decoder prefill took 381.6 seconds
versus 433.6 seconds. Recorded vision features, projected embeddings, final
hidden-state sample and cached logits remained bitwise identical to the previous
native results. CUDA comparisons passed the
unchanged gates, and both expected tokens, including EOS, matched. Peak cached
MLX allocation remained approximately 12.37 GiB. These are single-case,
instrumented full-model timings, not repeated full-model statistics.

The same run completed its fresh full-prefix uncached continuation without an
interruption, in 389.0 seconds. Its logits were bitwise unchanged from the prior
native result, with source NRMSE 0.0000297 and zero tolerance violations. Both
cached and uncached generation reached the expected EOS. The longer three
qualification cases remain unqualified. Local scripts, raw timings, checksums
and receipts are retained in `outputs/bf16-followup-20261009/`.
