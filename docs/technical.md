# Native NV-Reason-CT source details

The package supports macOS arm64 with MLX on Metal only. Shared runtime admission
raises when Metal is unavailable; learned operations have no host fallback.
NIfTI IO, geometry, cropping and tokenization are host stages, not inference
backends.

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
single-token recurrence preserve the pinned source arithmetic. Operator source
pins include PyTorch 2.13.0 revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`, the Flash source submodule
`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, and MLX 0.32.2 layout helpers.
MIT/BSD notices remain in `THIRD_PARTY_NOTICES.md`.

FP32 prefill resolves each strictly lower 64-by-64 attention matrix in one
Metal dispatch. The kernel keeps products separately rounded and reproduces
MLX 0.32.4's column-sum order: an identity at row 1, up to eight interleaved
accumulators below row 32, then a 32-lane SIMD reduction. Threadgroup barriers
preserve the dependency between successive rows without 63 host evaluations.
Source-BF16 prefill keeps its existing gradual-underflow reduction and
materialization boundaries.

`source_arithmetic_assets` contains Metal kernels and losslessly
encoded, model-independent unary/rotary constants. Their original encoding and
source details are retained; they are not selectable execution backends or
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
[verification scope](verification.md) for measured evidence and limits.

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

See [verification scope](verification.md). These component checks do not
establish full-model parity, clinical validation or report accuracy.

## Reusable CT conversation sessions

```python
from mlx_reason_ct import CTSession

with CTSession("ct.nii.gz", model_dir="converted-bundle") as session:
    first = session.report("out/first")
    followup = session.report("out/followup", prompt="Describe the lung findings.")
```

`CTSession` lazily loads one model and encodes one CT on its first report. Further
questions reuse the model and projected CT embeddings and include completed
responses as history. Failed or truncated responses retain the existing durable
failure outputs but do not enter history. `reset()` clears history/cache while
retaining the encoding; `close()` releases owned state. Calls serialize per session.
The session stores the raw decoded response as history, before the report writer
separates thinking or removes a trailing empty answer tag; the chat template controls
how that text is rendered on a later turn.
Changes to the source file's identity, size or modification time are rejected.
A new CT, model or precision profile requires a new session.

In `run.json`, `ct_geometry.anatomy_fallback` records that region bounds are absent.
It is also true for an intentionally centered `anatomy_region=None` crop, so read it
alongside `generation.anatomy_region` before interpreting it as a fallback.

KV caches grow in chunks of 256 tokens by default; `cache_chunk_size=0` selects
the original concatenating cache. Capacity is distinct from the logical offset:
attention sees only valid keys, and no tokens are evicted. Cache byte telemetry
includes spare capacity. Linear recurrent/conv state retains its existing layout.
The standalone `generate_report` API keeps its original cache policy.

`reuse_prefix=True` additionally retains an evaluated prefill snapshot. Reuse
requires the complete previous prompt to match token IDs, all three MRoPE axes
and image markers. Partial common prefixes rebuild prefill because recurrent
state cannot be cropped backwards. Decoder updates use separate array handles
from the snapshot. Prefix reuse can retain an extra cache and changes prefill
partitioning, so it is opt-in pending target-profile numerical qualification.
Metrics record `reused_prefix_tokens` and `cache_chunk_size`.

With the pinned template and `enable_thinking=False`, a completed assistant turn
does not retain the empty thinking block emitted by the generation prompt. The
ordinary follow-up therefore fails complete-prefix admission. The measured
two-turn conversation reused zero prefix tokens and retained an extra cache;
leave prefix reuse disabled for this workload. This does not affect reuse of
the loaded model and CT embeddings.

`tests/test_sessions.py` covers host lifecycle, source invalidation, exact-prefix
admission and failed-report history. The capacity/snapshot and prefix tests in
`tests/test_source_arithmetic.py` require Metal. Run with `MLX_ENABLE_TF32=0` and
qualify released-model follow-up tokens under the chosen precision profile before
adoption. Native FP32 timings and exact tokens for one synthetic CT are recorded
below; they do
not establish report quality or the other precision profiles.

## CT session measurement (2026-10-28)

Apple M5 Max with 128 GiB unified memory, macOS 27.2, Python 3.13.17,
MLX 0.32.4, `MLX_ENABLE_TF32=0`. The native FP32 model bundle was downloaded at
`MedMLX/MLX-Reason-CT` revision `f60c688952468934acc0630090ef08832ca88d31`;
the NVIDIA source checkpoint remains
`386b93e034983f6c1fc841a43833a1b6a0cd9c13`.

The input was a synthetic 192-cubed HU NIfTI, 2 mm isotropic: an ellipsoid with
radii `(70, 60, 80)`, center 95.5, interior `35 + 10*cos(z/12)` HU and -1000 HU
outside. `anatomy_region=None` selected the pinned centered crop. Each mode
answered `Answer only CT.` twice, carrying the first completed answer into the
second request, with thinking disabled and a 32-token upper limit. Prompts held
13,842 and 13,857 tokens. All 18 requests produced exactly `[1123, 248046]`
(`CT` and EOS), without truncation.

| Mode | First request median (s) | Follow-up median (s) | Follow-up range (s) | Follow-up peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: |
| `generate_report`, original cache | 18.394 | 18.309 | 17.107–18.821 | 21.19 |
| `CTSession`, capacity 256 | 18.283 | 15.706 | 14.502–15.769 | 21.20 |
| Session plus optional prefix snapshot | 17.802 | 15.524 | 14.795–15.744 | 22.11 |

Each mode ran in three fresh processes, reversing mode order on the middle
repetition. Both requests were measured; there was no excluded warmup. The
allocator cache was cleared before each request, but OS file caches were not
flushed. Timings include the public API's host preprocessing, model loading,
vision, prefill, greedy decoding and persisted outputs. Session follow-ups
reuse the model and projected CT embeddings. Completion was synchronized.
Peak is at least the active baseline, includes retained model/cache arrays and
excludes cached free allocations. Other host conversion jobs were present.

Model/CT reuse reduced median follow-up latency by **14.2%** on this short-answer
case. Peak active memory was essentially unchanged; the session keeps about
16.36 GiB active between requests to avoid repeating work. This case does not
measure long-response throughput or report accuracy. The capacity cache held
975,437,824 bytes before and after its one generated-token decode, versus
960,823,296 to 960,888,832 bytes for concatenation; spare capacity is counted.

Prefix snapshots reused **zero tokens** in every follow-up. The pinned template
omits the initial empty thinking block when formatting completed history; its
first mismatch is at index 13,838. Complete-prefix admission correctly rebuilt
prefill. The snapshot added about 0.91 GiB to follow-up peak memory and supplied
no demonstrated benefit; the small timing difference is not a prefix speedup.
Prefix reuse remains disabled by default. No tolerance or admission rule was
relaxed to force a hit. The two affected native capacity/snapshot and prefix
tests passed; these small references do not qualify other precision profiles.

Raw samples, output records and runnable probes are not included in this
repository. The source and arrays
are synthetic; model weights and generated output files are excluded from Git.

## Historical FP32 comparison

A task-output comparison recorded outside this repository at native revision
`eb2c10baf688b32eee082dbb8973e88b4555db7a` records greedy-token agreement against
the source CPU FP32 profile on one recorded case. Its result files are not
included here. This historical result does not establish original CUDA BF16
parity or diagnostic accuracy. The model/checkpoint revision remains pinned in
the technical notes above.

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
change. On this one case, the vision and hidden boundaries also matched the
recorded CUDA source values exactly; cached source-logit comparisons pass the
unchanged `atol=0.125`, `rtol=0.01`, normalized RMSE `<=0.005` gates. The CUDA
source values are not included in this repository. Exact tokens and EOS termination are retained.
Cached peak MLX allocation is approximately 12.37 GiB in both native runs.

The initial fast-K8/batching uncached pass was interrupted by a system GPU reset
(MLX reported an innocent-victim error).
A fresh-model retry reused the verified native vision features and first token,
then rebuilt the full prefix without a decoder cache. It reached EOS with
bitwise-unchanged native logits, zero source tolerance violations and normalized
RMSE 0.0000297. This is recovered continuation evidence, not an uninterrupted
full-model qualification run.

Isolated attention probes used one warmup and seven alternating synchronized
samples at the existing launch sizes: median speedups were 1.26 times for vision
and 1.36 times for decoder attention at full context. Projection batch probes
covered ten model shapes plus ragged, wide-exponent and partitioned cases; all
outputs matched bitwise. These probes support the execution changes, while the
full-model run checks their integration.

This evidence does not complete
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
qualification cases remain unqualified.

## FP32 triangular-prefill measurement (2026-10-28)

The 64-row triangular attention solve now uses one Metal dispatch per batch of
matrices, replacing 63 evaluated row updates. This applies only to FP32 prefill;
the source-BF16 path keeps its original reduction implementation. The baseline
was checkout `1d3ba549f98272b7a836e3d9b78e2649f5f90a1e`, with no prior local edits.

Measurements used Apple M5 Max, macOS 27.2 arm64, Python 3.13.17, MLX 0.32.4 and
`MLX_ENABLE_TF32=0`. Each case has batch 1, 16 query/key heads, 32 value heads,
key/value width 128 and a nonzero initial state. NumPy RNG seed 991 generates
normal queries, keys, values and initial state (scaled by 0.1); queries and keys
are L2 normalized, queries are additionally divided by sqrt(128), negative log
decays are uniform over [-0.2, -0.001], and beta is uniform over [0.1, 0.9].
Cases consume the generator in the order below. Inputs are materialized before
timing. Two warmups precede seven paired samples, with baseline/candidate order
alternating each sample. Each interval includes the complete `chunk_delta` call,
output evaluation and Metal synchronization.

| Tokens | Baseline median (range), ms | Candidate median (range), ms | Median speedup |
| --- | --- | --- | --- |
| 64 | 17.407 (16.994–20.487) | 3.014 (2.477–3.493) | 5.78x |
| 129 | 22.210 (19.835–27.727) | 7.122 (5.662–8.588) | 3.12x |
| 1,024 | 28.918 (28.428–32.554) | 12.951 (11.386–20.166) | 2.23x |
| 13,824 | 157.504 (151.523–167.298) | 103.330 (99.114–108.302) | 1.52x |

Output and final state matched the baseline bitwise for every FP32 timing case,
including the ragged 129-token case. Source-BF16 baseline/candidate outputs and
states also matched bitwise at 64 and 129 tokens. Ten targeted tests passed:
three exact triangular-expression comparisons (1, 32 and 513 matrices), five
independent recurrence/continuation cases (1, 63, 64, 65 and 129 tokens), and two
decay-scan references. Existing tolerances were unchanged. These are synthetic
prefill-operator results; no full CT model, checkpoint download, generated report
or full-model qualification was run. This does not establish whole-model latency
or clinical validity.
