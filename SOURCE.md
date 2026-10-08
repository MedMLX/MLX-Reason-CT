# Native NV-Reason-CT source provenance

NVIDIA: https://huggingface.co/nvidia/NV-Reason-CT/tree/386b93e034983f6c1fc841a43833a1b6a0cd9c13
(OpenMDW-1.1). RadNN source admission binds the pinned code, processor, chat template,
tokenizer, configurations and weights. The standalone converted manifest records
the source weight hash and verifies the staged runtime-file hashes. The staged upstream LICENSE is required.

The anatomy policy is derived from the pinned processor. Physical resampling
shape/affine arithmetic follows MONAI 1.6.0 (Apache-2.0). Primus follows
dynamic-network-architectures 0.4.3 (Apache-2.0), with EVA attention/SwiGLU/rotary
semantics from timm 1.0.22 (Apache-2.0). The multimodal merger, text decoder,
MRoPE and precision boundaries follow Transformers 5.10.4 (Apache-2.0).

The native recurrent Metal kernel is vendored from the published MLX-LM 0.32.0
`mlx_lm/models/gated_delta.py` (MIT, Copyright Apple), without arithmetic changes.
Its SHA-256 is `f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428`.
This module imports only standard-library and MLX modules, so native execution
does not load the MLX-LM package initializer or its Transformers dependencies.

The installed MLX-LM Qwen3.5 and Qwen3-Next implementations were inspected for
reuse. The reusable single-token gated-delta kernel has the required state and
Metal execution semantics. The native prefill, normalization, rotary and
activation owners preserve the pinned Transformers/CUDA arithmetic boundaries;
substituting the general language-model path did not establish those boundaries.

The native runtime runs all learned arithmetic on Metal. Standalone inference
defaults to `float32`. The RadNN adapter preserves its `bfloat16` default, which
selects the internal `source_bfloat16` implementation. Public `bfloat16_fp32`
selects the historical internal `bfloat16` accumulation profile. The historical
qualification results below describe RadNN receipts; they do not establish a
fresh qualification of this standalone package release.
CPU ownership: NIfTI IO, geometry, linear CT resampling, deterministic enclosed-air
crop selection, HU clipping/scaling, local Jinja rendering and Rust tokenizers.
Offline conversion expands the source BF16 bit patterns losslessly to FP32.
The disconnected 2D vision tower, inactive mask token and duplicate tied LM head
are recorded as exclusions; the source processor rejects 2D images and videos.

The CUDA reporting path retains its upstream BF16/SDPA configuration. The FP32
qualification oracle preserves the source's mathematical GQA attention and
bounds only its query workspace in 256-token blocks, avoiding the unchunked
11.42 GiB allocation. A memory-efficient SDPA surrogate was rejected after
full-model comparison showed it exceeded the fixed hidden-state tolerance.
All active source networks execute on CUDA; disconnected 2D parameters remain
on CPU. The `source_bfloat16` arithmetic retains the source default dispatch.

The original BF16 cache in Transformers 5.10.4 inherits its convolution dtype:
it stores the FP32 gated-delta recurrent result in BF16 and widens that stored
value when decoding the next token. The native diagnostic preserves this
boundary. Captured CUDA values independently test storage and reuse; the source
also exhibits shape-dependent BF16 GEMV/GEMM rounding between cached decoding
and full-sequence replay.

Qwen's source L2 normalization rounds its elementwise squares to BF16, sums
those values in FP32 opmath, then returns a BF16 sum and normalization output.
The native diagnostic widens only the summation operands to preserve these
boundaries. Ten captured CUDA Q/K arrays match exactly after this correction.

The original BF16 diagnostic also preserves the pinned RTX 4090 / cuBLAS 13.1.1
long-context GEMM storage boundaries. Transparent source captures and launch
metadata show two K slices for the 2560-to-32 A/B projections and three 3072-wide
K slices for the 9216-to-2560 MLP down projection. Each slice accumulates in FP32,
then stores BF16 C before the next slice. The native decoder keeps the original
full sequence length when applying this policy to its bounded MLP chunks;
single-token decode retains the source GEMV behavior. Independent sparse CUDA
half-way fixtures detect missing stores and dropped chunk context. This is a
correction to the native original-profile implementation; source arithmetic,
source dispatch, fixed gates and public profiles are unchanged. It does not
resolve accumulated original-BF16 cross-backend drift or source self-cache failure.

The separately qualified internal `bfloat16` profile (public `bfloat16_fp32`)
changes the numerical contract on
both qualification hosts. Parameters remain BF16. The patch convolution keeps
its BF16 output and bias boundary; later vision projections use BF16-rounded
operands with FP32 outputs. Vision normalization, attention and residuals, the
multimodal merger outputs, and the entire decoder use FP32 arithmetic. Decoder
convolution, recurrent state and full-attention KV storage stay FP32. This is
not parity with NVIDIA's original BF16 execution. The public CUDA path remains
the original upstream profile; the alternative oracle is qualification-only.

The multi-token decoder path implements Transformers' 64-token chunked
gated-delta algebra. Its decay prefix scan follows PyTorch 2.13.0
`ATen/native/cuda/ScanUtils.cuh` (BSD terms retained in `src/mlx_reason_ct/licenses/PyTorch-BSD.txt`).
The source's row-dependent Sklansky summation order matters when cumulative
decays are subtracted inside exponentials. Single-token decode reuses the
MLX-LM kernel. CUDA SiLU, sigmoid and Softplus formulas and their FP32/BF16
cast boundaries are retained explicitly; mathematically equivalent activation
formulas caused measurable long-context drift during qualification.

Activation formulas were checked directly against PyTorch 2.13.0's
[SiLU](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu)
and [Softplus](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu)
kernels. The installed PyTorch source revision is
`cf30153c4c131c8164ee7798e5022d810682e2cb`.

The original-source BF16 profile uses `source_arithmetic.py` and its packaged
Metal kernels. They preserve measured RTX4090 HMMA K8 carries, GEMVX lane
reduction, Welford/mean statistics, BF16 stores, FP32 recurrence, and Flash
denominator/LSE/combine order. The source Flash submodule is
`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`; fragment/layout helpers come from
MLX0.32.2. MIT/BSD notices are retained in `src/mlx_reason_ct/licenses/`. The source singleton
GQA16/4 head256 dispatch has independent CUDA controls at both sides of every
54..64 split transition, spanning13824..16384 keys.

The hash-bound unary assets losslessly encode model-independent CUDA rsqrt,
MUFU.EX2/LG2, finite-BF16 SiLU/sigmoid and fixed nonlearned Primus rotary values.
They contain no learned parameters, neural intermediates or case-specific
outputs. CPU decoding restores their original uint32 bytes; learned arithmetic
executes on Metal. FP32 and the alternate BF16-storage/FP32-accumulation profile
retain their existing arithmetic. Integration alone does not qualify the
original profile; the current acceptance contract is recorded below.

The BF16 activation payloads numerically widen each of the 65,536 uint16
storage patterns to its own uint32 table entry. Packing pairs by reinterpreting
uint16 bytes as uint32 changes both the table size and the kernel lookup value.
All entries are checked against the independently captured source tables.

RMS squared means retain the pinned ATen `Reduce.cuh` row-dependent dispatch.
Four-element input vectors use 32 to 512 lanes according to reduction width
and output row count, followed by shared-memory and warp reductions in source
order. Cached single-row RMS and long prefill RMS need different trees: a
one-ULP mean difference can change a BF16 halfway result and amplify through
later layers. Independent CUDA fixtures cover all three supported widths and
both sides of the row-count transitions. FP32 and the alternative profile keep
their existing MLX reductions.

Public profile selection changes no learned arithmetic or tolerance. Original
BF16 source-matched cached generation passes all 507 consumed logits/tokens
across four cases with the restored execution implementation. Under the
explicitly authorized `original_bf16_mode_matched_20261006` contract, source and
native cached-versus-uncached numeric differences remain failed diagnostics.
Every mode-matched numeric comparison, exact token/text comparison, and complete
cached and true-uncached trace remains required. Original failed receipts and
their hashes are preserved. Full qualification remains incomplete while the
three longer native true-uncached greedy traces are pending.

The provisional projection operand reuse and larger row/MLP batches were
rejected after the bounded full-prefix check: unchanged logits and passing
source parity, but 1078.8851 seconds versus the saved 1035.1445-second baseline.
Both execution files were restored byte-for-byte from their pre-change backups.
The decision and archived candidate files are under
`build/nv-reason-ct/bf16-verify-20261006/`. These single-prefix timings establish
no general performance gain; precision, source cast boundaries and tolerances
remain unchanged.
