# NV-Reason-CT MLX

Standalone FP32 [NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT) on Apple Silicon. MLX port by **Joseph Sandoval**.
[Weights](https://huggingface.co/josand/NV-Reason-CT-MLX) · [Details](docs/usage.md)

Requires macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/).
Weights: 17.4 GB. Observed MLX memory use: ~22.4 GB, plus system overhead.

## Quick start

```bash
git clone https://github.com/sandovaljoseph/NV-Reason-CT-MLX.git
cd NV-Reason-CT-MLX
make env

uv run nv-reason-ct-mlx download \
  --revision e15558ae30c8ad6c25c0bfcce7467cc72cd0b2f2 --model-dir models

uv run nv-reason-ct-mlx report \
  --input ct.nii.gz --model-dir models --output-dir outputs
```

Input: one 3D HU NIfTI CT. Output: report text, response JSON and run JSON.
For abdomen, add `--anatomy-region abdomen --prompt "write a structured abdomen CT report"`.

FP32 weights preserve the original BF16 values exactly. Fresh standalone GPU
verification is pending; [completed checks](docs/standalone-verification.json).

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

Code: [Apache-2.0](LICENSE), Joseph Sandoval.
Weights: [OpenMDW-1.1](THIRD_PARTY_NOTICES.md#nvidia-openmdw-11), NVIDIA CORPORATION & AFFILIATES.
[Third-party notices](THIRD_PARTY_NOTICES.md).
