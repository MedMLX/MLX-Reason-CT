# MLX-Reason-CT

Native MLX port of [NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT)
for Apple Silicon. Run 3D CT reasoning and report generation locally on Mac.

- Chest and abdomen CT
- Structured reports and CT question answering
- Local NIfTI input with Apple Silicon GPU acceleration
- FP32 inference using MLX; no PyTorch or CUDA required

MLX port by **Joseph Sandoval**.
[Weights](https://huggingface.co/josand/MLX-Reason-CT) · [Usage](docs/usage.md)

## Requirements

Apple Silicon Mac, macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/).
Weights occupy 17.4 GB. Measured peak MLX memory use is about 22.7 GB;
allow additional unified memory for preprocessing, macOS and other apps.

## Quick start

```bash
git clone https://github.com/sandovaljoseph/MLX-Reason-CT.git
cd MLX-Reason-CT
make env

uv run mlx-reason-ct download \
  --revision e15558ae30c8ad6c25c0bfcce7467cc72cd0b2f2 \
  --model-dir models
```

Generate a chest CT report:

```bash
uv run mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs
```

For abdomen CT:

```bash
uv run mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs \
  --anatomy-region abdomen
```

To ask a question about the CT, set `--prompt` to your question.

## Input and output

Input is one 3D CT volume in NIfTI format (`.nii` or `.nii.gz`) with Hounsfield
Unit values and valid spatial geometry. DICOM and 2D images are unsupported.

The upstream crop locates the chest from enclosed air. On whole-body scans,
especially with arms raised, it can select the head and neck instead; crop such
volumes to the chest or abdomen before running.

Each run writes:

- `report.txt` — generated response
- `model_response.json` — response, reasoning (with `--enable-thinking`) and generation metadata
- `run.json` — execution metadata

## Technical details

This release uses FP32 inference with weights converted directly from the
original BF16 checkpoint, without retraining. Implementation and verification
details, including token-exact parity with the CUDA source model on real CT, are in
[docs](docs/usage.md).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Code: [Apache-2.0](LICENSE), Joseph Sandoval.
Weights: [OpenMDW-1.1](THIRD_PARTY_NOTICES.md#nvidia-openmdw-11), NVIDIA CORPORATION & AFFILIATES.
[Third-party notices](THIRD_PARTY_NOTICES.md).
