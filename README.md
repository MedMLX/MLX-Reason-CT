# MLX-Reason-CT

Native MLX port of [NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT)
for Apple Silicon. Run 3D CT reasoning and report generation locally on Mac.

- Chest and abdomen CT
- Structured reports and CT question answering
- Local NIfTI input with Apple Silicon GPU acceleration
- FP32 inference using MLX; no PyTorch or CUDA required

MLX port by **Joseph Sandoval**.
[PyPI](https://pypi.org/project/mlx-reason-ct/) ·
[Weights](https://huggingface.co/josand/MLX-Reason-CT) ·
[Usage](https://github.com/sandovaljoseph/MLX-Reason-CT/blob/main/docs/usage.md)

## Requirements

Apple Silicon Mac, macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/).
Weights occupy 17.4 GB. Measured peak MLX memory use is about 22.7 GB;
allow additional unified memory for preprocessing, macOS and other apps.

## Quick start

```bash
uv tool install --python 3.12 mlx-reason-ct

mlx-reason-ct download \
  --revision e15558ae30c8ad6c25c0bfcce7467cc72cd0b2f2 \
  --model-dir models
```

Generate a chest CT report:

```bash
mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs
```

For abdomen CT:

```bash
mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs \
  --anatomy-region abdomen
```

To ask a question about the CT, set `--prompt` to your question.

For an existing Python 3.12 environment, install with `pip install mlx-reason-ct`.
For source development, clone [this repository](https://github.com/sandovaljoseph/MLX-Reason-CT)
and run `make env`; use
`uv run mlx-reason-ct` for the commands above.

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
[docs](https://github.com/sandovaljoseph/MLX-Reason-CT/blob/main/docs/usage.md).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Code: [Apache-2.0](https://github.com/sandovaljoseph/MLX-Reason-CT/blob/main/LICENSE), Joseph Sandoval.
Weights: [OpenMDW-1.1](https://huggingface.co/josand/MLX-Reason-CT/blob/main/LICENSE), NVIDIA CORPORATION & AFFILIATES.
[Third-party notices](https://github.com/sandovaljoseph/MLX-Reason-CT/blob/main/THIRD_PARTY_NOTICES.md).
