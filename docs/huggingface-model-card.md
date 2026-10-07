---
license: openmdw-1.1
language:
- en
pipeline_tag: image-text-to-text
base_model: nvidia/NV-Reason-CT
tags:
- native-mlx
- apple-silicon
- fp32
- ct
- medical-imaging
- nv-reason-ct
---

# NV-Reason-CT MLX

Native MLX port of [NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT)
for Apple Silicon. Run 3D CT reasoning and report generation locally on Mac.

- Chest and abdomen CT
- Structured reports and CT question answering
- Local NIfTI input with Apple Silicon GPU acceleration
- FP32 inference using MLX; no PyTorch or CUDA required

MLX port by **Joseph Sandoval**.
[GitHub](https://github.com/sandovaljoseph/NV-Reason-CT-MLX) ·
[Usage](https://github.com/sandovaljoseph/NV-Reason-CT-MLX/blob/main/docs/usage.md)

**Requires the companion `nv-reason-ct-mlx` runtime.** Generic `mlx-vlm` and
`mlx-lm` examples do not support this 3D CT model.

## Requirements

Apple Silicon Mac, macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/).
Weights occupy 17.4 GB. Measured peak MLX memory use is about 22.4 GB;
allow additional unified memory for preprocessing, macOS and other apps.

## Quick start

```bash
git clone https://github.com/sandovaljoseph/NV-Reason-CT-MLX.git
cd NV-Reason-CT-MLX
make env

uv run nv-reason-ct-mlx download \
  --revision e15558ae30c8ad6c25c0bfcce7467cc72cd0b2f2 \
  --model-dir models
```

Generate a chest CT report:

```bash
uv run nv-reason-ct-mlx report \
  --input ct.nii.gz --model-dir models --output-dir outputs
```

For abdomen CT:

```bash
uv run nv-reason-ct-mlx report \
  --input ct.nii.gz --model-dir models --output-dir outputs \
  --anatomy-region abdomen --prompt "write a structured abdomen CT report"
```

To ask a question about the CT, set `--prompt` to your question.

## Input and output

Input is one 3D CT volume in NIfTI format (`.nii` or `.nii.gz`) with Hounsfield
Unit values and valid spatial geometry. DICOM and 2D images are unsupported.

Each run writes:

- `report.txt` — generated response
- `model_response.json` — response and generation metadata
- `run.json` — execution metadata

## Technical details

This release uses FP32 inference with weights converted directly from the
original BF16 checkpoint, without retraining.
[Implementation and verification details](https://github.com/sandovaljoseph/NV-Reason-CT-MLX/blob/main/docs/usage.md),
including the pending standalone GPU check, are in the companion repository.
[Upstream model card and limitations](https://huggingface.co/josand/NV-Reason-CT-MLX/blob/main/upstream_model_card.md).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Weights: [OpenMDW-1.1](https://huggingface.co/josand/NV-Reason-CT-MLX/blob/main/LICENSE),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
Underlying Qwen3.5 terms: [Apache-2.0](https://huggingface.co/josand/NV-Reason-CT-MLX/blob/main/APACHE-2.0.txt).

Port code: [Apache-2.0](https://github.com/sandovaljoseph/NV-Reason-CT-MLX/blob/main/LICENSE),
Copyright (c) 2026 Joseph Sandoval.
[Third-party notices](https://github.com/sandovaljoseph/NV-Reason-CT-MLX/blob/main/THIRD_PARTY_NOTICES.md).
