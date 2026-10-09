---
license: openmdw-1.1
language:
- en
library_name: mlx
base_model: nvidia/NV-Reason-CT
pipeline_tag: image-text-to-text
inference: false
tags:
- mlx
- native-mlx
- apple-silicon
- fp32
- ct
- medical-imaging
- nv-reason-ct
- 3d-vlm
- vision-language
- radiology
- report-generation
---

# MLX-Reason-CT

Native MLX port of [NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT)
for Apple Silicon. Run 3D CT reasoning and report generation locally on Mac.

- Chest and abdomen CT
- Structured reports and CT question answering
- Local NIfTI input with Apple Silicon GPU acceleration
- FP32 by default, with explicit BF16 arithmetic profiles on Metal

MLX port by **Joseph Sandoval**.
[GitHub releases](https://github.com/MedMLX/MLX-Reason-CT/releases) ·
[Usage](https://github.com/MedMLX/MLX-Reason-CT/blob/main/docs/usage.md) ·
[Paper](https://arxiv.org/abs/2609.27511) ·
[HF collection](https://huggingface.co/collections/josand/medmlx-mlx-reason-ct-6ac85b2663d2ca5df1887db2)

**Requires the companion `mlx-reason-ct` runtime.** Follow the quick start below.
Hugging Face's generated "Use this model" snippet uses `mlx-vlm`, which does not
support this 3D CT architecture; generic `mlx-lm` loaders are also incompatible.

## Requirements

Supported platform: macOS arm64 on an Apple Silicon Mac with Metal GPU access,
Python 3.12 and [uv](https://docs.astral.sh/uv/). Inference fails explicitly when
Metal is unavailable.

`v0.2.2` uses the public Apache-2.0 `medmlx-core@v0.1.2` runtime.

Weights occupy 17.4 GB. Measured peak MLX memory use is about 22.7 GB;
allow additional unified memory for preprocessing, macOS and other apps.

## Quick start

```bash
uv tool install --python 3.12 \
  https://github.com/MedMLX/MLX-Reason-CT/releases/download/v0.2.2/mlx_reason_ct-0.2.2-py3-none-any.whl

mlx-reason-ct download \
  --revision c690a63888b9c6c9bd006687335fbd650eb60275 \
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

For an existing Python 3.12 environment, install the same wheel with `pip install`.
For source development, clone [the repository](https://github.com/MedMLX/MLX-Reason-CT)
and run `make env`; use `uv run mlx-reason-ct` for the commands above.

For a generated input with no patient data, follow the
[synthetic CT walkthrough](https://github.com/MedMLX/MLX-Reason-CT/blob/main/docs/synthetic-walkthrough.md).
It covers installation, bundle verification, input generation and completion checks.

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

Inference defaults to FP32 with weights converted directly from the
original BF16 checkpoint, without retraining.
[Implementation and verification details](https://github.com/MedMLX/MLX-Reason-CT/blob/main/docs/usage.md),
and the limits of its darwin-arm64 component references are in the companion repository.
[Upstream model card and limitations](https://huggingface.co/nvidia/NV-Reason-CT).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Weights: [OpenMDW-1.1](https://huggingface.co/josand/MLX-Reason-CT/blob/main/LICENSE),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
Underlying Qwen3.5 terms: [Apache-2.0](https://huggingface.co/josand/MLX-Reason-CT/blob/main/APACHE-2.0.txt).

Port code: [Apache-2.0](https://github.com/MedMLX/MLX-Reason-CT/blob/main/LICENSE),
Copyright (c) 2026 Joseph Sandoval.
[Third-party notices](https://github.com/MedMLX/MLX-Reason-CT/blob/main/THIRD_PARTY_NOTICES.md).
