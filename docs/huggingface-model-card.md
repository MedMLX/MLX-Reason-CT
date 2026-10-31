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
[PyPI](https://pypi.org/project/mlx-reason-ct/) ·
[Usage](https://github.com/MedMLX/mlx-reason-ct/blob/main/docs/usage.md) ·
[Paper](https://arxiv.org/abs/2609.27511) ·
[Weights](https://huggingface.co/MedMLX/MLX-Reason-CT)

**Requires the companion `mlx-reason-ct` runtime.** Follow the quick start below.
Hugging Face's generated "Use this model" snippet uses `mlx-vlm`, which does not
support this 3D CT architecture; generic `mlx-lm` loaders are also incompatible.

## Requirements

Supported platform: macOS arm64 on an Apple Silicon Mac with Metal GPU access,
Python 3.13 or 3.14 and [uv](https://docs.astral.sh/uv/). Inference fails explicitly when
Metal is unavailable.

The current release uses the public Apache-2.0 `medmlx-core==0.2.5` runtime from PyPI.

Weights occupy 17.4 GB. Measured peak MLX memory use is about 22.7 GB;
allow additional unified memory for preprocessing, macOS and other apps.

## Quick start

```bash
uv tool install --python 3.13 mlx-reason-ct

mlx-reason-ct download \
  --repo MedMLX/MLX-Reason-CT \
  --revision 5e5b5202fd04ac9f63ede203ebc9327aa676ae94 \
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

To ask a question about the CT, set `--prompt` to your question. Use
`--anatomy-region none` for the processor's plain centered crop and `--history`
for follow-up questions about the same CT; see the
[usage guide](https://github.com/MedMLX/mlx-reason-ct/blob/main/docs/usage.md).

For an existing Python 3.13 or 3.14 environment, use
`pip install mlx-reason-ct`. Wheel and source archives are also
available on [PyPI](https://pypi.org/project/mlx-reason-ct/).
For source development, clone [the repository](https://github.com/MedMLX/mlx-reason-ct)
and run `make env`; use `uv run mlx-reason-ct` for the commands above.

For a generated input with no patient data, follow the
[synthetic CT walkthrough](https://github.com/MedMLX/mlx-reason-ct/blob/main/docs/synthetic-walkthrough.md).
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

The runtime removes a trailing empty `<answer></answer>` element from the saved
report in both text and JSON, while preserving reasoning and nonempty answer elements.

## Technical details

Inference defaults to FP32 with weights converted directly from the
original BF16 checkpoint, without retraining.
[Implementation and verification details](https://github.com/MedMLX/mlx-reason-ct/blob/main/docs/technical.md),
and the limits of its darwin-arm64 component references are in the companion repository.
[Upstream model card and limitations](https://huggingface.co/nvidia/NV-Reason-CT).

The v0.3.2 release passed 144 synthetic-fixture tests and completed a default
FP32 report from a freshly installed wheel on a synthetic CT (180 generated tokens,
EOS reached, text and JSON outputs consistent). This verifies execution and the
documented component contracts; it does not establish clinical accuracy or
full-model agreement with the original BF16 implementation.

Version 0.2.3 includes faster source-BF16 projection and attention kernels.
On one M1 Max synthetic case, cached full-model time fell from 17:54 in the
preceding source build to 14:49, with bitwise-unchanged recorded outputs.
Full BF16 qualification remains incomplete; see the
[benchmark and limits](https://github.com/MedMLX/mlx-reason-ct/blob/main/docs/technical.md#source-bf16-optimization-measurement-2026-10-09).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Weights: [OpenMDW-1.1](https://huggingface.co/MedMLX/MLX-Reason-CT/blob/main/LICENSE),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
Underlying Qwen3.5 terms: [Apache-2.0](https://huggingface.co/MedMLX/MLX-Reason-CT/blob/main/APACHE-2.0.txt).

Port code: [Apache-2.0](https://github.com/MedMLX/mlx-reason-ct/blob/main/LICENSE).
[Third-party notices](https://github.com/MedMLX/mlx-reason-ct/blob/main/THIRD_PARTY_NOTICES.md).
