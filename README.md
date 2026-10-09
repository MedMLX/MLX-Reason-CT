# MLX-Reason-CT

Native MLX port of [NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT)
for Apple Silicon. Run 3D CT reasoning and report generation locally on Mac.

- Chest and abdomen CT
- Structured reports and CT question answering
- Local NIfTI input with Apple Silicon GPU acceleration
- FP32 by default, with explicit BF16 arithmetic profiles on Metal

MLX port by **Joseph Sandoval**.
[Releases](https://github.com/MedMLX/MLX-Reason-CT/releases) ·
[Weights](https://huggingface.co/josand/MLX-Reason-CT) ·
[Usage](https://github.com/MedMLX/MLX-Reason-CT/blob/main/docs/usage.md) ·
[Paper](https://arxiv.org/abs/2609.27511) ·
[HF collection](https://huggingface.co/collections/josand/medmlx-mlx-reason-ct-6ac85b2663d2ca5df1887db2)

## Requirements

Supported platform: macOS arm64 on an Apple Silicon Mac with Metal GPU access,
Python 3.12 and [uv](https://docs.astral.sh/uv/). Inference fails explicitly when
Metal is unavailable.

`v0.2.3` uses the public Apache-2.0 `medmlx-core==0.1.2` runtime from PyPI.

Weights occupy 17.4 GB. Measured peak MLX memory use is about 22.7 GB;
allow additional unified memory for preprocessing, macOS and other apps.

## Quick start

```bash
uv tool install --python 3.12 mlx-reason-ct==0.2.3

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

For an existing Python 3.12 environment, use `pip install mlx-reason-ct==0.2.3`.
Wheel and source archives are also available in the
[GitHub release](https://github.com/MedMLX/MLX-Reason-CT/releases/tag/v0.2.3).
For source development, clone [this repository](https://github.com/MedMLX/MLX-Reason-CT)
and run `make env`; use
`uv run mlx-reason-ct` for the commands above.

For a generated input with no patient data, follow the
[synthetic CT walkthrough](https://github.com/MedMLX/MLX-Reason-CT/blob/v0.2.3/docs/synthetic-walkthrough.md). It covers installation,
bundle verification, input generation and completion checks.

## Python

```python
from mlx_reason_ct.medmlx import run

result = run(image_path="ct.nii.gz", weights_path="models", output_dir="outputs")
print(result["outputs"]["report"])
```

Takes the same options as the CLI (`prompt`, `anatomy_region`, `enable_thinking`,
`precision`, `overwrite`). The model is also discoverable by
[medmlx-mcp](https://github.com/MedMLX/medmlx-mcp) as `mlx-reason-ct`. Full runner
contract in [docs/usage.md](https://github.com/MedMLX/MLX-Reason-CT/blob/v0.2.3/docs/usage.md#medmlx-runner).

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
original BF16 checkpoint, without retraining. Implementation and verification
details are in [docs](https://github.com/MedMLX/MLX-Reason-CT/blob/v0.2.3/docs/usage.md). Seeded test references are recorded on
darwin-arm64; [verification scope](https://github.com/MedMLX/MLX-Reason-CT/blob/v0.2.3/docs/verification.md) explains their bounds
and what they do not establish.

Version 0.2.3 includes faster source-BF16 projection and attention kernels.
On one M1 Max synthetic case, cached full-model time fell from 17:54 in the
preceding source build to 14:49, with bitwise-unchanged recorded outputs.
Full BF16 qualification remains incomplete; see the
[benchmark and limits](https://github.com/MedMLX/MLX-Reason-CT/blob/v0.2.3/docs/verification.md#source-bf16-optimization-measurement-2026-10-09).

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Code: [Apache-2.0](https://github.com/MedMLX/MLX-Reason-CT/blob/main/LICENSE), Joseph Sandoval.
Weights: [OpenMDW-1.1](https://huggingface.co/josand/MLX-Reason-CT/blob/main/LICENSE), NVIDIA CORPORATION & AFFILIATES.
[Third-party notices](https://github.com/MedMLX/MLX-Reason-CT/blob/main/THIRD_PARTY_NOTICES.md).
