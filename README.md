# MLX-Reason-CT

Native MLX port of [NVIDIA NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT)
for Apple Silicon. Run 3D CT reasoning and report generation locally on Mac.

- Chest and abdomen CT
- Structured reports and CT question answering
- Local NIfTI input with Apple Silicon GPU acceleration
- FP32 by default, with explicit BF16 arithmetic profiles on Metal

MLX port by **Joseph Sandoval**.
[PyPI](https://pypi.org/project/mlx-reason-ct/) ·
[Weights](https://huggingface.co/josand/MLX-Reason-CT) ·
[Usage](https://github.com/MedMLX/MLX-Reason-CT/blob/main/docs/usage.md)

## Requirements

Supported platform: macOS arm64 on an Apple Silicon Mac with Metal GPU access,
Python 3.12 and [uv](https://docs.astral.sh/uv/). Inference fails explicitly when
Metal is unavailable.
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
For source development, clone [this repository](https://github.com/MedMLX/MLX-Reason-CT)
and run `make env`; use
`uv run mlx-reason-ct` for the commands above.

## MedMLX runner

The package registers `mlx-reason-ct` in the `medmlx.models` entry point group
with task `generation`, for discovery by `medmlx-mcp`. Importing the package or
its declaration does not load MLX; runtime readiness is checked separately.

```python
from mlx_reason_ct.medmlx import run

result = run(
    image_path="ct.nii.gz",
    weights_path="models",
    output_dir="outputs",
    anatomy_region="chest",
    max_new_tokens=512,
)
print(result["outputs"]["report"])
```

`weights_path` is the explicit converted bundle root containing
`mlx/manifest.json`, the FP32 shards, tokenizer and chat template, as produced
by `mlx-reason-ct download` or `mlx-reason-ct convert`. It is a directory.
All arguments are keyword-only. Set `prompt` to a CT question,
`enable_thinking=True` to retain reasoning, or `anatomy_region="abdomen"` for
an abdominal report. `precision="float32"` remains the default;
`precision="bfloat16"` selects the source BF16 arithmetic profile and
`precision="bfloat16_fp32"` selects BF16 weights with FP32 accumulation and
decoding. These profiles retain the upstream qualification limits described in
[SOURCE.md](SOURCE.md). Generation remains greedy. Existing output files require
`overwrite=True`; this guard also applies to the public `generate_report` API.
The CLI accepts `--overwrite` for the same action and `--precision` for profile selection.

The return value contains `outputs` (`report`, `response`, `run`) as absolute
POSIX paths and JSON-compatible `details`. Missing or mismatched bundles raise
`medmlx_core.AssetNotReadyError`; bad inputs raise `InvalidInputError`,
unavailable dependencies or Metal raise `MissingDependencyError`, and generation
failures raise `ModelExecutionError`. Empty or truncated responses retain partial
files and failure metadata, then raise; increase `max_new_tokens` for truncation.
JSON records carry `schema_version=1`. Reports require human review.

Runner tests cover synthetic input, asset validation and report orchestration.
The fixed production network has no tiny configurable variant; these tests do
not qualify learned report generation or clinical performance.

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
details are in [docs](docs/usage.md). Seeded test references are recorded on
darwin-arm64; [verification scope](docs/verification.md) explains their bounds
and what they do not establish.

## Intended use

Intended for research and education, not clinical diagnosis or treatment decisions.
Outputs require human review.

## License

Code: [Apache-2.0](https://github.com/MedMLX/MLX-Reason-CT/blob/main/LICENSE), Joseph Sandoval.
Weights: [OpenMDW-1.1](https://huggingface.co/josand/MLX-Reason-CT/blob/main/LICENSE), NVIDIA CORPORATION & AFFILIATES.
[Third-party notices](https://github.com/MedMLX/MLX-Reason-CT/blob/main/THIRD_PARTY_NOTICES.md).
