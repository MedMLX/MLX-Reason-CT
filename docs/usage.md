# Usage

Start with the [synthetic CT walkthrough](synthetic-walkthrough.md) to exercise
installation and the report workflow using a generated geometric phantom.

## CT question answering

Use `--prompt` to ask a question about the volume. Set `--anatomy-region` to
`chest` (the default) or `abdomen` to match the CT.

```bash
uv run mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs \
  --prompt "What imaging modality is shown? Answer briefly."
```

## Python API

The examples below use `mlx-reason-ct` 0.2.3, which includes the MedMLX runner,
explicit precision profiles and overwrite controls.

```python
from mlx_reason_ct import generate_report

generate_report(
    "ct.nii.gz",
    "outputs",
    model_dir="models",
    anatomy_region="chest",
    prompt="write a structured chest CT report",
    max_new_tokens=512,
)
```

Without `--prompt`, a structured report for `--anatomy-region` is requested.
Inference runs locally on the Apple Silicon GPU, using FP32 by default. Add `--enable-thinking` to
use the source template's thinking mode; the reasoning is saved as `thinking` in
`model_response.json` and `report.txt` holds only the answer. Each request owns
its decoder state.

Input must be a finite scalar 3D HU NIfTI CT with a coded spatial transform.
Conflicting qform/sform transforms are rejected. Physical units are normalized
to millimeters. Preprocessing warns when source anatomy selection falls back
to a superior-edge crop. DICOM, 2D images, video and multiple-volume prompts
are unsupported.

Outputs: `report.txt`, `model_response.json`, `run.json`. An empty or truncated
response raises an error after retaining partial output and a failed completion
record. Increase `--max-new-tokens` if needed; the supported range is 1–8192.

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
decoding. Their numerical behavior and verification limitations are described in
[SOURCE.md](../SOURCE.md). Generation remains greedy. Existing output files require
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
not establish learned report accuracy or clinical performance.

## Weights

```bash
uv run mlx-reason-ct verify --model-dir models
uv run mlx-reason-ct convert --source-dir /path/to/upstream --output-dir models
```

Conversion requires the complete pinned upstream snapshot at revision
`386b93e034983f6c1fc841a43833a1b6a0cd9c13`. It expands BF16 bit patterns to FP32
without retraining, rejects changed source assets and validates tied weights.
The disconnected 2D tower, unused mask token and duplicate LM head are excluded.
The converted bundle verifies runtime metadata and all 35 shard hashes.

The published model bundle contains:

| Purpose | Files |
| --- | --- |
| Model presentation | `README.md`, `.gitattributes` |
| License and attribution | NVIDIA `LICENSE`, Qwen `APACHE-2.0.txt` |
| Model configuration | `config.json`, `generation_config.json` |
| Text processing | `chat_template.jinja`, `tokenizer.json` |
| CT preprocessing | `image_processor_3d/preprocessor_config.json` |
| Converted weights | `mlx/manifest.json`, 35 FP32 safetensors shards |

## Implementation

The runtime implements the 3D vision encoder, multimodal projection and Qwen3.5
hybrid decoder in MLX, including recurrent state and the attention KV cache.
CT loading and preprocessing run on the host. The supported platform is macOS
arm64 on Apple Silicon with MLX on Metal. Learned execution has no host fallback
and fails explicitly when Metal is unavailable.

## Development and verification

```bash
make env
make qa
make build
```

[Verification scope](verification.md) describes the darwin-arm64 references,
rounding bounds, host-stage checks and Metal tests. These checks establish no
full-model parity, clinical validation or report accuracy.

See [third-party terms](../THIRD_PARTY_NOTICES.md).
