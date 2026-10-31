# Usage

Start with the [synthetic CT walkthrough](synthetic-walkthrough.md) to exercise
installation and the report workflow using a generated geometric phantom.

This page documents the current source checkout; commands with `uv run` assume
`make env` has been run there. The published package includes plain
centered cropping (`--anatomy-region none` / `anatomy_region=None`) and follow-up
history (`--history` / `history`).

## CT question answering

Use `--prompt` to ask a question about the volume. Set `--anatomy-region` to
`chest` (the default) or `abdomen` to match the CT.

`none` selects the pinned processor's own plain
centered crop (upstream's own fallback for geometries where its chest/abdomen
heuristic is unreliable). Pass `--history` with a path to a JSON file of prior
`[question, answer]` turns to ask a follow-up question about the same CT,
matching upstream's documented multi-turn/follow-up support; only the first
turn carries the CT volume.

```bash
uv run mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs \
  --prompt "What imaging modality is shown? Answer briefly."
```

## Python API

The Python examples below also work with the published `mlx-reason-ct`
package.

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
`model_response.json` and `report.txt` holds only the answer. Each one-shot
`generate_report` request recomputes its decoder state from `history` and the new
`prompt`. The separate [`CTSession` API](technical.md#reusable-ct-conversation-sessions)
can reuse the model and CT embeddings across requests.

Version 0.2.4 removes a trailing empty `<answer></answer>` element from the saved
report in both text and JSON, while preserving reasoning and nonempty answer elements.

Input must be a finite scalar 3D HU NIfTI CT with a coded spatial transform.
Conflicting qform/sform transforms are rejected. Physical units are normalized
to millimeters; unknown spatial units are assumed to be millimeters. HU values are
required by the model but cannot be verified from NIfTI metadata. Preprocessing warns when a requested chest/abdomen region
falls back to a superior-edge crop. Pass
`anatomy_region=None`/`--anatomy-region none` for a plain centered crop instead,
with no such warning. Upstream's own
model card scopes input to one NIfTI CT volume with HU intensities per
request; DICOM, 2D images and video are unsupported here and upstream alike.
Use `history` for follow-up questions about that same volume.

Outputs: `report.txt`, `model_response.json`, `run.json`. An empty or truncated
response raises an error after retaining partial output and a failed completion
record. Increase `--max-new-tokens` if needed; the supported range is 1–8192.

## MedMLX runner

The package registers `mlx-reason-ct` in the `medmlx.models` entry point group
with task `generation`, for discovery by `medmlx`. Importing the package or
its declaration does not load MLX; `medmlx` probes the runtime in an isolated
process before inference.

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

`weights_path` is the local bundle root downloaded from `MedMLX/MLX-Reason-CT`.
It contains `mlx/manifest.json`, the 35 FP32 shards under `mlx/`, and the
runtime files listed below.
All arguments are keyword-only. Set `prompt` to a CT question,
`enable_thinking=True` to retain reasoning, or `anatomy_region="abdomen"` for
an abdominal report. `anatomy_region=None` selects a
plain centered crop with no chest/abdomen heuristic. Pass
`history=[(question, answer), ...]` with prior completed turns, oldest
first, to ask a follow-up question about the same CT; `prompt` (or the default
structured-report request) is the new, still-unanswered turn, and only the
first turn carries the CT volume.
`precision="float32"` remains the default;
`precision="bfloat16"` selects the source BF16 arithmetic profile and
`precision="bfloat16_fp32"` selects BF16 weights with FP32 accumulation and
decoding. Their numerical behavior and verification limitations are described in
[source and arithmetic contracts](technical.md). Generation remains greedy. Existing output files require
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
The network has a fixed full size with no tiny configurable variant; these tests
do not establish learned report accuracy or clinical performance.

## Tool conversations

`generate_report` and the MedMLX `run` provider accept `tools` (native function
schemas), `tool_history` (alternating assistant/tool text exchanges, ending in a
tool result), and optional `response_choices` (complete permitted responses).
The opening `prompt` carries the CT once. Tool history is exclusive with the
ordinary Q/A `history`; the checkpoint's own template renders the schemas and
tool messages. Returned call text is inert: the caller validates and executes it.

Install the `tools` extra to use `response_choices` with `enable_thinking=False`.
LLGuidance constrains greedy selection to canonical tokenizations of the supplied
choices, including special tool tokens. Logits and argmax remain on MLX Metal;
unconstrained generation retains the existing numerical path. A caller must
still reject incomplete responses and enforce tool permissions and arguments.
This interface does not establish native tool-following accuracy or clinical
qualification for the checkpoint.

The MedMLX source checkout includes a `reason-ct-tools` workflow combining
nodule detection, MedSAM2 masks, physical measurements and local review.

## Weights

```bash
uv run mlx-reason-ct download --repo MedMLX/MLX-Reason-CT \
  --revision 5e5b5202fd04ac9f63ede203ebc9327aa676ae94 \
  --model-dir models
uv run mlx-reason-ct verify --model-dir models
```

The published bundle expands the pinned source BF16 weights to FP32 without
retraining and excludes the disconnected 2D tower, unused mask token and
duplicate LM head. `verify` checks manifest metadata and the presence of runtime
files and all 35 shards; it does not read their contents or establish integrity.
The CLI `download` command defaults to `MedMLX/MLX-Reason-CT`; `--repo` overrides it.

The published model bundle contains:

| Purpose | Files |
| --- | --- |
| Model presentation | `README.md`, `.gitattributes` |
| License and attribution | NVIDIA `LICENSE`, Qwen `APACHE-2.0.txt` |
| Model configuration | `config.json`, `generation_config.json` |
| Text processing | `chat_template.jinja`, `tokenizer.json` |
| CT preprocessing | `image_processor_3d/preprocessor_config.json` |
| Bundle manifest | `mlx/manifest.json` |
| Converted weights | `mlx/vision.safetensors`, `mlx/embedding.safetensors`, `mlx/norm.safetensors`, `mlx/layer_00.safetensors` through `mlx/layer_31.safetensors` (35 FP32 shards) |

## Implementation

The runtime implements the 3D vision encoder, multimodal projection and Qwen3.5
hybrid decoder in MLX, including recurrent state and the attention KV cache.
CT loading and preprocessing run on the host. The supported platform is macOS
arm64 on Apple Silicon with MLX on Metal. Learned execution has no host fallback
and fails explicitly when Metal is unavailable.

## Development and verification

Use the existing environment and select checks for the changed behavior:

```bash
make qa  # static checks only
make test TESTS=tests/path/to/test.py PYTEST_ARGS=-q
```

Run `make env` only when preparing or updating dependencies. Package builds
(`make build`) and full model checks are explicit tasks; documentation or
metadata changes do not require them.

[Verification scope](verification.md) describes the darwin-arm64 references,
rounding bounds, host-stage checks and Metal tests. These checks establish no
full-model parity, clinical validation or report accuracy.

See [third-party terms](../THIRD_PARTY_NOTICES.md).
