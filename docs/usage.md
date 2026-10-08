# Usage

## CT question answering

Use `--prompt` to ask a question about the volume. Set `--anatomy-region` to
`chest` (the default) or `abdomen` to match the CT.

```bash
uv run mlx-reason-ct report \
  --input ct.nii.gz --model-dir models --output-dir outputs \
  --prompt "What imaging modality is shown? Answer briefly."
```

## Python API

```python
from mlx_reason_ct import generate_report

generate_report(
    "ct.nii.gz", "outputs",
    model_dir="models",
    anatomy_region="chest",
    prompt="write a structured chest CT report",
    max_new_tokens=512,
)
```

Without `--prompt`, a structured report for `--anatomy-region` is requested.
Inference is local and uses FP32 MLX on the Apple Silicon GPU. Add `--enable-thinking` to
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

## Weights

```bash
uv run mlx-reason-ct verify --model-dir models
uv run mlx-reason-ct convert --source-dir /path/to/upstream --output-dir models
```

Conversion requires the complete pinned upstream snapshot at revision
`386b93e034983f6c1fc841a43833a1b6a0cd9c13`. It expands BF16 bit patterns to FP32
without retraining, rejects changed source assets and validates tied weights.
The disconnected 2D tower, unused mask token and duplicate LM head are excluded.
The portable bundle verifies runtime metadata and all 35 shard hashes.

## Implementation

The runtime implements the 3D vision encoder, multimodal projection and Qwen3.5
hybrid decoder in MLX, including recurrent state and the attention KV cache.
CT loading and preprocessing run on CPU. Inference requires macOS on Apple
Silicon; preprocessing, conversion and bundle verification are portable.

## Development and verification

```bash
make qa
make build
```

[Historical FP32 qualification](fp32-engineering-qualification.json) covers four
synthetic cases. [Standalone verification](standalone-verification.json) records
package, CPU preprocessing and weight checks. [Real-CT parity](real-ct-cuda-parity.json)
records native generation on four de-identified real CT volumes (five requests,
1,320 generated tokens) matching the pinned FP32 PyTorch CUDA model token for
token, with image embeddings and first-step logits within the FP32 tolerances.
These checks establish no clinical or report accuracy.

See [third-party terms](../THIRD_PARTY_NOTICES.md).

Python callers can select `precision="bfloat16"` for source BF16 arithmetic or `precision="bfloat16_fp32"` for BF16 weights with FP32 arithmetic. The CLI default remains FP32. The native `NativeModel` profiles retain the arithmetic identities used by the qualification fixtures; this does not establish clinical validation.
