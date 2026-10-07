# Usage details

## Python API

```python
from nv_reason_ct_mlx import generate_report

generate_report(
    "ct.nii.gz", "outputs",
    model_dir="models",
    anatomy_region="chest",
    prompt="write a structured chest CT report",
    max_new_tokens=512,
)
```

Inference is local and uses FP32 Metal arithmetic. Add `--enable-thinking` to
use the source template's thinking mode. Each request owns its decoder state.

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
uv run nv-reason-ct-mlx verify --model-dir models
uv run nv-reason-ct-mlx convert --source-dir /path/to/upstream --output-dir models
```

Conversion requires the complete pinned upstream snapshot at revision
`386b93e034983f6c1fc841a43833a1b6a0cd9c13`. It expands BF16 bit patterns to FP32
without retraining, rejects changed source assets and validates tied weights.
The disconnected 2D tower, unused mask token and duplicate LM head are excluded.
The portable bundle verifies runtime metadata and all 35 shard hashes.

## Development and evidence

```bash
make qa
make build
```

[Historical FP32 qualification](fp32-engineering-qualification.json) covers four
synthetic cases. [Standalone verification](standalone-verification.json) records
package, CPU preprocessing and weight checks. Fresh standalone learned inference
is pending. These checks establish no clinical or report accuracy.

See [third-party terms](../THIRD_PARTY_NOTICES.md).
