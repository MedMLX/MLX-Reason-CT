# mlx-reason-ct

Generate reports and ask questions about 3D chest and abdomen CT, locally on Apple Silicon.

[Docs](https://github.com/MedMLX/mlx-reason-ct/blob/main/docs/technical.md) · [Model weights](https://huggingface.co/MedMLX/MLX-Reason-CT) · [MedMLX](https://github.com/MedMLX)

| Input | Output |
| :--- | :--- |
| 3D CT · NIfTI | Reports and CT question answering |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "mlx-reason-ct>=0.3.5"
```

## Run

```python
from mlx_reason_ct.medmlx import run

result = run(image_path="ct.nii.gz", output_dir="outputs")
print(result["outputs"]["report"])
```

Omitting `weights_path` downloads `MedMLX/MLX-Reason-CT` once and reuses its local
cache. An explicit model folder keeps the converted runtime files and the `mlx/`
shard layout. No conversion manifest or source-identity sidecar is required. Set
`anatomy_region="abdomen"` for abdominal CT, or use `prompt` to ask a question.

Allow more than 22.7 GB unified memory for the model, preprocessing and macOS.
Crop whole-body scans to the chest or abdomen before running.

Downloaded commits are returned in `details.weights_revision`; explicit local
weights report `null` unless the folder carries a MedMLX download revision.

## License

Port contributions: [Apache-2.0](https://github.com/MedMLX/mlx-reason-ct/blob/main/LICENSE). NVIDIA model materials and weights: OpenMDW-1.1. See [third-party notices](https://github.com/MedMLX/mlx-reason-ct/blob/main/THIRD_PARTY_NOTICES.md).

For research and education. Outputs require human review.
