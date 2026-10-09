# Synthetic CT walkthrough

This walkthrough uses a generated geometric phantom with HU-like intensities,
not patient data. It checks the installation and CT report workflow. The phantom
is not a realistic clinical scan, and its generated text cannot establish report
accuracy or clinical performance. Outputs require human review.

## Install and verify the bundle

Use macOS arm64, Python 3.12, [uv](https://docs.astral.sh/uv/) and an Apple Silicon
GPU with Metal access. The FP32 weights occupy 17.4 GB; the previously measured
peak MLX memory is about 22.7 GB, plus preprocessing, macOS and other applications.
Allow disk space for both the download cache and local bundle.

Run these commands in a new working directory:

```bash
uv tool install --python 3.12 mlx-reason-ct==0.2.3

mlx-reason-ct download \
  --revision c690a63888b9c6c9bd006687335fbd650eb60275 \
  --model-dir models

mlx-reason-ct verify --model-dir models
```

If `uv` reports that its executable directory is missing from `PATH`, run
`uv tool update-shell` and open a new terminal before invoking `mlx-reason-ct`.
The public `medmlx-core==0.1.2` dependency installs from PyPI with the package.

Verification prints JSON containing `"verified": true`, `"dtype": "float32"`,
`"shards": 35` and source revision
`386b93e034983f6c1fc841a43833a1b6a0cd9c13`. The source revision identifies NVIDIA's
checkpoint; the download revision identifies the converted Hub bundle.

## Generate the phantom

The following command creates `synthetic-ct/input.nii.gz`: a 192 × 192 × 192
FP32 array at 2-mm spacing, with matching coded qform/sform transforms and
millimeter units. Background is -1000, the body is 40, two enclosed air regions
are -800, and a small dense region is 700. It refuses to replace an existing input.

```bash
uv run --no-project --python 3.12 \
  --with numpy==2.3.5 --with nibabel==5.4.2 python - <<'PY'
from pathlib import Path

import nibabel as nib
import numpy as np

path = Path("synthetic-ct/input.nii.gz")
if path.exists():
    raise FileExistsError(path)
path.parent.mkdir(parents=True, exist_ok=True)

x, y, z = np.ogrid[:192, :192, :192]
body = ((x - 96) / 74) ** 2 + ((y - 96) / 66) ** 2 + ((z - 96) / 86) ** 2 < 1
air = (
    (((x - 65) / 22) ** 2 + ((y - 96) / 34) ** 2 < 1)
    | (((x - 127) / 22) ** 2 + ((y - 96) / 34) ** 2 < 1)
) & (z >= 60) & (z < 145)
dense = ((x - 96) / 8) ** 2 + ((y - 136) / 8) ** 2 < 1
values = np.full((192, 192, 192), -1000.0, dtype=np.float32)
values[body] = 40.0
values[body & air] = -800.0
values[body & dense] = 700.0

affine = np.diag([-2.0, -2.0, 2.0, 1.0])
affine[:3, 3] = (192.0, 192.0, -192.0)
image = nib.Nifti1Image(values, affine)
image.header.set_xyzt_units("mm")
image.set_qform(affine, code=1)
image.set_sform(affine, code=1)
image.header["descrip"] = b"Synthetic geometric CT phantom; no patient data"
nib.save(image, path)
print(path)
PY
```

## Run a short CT question

Use the companion runtime to exercise preprocessing, vision encoding, decoding
and report writing. The prompt requests a short answer to limit generation time:

```bash
mlx-reason-ct report \
  --input synthetic-ct/input.nii.gz \
  --model-dir models \
  --output-dir synthetic-ct/report \
  --anatomy-region chest \
  --precision float32 \
  --max-new-tokens 128 \
  --prompt "What imaging modality is shown? Answer briefly."
```

A completed run writes:

```text
synthetic-ct/
  input.nii.gz
  report/
    report.txt
    model_response.json
    run.json
```

`report.txt` contains the generated answer. `model_response.json` contains the
answer and generation metrics; `run.json` records geometry, source provenance,
precision, runtime and completion status. Actual answer wording is model output,
not a fixed expected result.

Inspect completion with a separate host-only command:

```bash
uv run --no-project --python 3.12 python - <<'PY'
import json
from pathlib import Path

directory = Path("synthetic-ct/report")
run = json.loads((directory / "run.json").read_text())
response = json.loads((directory / "model_response.json").read_text())
assert run["status"] == "succeeded"
assert run["generation"]["terminated_by_eos"]
assert not run["generation"]["truncated_by_max_new_tokens"]
assert run["generation"]["precision"] == "float32"
assert run["requires_human_review"] and response["requires_human_review"]
assert (directory / "report.txt").read_text().strip()
assert response["report"].strip() == (directory / "report.txt").read_text().strip()
print("Completed synthetic workflow; review report.txt manually.")
PY
```

If generation reaches the token limit, the CLI exits with an error and retains
partial output with `status: failed`. Increase `--max-new-tokens` and use a new
output directory, or pass `--overwrite` to deliberately replace those files.
To request the default structured report, omit `--prompt` and use a larger
token limit, such as 512. No generic `mlx-lm` or `mlx-vlm` loader is used.
