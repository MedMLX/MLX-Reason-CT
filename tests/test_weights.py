"""Published bundle admission checks."""

import json
from pathlib import Path

import pytest

from mlx_reason_ct import mlx_weights as weights
from mlx_reason_ct.errors import InvalidInputError


def make_bundle(directory: Path) -> Path:
    """Create a minimal bundle for manifest and asset admission tests."""
    directory.mkdir()
    for name in weights.RUNTIME_FILES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic " + name)
    shard_dir = directory / "mlx"
    shard_dir.mkdir()
    for name in weights.SHARDS:
        (shard_dir / name).write_bytes(b"placeholder")
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "engine": weights.ENGINE,
                "revision": weights.REVISION,
                "dtype": "float32",
                "shards": list(weights.SHARDS),
            }
        )
    )
    return directory


@pytest.mark.parametrize("corruption", ["runtime", "shard", "dtype", "inventory", "revision"])
def test_bundle_rejects_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    manifest_path = bundle / "mlx/manifest.json"
    weights.read_manifest(bundle)  # The positive control reaches the same admission owner.
    if corruption == "runtime":
        (bundle / "config.json").unlink()
        message = "runtime asset missing"
    elif corruption == "shard":
        (bundle / "mlx/vision.safetensors").unlink()
        message = "shard missing"
    else:
        manifest = json.loads(manifest_path.read_text())
        if corruption == "inventory":
            manifest["shards"].remove("vision.safetensors")
            message = "invalid shard inventory"
        else:
            manifest[corruption] = "incorrect"
            message = "provenance differs"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidInputError, match=message):
        weights.read_manifest(bundle)
