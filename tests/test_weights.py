"""Exact conversion and portable admission without the source checkpoint."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from mlx_reason_ct import mlx_weights as weights
from mlx_reason_ct.errors import InvalidInputError


def source_checkpoint(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid: str | None = None,
) -> Path:
    directory.mkdir()
    # Explicit BF16 storage encodes 1.0 and -2.5; no converter computes expectations.
    raw = struct.pack("<HH", 0x3F80, 0xC020)
    names = [
        "model.vision3d.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    names += [f"model.language_model.layers.{i}.weight" for i in range(32)]
    header = {}
    payload = bytearray()
    for name in names:
        value = raw
        if invalid == "nonfinite" and name == "model.vision3d.weight":
            value = struct.pack("<HH", 0x7F80, 0xC020)
        if invalid == "tied" and name == "lm_head.weight":
            value = struct.pack("<HH", 0x4000, 0xC020)
        offset = len(payload)
        header[name] = {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [offset, offset + len(value)],
        }
        payload.extend(value)
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    runtime = [
        "LICENSE",
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "image_processor_3d/preprocessor_config.json",
    ]
    for name in runtime:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic " + name)
    pins = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in [*runtime, "model.safetensors"]
    }
    monkeypatch.setattr(weights, "PINS", pins)
    return directory


def test_conversion_preserves_bits_and_portable_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_checkpoint(tmp_path / "source", monkeypatch)
    bundle = tmp_path / "bundle"
    manifest_path = weights.convert_checkpoint(source, bundle)
    manifest = weights.read_manifest(bundle)
    assert manifest_path == bundle / "mlx/manifest.json"
    assert manifest["engine"] == "nv_reason_ct_mlx.fp32.v1"
    assert manifest["dtype"] == "float32"
    assert len(manifest["shards"]) == 35
    for filename in manifest["shards"]:
        arrays = load_file(bundle / "mlx" / filename)
        for values in arrays.values():
            np.testing.assert_array_equal(values, np.array([1.0, -2.5], dtype=np.float32))
    assert manifest["excluded_source_tensors"] == ["lm_head.weight"]


@pytest.mark.parametrize("invalid", ["nonfinite", "tied"])
def test_conversion_rejects_invalid_source_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    source = source_checkpoint(tmp_path / "source", monkeypatch, invalid=invalid)
    expected = "Nonfinite" if invalid == "nonfinite" else "tied token embedding"
    with pytest.raises(InvalidInputError, match=expected):
        weights.convert_checkpoint(source, tmp_path / "bundle")


@pytest.mark.parametrize("corruption", ["runtime", "shard", "dtype", "inventory", "revision"])
def test_portable_bundle_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    source = source_checkpoint(tmp_path / "source", monkeypatch)
    bundle = tmp_path / "bundle"
    manifest_path = weights.convert_checkpoint(source, bundle)
    weights.read_manifest(bundle)  # The positive control reaches the same admission owner.
    if corruption == "runtime":
        (bundle / "config.json").write_text("changed")
        message = "runtime asset changed"
    elif corruption == "shard":
        (bundle / "mlx/vision.safetensors").write_bytes(b"changed")
        message = "integrity failure"
    else:
        manifest = json.loads(manifest_path.read_text())
        if corruption == "inventory":
            del manifest["shards"]["vision.safetensors"]
            message = "invalid shard inventory"
        else:
            manifest[corruption] = "incorrect"
            message = "provenance differs"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidInputError, match=message):
        weights.read_manifest(bundle)
