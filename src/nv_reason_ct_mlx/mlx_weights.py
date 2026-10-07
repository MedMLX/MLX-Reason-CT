"""Offline exact BF16-to-FP32 conversion and provenance-bound native cache admission."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np

from nv_reason_ct_mlx.errors import InvalidInputError
from nv_reason_ct_mlx.integrity import file_sha256

REVISION = "386b93e034983f6c1fc841a43833a1b6a0cd9c13"
ENGINE = "nv_reason_ct_mlx.fp32.v1"
MANIFEST = "mlx/manifest.json"
SHARDS = (
    "vision.safetensors",
    "embedding.safetensors",
    "norm.safetensors",
    *(f"layer_{i:02d}.safetensors" for i in range(32)),
)
PINS: dict[str, str] = json.loads(Path(__file__).with_name("source_pins.json").read_text())
RUNTIME_FILES = (
    "LICENSE",
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "image_processor_3d/preprocessor_config.json",
)


def validate_source(model_dir: Path) -> None:
    for name, expected in PINS.items():
        path = model_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise InvalidInputError(f"NV-Reason-CT pinned asset missing or changed: {name}")


def read_manifest(model_dir: Path) -> dict[str, Any]:
    for name in RUNTIME_FILES:
        if file_sha256(model_dir / name) != PINS[name]:
            raise InvalidInputError(f"NV-Reason-CT runtime asset changed: {name}")
    payload: object = json.loads((model_dir / MANIFEST).read_text())
    if not isinstance(payload, dict):
        raise InvalidInputError("NV-Reason-CT MLX cache manifest must be an object")
    manifest = cast(dict[str, Any], payload)
    if (
        manifest.get("engine") != ENGINE
        or manifest.get("revision") != REVISION
        or manifest.get("source_sha256") != PINS
        or manifest.get("dtype") != "float32"
        or manifest.get("runtime_sha256") != {name: PINS[name] for name in RUNTIME_FILES}
    ):
        raise InvalidInputError(
            "NV-Reason-CT MLX cache provenance differs from the pinned conversion"
        )
    inventory: object = manifest.get("shards", {})
    if not isinstance(inventory, dict):
        raise InvalidInputError("NV-Reason-CT MLX cache shard inventory must be an object")
    shards = cast(dict[str, str], inventory)
    if set(shards) != set(SHARDS):
        raise InvalidInputError("NV-Reason-CT MLX cache has an invalid shard inventory")
    for name, digest in shards.items():
        if file_sha256(model_dir / "mlx" / name) != digest:
            raise InvalidInputError(f"NV-Reason-CT MLX cache integrity failure: {name}")
    return manifest


def convert_checkpoint(model_dir: Path, output_dir: Path) -> Path:
    """Convert offline using bounded CPU mappings; no model execution or network IO."""
    safetensors: Any = import_module("safetensors.numpy")

    validate_source(model_dir)
    source = model_dir / "model.safetensors"
    with source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    mapped = np.memmap(source, mode="r", dtype=np.uint8, offset=8 + size)

    def tensor(name: str) -> np.ndarray:
        spec = header[name]
        if spec["dtype"] != "BF16":
            raise InvalidInputError(f"Pinned NV-Reason-CT tensor is not BF16: {name}")
        low, high = spec["data_offsets"]
        raw = mapped[low:high].view("<u2").astype(np.uint32)
        result = (raw << 16).view(np.float32).reshape(spec["shape"])
        if not np.isfinite(result).all():
            raise InvalidInputError(f"Nonfinite checkpoint tensor: {name}")
        return result

    groups: dict[str, list[str]] = defaultdict(list)
    ignored: list[str] = []
    for name in header:
        if name == "__metadata__":
            continue
        if name.startswith("model.vision3d.") and not name.endswith("mask_token"):
            group = "vision"
        elif name.startswith("model.language_model.layers."):
            group = f"layer_{int(name.split('.')[3]):02d}"
        elif name == "model.language_model.embed_tokens.weight":
            group = "embedding"
        elif name == "model.language_model.norm.weight":
            group = "norm"
        elif (
            name.startswith("model.visual.")
            or name.endswith("mask_token")
            or name == "lm_head.weight"
        ):
            ignored.append(name)
            continue
        else:
            raise InvalidInputError(f"Unexpected NV-Reason-CT checkpoint tensor: {name}")
        groups[group].append(name)
    # Transformers ties these parameters. Refuse ambiguous checkpoint aliases.
    if not np.array_equal(
        tensor("lm_head.weight"), tensor("model.language_model.embed_tokens.weight")
    ):
        raise InvalidInputError("NV-Reason-CT tied token embedding and LM head differ")
    if {group + ".safetensors" for group in groups} != set(SHARDS):
        raise InvalidInputError("Pinned checkpoint is missing required FP32 shard groups")
    destination = output_dir / "mlx"
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination, prefix="convert-") as temporary:
        staged = Path(temporary)
        hashes: dict[str, str] = {}
        for group, names in groups.items():
            filename = group + ".safetensors"
            arrays = {name: tensor(name) for name in names}
            safetensors.save_file(arrays, staged / filename)
            del arrays
            hashes[filename] = file_sha256(staged / filename)
        manifest = {
            "engine": ENGINE,
            "revision": REVISION,
            "source_sha256": PINS,
            "dtype": "float32",
            "runtime_sha256": {name: PINS[name] for name in RUNTIME_FILES},
            "conversion": "lossless BF16 bit expansion; source tensor layouts",
            "shards": hashes,
            "excluded_source_tensors": ignored,
            "exclusion_reason": (
                "2D image/video tower rejected by source processor; "
                "inactive mask token; tied LM head"
            ),
        }
        for name in RUNTIME_FILES:
            target = output_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_dir / name, target)
        # The manifest is the final admission marker, never a partially written conversion.
        (destination / "manifest.json").unlink(missing_ok=True)
        for name in hashes:
            (staged / name).replace(destination / name)
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staged / "manifest.json").replace(destination / "manifest.json")
    return destination / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(convert_checkpoint(args.source_dir, args.output_dir))


if __name__ == "__main__":
    main()
