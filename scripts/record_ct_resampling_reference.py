"""Record synthetic host-grid fixtures with the original MONAI/Torch operators.

Run in the pinned source CPU environment; neither the native resampler nor model
weights participate. The fixture metadata records the actual source runtime.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]


class Tensor(Protocol):
    """A float32 CPU tensor exposing only the shape operations the recorder needs."""

    def unsqueeze(self, dim: int) -> Tensor:
        """Return a view with a size-one dimension inserted at ``dim``."""
        ...

    def squeeze(self, dim: int) -> Tensor:
        """Return a view with the size-one dimension at ``dim`` removed."""
        ...

    def numpy(self) -> FloatArray:
        """Return the tensor contents as a float32 array."""
        ...


class Torch(Protocol):
    """The Torch entry points used to run the source operator."""

    def set_num_threads(self, count: int) -> None:
        """Limit the intra-op thread pool to ``count`` threads."""
        ...

    def from_numpy(self, array: FloatArray) -> Tensor:
        """Share a float32 NumPy array as a CPU tensor without copying."""
        ...


class Transform(Protocol):
    """A configured MONAI affine resampler."""

    def __call__(self, src: Tensor, theta: Tensor, spatial_size: tuple[int, int, int]) -> Tensor:
        """Sample ``src`` at ``spatial_size`` through the 4x4 output-to-input ``theta``."""
        ...


class TransformFactory(Protocol):
    """Builds MONAI's ``AffineTransform`` from the pinned keyword options."""

    def __call__(
        self,
        *,
        normalized: bool,
        mode: str,
        padding_mode: str,
        align_corners: bool,
        reverse_indexing: bool,
    ) -> Transform:
        """Return a transform set to the given interpolation and indexing options."""
        ...


def main() -> None:
    """Write the seeded MONAI resampling fixtures and metadata to ``--output``.

    Requires torch 2.13.0 and MONAI 1.6.0, the pinned source operators. Torch runs
    single-threaded so threaded BLAS shard tails cannot change the recorded values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = cast(Path, args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch = cast(Torch, import_module("torch"))
    factory = cast(
        TransformFactory, getattr(import_module("monai.networks.layers"), "AffineTransform")
    )
    assert version("torch") == "2.13.0" and version("monai") == "1.6.0"
    # Isolate the operator's K4 arithmetic from threaded BLAS shard tails.
    torch.set_num_threads(1)
    transform = factory(
        normalized=False,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
        reverse_indexing=True,
    )
    rng = np.random.default_rng(165)
    mapping = np.array(
        [
            [0.73, 0.002, -0.03, 1.25],
            [0.01, 1.1, 0.003, -2.4],
            [0.03, -0.005, 1.8, 0.2],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    cases = [
        ((9, 7, 11), (13, 8, 21)),
        ((15, 17, 19), (11, 23, 13)),
        ((32, 35, 41), (31, 39, 27)),
        ((3, 5, 7), (2, 3, 5)),
    ]
    for index, (source_shape, target_shape) in enumerate(cases):
        values = rng.uniform(-1050, 1100, source_shape).astype(np.float32)
        result = (
            transform(
                torch.from_numpy(values).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(mapping.astype(np.float32)),
                target_shape,
            )
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )
        np.savez_compressed(
            output / f"ct_resampling-{index}.npz",
            values=values,
            mapping=mapping,
            expected=result,
        )
    metadata = {
        "recorded_on": {"system": platform.system(), "machine": platform.machine()},
        "versions": {name: version(name) for name in ("torch", "monai", "numpy")},
        "seed": 165,
        "torch_threads": 1,
        "cases": len(cases),
        "units": "HU",
        "dtype": "float32",
        "oracle": "MONAI AffineTransform, normalized=False, bilinear, border, align_corners=False",
        "generator": "scripts/record_ct_resampling_reference.py",
    }
    (output / "ct_resampling.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
