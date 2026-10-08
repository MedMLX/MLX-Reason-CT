"""Independent physical geometry and anatomy-selection expectations."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mlx_reason_ct.ct_crop import anatomy_crop, load_anatomy_ct

nib: Any = import_module("nibabel")


def test_air_padding_preserves_hu_and_voxel_world_coordinates(tmp_path: Path) -> None:
    values = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    affine = np.diag([-2.1, -2.0, 2.0, 1.0])
    affine[:3, 3] = (10.0, 20.0, 30.0)
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "small.nii.gz"
    nib.save(image, path)
    with pytest.warns(RuntimeWarning, match="superior-edge"):
        result = load_anatomy_ct(path, "chest", roi=(8, 8, 10))
    assert result.resampled_shape == (4, 6, 8)
    assert result.padding == ((2, 2), (1, 1), (1, 1))
    assert result.start == (0, 0, 0)
    np.testing.assert_array_equal(result.values[2:6, 1:7, 1:9], values)
    assert np.all(result.values[:2] == -1000)
    assert np.all(result.values[:, :, 0] == -1000)
    np.testing.assert_allclose(result.affine[:3, 3], [14.2, 22.0, 28.0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        result.affine @ [2, 1, 1, 1], [10.0, 20.0, 30.0, 1.0], atol=1e-6, rtol=0
    )


def test_oblique_anisotropic_resampling_obeys_independent_linear_ramp(tmp_path: Path) -> None:
    x, y, z = np.indices((9, 7, 11))
    values = (x + 10 * y + 100 * z).astype(np.float32)
    angle = 0.17
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-3.0, -2.4, 4.0])
    affine[:3, 3] = (12.0, -30.0, 4.0)
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "oblique.nii.gz"
    nib.save(image, path)
    with pytest.warns(RuntimeWarning, match="superior-edge"):
        result = load_anatomy_ct(path, "abdomen", roi=(13, 8, 21))
    assert result.resampled_shape == (13, 8, 21)
    assert result.start == (0, 0, 0)
    np.testing.assert_allclose(
        result.affine[:3, :3], rotation @ np.diag([-2.0, -2.0, 2.0]), atol=1e-6, rtol=0
    )
    # Output index (4, 3, 8) samples input (8/3, 6/2.4, 16/4).
    assert float(result.values[4, 3, 8]) == pytest.approx(8 / 3 + 25 + 400, abs=0.0001)
    np.testing.assert_allclose(result.affine[:3, 3], [12.0, -30.0, 4.0], atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "region,start,bounds",
    [
        ("chest", (2, 2, 104), (150, 249)),
        ("abdomen", (2, 2, 30), (51, 200)),
    ],
)
def test_enclosed_air_selects_chest_and_abdomen_without_bowel_shift(
    region: str,
    start: tuple[int, ...],
    bounds: tuple[int, int],
) -> None:
    x, y, z = np.ogrid[:100, :100, :300]
    body = ((x - 50) / 45) ** 2 + ((y - 50) / 45) ** 2 < 1
    lungs = (((x - 30) / 12) ** 2 + ((y - 50) / 25) ** 2 < 1) | (
        ((x - 70) / 12) ** 2 + ((y - 50) / 25) ** 2 < 1
    )
    values = np.broadcast_to(np.where(body, 40.0, -1000.0), (100, 100, 300)).copy()
    values = np.where(lungs & (z >= 150) & (z < 250), -800.0, values)
    # Enclosed bowel gas has a >40-mm gap from the superior cluster.
    bowel = ((x - 50) / 20) ** 2 + ((y - 50) / 25) ** 2 < 1
    values = np.where(bowel & (z > 15) & (z < 60), -800.0, values)
    actual = anatomy_crop(values, region, (96, 96, 192), spacing=(2.0, 2.0, 2.0))
    assert actual == (start, bounds)
