"""Physical CT resampling and enclosed-air anatomy crops, independent of runtimes.

The crop policy follows NVIDIA NV-Reason-CT revision 386b93e (OpenMDW-1.1).
Spacing/extent arithmetic follows MONAI 1.6.0 (Apache-2.0).
See THIRD_PARTY_NOTICES.md. Affines map voxel indices to NIfTI RAS millimeters;
LPS names array axis directions, not the affine world coordinate convention.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from mlx_reason_ct.nifti import load_ct

ndimage: Any = import_module("scipy.ndimage")


@dataclass(frozen=True)
class CTCrop:
    """HU crop with explicit source and resampled/padded voxel geometry."""

    values: np.ndarray
    affine: np.ndarray
    source_affine: np.ndarray
    resampled_affine: np.ndarray
    resampled_shape: tuple[int, ...]
    padding: tuple[tuple[int, int], ...]
    start: tuple[int, ...]
    bounds: tuple[int, int] | None

    def geometry(self) -> dict[str, object]:
        return {
            "source_affine_ras_mm": self.source_affine.tolist(),
            "resampled_affine_ras_mm": self.resampled_affine.tolist(),
            "crop_affine_ras_mm": self.affine.tolist(),
            "resampled_shape_xyz": list(self.resampled_shape),
            "padding_xyz": [list(p) for p in self.padding],
            "crop_start_padded_xyz": list(self.start),
            "region_bounds_padded_z": None if self.bounds is None else list(self.bounds),
            "anatomy_fallback": self.bounds is None,
            "array_orientation": "LPS",
            "array_axes": "xyz",
            "units": "HU",
        }


def spacing_grid(
    shape: tuple[int, ...],
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    keep_fraction: float = 0.1,
) -> tuple[tuple[int, ...], np.ndarray]:
    """MONAI's non-diagonal voxel-center extent, including its spacing tolerance."""
    rzs = affine[:3, :3]
    original = np.linalg.norm(rzs, axis=0)
    target = np.asarray(spacing, dtype=np.float64)
    target = np.where(
        (original >= target * (1 - keep_fraction) - 1e-3)
        & (original <= target * (1 + keep_fraction) + 1e-3),
        original,
        target,
    )
    zoom_shear = np.linalg.cholesky(rzs.T @ rzs).T
    destination = np.eye(4)
    destination[:3, :3] = rzs @ np.linalg.inv(zoom_shear) @ np.diag(target)
    corners = np.asarray(np.meshgrid(*[(0.0, n - 1.0) for n in shape], indexing="ij"))
    corners = np.concatenate((corners.reshape(3, -1), np.ones((1, 8))))
    out_corners = np.linalg.solve(destination, affine) @ corners
    size: np.ndarray = np.asarray(
        np.round(np.ptp(out_corners[:3] / out_corners[3], axis=1) + 1), dtype=np.int64
    )
    world = affine @ corners
    for i in range(8):
        if np.allclose(np.min(out_corners[:3] - out_corners[:3, i : i + 1], axis=1), 0, rtol=1e-3):
            destination[:3, 3] = world[:3, i]
            break
    else:
        destination[:3, 3] = (
            rzs @ (np.asarray(shape) / 2) + affine[:3, 3] - destination[:3, :3] @ (size / 2)
        )
    return tuple(int(x) for x in size), destination


def lung_air_bounds(
    values: np.ndarray, *, spacing: tuple[float, float, float]
) -> tuple[int, int, int] | None:
    """Return body base and superior enclosed-air cluster bounds in LPS axes.

    This intensity heuristic is a crop selector, never a segmentation or diagnosis.
    Thresholds and connectivity are the pinned NVIDIA processor's policy.
    """
    body = values > -500
    body_z = np.flatnonzero(body.any(axis=(0, 1)))
    if not len(body_z):
        return None
    air = np.empty(body.shape, dtype=bool)
    for z in range(body.shape[2]):
        air[:, :, z] = ndimage.binary_fill_holes(body[:, :, z]) & ~body[:, :, z]
    labels, count = ndimage.label(air)
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    candidates = np.flatnonzero(sizes >= 0.1 * sizes.max())
    summaries: list[tuple[int, int, int, int]] = []
    for component in candidates[candidates != 0]:
        zs = np.where(labels == component)[2]
        summaries.append((int(component), int(sizes[component]), int(zs.min()), int(zs.max())))
    summaries.sort(key=lambda item: (item[3], item[1]), reverse=True)
    if not summaries:
        return None
    first = summaries[0]
    selected, z_min, z_max = [first[0]], first[2], first[3]
    for component, _size, low, high in summaries[1:]:
        if max(0, z_min - high - 1) * spacing[2] > 40:
            continue
        next_low = min(z_min, low)
        if (z_max - next_low + 1) * spacing[2] > 500:
            continue
        selected.append(component)
        z_min, z_max = next_low, max(z_max, high)
    lung = np.isin(labels, selected)
    zs = np.flatnonzero(lung.any(axis=(0, 1)))
    low = max(int(zs.min()), int(zs.max()) - max(1, int(np.ceil(500 / spacing[2]))) + 1)
    lung[:, :, :low] = False
    zs = np.flatnonzero(lung.any(axis=(0, 1)))
    if not len(zs) or float(lung.sum()) * float(np.prod(spacing)) / 1000 < 250:
        return None
    return int(body_z.min()), int(zs.min()), int(zs.max())


def anatomy_crop(
    values: np.ndarray,
    region: str,
    roi: tuple[int, int, int],
    *,
    spacing: tuple[float, float, float],
) -> tuple[tuple[int, ...], tuple[int, int] | None]:
    """Choose the pinned chest/abdomen crop in an already padded LPS HU array."""
    if region not in {"chest", "abdomen"}:
        raise ValueError("anatomy_region must be 'chest' or 'abdomen'")
    if any(n < r for n, r in zip(values.shape, roi, strict=True)):
        raise ValueError("Anatomy crop requires padding to at least the ROI size")
    lungs = lung_air_bounds(values, spacing=spacing)
    bounds = None
    if lungs is not None:
        body_low, lung_low, lung_high = lungs
        if region == "chest":
            bounds = (lung_low, lung_high)
        else:
            high = min(values.shape[2] - 1, lung_low + int(np.ceil(100 / spacing[2])))
            low = max(body_low, high - int(np.ceil(300 / spacing[2])) + 1)
            if low <= high:
                bounds = (low, high)
    centers = [(values.shape[i] - 1) / 2 for i in range(3)]
    if bounds is not None:
        centers[2] = (bounds[0] + bounds[1]) / 2
    start = [
        max(0, min(int(round(c - r / 2)), n - r))
        for c, r, n in zip(centers, roi, values.shape, strict=True)
    ]
    if bounds is None:
        warnings.warn(
            f"Could not determine {region} z bounds; using the source superior-edge crop.",
            RuntimeWarning,
            stacklevel=2,
        )
        start[2] = values.shape[2] - roi[2]
    return tuple(start), bounds


def load_anatomy_ct(
    path: Path,
    region: str,
    *,
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    roi: tuple[int, int, int] = (192, 192, 192),
) -> CTCrop:
    """Read, orient, resample, air-pad and select a CT crop on the CPU."""
    nib: Any = import_module("nibabel")
    values, source_affine = load_ct(path)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("CT input must be a finite scalar 3D HU volume")
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(source_affine),
        nib.orientations.axcodes2ornt("LPS"),
    )
    orientation_affine: np.ndarray = nib.orientations.inv_ornt_aff(transform, values.shape)
    affine = source_affine @ orientation_affine
    values = nib.orientations.apply_orientation(values, transform)
    shape, resampled_affine = spacing_grid(values.shape, affine, spacing)
    mapping: np.ndarray = np.asarray(np.linalg.solve(affine, resampled_affine))
    if shape != values.shape or not np.allclose(mapping, np.eye(4), atol=1e-3):
        values = ndimage.affine_transform(
            values,
            mapping[:3, :3],
            mapping[:3, 3],
            output_shape=shape,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    widths = tuple(
        (max(0, r - n) // 2, max(0, r - n) - max(0, r - n) // 2)
        for n, r in zip(shape, roi, strict=True)
    )
    values = np.pad(values, widths, constant_values=-1000.0)
    start, bounds = anatomy_crop(values, region, roi, spacing=spacing)
    values = values[tuple(slice(a, a + r) for a, r in zip(start, roi, strict=True))]
    crop_affine = resampled_affine.copy()
    crop_affine[:3, 3] += crop_affine[:3, :3] @ (
        np.asarray(start) - np.asarray([p[0] for p in widths])
    )
    return CTCrop(
        np.ascontiguousarray(values),
        crop_affine,
        source_affine,
        resampled_affine,
        shape,
        widths,
        start,
        bounds,
    )
