"""Independent portable input, prompt and durable response expectations."""

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nv_reason_ct_mlx.errors import InvalidInputError, InvalidPromptError, ModelExecutionError
from nv_reason_ct_mlx.nifti import load_ct
from nv_reason_ct_mlx.processor_mlx import volume_positions
from nv_reason_ct_mlx.response import save_response

nib: Any = import_module("nibabel")


@pytest.mark.parametrize("unit,scale", [("mm", 1.0), ("meter", 0.001), ("micron", 1000)])
def test_ct_units_and_scaled_hu(tmp_path: Path, unit: str, scale: float) -> None:
    image = nib.Nifti1Image(
        np.full((2, 3, 4), 10, dtype=np.int16), np.diag([2 * scale, 3 * scale, 4 * scale, 1])
    )
    image.header.set_xyzt_units(unit)
    image.header.set_slope_inter(2, -1000)
    path = tmp_path / "ct.nii.gz"
    nib.save(image, path)
    values, affine = load_ct(path)
    np.testing.assert_array_equal(values, np.full((2, 3, 4), -980, dtype=np.float32))
    np.testing.assert_allclose(affine, np.diag([2, 3, 4, 1]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("invalid", ["uncoded", "conflicting", "nonfinite", "4d"])
def test_ct_rejects_ambiguous_geometry_or_voxels(tmp_path: Path, invalid: str) -> None:
    shape = (2, 3, 4, 2) if invalid == "4d" else (2, 3, 4)
    values = np.zeros(shape, dtype=np.float32)
    if invalid == "nonfinite":
        values.flat[0] = np.nan
    image = nib.Nifti1Image(values, np.diag([2, 2, 2, 1]))
    if invalid == "uncoded":
        image.set_qform(None, code=0)
        image.set_sform(None, code=0)
    elif invalid == "conflicting":
        image.set_qform(np.diag([3, 2, 2, 1]), code=1)
    path = tmp_path / "ct.nii.gz"
    nib.save(image, path)
    expected = {
        "uncoded": "no coded",
        "conflicting": "different spatial",
        "nonfinite": "finite scalar 3D",
        "4d": "finite scalar 3D",
    }[invalid]
    with pytest.raises(InvalidInputError, match=expected):
        load_ct(path)


def test_volume_prompt_positions_and_single_marker_guard() -> None:
    ids = np.array([10, 99, 99, 99, 99, 11])
    positions, delta, start = volume_positions(ids, 99, (1, 2, 2))
    expected = np.array([[0, 1, 1, 1, 1, 3], [0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]])[:, None]
    np.testing.assert_array_equal(positions, expected)
    assert (delta, start) == (-2, 1)
    with pytest.raises(InvalidPromptError, match="exactly one"):
        volume_positions(np.array([10, 99, 11, 99, 99, 99]), 99, (1, 2, 2))


@pytest.mark.parametrize(
    "report,complete", [("Draft report", True), ("Partial", False), ("", True)]
)
def test_response_completion_and_partial_persistence(
    tmp_path: Path, report: str, complete: bool
) -> None:
    metrics = {"terminated_by_eos": complete, "generated_tokens": 2}
    if report and complete:
        outputs = save_response(tmp_path, report, metrics)
        assert outputs == {
            "report": str(tmp_path / "report.txt"),
            "response": str(tmp_path / "model_response.json"),
        }
    else:
        with pytest.raises(ModelExecutionError, match="empty or incomplete"):
            save_response(tmp_path, report, metrics)
    assert (tmp_path / "report.txt").read_text() == report + "\n"
    assert json.loads((tmp_path / "model_response.json").read_text()) == {
        "report": report,
        "requires_human_review": True,
        "generation": metrics,
    }


def test_thinking_response_separates_reasoning_from_report(tmp_path: Path) -> None:
    metrics = {"terminated_by_eos": True, "generated_tokens": 9}
    save_response(tmp_path, "Plan the report.\n</think>\n\nFINDINGS: Normal.", metrics)
    assert (tmp_path / "report.txt").read_text() == "FINDINGS: Normal.\n"
    assert json.loads((tmp_path / "model_response.json").read_text()) == {
        "report": "FINDINGS: Normal.",
        "thinking": "Plan the report.",
        "requires_human_review": True,
        "generation": metrics,
    }
    with pytest.raises(ModelExecutionError, match="empty or incomplete"):
        save_response(tmp_path, "Reasoning only\n</think>\n\n", metrics)


@pytest.mark.parametrize("region", ["chest", "abdomen"])
def test_default_prompt_requests_report_for_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, region: str
) -> None:
    api = import_module("nv_reason_ct_mlx.api")
    seen: dict[str, str] = {}

    class StopAfterPrompt(Exception):
        pass

    class FakeProcessor:
        def __init__(self, model_dir: Path) -> None:
            pass

        def image(self, source: Path, anatomy_region: str) -> tuple[None, None]:
            return None, None

        def prompt(self, question: str, enable_thinking: bool) -> None:
            seen["prompt"] = question
            raise StopAfterPrompt

    monkeypatch.setattr(api, "read_manifest", lambda model_dir: {})
    monkeypatch.setattr(api, "Processor", FakeProcessor)
    with pytest.raises(StopAfterPrompt):
        api.generate_report("ct.nii.gz", tmp_path, model_dir=tmp_path, anatomy_region=region)
    assert seen["prompt"] == f"write a structured {region} CT report"
