"""Default folders and explicit overrides share the offline report API."""

from pathlib import Path

import pytest

from mlx_reason_ct import api, medmlx


@pytest.mark.parametrize("automatic", [True, False])
def test_report_defaults_to_model_folder_and_records_weights_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    automatic: bool,
) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    revision = "a" * 40
    (folder / ".medmlx-revision").write_text(revision)
    image = tmp_path / "ct.nii.gz"
    image.write_bytes(b"mock report API does not read this input")
    downloads: list[str] = []
    selected: list[object] = []

    def download(model_id: str) -> Path:
        downloads.append(model_id)
        return folder

    def report(*args: object, **kwargs: object) -> dict[str, object]:
        selected.append(kwargs["model_dir"])
        return {"backend": "mlx"}

    monkeypatch.setattr(medmlx, "weights_dir", download)
    monkeypatch.setattr(api, "generate_report", report)
    result = medmlx.run(
        image_path=image,
        output_dir=tmp_path / "out",
        weights_path=None if automatic else folder,
    )
    assert downloads == (["mlx-reason-ct"] if automatic else [])
    assert selected == [folder]
    assert result["details"] == {"backend": "mlx", "weights_revision": revision}
