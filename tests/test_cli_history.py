"""CLI JSON history must retain the public API's strict conversation contract."""

import json
import sys
from pathlib import Path

import pytest

from mlx_reason_ct import cli
from mlx_reason_ct.errors import InvalidInputError


def _arguments(history_file: Path, output_dir: Path) -> list[str]:
    return [
        "mlx-reason-ct",
        "report",
        "--input",
        "ct.nii.gz",
        "--model-dir",
        "models",
        "--output-dir",
        str(output_dir),
        "--history",
        str(history_file),
    ]


@pytest.mark.parametrize(
    "history",
    [
        ["hi"],
        [["question", None]],
        [["question", 42]],
        [[True, "answer"]],
        [[{"question": "text"}, "answer"]],
        [["question"]],
        {"hi": "answer"},
    ],
)
def test_cli_rejects_malformed_history_without_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: object
) -> None:
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(history), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _arguments(history_file, tmp_path / "output"))

    with pytest.raises(InvalidInputError, match="history"):
        cli.main()


def test_cli_preserves_valid_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = [["Describe the CT.", "FINDINGS: clear."], ["Any change?", "No change."]]
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(history), encoding="utf-8")
    seen: dict[str, object] = {}

    def generate_report(*args: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(cli, "generate_report", generate_report)
    monkeypatch.setattr(sys, "argv", _arguments(history_file, tmp_path / "output"))

    assert cli.main() == 0
    assert json.loads(json.dumps(seen["history"])) == history
