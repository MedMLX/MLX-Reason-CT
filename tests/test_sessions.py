"""Session ownership, durable failures, and exact-prefix admission without a GPU."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from mlx_reason_ct import CTSession, api
from mlx_reason_ct.ct_crop import CTCrop
from mlx_reason_ct.errors import InvalidInputError, ModelExecutionError
from mlx_reason_ct.generation_state import PromptPrefix, cache_capacity
from mlx_reason_ct.processor_mlx import VolumePrompt

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array
    from mlx_reason_ct.mlx_model import DecoderCache


def prompt(length: int = 4) -> VolumePrompt:
    ids = np.arange(length, dtype=np.int64)[None]
    positions = np.tile(ids[None], (3, 1, 1))
    markers = np.zeros_like(ids)
    markers[:, 1:3] = 1
    return VolumePrompt("fixture", ids, positions, np.ones_like(ids), markers, 0, 1)


@pytest.mark.parametrize("required,expected", [(0, 0), (1, 256), (256, 256), (257, 512)])
def test_capacity_grows_only_at_boundaries(required: int, expected: int) -> None:
    assert cache_capacity(required, 256) == expected
    assert cache_capacity(required, 0) == required


@pytest.mark.parametrize("chunk", [-256, 1, 255, True])
def test_cache_rejects_invalid_chunk(chunk: int) -> None:
    with pytest.raises(InvalidInputError, match="multiple of 256"):
        cache_capacity(1, chunk)


def test_prefix_requires_complete_tokens_positions_and_image_markers() -> None:
    old = prompt()
    saved = PromptPrefix(old, cast("DecoderCache", object()), cast("Array", object()))
    assert saved.matching_tokens(prompt(6)) == 4
    assert saved.matching_tokens(prompt(3)) == 0
    changed = prompt(6)
    changed.input_ids[0, 2] += 1
    assert saved.matching_tokens(changed) == 0
    changed = prompt(6)
    changed.position_ids[2, 0, 3] += 1
    assert saved.matching_tokens(changed) == 0
    changed = prompt(6)
    changed.mm_token_type_ids[0, 2] = 0
    assert saved.matching_tokens(changed) == 0
    assert saved.matching_tokens(replace(prompt(6), image_start=2)) == 0
    assert PromptPrefix().matching_tokens(old) == 0


def test_session_reuses_encoding_and_preserves_history_on_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # These doubles exercise orchestration only. Native cache and token numerical
    # behavior has a separate Metal gate; no NumPy inference backend is supplied.
    source = tmp_path / "ct.nii.gz"
    source.write_bytes(b"synthetic input marker")
    calls: list[str] = []
    histories: list[tuple[tuple[str, str], ...]] = []
    complete = [True]
    pixels = np.zeros((1, 1, 1), dtype=np.float32)
    crop = CTCrop(
        pixels, np.eye(4), np.eye(4), np.eye(4), (1, 1, 1), ((0, 0),) * 3, (0, 0, 0), None
    )

    class Processor:
        def image(self, source: Path, anatomy: str | None) -> tuple[np.ndarray, CTCrop]:
            calls.append("image")
            return pixels, crop

        def prompt(
            self, question: str, thinking: bool, history: tuple[tuple[str, str], ...]
        ) -> VolumePrompt:
            histories.append(history)
            return prompt()

        def decode(self, tokens: list[int]) -> str:
            return "Synthetic response."

    class Model:
        dtype = "float32"

        def vision(self, value: object) -> tuple[np.ndarray, np.ndarray]:
            calls.append("vision")
            return pixels, pixels

    def load_assets(path: Path) -> Processor:
        calls.append("assets")
        return Processor()

    def load_model(path: Path, *, precision: str) -> Model:
        calls.append("model")
        assert precision == "float32"
        return Model()

    def no_op(*values: object) -> None:
        pass

    def array(value: object) -> object:
        return value

    runtime = SimpleNamespace(
        array=array,
        eval=no_op,
        isfinite=np.isfinite,
        all=np.all,
        reset_peak_memory=no_op,
        get_peak_memory=lambda: 0,
    )

    def generate(
        model: object,
        inputs: VolumePrompt,
        embeddings: object,
        *,
        max_new_tokens: int,
        prefix: PromptPrefix | None,
        cache_chunk_size: int,
    ) -> tuple[list[int], dict[str, object]]:
        assert cache_chunk_size == 256 and prefix is not None
        calls.append("generate")
        return [7, 9], {"terminated_by_eos": complete[0], "generated_tokens": 2}

    monkeypatch.setattr(api, "_load_assets", load_assets)
    monkeypatch.setattr(api, "_load_model", load_model)
    monkeypatch.setattr(api, "import_mlx", lambda: runtime)
    monkeypatch.setattr(api, "generate", generate)
    with CTSession(source, model_dir=tmp_path, reuse_prefix=True) as session:
        assert not calls
        result = session.report(tmp_path / "first", prompt="First")
        assert result["status"] == "succeeded"
        session.report(tmp_path / "second", prompt="Second")
        assert (
            calls.count("assets")
            == calls.count("model")
            == calls.count("image")
            == calls.count("vision")
            == 1
        )
        assert histories == [(), (("First", "Synthetic response."),)]
        before = session.history
        complete[0] = False
        with pytest.raises(ModelExecutionError, match="incomplete"):
            session.report(tmp_path / "partial", prompt="Third")
        assert session.history == before
        assert json.loads((tmp_path / "partial/run.json").read_text())["status"] == "failed"
        complete[0] = True
        session.reset()
        assert session.history == ()
        session.report(tmp_path / "reset", prompt="Fresh")
        assert calls.count("vision") == 1 and histories[-1] == ()
        source.write_bytes(b"changed input marker with different length")
        with pytest.raises(InvalidInputError, match="CT file changed"):
            session.report(tmp_path / "changed")
    with pytest.raises(InvalidInputError, match="closed"):
        session.report(tmp_path / "closed")
    session.close()
