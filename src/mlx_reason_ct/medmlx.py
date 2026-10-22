"""MedMLX declaration and offline CT report generation with explicit converted assets."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)

from mlx_reason_ct.response import ReportFiles

MODEL: dict[str, str] = {
    "id": "mlx-reason-ct",
    "label": "NV-Reason-CT",
    "task": "generation",
    "run": "mlx_reason_ct.medmlx:run",
    "summary": "CT reporting and question answering with NV-Reason-CT on Metal; FP32 by default.",
}


def run(
    *,
    image_path: str | PathLike[str],
    weights_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    overwrite: bool = False,
    prompt: str | None = None,
    anatomy_region: str | None = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    precision: str = "float32",
    history: Sequence[tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Generate a report from one scalar 3D HU NIfTI CT on Apple Silicon Metal.

    ``weights_path`` is the explicit converted bundle root containing
    ``mlx/manifest.json``, its FP32 shards, tokenizer and chat template. Writes
    ``report.txt``, ``model_response.json`` and ``run.json`` in ``output_dir``.
    A custom ``prompt`` asks a CT question; otherwise a structured report for
    ``anatomy_region`` (chest or abdomen) is requested. Pass ``anatomy_region=None``
    for the pinned processor's own centered crop, used upstream when the
    chest/abdomen heuristic is unreliable for a given scan. Generation is greedy.
    ``precision`` selects float32 (default), source-matched bfloat16 arithmetic,
    or bfloat16_fp32 (BF16 weights with FP32 accumulation and decoding).
    ``history`` continues a conversation about the same CT: already-answered
    ``(question, answer)`` turns, oldest first; ``prompt`` is the new, still-
    unanswered turn. Empty or truncated responses raise ModelExecutionError
    after retaining partial output and failure metadata. All reports require
    human review.
    """
    image = Path(image_path).expanduser().resolve()
    if not image.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"image_path must be a NIfTI CT file (.nii or .nii.gz): {image}")
    if not image.is_file():
        raise InvalidInputError(f"image_path does not exist: {image}")
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be a boolean")
    files = ReportFiles(Path(output_dir).expanduser().resolve())
    files.require_writable(overwrite=overwrite)
    weights = Path(weights_path).expanduser().resolve()
    if not weights.is_dir():
        raise AssetNotReadyError(
            f"NV-Reason-CT converted weights directory not found: {weights}",
            hint="Pass the converted bundle root containing mlx/manifest.json and FP32 shards.",
        )
    try:
        from mlx_reason_ct.api import generate_report

        details = generate_report(
            image,
            files.directory,
            model_dir=weights,
            prompt=prompt,
            anatomy_region=anatomy_region,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            precision=precision,
            overwrite=overwrite,
            history=history,
        )
    except (InvalidInputError, AssetNotReadyError, MissingDependencyError, ModelExecutionError):
        raise
    except ImportError as error:
        raise MissingDependencyError(
            f"NV-Reason-CT dependency is unavailable: {error}; install mlx-reason-ct.",
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise ModelExecutionError(
            f"NV-Reason-CT generation failed: {error}; check output permissions and Metal memory.",
        ) from error
    return {"outputs": files.outputs(), "details": details}
