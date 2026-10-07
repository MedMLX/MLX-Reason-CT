"""Shared durable text/completion contract for CUDA and native MLX."""

from __future__ import annotations

import json
from pathlib import Path

from nv_reason_ct_mlx.errors import ModelExecutionError


def save_response(output_dir: Path, response: str, metrics: dict[str, object]) -> dict[str, str]:
    report_path = output_dir / "report.txt"
    response_path = output_dir / "model_response.json"
    report_path.write_text(response + "\n", encoding="utf-8")
    response_path.write_text(
        json.dumps(
            {
                "report": response,
                "requires_human_review": True,
                "generation": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not response.strip() or not metrics["terminated_by_eos"]:
        raise ModelExecutionError(
            "NV-Reason-CT returned an empty or incomplete response. "
            "Partial output was retained; increase max_new_tokens for a truncated response."
        )
    return {"report": str(report_path), "response": str(response_path)}
