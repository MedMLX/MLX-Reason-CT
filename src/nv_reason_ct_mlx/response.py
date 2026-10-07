"""Durable report text and completion contract for native MLX generation."""

from __future__ import annotations

import json
from pathlib import Path

from nv_reason_ct_mlx.errors import ModelExecutionError

THINK_END = "</think>"


def split_thinking(response: str) -> tuple[str | None, str]:
    """Separate template thinking-mode reasoning from the final answer.

    With thinking enabled the template opens ``<think>`` in the prompt, so the
    decoded output is ``reasoning</think>answer``. Without the closing tag the
    whole response is the answer.
    """
    if THINK_END not in response:
        return None, response
    reasoning, answer = response.split(THINK_END, 1)
    return reasoning.strip(), answer.strip()


def save_response(output_dir: Path, response: str, metrics: dict[str, object]) -> dict[str, str]:
    report_path = output_dir / "report.txt"
    response_path = output_dir / "model_response.json"
    thinking, report = split_thinking(response)
    report_path.write_text(report + "\n", encoding="utf-8")
    record: dict[str, object] = {"report": report}
    if thinking is not None:
        record["thinking"] = thinking
    record.update(requires_human_review=True, generation=metrics)
    response_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not report.strip() or not metrics["terminated_by_eos"]:
        raise ModelExecutionError(
            "NV-Reason-CT returned an empty or incomplete response. "
            "Partial output was retained; increase max_new_tokens for a truncated response."
        )
    return {"report": str(report_path), "response": str(response_path)}
