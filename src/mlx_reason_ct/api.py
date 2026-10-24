"""Offline generation with pinned arithmetic profiles and durable completion status."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

from mlx_reason_ct._payloads import BundleManifest

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, GenerationModel
    from mlx_reason_ct.mlx_model import NativeModel

from mlx_reason_ct.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from mlx_reason_ct.mlx_weights import read_manifest
from mlx_reason_ct.processor_mlx import Processor, VolumePrompt
from mlx_reason_ct.response import ReportFiles, save_response
from mlx_reason_ct.runtime import host_info, import_mlx

PRECISION_PROFILES: dict[str, str] = {
    "float32": "float32",
    "bfloat16": "source_bfloat16",
    "bfloat16_fp32": "bfloat16",
}


def _load_assets(model_dir: Path) -> tuple[BundleManifest, Processor]:
    try:
        return read_manifest(model_dir), Processor(model_dir)
    except (InvalidInputError, OSError, ValueError, KeyError, TypeError) as error:
        raise AssetNotReadyError(
            f"NV-Reason-CT converted bundle is missing or mismatched at {model_dir}: {error}",
            reason=f"invalid converted bundle {model_dir}",
            hint="Pass the bundle root containing mlx/manifest.json, tokenizer and FP32 shards; "
            "download it with mlx-reason-ct download or convert the pinned source checkpoint.",
        ) from error


def _load_model(model_dir: Path, *, precision: str) -> NativeModel:
    from mlx_reason_ct.mlx_model import NativeModel

    try:
        return NativeModel(model_dir, precision=precision)
    except (InvalidInputError, OSError, ValueError, KeyError, RuntimeError) as error:
        raise AssetNotReadyError(
            f"Cannot load NV-Reason-CT FP32 tensors at {model_dir}: {error}",
            hint="Verify the converted bundle with mlx-reason-ct verify --model-dir PATH.",
        ) from error


def _validate_generation_options(
    prompt: str | None,
    anatomy_region: str | None,
    enable_thinking: bool,
    max_new_tokens: int,
    overwrite: bool,
    precision: str,
    history: Sequence[tuple[str, str]] | None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(cast(object, precision), str) or precision not in PRECISION_PROFILES:
        raise InvalidInputError(
            "Unsupported NV-Reason-CT MLX precision profile; "
            "choose float32, bfloat16 or bfloat16_fp32."
        )
    if anatomy_region is not None and anatomy_region not in {"chest", "abdomen"}:
        raise InvalidInputError("anatomy_region must be chest, abdomen, or None")
    if prompt is None:
        prompt = (
            f"write a structured {anatomy_region} CT report"
            if anatomy_region is not None
            else "write a structured CT report"
        )
    # Annotations do not bind callers, so validate the runtime values.
    token_limit = cast(object, max_new_tokens)
    if (
        isinstance(token_limit, bool)
        or not isinstance(token_limit, int)
        or not (1 <= token_limit <= 8192)
    ):
        raise InvalidInputError("max_new_tokens must be an integer from 1 to 8192")
    prompt_text = cast(object, prompt)
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise InvalidInputError("prompt must be nonempty text")
    if not isinstance(cast(object, enable_thinking), bool) or not isinstance(
        cast(object, overwrite), bool
    ):
        raise InvalidInputError("enable_thinking and overwrite must be booleans")
    resolved_history = _validate_history(history)
    return prompt, resolved_history


def _validate_history(
    history: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate prior ``(question, answer)`` turns for a follow-up request.

    Upstream's own chat template supports replaying completed turns ahead of
    a new question (the model card documents "follow-up questions and
    clarification requests"); each entry here is one already-answered turn,
    oldest first, with the CT volume attached only once, on the first turn.
    """
    if history is None:
        return ()
    turns = cast(object, history)
    if isinstance(turns, str | bytes) or not isinstance(turns, Sequence):
        raise InvalidInputError("history must be a sequence of (question, answer) pairs")
    resolved: list[tuple[str, str]] = []
    for turn in cast(Sequence[object], turns):
        if isinstance(turn, str | bytes) or not isinstance(turn, Sequence):
            raise InvalidInputError("Each history turn must be a (question, answer) pair")
        values = cast(Sequence[object], turn)
        if len(values) != 2:
            raise InvalidInputError("Each history turn must be a (question, answer) pair")
        question, answer = values
        if (
            not isinstance(question, str)
            or not isinstance(answer, str)
            or not question.strip()
            or not answer.strip()
        ):
            raise InvalidInputError("Each history turn must hold nonempty question/answer text")
        resolved.append((question, answer))
    return tuple(resolved)


def generate(
    model: GenerationModel,
    inputs: VolumePrompt,
    embeddings: Array,
    *,
    max_new_tokens: int,
) -> tuple[list[int], dict[str, object]]:
    """Greedy generation with the pinned EOS set and request-owned hybrid cache.

    Args:
        model: loaded network.
        inputs: tokenized prompt; its image placeholders start at ``image_start``.
        embeddings: projected CT embeddings, shape ``(1, N, hidden)``, replacing
            the ``N`` placeholder token embeddings.
        max_new_tokens: generation limit, including the EOS token if produced.

    Returns:
        ``(token_ids, metrics)``. Generation stops at the first EOS id or at
        ``max_new_tokens``; ``metrics`` records whether it ended at EOS or was
        truncated, token counts, and decoder-cache sizes.

    Raises:
        ModelExecutionError: a logit tensor is not finite.
    """
    mx = import_mlx()
    tokens = mx.array(inputs.input_ids)
    start = inputs.image_start
    length = embeddings.shape[1]
    hidden = mx.concatenate(
        [model.embed(tokens[:, :start]), embeddings, model.embed(tokens[:, start + length :])],
        axis=1,
    )
    hidden, cache = model.decode(hidden, mx.array(inputs.position_ids))
    # Only the last position is projected to logits; the prompt-length hidden states
    # are dropped afterwards.
    logit = model.logits(hidden[:, -1:])
    mx.eval(logit, *cache.arrays())
    del hidden
    initial_bytes = cache.nbytes
    growth: list[dict[str, int]] = [{"decoded_tokens": 0, "cache_bytes": initial_bytes}]
    output: list[int] = []
    for index in range(max_new_tokens):
        if not bool(mx.all(mx.isfinite(logit)).item()):
            raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
        token = int(cast(int, mx.argmax(logit[0, -1]).item()))
        output.append(token)
        if token in model.eos_ids or index + 1 == max_new_tokens:
            break
        # Image tokens consume fewer MRoPE positions than sequence slots, so new text
        # positions are the cache length shifted by rope_delta.
        position = mx.full((3, 1, 1), cache.offset + inputs.rope_delta, dtype=mx.int64)
        hidden, cache = model.decode(model.embed(mx.array([[token]])), position, cache)
        logit = model.logits(hidden)
        mx.eval(logit, *cache.arrays())
        if len(output) in {1, 8, 32, 128, 256, 512, 1024, 2048, 4096}:
            growth.append({"decoded_tokens": len(output), "cache_bytes": cache.nbytes})
    mx.synchronize()
    # The last sampled token is never fed back, so the cache holds one fewer token.
    growth.append({"decoded_tokens": len(output) - 1, "cache_bytes": cache.nbytes})
    complete = bool(output) and output[-1] in model.eos_ids
    return output, {
        "generated_tokens": len(output),
        "terminated_by_eos": complete,
        "truncated_by_max_new_tokens": len(output) >= max_new_tokens and not complete,
        "prompt_tokens": int(tokens.shape[1]),
        "initial_cache_bytes": initial_bytes,
        "final_cache_bytes": cache.nbytes,
        "cache_growth": growth,
    }


def generate_report(
    source: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    prompt: str | None = None,
    anatomy_region: str | None = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    precision: str = "float32",
    overwrite: bool = False,
    history: Sequence[tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Run one HU NIfTI CT locally on Metal; retain partial output on truncation.

    Without a prompt, a structured report for ``anatomy_region`` is requested.
    ``anatomy_region`` is ``"chest"``, ``"abdomen"``, or ``None`` for the
    pinned processor's own centered crop, used upstream when its chest/abdomen
    heuristic is unreliable for a given geometry. ``history`` continues a
    conversation about the same CT: a sequence of already-answered
    ``(question, answer)`` turns, oldest first, with ``prompt``/the default
    report request as the new, still-unanswered turn. Existing report files
    require ``overwrite=True`` to replace.

    ``source`` is a HU NIfTI CT and ``model_dir`` the converted bundle root.
    ``precision`` is a public profile name (see ``PRECISION_PROFILES``) and
    ``max_new_tokens`` must be an integer in 1-8192. Generation is greedy.

    Returns:
        The ``run.json`` record: status, generation metrics, CT geometry,
        generated token ids and runtime details. The report and response are
        written to ``output_dir`` next to it. The record always sets
        ``requires_human_review`` and ``clinical_validation: false``.

    Raises:
        InvalidInputError: invalid options, unreadable CT, or existing outputs
            without ``overwrite``.
        AssetNotReadyError: the converted bundle is missing or mismatched.
        MissingDependencyError: Metal or MLX is unavailable.
        ModelExecutionError: non-finite activations, or an empty or truncated
            response (partial files and a ``status: failed`` run record are kept).
    """
    prompt, resolved_history = _validate_generation_options(
        prompt, anatomy_region, enable_thinking, max_new_tokens, overwrite, precision, history
    )
    model_dir = Path(model_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    files = ReportFiles(output_dir)
    files.require_writable(overwrite=overwrite)
    manifest, processor = _load_assets(model_dir)
    pixels, crop = processor.image(Path(source), anatomy_region)
    inputs = processor.prompt(prompt, enable_thinking, resolved_history)
    mx = import_mlx()
    started = perf_counter()
    mx.reset_peak_memory()
    native_profile = PRECISION_PROFILES[precision]
    model = _load_model(model_dir, precision=native_profile)
    features, embeddings = model.vision(mx.array(pixels))
    if not bool(mx.all(mx.isfinite(embeddings)).item()):
        raise ModelExecutionError("NV-Reason-CT produced nonfinite image embeddings")
    # Only the projected embeddings are needed for decoding.
    del features
    tokens, metrics = generate(model, inputs, embeddings, max_new_tokens=max_new_tokens)
    metrics.update(
        {
            "elapsed_s": round(perf_counter() - started, 4),
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "precision": precision,
            "learned_weight_dtype": str(model.dtype),
            "decoder_arithmetic": "bfloat16" if precision == "bfloat16" else "float32",
            "native_arithmetic_profile": native_profile,
            "synchronization": "mx.eval per layer/token and mx.synchronize",
            "anatomy_region": anatomy_region,
            "prior_turns": len(resolved_history),
        }
    )
    files.require_writable(overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": 1,
        "schema": "nv_reason_ct_mlx.run.v1",
        "status": "succeeded",
        "source": str(Path(source)),
        "model_revision": manifest["revision"],
        "bundle_engine": manifest["engine"],
        "generation": metrics,
        "ct_geometry": crop.geometry(),
        "generated_token_ids": tokens,
        "runtime": host_info(),
        "requires_human_review": True,
        "clinical_validation": False,
    }
    # Completion failure is recorded after retaining report and response files.
    try:
        save_response(output_dir, processor.decode(tokens), metrics)
    except ModelExecutionError as error:
        result.update(status="failed", error=str(error))
        Path(files.outputs()["run"]).write_text(json.dumps(result, indent=2) + "\n")
        raise
    Path(files.outputs()["run"]).write_text(json.dumps(result, indent=2) + "\n")
    return result
