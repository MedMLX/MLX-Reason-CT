"""Offline FP32 generation with request-owned state and durable completion status."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from mlx_reason_ct.errors import InvalidInputError, ModelExecutionError
from mlx_reason_ct.mlx_weights import read_manifest
from mlx_reason_ct.processor_mlx import Processor, VolumePrompt
from mlx_reason_ct.response import save_response
from mlx_reason_ct.runtime import host_info, import_mlx


def generate(
    model: Any,
    inputs: VolumePrompt,
    embeddings: Any,
    *,
    max_new_tokens: int,
) -> tuple[list[int], dict[str, object]]:
    """Greedy generation with the pinned EOS set and request-owned hybrid cache."""
    mx = import_mlx()
    tokens = mx.array(inputs.input_ids)
    start = inputs.image_start
    length = embeddings.shape[1]
    hidden = mx.concatenate(
        [model.embed(tokens[:, :start]), embeddings, model.embed(tokens[:, start + length :])],
        axis=1,
    )
    hidden, cache = model.decode(hidden, mx.array(inputs.position_ids))
    logit = model.logits(hidden[:, -1:])
    mx.eval(logit, *cache.arrays())
    del hidden
    initial_bytes = cache.nbytes
    growth: list[dict[str, int]] = [{"decoded_tokens": 0, "cache_bytes": initial_bytes}]
    output: list[int] = []
    for index in range(max_new_tokens):
        if not bool(mx.all(mx.isfinite(logit)).item()):
            raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
        token = int(mx.argmax(logit[0, -1]).item())
        output.append(token)
        if token in model.eos_ids or index + 1 == max_new_tokens:
            break
        position = mx.full((3, 1, 1), cache.offset + inputs.rope_delta, dtype=mx.int64)
        hidden, cache = model.decode(model.embed(mx.array([[token]])), position, cache)
        logit = model.logits(hidden)
        mx.eval(logit, *cache.arrays())
        if len(output) in {1, 8, 32, 128, 256, 512, 1024, 2048, 4096}:
            growth.append({"decoded_tokens": len(output), "cache_bytes": cache.nbytes})
    mx.synchronize()
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
    anatomy_region: str = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
) -> dict[str, object]:
    """Run one HU NIfTI CT locally on Metal; retain partial output on truncation.

    Without a prompt, a structured report for ``anatomy_region`` is requested.
    """
    if anatomy_region not in {"chest", "abdomen"}:
        raise InvalidInputError("anatomy_region must be chest or abdomen")
    if prompt is None:
        prompt = f"write a structured {anatomy_region} CT report"
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
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    manifest = read_manifest(model_dir)
    processor = Processor(model_dir)
    pixels, crop = processor.image(Path(source), anatomy_region)
    inputs = processor.prompt(prompt, enable_thinking)
    mx = import_mlx()
    from mlx_reason_ct.mlx_model import NativeModel

    started = perf_counter()
    mx.reset_peak_memory()
    model = NativeModel(model_dir)
    features, embeddings = model.vision(mx.array(pixels))
    if not bool(mx.all(mx.isfinite(embeddings)).item()):
        raise ModelExecutionError("NV-Reason-CT produced nonfinite image embeddings")
    del features
    tokens, metrics = generate(model, inputs, embeddings, max_new_tokens=max_new_tokens)
    metrics.update(
        {
            "elapsed_s": round(perf_counter() - started, 4),
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "precision": "float32",
            "synchronization": "mx.eval per layer/token and mx.synchronize",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
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
        (output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
