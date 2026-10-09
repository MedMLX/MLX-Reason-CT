"""Offline generation with pinned arithmetic profiles and durable completion status."""

from __future__ import annotations

import json
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
    anatomy_region: str,
    enable_thinking: bool,
    max_new_tokens: int,
    overwrite: bool,
    precision: str,
) -> str:
    if not isinstance(cast(object, precision), str) or precision not in PRECISION_PROFILES:
        raise InvalidInputError(
            "Unsupported NV-Reason-CT MLX precision profile; "
            "choose float32, bfloat16 or bfloat16_fp32."
        )
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
    if not isinstance(cast(object, enable_thinking), bool) or not isinstance(
        cast(object, overwrite), bool
    ):
        raise InvalidInputError("enable_thinking and overwrite must be booleans")
    return prompt


def generate(
    model: GenerationModel,
    inputs: VolumePrompt,
    embeddings: Array,
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
        token = int(cast(int, mx.argmax(logit[0, -1]).item()))
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
    precision: str = "float32",
    overwrite: bool = False,
) -> dict[str, object]:
    """Run one HU NIfTI CT locally on Metal; retain partial output on truncation.

    Without a prompt, a structured report for ``anatomy_region`` is requested.
    Existing report files require ``overwrite=True`` to replace.
    """
    prompt = _validate_generation_options(
        prompt, anatomy_region, enable_thinking, max_new_tokens, overwrite, precision
    )
    model_dir = Path(model_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    files = ReportFiles(output_dir)
    files.require_writable(overwrite=overwrite)
    manifest, processor = _load_assets(model_dir)
    pixels, crop = processor.image(Path(source), anatomy_region)
    inputs = processor.prompt(prompt, enable_thinking)
    mx = import_mlx()
    started = perf_counter()
    mx.reset_peak_memory()
    native_profile = PRECISION_PROFILES[precision]
    model = _load_model(model_dir, precision=native_profile)
    features, embeddings = model.vision(mx.array(pixels))
    if not bool(mx.all(mx.isfinite(embeddings)).item()):
        raise ModelExecutionError("NV-Reason-CT produced nonfinite image embeddings")
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
        "source_sha256": manifest["source_sha256"],
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
