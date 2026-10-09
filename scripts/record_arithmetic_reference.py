"""Record seeded analytic references on macOS arm64 without a learned model."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mlx_reason_ct._payloads import JsonValue

type Array = NDArray[np.float32]
type FloatInput = NDArray[np.float32] | NDArray[np.float64]


def bfloat16(values: FloatInput) -> Array:
    """Round finite FP32 values to BF16, with ties to even, then widen."""
    bits = values.astype(np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def layer_norm(rng: np.random.Generator) -> dict[str, JsonValue]:
    values = bfloat16(rng.normal(size=(1, 2, 864)).astype(np.float32))
    weight = bfloat16(rng.uniform(0.75, 1.25, 864).astype(np.float32))
    bias = bfloat16(rng.uniform(-0.1, 0.1, 864).astype(np.float32))
    fp = values.astype(np.float64)
    centered = fp - fp.mean(axis=-1, keepdims=True)
    result = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
    result = (result * weight + bias).astype(np.float32)
    return {
        "input": values.tolist(),
        "weight": weight.tolist(),
        "bias": bias.tolist(),
        "eps": 1e-5,
        "float32": result.tolist(),
        "bfloat16": bfloat16(result).tolist(),
    }


def activations(rng: np.random.Generator) -> dict[str, JsonValue]:
    values = bfloat16(np.concatenate([[-8.0, 0.0, 8.0], rng.uniform(-8, 8, 61)]))
    fp = values.astype(np.float64)
    sigmoid = (1 / (1 + np.exp(-fp))).astype(np.float32)
    silu = (fp / (1 + np.exp(-fp))).astype(np.float32)
    return {
        "input": values.tolist(),
        "sigmoid_bf16": bfloat16(sigmoid).tolist(),
        "silu_bf16": bfloat16(silu).tolist(),
        "silu_fp32": silu.tolist(),
    }


def mean_squares() -> dict[str, JsonValue]:
    cases: dict[str, JsonValue] = {}
    for width in (128, 256, 2560):
        for rows in (1, 2, 17):
            seed = 431 + width * 100 + rows
            rng = np.random.Generator(np.random.PCG64(seed))
            values = bfloat16(rng.normal(0, 0.3, (1, rows, width)).astype(np.float32))
            expected = np.mean(values.astype(np.float64) ** 2, axis=-1, keepdims=True)
            cases[f"w{width}-r{rows}"] = {"seed": seed, "expected": expected.tolist()}
    return cases


def projections(rng: np.random.Generator) -> dict[str, JsonValue]:
    values = rng.normal(0, 0.3, (1, 2, 32)).astype(np.float32)
    weight = bfloat16(rng.normal(0, 0.3, (8, 32)).astype(np.float32))
    bias = bfloat16(rng.normal(0, 0.1, 8).astype(np.float32))
    raw = values.astype(np.float64) @ weight.astype(np.float64).T + bias
    rounded = bfloat16(values).astype(np.float64) @ weight.astype(np.float64).T + bias
    scale = np.abs(values.astype(np.float64)) @ np.abs(weight.astype(np.float64)).T
    rounded_scale = np.abs(bfloat16(values).astype(np.float64)) @ np.abs(weight).T
    return {
        "input": values.tolist(),
        "weight": weight.tolist(),
        "bias": bias.tolist(),
        "float32": raw.astype(np.float32).tolist(),
        "rounded_operands": rounded.astype(np.float32).tolist(),
        "bfloat16": bfloat16(rounded.astype(np.float32)).tolist(),
        "error_scale": (scale + np.abs(bias)).tolist(),
        "rounded_error_scale": (rounded_scale + np.abs(bias)).tolist(),
    }


def cache_normalization(rng: np.random.Generator) -> dict[str, JsonValue]:
    references: dict[str, JsonValue] = {}
    for kind in ("query", "key"):
        values = bfloat16(rng.normal(0, 0.3, 128).astype(np.float32))
        fp = values.astype(np.float64)
        ordinary = (fp / np.sqrt(np.sum(fp**2) + 1e-6)).astype(np.float32)
        squared = bfloat16(values * values)
        total = bfloat16(np.array(np.sum(squared.astype(np.float64)), dtype=np.float32))
        total = bfloat16(total + np.float32(1e-6))
        inverse = bfloat16(np.array(1 / np.sqrt(total.astype(np.float64)), dtype=np.float32))
        original = bfloat16(values * inverse)
        if kind == "query":
            ordinary = ordinary * np.float32(128**-0.5)
            original = original * np.float32(128**-0.5)
        references[kind] = {
            "input": values.tolist(),
            "float32": ordinary.tolist(),
            "bfloat16": original.tolist(),
        }
    references["recurrent"] = rng.normal(0, 0.3, 4).astype(np.float32).tolist()
    return references


def record() -> dict[str, JsonValue]:
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Record test references on macOS arm64 only")
    rng = np.random.Generator(np.random.PCG64(431))
    decay = -rng.uniform(0.001, 0.02, 64).astype(np.float32)
    return {
        "schema_version": 1,
        "recorded_on": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "method": "Seeded NumPy float64 formulas with explicit BF16 storage rounding; no Metal run",
        "seed": 431,
        "decay": {
            "input": decay.tolist(),
            "expected": np.cumsum(decay.astype(np.float64)).tolist(),
        },
        "layer_norm": layer_norm(rng),
        "activation": activations(rng),
        "mean_square": mean_squares(),
        "projection": projections(rng),
        "cache": cache_normalization(rng),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/nv_reason_ct/arithmetic.json")
    )
    args = parser.parse_args()
    reference = record()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
