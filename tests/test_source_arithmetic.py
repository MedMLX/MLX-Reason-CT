"""Independent generation/cache recurrence and captured CUDA source arithmetic oracles."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _require_metal() -> None:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Native source arithmetic requires Metal")


@pytest.mark.parametrize(
    "sequence,limit,eos,truncated,expected",
    [
        ([3, 9, 4], 8, True, False, [3, 9]),
        ([10, 3], 8, True, False, [10]),
        ([3, 4, 5], 2, False, True, [3, 4]),
        ([3, 9, 4], 2, True, False, [3, 9]),
    ],
)
def test_native_greedy_generation_stops_before_consuming_eos_and_obeys_limit(
    sequence: list[int],
    limit: int,
    eos: bool,
    truncated: bool,
    expected: list[int],
) -> None:
    _require_metal()
    from mlx_reason_ct.api import generate
    from mlx_reason_ct.mlx_model import DecoderCache, mx
    from mlx_reason_ct.processor_mlx import VolumePrompt

    seen_positions: list[list[int]] = []

    class LogitOracle:
        eos_ids = (9, 10)
        step = 0

        def embed(self, ids: Any) -> Any:
            return mx.zeros((*ids.shape, 1))

        def decode(self, embeddings: Any, positions: Any, cache: Any = None) -> tuple[Any, Any]:
            seen_positions.append(np.array(positions).reshape(-1).tolist())
            if cache is None:
                cache = DecoderCache()
            cache.offset += embeddings.shape[1]
            return embeddings, cache

        def logits(self, hidden: Any) -> Any:
            values = np.full((1, 1, 11), -2.0, dtype=np.float32)
            values[0, 0, sequence[self.step]] = 2.0
            self.step += 1
            return mx.array(values)

    inputs = VolumePrompt(
        "",
        np.array([[1, 2, 3]]),
        np.array([[[0, 1, 1]]] * 3),
        np.ones((1, 3)),
        np.array([[0, 1, 0]]),
        -1,
        1,
    )
    tokens, metrics = generate(LogitOracle(), inputs, mx.zeros((1, 1, 1)), max_new_tokens=limit)
    assert tokens == expected
    assert metrics["terminated_by_eos"] is eos
    assert metrics["truncated_by_max_new_tokens"] is truncated
    assert len(seen_positions) == len(expected)
    if len(expected) > 1:
        assert seen_positions[1] == [2, 2, 2]


@pytest.mark.parametrize("length", [1, 63, 64, 65, 129])
def test_native_chunk_delta_matches_independent_recurrence_and_continuation(length: int) -> None:
    _require_metal()
    from mlx_reason_ct.chunk_delta import chunk_delta, mx

    rng = np.random.default_rng(431)
    q = rng.normal(size=(1, length, 2, 8)).astype(np.float32) * 0.1
    k = rng.normal(size=q.shape).astype(np.float32) * 0.1
    v = rng.normal(size=(1, length, 4, 6)).astype(np.float32)
    g = -rng.uniform(0.001, 0.2, size=(1, length, 4)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=g.shape).astype(np.float32)
    initial = rng.normal(size=(1, 4, 6, 8)).astype(np.float32) * 0.1
    # Independent scalar time recurrence in double precision, including a
    # non-square V,K state and nonzero prior state to catch axis/cache errors.
    expected_state = initial.astype(np.float64).transpose(0, 1, 3, 2).copy()
    expected: list[np.ndarray[Any, Any]] = []
    for t in range(length):
        qt = np.repeat(q[:, t], 2, axis=1).astype(np.float64)
        kt = np.repeat(k[:, t], 2, axis=1).astype(np.float64)
        expected_state *= np.exp(g[:, t].astype(np.float64))[..., None, None]
        delta = (v[:, t] - np.sum(expected_state * kt[..., None], axis=-2)) * beta[:, t, :, None]
        expected_state += kt[..., None] * delta[..., None, :]
        expected.append(np.sum(expected_state * qt[..., None], axis=-2))
    actual, state = chunk_delta(*map(mx.array, (q, k, v, g, beta, initial)))
    np.testing.assert_allclose(np.array(actual), np.stack(expected, axis=1), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(
        np.array(state), expected_state.transpose(0, 1, 3, 2), atol=2e-6, rtol=2e-5
    )
    if length > 1:
        split = min(62, length - 1)
        first, prior = chunk_delta(
            mx.array(q[:, :split]),
            mx.array(k[:, :split]),
            mx.array(v[:, :split]),
            mx.array(g[:, :split]),
            mx.array(beta[:, :split]),
            mx.array(initial),
        )
        rest, continued = chunk_delta(
            mx.array(q[:, split:]),
            mx.array(k[:, split:]),
            mx.array(v[:, split:]),
            mx.array(g[:, split:]),
            mx.array(beta[:, split:]),
            prior,
        )
        np.testing.assert_allclose(
            np.array(mx.concatenate([first, rest], axis=1)),
            np.stack(expected, axis=1),
            atol=2e-6,
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            np.array(continued), expected_state.transpose(0, 1, 3, 2), atol=2e-6, rtol=2e-5
        )


def test_native_decay_scan_preserves_captured_cuda_rounding() -> None:
    _require_metal()
    from mlx_reason_ct.chunk_delta import mx, source_cumsum

    path = Path(__file__).parent / "fixtures/nv_reason_ct/cuda_decay_scan.json"
    fixture = json.loads(path.read_text())
    values = mx.broadcast_to(mx.array(fixture["input"], dtype=mx.float32), fixture["shape"])
    actual = np.array(source_cumsum(values)[0, 0, 0]).view(np.uint32)
    np.testing.assert_array_equal(actual, np.array(fixture["expected_bits"], dtype=np.uint32))


@pytest.mark.parametrize("mixed", [False, True])
def test_native_bf16_layer_norm_preserves_cuda_fp32_affine_boundary(mixed: bool) -> None:
    _require_metal()
    from mlx_reason_ct.mlx_model import NativeModel, mx

    path = Path(__file__).parent / "fixtures/nv_reason_ct/cuda_bf16_layer_norm.json"
    fixture = json.loads(path.read_text())
    model = object.__new__(NativeModel)
    model.accumulate_float32 = mixed
    model.weights = {
        "norm.weight": mx.array(fixture["weight"], dtype=mx.bfloat16),
        "norm.bias": mx.array(fixture["bias"], dtype=mx.bfloat16),
    }
    result = model.norm(mx.array(fixture["input"], dtype=mx.bfloat16), "norm", fixture["eps"])
    assert result.dtype == (mx.float32 if mixed else mx.bfloat16)
    actual = np.array(result.astype(mx.float32))
    if mixed:
        np.testing.assert_allclose(actual, fixture["expected_float32"], atol=1e-6, rtol=0)
    else:
        np.testing.assert_array_equal(actual, np.array(fixture["expected"], dtype=np.float32))


@pytest.mark.parametrize("operation", ["sigmoid_bf16", "silu_bf16", "silu_fp32"])
def test_native_source_activation_preserves_independent_cuda_storage(operation: str) -> None:
    _require_metal()
    from mlx_reason_ct import source_arithmetic
    from mlx_reason_ct.mlx_model import NativeModel, mx

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/nv_reason_ct/cuda_bf16_activation.json").read_text()
    )
    inputs = (np.array(fixture["input_bf16_bits"], dtype=np.uint32) << 16).view(np.float32)
    values = mx.array(inputs, dtype=mx.float32 if operation == "silu_fp32" else mx.bfloat16)
    actual = (
        NativeModel.sigmoid(values)
        if operation == "sigmoid_bf16"
        else NativeModel.silu(values)
        if operation == "silu_bf16"
        else source_arithmetic.silu(values)
    )
    bits = np.array(actual.astype(mx.float32)).view(np.uint32)
    if operation != "silu_fp32":
        bits = bits >> 16
    np.testing.assert_array_equal(
        bits,
        np.array(fixture["expected_storage_bits"][operation], dtype=np.uint32),
    )


@pytest.mark.parametrize("width", [128, 256, 2560])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 7, 8, 15, 16, 17, 32])
def test_native_source_rms_mean_preserves_cuda_row_dispatch(width: int, rows: int) -> None:
    import hashlib

    _require_metal()
    from mlx_reason_ct import source_arithmetic
    from mlx_reason_ct.mlx_model import mx

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/nv_reason_ct/cuda_rms_mean_dispatch.json").read_text()
    )
    rng = np.random.default_rng(fixture["seed"])
    # Recreate the independently captured CUDA inputs, including BF16 ties-to-even.
    # Neither input rounding nor expected means use the production implementation.
    inputs_by_case = {
        (fixture_width, fixture_rows): rng.normal(
            0,
            0.3,
            (1, fixture_rows, fixture_width),
        ).astype(np.float32)
        for fixture_width in fixture["widths"]
        for fixture_rows in fixture["row_counts"]
    }
    values = inputs_by_case[(width, rows)]
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    inputs = rounded.view(np.float32)
    expected = fixture["fixtures"][f"w{width}-r{rows}"]
    assert hashlib.sha256(inputs.tobytes()).hexdigest() == expected["input_float32_storage_sha256"]
    square = mx.array(inputs) * mx.array(inputs)
    actual = np.array(source_arithmetic.mean_square(square)).reshape(-1).view(np.uint32)
    np.testing.assert_array_equal(
        actual,
        np.array(expected["expected_mean_float32_bits"], dtype=np.uint32),
    )


@pytest.mark.parametrize(
    "tower,precision,source_operator,length,source_length",
    [
        ("vision3d", "bfloat16", None, 2, None),
        ("language_model", "bfloat16", None, 2, None),
        ("language_model", "source_bfloat16", "in_proj_a", 13844, None),
        ("language_model", "source_bfloat16", "in_proj_a", 1, None),
        ("language_model", "source_bfloat16", "mlp.down_proj", 13844, None),
        ("language_model", "source_bfloat16", "mlp.down_proj", 1, None),
        ("language_model", "source_bfloat16", "mlp.down_proj", 256, 13844),
        ("language_model", "float32", "in_proj_a", 13844, None),
        ("language_model", "bfloat16", "in_proj_a", 13844, None),
    ],
)
def test_native_projection_preserves_independent_cuda_precision_boundaries(
    tower: str,
    precision: str,
    source_operator: str | None,
    length: int,
    source_length: int | None,
) -> None:
    _require_metal()
    from mlx_reason_ct.mlx_model import NativeModel, mx

    model = object.__new__(NativeModel)
    model.precision = precision
    model.accumulate_float32 = precision == "bfloat16"
    model.dtype = mx.float32 if precision == "float32" else mx.bfloat16
    if source_operator is not None:
        path = Path(__file__).parent / "fixtures/nv_reason_ct/cuda_bf16_projection_storage.json"
        fixture = json.loads(path.read_text())
        case = next(row for row in fixture["cases"] if row["name"] == source_operator)
        name = "model.language_model.layers.0." + (
            "linear_attn." + source_operator if source_operator == "in_proj_a" else source_operator
        )
        weights = np.zeros((case["output_width"], case["input_width"]), dtype=np.float32)
        inputs = np.zeros((1, length, case["input_width"]), dtype=np.float32)
        for index, value in case["weight_sparse_per_row"]:
            weights[:, index] = value
        for index, value in case["input_sparse"]:
            inputs[:, :, index] = value
        model.weights = {name + ".weight": mx.array(weights, dtype=model.dtype)}
        result = model.linear(
            mx.array(inputs, dtype=mx.bfloat16 if precision == "source_bfloat16" else mx.float32),
            name,
            source_length=source_length,
        )
        expected = (
            case["expected_prefill"]
            if precision == "source_bfloat16"
            and (source_length or length) == case["context_length"]
            else case["expected_cached"]
        )
        assert result.dtype == (mx.bfloat16 if precision == "source_bfloat16" else mx.float32)
        np.testing.assert_array_equal(
            np.array(result.astype(mx.float32)),
            np.full((1, length, case["output_width"]), expected, dtype=np.float32),
        )
        return
    path = Path(__file__).parent / "fixtures/nv_reason_ct/cuda_mixed_projection.json"
    fixture = json.loads(path.read_text())
    name = f"model.{tower}.projection"
    model.weights = {
        name + ".weight": mx.array(fixture["weight"], dtype=mx.bfloat16),
        name + ".bias": mx.array(fixture["bias"], dtype=mx.bfloat16),
    }
    result = model.linear(mx.array(fixture["input"], dtype=mx.float32), name)
    assert result.dtype == mx.float32
    expected = fixture["vision_expected" if tower == "vision3d" else "decoder_expected"]
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "source_bfloat16"])
def test_native_recurrent_cache_preserves_source_storage_boundary(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
) -> None:
    _require_metal()
    from mlx_reason_ct import mlx_model

    mx = mlx_model.mx
    fixture_path = Path(__file__).parent / "fixtures/nv_reason_ct/cuda_bf16_recurrent_cache.json"
    fixture = json.loads(fixture_path.read_text())
    norm_fixture = json.loads((fixture_path.parent / "cuda_bf16_l2norm.json").read_text())
    expected_values = fixture["stored_values" if precision == "source_bfloat16" else "input_values"]
    expected = np.broadcast_to(np.array(expected_values, dtype=np.float32), (1, 32, 128, 32, 4))
    expected = expected.reshape(1, 32, 128, 128)
    emitted = mx.broadcast_to(mx.array(fixture["input_values"]), (1, 32, 128, 32, 4))
    emitted = emitted.reshape(1, 32, 128, 128)
    observed: list[Any] = []
    observed_qk: list[tuple[Any, Any]] = []

    def computation(
        q: Any,
        k: Any,
        v: Any,
        g: Any,
        beta: Any,
        prior: Any,
        *,
        source_bfloat16: bool = False,
    ) -> tuple[Any, Any]:
        observed.append(prior)
        observed_qk.append((q, k))
        return mx.zeros(v.shape, dtype=mx.float32), emitted

    def projection(x: Any, name: str, *, source_length: int | None = None) -> Any:
        size = 8192 if name.endswith("in_proj_qkv") else 4096 if name.endswith("in_proj_z") else 32
        if name.endswith("out_proj"):
            size = 2560
        return mx.zeros((*x.shape[:2], size), dtype=x.dtype)

    model = object.__new__(mlx_model.NativeModel)
    model.precision = precision
    model.accumulate_float32 = precision == "bfloat16"
    model.dtype = mx.float32 if precision == "float32" else mx.bfloat16
    model.linear = projection
    # Supply captured post-convolution activations; the owner still normalizes
    # Q/K and passes them into recurrence, then stores/reuses its real cache.
    model.silu = lambda x: x
    raw_qk = [mx.tile(mx.array(norm_fixture[kind]["input"]), (16,)) for kind in ("query", "key")]
    raw = mx.concatenate([*raw_qk, mx.zeros(4096)]).astype(model.dtype)

    def convolution(value: Any, weight: Any, *, groups: int) -> Any:
        return mx.broadcast_to(raw, (1, value.shape[1] - 3, 8192))

    monkeypatch.setattr(mx, "conv1d", convolution)
    monkeypatch.setattr(
        model,
        "delta",
        SimpleNamespace(gated_delta_kernel=computation),
        raising=False,
    )
    model.weights = {
        "attention.conv1d.weight": mx.zeros((8192, 1, 4), dtype=model.dtype),
        "attention.A_log": mx.zeros(32, dtype=model.dtype),
        "attention.dt_bias": mx.zeros(32, dtype=model.dtype),
        "attention.norm.weight": mx.ones(128, dtype=model.dtype),
    }
    monkeypatch.setattr(mlx_model, "chunk_delta", computation)
    from mlx_reason_ct import source_arithmetic

    monkeypatch.setattr(source_arithmetic, "gated_delta", computation)
    state: dict[str, Any] = {}
    activation_dtype = mx.float32 if model.accumulate_float32 else model.dtype
    model.linear_attention(mx.zeros((1, 2, 2560), dtype=activation_dtype), "attention", state)
    for index, kind in enumerate(("query", "key")):
        if precision == "source_bfloat16":
            normalized = np.array(norm_fixture[kind]["expected"], dtype=np.float32)
        else:
            raw_values = np.array(norm_fixture[kind]["input"], dtype=np.float64)
            normalized = (raw_values / np.sqrt(np.sum(raw_values**2) + 1e-6)).astype(np.float32)
        if kind == "query":
            normalized = normalized * np.float32(128**-0.5)
        expected_qk = np.broadcast_to(normalized, (1, 2, 16, 128))
        actual_qk = np.array(observed_qk[0][index])
        if precision == "source_bfloat16":
            np.testing.assert_array_equal(actual_qk, expected_qk)
        else:
            np.testing.assert_allclose(actual_qk, expected_qk, atol=1e-6, rtol=0)
    expected_dtype = mx.bfloat16 if precision == "source_bfloat16" else mx.float32
    assert state["recurrent"].dtype == expected_dtype
    np.testing.assert_array_equal(np.array(state["recurrent"].astype(mx.float32)), expected)
    model.linear_attention(mx.zeros((1, 1, 2560), dtype=activation_dtype), "attention", state)
    assert observed[1].dtype == mx.float32
    np.testing.assert_array_equal(np.array(observed[1]), expected)
