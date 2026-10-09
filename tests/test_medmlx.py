"""MedMLX discovery, fail-fast boundaries and shared report orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from importlib import import_module
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TypedDict, cast, get_type_hints

import numpy as np
import pytest
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from test_weights import source_checkpoint

from mlx_reason_ct import api, mlx_weights
from mlx_reason_ct._host_types import HostArray, Nibabel, Tokenizers
from mlx_reason_ct.errors import InvalidPromptError
from mlx_reason_ct.medmlx import MODEL, readiness, run
from mlx_reason_ct.processor_mlx import VolumePrompt

nib = cast(Nibabel, import_module("nibabel"))
tokenizers = cast(Tokenizers, import_module("tokenizers"))


class GenerationControls(TypedDict, total=False):
    prompt: str
    anatomy_region: str
    enable_thinking: bool
    max_new_tokens: int
    precision: str


@pytest.fixture
def image(tmp_path: Path) -> Path:
    values = np.random.Generator(np.random.PCG64(3)).uniform(-1000, 1000, (2, 2, 2))
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(values.astype(np.float32), np.eye(4)), path)
    return path


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = source_checkpoint(tmp_path / "source", monkeypatch)
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "FINDINGS:": 1, "Normal.": 2, "<|image_pad|>": 3, "[EOS]": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.add_special_tokens(["<|image_pad|>", "[EOS]"])
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "config.json").write_text("{}")
    (source / "generation_config.json").write_text('{"eos_token_id": [4]}')
    (source / "chat_template.jinja").write_text(
        "<|image_pad|>{{ messages[0]['content'][1]['text'] }}"
    )
    (source / "image_processor_3d/preprocessor_config.json").write_text(
        json.dumps({"final_grid_size": [1, 1, 1], "pixdim": [1, 1, 1], "spatial_size": [2, 2, 2]})
    )
    monkeypatch.setattr(
        mlx_weights,
        "PINS",
        {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in mlx_weights.PINS
        },
    )
    path = tmp_path / "bundle"
    mlx_weights.convert_checkpoint(source, path)
    return path


def test_declaration_entry_point_and_signature() -> None:
    assert "Operating System :: MacOS :: MacOS X" in (
        distribution("mlx-reason-ct").metadata.get_all("Classifier") or []
    )
    assert issubclass(InvalidPromptError, InvalidInputError)
    assert MODEL["id"] == "mlx-reason-ct"
    assert MODEL["task"] == "generation"
    assert MODEL["run"] == "mlx_reason_ct.medmlx:run"
    entry = next(
        entry
        for entry in distribution("mlx-reason-ct").entry_points
        if entry.group == "medmlx.models"
    )
    assert entry.name == "mlx-reason-ct"
    assert entry.value == "mlx_reason_ct.medmlx:MODEL"
    assert entry.load() is MODEL
    parameters = inspect.signature(run).parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert parameters["overwrite"].default is False
    assert get_type_hints(run)["overwrite"] is bool
    assert set(parameters) == {
        "image_path",
        "weights_path",
        "output_dir",
        "overwrite",
        "prompt",
        "anatomy_region",
        "enable_thinking",
        "max_new_tokens",
        "precision",
    }


def test_declaration_and_public_names_load_no_mlx_or_reference_stack() -> None:
    code = """
import importlib.abc
import sys
class BlockMlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'monai'}:
            raise ImportError('blocked dependency: ' + fullname)
sys.meta_path.insert(0, BlockMlx())
import mlx_reason_ct
assert 'mlx_reason_ct.api' not in sys.modules
import mlx_reason_ct.medmlx as m
assert m.MODEL['run'] == 'mlx_reason_ct.medmlx:run'
assert m.readiness()['ready'] is False
from mlx_reason_ct import (
    generate_report, InvalidInputError, InvalidPromptError, ModelExecutionError,
)
assert callable(generate_report)
assert 'generate_report' in dir(mlx_reason_ct)
try:
    mlx_reason_ct.unknown_name
except AttributeError:
    pass
else:
    raise AssertionError('unknown name was accepted')
loaded = {name.split('.')[0] for name in sys.modules}
assert not loaded & {'mlx', 'torch', 'monai'}, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "name,message", [("missing.nii.gz", "does not exist"), ("ct.png", "must be a NIfTI")]
)
def test_bad_image_path(tmp_path: Path, name: str, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        run(image_path=tmp_path / name, weights_path=tmp_path, output_dir=tmp_path / "out")


@pytest.mark.parametrize("invalid", ["garbage", "4d", "nonfinite", "uncoded"])
def test_bad_image_contents(tmp_path: Path, bundle: Path, invalid: str) -> None:
    path = tmp_path / "invalid.nii.gz"
    if invalid == "garbage":
        path.write_bytes(b"not NIfTI")
    else:
        values = np.zeros((2, 2, 2, 2) if invalid == "4d" else (2, 2, 2), dtype=np.float32)
        if invalid == "nonfinite":
            values.flat[0] = np.nan
        volume = nib.Nifti1Image(values, np.eye(4))
        if invalid == "uncoded":
            volume.set_qform(None, code=0)
            volume.set_sform(None, code=0)
        nib.save(volume, path)
    with pytest.raises(InvalidInputError):
        run(image_path=path, weights_path=bundle, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("invalid", ["missing", "file", "manifest", "runtime", "shard"])
def test_bad_weights(tmp_path: Path, image: Path, bundle: Path, invalid: str) -> None:
    if invalid == "missing":
        bundle = tmp_path / "missing"
    elif invalid == "file":
        bundle = tmp_path / "unconverted.safetensors"
        bundle.write_bytes(b"not a converted bundle directory")
    elif invalid == "manifest":
        (bundle / "mlx/manifest.json").write_text("{")
    elif invalid == "runtime":
        (bundle / "config.json").write_text("changed")
    else:
        (bundle / "mlx/vision.safetensors").write_bytes(b"changed")
    with pytest.raises(AssetNotReadyError, match="converted") as caught:
        run(image_path=image, weights_path=bundle, output_dir=tmp_path / "out")
    assert caught.value.hint
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["report.txt", "model_response.json", "run.json"])
def test_overwrite_refusal(tmp_path: Path, image: Path, name: str) -> None:
    output = tmp_path / name
    output.write_text("keep")
    with pytest.raises(InvalidInputError, match="overwrite=true"):
        run(image_path=image, weights_path=tmp_path, output_dir=tmp_path)
    with pytest.raises(InvalidInputError, match="overwrite=true"):
        api.generate_report(image, tmp_path, model_dir=tmp_path)
    assert output.read_text() == "keep"


def test_output_directory_and_boolean_guards(tmp_path: Path, image: Path) -> None:
    output = tmp_path / "taken"
    output.write_text("keep")
    with pytest.raises(InvalidInputError, match="not a directory"):
        run(image_path=image, weights_path=tmp_path, output_dir=output)
    with pytest.raises(InvalidInputError, match="overwrite must be a boolean"):
        run(
            image_path=image,
            weights_path=tmp_path,
            output_dir=tmp_path,
            overwrite=cast(bool, "yes"),
        )


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"prompt": " "}, "prompt"),
        ({"anatomy_region": "brain"}, "anatomy_region"),
        ({"enable_thinking": "yes"}, "booleans"),
        ({"precision": "float16"}, "precision profile"),
        ({"precision": {"name": "float32"}}, "precision profile"),
    ],
)
def test_generation_controls(
    tmp_path: Path, image: Path, arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        run(
            image_path=image,
            weights_path=tmp_path,
            output_dir=tmp_path / "out",
            **cast(GenerationControls, arguments),
        )


def test_runtime_uses_core_and_preserves_metal_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    from medmlx_core import runtime as core_runtime

    from mlx_reason_ct import runtime

    sentinel = object()
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(core_runtime, "import_mlx", lambda: sentinel)
    assert runtime.import_mlx() is sentinel
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    with pytest.raises(MissingDependencyError, match="Apple Silicon"):
        runtime.import_mlx()
    assert readiness()["ready"] is False


@pytest.mark.parametrize("error", [ImportError("missing mlx"), OSError("broken library")])
def test_runtime_load_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch, error: ImportError | OSError
) -> None:
    from medmlx_core import runtime as core_runtime

    from mlx_reason_ct import runtime

    def fail() -> None:
        raise error

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(core_runtime, "import_mlx", fail)
    with pytest.raises(MissingDependencyError, match="Metal runtime"):
        runtime.import_mlx()
    assert readiness()["ready"] is False


@pytest.fixture
def synthetic_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real assets/processor/writers; replace fixed-size learned computation."""
    mx = SimpleNamespace(
        array=np.asarray,
        all=np.all,
        isfinite=np.isfinite,
        reset_peak_memory=lambda: None,
        get_peak_memory=lambda: 0,
    )
    rng = np.random.Generator(np.random.PCG64(3))
    embeddings = rng.standard_normal((1, 1, 4), dtype=np.float32)

    class Model:
        def __init__(self, model_dir: Path, *, precision: str) -> None:
            assert model_dir.is_dir()
            self.dtype = "float32" if precision == "float32" else "bfloat16"

        def vision(self, pixels: HostArray) -> tuple[HostArray, HostArray]:
            assert pixels.shape == (1, 1, 2, 2, 2)
            assert pixels.dtype == np.float32
            return embeddings, embeddings

    def generate(
        model: object, inputs: VolumePrompt, embeddings: HostArray, *, max_new_tokens: int
    ) -> tuple[list[int], dict[str, object]]:
        assert inputs.input_ids.shape == (1, 2)
        assert max_new_tokens == 16
        return [1, 2, 4], {"terminated_by_eos": True, "generated_tokens": 3}

    monkeypatch.setattr(api, "import_mlx", lambda: mx)

    module = ModuleType("mlx_reason_ct.mlx_model")
    setattr(module, "NativeModel", Model)
    monkeypatch.setitem(sys.modules, "mlx_reason_ct.mlx_model", module)
    monkeypatch.setattr(api, "generate", generate)


@pytest.mark.parametrize(
    "precision,native_profile,arithmetic,weight_dtype",
    [
        ("float32", "float32", "float32", "float32"),
        ("bfloat16", "source_bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16_fp32", "bfloat16", "float32", "bfloat16"),
    ],
)
def test_runner_uses_public_api_and_writes_json_outputs(
    tmp_path: Path,
    image: Path,
    bundle: Path,
    synthetic_generation: None,
    precision: str,
    native_profile: str,
    arithmetic: str,
    weight_dtype: str,
) -> None:
    output = tmp_path / "out"
    with pytest.warns(RuntimeWarning, match="superior-edge"):
        result = run(
            image_path=image,
            weights_path=bundle,
            output_dir=output,
            prompt="Describe CT",
            anatomy_region="abdomen",
            enable_thinking=True,
            max_new_tokens=16,
            precision=precision,
        )
    outputs = cast(dict[str, str], result["outputs"])
    assert set(outputs) == {"report", "response", "run"}
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in outputs.values())
    assert Path(outputs["report"]).read_text() == "FINDINGS: Normal.\n"
    response = json.loads(Path(outputs["response"]).read_text())
    assert response["schema_version"] == 1
    assert response["requires_human_review"] is True
    details = json.loads(json.dumps(result["details"], allow_nan=False))
    assert details == json.loads(Path(outputs["run"]).read_text())
    assert details["status"] == "succeeded"
    assert details["clinical_validation"] is False
    assert details["generated_token_ids"] == [1, 2, 4]
    assert details["source_sha256"] == mlx_weights.PINS
    assert details["generation"]["precision"] == precision
    assert details["generation"]["native_arithmetic_profile"] == native_profile
    assert details["generation"]["decoder_arithmetic"] == arithmetic
    assert details["generation"]["learned_weight_dtype"] == weight_dtype
    Path(outputs["report"]).write_text("old")
    with pytest.warns(RuntimeWarning, match="superior-edge"):
        run(
            image_path=image,
            weights_path=bundle,
            output_dir=output,
            overwrite=True,
            max_new_tokens=16,
        )
    assert Path(outputs["report"]).read_text() == "FINDINGS: Normal.\n"


def test_partial_generation_retains_failure_record(
    tmp_path: Path,
    image: Path,
    bundle: Path,
    synthetic_generation: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def generate(*args: object, **kwargs: object) -> tuple[list[int], dict[str, object]]:
        return [1, 2], {"terminated_by_eos": False, "generated_tokens": 2}

    monkeypatch.setattr(api, "generate", generate)
    with pytest.warns(RuntimeWarning, match="superior-edge"):
        with pytest.raises(ModelExecutionError, match="incomplete"):
            run(image_path=image, weights_path=bundle, output_dir=tmp_path / "out")
    assert (tmp_path / "out/report.txt").read_text() == "FINDINGS: Normal.\n"
    assert json.loads((tmp_path / "out/run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "error,expected",
    [
        (ImportError("tokenizers absent"), MissingDependencyError),
        (RuntimeError("Metal allocation failed"), ModelExecutionError),
        (OSError("disk full"), ModelExecutionError),
    ],
)
def test_runner_execution_error_types(
    tmp_path: Path,
    image: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: type[Exception],
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise error

    monkeypatch.setattr(api, "generate_report", fail)
    with pytest.raises(expected):
        run(image_path=image, weights_path=tmp_path, output_dir=tmp_path / "out")


def test_cli_passes_controls_to_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx_reason_ct import cli

    seen: dict[str, object] = {}

    def generate_report(*args: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "succeeded"}

    monkeypatch.setattr(cli, "generate_report", generate_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlx-reason-ct",
            "report",
            "--input",
            "ct.nii.gz",
            "--model-dir",
            "models",
            "--output-dir",
            "out",
            "--overwrite",
            "--precision",
            "bfloat16",
        ],
    )
    assert cli.main() == 0
    assert seen["overwrite"] is True
    assert seen["precision"] == "bfloat16"
