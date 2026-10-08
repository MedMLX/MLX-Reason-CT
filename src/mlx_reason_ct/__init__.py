"""MLX-Reason-CT: CT reporting on Apple Silicon with lazy public exports."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_reason_ct.api import generate_report
    from mlx_reason_ct.errors import InvalidInputError, InvalidPromptError, ModelExecutionError

__all__ = ["generate_report", "InvalidInputError", "InvalidPromptError", "ModelExecutionError"]

_EXPORTS = {
    "generate_report": "mlx_reason_ct.api",
    "InvalidInputError": "mlx_reason_ct.errors",
    "InvalidPromptError": "mlx_reason_ct.errors",
    "ModelExecutionError": "mlx_reason_ct.errors",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
