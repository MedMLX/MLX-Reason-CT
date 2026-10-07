"""Require Apple Silicon Metal explicitly before learned computation."""

import platform
from importlib import import_module
from importlib.metadata import version
from typing import Any

from nv_reason_ct_mlx.errors import MissingDependencyError


def import_mlx() -> Any:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise MissingDependencyError("Inference requires macOS on Apple Silicon")
    mx: Any = import_module("mlx.core")
    if not mx.metal.is_available():
        raise MissingDependencyError("Inference requires the Metal GPU")
    mx.set_default_device(mx.gpu)
    return mx


def host_info() -> dict[str, str]:
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": version("mlx"),
    }
