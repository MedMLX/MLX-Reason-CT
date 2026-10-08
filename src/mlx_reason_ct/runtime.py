"""Require Apple Silicon Metal explicitly before learned computation."""

import platform
from importlib.metadata import version
from typing import Any

from mlx_reason_ct.errors import MissingDependencyError


def import_mlx() -> Any:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise MissingDependencyError(
            "NV-Reason-CT inference requires macOS on Apple Silicon with Metal",
            hint="Run on an Apple Silicon Mac with GPU access; install mlx-reason-ct.",
        )
    from medmlx_core.runtime import (
        import_mlx as shared_import_mlx,
    )

    try:
        return shared_import_mlx()
    except (ImportError, OSError) as error:
        raise MissingDependencyError(
            "NV-Reason-CT cannot load the MLX Metal runtime; "
            "check that MLX is installed and the Metal GPU is accessible.",
            hint="Install mlx-reason-ct on an Apple Silicon Mac with Metal GPU access.",
        ) from error


def host_info() -> dict[str, str]:
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": version("mlx"),
    }
