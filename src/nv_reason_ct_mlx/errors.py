"""Errors at the standalone input and execution boundaries."""


class InvalidInputError(ValueError):
    """An input or model bundle violates the supported contract."""


class InvalidPromptError(InvalidInputError):
    """A prompt cannot represent exactly one CT volume."""


class ModelExecutionError(RuntimeError):
    """Inference produced invalid or incomplete output."""


class MissingDependencyError(RuntimeError):
    """The required Metal runtime is unavailable."""
