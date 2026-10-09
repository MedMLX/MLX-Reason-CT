"""Pinned JSON asset schemas; host tooling need not load the native runtime."""

from typing import NotRequired, TypedDict

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None


class ArithmeticTable(TypedDict):
    file: str
    sha256: str
    source_npy_sha256: str


class ArithmeticManifest(TypedDict):
    tables: dict[str, ArithmeticTable]
    kernels_sha256: str


class EncodedTable(TypedDict):
    dtype: str
    codec: str
    decoded_bytes: int
    source_npy_sha256: str
    decoded_sha256: str
    shape: list[int]


class KernelSpec(TypedDict):
    inputs: list[str]
    outputs: list[str]
    header: str
    source: str
    atomic_outputs: NotRequired[bool]


class BundleManifest(TypedDict):
    engine: str
    revision: str
    source_sha256: dict[str, str]
    dtype: str
    runtime_sha256: dict[str, str]
    conversion: str
    shards: dict[str, str]
    excluded_source_tensors: list[str]
    exclusion_reason: str
