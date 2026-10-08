"""Offline CPU tokenizer/template and CT preprocessing for the native runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mlx_reason_ct.ct_crop import CTCrop, load_anatomy_ct
from mlx_reason_ct.errors import InvalidPromptError


@dataclass(frozen=True)
class VolumePrompt:
    text: str
    input_ids: np.ndarray
    position_ids: np.ndarray
    attention_mask: np.ndarray
    mm_token_type_ids: np.ndarray
    rope_delta: int
    image_start: int


def volume_positions(
    ids: np.ndarray, image_token_id: int, grid: tuple[int, int, int]
) -> tuple[np.ndarray, int, int]:
    """Pinned unmerged 3D MRoPE indices for the public single-volume request."""
    indices = np.flatnonzero(ids == image_token_id)
    count = int(np.prod(grid))
    if len(indices) != count or not np.array_equal(
        indices, np.arange(int(indices[0]), int(indices[0]) + count)
    ):
        raise InvalidPromptError("The prompt must contain exactly one CT image marker")
    start = int(indices[0])
    end = start + count
    before = np.broadcast_to(np.arange(start), (3, start))
    visual = np.indices(grid).reshape(3, -1) + start
    after = np.broadcast_to(np.arange(len(ids) - end) + start + max(grid[1:]), (3, len(ids) - end))
    positions = np.concatenate([before, visual, after], axis=1).astype(np.int64)
    return positions[:, None], int(positions.max() + 1 - len(ids)), start


class Processor:
    """Owns only CPU text and non-neural image preparation; no Transformers import."""

    def __init__(self, model_dir: Path) -> None:
        tokenizers: Any = import_module("tokenizers")
        self.tokenizer: Any = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.image_token = "<|image_pad|>"
        self.image_id: int = int(self.tokenizer.token_to_id(self.image_token))
        config = json.loads((model_dir / "image_processor_3d/preprocessor_config.json").read_text())
        self.grid: tuple[int, int, int] = tuple(config["final_grid_size"])
        self.spacing: tuple[float, float, float] = tuple(config["pixdim"])
        self.shape: tuple[int, int, int] = tuple(config["spatial_size"])
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

        def fail(message: str) -> None:
            raise InvalidPromptError(message)

        cast(dict[str, Any], environment.globals)["raise_exception"] = fail
        self.template: Template = environment.from_string(
            (model_dir / "chat_template.jinja").read_text()
        )

    def prompt(self, question: str, enable_thinking: bool) -> VolumePrompt:
        text = self.template.render(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if text.count(self.image_token) != 1:
            raise InvalidPromptError("The prompt must contain exactly one CT image marker")
        expanded = text.replace(self.image_token, self.image_token * int(np.prod(self.grid)))
        ids = np.asarray(
            self.tokenizer.encode(expanded, add_special_tokens=False).ids, dtype=np.int64
        )
        positions, delta, start = volume_positions(ids, self.image_id, self.grid)
        return VolumePrompt(
            text,
            ids[None],
            positions,
            np.ones((1, len(ids)), dtype=np.int64),
            (ids == self.image_id).astype(np.int64)[None],
            delta,
            start,
        )

    def image(self, source: Path, region: str) -> tuple[np.ndarray, CTCrop]:
        crop = load_anatomy_ct(source, region, spacing=self.spacing, roi=self.shape)
        pixels = np.clip(crop.values, -1000.0, 1000.0) / np.float32(1000.0)
        return np.ascontiguousarray(pixels.transpose(2, 1, 0)[None, None]), crop

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
