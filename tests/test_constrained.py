"""Constrained selection preserves special tokens and excludes illegal high logits."""

from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models

from mlx_reason_ct._host_types import TokenizerContract
from mlx_reason_ct.constrained import ChoiceConstraint
from mlx_reason_ct.errors import InvalidInputError


def test_greedy_choice_uses_legal_logits_and_requires_completion(tmp_path: Path) -> None:
    pytest.importorskip("llguidance")
    vocabulary = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(dict.fromkeys("alphabetz"))}}
    raw = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="[UNK]"))
    raw.decoder = decoders.ByteLevel()
    tokenizer = cast(TokenizerContract, raw)
    tokenizer.add_special_tokens(["[EOS]", "<tool_call>"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    eos = tokenizer.token_to_id("[EOS]")
    assert eos is not None
    constraint = ChoiceConstraint(path, ["<tool_call>alpha", "<tool_call>beta"])
    logits = np.zeros(len(vocabulary) + 2 + 8, dtype=np.float32)
    logits[vocabulary["z"]] = 1000  # The unconstrained maximum is never legal here.
    logits[eos] = 900  # EOS cannot win before the selected choice is complete.
    logits[vocabulary["b"]] = 10
    logits[vocabulary["a"]] = 1
    output: list[int] = []
    for _ in range(20):
        token = constraint.sample(mx.array(logits), (eos,))
        if token == eos:
            break
        output.append(token)
    else:
        pytest.fail("Constrained response did not terminate")
    assert tokenizer.decode(output, skip_special_tokens=False) == "<tool_call>beta"
    assert vocabulary["z"] not in output
    with pytest.raises(InvalidInputError, match="EOS token"):
        ChoiceConstraint(path, ["[EOS]beta"]).sample(mx.array(logits), (eos,))
