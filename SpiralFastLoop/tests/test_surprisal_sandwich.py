from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spiralfastloop.extras import surprisal_sandwich as sr
from spiralfastloop.extras.surprisal_sandwich import AntiTopKMiddle, CoherenceTailBoost


def test_anti_topk_middle_rejects_invalid_window() -> None:
    for kwargs in (
        {"start_frac": -0.1, "end_frac": 0.7},
        {"start_frac": 0.7, "end_frac": 0.7},
        {"start_frac": 0.8, "end_frac": 0.7},
        {"start_frac": 0.1, "end_frac": 1.1},
    ):
        with pytest.raises(ValueError):
            AntiTopKMiddle(**kwargs)


def test_anti_topk_middle_caps_topk_to_vocab_width() -> None:
    processor = AntiTopKMiddle(start_frac=0.0, end_frac=1.0, topk=5, alpha=10.0)
    processor.max_steps = 1

    scores = torch.tensor([[1.0, 3.0, 2.0]])
    output = processor(torch.tensor([[0]]), scores.clone())

    assert torch.allclose(output, torch.tensor([[-9.0, -7.0, -8.0]]))


def test_anti_topk_middle_rejects_invalid_topk_or_alpha() -> None:
    with pytest.raises(ValueError):
        AntiTopKMiddle(topk=0)
    with pytest.raises(ValueError):
        AntiTopKMiddle(alpha=float("nan"))


def test_coherence_tail_boost_rejects_invalid_fraction_or_mu() -> None:
    with pytest.raises(ValueError):
        CoherenceTailBoost(start_frac=1.1)
    with pytest.raises(ValueError):
        CoherenceTailBoost(mu=float("inf"))


def test_coherence_tail_boost_skips_mismatched_vocab_logits() -> None:
    class Tokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"a": 0, "b": 1}

    class TinyModel:
        device = torch.device("cpu")

        def __call__(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
            logits = torch.ones((input_ids.shape[0], 1, 3))
            return SimpleNamespace(logits=logits, past_key_values="past")

    processor = CoherenceTailBoost(
        start_frac=0.0,
        mu=0.5,
        tiny_model=TinyModel(),
        primary_tokenizer=Tokenizer(),
        tiny_tokenizer=Tokenizer(),
    )
    processor.max_steps = 1
    scores = torch.zeros((1, 5))

    output = processor(torch.tensor([[0]]), scores.clone())

    assert torch.equal(output, scores)


def test_surprise_repair_generate_requires_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "AutoTokenizer", None)
    monkeypatch.setattr(sr, "AutoModelForCausalLM", None)
    monkeypatch.setattr(sr, "LogitsProcessorList", None)

    with pytest.raises(ImportError, match="requires transformers"):
        sr.surprise_repair_generate("hello", "missing-model")
