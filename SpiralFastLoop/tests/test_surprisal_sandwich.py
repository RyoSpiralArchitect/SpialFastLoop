from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spiralfastloop.extras import surprisal_sandwich as sr
from spiralfastloop.extras.surprisal_sandwich import AntiTopKMiddle, CoherenceTailBoost


def test_anti_topk_middle_rejects_invalid_window() -> None:
    for kwargs in (
        {"start_frac": -0.1, "end_frac": 0.7},
        {"start_frac": True, "end_frac": 0.7},
        {"start_frac": 0.7, "end_frac": 0.7},
        {"start_frac": 0.8, "end_frac": 0.7},
        {"start_frac": 0.1, "end_frac": True},
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


@pytest.mark.parametrize("topk", [0, -1, 1.5, "2", True])
def test_anti_topk_middle_rejects_invalid_topk(topk: object) -> None:
    with pytest.raises(ValueError, match="topk"):
        AntiTopKMiddle(topk=topk)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), True, object()])
def test_anti_topk_middle_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        AntiTopKMiddle(alpha=alpha)  # type: ignore[arg-type]


def test_coherence_tail_boost_rejects_invalid_fraction_or_mu() -> None:
    with pytest.raises(ValueError):
        CoherenceTailBoost(start_frac=1.1)
    with pytest.raises(ValueError, match="start_frac"):
        CoherenceTailBoost(start_frac=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CoherenceTailBoost(mu=float("inf"))
    with pytest.raises(ValueError, match="mu"):
        CoherenceTailBoost(mu=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mu"):
        CoherenceTailBoost(mu=object())  # type: ignore[arg-type]


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


def _pretend_transformers_are_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "AutoTokenizer", object())
    monkeypatch.setattr(sr, "AutoModelForCausalLM", object())
    monkeypatch.setattr(sr, "LogitsProcessorList", object())


@pytest.mark.parametrize("max_new_tokens", [0, -1, 1.5, "2", True])
def test_surprise_repair_generate_rejects_invalid_max_new_tokens_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    max_new_tokens: object,
) -> None:
    _pretend_transformers_are_available(monkeypatch)

    with pytest.raises(ValueError, match="max_new_tokens"):
        sr.surprise_repair_generate(
            "hello",
            "unused-model",
            max_new_tokens=max_new_tokens,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "middle",
    [
        object(),
        (0.1,),
        (0.1, 0.2, 0.3),
        (True, 0.7),
        (0.1, True),
        (0.7, 0.7),
        (0.8, 0.7),
        (0.1, float("nan")),
    ],
)
def test_surprise_repair_generate_rejects_invalid_middle_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    middle: object,
) -> None:
    _pretend_transformers_are_available(monkeypatch)

    with pytest.raises(ValueError):
        sr.surprise_repair_generate(
            "hello",
            "unused-model",
            middle=middle,  # type: ignore[arg-type]
        )
