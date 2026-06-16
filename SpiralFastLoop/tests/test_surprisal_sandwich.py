from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spiralfastloop.extras import surprisal_sandwich as sr
from spiralfastloop.extras.surprisal_sandwich import AntiTopKMiddle, CoherenceTailBoost


class _FailingIterable:
    def __iter__(self):
        yield 0.1
        raise RuntimeError("iteration failed")


def test_anti_topk_middle_rejects_invalid_window() -> None:
    for kwargs in (
        {"start_frac": -0.1, "end_frac": 0.7},
        {"start_frac": True, "end_frac": 0.7},
        {"start_frac": "0.1", "end_frac": 0.7},
        {"start_frac": b"0.1", "end_frac": 0.7},
        {"start_frac": 0.7, "end_frac": 0.7},
        {"start_frac": 0.8, "end_frac": 0.7},
        {"start_frac": 0.1, "end_frac": True},
        {"start_frac": 0.1, "end_frac": "0.7"},
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


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), True, "10.0", b"10.0", object()])
def test_anti_topk_middle_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        AntiTopKMiddle(alpha=alpha)  # type: ignore[arg-type]


def test_anti_topk_middle_rejects_invalid_call_tensors_without_advancing() -> None:
    processor = AntiTopKMiddle(start_frac=0.0, end_frac=1.0, topk=2)
    processor.max_steps = 1
    cases = [
        (object(), torch.zeros((1, 3)), TypeError, "input_ids"),
        (torch.tensor([0]), torch.zeros((1, 3)), ValueError, "input_ids"),
        (torch.tensor([[0]]), object(), TypeError, "scores"),
        (torch.tensor([[0]]), torch.zeros(3), ValueError, "scores"),
        (torch.tensor([[0], [1]]), torch.zeros((1, 3)), ValueError, "batch"),
        (torch.tensor([[0]]), torch.ones((1, 3), dtype=torch.int64), ValueError, "scores"),
    ]

    for input_ids, scores, error_type, match in cases:
        with pytest.raises(error_type, match=match):
            processor(input_ids, scores)  # type: ignore[arg-type]
        assert processor.step == 0


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, "2", True])
def test_anti_topk_middle_rejects_invalid_max_steps_without_advancing(max_steps: object) -> None:
    processor = AntiTopKMiddle(start_frac=0.0, end_frac=1.0, topk=2)
    processor.max_steps = max_steps  # type: ignore[assignment]

    with pytest.raises(ValueError, match="max_steps"):
        processor(torch.tensor([[0]]), torch.zeros((1, 3)))
    assert processor.step == 0


def test_coherence_tail_boost_rejects_invalid_fraction_or_mu() -> None:
    with pytest.raises(ValueError):
        CoherenceTailBoost(start_frac=1.1)
    with pytest.raises(ValueError, match="start_frac"):
        CoherenceTailBoost(start_frac=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start_frac"):
        CoherenceTailBoost(start_frac="0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CoherenceTailBoost(mu=float("inf"))
    with pytest.raises(ValueError, match="mu"):
        CoherenceTailBoost(mu=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mu"):
        CoherenceTailBoost(mu="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mu"):
        CoherenceTailBoost(mu=object())  # type: ignore[arg-type]


def test_surprise_repair_generate_rejects_malformed_middle_before_loading_models() -> None:
    with pytest.raises(ValueError, match="middle"):
        sr.surprise_repair_generate(
            "hello",
            "main-model",
            middle=_FailingIterable(),  # type: ignore[arg-type]
        )


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


def test_coherence_tail_boost_rejects_invalid_call_tensors_without_advancing() -> None:
    class TinyModel:
        device = torch.device("cpu")

        def __call__(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
            logits = torch.ones((input_ids.shape[0], 1, 3))
            return SimpleNamespace(logits=logits, past_key_values="past")

    processor = CoherenceTailBoost(start_frac=0.0, tiny_model=TinyModel())
    processor.max_steps = 1

    with pytest.raises(TypeError, match="input_ids"):
        processor(object(), torch.zeros((1, 3)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input_ids"):
        processor(torch.tensor([0]), torch.zeros((1, 3)))
    with pytest.raises(TypeError, match="scores"):
        processor(torch.tensor([[0]]), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scores"):
        processor(torch.tensor([[0]]), torch.zeros(3))
    with pytest.raises(ValueError, match="batch"):
        processor(torch.tensor([[0], [1]]), torch.zeros((1, 3)))
    with pytest.raises(ValueError, match="scores"):
        processor(torch.tensor([[0]]), torch.ones((1, 3), dtype=torch.int64))

    assert processor.step == 0
    assert processor.past is None


def test_coherence_tail_boost_rejects_invalid_max_steps_without_advancing() -> None:
    processor = CoherenceTailBoost(start_frac=0.0, tiny_model=object())
    processor.max_steps = "2"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="max_steps"):
        processor(torch.tensor([[0]]), torch.zeros((1, 3)))
    assert processor.step == 0


def test_coherence_tail_boost_rejects_invalid_tiny_tokenizer_outputs_without_past() -> None:
    class PrimaryTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"a": 0}

        def decode(self, *_: object, **__: object) -> str:
            return "a"

    class TinyTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"b": 0}

        def __call__(self, *_: object, **__: object) -> dict[str, object]:
            return {}

    class TinyModel:
        device = torch.device("cpu")

    processor = CoherenceTailBoost(
        start_frac=0.0,
        tiny_model=TinyModel(),
        primary_tokenizer=PrimaryTokenizer(),
        tiny_tokenizer=TinyTokenizer(),
    )
    processor.max_steps = 1

    with pytest.raises(ValueError, match="tiny_tokenizer"):
        processor(torch.tensor([[0]]), torch.zeros((1, 3)))
    assert processor.step == 0
    assert processor.past is None


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
        "ab",
        b"ab",
        {0.1: 0.2, 0.3: 0.4},
        (0.1,),
        (0.1, 0.2, 0.3),
        ("0.1", 0.7),
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


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"prompt": object()}, "prompt"),
        ({"main_name": ""}, "main_name"),
        ({"main_name": "   "}, "main_name"),
        ({"main_name": object()}, "main_name"),
        ({"tiny_name": ""}, "tiny_name"),
        ({"tiny_name": object()}, "tiny_name"),
        ({"alpha": "10.0"}, "alpha"),
        ({"alpha": b"10.0"}, "alpha"),
        ({"topk": 0}, "topk"),
        ({"mu": "0.4"}, "mu"),
    ],
)
def test_surprise_repair_generate_rejects_invalid_settings_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    field: str,
) -> None:
    _pretend_transformers_are_available(monkeypatch)
    call_kwargs: dict[str, object] = {
        "prompt": "hello",
        "main_name": "unused-model",
    }
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=field):
        sr.surprise_repair_generate(**call_kwargs)  # type: ignore[arg-type]
