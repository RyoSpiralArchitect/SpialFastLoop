# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

"""Surprisal Sandwich generation helpers built on top of ``transformers``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import torch

from ..utils import _positive_int_setting, _strict_finite_float_setting

if TYPE_CHECKING:
    class _LogitsProcessorBase:
        pass
else:
    try:
        from transformers import LogitsProcessor as _LogitsProcessorBase
    except ImportError:  # pragma: no cover - exercised when optional extras are absent
        class _LogitsProcessorBase:
            """Fallback base class so lightweight processors remain importable."""

            pass

AutoModelForCausalLM: Any
AutoTokenizer: Any
LogitsProcessorList: Any
try:
    from transformers import (
        AutoModelForCausalLM as _AutoModelForCausalLM,
        AutoTokenizer as _AutoTokenizer,
        LogitsProcessorList as _LogitsProcessorList,
    )
except ImportError:  # pragma: no cover - exercised when optional extras are absent
    AutoModelForCausalLM = None
    AutoTokenizer = None
    LogitsProcessorList = None
else:
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    LogitsProcessorList = _LogitsProcessorList

__all__ = [
    "AntiTopKMiddle",
    "CoherenceTailBoost",
    "surprise_repair_generate",
]


def _string_setting(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized if not allow_empty else value


def _optional_model_name_setting(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _string_setting(value, name)


def _validate_fraction(name: str, value: float) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"{name} must be a finite value in [0, 1].")
    value = _strict_finite_float_setting(value, name)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1].")
    return value


def _validate_finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"{name} must be finite.")
    normalized = _strict_finite_float_setting(value, name)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _validate_window(start_frac: float, end_frac: float) -> tuple[float, float]:
    start = _validate_fraction("start_frac", start_frac)
    end = _validate_fraction("end_frac", end_frac)
    if start >= end:
        raise ValueError("start_frac must be less than end_frac.")
    return start, end


def _validate_middle(middle: Any) -> tuple[float, float]:
    if isinstance(middle, (str, bytes, bytearray, Mapping)):
        raise ValueError("middle must contain exactly two fractions.")
    try:
        values = tuple(middle)
    except TypeError as exc:
        raise ValueError("middle must contain exactly two fractions.") from exc
    if len(values) != 2:
        raise ValueError("middle must contain exactly two fractions.")
    start_frac, end_frac = values
    return _validate_window(start_frac, end_frac)


def _max_steps_setting(value: Any) -> Optional[int]:
    if value is None:
        return None
    return _positive_int_setting(value, "max_steps")


def _processor_tensor_inputs(
    input_ids: Any,
    scores: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor.")
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor.")
    if input_ids.ndim < 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, vocab].")
    if scores.shape[-1] <= 0:
        raise ValueError("scores must have a non-empty vocabulary dimension.")
    if input_ids.shape[0] != scores.shape[0]:
        raise ValueError("input_ids and scores batch dimensions must match.")
    if not torch.is_floating_point(scores):
        raise ValueError("scores must be a floating-point tensor.")
    return input_ids, scores


def _require_transformers() -> None:
    if AutoTokenizer is None or AutoModelForCausalLM is None or LogitsProcessorList is None:
        raise ImportError(
            "surprise_repair_generate requires transformers; install with "
            "`pip install spiralfastloop[extras]`."
        )


class AntiTopKMiddle(_LogitsProcessorBase):
    """
    In the middle span of generation, apply a strong penalty to the current top-K tokens
    (i.e., "what would most naturally come next") to inject surprise.
    """
    def __init__(
        self,
        start_frac: float = 0.45,
        end_frac: float = 0.7,
        topk: int = 5,
        alpha: float = 10.0,
    ) -> None:
        start, end = _validate_window(start_frac, end_frac)
        self.sf: float = start
        self.ef: float = end
        self.topk: int = _positive_int_setting(topk, "topk")
        self.alpha: float = _validate_finite_float("alpha", alpha)
        self.step: int = 0
        self.max_steps: Optional[int] = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        max_steps = _max_steps_setting(self.max_steps)
        if max_steps is None:
            return scores
        _, scores = _processor_tensor_inputs(input_ids, scores)
        pos = self.step
        if self.sf * max_steps <= pos < self.ef * max_steps:
            k = min(self.topk, scores.shape[-1])
            if k <= 0:
                return scores
            vals, idx = torch.topk(scores, k, dim=-1)
            scores.scatter_(dim=-1, index=idx, src=vals - self.alpha)
        self.step = pos + 1
        return scores

class CoherenceTailBoost(_LogitsProcessorBase):
    """
    In the tail region, lightly boost coherence via a tiny LM's logits (optional).
    """
    def __init__(
        self,
        start_frac: float = 0.7,
        mu: float = 0.4,
        tiny_model: Optional[Any] = None,
        primary_tokenizer: Optional[Any] = None,
        tiny_tokenizer: Optional[Any] = None,
    ) -> None:
        self.sf: float = _validate_fraction("start_frac", start_frac)
        self.mu: float = _validate_finite_float("mu", mu)
        self.tiny: Optional[Any] = tiny_model
        self.step: int = 0
        self.max_steps: Optional[int] = None
        self.past: Any = None
        self.primary_tokenizer = primary_tokenizer
        self.tiny_tokenizer = tiny_tokenizer
        self._tokenizers_compatible: bool = False
        base_tokenizer = primary_tokenizer
        tiny_tok = tiny_tokenizer
        if self.tiny is not None and tiny_tok is not None and base_tokenizer is not None:
            try:
                self._tokenizers_compatible = (
                    base_tokenizer.get_vocab() == tiny_tok.get_vocab()
                )
            except Exception:
                self._tokenizers_compatible = False

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        max_steps = _max_steps_setting(self.max_steps)
        if max_steps is None or self.tiny is None:
            return scores
        input_ids, scores = _processor_tensor_inputs(input_ids, scores)
        pos = self.step
        if pos >= self.sf * max_steps:
            if self._tokenizers_compatible:
                tiny_input_ids = input_ids.to(self.tiny.device)
                past = self.past
            else:
                if self.primary_tokenizer is None or self.tiny_tokenizer is None:
                    self.step = pos + 1
                    return scores
                text = self.primary_tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                tiny_encoding = self.tiny_tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                if not isinstance(tiny_encoding, Mapping) or "input_ids" not in tiny_encoding:
                    raise ValueError("tiny_tokenizer must return a mapping with input_ids.")
                tiny_input_ids = tiny_encoding["input_ids"].to(self.tiny.device)
                past = None

            out = self.tiny(
                tiny_input_ids,
                use_cache=True,
                past_key_values=past,
            )
            next_past = out.past_key_values if self._tokenizers_compatible else None
            tiny_logits = out.logits[:, -1, :]
            if tiny_logits.shape != scores.shape:
                self.step = pos + 1
                return scores
            scores = scores + self.mu * tiny_logits
            self.past = next_past
        self.step = pos + 1
        return scores

@torch.no_grad()
def surprise_repair_generate(
    prompt: str,
    main_name: str,
    tiny_name: Optional[str] = None,
    max_new_tokens: int = 64,
    middle: Tuple[float, float] = (0.45, 0.7),
    alpha: float = 10.0,
    topk: int = 5,
    mu: float = 0.4,
    **genkw: Any,
) -> str:
    max_new_tokens = _positive_int_setting(max_new_tokens, "max_new_tokens")
    middle_start, middle_end = _validate_middle(middle)
    prompt_value = _string_setting(prompt, "prompt", allow_empty=True)
    main_name_value = _string_setting(main_name, "main_name")
    tiny_name_value = _optional_model_name_setting(tiny_name, "tiny_name")
    alpha_value = _validate_finite_float("alpha", alpha)
    topk_value = _positive_int_setting(topk, "topk")
    mu_value = _validate_finite_float("mu", mu)
    _require_transformers()
    tok: Any = AutoTokenizer.from_pretrained(main_name_value)
    main: Any = AutoModelForCausalLM.from_pretrained(main_name_value, device_map="auto").eval()

    tiny: Optional[Any] = None
    tiny_tok: Optional[Any] = None
    if tiny_name_value is not None:
        tiny_tok = AutoTokenizer.from_pretrained(tiny_name_value)
        tiny = AutoModelForCausalLM.from_pretrained(tiny_name_value, device_map="auto").eval()

    anti = AntiTopKMiddle(
        start_frac=middle_start,
        end_frac=middle_end,
        topk=topk_value,
        alpha=alpha_value,
    )
    coh = CoherenceTailBoost(
        start_frac=middle_end,
        mu=mu_value,
        tiny_model=tiny,
        primary_tokenizer=tok,
        tiny_tokenizer=tiny_tok,
    )

    processors = LogitsProcessorList([anti, coh])
    ids = tok(prompt_value, return_tensors="pt").to(main.device)

    anti.max_steps = max_new_tokens
    coh.max_steps = max_new_tokens

    out = main.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        logits_processor=processors,
        **genkw,
    )
    return cast(str, tok.decode(out[0], skip_special_tokens=True))
