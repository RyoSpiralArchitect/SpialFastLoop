# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

"""Surprisal Sandwich generation helpers built on top of ``transformers``."""

from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

__all__ = [
    "AntiTopKMiddle",
    "CoherenceTailBoost",
    "surprise_repair_generate",
]

class AntiTopKMiddle(LogitsProcessor):
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
        assert 0.0 <= start_frac < end_frac <= 1.0
        self.sf: float = start_frac
        self.ef: float = end_frac
        self.topk: int = topk
        self.alpha: float = alpha
        self.step: int = 0
        self.max_steps: Optional[int] = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.max_steps is None:
            return scores
        pos = self.step
        self.step += 1
        if self.sf * self.max_steps <= pos < self.ef * self.max_steps:
            vals, idx = torch.topk(scores, self.topk, dim=-1)
            scores.scatter_(dim=-1, index=idx, src=vals - self.alpha)
        return scores

class CoherenceTailBoost(LogitsProcessor):
    """
    In the tail region, lightly boost coherence via a tiny LM's logits (optional).
    """
    def __init__(
        self,
        start_frac: float = 0.7,
        mu: float = 0.4,
        tiny_model: Optional[AutoModelForCausalLM] = None,
        primary_tokenizer: Optional[AutoTokenizer] = None,
        tiny_tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.sf: float = start_frac
        self.mu: float = mu
        self.tiny: Optional[AutoModelForCausalLM] = tiny_model
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
        if self.max_steps is None or self.tiny is None:
            return scores
        pos = self.step
        self.step += 1
        if pos >= self.sf * self.max_steps:
            if self._tokenizers_compatible:
                tiny_input_ids = input_ids.to(self.tiny.device)
                past = self.past
            else:
                if self.primary_tokenizer is None or self.tiny_tokenizer is None:
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
                tiny_input_ids = tiny_encoding["input_ids"].to(self.tiny.device)
                past = None

            out = self.tiny(
                tiny_input_ids,
                use_cache=True,
                past_key_values=past,
            )
            self.past = out.past_key_values if self._tokenizers_compatible else None
            tiny_logits = out.logits[:, -1, :]
            scores = scores + self.mu * tiny_logits
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
    tok = AutoTokenizer.from_pretrained(main_name)
    main = AutoModelForCausalLM.from_pretrained(main_name, device_map="auto").eval()

    tiny: Optional[AutoModelForCausalLM] = None
    tiny_tok: Optional[AutoTokenizer] = None
    if tiny_name is not None:
        tiny_tok = AutoTokenizer.from_pretrained(tiny_name)
        tiny = AutoModelForCausalLM.from_pretrained(tiny_name, device_map="auto").eval()

    anti = AntiTopKMiddle(start_frac=middle[0], end_frac=middle[1], topk=topk, alpha=alpha)
    coh = CoherenceTailBoost(
        start_frac=middle[1],
        mu=mu,
        tiny_model=tiny,
        primary_tokenizer=tok,
        tiny_tokenizer=tiny_tok,
    )

    processors = LogitsProcessorList([anti, coh])
    ids = tok(prompt, return_tensors="pt").to(main.device)

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
