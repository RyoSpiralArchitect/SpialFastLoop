# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

"""Trigger utilities for dynamically mixing harder samples into training."""

from dataclasses import dataclass
from math import floor
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..engine import TriggerResult

__all__ = ["LossStdConfig", "LossStdTrigger"]


@dataclass
class LossStdConfig:
    """Configuration controlling when and how much to inject extra samples."""

    std_threshold: float = 0.15
    inject_ratio: float = 0.08  # fraction of batch to add at most
    weight_alpha: float = 1.2  # weight for injected samples
    budget_frac: float = 0.03  # token/sample budget per epoch (approx)
    pulse_every: int = 800  # force a pulse every N steps
    max_injected_per_step: int = 128


class LossStdTrigger:
    """Trigger hook for requesting harder samples when the batch looks too easy.

    The trigger maintains a fractional budget and automatically resets its running
    totals whenever the provided step counter decreases (e.g., at the start of a
    new epoch). Fractional budget allowances are accumulated across calls so that
    tiny per-step credits eventually release a whole extra sample. Forced pulses
    fire at most once per unique step value so repeated step callbacks from
    gradient accumulation do not spam extra requests.
    """

    def __init__(
        self,
        provider: Callable[[int, str, Dict[str, Any]], Tuple[Any, Any]],
        cfg: Optional[LossStdConfig] = None,
    ) -> None:
        self.provider = provider
        self.cfg = cfg or LossStdConfig()
        self.spent = 0  # approximate budget spent (injected samples)
        # Count of baseline samples the trigger has observed (without injections).
        self.total = 0
        self._last_step: Optional[int] = None
        self._last_pulse_step: Optional[int] = None
        # Accumulate fractional budget so tiny allowances eventually release
        # whole extra samples instead of being lost to flooring.
        self._budget_buffer: float = 0.0

    def _reset_budget_counters(self) -> None:
        """Reset running totals when a new epoch begins."""
        self.spent = 0
        self.total = 0
        self._last_pulse_step = None
        self._budget_buffer = 0.0

    def __call__(self, ctx: Dict[str, Any]) -> Optional[TriggerResult]:
        loss_vec: torch.Tensor = ctx["loss_vec"].detach()
        if loss_vec.numel() == 0:
            return None

        device = ctx["device"]
        raw_step = ctx.get("step")
        step = int(raw_step) if raw_step is not None else 0
        has_step = raw_step is not None
        if has_step:
            if self._last_step is not None and step < self._last_step:
                self._reset_budget_counters()
            self._last_step = step

        batch = loss_vec.numel()
        self.total += batch

        coefvar = loss_vec.std(unbiased=False) / (loss_vec.mean().abs() + 1e-8)
        pulse_due = (
            self.cfg.pulse_every > 0 and step > 0 and step % self.cfg.pulse_every == 0
        )
        force_pulse = pulse_due and step != self._last_pulse_step
        need = coefvar.item() <= self.cfg.std_threshold or force_pulse

        budget_ok = self.spent <= self.cfg.budget_frac * max(1, self.total)
        if not (need and budget_ok):
            if force_pulse and has_step:
                self._last_pulse_step = step
            return None

        requested = min(int(batch * self.cfg.inject_ratio), self.cfg.max_injected_per_step)
        if requested <= 0:
            if force_pulse and has_step:
                self._last_pulse_step = step
            return None

        budget_limit = self.cfg.budget_frac * max(1, self.total)
        remaining_budget = budget_limit - self.spent
        available_budget = max(0.0, remaining_budget + self._budget_buffer)
        if available_budget <= 0.0:
            self._budget_buffer = 0.0
            if force_pulse and has_step:
                self._last_pulse_step = step
            return None

        allowed_whole = int(available_budget)
        fractional_credit = max(0.0, available_budget - allowed_whole)
        if allowed_whole <= 0:
            self._budget_buffer = fractional_credit
            if force_pulse and has_step:
                self._last_pulse_step = step
            return None
        requested = min(requested, allowed_whole)
        if requested <= 0:
            self._budget_buffer = fractional_credit
            if force_pulse and has_step:
                self._last_pulse_step = step
            return None

        extra_x, extra_y = self.provider(requested, device, ctx)
        self.spent += requested
        leftover = max(0.0, available_budget - requested)
        self._budget_buffer = max(0.0, leftover - floor(leftover))
        if force_pulse:
            self._last_pulse_step = step

        # weights: original ones at 1.0, injected at alpha
        weights = torch.ones(batch + requested, device=loss_vec.device)
        weights[-requested:] = self.cfg.weight_alpha
        return TriggerResult(extra_inputs=extra_x, extra_targets=extra_y, weights=weights)
