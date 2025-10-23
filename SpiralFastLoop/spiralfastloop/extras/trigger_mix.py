# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

"""Trigger utilities for dynamically mixing harder samples into training."""

from dataclasses import dataclass
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
    """Trigger hook for requesting harder samples when the batch looks too easy."""

    def __init__(
        self,
        provider: Callable[[int, str, Dict[str, Any]], Tuple[Any, Any]],
        cfg: Optional[LossStdConfig] = None,
    ) -> None:
        self.provider = provider
        self.cfg = cfg or LossStdConfig()
        self.spent = 0  # approximate budget spent (samples)
        self.total = 0  # approximate total samples seen

    def __call__(self, ctx: Dict[str, Any]) -> Optional[TriggerResult]:
        loss_vec: torch.Tensor = ctx["loss_vec"].detach()
        if loss_vec.numel() == 0:
            return None

        device = ctx["device"]
        step = int(ctx.get("step", 0))

        batch = loss_vec.numel()
        self.total += batch

        coefvar = loss_vec.std(unbiased=False) / (loss_vec.mean().abs() + 1e-8)
        pulse_due = (
            self.cfg.pulse_every > 0 and step > 0 and step % self.cfg.pulse_every == 0
        )
        need = coefvar.item() <= self.cfg.std_threshold or pulse_due

        budget_ok = self.spent <= self.cfg.budget_frac * max(1, self.total)
        if not (need and budget_ok):
            return None

        requested = min(int(batch * self.cfg.inject_ratio), self.cfg.max_injected_per_step)
        if requested <= 0:
            return None

        budget_limit = self.cfg.budget_frac * max(1, self.total)
        remaining_budget = budget_limit - self.spent
        if remaining_budget <= 0:
            return None

        allowed_whole = int(remaining_budget)
        if allowed_whole <= 0:
            return None
        requested = min(requested, allowed_whole)
        if requested <= 0:
            return None

        extra_x, extra_y = self.provider(requested, device, ctx)
        self.spent += requested
        self.total += requested

        # weights: original ones at 1.0, injected at alpha
        weights = torch.ones(batch + requested, device=loss_vec.device)
        weights[-requested:] = self.cfg.weight_alpha
        return TriggerResult(extra_inputs=extra_x, extra_targets=extra_y, weights=weights)
