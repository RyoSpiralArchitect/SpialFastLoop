# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

"""Trigger utilities for dynamically mixing harder samples into training."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

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

    def __post_init__(self) -> None:
        if self.std_threshold < 0:
            raise ValueError("std_threshold must be non-negative.")
        if self.inject_ratio < 0:
            raise ValueError("inject_ratio must be non-negative.")
        if self.weight_alpha <= 0:
            raise ValueError("weight_alpha must be strictly positive.")
        if self.budget_frac < 0:
            raise ValueError("budget_frac must be non-negative.")
        self.pulse_every = int(self.pulse_every)
        if self.pulse_every < 0:
            raise ValueError("pulse_every must be non-negative.")
        self.max_injected_per_step = int(self.max_injected_per_step)
        if self.max_injected_per_step < 0:
            raise ValueError("max_injected_per_step must be non-negative.")


class LossStdTrigger:
    """Trigger hook for requesting harder samples when the batch looks too easy."""

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

        # weights: original ones at 1.0, injected at alpha
        weights = torch.ones(batch + requested, device=loss_vec.device)
        weights[-requested:] = self.cfg.weight_alpha
        return TriggerResult(extra_inputs=extra_x, extra_targets=extra_y, weights=weights)

    def reset(self) -> None:
        """Reset the running budget counters to start a fresh schedule."""

        self.spent = 0
        self.total = 0

    def state_dict(self) -> Dict[str, int]:
        """Return a serialisable snapshot of the trigger's running state."""

        return {"spent": int(self.spent), "total": int(self.total)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the trigger's state from :meth:`state_dict` output."""

        if not isinstance(state, Mapping):
            raise TypeError("LossStdTrigger.load_state_dict expects a mapping input.")
        try:
            spent = int(state.get("spent", 0))
            total = int(state.get("total", 0))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("LossStdTrigger state must contain integer 'spent' and 'total'.") from exc
        if spent < 0 or total < 0:
            raise ValueError("LossStdTrigger state counters must be non-negative.")
        self.spent = spent
        self.total = total
