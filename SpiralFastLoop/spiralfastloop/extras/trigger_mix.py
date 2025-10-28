# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

"""Trigger utilities for dynamically mixing harder samples into training.

The helpers in this module work with per-sample losses whose magnitudes are
typically ``O(1)`` but may accumulate fractional budget credits over many calls.
Two tiny epsilons are exposed to make the behaviour easy to reason about and
retune:

``FRACTION_NORMALIZATION_EPS``
    Drops rounding residue when tracking fractional budgets so we do not leak
    microscopic negative numbers back into subsequent calls.

``COEFVAR_STABILIZER``
    Prevents division-by-zero when the mean per-sample loss is extremely close
    to zero during coefficient-of-variation checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..engine import TriggerResult
from ..metrics import GLOBAL_NORMALIZATION_METRICS, NormalizationMetricsCollector

# Exposed tolerances so downstream callers (or tests) can tune them if their
# loss scales differ drastically from the default cross-entropy-ish regime.
FRACTION_NORMALIZATION_EPS = 1e-12
COEFVAR_STABILIZER = 1e-8

__all__ = [
    "LossStdConfig",
    "LossStdTrigger",
    "FRACTION_NORMALIZATION_EPS",
    "COEFVAR_STABILIZER",
]


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
        normalization_metrics: Optional[NormalizationMetricsCollector] = None,
    ) -> None:
        self.provider = provider
        self.cfg = cfg or LossStdConfig()
        self.spent: int = 0  # approximate budget spent (injected samples)
        # Count of baseline samples the trigger has observed (without injections).
        self.total: int = 0
        self._last_step: Optional[int] = None
        self._last_pulse_step: Optional[int] = None
        # Accumulate fractional budget so tiny allowances eventually release
        # whole extra samples instead of being lost to flooring.
        self._budget_buffer: float = 0.0
        self._norm_metrics = normalization_metrics or GLOBAL_NORMALIZATION_METRICS

    def _drop_rounding_noise(self, value: float, *, context: str = "budget_buffer") -> float:
        """Elide microscopic float residue that should count as zero.

        The fractional budget buffer is dimensionless (counts of samples) so
        residue below :data:`FRACTION_NORMALIZATION_EPS` is too small to be
        meaningful.  The epsilon is intentionally module-level so tests and
        downstream users can retune it for different numerical regimes.
        """

        normalized = 0.0 if abs(value) < FRACTION_NORMALIZATION_EPS else value
        if self._norm_metrics is not None:
            self._norm_metrics.record(value, normalized, context=context)
        return normalized

    def _reset_budget_counters(self) -> None:
        """Reset running totals when a new epoch begins."""
        self.spent = 0
        self.total = 0
        self._last_pulse_step = None
        self._budget_buffer = 0.0

    def __call__(self, ctx: Dict[str, Any]) -> Optional[TriggerResult]:
        loss_tensor = ctx["loss_vec"]
        if not isinstance(loss_tensor, torch.Tensor):
            raise TypeError("Loss vector in trigger context must be a torch.Tensor.")
        loss_vec = loss_tensor.detach()
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

        coefvar = loss_vec.std(unbiased=False) / (
            loss_vec.mean().abs() + COEFVAR_STABILIZER
        )
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
        fractional_credit = self._drop_rounding_noise(
            max(0.0, available_budget - allowed_whole), context="fractional_credit"
        )
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
        leftover_available = max(0.0, available_budget - requested)
        remaining_budget_after = max(0.0, remaining_budget - requested)
        carryover_credit = self._drop_rounding_noise(
            max(0.0, leftover_available - remaining_budget_after), context="carryover_credit"
        )
        self._budget_buffer = carryover_credit
        if force_pulse:
            self._last_pulse_step = step

        # weights: original ones at 1.0, injected at alpha
        weights = torch.ones(batch + requested, device=loss_vec.device)
        weights[-requested:] = self.cfg.weight_alpha
        return TriggerResult(extra_inputs=extra_x, extra_targets=extra_y, weights=weights)
