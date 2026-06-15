# SPDX-License-Identifier: Apache-2.0
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

from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple

import torch

from ..engine import TriggerResult
from ..metrics import GLOBAL_NORMALIZATION_METRICS, NormalizationMetricsCollector
from ..utils import (
    _device_setting,
    _non_negative_finite_float_setting,
    _non_negative_int_setting,
    _optional_positive_int_setting,
    _positive_int_setting,
)

# Exposed tolerances so downstream callers (or tests) can tune them if their
# loss scales differ drastically from the default cross-entropy-ish regime.
FRACTION_NORMALIZATION_EPS = 1e-12
COEFVAR_STABILIZER = 1e-8

__all__ = [
    "HardSampleBuffer",
    "HardSampleProvider",
    "LossStdConfig",
    "LossStdTrigger",
    "FRACTION_NORMALIZATION_EPS",
    "COEFVAR_STABILIZER",
]


def _detach_to_cpu(batch: Any) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu()
    if isinstance(batch, dict):
        return {k: _detach_to_cpu(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        converted = [_detach_to_cpu(v) for v in batch]
        if isinstance(batch, tuple):
            if hasattr(batch, "_fields"):
                return type(batch)(*converted)
            return tuple(converted)
        return converted
    return batch


def _select_indices(batch: Any, indices: Sequence[int]) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.index_select(0, torch.tensor(indices, device=batch.device))
    if isinstance(batch, dict):
        if not batch and len(indices) > 0:
            raise ValueError("batch mapping must not be empty when selecting samples.")
        return {k: _select_indices(v, indices) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            if len(indices) > 0:
                raise ValueError("batch sequence must not be empty when selecting samples.")
            return batch
        if isinstance(batch[0], torch.Tensor):
            selected = [_select_indices(v, indices) for v in batch]
            if isinstance(batch, tuple):
                if hasattr(batch, "_fields"):
                    return type(batch)(*selected)
                return tuple(selected)
            return selected
        selected = [batch[i] for i in indices]
        if isinstance(batch, tuple):
            if hasattr(batch, "_fields"):
                return type(batch)(*selected)
            return tuple(selected)
        return selected
    raise TypeError("Unsupported batch structure for hard-sample selection.")


def _split_batch(batch: Any, batch_size: int) -> list[Any]:
    if isinstance(batch, torch.Tensor):
        return [batch[i] for i in range(batch_size)]
    if isinstance(batch, dict):
        if not batch and batch_size > 0:
            raise ValueError("batch mapping must not be empty when splitting samples.")
        per_key = {k: _split_batch(v, batch_size) for k, v in batch.items()}
        return [{k: per_key[k][i] for k in per_key} for i in range(batch_size)]
    if isinstance(batch, (list, tuple)):
        if len(batch) == batch_size and not (batch and isinstance(batch[0], torch.Tensor)):
            return list(batch)
        per_item = [_split_batch(item, batch_size) for item in batch]
        samples: list[Any] = []
        for i in range(batch_size):
            assembled = [per_item[j][i] for j in range(len(per_item))]
            if isinstance(batch, tuple):
                if hasattr(batch, "_fields"):
                    samples.append(type(batch)(*assembled))
                else:
                    samples.append(tuple(assembled))
            else:
                samples.append(assembled)
        return samples
    return [batch for _ in range(batch_size)]


def _trigger_context_setting(ctx: Any) -> Dict[str, Any]:
    if not isinstance(ctx, Mapping):
        raise ValueError("ctx must be a mapping")
    return dict(ctx)


def _callable_setting(value: Any, name: str) -> Any:
    if not callable(value):
        raise ValueError(f"{name} must be callable")
    return value


def _finite_loss_vector_setting(loss_vec: Any) -> torch.Tensor:
    if not isinstance(loss_vec, torch.Tensor):
        raise TypeError("loss_vec must be a torch.Tensor")
    if not torch.is_floating_point(loss_vec):
        raise ValueError("loss_vec must be a floating-point tensor")
    if not torch.isfinite(loss_vec).all().item():
        raise ValueError("loss_vec must contain only finite values")
    return loss_vec


class HardSampleBuffer:
    """Ring buffer of hard samples to support trigger-based injections."""

    def __init__(self, *, max_samples: int = 2048) -> None:
        self.max_samples = _non_negative_int_setting(max_samples, "max_samples")
        self._inputs: Deque[Any] = deque(maxlen=self.max_samples)
        self._targets: Deque[Any] = deque(maxlen=self.max_samples)

    def __len__(self) -> int:
        return len(self._inputs)

    def add_batch(
        self,
        inputs: Any,
        targets: Any,
        loss_vec: torch.Tensor,
        *,
        top_k: Optional[int] = None,
    ) -> None:
        if self.max_samples <= 0:
            return
        loss_vec = _finite_loss_vector_setting(loss_vec)
        if loss_vec.ndim != 1:
            raise ValueError("loss_vec must be a 1D tensor for hard-sample selection.")
        batch_size = loss_vec.shape[0]
        if batch_size == 0:
            return
        requested_top_k = _optional_positive_int_setting(top_k, "top_k")
        k = batch_size if requested_top_k is None else min(batch_size, requested_top_k)
        _, indices = torch.topk(loss_vec, k=k, largest=True)
        try:
            selected_indices = indices.tolist()
            selected_inputs = _select_indices(inputs, selected_indices)
            selected_targets = _select_indices(targets, selected_indices)
            cpu_inputs = _detach_to_cpu(selected_inputs)
            cpu_targets = _detach_to_cpu(selected_targets)
            input_samples = _split_batch(cpu_inputs, k)
            target_samples = _split_batch(cpu_targets, k)
        except Exception as exc:
            raise ValueError("inputs and targets must match loss_vec batch dimension") from exc
        if len(input_samples) != k or len(target_samples) != k:
            raise ValueError("inputs and targets must match loss_vec batch dimension")
        for item_in, item_tgt in zip(input_samples, target_samples):
            self._inputs.append(item_in)
            self._targets.append(item_tgt)

    def sample(self, num_samples: int) -> Tuple[Any, Any]:
        if len(self._inputs) == 0:
            raise ValueError("HardSampleBuffer is empty; cannot sample.")
        requested = _positive_int_setting(num_samples, "num_samples")
        indices = torch.randint(0, len(self._inputs), (requested,))
        samples_in = [self._inputs[i] for i in indices.tolist()]
        samples_tgt = [self._targets[i] for i in indices.tolist()]
        if isinstance(samples_in[0], torch.Tensor):
            return torch.stack(samples_in, dim=0), torch.stack(samples_tgt, dim=0)
        return samples_in, samples_tgt


class HardSampleProvider:
    """Provider that pulls from a HardSampleBuffer and optionally augments."""

    def __init__(
        self,
        buffer: HardSampleBuffer,
        *,
        augmenter: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        fallback: Optional[Callable[[int, str, Dict[str, Any]], Tuple[Any, Any]]] = None,
        select_top_k: Optional[int] = None,
    ) -> None:
        if not isinstance(buffer, HardSampleBuffer):
            raise ValueError("buffer must be a HardSampleBuffer")
        if augmenter is not None:
            _callable_setting(augmenter, "augmenter")
        if fallback is not None:
            _callable_setting(fallback, "fallback")
        self.buffer = buffer
        self.augmenter = augmenter
        self.fallback = fallback
        self.select_top_k = _optional_positive_int_setting(select_top_k, "select_top_k")

    def observe(self, ctx: Dict[str, Any]) -> None:
        ctx_value = _trigger_context_setting(ctx)
        loss_vec = ctx_value.get("loss_vec")
        if not isinstance(loss_vec, torch.Tensor):
            return
        if "inputs" not in ctx_value or "targets" not in ctx_value:
            raise ValueError("ctx must include inputs and targets when loss_vec is a tensor")
        self.buffer.add_batch(
            ctx_value["inputs"],
            ctx_value["targets"],
            _finite_loss_vector_setting(loss_vec),
            top_k=self.select_top_k,
        )

    def __call__(self, requested: int, device: str, ctx: Dict[str, Any]) -> Tuple[Any, Any]:
        requested_value = _positive_int_setting(requested, "requested")
        device_value = _device_setting(device)
        ctx_value = _trigger_context_setting(ctx)
        try:
            inputs, targets = self.buffer.sample(requested_value)
        except ValueError:
            if self.fallback is None:
                raise
            inputs, targets = self.fallback(requested_value, device_value, ctx_value)
        if self.augmenter is not None:
            inputs, targets = self.augmenter(inputs, targets)
        return inputs, targets


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
        self.std_threshold = _non_negative_finite_float_setting(
            self.std_threshold,
            "std_threshold",
        )
        self.inject_ratio = _non_negative_finite_float_setting(
            self.inject_ratio,
            "inject_ratio",
        )
        self.weight_alpha = _non_negative_finite_float_setting(
            self.weight_alpha,
            "weight_alpha",
        )
        self.budget_frac = _non_negative_finite_float_setting(
            self.budget_frac,
            "budget_frac",
        )
        self.pulse_every = _non_negative_int_setting(self.pulse_every, "pulse_every")
        self.max_injected_per_step = _non_negative_int_setting(
            self.max_injected_per_step,
            "max_injected_per_step",
        )


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
        _callable_setting(provider, "provider")
        if cfg is not None and not isinstance(cfg, LossStdConfig):
            raise ValueError("cfg must be a LossStdConfig or None")
        if normalization_metrics is not None and not isinstance(
            normalization_metrics,
            NormalizationMetricsCollector,
        ):
            raise ValueError(
                "normalization_metrics must be a NormalizationMetricsCollector or None"
            )
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

    def observe(self, ctx: Dict[str, Any]) -> None:
        provider = self.provider
        observe = getattr(provider, "observe", None)
        if observe is not None:
            _callable_setting(observe, "provider.observe")
            observe(_trigger_context_setting(ctx))

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
        ctx_value = _trigger_context_setting(ctx)
        if "loss_vec" not in ctx_value:
            raise ValueError("ctx must include loss_vec")
        if "device" not in ctx_value:
            raise ValueError("ctx must include device")

        loss_tensor = ctx_value["loss_vec"]
        if not isinstance(loss_tensor, torch.Tensor):
            raise TypeError("Loss vector in trigger context must be a torch.Tensor.")
        loss_vec = _finite_loss_vector_setting(loss_tensor.detach())
        if loss_vec.numel() == 0:
            return None

        device = _device_setting(ctx_value["device"])
        raw_step = ctx_value.get("step")
        step = _non_negative_int_setting(raw_step, "step") if raw_step is not None else 0
        has_step = raw_step is not None
        spent = self.spent
        total = self.total
        budget_buffer = self._budget_buffer
        last_pulse_step = self._last_pulse_step
        if has_step:
            if self._last_step is not None and step < self._last_step:
                spent = 0
                total = 0
                last_pulse_step = None
                budget_buffer = 0.0

        batch = loss_vec.numel()
        next_total = total + batch

        coefvar = loss_vec.std(unbiased=False) / (
            loss_vec.mean().abs() + COEFVAR_STABILIZER
        )
        pulse_due = (
            self.cfg.pulse_every > 0 and step > 0 and step % self.cfg.pulse_every == 0
        )
        force_pulse = pulse_due and step != last_pulse_step
        need = coefvar.item() <= self.cfg.std_threshold or force_pulse

        def commit_state(
            *,
            spent_value: int = spent,
            total_value: int = next_total,
            budget_buffer_value: float = budget_buffer,
            pulse_step_value: Optional[int] = last_pulse_step,
        ) -> None:
            self.spent = spent_value
            self.total = total_value
            self._budget_buffer = budget_buffer_value
            self._last_pulse_step = pulse_step_value
            if has_step:
                self._last_step = step

        budget_ok = spent <= self.cfg.budget_frac * max(1, next_total)
        if not (need and budget_ok):
            if force_pulse and has_step:
                last_pulse_step = step
            commit_state(pulse_step_value=last_pulse_step)
            return None

        requested = min(int(batch * self.cfg.inject_ratio), self.cfg.max_injected_per_step)
        if requested <= 0:
            if force_pulse and has_step:
                last_pulse_step = step
            commit_state(pulse_step_value=last_pulse_step)
            return None

        budget_limit = self.cfg.budget_frac * max(1, next_total)
        remaining_budget = budget_limit - spent
        available_budget = max(0.0, remaining_budget + budget_buffer)
        if available_budget <= 0.0:
            if force_pulse and has_step:
                last_pulse_step = step
            commit_state(budget_buffer_value=0.0, pulse_step_value=last_pulse_step)
            return None

        allowed_whole = int(available_budget)
        if allowed_whole <= 0:
            fractional_credit = self._drop_rounding_noise(
                max(0.0, available_budget - allowed_whole), context="fractional_credit"
            )
            if force_pulse and has_step:
                last_pulse_step = step
            commit_state(
                budget_buffer_value=fractional_credit,
                pulse_step_value=last_pulse_step,
            )
            return None
        requested = min(requested, allowed_whole)
        if requested <= 0:
            if force_pulse and has_step:
                last_pulse_step = step
            commit_state(pulse_step_value=last_pulse_step)
            return None

        extra_x, extra_y = self.provider(requested, device, ctx_value)
        leftover_available = max(0.0, available_budget - requested)
        remaining_budget_after = max(0.0, remaining_budget - requested)
        carryover_credit = self._drop_rounding_noise(
            max(0.0, leftover_available - remaining_budget_after), context="carryover_credit"
        )
        if force_pulse:
            last_pulse_step = step
        commit_state(
            spent_value=spent + requested,
            budget_buffer_value=carryover_credit,
            pulse_step_value=last_pulse_step,
        )

        # weights: original ones at 1.0, injected at alpha
        weights = torch.ones(batch + requested, device=loss_vec.device)
        weights[-requested:] = self.cfg.weight_alpha
        return TriggerResult(extra_inputs=extra_x, extra_targets=extra_y, weights=weights)
