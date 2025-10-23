# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

\
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn

from .utils import (
    get_best_device, get_amp_policy, autocast_ctx, to_device,
    ThroughputMeter, maybe_channels_last, safe_compile
)

@dataclass
class TriggerResult:
    extra_inputs: Any = None
    extra_targets: Any = None
    weights: Optional[torch.Tensor] = None  # shape [B_total] or None


@contextmanager
def _temporary_reduction(criterion: Any, reduction: str):
    if not hasattr(criterion, "reduction"):
        yield False
        return
    try:
        old_reduction = getattr(criterion, "reduction")
    except Exception:
        yield False
        return
    try:
        setattr(criterion, "reduction", reduction)
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        try:
            setattr(criterion, "reduction", old_reduction)
        except Exception:
            pass


def _concat_batches(base: Any, extra: Any, *, dim: int = 0) -> Any:
    """Concatenate ``extra`` onto ``base`` preserving the original structure."""
    if base is None:
        if extra is None:
            return None
        raise ValueError("Cannot concatenate extra data onto a None base.")

    if isinstance(base, torch.Tensor):
        if not isinstance(extra, torch.Tensor):
            raise TypeError("Mismatched types when concatenating trigger inputs.")
        return torch.cat([base, extra], dim=dim)

    if isinstance(base, (list, tuple)):
        if not isinstance(extra, type(base)) or len(base) != len(extra):
            raise TypeError("Trigger extras must match the structure of the original inputs.")
        concatenated = [
            _concat_batches(b, e, dim=dim)
            for b, e in zip(base, extra)
        ]
        return type(base)(concatenated)

    if isinstance(base, dict):
        if not isinstance(extra, dict) or base.keys() != extra.keys():
            raise TypeError("Trigger extras must provide the same keys as the original inputs.")
        return {
            key: _concat_batches(base[key], extra[key], dim=dim)
            for key in base
        }

    raise TypeError(
        "Unsupported batch structure for trigger concatenation: "
        f"{type(base).__name__}"
    )


def _infer_batch_size(data: Any) -> Optional[int]:
    if isinstance(data, torch.Tensor):
        return data.shape[0]
    if isinstance(data, (list, tuple)):
        for item in data:
            size = _infer_batch_size(item)
            if size is not None:
                return size
        return None
    if isinstance(data, dict):
        for value in data.values():
            size = _infer_batch_size(value)
            if size is not None:
                return size
    return None


def _compute_per_sample_loss(criterion: Any, outputs: Any, targets: Any) -> torch.Tensor:
    with _temporary_reduction(criterion, "none") as reduction_set:
        if not reduction_set:
            raise RuntimeError("Criterion does not support reduction='none'.")
        return criterion(outputs, targets)

class FastTrainer:
    """
    A fast, practical PyTorch training loop with:
      - Auto device (CUDA/MPS/CPU)
      - AMP (bf16/fp16 auto)
      - Gradient accumulation
      - Data transfer tweaks (non_blocking, pin_memory recommended at loader)
      - torch.compile (best-effort)
      - Sync reduction (.item() minimized, zero_grad(set_to_none=True))
      - Optional Trigger hook for dynamic hard-sample injection (loss_std-driven)
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any] = None, *,
                 device: Optional[str] = None,
                 use_amp: Optional[bool] = "auto",
                 compile_mode: str = "reduce-overhead",
                 grad_accum: int = 1,
                 channels_last: bool = False,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 50,
                 trigger_hook: Optional[Callable[[Dict[str, Any]], Optional[TriggerResult]]] = None):
        self.device = device or get_best_device()
        self.model = model.to(self.device)
        self.model = maybe_channels_last(self.model, channels_last=channels_last)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum = max(1, int(grad_accum))
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.trigger_hook = trigger_hook

        # AMP policy
        self.amp_enabled, self.amp_dtype, use_scaler = get_amp_policy(self.device, use_amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(use_scaler and self.amp_enabled))

        # torch.compile best-effort (skip CPU)
        self.compiled = False
        if self.device != "cpu":
            self.model, self.compiled = safe_compile(self.model, mode=compile_mode)

        # CUDA fast matmul precision
        if self.device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def train_one_epoch(self, loader, criterion, *, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Train for one epoch (or a fixed number of steps if steps is provided).
        Expects criterion to support reduction='mean'. If trigger_hook is set and
        you want per-sample logic, pass a criterion that supports reduction='none'.
        """
        self.model.train()
        meter = ThroughputMeter()

        # Detect if criterion supports reduction='none'
        supports_per_sample = False
        if criterion is not None:
            with _temporary_reduction(criterion, "none") as reduction_set:
                supports_per_sample = reduction_set

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_items = 0
        step_idx = 0

        for batch in loader:
            step_idx += 1
            if steps is not None and step_idx > steps:
                break

            batch = to_device(batch, self.device, non_blocking=True)
            # Support (inputs, targets) or dict with 'inputs','targets'
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            elif isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                inputs, targets = batch["inputs"], batch["targets"]
            else:
                # Fallback: treat entire batch as inputs, no targets (self-supervised / user handles loss)
                inputs, targets = batch, None

            with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                outputs = self.model(inputs)

                if criterion is not None:
                    if supports_per_sample and self.trigger_hook is not None:
                        # per-sample loss for trigger decisions
                        loss_vec = _compute_per_sample_loss(criterion, outputs, targets)
                        # Trigger may inject extra samples (e.g., hard examples)
                        trig_result = self.trigger_hook({
                            "inputs": inputs, "targets": targets, "outputs": outputs,
                            "loss_vec": loss_vec, "device": self.device, "step": step_idx
                        })
                        if trig_result is not None and trig_result.extra_inputs is not None:
                            extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                            combined_inputs = _concat_batches(inputs, extra_x)

                            if trig_result.extra_targets is not None:
                                if targets is None:
                                    raise ValueError(
                                        "Trigger provided extra targets but the base batch has no targets."
                                    )
                                extra_y = to_device(trig_result.extra_targets, self.device, non_blocking=True)
                                combined_targets = _concat_batches(targets, extra_y)
                            else:
                                combined_targets = targets

                            inputs = combined_inputs
                            targets = combined_targets
                            outputs = self.model(inputs)
                            loss_vec = _compute_per_sample_loss(criterion, outputs, targets)

                            if trig_result.weights is not None:
                                weights = trig_result.weights.to(loss_vec.device, dtype=loss_vec.dtype)
                                if weights.shape[0] != loss_vec.shape[0]:
                                    raise ValueError(
                                        "Trigger weights must have the same batch dimension as the loss vector."
                                    )
                                if weights.ndim == 1 and loss_vec.ndim > 1:
                                    view_shape = (weights.shape[0],) + (1,) * (loss_vec.ndim - 1)
                                    weights_broadcast = weights.view(view_shape)
                                else:
                                    weights_broadcast = weights
                                denom = weights.sum()
                                eps = torch.finfo(denom.dtype).eps
                                if denom.abs() <= eps:
                                    raise ValueError("Trigger weights must not sum to zero.")
                                loss = (loss_vec * weights_broadcast).sum() / denom
                            else:
                                loss = loss_vec.mean()
                        else:
                            loss = loss_vec.mean()
                    else:
                        loss = criterion(outputs, targets)
                else:
                    raise ValueError(
                        "No criterion provided. Supply a criterion or pre-compute the loss before "
                        "calling train_one_epoch."
                    )

            loss_to_report = loss.detach()
            loss = loss / self.grad_accum

            # Backward
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step if accumulation boundary
            if step_idx % self.grad_accum == 0:
                if self.clip_grad_norm is not None:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

            # Metrics
            bs = _infer_batch_size(inputs) or 1
            meter.tick(bs)
            total_items += bs
            total_loss = total_loss + loss_to_report.detach().to(total_loss.dtype)

            if (step_idx % self.log_interval) == 0:
                m = meter.summary()
                avg_loss = (total_loss / step_idx).item()
                print(f"[Step {step_idx}] loss~{avg_loss:.4f} | "
                      f"thr={m['samples_per_sec']:.1f}/s p50={m['p50_s']*1e3:.1f}ms p95={m['p95_s']*1e3:.1f}ms",
                      flush=True)

        metrics = meter.summary()
        if self.device == "cuda":
            try:
                metrics["cuda_max_mem_bytes"] = torch.cuda.max_memory_allocated()
            except Exception:
                pass
        metrics["avg_loss"] = total_loss.item() / max(1, step_idx)
        metrics["steps"] = step_idx
        metrics["samples"] = total_items
        metrics["amp"] = self.amp_enabled
        metrics["compiled"] = self.compiled
        metrics["device"] = self.device
        return metrics
