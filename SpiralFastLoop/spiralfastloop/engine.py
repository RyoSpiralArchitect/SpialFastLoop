# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn

from .utils import (
    get_best_device, get_amp_policy, autocast_ctx, to_device,
    ThroughputMeter, maybe_channels_last, safe_compile, dataloader_from_dataset
)


def _concatenate_batches(base: Any, extra: Any) -> Any:
    """Concatenate two batched structures along their first dimension."""
    if base is None:
        return extra
    if isinstance(base, torch.Tensor):
        if not isinstance(extra, torch.Tensor):
            raise TypeError("Extra inputs must mirror tensor structure of original batch.")
        return torch.cat([base, extra], dim=0)
    if isinstance(base, Mapping):
        if not isinstance(extra, Mapping):
            raise TypeError("Trigger extra batch must be a mapping matching the original batch.")
        if set(base.keys()) != set(extra.keys()):
            raise KeyError("Trigger extra batch keys must match the original batch keys.")
        return type(base)({k: _concatenate_batches(base[k], extra[k]) for k in base.keys()})
    if isinstance(base, Sequence) and not isinstance(base, (str, bytes)):
        if not isinstance(extra, Sequence) or len(base) != len(extra):
            raise TypeError("Trigger extra batch must match the sequence structure of the original batch.")
        concatenated = [_concatenate_batches(b, e) for b, e in zip(base, extra)]
        return type(base)(concatenated)
    raise TypeError("Unsupported batch structure for trigger concatenation.")


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    if isinstance(batch, Mapping):
        for value in batch.values():
            size = _infer_batch_size(value)
            if size is not None:
                return size
        return 1
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        for value in batch:
            size = _infer_batch_size(value)
            if size is not None:
                return size
        return 1
    return 1


@contextmanager
def _force_reduction(criterion: Any, reduction: str):
    """Temporarily set ``criterion.reduction`` if possible."""
    if not hasattr(criterion, "reduction"):
        yield False
        return
    old_reduction = getattr(criterion, "reduction")
    if old_reduction == reduction:
        yield True
        return
    try:
        criterion.reduction = reduction
    except Exception:
        yield False
        return
    success = getattr(criterion, "reduction", None) == reduction
    try:
        yield success
    finally:
        try:
            criterion.reduction = old_reduction
        except Exception:
            pass


def _per_sample_losses(loss_tensor: torch.Tensor) -> torch.Tensor:
    """Collapse criterion outputs into a 1-D per-sample vector."""
    if not isinstance(loss_tensor, torch.Tensor):
        raise TypeError("Criterion must return a tensor when reduction='none'.")
    if loss_tensor.ndim == 0:
        return loss_tensor.unsqueeze(0)
    if loss_tensor.shape[0] == 0:
        return loss_tensor.reshape(0)
    if loss_tensor.ndim == 1:
        return loss_tensor
    leading = loss_tensor.shape[0]
    return loss_tensor.reshape(leading, -1).mean(dim=1)


def _mean_loss(loss_vec: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        return loss_vec.mean()
    w = weights.to(loss_vec.device, dtype=loss_vec.dtype).reshape(-1)
    losses = loss_vec.reshape(-1)
    if losses.shape != w.shape:
        raise ValueError("Trigger weights must align with the per-sample loss vector.")
    denom = w.sum()
    if torch.isnan(denom) or torch.isinf(denom) or denom <= 0:
        raise ValueError("Trigger weights must sum to a positive finite value.")
    return (losses * w).sum() / denom

@dataclass
class TriggerResult:
    extra_inputs: Any = None
    extra_targets: Any = None
    weights: Optional[torch.Tensor] = None  # shape [B_total] or None

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
        with _force_reduction(criterion, "none") as ok:
            supports_per_sample = ok

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        total_weight = torch.zeros((), device=self.device, dtype=torch.float64)
        total_items = 0
        step_idx = 0
        optimizer_steps = 0

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

                if targets is not None and criterion is not None:
                    if supports_per_sample and self.trigger_hook is not None:
                        with _force_reduction(criterion, "none") as ok:
                            if not ok:
                                raise RuntimeError("Trigger requires criterion with reduction='none'.")
                            loss_tensor = criterion(outputs, targets)
                        loss_vec = _per_sample_losses(loss_tensor)
                        trig_ctx = {
                            "inputs": inputs,
                            "targets": targets,
                            "outputs": outputs,
                            "loss_vec": loss_vec.detach(),
                            "device": self.device,
                            "step": step_idx,
                        }
                        trig_result = self.trigger_hook(trig_ctx)
                        weights = None
                        if trig_result is not None:
                            if trig_result.extra_inputs is not None:
                                extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                                extra_y = (
                                    to_device(trig_result.extra_targets, self.device, non_blocking=True)
                                    if trig_result.extra_targets is not None
                                    else None
                                )
                                if extra_y is None:
                                    raise ValueError("Trigger provided extra inputs without matching targets.")
                                inputs = _concatenate_batches(inputs, extra_x)
                                targets = _concatenate_batches(targets, extra_y)
                                outputs = self.model(inputs)
                                with _force_reduction(criterion, "none") as ok2:
                                    if not ok2:
                                        raise RuntimeError("Trigger requires criterion with reduction='none'.")
                                    loss_tensor = criterion(outputs, targets)
                                loss_vec = _per_sample_losses(loss_tensor)
                            weights = trig_result.weights if trig_result.weights is not None else None
                        loss = _mean_loss(loss_vec, weights)
                    else:
                        loss = criterion(outputs, targets)
                        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                            loss = loss.mean()
                else:
                    raise ValueError("No criterion provided for supervised step; supply a loss function.")
                raw_loss = loss
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
                optimizer_steps += 1

            # Metrics
            bs = _infer_batch_size(inputs)
            meter.tick(bs)
            total_items += bs
            loss_detached = raw_loss.detach().to(device=total_loss.device, dtype=total_loss.dtype)
            batch_weight = total_loss.new_tensor(bs, dtype=total_loss.dtype)
            total_loss += loss_detached * batch_weight
            total_weight += batch_weight

            if (step_idx % self.log_interval) == 0:
                m = meter.summary()
                weight_value = total_weight.item()
                avg_loss = (total_loss / total_weight).item() if weight_value > 0 else 0.0
                print(f"[Step {step_idx}] loss~{avg_loss:.4f} | "
                      f"thr={m['samples_per_sec']:.1f}/s p50={m['p50_s']*1e3:.1f}ms p95={m['p95_s']*1e3:.1f}ms",
                      flush=True)

        metrics = meter.summary()
        if self.device == "cuda":
            try:
                metrics["cuda_max_mem_bytes"] = torch.cuda.max_memory_allocated()
            except Exception:
                pass
        weight_value = total_weight.item()
        if weight_value > 0:
            metrics["avg_loss"] = (total_loss / total_weight).item()
        else:
            metrics["avg_loss"] = 0.0
        metrics["steps"] = step_idx
        metrics["optimizer_steps"] = optimizer_steps
        metrics["samples"] = total_items
        metrics["amp"] = self.amp_enabled
        metrics["compiled"] = self.compiled
        metrics["device"] = self.device
        return metrics


def recommended_dataloader(dataset, *, batch_size: int, device: str, **kwargs):
    """Backward-compatible alias for :func:`spiralfastloop.utils.dataloader_from_dataset`."""
    return dataloader_from_dataset(dataset, batch_size=batch_size, device=device, **kwargs)
