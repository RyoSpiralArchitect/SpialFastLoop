# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn

from .utils import (
    get_best_device, get_amp_policy, autocast_ctx, to_device,
    ThroughputMeter, maybe_channels_last, safe_compile
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
        merged = {k: _concatenate_batches(base[k], extra[k]) for k in base.keys()}
        default_factory = getattr(base, "default_factory", None)
        if hasattr(base, "default_factory"):
            new_mapping = type(base)(default_factory)
            new_mapping.update(merged)
            return new_mapping
        try:
            return type(base)(merged)
        except TypeError:
            new_mapping = type(base)()
            new_mapping.update(merged)
            return new_mapping
    if isinstance(base, list):
        if not isinstance(extra, (list, tuple)) or len(base) != len(extra):
            raise TypeError("Trigger extra batch must match the list structure of the original batch.")
        concatenated = [_concatenate_batches(b, e) for b, e in zip(base, extra)]
        return type(base)(concatenated)
    if isinstance(base, tuple):
        if not isinstance(extra, (list, tuple)) or len(base) != len(extra):
            raise TypeError("Trigger extra batch must match the tuple structure of the original batch.")
        concatenated = [_concatenate_batches(b, e) for b, e in zip(base, extra)]
        if hasattr(base, "_fields"):
            return type(base)(*concatenated)
        return tuple(concatenated)
    if base is None and extra is None:
        return None
    raise TypeError("Unsupported batch structure for trigger concatenation.")


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    if isinstance(batch, Mapping):
        candidate_values = []
        for value in batch.values():
            if value is None:
                continue
            try:
                candidate_values.append(_infer_batch_size(value))
            except (TypeError, ValueError):
                continue
        if not candidate_values:
            raise ValueError("Unable to infer batch size from mapping inputs provided by trigger.")
        unique = set(candidate_values)
        if len(unique) != 1:
            raise ValueError("Inconsistent batch sizes detected in mapping inputs provided by trigger.")
        return candidate_values[0]
    if isinstance(batch, (list, tuple)):
        candidate_values = []
        for value in batch:
            if value is None:
                continue
            try:
                candidate_values.append(_infer_batch_size(value))
            except (TypeError, ValueError):
                continue
        if not candidate_values:
            if hasattr(batch, "__len__"):
                return len(batch)
            raise ValueError("Unable to infer batch size from sequence inputs provided by trigger.")
        unique = set(candidate_values)
        if len(unique) != 1:
            raise ValueError("Inconsistent batch sizes detected in sequence inputs provided by trigger.")
        return candidate_values[0]
    if batch is None:
        raise ValueError("Cannot infer batch size from None input.")
    raise TypeError("Unsupported batch structure for inferring batch size.")


def _ensure_loss_vector(loss_tensor: torch.Tensor) -> torch.Tensor:
    if loss_tensor.ndim == 0:
        return loss_tensor.unsqueeze(0)
    if loss_tensor.ndim == 1:
        return loss_tensor
    if loss_tensor.shape[0] <= 0:
        raise ValueError("Loss tensor must have a non-zero batch dimension.")
    return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)

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
        if hasattr(criterion, "reduction"):
            old_reduction = getattr(criterion, "reduction")
            try:
                criterion.reduction = "none"
                supports_per_sample = getattr(criterion, "reduction", None) == "none"
            except Exception:
                supports_per_sample = False
            finally:
                try:
                    criterion.reduction = old_reduction
                except Exception:
                    pass

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

            batch_size = None
            loss_weight_tensor = None

            with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                outputs = self.model(inputs)

                if targets is not None and criterion is not None:
                    if supports_per_sample and self.trigger_hook is not None:
                        reduction_to_restore = None
                        if hasattr(criterion, "reduction") and getattr(criterion, "reduction") != "none":
                            reduction_to_restore = getattr(criterion, "reduction")
                            criterion.reduction = "none"
                        try:
                            # per-sample loss for trigger decisions
                            loss_vec = _ensure_loss_vector(criterion(outputs, targets))
                        finally:
                            if reduction_to_restore is not None:
                                criterion.reduction = reduction_to_restore
                        batch_size = loss_vec.shape[0]
                        # Trigger may inject extra samples (e.g., hard examples)
                        trig_result = self.trigger_hook({
                            "inputs": inputs, "targets": targets, "outputs": outputs,
                            "loss_vec": loss_vec, "device": self.device, "step": step_idx
                        })
                        weights = None
                        if trig_result is not None:
                            if trig_result.extra_inputs is not None:
                                # Concatenate and recompute outputs & loss_vec
                                extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                                extra_y = to_device(trig_result.extra_targets, self.device, non_blocking=True) if trig_result.extra_targets is not None else None
                                inputs = _concatenate_batches(inputs, extra_x)
                                if extra_y is None:
                                    raise ValueError("Trigger provided extra inputs without matching targets.")
                                targets = _concatenate_batches(targets, extra_y)
                                outputs = self.model(inputs)
                                reduction_to_restore = None
                                if hasattr(criterion, "reduction") and getattr(criterion, "reduction") != "none":
                                    reduction_to_restore = getattr(criterion, "reduction")
                                    criterion.reduction = "none"
                                try:
                                    loss_vec = _ensure_loss_vector(criterion(outputs, targets))
                                finally:
                                    if reduction_to_restore is not None:
                                        criterion.reduction = reduction_to_restore
                                batch_size = loss_vec.shape[0]
                            weights = trig_result.weights

                        if weights is not None:
                            w = weights.to(loss_vec.device, dtype=loss_vec.dtype)
                            if w.ndim != 1 or w.shape[0] != loss_vec.shape[0]:
                                raise ValueError("Trigger weights must be a 1D tensor that matches the concatenated batch size.")
                            weight_sum = w.sum()
                            weight_sum_detached = weight_sum.detach()
                            if not torch.isfinite(weight_sum_detached):
                                raise ValueError("Trigger weights must be finite.")
                            if weight_sum_detached.item() <= 0:
                                raise ValueError("Trigger weights must sum to a positive value.")
                            loss = (loss_vec * w).sum() / weight_sum
                            loss_weight_tensor = weight_sum_detached.to(device=total_loss.device, dtype=total_loss.dtype)
                        else:
                            loss = loss_vec.mean()
                            loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
                    else:
                        loss = criterion(outputs, targets)
                        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                            loss = loss.mean()
                        reference = targets if targets is not None else inputs
                        batch_size = _infer_batch_size(reference)
                        loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
                    if batch_size is None:
                        reference = targets if targets is not None else inputs
                        batch_size = _infer_batch_size(reference)
                        if loss_weight_tensor is None:
                            loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
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
            if batch_size is None:
                reference = targets if targets is not None else inputs
                batch_size = _infer_batch_size(reference)
            if loss_weight_tensor is None:
                loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)

            meter.tick(int(batch_size))
            total_items += int(batch_size)
            loss_detached = raw_loss.detach().to(device=total_loss.device, dtype=total_loss.dtype)
            total_loss += loss_detached * loss_weight_tensor
            total_weight += loss_weight_tensor

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
