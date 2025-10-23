# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

\
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
        try:
            if hasattr(criterion, "reduction"):
                old = criterion.reduction
                criterion.reduction = "none"
                criterion.reduction = old
                supports_per_sample = True
        except Exception:
            supports_per_sample = False

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
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

                if targets is not None and criterion is not None:
                    if supports_per_sample and self.trigger_hook is not None:
                        # per-sample loss for trigger decisions
                        loss_vec = criterion(outputs, targets)
                        # Trigger may inject extra samples (e.g., hard examples)
                        trig_result = self.trigger_hook({
                            "inputs": inputs, "targets": targets, "outputs": outputs,
                            "loss_vec": loss_vec, "device": self.device, "step": step_idx
                        })
                        if trig_result is not None and trig_result.extra_inputs is not None:
                            # Concatenate and recompute outputs & loss_vec
                            extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                            extra_y = to_device(trig_result.extra_targets, self.device, non_blocking=True) if trig_result.extra_targets is not None else None
                            # Recompute with concatenated batch
                            if isinstance(inputs, torch.Tensor):
                                cat_inputs = torch.cat([inputs, extra_x], dim=0)
                            elif isinstance(inputs, (list, tuple)):
                                cat_inputs = type(inputs)(torch.cat([inputs[0], extra_x], dim=0))  # simplistic
                            else:
                                cat_inputs = extra_x  # user responsibility for exotic structures
                            outputs = self.model(cat_inputs)
                            if extra_y is not None:
                                targets = torch.cat([targets, extra_y], dim=0)
                            loss_vec = criterion(outputs, targets)

                            if trig_result.weights is not None:
                                w = trig_result.weights.to(loss_vec.device)
                                loss = (loss_vec * w).mean()
                            else:
                                loss = loss_vec.mean()
                        else:
                            loss = loss_vec.mean()
                    else:
                        loss = criterion(outputs, targets)
                else:
                    # User handles their own loss externally
                    if isinstance(outputs, torch.Tensor):
                        loss = outputs.mean()  # placeholder to keep graph moving
                    else:
                        raise ValueError("No criterion provided and outputs are not a tensor.")

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
            bs = None
            if isinstance(inputs, torch.Tensor):
                bs = inputs.shape[0]
            elif isinstance(inputs, (list, tuple)) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                bs = inputs[0].shape[0]
            else:
                bs = 1
            meter.tick(bs)
            total_items += bs
            total_loss += float(loss.detach().cpu())  # avoid .item() to reduce sync; cast via cpu()

            if (step_idx % self.log_interval) == 0:
                m = meter.summary()
                print(f"[Step {step_idx}] loss~{total_loss/(step_idx):.4f} | "
                      f"thr={m['samples_per_sec']:.1f}/s p50={m['p50_s']*1e3:.1f}ms p95={m['p95_s']*1e3:.1f}ms",
                      flush=True)

        metrics = meter.summary()
        if self.device == "cuda":
            try:
                metrics["cuda_max_mem_bytes"] = torch.cuda.max_memory_allocated()
            except Exception:
                pass
        metrics["avg_loss"] = total_loss / max(1, step_idx)
        metrics["steps"] = step_idx
        metrics["samples"] = total_items
        metrics["amp"] = self.amp_enabled
        metrics["compiled"] = self.compiled
        metrics["device"] = self.device
        return metrics
