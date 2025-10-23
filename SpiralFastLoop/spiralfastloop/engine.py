# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ryō

\
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Sequence, Mapping

import torch
import torch.nn as nn

from .utils import (
    get_best_device, get_amp_policy, autocast_ctx, to_device,
    ThroughputMeter, maybe_channels_last, safe_compile
)


def _concat_batches(primary: Any, extra: Any, *, dim: int = 0) -> Any:
    """Concatenate two batches that share the same nested structure."""
    if extra is None:
        return primary

    if torch.is_tensor(primary):
        if not torch.is_tensor(extra):
            raise TypeError("Extra batch must mirror tensor structure of primary batch")
        return torch.cat([primary, extra], dim=dim)

    if isinstance(primary, (list, tuple)):
        if not isinstance(extra, (list, tuple)):
            raise TypeError("Extra batch must mirror list/tuple structure of primary batch")
        if len(primary) != len(extra):
            raise ValueError("Extra batch must match list/tuple length of primary batch")
        concatenated = [_concat_batches(p, e, dim=dim) for p, e in zip(primary, extra)]
        return type(primary)(concatenated)

    if isinstance(primary, Mapping):
        if not isinstance(extra, Mapping):
            raise TypeError("Extra batch must mirror mapping structure of primary batch")
        if primary.keys() != extra.keys():
            missing = set(primary.keys()) ^ set(extra.keys())
            raise ValueError(f"Extra batch mapping keys must match primary batch keys: {missing}")
        return type(primary)({k: _concat_batches(primary[k], extra[k], dim=dim) for k in primary.keys()})

    raise TypeError("Unsupported batch type for concatenation; provide matching tensors, sequences, or mappings")


def _infer_batch_size(inputs: Any) -> int:
    """Best-effort batch-size inference for logging/throughput metrics."""
    if torch.is_tensor(inputs):
        return inputs.shape[0]
    if isinstance(inputs, Sequence) and inputs:
        for item in inputs:
            if torch.is_tensor(item):
                return item.shape[0]
    if isinstance(inputs, Mapping):
        for value in inputs.values():
            if torch.is_tensor(value):
                return value.shape[0]
    return 1

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
                            extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                            extra_y = (
                                to_device(trig_result.extra_targets, self.device, non_blocking=True)
                                if trig_result.extra_targets is not None else None
                            )

                            inputs = _concat_batches(inputs, extra_x)
                            if extra_y is not None:
                                targets = _concat_batches(targets, extra_y)

                            outputs = self.model(inputs)
                            loss_vec = criterion(outputs, targets)

                            if trig_result.weights is not None:
                                w = trig_result.weights.to(loss_vec.device)
                                if w.shape[0] != loss_vec.shape[0]:
                                    raise ValueError("Trigger weights must match concatenated batch length")
                                broadcast_dims = (1,) * max(0, loss_vec.dim() - 1)
                                loss = (loss_vec * w.view(w.shape[0], *broadcast_dims)).mean()
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

                effective_loss = loss
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
            bs = _infer_batch_size(inputs)
            meter.tick(bs)
            total_items += bs
            total_loss += float(effective_loss.detach().cpu())  # avoid .item() to reduce sync; cast via cpu()

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
