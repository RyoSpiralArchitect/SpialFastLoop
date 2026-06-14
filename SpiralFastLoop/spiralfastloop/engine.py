# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, cast

import fnmatch
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .utils import (
    AmpSetting,
    PhaseProfiler,
    ThroughputMeter,
    autocast_ctx,
    dataloader_from_dataset,
    distributed_sum,
    get_distributed_context,
    get_amp_policy,
    get_best_device,
    init_distributed,
    maybe_channels_last,
    safe_compile_with_diagnostics,
    to_device,
)
from .logging_utils import MetricsLogger

recommended_dataloader = dataloader_from_dataset


def _configure_cuda_backends(
    enable_tf32: bool,
    cudnn_benchmark: bool,
    reduced_precision_reduction: bool = True,
    enable_flash_sdp: Optional[bool] = True,
    enable_mem_efficient_sdp: Optional[bool] = True,
    enable_math_sdp: Optional[bool] = False,
    torch_mod: Any = torch,
) -> None:
    """Toggle CUDA backend knobs with a safe, testable helper."""
    try:
        cuda_backends = getattr(getattr(torch_mod, "backends", None), "cuda", None)
        cuda_module = cuda_backends
        matmul_backend = getattr(cuda_backends, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = bool(enable_tf32)
        for attr in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
            if matmul_backend is not None and hasattr(matmul_backend, attr):
                setattr(matmul_backend, attr, bool(reduced_precision_reduction))
        sdp_toggles = (
            ("enable_flash_sdp", enable_flash_sdp),
            ("enable_mem_efficient_sdp", enable_mem_efficient_sdp),
            ("enable_math_sdp", enable_math_sdp),
        )
        for fn_name, value in sdp_toggles:
            if value is None:
                continue
            fn = getattr(cuda_module, fn_name, None)
            if callable(fn):
                fn(bool(value))
    except Exception:
        pass
    try:
        cudnn_backend = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = bool(cudnn_benchmark)
    except Exception:
        pass


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
        mapping_type = cast(Any, type(base))
        if hasattr(base, "default_factory"):
            default_factory = getattr(base, "default_factory")
            new_mapping = mapping_type(default_factory)
            cast(MutableMapping[Any, Any], new_mapping).update(merged)
            return new_mapping
        new_mapping = mapping_type()
        cast(MutableMapping[Any, Any], new_mapping).update(merged)
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
        return int(batch.shape[0])
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


def _metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

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
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        *,
        device: Optional[str] = None,
        use_amp: AmpSetting = "auto",
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        grad_accum: int = 1,
        channels_last: bool = False,
        clip_grad_norm: Optional[float] = None,
        log_interval: int = 50,
        trigger_hook: Optional[Callable[[Dict[str, Any]], Optional[TriggerResult]]] = None,
        logger: Optional[MetricsLogger] = None,
        distributed: Optional[bool] = None,
        distributed_backend: Optional[str] = None,
        ddp_kwargs: Optional[Dict[str, Any]] = None,
        log_on_rank0: bool = True,
        enable_tf32: bool = True,
        cudnn_benchmark: bool = True,
        reduced_precision_reduction: bool = True,
        enable_flash_sdp: Optional[bool] = True,
        enable_mem_efficient_sdp: Optional[bool] = True,
        enable_math_sdp: Optional[bool] = False,
        meter_fast_mode: bool = False,
    ) -> None:
        if distributed is True:
            self.dist_ctx = init_distributed(backend=distributed_backend)
        else:
            self.dist_ctx = get_distributed_context()
        base_device = device or get_best_device()
        if self.dist_ctx.world_size > 1 and base_device.startswith("cuda"):
            torch.cuda.set_device(self.dist_ctx.local_rank)
            self.device = f"cuda:{self.dist_ctx.local_rank}"
        else:
            self.device = base_device
        self.model = model.to(self.device)
        self.model = maybe_channels_last(self.model, channels_last=channels_last)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum = max(1, int(grad_accum))
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = max(0, int(log_interval))
        self.trigger_hook = trigger_hook
        self.logger = logger
        self.log_on_rank0 = log_on_rank0
        self.meter_fast_mode = bool(meter_fast_mode)
        self.compile_requested = bool(use_compile)

        # AMP policy
        self.amp_enabled, self.amp_dtype, use_scaler = get_amp_policy(self.device, use_amp)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=(use_scaler and self.amp_enabled))
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(use_scaler and self.amp_enabled))

        # torch.compile best-effort (skip CPU)
        self.compiled = False
        self.compile_init_time_s = 0.0
        self.compile_fallback_reason = ""
        if self.compile_requested and self.device != "cpu":
            compile_started_at = time.perf_counter()
            compile_result = safe_compile_with_diagnostics(self.model, mode=compile_mode)
            self.model = compile_result.model
            self.compiled = compile_result.compiled
            self.compile_fallback_reason = compile_result.fallback_reason
            self.compile_init_time_s = time.perf_counter() - compile_started_at
        elif self.compile_requested:
            self.compile_fallback_reason = "cpu_device"
        else:
            self.compile_fallback_reason = "not_requested"

        # DDP wrap if requested and initialized
        self.using_ddp = False
        if self.dist_ctx.world_size > 1:
            ddp_kwargs = ddp_kwargs or {}
            if self.device.startswith("cuda"):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.dist_ctx.local_rank],
                    output_device=self.dist_ctx.local_rank,
                    **ddp_kwargs,
                )
            else:
                self.model = DistributedDataParallel(self.model, **ddp_kwargs)
            self.using_ddp = True

        # CUDA fast matmul precision
        if self.device.startswith("cuda"):
            _configure_cuda_backends(
                enable_tf32,
                cudnn_benchmark,
                reduced_precision_reduction,
                enable_flash_sdp,
                enable_mem_efficient_sdp,
                enable_math_sdp,
            )
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def _should_log(self) -> bool:
        return not self.log_on_rank0 or self.dist_ctx.is_primary

    def _log_metrics(
        self,
        stage: str,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        mode: str = "step",
    ) -> None:
        if not self._should_log():
            return
        if self.logger is not None:
            self.logger.log_metrics(stage, metrics, step=step, epoch=epoch, mode=mode)
        elif step is not None:
            summary = ", ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"[{stage}:{mode}] step={step} {summary}", flush=True)

    def _install_profile_model_hooks(
        self,
        profiler: PhaseProfiler,
        *,
        depth: int = 1,
        max_modules: int = 64,
        include: Optional[str | Sequence[str]] = None,
    ) -> list[Any]:
        if not profiler.enabled:
            return []
        root = getattr(self.model, "module", self.model)
        root = getattr(root, "_orig_mod", root)
        max_depth = max(1, int(depth))
        if include is None:
            include_patterns: list[str] = []
        elif isinstance(include, str):
            include_patterns = [item.strip() for item in include.split(",") if item.strip()]
        else:
            include_patterns = [str(item).strip() for item in include if str(item).strip()]

        def matches_include(name: str) -> bool:
            if not include_patterns:
                return True
            return any(
                name == pattern
                or name.startswith(pattern + ".")
                or fnmatch.fnmatchcase(name, pattern)
                for pattern in include_patterns
            )

        selected = []
        for name, module in root.named_modules():
            if not name:
                continue
            if len(name.split(".")) == max_depth and matches_include(name):
                selected.append((name, module))
            if len(selected) >= max(1, int(max_modules)):
                break

        handles: list[Any] = []

        def disable_dynamo_if_available(hook: Callable[..., Any]) -> Callable[..., Any]:
            try:
                dynamo = getattr(torch, "_dynamo", None)
                disable = getattr(dynamo, "disable", None)
                if callable(disable):
                    return cast(Callable[..., Any], disable(hook))
            except Exception:
                pass
            return hook

        def forward_pre(label: str) -> Callable[[nn.Module, Any], None]:
            def hook(_module: nn.Module, _inputs: Any) -> None:
                profiler.start_detail("forward", label)
            return hook

        def forward_post(label: str) -> Callable[[nn.Module, Any, Any], None]:
            def hook(_module: nn.Module, _inputs: Any, _outputs: Any) -> None:
                profiler.stop_detail("forward", label)
            return hook

        def grad_ready(label: str) -> Callable[[torch.Tensor], torch.Tensor]:
            def hook(grad: torch.Tensor) -> torch.Tensor:
                profiler.record_event_since_start("backward", "backward_grad_ready", label)
                return grad
            return hook

        seen_params = set()
        for name, module in selected:
            label = f"model.{name}"
            try:
                handles.append(module.register_forward_pre_hook(disable_dynamo_if_available(forward_pre(label))))
                handles.append(module.register_forward_hook(disable_dynamo_if_available(forward_post(label))))
            except Exception:
                pass
            for param in module.parameters(recurse=True):
                param_id = id(param)
                if param_id in seen_params or not param.requires_grad:
                    continue
                seen_params.add(param_id)
                try:
                    handles.append(param.register_hook(grad_ready(label)))
                except Exception:
                    pass
        return handles

    def train_one_epoch(
        self,
        loader: Iterable[Any],
        criterion: Any,
        *,
        steps: Optional[int] = None,
        epoch: Optional[int] = None,
        collect_profile: bool = False,
        profile_sync: bool = False,
        profile_distribution: bool = True,
        profile_window: int = 512,
        profile_model: bool = False,
        profile_model_depth: int = 1,
        profile_model_max_modules: int = 64,
        profile_model_include: Optional[str | Sequence[str]] = None,
        warmup_steps: int = 0,
    ) -> Dict[str, Any]:
        """
        Train for one epoch (or a fixed number of steps if steps is provided).
        Expects criterion to support reduction='mean'. If trigger_hook is set and
        you want per-sample logic, pass a criterion that supports reduction='none'.
        """
        self.model.train()
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler) and epoch is not None:
            sampler.set_epoch(epoch)
        warmup_step_limit = max(0, int(warmup_steps))
        meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        warmup_meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        steady_meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        profiler = PhaseProfiler(
            enabled=collect_profile,
            device=self.device,
            sync=profile_sync,
            track_distribution=profile_distribution,
            window=profile_window,
        )
        profile_hook_handles = self._install_profile_model_hooks(
            profiler,
            depth=profile_model_depth,
            max_modules=profile_model_max_modules,
            include=profile_model_include,
        ) if (collect_profile and profile_model) else []

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
        if self.device.startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        elif self.device.startswith("mps"):
            try:
                reset_mps_peak = getattr(torch.mps, "reset_peak_memory_stats", None)
                if callable(reset_mps_peak):
                    reset_mps_peak()
            except Exception:
                pass

        metric_dtype = torch.float32 if self.device.startswith("mps") else torch.float64
        total_loss = torch.zeros((), device=self.device, dtype=metric_dtype)
        total_weight = torch.zeros((), device=self.device, dtype=metric_dtype)
        warmup_loss = torch.zeros((), device=self.device, dtype=metric_dtype)
        warmup_weight = torch.zeros((), device=self.device, dtype=metric_dtype)
        steady_loss = torch.zeros((), device=self.device, dtype=metric_dtype)
        steady_weight = torch.zeros((), device=self.device, dtype=metric_dtype)
        total_items = 0
        warmup_items = 0
        steady_items = 0
        step_idx = 0
        optimizer_steps = 0
        warmup_recorded_steps = 0
        steady_recorded_steps = 0
        warmup_optimizer_steps = 0
        steady_optimizer_steps = 0

        def profiled_batches() -> Iterable[tuple[Any, float]]:
            nonlocal step_idx
            data_iter = iter(loader)
            try:
                while steps is None or step_idx < steps:
                    step_started_at = time.perf_counter()
                    profiler.start("data_wait")
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        profiler.cancel("data_wait")
                        break
                    profiler.stop("data_wait")
                    step_idx += 1
                    yield batch, step_started_at
            finally:
                for handle in profile_hook_handles:
                    try:
                        handle.remove()
                    except Exception:
                        pass

        for batch, step_started_at in profiled_batches():

            profiler.start("transfer")
            batch = to_device(batch, self.device, non_blocking=True)
            profiler.stop("transfer")
            # Support (inputs, targets) or dict with 'inputs','targets'
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            elif isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                inputs, targets = batch["inputs"], batch["targets"]
            else:
                # Fallback: treat entire batch as inputs, no targets (self-supervised / user handles loss)
                inputs, targets = batch, None

            batch_size: Optional[int] = None
            loss_weight_tensor: Optional[torch.Tensor] = None

            with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                profiler.start("forward")
                outputs = self.model(inputs)
                profiler.stop("forward")

                if targets is not None and criterion is not None:
                    if supports_per_sample and self.trigger_hook is not None:
                        reduction_to_restore = None
                        if hasattr(criterion, "reduction") and getattr(criterion, "reduction") != "none":
                            reduction_to_restore = getattr(criterion, "reduction")
                            criterion.reduction = "none"
                        try:
                            # per-sample loss for trigger decisions
                            profiler.start("loss")
                            loss_vec = _ensure_loss_vector(criterion(outputs, targets))
                            profiler.stop("loss")
                        finally:
                            if reduction_to_restore is not None:
                                criterion.reduction = reduction_to_restore
                        batch_size = loss_vec.shape[0]
                        trigger_ctx = {
                            "inputs": inputs,
                            "targets": targets,
                            "outputs": outputs,
                            "loss_vec": loss_vec,
                            "device": self.device,
                            "step": step_idx,
                        }
                        if hasattr(self.trigger_hook, "observe"):
                            self.trigger_hook.observe(trigger_ctx)
                        # Trigger may inject extra samples (e.g., hard examples)
                        profiler.start("trigger")
                        trig_result = self.trigger_hook(trigger_ctx)
                        profiler.stop("trigger")
                        weights: Optional[torch.Tensor] = None
                        if trig_result is not None:
                            if trig_result.extra_inputs is not None:
                                # Concatenate and recompute outputs & loss_vec
                                profiler.start("inject_transfer")
                                extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                                extra_y = (
                                    to_device(trig_result.extra_targets, self.device, non_blocking=True)
                                    if trig_result.extra_targets is not None
                                    else None
                                )
                                inputs = _concatenate_batches(inputs, extra_x)
                                if extra_y is None:
                                    raise ValueError("Trigger provided extra inputs without matching targets.")
                                targets = _concatenate_batches(targets, extra_y)
                                profiler.stop("inject_transfer")
                                profiler.start("forward")
                                outputs = self.model(inputs)
                                profiler.stop("forward")
                                reduction_to_restore = None
                                if hasattr(criterion, "reduction") and getattr(criterion, "reduction") != "none":
                                    reduction_to_restore = getattr(criterion, "reduction")
                                    criterion.reduction = "none"
                                try:
                                    profiler.start("loss")
                                    loss_vec = _ensure_loss_vector(criterion(outputs, targets))
                                    profiler.stop("loss")
                                finally:
                                    if reduction_to_restore is not None:
                                        criterion.reduction = reduction_to_restore
                                batch_size = loss_vec.shape[0]
                            weights = trig_result.weights

                        if weights is not None:
                            profiler.start("loss_reduce")
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
                            profiler.stop("loss_reduce")
                        else:
                            profiler.start("loss_reduce")
                            loss = loss_vec.mean()
                            loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
                            profiler.stop("loss_reduce")
                    else:
                        profiler.start("loss")
                        loss = criterion(outputs, targets)
                        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                            loss = loss.mean()
                        profiler.stop("loss")
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
            profiler.start("backward")
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            profiler.stop("backward")

            # Step if accumulation boundary
            if step_idx % self.grad_accum == 0:
                profiler.start("optimizer")
                try:
                    if self.clip_grad_norm is not None:
                        profiler.start_detail("optimizer", "clip_grad_norm")
                        try:
                            if self.scaler.is_enabled():
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        finally:
                            profiler.stop_detail("optimizer", "clip_grad_norm")

                    if self.scaler.is_enabled():
                        profiler.start_detail("optimizer", "scaler.step")
                        try:
                            self.scaler.step(self.optimizer)
                        finally:
                            profiler.stop_detail("optimizer", "scaler.step")
                        profiler.start_detail("optimizer", "scaler.update")
                        try:
                            self.scaler.update()
                        finally:
                            profiler.stop_detail("optimizer", "scaler.update")
                    else:
                        profiler.start_detail("optimizer", "optimizer.step")
                        try:
                            self.optimizer.step()
                        finally:
                            profiler.stop_detail("optimizer", "optimizer.step")

                    profiler.start_detail("optimizer", "zero_grad")
                    try:
                        self.optimizer.zero_grad(set_to_none=True)
                    finally:
                        profiler.stop_detail("optimizer", "zero_grad")
                    if self.scheduler is not None:
                        profiler.start_detail("optimizer", "scheduler.step")
                        try:
                            try:
                                self.scheduler.step()
                            except Exception:
                                pass
                        finally:
                            profiler.stop_detail("optimizer", "scheduler.step")
                    optimizer_steps += 1
                    if warmup_step_limit > 0 and step_idx <= warmup_step_limit:
                        warmup_optimizer_steps += 1
                    else:
                        steady_optimizer_steps += 1
                finally:
                    profiler.stop("optimizer")

            # Metrics
            profiler.start("metrics")
            if batch_size is None:
                reference = targets if targets is not None else inputs
                batch_size = _infer_batch_size(reference)
            if loss_weight_tensor is None:
                loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)

            batch_size_int = int(batch_size)
            batch_duration_s = max(0.0, time.perf_counter() - step_started_at)
            meter.record(batch_duration_s, batch_size_int)
            total_items += batch_size_int
            loss_detached = raw_loss.detach().to(device=total_loss.device, dtype=total_loss.dtype)
            total_loss += loss_detached * loss_weight_tensor
            total_weight += loss_weight_tensor
            if warmup_step_limit > 0 and step_idx <= warmup_step_limit:
                warmup_meter.record(batch_duration_s, batch_size_int)
                warmup_items += batch_size_int
                warmup_loss += loss_detached * loss_weight_tensor
                warmup_weight += loss_weight_tensor
                warmup_recorded_steps += 1
            else:
                steady_meter.record(batch_duration_s, batch_size_int)
                steady_items += batch_size_int
                steady_loss += loss_detached * loss_weight_tensor
                steady_weight += loss_weight_tensor
                steady_recorded_steps += 1
            profiler.stop("metrics")

            if self.log_interval > 0 and (step_idx % self.log_interval) == 0:
                m = meter.summary()
                steady_m = steady_meter.summary()
                weight_value = total_weight.item()
                avg_loss = (total_loss / total_weight).item() if weight_value > 0 else 0.0
                self._log_metrics(
                    "train",
                    {
                        "avg_loss": avg_loss,
                        "samples_per_sec": m["samples_per_sec"],
                        "p50_s": m["p50_s"],
                        "p95_s": m["p95_s"],
                        "p99_s": m["p99_s"],
                        "steady_samples_per_sec": steady_m["samples_per_sec"],
                    },
                    step=step_idx,
                    epoch=epoch,
                    mode="step",
                )

        metrics: Dict[str, Any] = dict(meter.summary())
        if self.device.startswith("cuda"):
            try:
                metrics["cuda_current_mem_bytes"] = torch.cuda.memory_allocated()
                metrics["cuda_max_mem_bytes"] = torch.cuda.max_memory_allocated()
                metrics["cuda_reserved_mem_bytes"] = torch.cuda.memory_reserved()
                metrics["cuda_max_reserved_mem_bytes"] = torch.cuda.max_memory_reserved()
            except Exception:
                pass
        elif self.device.startswith("mps"):
            try:
                metrics["mps_current_mem_bytes"] = torch.mps.current_allocated_memory()
            except Exception:
                pass
            try:
                metrics["mps_driver_mem_bytes"] = torch.mps.driver_allocated_memory()
            except Exception:
                pass
            try:
                metrics["mps_recommended_max_mem_bytes"] = torch.mps.recommended_max_memory()
            except Exception:
                pass
            try:
                max_mps_memory = getattr(torch.mps, "max_memory_allocated", None)
                if callable(max_mps_memory):
                    metrics["mps_max_mem_bytes"] = max_mps_memory()
            except Exception:
                pass
        if self.dist_ctx.world_size > 1:
            total_loss = distributed_sum(total_loss)
            total_weight = distributed_sum(total_weight)
            warmup_loss = distributed_sum(warmup_loss)
            warmup_weight = distributed_sum(warmup_weight)
            steady_loss = distributed_sum(steady_loss)
            steady_weight = distributed_sum(steady_weight)
            total_items = int(distributed_sum(torch.tensor(total_items, device=total_loss.device)).item())
            warmup_items = int(distributed_sum(torch.tensor(warmup_items, device=total_loss.device)).item())
            steady_items = int(distributed_sum(torch.tensor(steady_items, device=total_loss.device)).item())
        weight_value = total_weight.item()
        if weight_value > 0:
            metrics["avg_loss"] = (total_loss / total_weight).item()
        else:
            metrics["avg_loss"] = 0.0
        warmup_weight_value = warmup_weight.item()
        steady_weight_value = steady_weight.item()
        if warmup_weight_value > 0:
            metrics["warmup_avg_loss"] = (warmup_loss / warmup_weight).item()
        else:
            metrics["warmup_avg_loss"] = 0.0
        if steady_weight_value > 0:
            metrics["steady_avg_loss"] = (steady_loss / steady_weight).item()
        else:
            metrics["steady_avg_loss"] = 0.0
        metrics["steps"] = step_idx
        metrics["optimizer_steps"] = optimizer_steps
        metrics["samples"] = total_items
        warmup_summary = warmup_meter.summary()
        steady_summary = steady_meter.summary()
        for key, value in warmup_summary.items():
            metrics[f"warmup_{key}"] = value
        for key, value in steady_summary.items():
            metrics[f"steady_{key}"] = value
        metrics["warmup_steps"] = warmup_recorded_steps
        metrics["steady_steps"] = steady_recorded_steps
        metrics["warmup_samples"] = warmup_items
        metrics["steady_samples"] = steady_items
        metrics["warmup_optimizer_steps"] = warmup_optimizer_steps
        metrics["steady_optimizer_steps"] = steady_optimizer_steps
        metrics["cold_start_steps"] = warmup_recorded_steps
        metrics["cold_start_time_s"] = warmup_summary["total_time_s"]
        metrics["cold_start_samples_per_sec"] = warmup_summary["samples_per_sec"]
        metrics["reported_samples_per_sec"] = (
            steady_summary["samples_per_sec"] if steady_recorded_steps > 0 else metrics["samples_per_sec"]
        )
        metrics["amp"] = self.amp_enabled
        metrics["compile_requested"] = self.compile_requested
        metrics["compiled"] = self.compiled
        metrics["compile_init_time_s"] = self.compile_init_time_s
        metrics["compile_fallback_reason"] = self.compile_fallback_reason
        metrics["device"] = self.device
        metrics["world_size"] = self.dist_ctx.world_size
        metrics["rank"] = self.dist_ctx.rank
        if collect_profile:
            metrics["profile"] = profiler.summary()
        self._log_metrics("train", metrics, epoch=epoch, mode="epoch")
        return metrics

    def fit(
        self,
        dataset: torch.utils.data.Dataset[Any],
        criterion: Any,
        *,
        batch_size: int = 256,
        steps: Optional[int] = None,
        epoch: Optional[int] = None,
        **loader_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train on a dataset with a minimal-parameter entrypoint."""
        loader = dataloader_from_dataset(
            dataset,
            batch_size=batch_size,
            device=self.device,
            **loader_kwargs,
        )
        return self.train_one_epoch(loader, criterion, steps=steps, epoch=epoch)

    def evaluate(
        self,
        loader: Iterable[Any],
        criterion: Optional[Any] = None,
        *,
        metrics_fn: Optional[Callable[[Any, Any, Any], Dict[str, Any]]] = None,
        steps: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a validation/evaluation loop without gradient updates."""
        self.model.eval()
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler) and epoch is not None:
            sampler.set_epoch(epoch)
        meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        metric_dtype = torch.float32 if self.device.startswith("mps") else torch.float64
        total_loss = torch.zeros((), device=self.device, dtype=metric_dtype)
        total_weight = torch.zeros((), device=self.device, dtype=metric_dtype)
        total_items = 0
        step_idx = 0
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}

        with torch.no_grad():
            for batch in loader:
                if steps is not None and step_idx >= steps:
                    break
                step_idx += 1
                batch = to_device(batch, self.device, non_blocking=True)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                elif isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                    inputs, targets = batch["inputs"], batch["targets"]
                else:
                    inputs, targets = batch, None

                with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                    outputs = self.model(inputs)
                    if criterion is not None and targets is not None:
                        loss = criterion(outputs, targets)
                        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                            loss = loss.mean()
                    else:
                        loss = None
                reference = targets if targets is not None else inputs
                batch_size = _infer_batch_size(reference)
                meter.tick(int(batch_size))
                total_items += int(batch_size)

                if loss is not None:
                    loss_detached = loss.detach().to(device=total_loss.device, dtype=total_loss.dtype)
                    weight_tensor = total_weight.new_tensor(batch_size, dtype=total_weight.dtype)
                    total_loss += loss_detached * weight_tensor
                    total_weight += weight_tensor

                if metrics_fn is not None:
                    extra = metrics_fn(outputs, targets, inputs)
                    for key, value in extra.items():
                        metric_value = _metric_to_float(value)
                        metric_sums[key] = metric_sums.get(key, 0.0) + metric_value * batch_size
                        metric_weights[key] = metric_weights.get(key, 0.0) + batch_size

                if self.log_interval > 0 and (step_idx % self.log_interval) == 0:
                    m = meter.summary()
                    self._log_metrics(
                        "eval",
                        {
                            "samples_per_sec": m["samples_per_sec"],
                            "p50_s": m["p50_s"],
                            "p95_s": m["p95_s"],
                            "p99_s": m["p99_s"],
                        },
                        step=step_idx,
                        epoch=epoch,
                        mode="step",
                    )

        if self.dist_ctx.world_size > 1:
            total_loss = distributed_sum(total_loss)
            total_weight = distributed_sum(total_weight)
            total_items = int(distributed_sum(torch.tensor(total_items, device=total_loss.device)).item())
            for key in list(metric_sums.keys()):
                metric_sums[key] = float(
                    distributed_sum(torch.tensor(metric_sums[key], device=total_loss.device)).item()
                )
                metric_weights[key] = float(
                    distributed_sum(torch.tensor(metric_weights[key], device=total_loss.device)).item()
                )
        metrics: Dict[str, Any] = dict(meter.summary())
        weight_value = total_weight.item()
        if weight_value > 0:
            metrics["avg_loss"] = (total_loss / total_weight).item()
        else:
            metrics["avg_loss"] = 0.0
        metrics["steps"] = step_idx
        metrics["samples"] = total_items
        metrics["device"] = self.device
        metrics["world_size"] = self.dist_ctx.world_size
        metrics["rank"] = self.dist_ctx.rank

        for key, total in metric_sums.items():
            denom = metric_weights.get(key, 0.0)
            metrics[key] = total / denom if denom else 0.0
        self._log_metrics("eval", metrics, epoch=epoch, mode="epoch")
        return metrics

    def predict(
        self,
        loader: Iterable[Any],
        *,
        steps: Optional[int] = None,
        postprocess: Optional[Callable[[Any], Any]] = None,
    ) -> list[Any]:
        """Run inference and collect outputs on CPU."""
        self.model.eval()
        outputs_list: list[Any] = []
        step_idx = 0
        with torch.no_grad():
            for batch in loader:
                step_idx += 1
                if steps is not None and step_idx > steps:
                    break
                batch = to_device(batch, self.device, non_blocking=True)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs = batch[0]
                elif isinstance(batch, dict) and "inputs" in batch:
                    inputs = batch["inputs"]
                else:
                    inputs = batch
                with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                    outputs = self.model(inputs)
                if postprocess is not None:
                    outputs = postprocess(outputs)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.detach().cpu()
                outputs_list.append(outputs)
        return outputs_list
