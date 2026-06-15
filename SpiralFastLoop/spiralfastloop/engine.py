# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, cast

import fnmatch
import math
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .utils import (
    AmpSetting,
    PhaseProfiler,
    ThroughputMeter,
    _bool_setting,
    _device_setting,
    _module_setting,
    _non_negative_finite_float_setting,
    _non_negative_int_setting,
    _optional_bool_setting,
    _optional_non_empty_string_setting,
    _optional_positive_int_setting,
    _positive_int_setting,
    autocast_ctx,
    dataloader_from_dataset,
    distributed_max,
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

_PROFILE_PHASE_METRIC_NAMES = (
    "data_wait",
    "transfer",
    "forward",
    "loss",
    "loss_reduce",
    "trigger",
    "inject_transfer",
    "backward",
    "optimizer",
    "user_metrics",
    "postprocess",
    "collect_output",
    "metrics",
)
_PROFILE_METRIC_MISSING = object()

_BATCH_SIZE_INFERENCE_FAILURE_REASONS = (
    "tensor_scalar",
    "tensor_empty",
    "mapping_empty",
    "mapping_inconsistent",
    "sequence_empty",
    "sequence_inconsistent",
    "none",
    "unsupported_type",
)

_BATCH_SIZE_INFERENCE_FAILURE_MESSAGES = {
    "tensor_scalar": "Unable to infer batch size from scalar tensor.",
    "tensor_empty": "Tensor batch dimension must be non-zero.",
    "mapping_empty": "Unable to infer batch size from mapping input.",
    "mapping_inconsistent": "Inconsistent batch sizes detected in mapping input.",
    "sequence_empty": "Sequence batch dimension must be non-zero.",
    "sequence_inconsistent": "Inconsistent batch sizes detected in sequence input.",
    "none": "Cannot infer batch size from None input.",
    "unsupported_type": "Unsupported batch structure for inferring batch size.",
}

_USER_METRIC_RESERVED_NAMES = frozenset({
    "avg_loss",
    "avg_batch_s",
    "best_samples_per_sec",
    "compile_fallback_reason",
    "compile_init_time_s",
    "compile_requested",
    "compiled",
    "device",
    "eval_failed",
    "eval_failure_stage",
    "eval_failure_last_error",
    "headroom_ratio",
    "measured_steps",
    "p50_s",
    "p95_s",
    "p99_s",
    "postprocess_calls",
    "postprocess_failures",
    "postprocess_last_error",
    "postprocess_requested",
    "postprocess_successes",
    "predict_failed",
    "predict_failure_stage",
    "predict_failure_last_error",
    "profile",
    "rank",
    "reported_samples_per_sec",
    "samples",
    "samples_per_sec",
    "steps",
    "std_batch_s",
    "total_time_s",
    "train_failed",
    "train_failure_stage",
    "train_failure_last_error",
    "unmeasured_steps",
    "world_size",
})
_USER_METRIC_RESERVED_PREFIXES = (
    "batch_size_inference_",
    "cuda_",
    "metrics_fn_",
    "mps_",
    "profile_",
    "user_metric_",
)


def _format_exception_reason(exc: Exception, limit: int = 200) -> str:
    try:
        message = str(exc).strip()
    except Exception:
        message = ""
    if len(message) > limit:
        message = f"{message[:limit - 3]}..."
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _finite_profile_value(raw: Any) -> Optional[float]:
    if isinstance(raw, (bool, str)):
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _set_profile_metric(
    metrics: Dict[str, Any],
    name: str,
    raw: Any,
    invalid_fields: list[str],
    *,
    min_value: Optional[float] = 0.0,
    max_value: Optional[float] = None,
) -> Optional[float]:
    value = _finite_profile_value(raw)
    if value is None:
        invalid_fields.append(name)
        return None
    if min_value is not None and value < min_value:
        invalid_fields.append(name)
        return None
    if max_value is not None and value > max_value:
        invalid_fields.append(name)
        return None
    metrics[name] = value
    return value


def _set_profile_count_metric(
    metrics: Dict[str, Any],
    name: str,
    raw: Any,
    invalid_fields: list[str],
) -> Optional[int]:
    value = _finite_profile_value(raw)
    if value is None or value < 0.0 or not value.is_integer():
        invalid_fields.append(name)
        return None
    count = int(value)
    metrics[name] = count
    return count


def _record_profile_flat_metric_invalids(metrics: Dict[str, Any], invalid_fields: list[str]) -> None:
    metrics["profile_flat_metric_invalid_count"] = len(invalid_fields)
    if invalid_fields:
        metrics["profile_flat_metric_invalid_fields"] = invalid_fields


def _set_profile_metric_if_present(
    metrics: Dict[str, Any],
    name: str,
    source: Mapping[str, Any],
    source_name: str,
    invalid_fields: list[str],
    *,
    max_value: Optional[float] = None,
) -> None:
    if source_name not in source:
        return
    _set_profile_metric(
        metrics,
        name,
        source[source_name],
        invalid_fields,
        max_value=max_value,
    )


def _set_profile_count_metric_if_present(
    metrics: Dict[str, Any],
    name: str,
    source: Mapping[str, Any],
    source_name: str,
    invalid_fields: list[str],
) -> None:
    if source_name not in source:
        return
    _set_profile_count_metric(metrics, name, source[source_name], invalid_fields)


def _add_profile_backward_event_metrics(
    metrics: Dict[str, Any],
    profile: Mapping[str, Any],
    invalid_fields: list[str],
) -> None:
    events = profile.get("phase_events", {})
    if not isinstance(events, Mapping):
        return
    group = events.get("backward_grad_ready")
    if not isinstance(group, Mapping):
        return

    children = group.get("children", {})
    if isinstance(children, Mapping):
        metrics["profile_backward_grad_ready_child_count"] = len(children)
    _set_profile_metric_if_present(
        metrics,
        "profile_backward_grad_ready_parent_avg_ms",
        group,
        "parent_avg_ms",
        invalid_fields,
    )

    top_children = group.get("top_children", ())
    if not isinstance(top_children, Sequence) or isinstance(top_children, (str, bytes)):
        invalid_fields.append("profile_backward_grad_ready_top_child")
        return
    top_child = next((row for row in top_children if isinstance(row, Mapping)), None)
    if top_child is None:
        return

    _set_profile_metric_if_present(
        metrics,
        "profile_backward_grad_ready_top_avg_ms",
        top_child,
        "avg_ms",
        invalid_fields,
    )
    _set_profile_metric_if_present(
        metrics,
        "profile_backward_grad_ready_top_pct",
        top_child,
        "avg_pct_of_parent",
        invalid_fields,
        max_value=100.0,
    )
    _set_profile_count_metric_if_present(
        metrics,
        "profile_backward_grad_ready_top_calls",
        top_child,
        "calls",
        invalid_fields,
    )


def _add_profile_phase_metrics(metrics: Dict[str, Any], profile: Mapping[str, Any]) -> None:
    """Expose common phase timers as flat metrics for benchmark tables."""
    invalid_fields: list[str] = []
    _set_profile_metric(
        metrics,
        "profile_total_s",
        profile.get("profile_total_s", _PROFILE_METRIC_MISSING),
        invalid_fields,
    )
    for count_name in ("profile_open_phase_count", "profile_open_detail_count"):
        if count_name in profile:
            _set_profile_count_metric(metrics, count_name, profile[count_name], invalid_fields)
    phases = profile.get("phases", {})
    if not isinstance(phases, Mapping):
        _record_profile_flat_metric_invalids(metrics, invalid_fields)
        return

    forward_backward_time_s = 0.0
    forward_backward_pct = 0.0
    forward_backward_time_values = 0
    forward_backward_pct_values = 0
    for phase_name in _PROFILE_PHASE_METRIC_NAMES:
        row = phases.get(phase_name)
        if not isinstance(row, Mapping):
            continue
        time_s = _set_profile_metric(
            metrics,
            f"profile_{phase_name}_time_s",
            row.get("total_s", _PROFILE_METRIC_MISSING),
            invalid_fields,
        )
        pct = _set_profile_metric(
            metrics,
            f"profile_{phase_name}_pct",
            row.get("pct", _PROFILE_METRIC_MISSING),
            invalid_fields,
            max_value=100.0,
        )
        _set_profile_metric(
            metrics,
            f"profile_{phase_name}_avg_ms",
            row.get("avg_ms", _PROFILE_METRIC_MISSING),
            invalid_fields,
        )
        if phase_name in {"forward", "backward"}:
            if time_s is not None:
                forward_backward_time_s += time_s
                forward_backward_time_values += 1
            if pct is not None:
                forward_backward_pct += pct
                forward_backward_pct_values += 1
    if forward_backward_time_values > 0:
        metrics["profile_forward_backward_time_s"] = forward_backward_time_s
    if forward_backward_pct_values > 0:
        if forward_backward_pct <= 100.0:
            metrics["profile_forward_backward_pct"] = forward_backward_pct
        else:
            invalid_fields.append("profile_forward_backward_pct")
    _add_profile_backward_event_metrics(metrics, profile, invalid_fields)
    _record_profile_flat_metric_invalids(metrics, invalid_fields)


def _reset_device_peak_memory_stats(device: str) -> None:
    if device.startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    elif device.startswith("mps"):
        try:
            reset_mps_peak = getattr(torch.mps, "reset_peak_memory_stats", None)
            if callable(reset_mps_peak):
                reset_mps_peak()
        except Exception:
            pass


def _collect_device_memory_metrics(device: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if device.startswith("cuda"):
        try:
            metrics["cuda_current_mem_bytes"] = torch.cuda.memory_allocated(device)
            metrics["cuda_max_mem_bytes"] = torch.cuda.max_memory_allocated(device)
            metrics["cuda_reserved_mem_bytes"] = torch.cuda.memory_reserved(device)
            metrics["cuda_max_reserved_mem_bytes"] = torch.cuda.max_memory_reserved(device)
        except Exception:
            pass
    elif device.startswith("mps"):
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
    return metrics


def _profile_model_include_patterns(include: Optional[Union[str, Sequence[str]]]) -> list[str]:
    if include is None:
        return []
    if isinstance(include, str):
        return [item.strip() for item in include.split(",") if item.strip()]
    if not isinstance(include, Sequence):
        raise ValueError("profile_model_include must be a string, sequence of strings, or None")
    patterns = []
    for item in include:
        if not isinstance(item, str):
            raise ValueError("profile_model_include entries must be strings")
        pattern = item.strip()
        if pattern:
            patterns.append(pattern)
    return patterns


@dataclass
class _ProfileHookInstallResult:
    handles: list[Any]
    modules_selected: int = 0
    hook_failures: int = 0
    last_error: str = ""


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
    enable_tf32_value = _bool_setting(enable_tf32, "enable_tf32")
    cudnn_benchmark_value = _bool_setting(cudnn_benchmark, "cudnn_benchmark")
    reduced_precision_reduction_value = _bool_setting(
        reduced_precision_reduction,
        "reduced_precision_reduction",
    )
    enable_flash_sdp_value = _optional_bool_setting(enable_flash_sdp, "enable_flash_sdp")
    enable_mem_efficient_sdp_value = _optional_bool_setting(
        enable_mem_efficient_sdp,
        "enable_mem_efficient_sdp",
    )
    enable_math_sdp_value = _optional_bool_setting(enable_math_sdp, "enable_math_sdp")
    try:
        cuda_backends = getattr(getattr(torch_mod, "backends", None), "cuda", None)
        cuda_module = cuda_backends
        matmul_backend = getattr(cuda_backends, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = enable_tf32_value
        for attr in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
            if matmul_backend is not None and hasattr(matmul_backend, attr):
                setattr(matmul_backend, attr, reduced_precision_reduction_value)
        sdp_toggles = (
            ("enable_flash_sdp", enable_flash_sdp_value),
            ("enable_mem_efficient_sdp", enable_mem_efficient_sdp_value),
            ("enable_math_sdp", enable_math_sdp_value),
        )
        for fn_name, value in sdp_toggles:
            if value is None:
                continue
            fn = getattr(cuda_module, fn_name, None)
            if callable(fn):
                fn(value)
    except Exception:
        pass
    try:
        cudnn_backend = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = cudnn_benchmark_value
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
        if not isinstance(extra, list) or len(base) != len(extra):
            raise TypeError("Trigger extra batch must match the list structure of the original batch.")
        concatenated = [_concatenate_batches(b, e) for b, e in zip(base, extra)]
        return type(base)(concatenated)
    if isinstance(base, tuple):
        if not isinstance(extra, tuple) or len(base) != len(extra):
            raise TypeError("Trigger extra batch must match the tuple structure of the original batch.")
        base_namedtuple = hasattr(base, "_fields")
        extra_namedtuple = hasattr(extra, "_fields")
        if base_namedtuple != extra_namedtuple or (base_namedtuple and type(base) is not type(extra)):
            raise TypeError("Trigger extra batch must match the namedtuple structure of the original batch.")
        concatenated = [_concatenate_batches(b, e) for b, e in zip(base, extra)]
        if base_namedtuple:
            return type(base)(*concatenated)
        return tuple(concatenated)
    if base is None and extra is None:
        return None
    raise TypeError("Unsupported batch structure for trigger concatenation.")


def _new_batch_size_failure_counts() -> Dict[str, int]:
    return {reason: 0 for reason in _BATCH_SIZE_INFERENCE_FAILURE_REASONS}


def _record_batch_size_failure(counts: Dict[str, int], reason: str) -> None:
    if reason not in counts:
        reason = "unsupported_type"
    counts[reason] += 1


def _nonzero_batch_size_failure_counts(counts: Mapping[str, int]) -> Dict[str, int]:
    return {reason: count for reason, count in counts.items() if count}


def _add_batch_size_failure_metrics(metrics: Dict[str, Any], counts: Mapping[str, int]) -> None:
    metrics["batch_size_inference_failure_reasons"] = _nonzero_batch_size_failure_counts(counts)
    for reason in _BATCH_SIZE_INFERENCE_FAILURE_REASONS:
        metrics[f"batch_size_inference_{reason}_failures"] = counts.get(reason, 0)


def _distributed_sum_int(value: int, device: torch.device) -> int:
    return int(distributed_sum(torch.tensor(value, device=device)).item())


def _distributed_metric_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _distributed_sum_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=_distributed_metric_dtype(device))
    return float(distributed_sum(tensor).item())


def _distributed_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=_distributed_metric_dtype(device))
    return float(distributed_max(tensor).item())


def _non_negative_metric_value(raw: Any) -> float:
    value = _finite_profile_value(raw)
    if value is None or value < 0.0:
        return 0.0
    return value


def _apply_distributed_throughput_metrics(
    metrics: Dict[str, Any],
    *,
    samples: int,
    batches: int,
    device: torch.device,
) -> None:
    total_time_s = _distributed_max_float(
        _non_negative_metric_value(metrics.get("total_time_s")),
        device,
    )
    best_samples_per_sec = _distributed_sum_float(
        _non_negative_metric_value(metrics.get("best_samples_per_sec")),
        device,
    )
    window_samples = _distributed_sum_float(
        _non_negative_metric_value(metrics.get("window_samples")),
        device,
    )
    window_batches = _distributed_sum_float(
        _non_negative_metric_value(metrics.get("window_batches")),
        device,
    )
    window_time_s = _distributed_max_float(
        _non_negative_metric_value(metrics.get("window_time_s")),
        device,
    )
    ema_samples_per_sec = _distributed_sum_float(
        _non_negative_metric_value(metrics.get("ema_samples_per_sec")),
        device,
    )

    samples_per_sec = samples / total_time_s if total_time_s > 0.0 else 0.0
    window_samples_per_sec = (
        window_samples / window_time_s if window_time_s > 0.0 else 0.0
    )
    metrics["samples"] = float(samples)
    metrics["batches"] = float(batches)
    metrics["total_time_s"] = total_time_s
    metrics["avg_batch_s"] = total_time_s / batches if batches > 0 else 0.0
    metrics["samples_per_sec"] = samples_per_sec
    metrics["best_samples_per_sec"] = best_samples_per_sec
    metrics["headroom_ratio"] = (
        best_samples_per_sec / samples_per_sec if samples_per_sec > 0.0 else 0.0
    )
    metrics["window_samples"] = window_samples
    metrics["window_batches"] = window_batches
    metrics["window_time_s"] = window_time_s
    metrics["window_samples_per_sec"] = window_samples_per_sec
    metrics["ema_samples_per_sec"] = ema_samples_per_sec


def _try_infer_batch_size_with_reason(batch: Any) -> tuple[Optional[int], str]:
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            return None, "tensor_scalar"
        if batch.shape[0] <= 0:
            return None, "tensor_empty"
        return int(batch.shape[0]), ""
    if isinstance(batch, Mapping):
        candidate_values = []
        child_reasons = []
        for value in batch.values():
            if value is None:
                continue
            candidate, reason = _try_infer_batch_size_with_reason(value)
            if candidate is None:
                child_reasons.append(reason)
                continue
            candidate_values.append(candidate)
        if not candidate_values:
            if child_reasons and len(set(child_reasons)) == 1:
                return None, child_reasons[0]
            return None, "mapping_empty"
        if child_reasons:
            return None, "mapping_inconsistent"
        unique = set(candidate_values)
        if len(unique) != 1:
            return None, "mapping_inconsistent"
        return candidate_values[0], ""
    if isinstance(batch, (list, tuple)):
        candidate_values = []
        child_reasons = []
        for value in batch:
            if value is None:
                continue
            candidate, reason = _try_infer_batch_size_with_reason(value)
            if candidate is None:
                child_reasons.append(reason)
                continue
            candidate_values.append(candidate)
        if not candidate_values:
            length = len(batch)
            if length <= 0:
                return None, "sequence_empty"
            return length, ""
        if child_reasons:
            return None, "sequence_inconsistent"
        unique = set(candidate_values)
        if len(unique) != 1:
            return None, "sequence_inconsistent"
        return candidate_values[0], ""
    if batch is None:
        return None, "none"
    return None, "unsupported_type"


def _infer_batch_size(batch: Any) -> int:
    batch_size, reason = _try_infer_batch_size_with_reason(batch)
    if batch_size is not None:
        return batch_size
    message = _BATCH_SIZE_INFERENCE_FAILURE_MESSAGES.get(
        reason,
        _BATCH_SIZE_INFERENCE_FAILURE_MESSAGES["unsupported_type"],
    )
    if reason == "unsupported_type":
        raise TypeError(message)
    raise ValueError(message)


def _try_infer_batch_size(batch: Any) -> Optional[int]:
    batch_size, _reason = _try_infer_batch_size_with_reason(batch)
    return batch_size


def _ensure_loss_vector(loss_tensor: torch.Tensor) -> torch.Tensor:
    if loss_tensor.ndim == 0:
        return loss_tensor.unsqueeze(0)
    if loss_tensor.ndim == 1:
        return loss_tensor
    if loss_tensor.shape[0] <= 0:
        raise ValueError("Loss tensor must have a non-zero batch dimension.")
    return loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)


def _metric_to_float(value: Any) -> tuple[Optional[float], str]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None, "invalid"
        detached = value.detach()
        if detached.dtype == torch.bool or torch.is_complex(detached):
            return None, "invalid"
        try:
            if detached.numel() == 1:
                normalized = float(detached.cpu().item())
            else:
                normalized = float(detached.to(dtype=torch.float64).mean().cpu().item())
        except (RuntimeError, TypeError, ValueError):
            return None, "invalid"
    else:
        if isinstance(value, (bool, str, bytes, bytearray)):
            return None, "invalid"
        try:
            normalized = float(value)
        except Exception:
            return None, "invalid"
    if not math.isfinite(normalized):
        return None, "non_finite"
    return normalized, ""


def _user_metric_name(key: Any) -> Optional[str]:
    if not isinstance(key, str) or not key.strip():
        return None
    name = key.strip()
    if name in _USER_METRIC_RESERVED_NAMES:
        return None
    if any(name.startswith(prefix) for prefix in _USER_METRIC_RESERVED_PREFIXES):
        return None
    return name


def _detach_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return type(obj)(_detach_to_cpu(value) for value in obj)
    if isinstance(obj, tuple):
        converted_tuple = tuple(_detach_to_cpu(value) for value in obj)
        if hasattr(obj, "_fields"):
            return type(obj)(*converted_tuple)
        return tuple(converted_tuple)
    if isinstance(obj, MutableMapping):
        converted_mapping = {key: _detach_to_cpu(value) for key, value in obj.items()}
        mapping_type = cast(Any, type(obj))
        if hasattr(obj, "default_factory"):
            new_mapping = mapping_type(getattr(obj, "default_factory"))
        else:
            new_mapping = mapping_type()
        new_mapping.update(converted_mapping)
        return new_mapping
    if isinstance(obj, Mapping):
        converted_mapping = {key: _detach_to_cpu(value) for key, value in obj.items()}
        return cast(Any, type(obj))(converted_mapping)
    return obj


@dataclass
class TriggerResult:
    extra_inputs: Any = None
    extra_targets: Any = None
    weights: Optional[torch.Tensor] = None  # shape [B_total] or None


def _optimizer_setting(optimizer: Any) -> torch.optim.Optimizer:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a torch.optim.Optimizer")
    return optimizer


def _optional_scheduler_setting(scheduler: Any) -> Any:
    if scheduler is None:
        return None
    step = getattr(scheduler, "step", None)
    if not callable(step):
        raise ValueError("scheduler must provide a callable step() method")
    return scheduler


def _optional_trigger_hook_setting(trigger_hook: Any) -> Optional[Callable[[Dict[str, Any]], Optional[TriggerResult]]]:
    if trigger_hook is None:
        return None
    if not callable(trigger_hook):
        raise ValueError("trigger_hook must be callable")
    return cast(Callable[[Dict[str, Any]], Optional[TriggerResult]], trigger_hook)


def _optional_trigger_observe_setting(trigger_hook: Any) -> Optional[Callable[[Dict[str, Any]], None]]:
    observe = getattr(trigger_hook, "observe", None)
    if observe is None:
        return None
    if not callable(observe):
        raise ValueError("trigger_hook.observe must be callable")
    return cast(Callable[[Dict[str, Any]], None], observe)


def _optional_trigger_result_setting(result: Any) -> Optional[TriggerResult]:
    if result is None:
        return None
    if not isinstance(result, TriggerResult):
        raise ValueError("trigger_hook must return TriggerResult or None")
    if result.extra_inputs is not None and result.extra_targets is None:
        raise ValueError("TriggerResult.extra_targets must be provided when extra_inputs is set")
    if result.weights is not None and not isinstance(result.weights, torch.Tensor):
        raise ValueError("TriggerResult.weights must be a torch.Tensor or None")
    return result


def _optional_logger_setting(logger: Any) -> Optional[MetricsLogger]:
    if logger is None:
        return None
    log_metrics = getattr(logger, "log_metrics", None)
    if not callable(log_metrics):
        raise ValueError("logger must provide a callable log_metrics() method")
    return cast(MetricsLogger, logger)


def _optional_metrics_fn_setting(metrics_fn: Any) -> Optional[Callable[[Any, Any, Any], Any]]:
    if metrics_fn is None:
        return None
    if not callable(metrics_fn):
        raise ValueError("metrics_fn must be callable")
    return cast(Callable[[Any, Any, Any], Any], metrics_fn)


def _optional_postprocess_setting(postprocess: Any) -> Optional[Callable[[Any], Any]]:
    if postprocess is None:
        return None
    if not callable(postprocess):
        raise ValueError("postprocess must be callable")
    return cast(Callable[[Any], Any], postprocess)


def _optional_ddp_kwargs_setting(ddp_kwargs: Any) -> Dict[str, Any]:
    if ddp_kwargs is None:
        return {}
    if not isinstance(ddp_kwargs, Mapping):
        raise ValueError("ddp_kwargs must be a mapping or None")
    normalized: Dict[str, Any] = {}
    for key, value in ddp_kwargs.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("ddp_kwargs keys must be non-empty strings")
        normalized[key.strip()] = value
    return normalized


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
        model_value = _module_setting(model)
        optimizer_value = _optimizer_setting(optimizer)
        scheduler_value = _optional_scheduler_setting(scheduler)
        trigger_hook_value = _optional_trigger_hook_setting(trigger_hook)
        logger_value = _optional_logger_setting(logger)
        ddp_kwargs_value = _optional_ddp_kwargs_setting(ddp_kwargs)
        device_value = None if device is None else _device_setting(device)
        distributed_backend_value = _optional_non_empty_string_setting(
            distributed_backend,
            "distributed_backend",
        )
        distributed_value = _optional_bool_setting(distributed, "distributed")
        channels_last_value = _bool_setting(channels_last, "channels_last")
        use_compile_value = _bool_setting(use_compile, "use_compile")
        log_on_rank0_value = _bool_setting(log_on_rank0, "log_on_rank0")
        enable_tf32_value = _bool_setting(enable_tf32, "enable_tf32")
        cudnn_benchmark_value = _bool_setting(cudnn_benchmark, "cudnn_benchmark")
        reduced_precision_reduction_value = _bool_setting(
            reduced_precision_reduction,
            "reduced_precision_reduction",
        )
        enable_flash_sdp_value = _optional_bool_setting(enable_flash_sdp, "enable_flash_sdp")
        enable_mem_efficient_sdp_value = _optional_bool_setting(
            enable_mem_efficient_sdp,
            "enable_mem_efficient_sdp",
        )
        enable_math_sdp_value = _optional_bool_setting(enable_math_sdp, "enable_math_sdp")
        meter_fast_mode_value = _bool_setting(meter_fast_mode, "meter_fast_mode")

        if distributed_value is True:
            self.dist_ctx = init_distributed(backend=distributed_backend_value)
        else:
            self.dist_ctx = get_distributed_context()
        base_device = device_value or get_best_device()
        if self.dist_ctx.world_size > 1 and base_device.startswith("cuda"):
            torch.cuda.set_device(self.dist_ctx.local_rank)
            self.device = f"cuda:{self.dist_ctx.local_rank}"
        else:
            self.device = base_device
        amp_enabled, amp_dtype, use_scaler = get_amp_policy(self.device, use_amp)
        self.model = model_value.to(self.device)
        self.model = maybe_channels_last(self.model, channels_last=channels_last_value)
        self.optimizer = optimizer_value
        self.scheduler = scheduler_value
        self.grad_accum = _positive_int_setting(grad_accum, "grad_accum")
        self.clip_grad_norm = (
            None if clip_grad_norm is None else _non_negative_finite_float_setting(clip_grad_norm, "clip_grad_norm")
        )
        self.log_interval = _non_negative_int_setting(log_interval, "log_interval")
        self.trigger_hook = trigger_hook_value
        self.logger = logger_value
        self.log_on_rank0 = log_on_rank0_value
        self.meter_fast_mode = meter_fast_mode_value
        self.compile_requested = use_compile_value

        # AMP policy
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
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
            self.compile_init_time_s = _non_negative_finite_float_setting(
                time.perf_counter() - compile_started_at,
                "compile_init_time_s",
            )
        elif self.compile_requested:
            self.compile_fallback_reason = "cpu_device"
        else:
            self.compile_fallback_reason = "not_requested"

        # DDP wrap if requested and initialized
        self.using_ddp = False
        if self.dist_ctx.world_size > 1:
            if self.device.startswith("cuda"):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.dist_ctx.local_rank],
                    output_device=self.dist_ctx.local_rank,
                    **ddp_kwargs_value,
                )
            else:
                self.model = DistributedDataParallel(self.model, **ddp_kwargs_value)
            self.using_ddp = True

        # CUDA fast matmul precision
        if self.device.startswith("cuda"):
            _configure_cuda_backends(
                enable_tf32_value,
                cudnn_benchmark_value,
                reduced_precision_reduction_value,
                enable_flash_sdp_value,
                enable_mem_efficient_sdp_value,
                enable_math_sdp_value,
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
        include: Optional[Union[str, Sequence[str]]] = None,
    ) -> _ProfileHookInstallResult:
        if not profiler.enabled:
            return _ProfileHookInstallResult(handles=[])
        root = getattr(self.model, "module", self.model)
        root = getattr(root, "_orig_mod", root)
        max_depth = _positive_int_setting(depth, "profile_model_depth")
        max_selected_modules = _positive_int_setting(max_modules, "profile_model_max_modules")
        include_patterns = _profile_model_include_patterns(include)

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
            if len(selected) >= max_selected_modules:
                break

        handles: list[Any] = []
        hook_failures = 0
        hook_last_error = ""

        def record_hook_failure(exc: Exception) -> None:
            nonlocal hook_failures, hook_last_error
            hook_failures += 1
            hook_last_error = _format_exception_reason(exc)

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

        def register_forward_post_hook(module: nn.Module, hook: Callable[..., Any]) -> Any:
            try:
                return module.register_forward_hook(hook, always_call=True)
            except TypeError:
                return module.register_forward_hook(hook)

        def grad_ready(label: str) -> Callable[[torch.Tensor], torch.Tensor]:
            def hook(grad: torch.Tensor) -> torch.Tensor:
                profiler.record_event_since_start("backward", "backward_grad_ready", label)
                return grad
            return hook

        seen_params = set()
        for name, module in selected:
            label = f"model.{name}"
            pre_handle: Any = None
            try:
                pre_handle = module.register_forward_pre_hook(disable_dynamo_if_available(forward_pre(label)))
                post_handle = register_forward_post_hook(
                    module,
                    disable_dynamo_if_available(forward_post(label)),
                )
            except Exception as exc:
                record_hook_failure(exc)
                if pre_handle is not None:
                    try:
                        pre_handle.remove()
                    except Exception:
                        pass
            else:
                handles.append(pre_handle)
                handles.append(post_handle)
            for param in module.parameters(recurse=True):
                param_id = id(param)
                if param_id in seen_params or not param.requires_grad:
                    continue
                seen_params.add(param_id)
                try:
                    handles.append(param.register_hook(grad_ready(label)))
                except Exception as exc:
                    record_hook_failure(exc)
        return _ProfileHookInstallResult(
            handles=handles,
            modules_selected=len(selected),
            hook_failures=hook_failures,
            last_error=hook_last_error,
        )

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
        profile_model_include: Optional[Union[str, Sequence[str]]] = None,
        warmup_steps: int = 0,
    ) -> Dict[str, Any]:
        """
        Train for one epoch (or a fixed number of steps if steps is provided).
        Expects criterion to support reduction='mean'. If trigger_hook is set and
        you want per-sample logic, pass a criterion that supports reduction='none'.
        """
        step_limit = _optional_positive_int_setting(steps, "steps")
        warmup_step_limit = _non_negative_int_setting(warmup_steps, "warmup_steps")
        collect_profile_value = _bool_setting(collect_profile, "collect_profile")
        profile_sync_value = _bool_setting(profile_sync, "profile_sync")
        profile_distribution_value = _bool_setting(profile_distribution, "profile_distribution")
        profile_model_value = _bool_setting(profile_model, "profile_model")
        if step_limit is not None and warmup_step_limit > step_limit:
            raise ValueError("warmup_steps must be less than or equal to steps")
        profile_window_size = _positive_int_setting(profile_window, "profile_window")
        profile_model_depth_value = _positive_int_setting(profile_model_depth, "profile_model_depth")
        profile_model_max_modules_value = _positive_int_setting(
            profile_model_max_modules,
            "profile_model_max_modules",
        )
        self.model.train()
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler) and epoch is not None:
            sampler.set_epoch(epoch)
        meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        warmup_meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        steady_meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        profiler = PhaseProfiler(
            enabled=collect_profile_value,
            device=self.device,
            sync=profile_sync_value,
            track_distribution=profile_distribution_value,
            window=profile_window_size,
        )
        profile_model_enabled = collect_profile_value and profile_model_value
        if profile_model_enabled:
            profile_hook_result = self._install_profile_model_hooks(
                profiler,
                depth=profile_model_depth_value,
                max_modules=profile_model_max_modules_value,
                include=profile_model_include,
            )
        else:
            profile_hook_result = _ProfileHookInstallResult(handles=[])
        if not profile_model_value:
            profile_model_status = "not_requested"
        elif not collect_profile_value:
            profile_model_status = "collect_profile_disabled"
        elif profile_hook_result.modules_selected == 0:
            profile_model_status = "no_matching_modules"
        elif profile_hook_result.hook_failures > 0:
            profile_model_status = "hook_failures"
        else:
            profile_model_status = "ok"
        profile_hook_handles = profile_hook_result.handles

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
        _reset_device_peak_memory_stats(self.device)

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
        pending_accum_steps = 0
        partial_optimizer_steps = 0
        grad_accum_tail_steps = 0
        scheduler_step_failures = 0
        scheduler_last_error = ""
        warmup_recorded_steps = 0
        steady_recorded_steps = 0
        warmup_optimizer_steps = 0
        steady_optimizer_steps = 0
        train_failure_logged = False

        def build_train_failure_metrics(stage: str, exc: Exception) -> Dict[str, Any]:
            metrics: Dict[str, Any] = dict(meter.summary())
            metrics.update(_collect_device_memory_metrics(self.device))
            weight_value = total_weight.item()
            if weight_value > 0:
                metrics["avg_loss"] = (total_loss / total_weight).item()
            else:
                metrics["avg_loss"] = 0.0
            warmup_summary = warmup_meter.summary()
            steady_summary = steady_meter.summary()
            for key, value in warmup_summary.items():
                metrics[f"warmup_{key}"] = value
            for key, value in steady_summary.items():
                metrics[f"steady_{key}"] = value
            metrics["steps"] = step_idx
            metrics["optimizer_steps"] = optimizer_steps
            metrics["grad_accum"] = self.grad_accum
            metrics["partial_optimizer_steps"] = partial_optimizer_steps
            metrics["grad_accum_tail_steps"] = grad_accum_tail_steps
            metrics["scheduler_step_failures"] = scheduler_step_failures
            metrics["scheduler_last_error"] = scheduler_last_error
            metrics["samples"] = total_items
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
            metrics["profile_model_requested"] = profile_model_value
            metrics["profile_model_enabled"] = profile_model_enabled
            metrics["profile_model_status"] = profile_model_status
            metrics["profile_model_modules_selected"] = profile_hook_result.modules_selected
            metrics["profile_model_hook_count"] = len(profile_hook_result.handles)
            metrics["profile_model_hook_failures"] = profile_hook_result.hook_failures
            metrics["profile_model_hook_last_error"] = profile_hook_result.last_error
            metrics["device"] = self.device
            metrics["world_size"] = self.dist_ctx.world_size
            metrics["rank"] = self.dist_ctx.rank
            metrics["train_failed"] = True
            metrics["train_failure_stage"] = stage
            metrics["train_failure_last_error"] = _format_exception_reason(exc)
            if collect_profile_value:
                profile_summary = profiler.summary()
                metrics["profile"] = profile_summary
                _add_profile_phase_metrics(metrics, profile_summary)
            return metrics

        def log_train_failure(stage: str, exc: Exception) -> None:
            nonlocal train_failure_logged
            if train_failure_logged:
                return
            train_failure_logged = True
            try:
                self._log_metrics(
                    "train",
                    build_train_failure_metrics(stage, exc),
                    epoch=epoch,
                    mode="error",
                )
            except Exception:
                pass

        def rescale_accumulated_gradients(factor: float) -> None:
            if factor == 1.0:
                return
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.detach().mul_(factor)

        def run_optimizer_step(accumulated_steps: int) -> None:
            nonlocal optimizer_steps
            nonlocal partial_optimizer_steps
            nonlocal grad_accum_tail_steps
            nonlocal scheduler_step_failures
            nonlocal scheduler_last_error
            nonlocal warmup_optimizer_steps
            nonlocal steady_optimizer_steps
            profiler.start("optimizer")
            optimizer_error: Optional[Exception] = None
            try:
                if accumulated_steps < self.grad_accum:
                    partial_optimizer_steps += 1
                    grad_accum_tail_steps = accumulated_steps
                    scale_factor = self.grad_accum / max(1, accumulated_steps)
                    profiler.start_detail("optimizer", "grad_accum_rescale")
                    try:
                        rescale_accumulated_gradients(scale_factor)
                    finally:
                        profiler.stop_detail("optimizer", "grad_accum_rescale")

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
                        self.scheduler.step()
                    except Exception as exc:
                        scheduler_step_failures += 1
                        scheduler_last_error = _format_exception_reason(exc)
                    finally:
                        profiler.stop_detail("optimizer", "scheduler.step")
                optimizer_steps += 1
                if warmup_step_limit > 0 and step_idx <= warmup_step_limit:
                    warmup_optimizer_steps += 1
                else:
                    steady_optimizer_steps += 1
            except Exception as exc:
                optimizer_error = exc
                raise
            finally:
                profiler.stop("optimizer")
                if optimizer_error is not None:
                    log_train_failure("optimizer", optimizer_error)

        def profiled_batches() -> Iterable[tuple[Any, float]]:
            nonlocal step_idx
            data_iter = iter(loader)
            try:
                while step_limit is None or step_idx < step_limit:
                    step_started_at = time.perf_counter()
                    profiler.start("data_wait")
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        profiler.cancel("data_wait")
                        break
                    except Exception as exc:
                        profiler.cancel("data_wait")
                        log_train_failure("data_wait", exc)
                        raise
                    else:
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
            transfer_error: Optional[Exception] = None
            try:
                batch = to_device(batch, self.device, non_blocking=True)
            except Exception as exc:
                transfer_error = exc
                raise
            finally:
                profiler.stop("transfer")
                if transfer_error is not None:
                    log_train_failure("transfer", transfer_error)
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
                forward_error: Optional[Exception] = None
                try:
                    outputs = self.model(inputs)
                except Exception as exc:
                    forward_error = exc
                    raise
                finally:
                    profiler.stop("forward")
                    if forward_error is not None:
                        log_train_failure("forward", forward_error)

                if targets is not None and criterion is not None:
                    if supports_per_sample and self.trigger_hook is not None:
                        reduction_to_restore = None
                        if hasattr(criterion, "reduction") and getattr(criterion, "reduction") != "none":
                            reduction_to_restore = getattr(criterion, "reduction")
                            criterion.reduction = "none"
                        try:
                            # per-sample loss for trigger decisions
                            profiler.start("loss")
                            loss_error: Optional[Exception] = None
                            try:
                                loss_vec = _ensure_loss_vector(criterion(outputs, targets))
                            except Exception as exc:
                                loss_error = exc
                                raise
                            finally:
                                profiler.stop("loss")
                                if loss_error is not None:
                                    log_train_failure("loss", loss_error)
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
                        observe = _optional_trigger_observe_setting(self.trigger_hook)
                        if observe is not None:
                            try:
                                observe(trigger_ctx)
                            except Exception as exc:
                                log_train_failure("trigger_observe", exc)
                                raise
                        # Trigger may inject extra samples (e.g., hard examples)
                        profiler.start("trigger")
                        trigger_error: Optional[Exception] = None
                        try:
                            trig_result = _optional_trigger_result_setting(self.trigger_hook(trigger_ctx))
                        except Exception as exc:
                            trigger_error = exc
                            raise
                        finally:
                            profiler.stop("trigger")
                            if trigger_error is not None:
                                log_train_failure("trigger", trigger_error)
                        weights: Optional[torch.Tensor] = None
                        if trig_result is not None:
                            if trig_result.extra_inputs is not None:
                                # Concatenate and recompute outputs & loss_vec
                                profiler.start("inject_transfer")
                                inject_transfer_error: Optional[Exception] = None
                                try:
                                    extra_x = to_device(trig_result.extra_inputs, self.device, non_blocking=True)
                                    extra_y = (
                                        to_device(trig_result.extra_targets, self.device, non_blocking=True)
                                        if trig_result.extra_targets is not None
                                        else None
                                    )
                                    inputs = _concatenate_batches(inputs, extra_x)
                                    targets = _concatenate_batches(targets, extra_y)
                                except Exception as exc:
                                    inject_transfer_error = exc
                                    raise
                                finally:
                                    profiler.stop("inject_transfer")
                                    if inject_transfer_error is not None:
                                        log_train_failure("inject_transfer", inject_transfer_error)
                                profiler.start("forward")
                                forward_error = None
                                try:
                                    outputs = self.model(inputs)
                                except Exception as exc:
                                    forward_error = exc
                                    raise
                                finally:
                                    profiler.stop("forward")
                                    if forward_error is not None:
                                        log_train_failure("forward", forward_error)
                                reduction_to_restore = None
                                if hasattr(criterion, "reduction") and getattr(criterion, "reduction") != "none":
                                    reduction_to_restore = getattr(criterion, "reduction")
                                    criterion.reduction = "none"
                                try:
                                    profiler.start("loss")
                                    loss_error = None
                                    try:
                                        loss_vec = _ensure_loss_vector(criterion(outputs, targets))
                                    except Exception as exc:
                                        loss_error = exc
                                        raise
                                    finally:
                                        profiler.stop("loss")
                                        if loss_error is not None:
                                            log_train_failure("loss", loss_error)
                                finally:
                                    if reduction_to_restore is not None:
                                        criterion.reduction = reduction_to_restore
                                batch_size = loss_vec.shape[0]
                            weights = trig_result.weights

                        if weights is not None:
                            profiler.start("loss_reduce")
                            loss_reduce_error: Optional[Exception] = None
                            try:
                                w = weights.to(loss_vec.device, dtype=loss_vec.dtype)
                                if w.ndim != 1 or w.shape[0] != loss_vec.shape[0]:
                                    raise ValueError(
                                        "Trigger weights must be a 1D tensor that matches the concatenated batch size."
                                    )
                                weight_sum = w.sum()
                                weight_sum_detached = weight_sum.detach()
                                if not torch.isfinite(weight_sum_detached):
                                    raise ValueError("Trigger weights must be finite.")
                                if weight_sum_detached.item() <= 0:
                                    raise ValueError("Trigger weights must sum to a positive value.")
                                loss = (loss_vec * w).sum() / weight_sum
                                loss_weight_tensor = weight_sum_detached.to(
                                    device=total_loss.device,
                                    dtype=total_loss.dtype,
                                )
                            except Exception as exc:
                                loss_reduce_error = exc
                                raise
                            finally:
                                profiler.stop("loss_reduce")
                                if loss_reduce_error is not None:
                                    log_train_failure("loss_reduce", loss_reduce_error)
                        else:
                            profiler.start("loss_reduce")
                            loss_reduce_error = None
                            try:
                                loss = loss_vec.mean()
                                loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
                            except Exception as exc:
                                loss_reduce_error = exc
                                raise
                            finally:
                                profiler.stop("loss_reduce")
                                if loss_reduce_error is not None:
                                    log_train_failure("loss_reduce", loss_reduce_error)
                    else:
                        profiler.start("loss")
                        loss_error = None
                        try:
                            loss = criterion(outputs, targets)
                            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                                loss = loss.mean()
                        except Exception as exc:
                            loss_error = exc
                            raise
                        finally:
                            profiler.stop("loss")
                            if loss_error is not None:
                                log_train_failure("loss", loss_error)
                        reference = targets if targets is not None else inputs
                        batch_size = _infer_batch_size(reference)
                        loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
                    if batch_size is None:
                        reference = targets if targets is not None else inputs
                        batch_size = _infer_batch_size(reference)
                        if loss_weight_tensor is None:
                            loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)
                else:
                    loss_error = ValueError("No criterion provided for supervised step; supply a loss function.")
                    log_train_failure("loss", loss_error)
                    raise loss_error
                raw_loss = loss
                loss = loss / self.grad_accum

            # Backward
            profiler.start("backward")
            backward_error: Optional[Exception] = None
            try:
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except Exception as exc:
                backward_error = exc
                raise
            finally:
                profiler.stop("backward")
                if backward_error is not None:
                    log_train_failure("backward", backward_error)

            # Step if accumulation boundary
            pending_accum_steps += 1
            reached_accum_boundary = pending_accum_steps >= self.grad_accum
            reached_requested_steps = step_limit is not None and step_idx >= step_limit
            if reached_accum_boundary or reached_requested_steps:
                run_optimizer_step(pending_accum_steps)
                pending_accum_steps = 0

            # Metrics
            profiler.start("metrics")
            metrics_error: Optional[Exception] = None
            try:
                if batch_size is None:
                    reference = targets if targets is not None else inputs
                    batch_size = _infer_batch_size(reference)
                if loss_weight_tensor is None:
                    loss_weight_tensor = total_loss.new_tensor(batch_size, dtype=total_loss.dtype)

                batch_size_int = int(batch_size)
                batch_duration_s = _non_negative_finite_float_setting(
                    time.perf_counter() - step_started_at,
                    "batch_duration_s",
                )
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
            except Exception as exc:
                metrics_error = exc
                raise
            finally:
                profiler.stop("metrics")
                if metrics_error is not None:
                    log_train_failure("metrics", metrics_error)

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

        if pending_accum_steps > 0:
            run_optimizer_step(pending_accum_steps)

        metrics: Dict[str, Any] = dict(meter.summary())
        metrics.update(_collect_device_memory_metrics(self.device))
        counter_device = total_loss.device
        if self.dist_ctx.world_size > 1:
            total_loss = distributed_sum(total_loss)
            total_weight = distributed_sum(total_weight)
            warmup_loss = distributed_sum(warmup_loss)
            warmup_weight = distributed_sum(warmup_weight)
            steady_loss = distributed_sum(steady_loss)
            steady_weight = distributed_sum(steady_weight)
            total_items = _distributed_sum_int(total_items, counter_device)
            warmup_items = _distributed_sum_int(warmup_items, counter_device)
            steady_items = _distributed_sum_int(steady_items, counter_device)
            step_idx = _distributed_sum_int(step_idx, counter_device)
            optimizer_steps = _distributed_sum_int(optimizer_steps, counter_device)
            partial_optimizer_steps = _distributed_sum_int(partial_optimizer_steps, counter_device)
            grad_accum_tail_steps = _distributed_sum_int(grad_accum_tail_steps, counter_device)
            scheduler_step_failures = _distributed_sum_int(scheduler_step_failures, counter_device)
            warmup_recorded_steps = _distributed_sum_int(warmup_recorded_steps, counter_device)
            steady_recorded_steps = _distributed_sum_int(steady_recorded_steps, counter_device)
            warmup_optimizer_steps = _distributed_sum_int(warmup_optimizer_steps, counter_device)
            steady_optimizer_steps = _distributed_sum_int(steady_optimizer_steps, counter_device)
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
        warmup_summary = warmup_meter.summary()
        steady_summary = steady_meter.summary()
        if self.dist_ctx.world_size > 1:
            _apply_distributed_throughput_metrics(
                metrics,
                samples=total_items,
                batches=step_idx,
                device=counter_device,
            )
            _apply_distributed_throughput_metrics(
                warmup_summary,
                samples=warmup_items,
                batches=warmup_recorded_steps,
                device=counter_device,
            )
            _apply_distributed_throughput_metrics(
                steady_summary,
                samples=steady_items,
                batches=steady_recorded_steps,
                device=counter_device,
            )
        metrics["steps"] = step_idx
        metrics["batches"] = float(step_idx)
        metrics["optimizer_steps"] = optimizer_steps
        metrics["grad_accum"] = self.grad_accum
        metrics["partial_optimizer_steps"] = partial_optimizer_steps
        metrics["grad_accum_tail_steps"] = grad_accum_tail_steps
        metrics["scheduler_step_failures"] = scheduler_step_failures
        metrics["scheduler_last_error"] = scheduler_last_error
        metrics["samples"] = total_items
        for key, value in warmup_summary.items():
            metrics[f"warmup_{key}"] = value
        for key, value in steady_summary.items():
            metrics[f"steady_{key}"] = value
        metrics["warmup_steps"] = warmup_recorded_steps
        metrics["steady_steps"] = steady_recorded_steps
        metrics["warmup_batches"] = float(warmup_recorded_steps)
        metrics["steady_batches"] = float(steady_recorded_steps)
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
        metrics["profile_model_requested"] = profile_model_value
        metrics["profile_model_enabled"] = profile_model_enabled
        metrics["profile_model_status"] = profile_model_status
        metrics["profile_model_modules_selected"] = profile_hook_result.modules_selected
        metrics["profile_model_hook_count"] = len(profile_hook_result.handles)
        metrics["profile_model_hook_failures"] = profile_hook_result.hook_failures
        metrics["profile_model_hook_last_error"] = profile_hook_result.last_error
        metrics["device"] = self.device
        metrics["world_size"] = self.dist_ctx.world_size
        metrics["rank"] = self.dist_ctx.rank
        metrics["train_failed"] = False
        metrics["train_failure_stage"] = ""
        metrics["train_failure_last_error"] = ""
        if collect_profile_value:
            profile_summary = profiler.summary()
            metrics["profile"] = profile_summary
            _add_profile_phase_metrics(metrics, profile_summary)
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
        collect_profile: bool = False,
        profile_sync: bool = False,
        profile_distribution: bool = True,
        profile_window: int = 512,
        profile_model: bool = False,
        profile_model_depth: int = 1,
        profile_model_max_modules: int = 64,
        profile_model_include: Optional[Union[str, Sequence[str]]] = None,
        warmup_steps: int = 0,
        **loader_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train on a dataset with a minimal-parameter entrypoint."""
        collect_profile_value = _bool_setting(collect_profile, "collect_profile")
        profile_sync_value = _bool_setting(profile_sync, "profile_sync")
        profile_distribution_value = _bool_setting(profile_distribution, "profile_distribution")
        profile_model_value = _bool_setting(profile_model, "profile_model")
        loader = dataloader_from_dataset(
            dataset,
            batch_size=batch_size,
            device=self.device,
            **loader_kwargs,
        )
        return self.train_one_epoch(
            loader,
            criterion,
            steps=steps,
            epoch=epoch,
            collect_profile=collect_profile_value,
            profile_sync=profile_sync_value,
            profile_distribution=profile_distribution_value,
            profile_window=profile_window,
            profile_model=profile_model_value,
            profile_model_depth=profile_model_depth,
            profile_model_max_modules=profile_model_max_modules,
            profile_model_include=profile_model_include,
            warmup_steps=warmup_steps,
        )

    def evaluate(
        self,
        loader: Iterable[Any],
        criterion: Optional[Any] = None,
        *,
        metrics_fn: Optional[Callable[[Any, Any, Any], Dict[str, Any]]] = None,
        steps: Optional[int] = None,
        epoch: Optional[int] = None,
        collect_profile: bool = False,
        profile_sync: bool = False,
        profile_distribution: bool = True,
        profile_window: int = 512,
    ) -> Dict[str, Any]:
        """Run a validation/evaluation loop without gradient updates."""
        step_limit = _optional_positive_int_setting(steps, "steps")
        profile_window_size = _positive_int_setting(profile_window, "profile_window")
        collect_profile_value = _bool_setting(collect_profile, "collect_profile")
        profile_sync_value = _bool_setting(profile_sync, "profile_sync")
        profile_distribution_value = _bool_setting(profile_distribution, "profile_distribution")
        metrics_fn_value = _optional_metrics_fn_setting(metrics_fn)
        self.model.eval()
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler) and epoch is not None:
            sampler.set_epoch(epoch)
        meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        profiler = PhaseProfiler(
            enabled=collect_profile_value,
            device=self.device,
            sync=profile_sync_value,
            track_distribution=profile_distribution_value,
            window=profile_window_size,
        )
        _reset_device_peak_memory_stats(self.device)
        metric_dtype = torch.float32 if self.device.startswith("mps") else torch.float64
        total_loss = torch.zeros((), device=self.device, dtype=metric_dtype)
        total_weight = torch.zeros((), device=self.device, dtype=metric_dtype)
        total_items = 0
        step_idx = 0
        measured_steps = 0
        batch_size_inference_failures = 0
        batch_size_failure_counts = _new_batch_size_failure_counts()
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}
        user_metric_valid_count = 0
        user_metric_invalid_count = 0
        user_metric_non_finite_count = 0
        user_metric_unmeasured_count = 0
        metrics_fn_requested = metrics_fn_value is not None
        metrics_fn_calls = 0
        metrics_fn_failures = 0
        metrics_fn_last_error = ""
        eval_failure_logged = False

        def build_eval_metrics(
            *,
            failed: bool = False,
            failure_stage: str = "",
            failure_last_error: str = "",
        ) -> Dict[str, Any]:
            metrics: Dict[str, Any] = dict(meter.summary())
            metrics.update(_collect_device_memory_metrics(self.device))
            weight_value = total_weight.item()
            if weight_value > 0:
                metrics["avg_loss"] = (total_loss / total_weight).item()
            else:
                metrics["avg_loss"] = 0.0
            metrics["steps"] = step_idx
            metrics["measured_steps"] = measured_steps
            metrics["batches"] = float(measured_steps)
            metrics["unmeasured_steps"] = step_idx - measured_steps
            metrics["batch_size_inference_failures"] = batch_size_inference_failures
            _add_batch_size_failure_metrics(metrics, batch_size_failure_counts)
            metrics["samples"] = total_items
            if not failed and self.dist_ctx.world_size > 1:
                _apply_distributed_throughput_metrics(
                    metrics,
                    samples=total_items,
                    batches=measured_steps,
                    device=torch.device(self.device),
                )
                metrics["samples"] = total_items
            metrics["reported_samples_per_sec"] = metrics["samples_per_sec"]
            metrics["device"] = self.device
            metrics["world_size"] = self.dist_ctx.world_size
            metrics["rank"] = self.dist_ctx.rank
            metrics["metrics_fn_requested"] = metrics_fn_requested
            metrics["metrics_fn_calls"] = metrics_fn_calls
            metrics["metrics_fn_successes"] = metrics_fn_calls - metrics_fn_failures
            metrics["metrics_fn_failures"] = metrics_fn_failures
            metrics["metrics_fn_last_error"] = metrics_fn_last_error
            metrics["eval_failed"] = failed
            metrics["eval_failure_stage"] = failure_stage
            metrics["eval_failure_last_error"] = failure_last_error
            metrics["user_metric_valid_count"] = user_metric_valid_count
            metrics["user_metric_invalid_count"] = user_metric_invalid_count
            metrics["user_metric_non_finite_count"] = user_metric_non_finite_count
            metrics["user_metric_unmeasured_count"] = user_metric_unmeasured_count
            metrics["user_metric_skipped_count"] = (
                user_metric_invalid_count
                + user_metric_non_finite_count
                + user_metric_unmeasured_count
            )

            for key, total in metric_sums.items():
                denom = metric_weights.get(key, 0.0)
                metrics[key] = total / denom if denom else 0.0
            if collect_profile_value:
                profile_summary = profiler.summary()
                metrics["profile"] = profile_summary
                _add_profile_phase_metrics(metrics, profile_summary)
            return metrics

        def log_eval_failure(stage: str, exc: Exception) -> None:
            nonlocal eval_failure_logged
            if eval_failure_logged:
                return
            eval_failure_logged = True
            try:
                self._log_metrics(
                    "eval",
                    build_eval_metrics(
                        failed=True,
                        failure_stage=stage,
                        failure_last_error=_format_exception_reason(exc),
                    ),
                    epoch=epoch,
                    mode="error",
                )
            except Exception:
                pass

        with torch.no_grad():
            data_iter = iter(loader)
            while step_limit is None or step_idx < step_limit:
                step_started_at = time.perf_counter()
                profiler.start("data_wait")
                try:
                    batch = next(data_iter)
                except StopIteration:
                    profiler.cancel("data_wait")
                    break
                except Exception as exc:
                    profiler.cancel("data_wait")
                    log_eval_failure("data_wait", exc)
                    raise
                else:
                    profiler.stop("data_wait")
                step_idx += 1
                profiler.start("transfer")
                transfer_error: Optional[Exception] = None
                try:
                    batch = to_device(batch, self.device, non_blocking=True)
                except Exception as exc:
                    transfer_error = exc
                    raise
                finally:
                    profiler.stop("transfer")
                    if transfer_error is not None:
                        log_eval_failure("transfer", transfer_error)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                elif isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
                    inputs, targets = batch["inputs"], batch["targets"]
                else:
                    inputs, targets = batch, None

                with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                    profiler.start("forward")
                    forward_error: Optional[Exception] = None
                    try:
                        outputs = self.model(inputs)
                    except Exception as exc:
                        forward_error = exc
                        raise
                    finally:
                        profiler.stop("forward")
                        if forward_error is not None:
                            log_eval_failure("forward", forward_error)
                    if criterion is not None and targets is not None:
                        profiler.start("loss")
                        loss_error: Optional[Exception] = None
                        try:
                            loss = criterion(outputs, targets)
                            if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                                loss = loss.mean()
                        except Exception as exc:
                            loss_error = exc
                            raise
                        finally:
                            profiler.stop("loss")
                            if loss_error is not None:
                                log_eval_failure("loss", loss_error)
                    else:
                        loss = None
                reference = targets if targets is not None else inputs
                batch_size, batch_size_failure_reason = _try_infer_batch_size_with_reason(reference)
                batch_size_int = int(batch_size) if batch_size is not None else None

                extra_metrics: Any = None
                if metrics_fn_value is not None:
                    profiler.start("user_metrics")
                    try:
                        metrics_fn_calls += 1
                        extra_metrics = metrics_fn_value(outputs, targets, inputs)
                    except Exception as exc:
                        metrics_fn_failures += 1
                        metrics_fn_last_error = _format_exception_reason(exc)
                        log_eval_failure("user_metrics", exc)
                        raise
                    finally:
                        profiler.stop("user_metrics")

                profiler.start("metrics")
                metrics_error: Optional[Exception] = None
                try:
                    batch_duration_s = _non_negative_finite_float_setting(
                        time.perf_counter() - step_started_at,
                        "batch_duration_s",
                    )
                    if batch_size_int is not None:
                        meter.record(batch_duration_s, batch_size_int)
                        total_items += batch_size_int
                        measured_steps += 1
                    else:
                        batch_size_inference_failures += 1
                        _record_batch_size_failure(batch_size_failure_counts, batch_size_failure_reason)

                    if loss is not None and batch_size_int is not None:
                        loss_detached = loss.detach().to(device=total_loss.device, dtype=total_loss.dtype)
                        weight_tensor = total_weight.new_tensor(batch_size_int, dtype=total_weight.dtype)
                        total_loss += loss_detached * weight_tensor
                        total_weight += weight_tensor

                    if extra_metrics is not None:
                        if not isinstance(extra_metrics, Mapping):
                            user_metric_invalid_count += 1
                        else:
                            for raw_key, value in extra_metrics.items():
                                key = _user_metric_name(raw_key)
                                if key is None:
                                    user_metric_invalid_count += 1
                                    continue
                                metric_value, metric_error = _metric_to_float(value)
                                if metric_value is None:
                                    if metric_error == "non_finite":
                                        user_metric_non_finite_count += 1
                                    else:
                                        user_metric_invalid_count += 1
                                    continue
                                if batch_size_int is None:
                                    user_metric_unmeasured_count += 1
                                    continue
                                metric_sums[key] = metric_sums.get(key, 0.0) + metric_value * batch_size_int
                                metric_weights[key] = metric_weights.get(key, 0.0) + batch_size_int
                                user_metric_valid_count += 1
                except Exception as exc:
                    metrics_error = exc
                    raise
                finally:
                    profiler.stop("metrics")
                    if metrics_error is not None:
                        log_eval_failure("metrics", metrics_error)

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
            step_idx = int(distributed_sum(torch.tensor(step_idx, device=total_loss.device)).item())
            measured_steps = int(distributed_sum(torch.tensor(measured_steps, device=total_loss.device)).item())
            batch_size_inference_failures = int(
                distributed_sum(torch.tensor(batch_size_inference_failures, device=total_loss.device)).item()
            )
            for reason in list(batch_size_failure_counts.keys()):
                batch_size_failure_counts[reason] = int(
                    distributed_sum(torch.tensor(batch_size_failure_counts[reason], device=total_loss.device)).item()
                )
            metrics_fn_calls = int(
                distributed_sum(torch.tensor(metrics_fn_calls, device=total_loss.device)).item()
            )
            metrics_fn_failures = int(
                distributed_sum(torch.tensor(metrics_fn_failures, device=total_loss.device)).item()
            )
            user_metric_valid_count = int(
                distributed_sum(torch.tensor(user_metric_valid_count, device=total_loss.device)).item()
            )
            user_metric_invalid_count = int(
                distributed_sum(torch.tensor(user_metric_invalid_count, device=total_loss.device)).item()
            )
            user_metric_non_finite_count = int(
                distributed_sum(torch.tensor(user_metric_non_finite_count, device=total_loss.device)).item()
            )
            user_metric_unmeasured_count = int(
                distributed_sum(torch.tensor(user_metric_unmeasured_count, device=total_loss.device)).item()
            )
            for key in list(metric_sums.keys()):
                metric_sums[key] = float(
                    distributed_sum(torch.tensor(metric_sums[key], device=total_loss.device)).item()
                )
                metric_weights[key] = float(
                    distributed_sum(torch.tensor(metric_weights[key], device=total_loss.device)).item()
                )
        metrics = build_eval_metrics()
        self._log_metrics("eval", metrics, epoch=epoch, mode="epoch")
        return metrics

    def predict(
        self,
        loader: Iterable[Any],
        *,
        steps: Optional[int] = None,
        postprocess: Optional[Callable[[Any], Any]] = None,
        collect_profile: bool = False,
        profile_sync: bool = False,
        profile_distribution: bool = True,
        profile_window: int = 512,
        return_metrics: bool = False,
    ) -> Union[list[Any], tuple[list[Any], Dict[str, Any]]]:
        """Run inference and collect outputs on CPU.

        When ``return_metrics`` or ``collect_profile`` is true, returns
        ``(predictions, metrics)``. The default return value remains the
        prediction list for backward compatibility.
        """
        step_limit = _optional_positive_int_setting(steps, "steps")
        profile_window_size = _positive_int_setting(profile_window, "profile_window")
        collect_profile_value = _bool_setting(collect_profile, "collect_profile")
        profile_sync_value = _bool_setting(profile_sync, "profile_sync")
        profile_distribution_value = _bool_setting(profile_distribution, "profile_distribution")
        return_metrics_value = _bool_setting(return_metrics, "return_metrics")
        postprocess_value = _optional_postprocess_setting(postprocess)
        metrics_requested = return_metrics_value or collect_profile_value
        self.model.eval()
        outputs_list: list[Any] = []
        meter = ThroughputMeter(fast_mode=self.meter_fast_mode)
        profiler = PhaseProfiler(
            enabled=collect_profile_value,
            device=self.device,
            sync=profile_sync_value,
            track_distribution=profile_distribution_value,
            window=profile_window_size,
        )
        if metrics_requested:
            _reset_device_peak_memory_stats(self.device)
        total_items = 0
        step_idx = 0
        measured_steps = 0
        batch_size_inference_failures = 0
        batch_size_failure_counts = _new_batch_size_failure_counts()
        postprocess_requested = postprocess_value is not None
        postprocess_calls = 0
        postprocess_failures = 0
        postprocess_last_error = ""
        predict_failure_logged = False

        def build_predict_metrics(
            *,
            failed: bool = False,
            failure_stage: str = "",
            failure_last_error: str = "",
        ) -> Dict[str, Any]:
            metrics: Dict[str, Any] = dict(meter.summary())
            metrics.update(_collect_device_memory_metrics(self.device))
            metrics["steps"] = step_idx
            metrics["measured_steps"] = measured_steps
            metrics["batches"] = float(measured_steps)
            metrics["unmeasured_steps"] = step_idx - measured_steps
            metrics["batch_size_inference_failures"] = batch_size_inference_failures
            _add_batch_size_failure_metrics(metrics, batch_size_failure_counts)
            metrics["samples"] = total_items
            if not failed and self.dist_ctx.world_size > 1:
                _apply_distributed_throughput_metrics(
                    metrics,
                    samples=total_items,
                    batches=measured_steps,
                    device=torch.device(self.device),
                )
                metrics["samples"] = total_items
            metrics["reported_samples_per_sec"] = metrics["samples_per_sec"]
            metrics["device"] = self.device
            metrics["world_size"] = self.dist_ctx.world_size
            metrics["rank"] = self.dist_ctx.rank
            metrics["postprocess_requested"] = postprocess_requested
            metrics["postprocess_calls"] = postprocess_calls
            metrics["postprocess_successes"] = postprocess_calls - postprocess_failures
            metrics["postprocess_failures"] = postprocess_failures
            metrics["postprocess_last_error"] = postprocess_last_error
            metrics["predict_failed"] = failed
            metrics["predict_failure_stage"] = failure_stage
            metrics["predict_failure_last_error"] = failure_last_error
            if collect_profile_value:
                profile_summary = profiler.summary()
                metrics["profile"] = profile_summary
                _add_profile_phase_metrics(metrics, profile_summary)
            return metrics

        def log_predict_failure(stage: str, exc: Exception) -> None:
            nonlocal predict_failure_logged
            if predict_failure_logged or not metrics_requested:
                return
            predict_failure_logged = True
            try:
                self._log_metrics(
                    "predict",
                    build_predict_metrics(
                        failed=True,
                        failure_stage=stage,
                        failure_last_error=_format_exception_reason(exc),
                    ),
                    mode="error",
                )
            except Exception:
                pass

        with torch.no_grad():
            data_iter = iter(loader)
            while step_limit is None or step_idx < step_limit:
                step_started_at = time.perf_counter()
                profiler.start("data_wait")
                try:
                    batch = next(data_iter)
                except StopIteration:
                    profiler.cancel("data_wait")
                    break
                except Exception as exc:
                    profiler.cancel("data_wait")
                    log_predict_failure("data_wait", exc)
                    raise
                else:
                    profiler.stop("data_wait")
                step_idx += 1
                profiler.start("transfer")
                transfer_error: Optional[Exception] = None
                try:
                    batch = to_device(batch, self.device, non_blocking=True)
                except Exception as exc:
                    transfer_error = exc
                    raise
                finally:
                    profiler.stop("transfer")
                    if transfer_error is not None:
                        log_predict_failure("transfer", transfer_error)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs = batch[0]
                elif isinstance(batch, dict) and "inputs" in batch:
                    inputs = batch["inputs"]
                else:
                    inputs = batch
                with autocast_ctx(self.device, self.amp_enabled, self.amp_dtype):
                    profiler.start("forward")
                    forward_error: Optional[Exception] = None
                    try:
                        outputs = self.model(inputs)
                    except Exception as exc:
                        forward_error = exc
                        raise
                    finally:
                        profiler.stop("forward")
                        if forward_error is not None:
                            log_predict_failure("forward", forward_error)
                if postprocess_value is not None:
                    profiler.start("postprocess")
                    postprocess_error: Optional[Exception] = None
                    try:
                        postprocess_calls += 1
                        outputs = postprocess_value(outputs)
                    except Exception as exc:
                        postprocess_error = exc
                        postprocess_failures += 1
                        postprocess_last_error = _format_exception_reason(exc)
                        raise
                    finally:
                        profiler.stop("postprocess")
                        if postprocess_error is not None:
                            log_predict_failure("postprocess", postprocess_error)
                batch_size, batch_size_failure_reason = (
                    _try_infer_batch_size_with_reason(inputs) if metrics_requested else (None, "")
                )
                profiler.start("collect_output")
                collect_output_error: Optional[Exception] = None
                try:
                    outputs_list.append(_detach_to_cpu(outputs))
                except Exception as exc:
                    collect_output_error = exc
                    raise
                finally:
                    profiler.stop("collect_output")
                    if collect_output_error is not None:
                        log_predict_failure("collect_output", collect_output_error)
                if metrics_requested:
                    profiler.start("metrics")
                    metrics_error: Optional[Exception] = None
                    try:
                        if batch_size is not None:
                            batch_size_int = int(batch_size)
                            batch_duration_s = _non_negative_finite_float_setting(
                                time.perf_counter() - step_started_at,
                                "batch_duration_s",
                            )
                            meter.record(batch_duration_s, batch_size_int)
                            total_items += batch_size_int
                            measured_steps += 1
                        else:
                            batch_size_inference_failures += 1
                            _record_batch_size_failure(batch_size_failure_counts, batch_size_failure_reason)
                    except Exception as exc:
                        metrics_error = exc
                        raise
                    finally:
                        profiler.stop("metrics")
                        if metrics_error is not None:
                            log_predict_failure("metrics", metrics_error)
        if not metrics_requested:
            return outputs_list
        if self.dist_ctx.world_size > 1:
            counter_device = torch.device(self.device)
            step_idx = int(distributed_sum(torch.tensor(step_idx, device=counter_device)).item())
            total_items = int(distributed_sum(torch.tensor(total_items, device=counter_device)).item())
            measured_steps = int(distributed_sum(torch.tensor(measured_steps, device=counter_device)).item())
            batch_size_inference_failures = int(
                distributed_sum(torch.tensor(batch_size_inference_failures, device=counter_device)).item()
            )
            for reason in list(batch_size_failure_counts.keys()):
                batch_size_failure_counts[reason] = int(
                    distributed_sum(torch.tensor(batch_size_failure_counts[reason], device=counter_device)).item()
                )
            postprocess_calls = int(
                distributed_sum(torch.tensor(postprocess_calls, device=counter_device)).item()
            )
            postprocess_failures = int(
                distributed_sum(torch.tensor(postprocess_failures, device=counter_device)).item()
            )
        metrics = build_predict_metrics()
        self._log_metrics("predict", metrics, mode="epoch")
        return outputs_list, metrics
