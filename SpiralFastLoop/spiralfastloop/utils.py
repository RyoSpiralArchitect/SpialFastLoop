# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

import math
import operator
import os
import sys
import time
from collections import deque
from collections.abc import Mapping, MutableMapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset

SampleWindow = Union[deque[float], Tuple[()], list[float]]


@dataclass(frozen=True)
class CompileResult:
    model: nn.Module
    compiled: bool
    fallback_reason: str = ""


def get_best_device() -> str:
    """Pick the best available device among CUDA, MPS, CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class DistributedContext:
    is_initialized: bool
    rank: int
    world_size: int
    local_rank: int
    backend: Optional[str]

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_non_negative_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value >= 0 else default


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default


def _non_empty_string_setting(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_non_empty_string_setting(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _non_empty_string_setting(value, name)


def get_distributed_context() -> DistributedContext:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = _env_non_negative_int("LOCAL_RANK", rank)
        backend = torch.distributed.get_backend()
        return DistributedContext(
            is_initialized=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=backend,
        )
    rank = _env_non_negative_int("RANK", 0)
    world_size = _env_positive_int("WORLD_SIZE", 1)
    local_rank = _env_non_negative_int("LOCAL_RANK", rank)
    return DistributedContext(
        is_initialized=False,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=None,
    )


def init_distributed(
    *,
    backend: Optional[str] = None,
    init_method: str = "env://",
) -> DistributedContext:
    backend_value = _optional_non_empty_string_setting(backend, "backend")
    init_method_value = _non_empty_string_setting(init_method, "init_method")
    if not torch.distributed.is_available():
        return get_distributed_context()
    if torch.distributed.is_initialized():
        return get_distributed_context()
    world_size = _env_positive_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return get_distributed_context()
    if backend_value is None:
        backend_value = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend_value, init_method=init_method_value)
    return get_distributed_context()

AmpSetting = Union[bool, Literal["auto"], None]


def _device_setting(device: Any, name: str = "device") -> str:
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"{name} must be a non-empty device string")
    return device.strip()


def _compile_mode_setting(mode: Any) -> str:
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("mode must be a non-empty string")
    return mode.strip()


def _module_setting(model: Any, name: str = "model") -> nn.Module:
    if not isinstance(model, nn.Module):
        raise ValueError(f"{name} must be a torch.nn.Module")
    return model


def _device_type(device: Any) -> str:
    return _device_setting(device).split(":", 1)[0]


def _bool_setting(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean")


def _optional_bool_setting(value: Any, name: str) -> Optional[bool]:
    if value is None:
        return None
    return _bool_setting(value, name)


def _int_setting(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _positive_int_setting(value: Any, name: str) -> int:
    normalized = _int_setting(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _non_negative_int_setting(value: Any, name: str) -> int:
    normalized = _int_setting(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _optional_positive_int_setting(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _positive_int_setting(value, name)


def _finite_float_setting(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _non_negative_finite_float_setting(value: Any, name: str) -> float:
    normalized = _finite_float_setting(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return normalized


def _strict_finite_float_setting(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _profile_name_setting(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _time_value_setting(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(f"{name} must return a finite number")
    try:
        normalized = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must return a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must return a finite number")
    return normalized


def get_amp_policy(device: str, use_amp: AmpSetting = "auto") -> Tuple[bool, torch.dtype, bool]:
    """
    Decide AMP usage, dtype, and whether GradScaler should be used.

    Returns:
        enabled: bool
        amp_dtype: torch.dtype
        use_scaler: bool  # GradScaler only for CUDA
    """
    device_type = _device_type(device)
    if use_amp is None:
        use_amp = "auto"
    elif isinstance(use_amp, bool):
        if not use_amp:
            return False, torch.float32, False
    elif use_amp != "auto":
        raise ValueError("use_amp must be a boolean, 'auto', or None")

    if device_type == "cuda":
        # Prefer bf16 on Ampere+ (TF32/bf16), else fp16
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return True, torch.bfloat16, True
        else:
            return True, torch.float16, True
    elif device_type == "mps":
        # MPS AMP is fp16; GradScaler is not used on MPS.
        return True, torch.float16, False
    else:
        return False, torch.float32, False

def autocast_ctx(device: str, enabled: bool, amp_dtype: torch.dtype) -> AbstractContextManager[Any]:
    enabled_value = _bool_setting(enabled, "enabled")
    if not enabled_value:
        return nullcontext()
    return cast(AbstractContextManager[Any], torch.autocast(device_type=_device_type(device), dtype=amp_dtype))


def to_device(obj: Any, device: str, non_blocking: bool = True) -> Any:
    """Recursively move tensors (and nested structures) to device."""
    non_blocking_value = _bool_setting(non_blocking, "non_blocking")
    device_value = _device_setting(device)
    return _to_device(obj, device_value, non_blocking_value)


def _to_device(obj: Any, device: str, non_blocking: bool) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, list):
        return type(obj)(_to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, tuple):
        converted_tuple = tuple(_to_device(x, device, non_blocking) for x in obj)
        if hasattr(obj, "_fields"):
            return type(obj)(*converted_tuple)
        return type(obj)(converted_tuple)
    if isinstance(obj, MutableMapping):
        converted_mapping = {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
        mapping_type = cast(Any, type(obj))
        if hasattr(obj, "default_factory"):
            default_factory = getattr(obj, "default_factory")
            new_mapping = mapping_type(default_factory)
        else:
            new_mapping = mapping_type()
        new_mapping.update(converted_mapping)
        return new_mapping
    if isinstance(obj, Mapping):
        converted_mapping = {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
        mapping_type = cast(Any, type(obj))
        return mapping_type(converted_mapping)
    return obj


def _torch_default_device() -> Any:
    get_default_device = getattr(torch, "get_default_device", None)
    if get_default_device is None:
        return "cpu"
    return get_default_device()


def _seeded_dataloader_generator(seed: int) -> torch.Generator:
    default_device = _torch_default_device()
    try:
        generator = torch.Generator(device=default_device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def dataloader_from_dataset(
    dataset: Dataset[Any],
    batch_size: int,
    device: str,
    num_workers: Optional[int] = None,
    prefetch_factor: int = 2,
    persistent: bool = True,
    pin_memory: Optional[bool] = None,
    shuffle: bool = True,
    distributed: bool = False,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Create a DataLoader with sensible performance defaults."""
    batch_size = _positive_int_setting(batch_size, "batch_size")
    prefetch_factor = _positive_int_setting(prefetch_factor, "prefetch_factor")
    seed = _int_setting(seed, "seed")
    persistent_value = _bool_setting(persistent, "persistent")
    pin_memory_value = _optional_bool_setting(pin_memory, "pin_memory")
    shuffle_value = _bool_setting(shuffle, "shuffle")
    distributed_value = _bool_setting(distributed, "distributed")
    drop_last_value = _bool_setting(drop_last, "drop_last")
    device_value = _device_setting(device)
    resolved_device = get_best_device() if device_value == "auto" else device_value
    workers = num_workers
    if workers is None:
        try:
            cpu_count = os.cpu_count()
        except Exception:
            cpu_count = None
        if cpu_count is None:
            workers = 2
        else:
            workers = max(2, cpu_count // 2)
    else:
        workers = _non_negative_int_setting(workers, "num_workers")
    if pin_memory_value is None:
        pin_memory_value = (_device_type(resolved_device) == "cuda")
    sampler: Optional[DistributedSampler[Any]] = None
    if distributed_value:
        ctx = get_distributed_context()
        if ctx.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=shuffle_value,
                seed=seed,
                drop_last=drop_last_value,
            )
            shuffle_value = False
    generator = _seeded_dataloader_generator(seed)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "generator": generator,
        "shuffle": shuffle_value,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory_value,
        "drop_last": drop_last_value,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_value
    return DataLoader(dataset, **loader_kwargs)


def distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    tensor = value.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def distributed_max(value: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    tensor = value.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor


def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    tensor = distributed_sum(value)
    ctx = get_distributed_context()
    if ctx.world_size > 1:
        tensor = tensor / ctx.world_size
    return tensor

class _PSquareQuantile:
    __slots__ = ("quantile", "_initial", "_q", "_n", "_np", "_dn")

    """Streaming percentile estimator using the P² algorithm."""

    def __init__(self, quantile: float) -> None:
        quantile_value = _strict_finite_float_setting(quantile, "quantile")
        if not (0.0 < quantile_value < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = quantile_value
        self._initial: list[float] = []
        self._q: Optional[list[float]] = None
        self._n: Optional[list[int]] = None
        self._np: Optional[list[float]] = None
        self._dn: Optional[list[float]] = None

    def _state(self) -> Optional[tuple[list[float], list[int], list[float], list[float]]]:
        q_values = self._q
        positions = self._n
        desired = self._np
        increments = self._dn
        if q_values is None and positions is None and desired is None and increments is None:
            return None
        if q_values is None or positions is None or desired is None or increments is None:
            raise RuntimeError("P² quantile estimator state is inconsistent.")
        return q_values, positions, desired, increments

    def _initialized_state(self) -> tuple[list[float], list[int], list[float], list[float]]:
        state = self._state()
        if state is None:
            raise RuntimeError("P² quantile estimator is not initialized.")
        return state

    def add(self, value: float) -> None:
        value = _strict_finite_float_setting(value, "value")

        state = self._state()
        if state is None:
            initial = self._initial
            initial.append(value)
            if len(initial) == 5:
                initial.sort()
                q_values = initial.copy()
                q = self.quantile
                positions = [i + 1 for i in range(5)]
                desired = [
                    1.0,
                    1.0 + 2.0 * q,
                    1.0 + 4.0 * q,
                    3.0 + 2.0 * q,
                    5.0,
                ]
                increments = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]
                self._q, self._n, self._np, self._dn = q_values, positions, desired, increments
            return
        q_values, positions, desired, increments = state

        if value < q_values[0]:
            q_values[0] = value
            k = 0
        elif value >= q_values[4]:
            q_values[4] = value
            k = 3
        else:
            k = 0
            while k < 3 and value >= q_values[k + 1]:
                k += 1

        for i in range(k + 1, 5):
            positions[i] += 1

        for i in range(5):
            desired[i] += increments[i]

        for i in range(1, 4):
            d = desired[i] - positions[i]
            if (d >= 1.0 and positions[i + 1] - positions[i] > 1) or (d <= -1.0 and positions[i - 1] - positions[i] < -1):
                step = 1 if d > 0 else -1
                candidate = self._parabolic_update(i, step)
                lower = q_values[i - 1]
                upper = q_values[i + 1]
                if lower < candidate < upper:
                    q_values[i] = candidate
                else:
                    q_values[i] = self._linear_update(i, step)
                positions[i] += step

    def value(self) -> float:
        if self._q is not None:
            return float(self._q[2])
        if not self._initial:
            return 0.0
        ordered = sorted(self._initial)
        if len(ordered) == 1:
            return float(ordered[0])
        index = int(round(self.quantile * (len(ordered) - 1)))
        index = max(0, min(len(ordered) - 1, index))
        return float(ordered[index])

    def _parabolic_update(self, idx: int, step: int) -> float:
        q_values, positions, _, _ = self._initialized_state()
        numerator = step * (
            (positions[idx] - positions[idx - 1] + step) * (q_values[idx + 1] - q_values[idx]) / (positions[idx + 1] - positions[idx])
            + (positions[idx + 1] - positions[idx] - step) * (q_values[idx] - q_values[idx - 1]) / (positions[idx] - positions[idx - 1])
        )
        denominator = positions[idx + 1] - positions[idx - 1]
        if denominator == 0:
            return q_values[idx]
        return q_values[idx] + numerator / denominator

    def _linear_update(self, idx: int, step: int) -> float:
        q_values, positions, _, _ = self._initialized_state()
        neighbour = idx + step
        denominator = positions[neighbour] - positions[idx]
        if denominator == 0:
            return q_values[idx]
        return q_values[idx] + step * (q_values[neighbour] - q_values[idx]) / denominator


class ThroughputMeter:
    __slots__ = (
        "_time_fn",
        "_smoothing",
        "_window_limit",
        "_window_records",
        "_window_duration",
        "_window_samples",
        "_window_batches",
        "last",
        "samples",
        "_total_time",
        "_time_correction",
        "_batches",
        "_median",
        "_p95",
        "_p99",
        "_last_duration",
        "_min_duration",
        "_max_duration",
        "_mean_duration",
        "_m2_duration",
        "_ema_throughput",
        "_track_distribution",
        "_track_window",
        "_best_time_per_sample",
        "_fast_mode",
    )

    """Measure batch latencies and throughput with streaming quantile estimates.

    Set ``track_distribution=False`` to avoid P² percentile maintenance when you
    only need aggregate throughput numbers, reducing Python overhead inside
    tight training loops.

    Set ``track_window=False`` to skip the moving window book-keeping when you
    only care about global throughput.

    Set ``window=0`` to disable moving-window stats while retaining streaming
    percentile tracking.
    """

    class _BatchTimer(AbstractContextManager["ThroughputMeter._BatchTimer"]):
        __slots__ = ("_meter", "_batch_size", "_record_on_exception", "_start")

        def __init__(
            self,
            meter: "ThroughputMeter",
            batch_size: int,
            *,
            record_on_exception: bool,
        ) -> None:
            self._meter = meter
            self._batch_size = _positive_int_setting(batch_size, "batch_size")
            self._record_on_exception = _bool_setting(record_on_exception, "record_on_exception")
            self._start: Optional[float] = None

        def __enter__(self) -> "ThroughputMeter._BatchTimer":
            self._start = self._meter._now()
            return self

        def __exit__(
            self,
            exc_type: Optional[type[BaseException]],
            exc: Optional[BaseException],
            tb: Any,
        ) -> Literal[False]:
            end = self._meter._now()
            if self._start is None:
                return False
            should_record = exc_type is None or self._record_on_exception
            elapsed = end - self._start
            if not should_record:
                if elapsed >= 0.0:
                    self._meter.last = end
                return False
            duration = _non_negative_finite_float_setting(elapsed, "duration_s")
            self._meter.record(duration, self._batch_size)
            self._meter.last = end
            return False

    def __init__(
        self,
        *,
        time_fn: Optional[Callable[[], float]] = None,
        smoothing: Optional[float] = 0.2,
        window: int = 32,
        track_distribution: bool = True,
        track_window: bool = True,
        fast_mode: bool = False,
    ) -> None:
        fast_mode_value = _bool_setting(fast_mode, "fast_mode")
        if fast_mode_value:
            track_distribution_value = False
            track_window_value = False
            smoothing_value = None
        else:
            track_distribution_value = _bool_setting(
                track_distribution,
                "track_distribution",
            )
            track_window_value = _bool_setting(track_window, "track_window")
            smoothing_value = None
            if smoothing is not None:
                smoothing_value = _finite_float_setting(smoothing, "smoothing")
            if smoothing_value is not None and not (0.0 < smoothing_value <= 1.0):
                raise ValueError("smoothing must be in the interval (0, 1].")
        window_int = _non_negative_int_setting(window, "window")
        track_window_value = track_window_value and window_int > 0
        self._track_distribution = track_distribution_value
        self._track_window = track_window_value
        self._fast_mode = fast_mode_value
        if time_fn is None:
            self._time_fn = time.perf_counter
        elif callable(time_fn):
            self._time_fn = time_fn
        else:
            raise ValueError("time_fn must be callable")
        self._smoothing = smoothing_value
        self._window_limit = window_int if self._track_window else 0
        self._window_records: deque[tuple[float, int]] = deque()
        self._window_duration = 0.0
        self._window_samples = 0
        self._window_batches = 0
        self.reset()

    def reset(self) -> None:
        """Clear the meter's accumulated state while keeping the time source."""
        self.last = self._now()
        self.samples = 0
        self._total_time = 0.0
        self._time_correction = 0.0
        self._batches = 0
        self._median = _PSquareQuantile(0.5) if self._track_distribution else None
        self._p95 = _PSquareQuantile(0.95) if self._track_distribution else None
        self._p99 = _PSquareQuantile(0.99) if self._track_distribution else None
        self._last_duration = 0.0
        self._min_duration = math.inf
        self._max_duration = 0.0
        self._mean_duration = 0.0
        self._m2_duration = 0.0
        self._ema_throughput: Optional[float] = None
        self._window_records.clear()
        self._window_duration = 0.0
        self._window_samples = 0
        self._window_batches = 0
        self._best_time_per_sample: Optional[float] = None

    def tick(self, batch_size: int) -> None:
        batch_size_int = _positive_int_setting(batch_size, "batch_size")
        now = self._now()
        elapsed = _non_negative_finite_float_setting(now - self.last, "duration_s")
        self.last = now
        self.record(elapsed, batch_size_int)

    def record(self, duration_s: float, batch_size: int) -> None:
        duration = _non_negative_finite_float_setting(duration_s, "duration_s")
        batch_size_int = _positive_int_setting(batch_size, "batch_size")

        self.samples += batch_size_int
        self._accumulate_total_time(duration)
        self._batches += 1
        delta = duration - self._mean_duration
        self._mean_duration += delta / self._batches
        delta2 = duration - self._mean_duration
        self._m2_duration += delta * delta2

        if duration < self._min_duration:
            self._min_duration = duration
        if duration > self._max_duration:
            self._max_duration = duration

        if duration > 0.0:
            time_per_sample = duration / batch_size_int
            best = self._best_time_per_sample
            if best is None or time_per_sample < best:
                self._best_time_per_sample = time_per_sample

        if self._track_distribution:
            median = self._median
            p95 = self._p95
            p99 = self._p99
            if median is not None:
                median.add(duration)
            if p95 is not None:
                p95.add(duration)
            if p99 is not None:
                p99.add(duration)
        self._last_duration = duration

        window_limit = self._window_limit
        if window_limit:
            window_records = self._window_records
            if self._window_batches == window_limit:
                old_duration, old_samples = window_records.popleft()
                self._window_duration -= old_duration
                self._window_samples -= old_samples
                self._window_batches -= 1
            window_records.append((duration, batch_size_int))
            self._window_duration += duration
            self._window_samples += batch_size_int
            self._window_batches += 1

        smoothing = self._smoothing
        if smoothing is not None and duration > 0.0:
            throughput = batch_size_int / duration
            ema = self._ema_throughput
            if ema is None:
                self._ema_throughput = throughput
            else:
                self._ema_throughput = smoothing * throughput + (1.0 - smoothing) * ema
        elif self._ema_throughput is None and smoothing is not None:
            self._ema_throughput = 0.0

    def summary(self) -> Dict[str, float]:
        total = self._total_time
        thr = (self.samples / total) if total > 0.0 else 0.0
        batches = self._batches
        avg_batch = (total / batches) if batches > 0 else 0.0
        min_batch = self._min_duration if batches > 0 and math.isfinite(self._min_duration) else 0.0
        max_batch = self._max_duration if batches > 0 else 0.0
        ema = self._ema_throughput if self._ema_throughput is not None else 0.0
        best_sps = 0.0
        if self._best_time_per_sample:
            best_sps = 1.0 / self._best_time_per_sample
        headroom = (best_sps / thr) if thr > 0.0 else 0.0
        window_values = [duration for duration, _samples in self._window_records] if self._track_window else []
        if self._track_distribution and window_values:
            ordered = sorted(window_values)

            def window_percentile(quantile: float) -> float:
                index = int(round(quantile * (len(ordered) - 1)))
                index = max(0, min(len(ordered) - 1, index))
                return float(ordered[index])

            p50 = window_percentile(0.5)
            p95 = window_percentile(0.95)
            p99 = window_percentile(0.99)
        elif self._track_distribution and self._median is not None and self._p95 is not None and self._p99 is not None:
            p50 = self._median.value()
            p95 = self._p95.value()
            p99 = self._p99.value()
        else:
            p50 = 0.0
            p95 = 0.0
            p99 = 0.0
        std_batch = (self._m2_duration / (batches - 1)) ** 0.5 if batches > 1 else 0.0
        window_thr = 0.0
        if self._window_duration > 0.0 and self._window_batches > 0:
            window_thr = self._window_samples / self._window_duration
        return {
            "p50_s": p50,
            "p95_s": p95,
            "p99_s": p99,
            "std_batch_s": std_batch,
            "samples_per_sec": thr,
            "avg_batch_s": avg_batch,
            "total_time_s": total,
            "batches": float(batches),
            "samples": float(self.samples),
            "last_batch_s": self._last_duration if batches > 0 else 0.0,
            "min_batch_s": min_batch,
            "max_batch_s": max_batch,
            "ema_samples_per_sec": ema,
            "window_samples_per_sec": window_thr,
            "window_time_s": self._window_duration if self._window_batches > 0 else 0.0,
            "window_batches": float(self._window_batches),
            "window_samples": float(self._window_samples),
            "distribution_tracked": self._track_distribution,
            "window_tracked": self._track_window,
            "best_samples_per_sec": best_sps,
            "headroom_ratio": headroom,
        }

    def time_batch(
        self,
        batch_size: int,
        *,
        record_on_exception: bool = False,
    ) -> "ThroughputMeter._BatchTimer":
        return ThroughputMeter._BatchTimer(
            self,
            batch_size,
            record_on_exception=record_on_exception,
        )

    @property
    def distribution_tracked(self) -> bool:
        """Return whether percentile tracking is enabled."""

        return self._track_distribution

    @property
    def window_tracked(self) -> bool:
        """Return whether moving-window stats are maintained."""

        return self._track_window

    def _accumulate_total_time(self, duration: float) -> None:
        y = duration - self._time_correction
        t = self._total_time + y
        self._time_correction = (t - self._total_time) - y
        self._total_time = t

    @property
    def total_time(self) -> float:
        return self._total_time

    def _now(self) -> float:
        return _time_value_setting(self._time_fn(), "time_fn")


def synchronize_device(device: str) -> None:
    """Best-effort accelerator synchronization for precise profiling."""
    device_type = _device_type(device)
    if device_type == "cuda":
        try:
            torch.cuda.synchronize(torch.device(device))
        except Exception:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    elif device_type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


class PhaseProfiler:
    """Opt-in phase and module-detail timer for training-loop bottleneck analysis."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        device: str = "cpu",
        sync: bool = False,
        track_distribution: bool = True,
        window: int = 512,
    ) -> None:
        self.enabled = _bool_setting(enabled, "enabled")
        self.device = _device_setting(device)
        self.sync = _bool_setting(sync, "sync")
        self.track_distribution = _bool_setting(track_distribution, "track_distribution")
        self.window = _positive_int_setting(window, "window")
        self.totals: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self.samples: Dict[str, deque[float]] = {}
        self._starts: Dict[str, float] = {}
        self.detail_totals: Dict[str, Dict[str, float]] = {}
        self.detail_calls: Dict[str, Dict[str, int]] = {}
        self.detail_samples: Dict[str, Dict[str, deque[float]]] = {}
        self._detail_starts: Dict[tuple[str, str], list[float]] = {}
        self.event_totals: Dict[str, Dict[str, float]] = {}
        self.event_calls: Dict[str, Dict[str, int]] = {}
        self.event_samples: Dict[str, Dict[str, deque[float]]] = {}

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(round((percentile / 100.0) * (len(ordered) - 1)))
        index = max(0, min(len(ordered) - 1, index))
        return ordered[index]

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = math.fsum(values) / len(values)
        return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))

    def _record(self, name: str, seconds: float) -> None:
        name = _profile_name_setting(name, "name")
        duration = _non_negative_finite_float_setting(seconds, "seconds")
        self.totals[name] = self.totals.get(name, 0.0) + duration
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.track_distribution:
            bucket = self.samples.get(name)
            if bucket is None:
                bucket = deque(maxlen=self.window)
                self.samples[name] = bucket
            bucket.append(duration)

    def _record_detail(self, parent: str, name: str, seconds: float) -> None:
        parent = _profile_name_setting(parent, "parent")
        name = _profile_name_setting(name, "name")
        duration = _non_negative_finite_float_setting(seconds, "seconds")
        totals = self.detail_totals.setdefault(parent, {})
        calls = self.detail_calls.setdefault(parent, {})
        totals[name] = totals.get(name, 0.0) + duration
        calls[name] = calls.get(name, 0) + 1
        if self.track_distribution:
            samples = self.detail_samples.setdefault(parent, {})
            bucket = samples.get(name)
            if bucket is None:
                bucket = deque(maxlen=self.window)
                samples[name] = bucket
            bucket.append(duration)

    def _record_event(self, group: str, name: str, seconds: float) -> None:
        group = _profile_name_setting(group, "group")
        name = _profile_name_setting(name, "name")
        duration = _non_negative_finite_float_setting(seconds, "seconds")
        totals = self.event_totals.setdefault(group, {})
        calls = self.event_calls.setdefault(group, {})
        totals[name] = totals.get(name, 0.0) + duration
        calls[name] = calls.get(name, 0) + 1
        if self.track_distribution:
            samples = self.event_samples.setdefault(group, {})
            bucket = samples.get(name)
            if bucket is None:
                bucket = deque(maxlen=self.window)
                samples[name] = bucket
            bucket.append(duration)

    def _phase_row(
        self,
        total: float,
        calls: int,
        samples: SampleWindow,
        denom: float,
        pct_name: str = "pct",
    ) -> Dict[str, Any]:
        calls = max(1, int(calls))
        row: Dict[str, Any] = {
            "total_s": float(total),
            "avg_ms": (float(total) / calls) * 1e3,
            pct_name: (100.0 * float(total) / denom) if denom > 0 else 0.0,
            "calls": calls,
        }
        sample_values = list(samples or ())
        if self.track_distribution and sample_values:
            row.update({
                "sample_count": len(sample_values),
                "p50_ms": self._percentile(sample_values, 50) * 1e3,
                "p95_ms": self._percentile(sample_values, 95) * 1e3,
                "p99_ms": self._percentile(sample_values, 99) * 1e3,
                "std_ms": self._std(sample_values) * 1e3,
                "min_ms": min(sample_values) * 1e3,
                "max_ms": max(sample_values) * 1e3,
            })
        return row

    def _event_row(self, total: float, calls: int, samples: SampleWindow) -> Dict[str, Any]:
        calls = max(1, int(calls))
        sample_values = list(samples or ())
        row: Dict[str, Any] = {
            "total_s": float(total),
            "avg_ms": (float(total) / calls) * 1e3,
            "calls": calls,
            "sample_count": calls,
        }
        if sample_values:
            row.update({
                "window_sample_count": len(sample_values),
                "p50_ms": self._percentile(sample_values, 50) * 1e3,
                "p95_ms": self._percentile(sample_values, 95) * 1e3,
                "p99_ms": self._percentile(sample_values, 99) * 1e3,
                "std_ms": self._std(sample_values) * 1e3,
                "min_ms": min(sample_values) * 1e3,
                "max_ms": max(sample_values) * 1e3,
            })
        return row

    @staticmethod
    def _with_optional_p95(row: Dict[str, Any], source: Mapping[str, Any]) -> Dict[str, Any]:
        if "p95_ms" in source:
            row["p95_ms"] = source["p95_ms"]
        return row

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        key = _profile_name_setting(name, "name")
        if self.sync:
            synchronize_device(self.device)
        self._starts[key] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        key = _profile_name_setting(name, "name")
        if self.sync:
            synchronize_device(self.device)
        start = self._starts.get(key)
        if start is None:
            return
        exception_active = sys.exc_info()[0] is not None
        try:
            self._record(key, time.perf_counter() - start)
        except ValueError:
            if not exception_active:
                raise
            self._starts.pop(key, None)
            return
        self._starts.pop(key, None)

    def cancel(self, name: str) -> None:
        if self.enabled:
            self._starts.pop(_profile_name_setting(name, "name"), None)

    def start_detail(self, parent: str, name: str) -> None:
        if not self.enabled:
            return
        parent_key = _profile_name_setting(parent, "parent")
        name_key = _profile_name_setting(name, "name")
        if self.sync:
            synchronize_device(self.device)
        key = (parent_key, name_key)
        self._detail_starts.setdefault(key, []).append(time.perf_counter())

    def stop_detail(self, parent: str, name: str) -> None:
        if not self.enabled:
            return
        parent_key = _profile_name_setting(parent, "parent")
        name_key = _profile_name_setting(name, "name")
        if self.sync:
            synchronize_device(self.device)
        key = (parent_key, name_key)
        starts = self._detail_starts.get(key)
        if not starts:
            return
        start = starts[-1]
        exception_active = sys.exc_info()[0] is not None
        try:
            self._record_detail(parent_key, name_key, time.perf_counter() - start)
        except ValueError:
            if not exception_active:
                raise
            starts.pop()
            if not starts:
                self._detail_starts.pop(key, None)
            return
        starts.pop()
        if not starts:
            self._detail_starts.pop(key, None)

    def record_event_since_start(self, parent: str, group: str, name: str) -> None:
        if not self.enabled:
            return
        parent_key = _profile_name_setting(parent, "parent")
        group_key = _profile_name_setting(group, "group")
        name_key = _profile_name_setting(name, "name")
        if self.sync:
            synchronize_device(self.device)
        start = self._starts.get(parent_key)
        if start is None:
            return
        exception_active = sys.exc_info()[0] is not None
        try:
            self._record_event(group_key, name_key, time.perf_counter() - start)
        except ValueError:
            if not exception_active:
                raise

    def summary(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        denom = math.fsum(self.totals.values())
        phases = {
            name: self._phase_row(total, self.calls.get(name, 0), self.samples.get(name, ()), denom)
            for name, total in sorted(self.totals.items())
        }
        phase_breakdowns: Dict[str, Dict[str, Any]] = {}
        for parent, totals in sorted(self.detail_totals.items()):
            parent_total = self.totals.get(parent, math.fsum(totals.values()))
            children = {
                name: self._phase_row(
                    total,
                    self.detail_calls.get(parent, {}).get(name, 0),
                    self.detail_samples.get(parent, {}).get(name, ()),
                    parent_total,
                    pct_name="pct_of_parent",
                )
                for name, total in sorted(totals.items())
            }
            top_children = sorted(
                (
                    self._with_optional_p95(
                        {
                            "name": name,
                            "total_s": row["total_s"],
                            "pct_of_parent": row["pct_of_parent"],
                            "avg_ms": row["avg_ms"],
                            "calls": row["calls"],
                        },
                        row,
                    )
                    for name, row in children.items()
                ),
                key=lambda row: row["total_s"],
                reverse=True,
            )
            tracked_s = math.fsum(float(row["total_s"]) for row in children.values())
            untracked_s = max(0.0, parent_total - tracked_s)
            overtracked_s = max(0.0, tracked_s - parent_total)
            phase_breakdowns[parent] = {
                "parent_total_s": parent_total,
                "tracked_s": tracked_s,
                "untracked_s": untracked_s,
                "overtracked_s": overtracked_s,
                "children": children,
                "top_children": top_children,
            }
        phase_events: Dict[str, Dict[str, Any]] = {}
        for group, totals in sorted(self.event_totals.items()):
            children = {
                name: self._event_row(
                    total,
                    self.event_calls.get(group, {}).get(name, 0),
                    self.event_samples.get(group, {}).get(name, ()),
                )
                for name, total in sorted(totals.items())
            }
            top_children = sorted(
                (
                    self._with_optional_p95(
                        {
                            "name": name,
                            "total_s": row["total_s"],
                            "avg_ms": row["avg_ms"],
                            "calls": row["calls"],
                            "sample_count": row["sample_count"],
                        },
                        row,
                    )
                    for name, row in children.items()
                ),
                key=lambda row: row["avg_ms"],
                reverse=True,
            )
            phase_events[group] = {"children": children, "top_children": top_children}
        top_phases = sorted(
            (
                self._with_optional_p95(
                    {
                        "name": name,
                        "total_s": row["total_s"],
                        "pct": row["pct"],
                        "avg_ms": row["avg_ms"],
                        "calls": row["calls"],
                    },
                    row,
                )
                for name, row in phases.items()
            ),
            key=lambda row: row["total_s"],
            reverse=True,
        )
        open_details = [
            {"parent": parent, "name": name, "count": len(starts)}
            for (parent, name), starts in sorted(self._detail_starts.items())
            if starts
        ]
        open_detail_count = sum(len(starts) for starts in self._detail_starts.values() if starts)
        return {
            "profile_sync": self.sync,
            "profile_distribution": self.track_distribution,
            "profile_window": self.window if self.track_distribution else 0,
            "profile_total_s": denom,
            "profile_open_phase_count": len(self._starts),
            "profile_open_detail_count": open_detail_count,
            "profile_open_phases": sorted(self._starts.keys()),
            "profile_open_details": open_details,
            "phases": phases,
            "phase_breakdowns": phase_breakdowns,
            "phase_events": phase_events,
            "top_phases": top_phases,
        }


def maybe_channels_last(model: nn.Module, channels_last: bool = False) -> nn.Module:
    model_value = _module_setting(model)
    channels_last_value = _bool_setting(channels_last, "channels_last")
    if not channels_last_value:
        return model_value
    try:
        to_method = cast(Callable[..., nn.Module], getattr(model_value, "to"))
        return to_method(memory_format=torch.channels_last)
    except Exception:
        return model_value

def safe_compile(model: nn.Module, mode: str = "reduce-overhead") -> Tuple[nn.Module, bool]:
    """Compile model if torch.compile exists and succeeds."""
    result = safe_compile_with_diagnostics(model, mode=mode)
    return result.model, result.compiled


def _format_compile_exception(exc: Exception) -> str:
    try:
        message = str(exc).strip()
    except Exception:
        message = ""
    if len(message) > 200:
        message = f"{message[:197]}..."
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def safe_compile_with_diagnostics(model: nn.Module, mode: str = "reduce-overhead") -> CompileResult:
    """Compile model if possible and keep a compact fallback reason."""
    model_value = _module_setting(model)
    mode_value = _compile_mode_setting(mode)
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return CompileResult(model=model_value, compiled=False, fallback_reason="torch_compile_unavailable")
    try:
        m = compile_fn(model_value, mode=mode_value)
        if isinstance(m, nn.Module):
            return CompileResult(model=m, compiled=True)
        return CompileResult(
            model=model_value,
            compiled=False,
            fallback_reason=f"non_module_result:{type(m).__name__}",
        )
    except Exception as exc:
        return CompileResult(
            model=model_value,
            compiled=False,
            fallback_reason=_format_compile_exception(exc),
        )
