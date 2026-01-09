# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

import math
import os
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


def get_distributed_context() -> DistributedContext:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = _env_int("LOCAL_RANK", rank)
        backend = torch.distributed.get_backend()
        return DistributedContext(
            is_initialized=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=backend,
        )
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", rank)
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
    if not torch.distributed.is_available():
        return get_distributed_context()
    if torch.distributed.is_initialized():
        return get_distributed_context()
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return get_distributed_context()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method=init_method)
    return get_distributed_context()

AmpSetting = Union[bool, Literal["auto"], None]


def _device_type(device: str) -> str:
    return device.split(":", 1)[0]


def get_amp_policy(device: str, use_amp: AmpSetting = "auto") -> Tuple[bool, torch.dtype, bool]:
    """
    Decide AMP usage, dtype, and whether GradScaler should be used.

    Returns:
        enabled: bool
        amp_dtype: torch.dtype
        use_scaler: bool  # GradScaler only for CUDA
    """
    if use_amp is False:
        return False, torch.float32, False

    if use_amp is None:
        use_amp = "auto"

    device_type = _device_type(device)
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
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=_device_type(device), dtype=amp_dtype)

def to_device(obj: Any, device: str, non_blocking: bool = True) -> Any:
    """Recursively move tensors (and nested structures) to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, list):
        return type(obj)(to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, tuple):
        converted_tuple = tuple(to_device(x, device, non_blocking) for x in obj)
        if hasattr(obj, "_fields"):
            return type(obj)(*converted_tuple)
        return type(obj)(converted_tuple)
    if isinstance(obj, MutableMapping):
        converted_mapping = {k: to_device(v, device, non_blocking) for k, v in obj.items()}
        mapping_type = cast(Any, type(obj))
        if hasattr(obj, "default_factory"):
            default_factory = getattr(obj, "default_factory")
            new_mapping = mapping_type(default_factory)
        else:
            new_mapping = mapping_type()
        new_mapping.update(converted_mapping)
        return new_mapping
    if isinstance(obj, Mapping):
        converted_mapping = {k: to_device(v, device, non_blocking) for k, v in obj.items()}
        mapping_type = cast(Any, type(obj))
        return mapping_type(converted_mapping)
    return obj

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
    if pin_memory is None:
        pin_memory = (_device_type(device) == "cuda")
    sampler = None
    if distributed:
        ctx = get_distributed_context()
        if ctx.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
            shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    tensor = value.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
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
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = float(quantile)
        self._initial: list[float] = []
        self._q: Optional[list[float]] = None
        self._n: Optional[list[int]] = None
        self._np: Optional[list[float]] = None
        self._dn: Optional[list[float]] = None

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return

        q_values = self._q
        positions = self._n
        desired = self._np
        increments = self._dn

        if q_values is None or positions is None or desired is None or increments is None:
            initial = self._initial
            initial.append(float(value))
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

        assert positions is not None and desired is not None and increments is not None

        if value < q_values[0]:
            q_values[0] = float(value)
            k = 0
        elif value >= q_values[4]:
            q_values[4] = float(value)
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
        assert self._q is not None and self._n is not None
        q_values = self._q
        positions = self._n
        numerator = step * (
            (positions[idx] - positions[idx - 1] + step) * (q_values[idx + 1] - q_values[idx]) / (positions[idx + 1] - positions[idx])
            + (positions[idx + 1] - positions[idx] - step) * (q_values[idx] - q_values[idx - 1]) / (positions[idx] - positions[idx - 1])
        )
        denominator = positions[idx + 1] - positions[idx - 1]
        if denominator == 0:
            return q_values[idx]
        return q_values[idx] + numerator / denominator

    def _linear_update(self, idx: int, step: int) -> float:
        assert self._q is not None and self._n is not None
        q_values = self._q
        positions = self._n
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
        "_last_duration",
        "_min_duration",
        "_max_duration",
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
            self._batch_size = batch_size
            self._record_on_exception = record_on_exception
            self._start: Optional[float] = None

        def __enter__(self) -> "ThroughputMeter._BatchTimer":
            self._start = self._meter._time_fn()
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            end = self._meter._time_fn()
            self._meter.last = end
            if self._start is None:
                return False
            duration = max(0.0, end - self._start)
            should_record = exc_type is None or self._record_on_exception
            if should_record:
                self._meter.record(duration, self._batch_size)
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
        if fast_mode:
            track_distribution = False
            track_window = False
            smoothing = None
        if smoothing is not None:
            if not (0.0 < smoothing <= 1.0):
                raise ValueError("smoothing must be in the interval (0, 1].")
        window_int = int(window)
        if window_int < 0:
            raise ValueError("window must be non-negative.")
        self._track_distribution = bool(track_distribution)
        self._track_window = bool(track_window)
        self._fast_mode = bool(fast_mode)
        self._time_fn: Callable[[], float] = time_fn or time.perf_counter
        self._smoothing = smoothing
        self._window_limit = window_int if self._track_window else 0
        self._window_records: deque[tuple[float, int]] = deque()
        self._window_duration = 0.0
        self._window_samples = 0
        self._window_batches = 0
        self.reset()

    def reset(self) -> None:
        """Clear the meter's accumulated state while keeping the time source."""
        self.last = self._time_fn()
        self.samples = 0
        self._total_time = 0.0
        self._time_correction = 0.0
        self._batches = 0
        self._median = _PSquareQuantile(0.5) if self._track_distribution else None
        self._p95 = _PSquareQuantile(0.95) if self._track_distribution else None
        self._last_duration = 0.0
        self._min_duration = math.inf
        self._max_duration = 0.0
        self._ema_throughput: Optional[float] = None
        self._window_records.clear()
        self._window_duration = 0.0
        self._window_samples = 0
        self._window_batches = 0
        self._best_time_per_sample: Optional[float] = None

    def tick(self, batch_size: int) -> None:
        now = self._time_fn()
        elapsed = max(0.0, now - self.last)
        self.last = now
        self.record(elapsed, batch_size)

    def record(self, duration_s: float, batch_size: int) -> None:
        if duration_s < 0.0:
            raise ValueError("Duration must be non-negative.")
        if not math.isfinite(duration_s):
            raise ValueError("Duration must be finite.")
        batch_size_int = int(batch_size)
        if batch_size_int <= 0:
            raise ValueError("batch_size must be a positive integer.")
        duration = float(duration_s)

        self.samples += batch_size_int
        self._accumulate_total_time(duration)
        self._batches += 1

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
            if median is not None:
                median.add(duration)
            if p95 is not None:
                p95.add(duration)
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
        if self._track_distribution and self._median is not None and self._p95 is not None:
            p50 = self._median.value()
            p95 = self._p95.value()
        else:
            p50 = 0.0
            p95 = 0.0
        window_thr = 0.0
        if self._window_duration > 0.0 and self._window_batches > 0:
            window_thr = self._window_samples / self._window_duration
        return {
            "p50_s": p50,
            "p95_s": p95,
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

def maybe_channels_last(model: nn.Module, channels_last: bool = False) -> nn.Module:
    if not channels_last:
        return model
    try:
        to_method = cast(Callable[..., nn.Module], getattr(model, "to"))
        return to_method(memory_format=torch.channels_last)
    except Exception:
        return model

def safe_compile(model: nn.Module, mode: str = "reduce-overhead") -> Tuple[nn.Module, bool]:
    """Compile model if torch.compile exists and succeeds."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model, False
    try:
        m = compile_fn(model, mode=mode)
        assert isinstance(m, nn.Module)
        return m, True
    except Exception:
        return model, False
