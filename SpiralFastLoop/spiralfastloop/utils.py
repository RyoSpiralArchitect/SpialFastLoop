# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

import math
import os
import time
from collections import deque
from collections.abc import Mapping, MutableMapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

def get_best_device() -> str:
    """Pick the best available device among CUDA, MPS, CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

AmpSetting = Union[bool, Literal["auto"], None]


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

    if device == "cuda":
        # Prefer bf16 on Ampere+ (TF32/bf16), else fp16
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return True, torch.bfloat16, True
        else:
            return True, torch.float16, True
    elif device == "mps":
        # MPS AMP is fp16; GradScaler is not used on MPS.
        return True, torch.float16, False
    else:
        return False, torch.float32, False

def autocast_ctx(device: str, enabled: bool, amp_dtype: torch.dtype) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device, dtype=amp_dtype)

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
        pin_memory = (device == "cuda")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, prefetch_factor=prefetch_factor,
        persistent_workers=persistent, pin_memory=pin_memory
    )

class _PSquareQuantile:
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
        if self._q is None or self._n is None or self._np is None or self._dn is None:
            self._initial.append(float(value))
            if len(self._initial) == 5:
                self._initial.sort()
                self._q = self._initial.copy()
                self._n = [i + 1 for i in range(5)]
                q = self.quantile
                self._np = [
                    1.0,
                    1.0 + 2.0 * q,
                    1.0 + 4.0 * q,
                    3.0 + 2.0 * q,
                    5.0,
                ]
                self._dn = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]
            return

        q_values = self._q
        positions = self._n
        desired = self._np
        increments = self._dn
        assert q_values is not None and positions is not None and desired is not None and increments is not None

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
    """Measure batch latencies and throughput with streaming quantile estimates."""

    class _BatchTimer(AbstractContextManager["ThroughputMeter._BatchTimer"]):
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
    ) -> None:
        if smoothing is not None:
            if not (0.0 < smoothing <= 1.0):
                raise ValueError("smoothing must be in the interval (0, 1].")
        window_int = int(window)
        if window_int < 0:
            raise ValueError("window must be non-negative.")
        self._time_fn: Callable[[], float] = time_fn or time.perf_counter
        self._smoothing = smoothing
        self._window_limit = window_int
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
        self._median = _PSquareQuantile(0.5)
        self._p95 = _PSquareQuantile(0.95)
        self._last_duration = 0.0
        self._min_duration = math.inf
        self._max_duration = 0.0
        self._ema_throughput: Optional[float] = None
        self._window_records.clear()
        self._window_duration = 0.0
        self._window_samples = 0
        self._window_batches = 0

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
        self.samples += batch_size_int
        duration = float(duration_s)
        self._accumulate_total_time(duration)
        self._batches += 1
        self._median.add(duration)
        self._p95.add(duration)
        self._last_duration = duration
        if duration < self._min_duration:
            self._min_duration = duration
        if duration > self._max_duration:
            self._max_duration = duration

        if self._window_limit:
            if self._window_batches == self._window_limit:
                old_duration, old_samples = self._window_records.popleft()
                self._window_duration -= old_duration
                self._window_samples -= old_samples
                self._window_batches -= 1
            self._window_records.append((duration, batch_size_int))
            self._window_duration += duration
            self._window_samples += batch_size_int
            self._window_batches += 1

        if self._smoothing is not None and duration > 0.0:
            throughput = batch_size_int / duration
            if self._ema_throughput is None:
                self._ema_throughput = throughput
            else:
                alpha = self._smoothing
                assert alpha is not None
                self._ema_throughput = alpha * throughput + (1.0 - alpha) * self._ema_throughput
        elif self._ema_throughput is None and self._smoothing is not None:
            self._ema_throughput = 0.0

    def summary(self) -> Dict[str, float]:
        total = self._total_time
        thr = (self.samples / total) if total > 0.0 else 0.0
        batches = self._batches
        avg_batch = (total / batches) if batches > 0 else 0.0
        min_batch = self._min_duration if batches > 0 and math.isfinite(self._min_duration) else 0.0
        max_batch = self._max_duration if batches > 0 else 0.0
        ema = self._ema_throughput if self._ema_throughput is not None else 0.0
        window_thr = 0.0
        if self._window_duration > 0.0 and self._window_batches > 0:
            window_thr = self._window_samples / self._window_duration
        return {
            "p50_s": self._median.value(),
            "p95_s": self._p95.value(),
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
