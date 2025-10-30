# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

import math
import os
import time
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

    def __init__(self, *, time_fn: Optional[Callable[[], float]] = None) -> None:
        self._time_fn: Callable[[], float] = time_fn or time.perf_counter
        initial_time = self._time_fn()
        self.last: float = initial_time
        self.samples: int = 0
        self.total_time: float = 0.0
        self._median = _PSquareQuantile(0.5)
        self._p95 = _PSquareQuantile(0.95)

    def tick(self, batch_size: int) -> None:
        now = self._time_fn()
        elapsed = max(0.0, now - self.last)
        self.last = now
        self.record(elapsed, batch_size)

    def record(self, duration_s: float, batch_size: int) -> None:
        if duration_s < 0.0:
            raise ValueError("Duration must be non-negative.")
        self.samples += int(batch_size)
        self.total_time += float(duration_s)
        self._median.add(duration_s)
        self._p95.add(duration_s)

    def summary(self) -> Dict[str, float]:
        total = self.total_time
        thr = (self.samples / total) if total > 0.0 else 0.0
        return {
            "p50_s": self._median.value(),
            "p95_s": self._p95.value(),
            "samples_per_sec": thr,
        }

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
