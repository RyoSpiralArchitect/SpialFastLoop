# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

import os
import time
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Iterable, Literal, Optional, Tuple, TypeVar, Union, overload

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "get_best_device",
    "get_amp_policy",
    "autocast_ctx",
    "to_device",
    "dataloader_from_dataset",
    "ThroughputMeter",
    "maybe_channels_last",
    "safe_compile",
]


T = TypeVar("T")


def get_best_device() -> str:
    """Pick the best available device among CUDA, MPS, CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_amp_policy(
    device: str, use_amp: Union[bool, Literal["auto"], None] = "auto"
) -> Tuple[bool, torch.dtype, bool]:
    """
    Decide AMP usage, dtype, and whether GradScaler should be used.

    Returns:
        enabled: bool
        amp_dtype: torch.dtype
        use_scaler: bool  # GradScaler only for CUDA
    """
    if use_amp is False:
        return False, torch.float32, False

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


def autocast_ctx(
    device: str, enabled: bool, amp_dtype: torch.dtype
) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device, dtype=amp_dtype)


@overload
def to_device(
    obj: torch.Tensor, device: str, non_blocking: bool = True
) -> torch.Tensor:
    """Overload stub returning a tensor."""
    ...


@overload
def to_device(obj: Sequence[T], device: str, non_blocking: bool = True) -> Sequence[T]:
    """Overload stub returning a sequence."""
    ...


@overload
def to_device(
    obj: Mapping[Any, T], device: str, non_blocking: bool = True
) -> Mapping[Any, T]:
    """Overload stub returning a mapping."""
    ...


def to_device(obj: Any, device: str, non_blocking: bool = True) -> Any:
    """Recursively move tensors (and nested structures) to device."""

    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, list):
        return type(obj)(to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, tuple):
        converted = tuple(to_device(x, device, non_blocking) for x in obj)
        if hasattr(obj, "_fields"):
            return type(obj)(*converted)
        return type(obj)(converted)
    if isinstance(obj, Mapping):
        converted_dict: dict[Any, Any] = {
            k: to_device(v, device, non_blocking) for k, v in obj.items()
        }
        if isinstance(obj, MutableMapping):
            try:
                new_mapping: MutableMapping[Any, Any] = type(obj)()
                new_mapping.update(converted_dict)
                return new_mapping
            except Exception:
                pass
        return dict(converted_dict)
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
    if num_workers is None:
        try:
            cpu_count = os.cpu_count() or 0
            num_workers = max(2, cpu_count // 2) if cpu_count else 2
        except Exception:
            num_workers = 2
    if pin_memory is None:
        pin_memory = device == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent,
        pin_memory=pin_memory,
    )


class ThroughputMeter:
    """Measure batch latencies and throughput."""

    def __init__(self) -> None:
        self.batch_times: list[float] = []
        self.last = time.perf_counter()
        self.samples = 0

    def tick(self, batch_size: int) -> None:
        now = time.perf_counter()
        self.batch_times.append(now - self.last)
        self.samples += batch_size
        self.last = now

    @staticmethod
    def _percentile(xs: Iterable[float], percentile: float) -> float:
        xs_list = list(xs)
        if not xs_list:
            return 0.0
        xs_sorted = sorted(xs_list)
        idx = max(
            0,
            min(
                len(xs_sorted) - 1,
                int(round((percentile / 100.0) * (len(xs_sorted) - 1))),
            ),
        )
        return xs_sorted[idx]

    def summary(self) -> dict[str, float]:
        p50 = self._percentile(self.batch_times, 50)
        p95 = self._percentile(self.batch_times, 95)
        total = sum(self.batch_times) if self.batch_times else 0.0
        throughput = (self.samples / total) if total > 0 else 0.0
        return {"p50_s": p50, "p95_s": p95, "samples_per_sec": throughput}


def maybe_channels_last(
    model: torch.nn.Module, channels_last: bool = False
) -> torch.nn.Module:
    if not channels_last:
        return model
    try:
        return model.to(memory_format=torch.channels_last)
    except Exception:
        return model


def safe_compile(
    model: torch.nn.Module, mode: str = "reduce-overhead"
) -> tuple[torch.nn.Module, bool]:
    """Compile model if torch.compile exists and succeeds."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model, False
    try:
        m = compile_fn(model, mode=mode)
        return m, True
    except Exception:
        return model, False
