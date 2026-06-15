#!/usr/bin/env python3
"""Benchmark FastTrainer under heavy transactional and parallel workloads."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
from spiralfastloop.utils import (
    _bool_setting,
    _finite_float_setting,
    _int_setting,
    _non_negative_int_setting,
    _positive_int_setting,
    dataloader_from_dataset,
)
from scripts.json_utils import dump_json, dumps_json

BASE_SUMMARY_FIELDS = (
    "reported_samples_per_sec",
    "samples_per_sec",
    "steady_samples_per_sec",
    "end_to_end_wall_time_s",
    "setup_time_s",
    "wall_time_s",
    "cold_start_time_s",
)

BATCH_SUMMARY_FIELDS = (
    "p50_s",
    "p95_s",
    "p99_s",
    "std_batch_s",
    "avg_batch_s",
    "best_samples_per_sec",
    "headroom_ratio",
)

WORKLOAD_SUMMARY_FIELDS = (
    "steps",
    "samples",
    "optimizer_steps",
    "grad_accum",
    "partial_optimizer_steps",
    "grad_accum_tail_steps",
    "warmup_steps",
    "warmup_samples",
    "warmup_optimizer_steps",
    "warmup_samples_per_sec",
    "warmup_total_time_s",
    "warmup_p99_s",
    "cold_start_steps",
    "cold_start_samples_per_sec",
    "steady_steps",
    "steady_samples",
    "steady_optimizer_steps",
    "steady_total_time_s",
    "steady_p99_s",
)

PROFILE_SUMMARY_FIELDS = (
    "profile_total_s",
    "profile_flat_metric_invalid_count",
    "profile_data_wait_time_s",
    "profile_data_wait_pct",
    "profile_data_wait_avg_ms",
    "profile_transfer_time_s",
    "profile_transfer_pct",
    "profile_transfer_avg_ms",
    "profile_forward_time_s",
    "profile_forward_pct",
    "profile_forward_avg_ms",
    "profile_loss_time_s",
    "profile_loss_pct",
    "profile_loss_avg_ms",
    "profile_loss_reduce_time_s",
    "profile_loss_reduce_pct",
    "profile_loss_reduce_avg_ms",
    "profile_trigger_time_s",
    "profile_trigger_pct",
    "profile_trigger_avg_ms",
    "profile_inject_transfer_time_s",
    "profile_inject_transfer_pct",
    "profile_inject_transfer_avg_ms",
    "profile_backward_time_s",
    "profile_backward_pct",
    "profile_backward_avg_ms",
    "profile_forward_backward_time_s",
    "profile_forward_backward_pct",
    "profile_optimizer_time_s",
    "profile_optimizer_pct",
    "profile_optimizer_avg_ms",
    "profile_user_metrics_time_s",
    "profile_user_metrics_pct",
    "profile_user_metrics_avg_ms",
    "profile_postprocess_time_s",
    "profile_postprocess_pct",
    "profile_postprocess_avg_ms",
    "profile_collect_output_time_s",
    "profile_collect_output_pct",
    "profile_collect_output_avg_ms",
    "profile_metrics_time_s",
    "profile_metrics_pct",
    "profile_metrics_avg_ms",
)

DEVICE_MEMORY_SUMMARY_FIELDS = (
    "cuda_current_mem_bytes",
    "cuda_max_mem_bytes",
    "cuda_reserved_mem_bytes",
    "cuda_max_reserved_mem_bytes",
    "mps_current_mem_bytes",
    "mps_max_mem_bytes",
    "mps_driver_mem_bytes",
    "mps_recommended_max_mem_bytes",
)

SUMMARY_FIELDS = (
    BASE_SUMMARY_FIELDS
    + BATCH_SUMMARY_FIELDS
    + WORKLOAD_SUMMARY_FIELDS
    + PROFILE_SUMMARY_FIELDS
    + DEVICE_MEMORY_SUMMARY_FIELDS
)

BEST_RUN_FIELDS = (
    "run",
    "seed",
    "dataset_mode",
    "reported_samples_per_sec",
    "samples_per_sec",
    "steady_samples_per_sec",
    "p99_s",
    "std_batch_s",
    "best_samples_per_sec",
    "headroom_ratio",
    "end_to_end_wall_time_s",
    "setup_time_s",
    "wall_time_s",
    "steps",
    "samples",
    "optimizer_steps",
    "grad_accum",
    "partial_optimizer_steps",
    "grad_accum_tail_steps",
    "warmup_steps",
    "warmup_samples",
    "warmup_optimizer_steps",
    "warmup_samples_per_sec",
    "warmup_total_time_s",
    "warmup_p99_s",
    "cold_start_steps",
    "cold_start_time_s",
    "cold_start_samples_per_sec",
    "steady_steps",
    "steady_samples",
    "steady_optimizer_steps",
    "steady_total_time_s",
    "steady_p99_s",
    "profile_flat_metric_invalid_count",
    "profile_forward_backward_pct",
    "profile_forward_backward_time_s",
    "profile_forward_pct",
    "profile_loss_pct",
    "profile_loss_reduce_pct",
    "profile_backward_pct",
    "profile_optimizer_pct",
    "profile_user_metrics_pct",
    "profile_postprocess_pct",
    "profile_collect_output_pct",
    "profile_metrics_pct",
    "cuda_current_mem_bytes",
    "cuda_max_mem_bytes",
    "cuda_reserved_mem_bytes",
    "cuda_max_reserved_mem_bytes",
    "mps_current_mem_bytes",
    "mps_max_mem_bytes",
    "mps_driver_mem_bytes",
    "mps_recommended_max_mem_bytes",
)

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
DATASET_MODE_CHOICES = frozenset({"generated", "materialized"})
BEST_RUN_TEXT_FIELDS = frozenset({"dataset_mode"})


@dataclass
class BenchmarkResult:
    wall_time_s: float
    trainer_metrics: dict
    run_index: int

    def as_dict(self) -> dict:
        payload = {"wall_time_s": self.wall_time_s, "run": self.run_index}
        payload.update(self.trainer_metrics)
        return payload


def _summary_row(row: dict) -> dict:
    normalized = dict(row)
    if "reported_samples_per_sec" not in normalized:
        if "steady_samples_per_sec" in normalized:
            normalized["reported_samples_per_sec"] = normalized["steady_samples_per_sec"]
        elif "samples_per_sec" in normalized:
            normalized["reported_samples_per_sec"] = normalized["samples_per_sec"]
    if "end_to_end_wall_time_s" not in normalized and "wall_time_s" in normalized:
        normalized["end_to_end_wall_time_s"] = normalized["wall_time_s"]
    return normalized


def _finite_summary_value(raw: object) -> Optional[float]:
    if isinstance(raw, (bool, str)):
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _summary_choice(raw: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = raw.strip()
    if value not in allowed:
        raise ValueError(f"{name} has unsupported value: {value}")
    return value


def _compact_run(row: dict) -> dict:
    compact = {}
    for field in BEST_RUN_FIELDS:
        if field not in row:
            continue
        value = row[field]
        if field in BEST_RUN_TEXT_FIELDS:
            if field == "dataset_mode":
                value = _summary_choice(value, DATASET_MODE_CHOICES, field)
        elif _finite_summary_value(value) is None:
            continue
        compact[field] = value
    return compact


def _finite_metric_value(row: dict, field: str) -> Optional[float]:
    if field not in row:
        return None
    return _finite_summary_value(row[field])


def _best_finite_row(
    rows: list[dict],
    field: str,
    *,
    prefer_high: bool,
    sample_count_field: Optional[str] = None,
) -> Optional[dict]:
    candidates = []
    for row in rows:
        if sample_count_field is not None and sample_count_field in row:
            sample_count = _finite_metric_value(row, sample_count_field)
            if sample_count is None or sample_count <= 0.0:
                continue
        value = _finite_metric_value(row, field)
        if value is None:
            continue
        candidates.append((value, row))
    if not candidates:
        return None
    selector = max if prefer_high else min
    return selector(candidates, key=lambda item: item[0])[1]


def summary_fields_for_rows(rows: list[dict]) -> tuple[str, ...]:
    present_fields = set()
    for row in rows:
        present_fields.update(row.keys())
    batch_fields = tuple(field for field in BATCH_SUMMARY_FIELDS if field in present_fields)
    workload_fields = tuple(field for field in WORKLOAD_SUMMARY_FIELDS if field in present_fields)
    profile_fields = tuple(field for field in PROFILE_SUMMARY_FIELDS if field in present_fields)
    memory_fields = tuple(field for field in DEVICE_MEMORY_SUMMARY_FIELDS if field in present_fields)
    return BASE_SUMMARY_FIELDS + batch_fields + workload_fields + profile_fields + memory_fields


def count_profiled_rows(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if any(field in row for field in PROFILE_SUMMARY_FIELDS)
    )


def summarize_metric(rows: list[dict], field: str, *, missing_as_zero: bool = True) -> dict[str, float]:
    values = []
    sample_count = 0
    missing_count = 0
    non_finite_count = 0
    for row in rows:
        if field in row:
            value = _finite_summary_value(row[field])
            if value is None:
                non_finite_count += 1
                continue
            values.append(value)
            sample_count += 1
        elif missing_as_zero:
            values.append(0.0)
            missing_count += 1
    result: dict[str, float] = {}
    if not values:
        result.update({"mean": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0})
        if missing_count > 0 or non_finite_count > 0:
            result["sample_count"] = float(sample_count)
        if missing_count > 0:
            result["missing_count"] = float(missing_count)
        if non_finite_count > 0:
            result["non_finite_count"] = float(non_finite_count)
        return result
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    result.update({
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "stddev": math.sqrt(variance),
    })
    if sample_count != len(rows) or non_finite_count > 0:
        result["sample_count"] = float(sample_count)
    if missing_count > 0:
        result["missing_count"] = float(missing_count)
    if non_finite_count > 0:
        result["non_finite_count"] = float(non_finite_count)
    return result


def summarize_results(rows: list[dict]) -> dict:
    summary_rows = [_summary_row(row) for row in rows]
    summary: dict = {
        "runs": len(summary_rows),
        "best_reported": None,
        "best_end_to_end": None,
    }
    profiled_runs = count_profiled_rows(summary_rows)
    if profiled_runs > 0:
        summary["profiled_runs"] = profiled_runs
    for field in summary_fields_for_rows(summary_rows):
        missing_as_zero = field in BASE_SUMMARY_FIELDS
        for stat_name, value in summarize_metric(
            summary_rows,
            field,
            missing_as_zero=missing_as_zero,
        ).items():
            summary[f"{stat_name}_{field}"] = value

    if summary_rows:
        best_reported = _best_finite_row(
            summary_rows,
            "reported_samples_per_sec",
            prefer_high=True,
        )
        best_end_to_end = _best_finite_row(
            summary_rows,
            "end_to_end_wall_time_s",
            prefer_high=False,
        )
        summary["best_reported"] = _compact_run(best_reported) if best_reported is not None else None
        summary["best_end_to_end"] = _compact_run(best_end_to_end) if best_end_to_end is not None else None
    return summary


def _int_arg(raw: object) -> int:
    if isinstance(raw, str):
        try:
            return int(raw)
        except Exception as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
    try:
        return _int_setting(raw, "value")
    except Exception as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc


def positive_int_arg(raw: object) -> int:
    value = _int_arg(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def non_negative_int_arg(raw: object) -> int:
    value = _int_arg(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def positive_float_arg(raw: object) -> float:
    if isinstance(raw, bool):
        raise argparse.ArgumentTypeError("must be a number")
    try:
        value = float(raw)
    except Exception as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _positive_float_setting(raw: object, name: str) -> float:
    if isinstance(raw, str):
        raise ValueError(f"{name} must be a positive finite number")
    value = _finite_float_setting(raw, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _profile_model_include_setting(raw: object) -> object:
    if raw is None or isinstance(raw, str):
        return raw
    if not isinstance(raw, Sequence):
        raise ValueError("profile_model_include must be a string, sequence of strings, or None")
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("profile_model_include entries must be strings")
    return raw


def _path_setting(raw: object, name: str) -> str:
    if not isinstance(raw, (str, PathLike)):
        raise ValueError(f"{name} must be a non-empty path string")
    try:
        normalized = os.fsdecode(raw)
    except Exception as exc:
        raise ValueError(f"{name} must be a non-empty path string") from exc
    if not isinstance(normalized, str) or not normalized.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    return normalized


def _optional_path_setting(raw: object, name: str) -> Optional[str]:
    if raw is None:
        return None
    return _path_setting(raw, name)


def device_arg(raw: object) -> str:
    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError("must be one of auto, cpu, cuda, mps")
    if raw not in DEVICE_CHOICES:
        raise argparse.ArgumentTypeError("must be one of auto, cpu, cuda, mps")
    return raw


def _finite_display_value(raw: object) -> Optional[float]:
    if isinstance(raw, (bool, str)):
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_metric_value(
    raw: object,
    *,
    precision: int,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    value = _finite_display_value(raw)
    if value is None:
        return "n/a"
    return f"{value * scale:.{precision}f}{suffix}"


def _format_count(raw: object) -> str:
    if isinstance(raw, (bool, str)):
        return "n/a"
    try:
        return str(non_negative_int_arg(raw))
    except argparse.ArgumentTypeError:
        return "n/a"


def _has_positive_display_value(raw: object) -> bool:
    value = _finite_display_value(raw)
    return value is not None and value > 0.0


def _profile_row_name(row: dict[str, Any]) -> str:
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return "<unnamed>"


def _dict_value(raw: object) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _list_value(raw: object) -> list[object]:
    return raw if isinstance(raw, list) else []


def _profile_child_rows(profile: dict[str, Any], section: str, group: str) -> list[object]:
    sections = _dict_value(profile.get(section))
    group_profile = _dict_value(sections.get(group))
    return _list_value(group_profile.get("top_children"))


def validate_benchmark_args(args: argparse.Namespace) -> None:
    steps = _positive_int_setting(args.steps, "steps")
    warmup_steps = _non_negative_int_setting(args.warmup_steps, "warmup_steps")
    if warmup_steps > steps:
        raise ValueError("warmup-steps must be less than or equal to steps")
    for field in (
        "transactions",
        "feature_dim",
        "num_classes",
        "samples",
        "classes",
        "batch_size",
        "grad_accum",
        "prefetch_factor",
        "runs",
        "profile_window",
        "profile_model_depth",
        "profile_model_max_modules",
    ):
        if hasattr(args, field):
            _positive_int_setting(getattr(args, field), field)
    for field in ("workers", "log_interval"):
        if hasattr(args, field):
            _non_negative_int_setting(getattr(args, field), field)
    if hasattr(args, "seed"):
        _int_setting(args.seed, "seed")
    if hasattr(args, "learning_rate"):
        _positive_float_setting(args.learning_rate, "learning_rate")
    for field in ("compile", "collect_profile", "profile_sync", "profile_distribution", "profile_model"):
        if hasattr(args, field):
            _bool_setting(getattr(args, field), field)
    if hasattr(args, "profile_model_include"):
        _profile_model_include_setting(args.profile_model_include)
    for field in ("json_out", "summary_out"):
        if hasattr(args, field):
            _optional_path_setting(getattr(args, field), field)
    if hasattr(args, "dataset_mode") and args.dataset_mode not in {"generated", "materialized"}:
        raise ValueError("dataset_mode must be one of generated, materialized")
    if hasattr(args, "device"):
        try:
            device_arg(args.device)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"device {exc}") from exc


class SyntheticTransactionDataset(Dataset):
    """Synthetic tabular dataset that simulates transactional workloads."""

    def __init__(
        self,
        size: int,
        features: int,
        classes: int,
        *,
        seed: int = 17,
        materialized: bool = False,
    ) -> None:
        self.size = _positive_int_setting(size, "size")
        self.features = _positive_int_setting(features, "features")
        self.classes = _positive_int_setting(classes, "classes")
        self.seed = _int_setting(seed, "seed")
        self.materialized = _bool_setting(materialized, "materialized")
        self._features: Optional[torch.Tensor] = None
        self._targets: Optional[torch.Tensor] = None
        if self.materialized:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            self._features = torch.randn(self.size, self.features, generator=generator, device="cpu")
            self._targets = torch.randint(0, self.classes, (self.size,), generator=generator, device="cpu")

    def __len__(self) -> int:
        return self.size

    @property
    def materialized_bytes(self) -> int:
        if not self.materialized or self._features is None or self._targets is None:
            return 0
        feature_bytes = self._features.numel() * self._features.element_size()
        target_bytes = self._targets.numel() * self._targets.element_size()
        return int(feature_bytes + target_bytes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        if self.materialized:
            features = self._features
            targets = self._targets
            if features is None or targets is None:
                raise RuntimeError("materialized dataset storage was not initialized")
            return features[index], targets[index]
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        features = torch.randn(self.features, generator=generator, device="cpu")
        target = torch.randint(0, self.classes, (1,), generator=generator, device="cpu").squeeze(0)
        return features, target


def build_model(features: int, classes: int) -> nn.Module:
    features = _positive_int_setting(features, "features")
    classes = _positive_int_setting(classes, "classes")
    hidden = max(32, features * 2)
    return nn.Sequential(
        nn.Linear(features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    )


def run_once(args, run_index: int) -> BenchmarkResult:
    validate_benchmark_args(args)
    run_seed = _int_setting(args.seed, "seed") + _non_negative_int_setting(run_index, "run_index")
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    setup_start = time.perf_counter()
    dataset_start = time.perf_counter()
    dataset = SyntheticTransactionDataset(
        size=args.transactions,
        features=args.feature_dim,
        classes=args.num_classes,
        seed=run_seed,
        materialized=args.dataset_mode == "materialized",
    )
    dataset_setup_time_s = time.perf_counter() - dataset_start

    loader_start = time.perf_counter()
    loader = dataloader_from_dataset(
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent=args.workers > 0,
        seed=run_seed,
        shuffle=True,
    )
    loader_setup_time_s = time.perf_counter() - loader_start

    model_start = time.perf_counter()
    model = build_model(args.feature_dim, args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = FastTrainer(
        model,
        optimizer,
        scheduler=None,
        device=args.device,
        use_compile=args.compile,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
    )
    model_setup_time_s = time.perf_counter() - model_start
    setup_time_s = time.perf_counter() - setup_start

    start = time.perf_counter()
    metrics = trainer.train_one_epoch(
        loader,
        criterion,
        steps=args.steps,
        collect_profile=args.collect_profile,
        profile_sync=args.profile_sync,
        profile_distribution=args.profile_distribution,
        profile_window=args.profile_window,
        profile_model=args.profile_model,
        profile_model_depth=args.profile_model_depth,
        profile_model_max_modules=args.profile_model_max_modules,
        profile_model_include=args.profile_model_include,
        warmup_steps=args.warmup_steps,
    )
    wall = time.perf_counter() - start
    metrics.update({
        "batch_size": args.batch_size,
        "dataset_materialized_bytes": dataset.materialized_bytes,
        "dataset_setup_time_s": dataset_setup_time_s,
        "num_workers": args.workers,
        "loader_setup_time_s": loader_setup_time_s,
        "model_setup_time_s": model_setup_time_s,
        "setup_time_s": setup_time_s,
        "end_to_end_wall_time_s": setup_time_s + wall,
        "transactions": args.transactions,
        "dataset_mode": args.dataset_mode,
        "seed": run_seed,
    })
    return BenchmarkResult(wall_time_s=wall, trainer_metrics=metrics, run_index=run_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=positive_int_arg, default=100_000, help="Total synthetic transactions to sample.")
    parser.add_argument("--feature-dim", type=positive_int_arg, default=128, help="Width of each synthetic transaction vector.")
    parser.add_argument("--num-classes", type=positive_int_arg, default=32, help="Number of synthetic classification targets.")
    parser.add_argument(
        "--dataset-mode",
        choices=["generated", "materialized"],
        default="generated",
        help="Generate each sample on demand or precompute tensors once to reduce DataLoader noise.",
    )
    parser.add_argument("--batch-size", type=positive_int_arg, default=512, help="Batch size for the benchmark dataloader.")
    parser.add_argument("--grad-accum", type=positive_int_arg, default=2, help="Gradient accumulation factor.")
    parser.add_argument("--workers", type=non_negative_int_arg, default=4, help="Number of dataloader worker processes.")
    parser.add_argument("--prefetch-factor", type=positive_int_arg, default=4, help="Prefetch factor passed to the dataloader.")
    parser.add_argument("--device", type=device_arg, default="auto", help="Device override (auto/cuda/mps/cpu).")
    parser.add_argument("--steps", type=positive_int_arg, default=200, help="Number of training steps per run.")
    parser.add_argument("--log-interval", type=non_negative_int_arg, default=0, help="Step log interval; 0 disables step logs.")
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile for lower cold-start cost.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=non_negative_int_arg,
        default=0,
        help="Measure the first N steps separately and exclude them from steady-state throughput.",
    )
    parser.add_argument("--runs", type=positive_int_arg, default=3, help="How many repeated runs to execute.")
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4, help="Learning rate for the synthetic model.")
    parser.add_argument("--seed", type=_int_arg, default=1234, help="Base random seed for synthetic data.")
    parser.add_argument("--collect-profile", action="store_true", help="Collect train-loop phase timings.")
    parser.add_argument("--profile-sync", action="store_true", help="Synchronize accelerator around profiled phases.")
    parser.add_argument(
        "--no-profile-distribution",
        dest="profile_distribution",
        action="store_false",
        help="Skip p50/p95/p99/std samples and collect phase totals only.",
    )
    parser.add_argument("--profile-window", type=positive_int_arg, default=512, help="Per-phase sample window size.")
    parser.add_argument("--profile-model", action="store_true", help="Collect module-level forward/backward drilldowns.")
    parser.add_argument("--profile-model-depth", type=positive_int_arg, default=1, help="Exact module depth to profile.")
    parser.add_argument("--profile-model-max-modules", type=positive_int_arg, default=64, help="Maximum modules to hook.")
    parser.add_argument(
        "--profile-model-include",
        type=str,
        default=None,
        help="Comma list of module names/globs to include, for example 0,2 or 0.*.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to dump the benchmark results as JSON for dashboards.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Optional path to dump aggregate benchmark stats as JSON.",
    )
    args = parser.parse_args()
    try:
        validate_benchmark_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    if args.profile_model:
        args.collect_profile = True
    results = []
    for run_index in range(args.runs):
        result = run_once(args, run_index)
        results.append(result)
        metrics = result.as_dict()
        print(
            f"Run {run_index}: "
            f"wall={_format_metric_value(metrics.get('wall_time_s'), precision=2, suffix='s')} "
            f"setup={_format_metric_value(metrics.get('setup_time_s'), precision=2, suffix='s')} "
            f"e2e={_format_metric_value(metrics.get('end_to_end_wall_time_s', metrics.get('wall_time_s')), precision=2, suffix='s')} "
            f"thr={_format_metric_value(metrics.get('reported_samples_per_sec', metrics.get('samples_per_sec')), precision=1, suffix='/s')} "
            f"total_thr={_format_metric_value(metrics.get('samples_per_sec'), precision=1, suffix='/s')} "
            f"p99_batch={_format_metric_value(metrics.get('p99_s'), precision=2, scale=1e3, suffix='ms')} "
            f"std_batch={_format_metric_value(metrics.get('std_batch_s'), precision=2, scale=1e3, suffix='ms')} "
            f"avg_loss={_format_metric_value(metrics.get('avg_loss'), precision=4)}"
        )
        if _has_positive_display_value(metrics.get("warmup_steps")):
            print(
                f"  cold_start: steps={_format_count(metrics.get('cold_start_steps'))} "
                f"time={_format_metric_value(metrics.get('cold_start_time_s'), precision=2, suffix='s')} "
                f"thr={_format_metric_value(metrics.get('cold_start_samples_per_sec'), precision=1, suffix='/s')}"
            )
        if _has_positive_display_value(metrics.get("steady_steps")):
            print(
                f"  steady: steps={_format_count(metrics.get('steady_steps'))} "
                f"thr={_format_metric_value(metrics.get('steady_samples_per_sec'), precision=1, suffix='/s')} "
                f"p99_batch={_format_metric_value(metrics.get('steady_p99_s'), precision=2, scale=1e3, suffix='ms')}"
            )
        profile = _dict_value(metrics.get("profile"))
        if profile:
            top_phases = ", ".join(
                f"{_profile_row_name(row)}={_format_metric_value(row.get('pct'), precision=1, suffix='%')}"
                for row in _list_value(profile.get("top_phases"))[:4]
                if isinstance(row, dict)
            )
            print(f"  phases: {top_phases}")
            forward = _profile_child_rows(profile, "phase_breakdowns", "forward")
            if forward:
                top_forward = ", ".join(
                    f"{_profile_row_name(row)}="
                    f"{_format_metric_value(row.get('pct_of_parent'), precision=1, suffix='%')}"
                    for row in forward[:4]
                    if isinstance(row, dict)
                )
                print(f"  forward: {top_forward}")
            backward = _profile_child_rows(profile, "phase_events", "backward_grad_ready")
            if backward:
                top_backward = ", ".join(
                    f"{_profile_row_name(row)}={_format_metric_value(row.get('avg_ms'), precision=1, suffix='ms')}"
                    for row in backward[:4]
                    if isinstance(row, dict)
                )
                print(f"  backward_grad_ready: {top_backward}")
            optimizer = _profile_child_rows(profile, "phase_breakdowns", "optimizer")
            if optimizer:
                top_optimizer = ", ".join(
                    f"{_profile_row_name(row)}="
                    f"{_format_metric_value(row.get('pct_of_parent'), precision=1, suffix='%')}"
                    for row in optimizer[:4]
                    if isinstance(row, dict)
                )
                print(f"  optimizer: {top_optimizer}")

    payload = [result.as_dict() for result in results]
    aggregate = summarize_results(payload)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            dump_json(payload, handle)
        print(f"Wrote results to {out_path}")

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            dump_json(aggregate, handle)
        print(f"Wrote aggregate summary to {out_path}")

    print("Aggregate:", dumps_json(aggregate))


if __name__ == "__main__":
    main()
