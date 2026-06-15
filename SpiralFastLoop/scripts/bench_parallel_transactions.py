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
    _non_negative_finite_float_setting,
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
CRITICAL_SUMMARY_MISSING_FIELDS = frozenset({
    "reported_samples_per_sec",
    "end_to_end_wall_time_s",
})

SETUP_SUMMARY_FIELDS = (
    "dataset_setup_time_s",
    "loader_setup_time_s",
    "model_setup_time_s",
    "compile_init_time_s",
    "dataset_materialized_bytes",
)

CONFIG_SUMMARY_FIELDS = (
    "transactions",
    "batch_size",
    "num_workers",
    "world_size",
)

BATCH_SUMMARY_FIELDS = (
    "p50_s",
    "p95_s",
    "p99_s",
    "std_batch_s",
    "avg_batch_s",
    "last_batch_s",
    "min_batch_s",
    "max_batch_s",
    "batches",
    "best_samples_per_sec",
    "headroom_ratio",
    "ema_samples_per_sec",
    "window_samples_per_sec",
    "window_time_s",
    "window_batches",
    "window_samples",
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
DIAGNOSTIC_SUMMARY_FIELDS = (
    "scheduler_step_failures",
)
WORKLOAD_INTEGER_SUMMARY_FIELDS = frozenset({
    "steps",
    "samples",
    "optimizer_steps",
    "grad_accum",
    "partial_optimizer_steps",
    "grad_accum_tail_steps",
    "warmup_steps",
    "warmup_samples",
    "warmup_optimizer_steps",
    "cold_start_steps",
    "steady_steps",
    "steady_samples",
    "steady_optimizer_steps",
    "scheduler_step_failures",
})

PROFILE_PHASE_SUMMARY_FIELDS = (
    "profile_total_s",
    "profile_flat_metric_invalid_count",
    "profile_open_phase_count",
    "profile_open_detail_count",
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

PROFILE_DISTRIBUTION_SUMMARY_FIELD_NAMES = (
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "std_ms",
    "min_ms",
    "max_ms",
)
PROFILE_DISTRIBUTION_COUNT_SUMMARY_FIELD_NAMES = (
    "sample_count",
    "window_sample_count",
)
PROFILE_PHASE_SUMMARY_FIELD_NAMES = (
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
PROFILE_PHASE_DISTRIBUTION_SUMMARY_FIELDS = tuple(
    f"profile_{phase_name}_{field_name}"
    for phase_name in PROFILE_PHASE_SUMMARY_FIELD_NAMES
    for field_name in PROFILE_DISTRIBUTION_SUMMARY_FIELD_NAMES + PROFILE_DISTRIBUTION_COUNT_SUMMARY_FIELD_NAMES
)
PROFILE_PHASE_COUNT_SUMMARY_FIELDS = tuple(
    f"profile_{phase_name}_{field_name}"
    for phase_name in PROFILE_PHASE_SUMMARY_FIELD_NAMES
    for field_name in PROFILE_DISTRIBUTION_COUNT_SUMMARY_FIELD_NAMES
)


def _profile_top_summary_fields(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_top_{field_name}"
        for field_name in PROFILE_DISTRIBUTION_SUMMARY_FIELD_NAMES + PROFILE_DISTRIBUTION_COUNT_SUMMARY_FIELD_NAMES
    )


PROFILE_FORWARD_BREAKDOWN_SUMMARY_FIELDS = (
    "profile_forward_child_count",
    "profile_forward_tracked_time_s",
    "profile_forward_untracked_time_s",
    "profile_forward_overtracked_time_s",
    "profile_forward_coverage_pct",
    "profile_forward_untracked_pct",
    "profile_forward_overtracked_pct_of_parent",
    "profile_forward_top_time_s",
    "profile_forward_top_pct_of_parent",
    "profile_forward_top_avg_ms",
    *_profile_top_summary_fields("profile_forward"),
    "profile_forward_top_calls",
)

PROFILE_OPTIMIZER_BREAKDOWN_SUMMARY_FIELDS = (
    "profile_optimizer_child_count",
    "profile_optimizer_tracked_time_s",
    "profile_optimizer_untracked_time_s",
    "profile_optimizer_overtracked_time_s",
    "profile_optimizer_coverage_pct",
    "profile_optimizer_untracked_pct",
    "profile_optimizer_overtracked_pct_of_parent",
    "profile_optimizer_top_time_s",
    "profile_optimizer_top_pct_of_parent",
    "profile_optimizer_top_avg_ms",
    *_profile_top_summary_fields("profile_optimizer"),
    "profile_optimizer_top_calls",
)

PROFILE_EVENT_SUMMARY_FIELDS = (
    "profile_backward_grad_ready_child_count",
    "profile_backward_grad_ready_parent_avg_ms",
    "profile_backward_grad_ready_earliest_avg_ms",
    "profile_backward_grad_ready_latest_avg_ms",
    "profile_backward_grad_ready_span_avg_ms",
    "profile_backward_grad_ready_earliest_pct",
    "profile_backward_grad_ready_latest_pct",
    "profile_backward_grad_ready_span_pct",
    "profile_backward_grad_ready_top_avg_ms",
    "profile_backward_grad_ready_top_pct",
    *_profile_top_summary_fields("profile_backward_grad_ready"),
    "profile_backward_grad_ready_top_calls",
)

PROFILE_MODEL_SUMMARY_FIELDS = (
    "profile_model_modules_selected",
    "profile_model_hook_count",
    "profile_model_hook_failures",
)

PROFILE_SUMMARY_FIELDS = (
    PROFILE_PHASE_SUMMARY_FIELDS
    + PROFILE_PHASE_DISTRIBUTION_SUMMARY_FIELDS
    + PROFILE_FORWARD_BREAKDOWN_SUMMARY_FIELDS
    + PROFILE_OPTIMIZER_BREAKDOWN_SUMMARY_FIELDS
    + PROFILE_EVENT_SUMMARY_FIELDS
    + PROFILE_MODEL_SUMMARY_FIELDS
)
BOUNDED_PERCENT_SUMMARY_FIELDS = frozenset(
    field for field in PROFILE_SUMMARY_FIELDS
    if field.endswith("_pct")
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
SUMMARY_INTEGER_FIELDS = (
    WORKLOAD_INTEGER_SUMMARY_FIELDS
    | frozenset(PROFILE_PHASE_COUNT_SUMMARY_FIELDS)
    | frozenset({
        "batches",
        "batch_size",
        "num_workers",
        "dataset_materialized_bytes",
        "transactions",
        "window_batches",
        "window_samples",
        "world_size",
        "profile_flat_metric_invalid_count",
        "profile_open_phase_count",
        "profile_open_detail_count",
        "profile_forward_child_count",
        "profile_forward_top_sample_count",
        "profile_forward_top_window_sample_count",
        "profile_forward_top_calls",
        "profile_optimizer_child_count",
        "profile_optimizer_top_sample_count",
        "profile_optimizer_top_window_sample_count",
        "profile_optimizer_top_calls",
        "profile_backward_grad_ready_child_count",
        "profile_backward_grad_ready_top_sample_count",
        "profile_backward_grad_ready_top_window_sample_count",
        "profile_backward_grad_ready_top_calls",
        "profile_model_modules_selected",
        "profile_model_hook_count",
        "profile_model_hook_failures",
    })
    | frozenset(DEVICE_MEMORY_SUMMARY_FIELDS)
)

SUMMARY_FIELDS = (
    BASE_SUMMARY_FIELDS
    + SETUP_SUMMARY_FIELDS
    + CONFIG_SUMMARY_FIELDS
    + BATCH_SUMMARY_FIELDS
    + WORKLOAD_SUMMARY_FIELDS
    + DIAGNOSTIC_SUMMARY_FIELDS
    + PROFILE_SUMMARY_FIELDS
    + DEVICE_MEMORY_SUMMARY_FIELDS
)

BEST_RUN_FIELDS = (
    "run",
    "seed",
    "dataset_mode",
    "device",
    "amp",
    "compile_requested",
    "compiled",
    "compile_init_time_s",
    "compile_fallback_reason",
    "transactions",
    "batch_size",
    "num_workers",
    "world_size",
    "rank",
    "reported_samples_per_sec",
    "samples_per_sec",
    "steady_samples_per_sec",
    "p99_s",
    "std_batch_s",
    "avg_batch_s",
    "last_batch_s",
    "min_batch_s",
    "max_batch_s",
    "batches",
    "best_samples_per_sec",
    "headroom_ratio",
    "ema_samples_per_sec",
    "window_samples_per_sec",
    "window_time_s",
    "window_batches",
    "window_samples",
    "end_to_end_wall_time_s",
    "setup_time_s",
    "dataset_setup_time_s",
    "loader_setup_time_s",
    "model_setup_time_s",
    "dataset_materialized_bytes",
    "wall_time_s",
    "steps",
    "samples",
    "optimizer_steps",
    "grad_accum",
    "partial_optimizer_steps",
    "grad_accum_tail_steps",
    "scheduler_step_failures",
    "scheduler_last_error",
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
    "profile_open_phase_count",
    "profile_open_detail_count",
    "profile_model_requested",
    "profile_model_enabled",
    "profile_model_status",
    "profile_model_modules_selected",
    "profile_model_hook_count",
    "profile_model_hook_failures",
    "profile_model_hook_last_error",
    "profile_forward_backward_pct",
    "profile_forward_backward_time_s",
    "profile_forward_pct",
    "profile_forward_child_count",
    "profile_forward_tracked_time_s",
    "profile_forward_untracked_time_s",
    "profile_forward_overtracked_time_s",
    "profile_forward_coverage_pct",
    "profile_forward_untracked_pct",
    "profile_forward_overtracked_pct_of_parent",
    "profile_forward_top_time_s",
    "profile_forward_top_pct_of_parent",
    "profile_forward_top_avg_ms",
    "profile_forward_top_p50_ms",
    "profile_forward_top_p95_ms",
    "profile_forward_top_p99_ms",
    "profile_forward_top_std_ms",
    "profile_forward_top_min_ms",
    "profile_forward_top_max_ms",
    "profile_forward_top_sample_count",
    "profile_forward_top_window_sample_count",
    "profile_forward_top_calls",
    "profile_loss_pct",
    "profile_loss_reduce_pct",
    "profile_backward_pct",
    "profile_optimizer_pct",
    "profile_optimizer_child_count",
    "profile_optimizer_tracked_time_s",
    "profile_optimizer_untracked_time_s",
    "profile_optimizer_overtracked_time_s",
    "profile_optimizer_coverage_pct",
    "profile_optimizer_untracked_pct",
    "profile_optimizer_overtracked_pct_of_parent",
    "profile_optimizer_top_time_s",
    "profile_optimizer_top_pct_of_parent",
    "profile_optimizer_top_avg_ms",
    "profile_optimizer_top_p50_ms",
    "profile_optimizer_top_p95_ms",
    "profile_optimizer_top_p99_ms",
    "profile_optimizer_top_std_ms",
    "profile_optimizer_top_min_ms",
    "profile_optimizer_top_max_ms",
    "profile_optimizer_top_sample_count",
    "profile_optimizer_top_window_sample_count",
    "profile_optimizer_top_calls",
    "profile_user_metrics_pct",
    "profile_postprocess_pct",
    "profile_collect_output_pct",
    "profile_metrics_pct",
    "profile_backward_grad_ready_child_count",
    "profile_backward_grad_ready_parent_avg_ms",
    "profile_backward_grad_ready_earliest_avg_ms",
    "profile_backward_grad_ready_latest_avg_ms",
    "profile_backward_grad_ready_span_avg_ms",
    "profile_backward_grad_ready_earliest_pct",
    "profile_backward_grad_ready_latest_pct",
    "profile_backward_grad_ready_span_pct",
    "profile_backward_grad_ready_top_avg_ms",
    "profile_backward_grad_ready_top_pct",
    "profile_backward_grad_ready_top_p50_ms",
    "profile_backward_grad_ready_top_p95_ms",
    "profile_backward_grad_ready_top_p99_ms",
    "profile_backward_grad_ready_top_std_ms",
    "profile_backward_grad_ready_top_min_ms",
    "profile_backward_grad_ready_top_max_ms",
    "profile_backward_grad_ready_top_sample_count",
    "profile_backward_grad_ready_top_window_sample_count",
    "profile_backward_grad_ready_top_calls",
    *PROFILE_PHASE_DISTRIBUTION_SUMMARY_FIELDS,
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
PROFILE_MODEL_STATUS_ORDER = (
    "not_requested",
    "collect_profile_disabled",
    "no_matching_modules",
    "hook_failures",
    "ok",
)
PROFILE_MODEL_STATUS_CHOICES = frozenset(PROFILE_MODEL_STATUS_ORDER)
BEST_RUN_TEXT_FIELDS = frozenset({
    "compile_fallback_reason",
    "dataset_mode",
    "device",
    "profile_model_status",
    "profile_model_hook_last_error",
    "scheduler_last_error",
})
BEST_RUN_BOOL_FIELDS = frozenset({
    "amp",
    "compile_requested",
    "compiled",
    "profile_model_requested",
    "profile_model_enabled",
})
BEST_RUN_INTEGER_FIELDS = frozenset({
    "run",
    "batches",
    "batch_size",
    "dataset_materialized_bytes",
    "num_workers",
    "rank",
    "transactions",
    "window_batches",
    "window_samples",
    "world_size",
    "steps",
    "samples",
    "optimizer_steps",
    "grad_accum",
    "partial_optimizer_steps",
    "grad_accum_tail_steps",
    "scheduler_step_failures",
    "warmup_steps",
    "warmup_samples",
    "warmup_optimizer_steps",
    "cold_start_steps",
    "steady_steps",
    "steady_samples",
    "steady_optimizer_steps",
    *PROFILE_PHASE_COUNT_SUMMARY_FIELDS,
    "profile_flat_metric_invalid_count",
    "profile_open_phase_count",
    "profile_open_detail_count",
    "profile_forward_child_count",
    "profile_forward_top_sample_count",
    "profile_forward_top_window_sample_count",
    "profile_forward_top_calls",
    "profile_optimizer_child_count",
    "profile_optimizer_top_sample_count",
    "profile_optimizer_top_window_sample_count",
    "profile_optimizer_top_calls",
    "profile_backward_grad_ready_child_count",
    "profile_backward_grad_ready_top_sample_count",
    "profile_backward_grad_ready_top_window_sample_count",
    "profile_backward_grad_ready_top_calls",
    "profile_model_modules_selected",
    "profile_model_hook_count",
    "profile_model_hook_failures",
    "cuda_current_mem_bytes",
    "cuda_max_mem_bytes",
    "cuda_reserved_mem_bytes",
    "cuda_max_reserved_mem_bytes",
    "mps_current_mem_bytes",
    "mps_max_mem_bytes",
    "mps_driver_mem_bytes",
    "mps_recommended_max_mem_bytes",
})
BEST_RUN_POSITIVE_INTEGER_FIELDS = frozenset({
    "batch_size",
    "grad_accum",
    "transactions",
    "world_size",
})
BEST_RUN_POSITIVE_ONLY_INTEGER_FIELDS = frozenset({"scheduler_step_failures"})
PROFILE_MODEL_RESULT_FIELDS = frozenset({
    "profile_model_requested",
    "profile_model_enabled",
    "profile_model_status",
    "profile_model_modules_selected",
    "profile_model_hook_count",
    "profile_model_hook_failures",
    "profile_model_hook_last_error",
})
PROFILE_BOTTLENECK_CANDIDATE_LIMIT = 8
PROFILE_BOTTLENECK_SEVERITY_LEVELS = (
    ("high", 25.0),
    ("medium", 10.0),
    ("low", 0.0),
)
PROFILE_BOTTLENECK_SEVERITY_SCORE_UNIT = "profile_pct"
PROFILE_BOTTLENECK_CANDIDATE_SPECS = (
    {
        "name": "forward_phase",
        "label": "forward phase",
        "category": "phase_share",
        "metric": "profile_forward_pct",
        "unit": "profile_pct",
        "reason": "forward owns a large share of profiled loop time",
        "next_step": "inspect forward top-child and tail metrics",
        "details": (
            ("avg_ms", "profile_forward_avg_ms"),
            ("p95_ms", "profile_forward_p95_ms"),
            ("p99_ms", "profile_forward_p99_ms"),
            ("std_ms", "profile_forward_std_ms"),
        ),
    },
    {
        "name": "backward_phase",
        "label": "backward phase",
        "category": "phase_share",
        "metric": "profile_backward_pct",
        "unit": "profile_pct",
        "reason": "backward owns a large share of profiled loop time",
        "next_step": "inspect gradient-ready span and backward top-child metrics",
        "details": (
            ("avg_ms", "profile_backward_avg_ms"),
            ("p95_ms", "profile_backward_p95_ms"),
            ("p99_ms", "profile_backward_p99_ms"),
            ("std_ms", "profile_backward_std_ms"),
        ),
    },
    {
        "name": "optimizer_phase",
        "label": "optimizer phase",
        "category": "phase_share",
        "metric": "profile_optimizer_pct",
        "unit": "profile_pct",
        "reason": "optimizer owns a large share of profiled loop time",
        "next_step": "inspect optimizer top-child and tail metrics",
        "details": (
            ("avg_ms", "profile_optimizer_avg_ms"),
            ("p95_ms", "profile_optimizer_p95_ms"),
            ("p99_ms", "profile_optimizer_p99_ms"),
            ("std_ms", "profile_optimizer_std_ms"),
        ),
    },
    {
        "name": "forward_untracked",
        "label": "forward untracked time",
        "category": "coverage_gap",
        "metric": "profile_forward_untracked_pct",
        "parent_metric": "profile_forward_pct",
        "unit": "pct_of_parent",
        "reason": "a large share of forward time is outside child timers",
        "next_step": "increase model profiling coverage or include narrower module filters",
        "details": (
            ("coverage_pct", "profile_forward_coverage_pct"),
            ("untracked_time_s", "profile_forward_untracked_time_s"),
        ),
    },
    {
        "name": "optimizer_untracked",
        "label": "optimizer untracked time",
        "category": "coverage_gap",
        "metric": "profile_optimizer_untracked_pct",
        "parent_metric": "profile_optimizer_pct",
        "unit": "pct_of_parent",
        "reason": "a large share of optimizer time is outside child timers",
        "next_step": "add or inspect optimizer child timers before optimizing",
        "details": (
            ("coverage_pct", "profile_optimizer_coverage_pct"),
            ("untracked_time_s", "profile_optimizer_untracked_time_s"),
        ),
    },
    {
        "name": "forward_top_child",
        "label": "forward top child",
        "category": "child_hotspot",
        "metric": "profile_forward_top_pct_of_parent",
        "parent_metric": "profile_forward_pct",
        "unit": "pct_of_parent",
        "reason": "one forward child dominates its parent phase",
        "next_step": "drill into selected modules with depth/include filters",
        "details": (
            ("avg_ms", "profile_forward_top_avg_ms"),
            ("p95_ms", "profile_forward_top_p95_ms"),
            ("p99_ms", "profile_forward_top_p99_ms"),
            ("calls", "profile_forward_top_calls"),
            ("child_count", "profile_forward_child_count"),
            ("overtracked_pct_of_parent", "profile_forward_overtracked_pct_of_parent"),
        ),
    },
    {
        "name": "backward_readiness_span",
        "label": "backward readiness span",
        "category": "readiness_span",
        "metric": "profile_backward_grad_ready_span_pct",
        "parent_metric": "profile_backward_pct",
        "unit": "pct_of_parent",
        "reason": "gradient readiness is spread across a large part of backward time",
        "next_step": "look for long gaps between earliest and latest ready modules",
        "details": (
            ("span_avg_ms", "profile_backward_grad_ready_span_avg_ms"),
            ("earliest_pct", "profile_backward_grad_ready_earliest_pct"),
            ("latest_pct", "profile_backward_grad_ready_latest_pct"),
            ("earliest_avg_ms", "profile_backward_grad_ready_earliest_avg_ms"),
            ("latest_avg_ms", "profile_backward_grad_ready_latest_avg_ms"),
            ("child_count", "profile_backward_grad_ready_child_count"),
        ),
    },
    {
        "name": "backward_ready_top_child",
        "label": "backward ready top child",
        "category": "child_hotspot",
        "metric": "profile_backward_grad_ready_top_pct",
        "parent_metric": "profile_backward_pct",
        "unit": "pct_of_parent",
        "reason": "one module dominates gradient-ready timing",
        "next_step": "focus backward inspection on the slowest ready module",
        "details": (
            ("avg_ms", "profile_backward_grad_ready_top_avg_ms"),
            ("p95_ms", "profile_backward_grad_ready_top_p95_ms"),
            ("p99_ms", "profile_backward_grad_ready_top_p99_ms"),
            ("calls", "profile_backward_grad_ready_top_calls"),
            ("child_count", "profile_backward_grad_ready_child_count"),
        ),
    },
    {
        "name": "optimizer_top_child",
        "label": "optimizer top child",
        "category": "child_hotspot",
        "metric": "profile_optimizer_top_pct_of_parent",
        "parent_metric": "profile_optimizer_pct",
        "unit": "pct_of_parent",
        "reason": "one optimizer child dominates its parent phase",
        "next_step": "compare optimizer child timing and parameter-group behavior",
        "details": (
            ("avg_ms", "profile_optimizer_top_avg_ms"),
            ("p95_ms", "profile_optimizer_top_p95_ms"),
            ("p99_ms", "profile_optimizer_top_p99_ms"),
            ("calls", "profile_optimizer_top_calls"),
            ("child_count", "profile_optimizer_child_count"),
            ("overtracked_pct_of_parent", "profile_optimizer_overtracked_pct_of_parent"),
        ),
    },
)


@dataclass
class BenchmarkResult:
    wall_time_s: float
    trainer_metrics: dict
    run_index: int

    def as_dict(self) -> dict:
        payload = dict(self.trainer_metrics)
        payload["wall_time_s"] = self.wall_time_s
        payload["run"] = self.run_index
        return payload


def _summary_row(row: dict) -> dict:
    normalized = dict(row)
    if not _profile_model_fields_visible(normalized):
        for field in PROFILE_MODEL_RESULT_FIELDS:
            normalized.pop(field, None)
    if "reported_samples_per_sec" not in normalized:
        if "steady_samples_per_sec" in normalized:
            normalized["reported_samples_per_sec"] = normalized["steady_samples_per_sec"]
        elif "samples_per_sec" in normalized:
            normalized["reported_samples_per_sec"] = normalized["samples_per_sec"]
    if "end_to_end_wall_time_s" not in normalized and "wall_time_s" in normalized:
        normalized["end_to_end_wall_time_s"] = normalized["wall_time_s"]
    return normalized


def _profile_model_fields_visible(row: dict) -> bool:
    if row.get("profile_model_requested") is True:
        return True
    status_raw = row.get("profile_model_status")
    status = status_raw.strip() if isinstance(status_raw, str) else ""
    return bool(status) and status != "not_requested"


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


def _profile_bottleneck_severity(score: float) -> str:
    for severity, threshold in PROFILE_BOTTLENECK_SEVERITY_LEVELS:
        if score >= threshold:
            return severity
    return "low"


def _profile_bottleneck_severity_thresholds() -> dict[str, object]:
    return {
        "score_unit": PROFILE_BOTTLENECK_SEVERITY_SCORE_UNIT,
        "levels": [
            {"severity": severity, "min_score": threshold}
            for severity, threshold in PROFILE_BOTTLENECK_SEVERITY_LEVELS
        ],
    }


def _measured_summary_metric_value(summary: dict, metric: str) -> Optional[float]:
    field = f"mean_{metric}"
    if field not in summary:
        field = metric
    if field not in summary:
        return None
    value = _finite_summary_value(summary[field])
    if value is None:
        return None
    if value < _summary_metric_min_value(metric):
        return None
    max_value = _summary_metric_max_value(metric)
    if max_value is not None and value > max_value:
        return None

    sample_count_field = f"sample_count_{metric}"
    if field.startswith("mean_") and sample_count_field in summary:
        sample_count = _positive_sample_count_value(summary[sample_count_field])
        if sample_count is None:
            return None
    return value


def _profile_bottleneck_candidate(summary: dict, spec: dict[str, object]) -> Optional[dict[str, object]]:
    metric = spec["metric"]
    if not isinstance(metric, str):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        return None
    value = _measured_summary_metric_value(summary, metric)
    if value is None or value <= 0.0:
        return None

    unit = spec.get("unit", "profile_pct")
    score = value
    score_unit = unit
    candidate: dict[str, object] = {
        "name": name,
        "metric": metric,
        "value": value,
        "unit": unit,
    }
    for text_field in ("label", "category", "reason", "next_step"):
        text_value = spec.get(text_field)
        if isinstance(text_value, str) and text_value:
            candidate[text_field] = text_value
    parent_metric = spec.get("parent_metric")
    if isinstance(parent_metric, str):
        parent_value = _measured_summary_metric_value(summary, parent_metric)
        if parent_value is None or parent_value <= 0.0:
            return None
        score = parent_value * value / 100.0
        score_unit = "profile_pct"
        candidate["parent_metric"] = parent_metric
        candidate["parent_value"] = parent_value
    if score <= 0.0:
        return None
    candidate["score"] = score
    candidate["score_unit"] = score_unit
    candidate["severity"] = _profile_bottleneck_severity(score)

    details = spec.get("details", ())
    if isinstance(details, tuple):
        for detail in details:
            if not isinstance(detail, tuple) or len(detail) != 2:
                continue
            key, detail_metric = detail
            if not isinstance(key, str) or not isinstance(detail_metric, str):
                continue
            detail_value = _measured_summary_metric_value(summary, detail_metric)
            if detail_value is not None:
                candidate[key] = detail_value
    return candidate


def _ranked_profile_bottleneck_candidates(summary: dict) -> list[dict[str, object]]:
    candidates = []
    for spec in PROFILE_BOTTLENECK_CANDIDATE_SPECS:
        candidate = _profile_bottleneck_candidate(summary, spec)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (
            -float(candidate["score"]),
            -float(candidate["value"]),
            str(candidate["name"]),
        )
    )
    return [
        {**candidate, "rank": rank}
        for rank, candidate in enumerate(candidates, start=1)
    ]


def profile_bottleneck_candidates_for_summary(summary: dict) -> list[dict[str, object]]:
    return _ranked_profile_bottleneck_candidates(summary)[:PROFILE_BOTTLENECK_CANDIDATE_LIMIT]


def _profile_bottleneck_category_summary(
    candidates: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    category_summary: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        category_raw = candidate.get("category")
        category = category_raw if isinstance(category_raw, str) and category_raw else "uncategorized"
        score = _finite_summary_value(candidate.get("score"))
        if score is None:
            continue
        entry = category_summary.setdefault(
            category,
            {
                "count": 0,
                "max_score": score,
                "total_score": 0.0,
                "mean_score": 0.0,
                "pressure_score": 0.0,
                "pressure_score_unit": candidate.get("score_unit", ""),
                "score_unit": candidate.get("score_unit", ""),
                "top_candidate": candidate.get("name", ""),
                "top_rank": candidate.get("rank", 0),
                "top_severity": candidate.get("severity", ""),
                "severity_counts": {},
            },
        )
        entry["count"] = int(entry["count"]) + 1
        total_score = _finite_summary_value(entry.get("total_score"))
        entry["total_score"] = (total_score or 0.0) + score
        severity = candidate.get("severity")
        severity_counts = entry.get("severity_counts")
        if isinstance(severity, str) and severity and isinstance(severity_counts, dict):
            severity_counts[severity] = int(severity_counts.get(severity, 0)) + 1
        max_score = _finite_summary_value(entry.get("max_score"))
        if max_score is None or score > max_score:
            entry["max_score"] = score
            entry["score_unit"] = candidate.get("score_unit", "")
            entry["top_candidate"] = candidate.get("name", "")
            entry["top_rank"] = candidate.get("rank", 0)
            entry["top_severity"] = candidate.get("severity", "")
    for entry in category_summary.values():
        count = int(entry.get("count", 0))
        total_score = _finite_summary_value(entry.get("total_score"))
        if count > 0 and total_score is not None:
            entry["mean_score"] = total_score / count
        pressure_score = _profile_bottleneck_category_pressure_score(entry)
        if pressure_score is not None:
            entry["pressure_score"] = pressure_score
            entry["pressure_score_unit"] = entry.get("score_unit", "")
        severity_counts = entry.get("severity_counts")
        if isinstance(severity_counts, dict):
            entry["severity_counts"] = _ordered_profile_bottleneck_severity_counts(
                severity_counts
            )
    return category_summary


def _ranked_profile_bottleneck_category_items(
    category_summary: dict[str, object],
) -> list[tuple[str, dict[str, object]]]:
    ranked_items: list[tuple[str, dict[str, object]]] = []
    for category_name, raw_entry in category_summary.items():
        if not isinstance(category_name, str) or not isinstance(raw_entry, dict):
            continue
        max_score = _finite_summary_value(raw_entry.get("max_score"))
        top_candidate = raw_entry.get("top_candidate")
        if (
            max_score is None
            or max_score < 0.0
            or not isinstance(top_candidate, str)
            or not top_candidate
        ):
            continue
        ranked_items.append((category_name, raw_entry))

    ranked_items.sort(
        key=lambda item: _profile_bottleneck_category_rank_key(item[0], item[1])
    )
    return ranked_items


def _profile_bottleneck_category_pressure_score(
    entry: dict[str, object],
) -> Optional[float]:
    total_score = _finite_summary_value(entry.get("total_score"))
    if total_score is not None and total_score >= 0.0:
        return total_score

    max_score = _finite_summary_value(entry.get("max_score"))
    if max_score is not None and max_score >= 0.0:
        return max_score
    return None


def _profile_bottleneck_category_rank_key(
    category_name: str,
    entry: dict[str, object],
) -> tuple[float, float, float, str]:
    pressure_score = _profile_bottleneck_category_pressure_score(entry)
    max_score = _finite_summary_value(entry.get("max_score"))
    rank = _finite_summary_value(entry.get("top_rank"))
    return (
        -(pressure_score if pressure_score is not None else -1.0),
        -(max_score if max_score is not None else -1.0),
        rank if rank is not None else float("inf"),
        category_name,
    )


def _profile_bottleneck_top_category(
    category_summary: dict[str, dict[str, object]],
) -> Optional[dict[str, object]]:
    ranked_items = _ranked_profile_bottleneck_category_items(category_summary)
    if not ranked_items:
        return None
    category_name, entry = ranked_items[0]
    return {**entry, "category": category_name}


def _profile_bottleneck_severity_counts(
    candidates: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        severity = candidate.get("severity")
        if isinstance(severity, str) and severity:
            counts[severity] = counts.get(severity, 0) + 1

    return _ordered_profile_bottleneck_severity_counts(counts)


def _ordered_profile_bottleneck_severity_counts(
    counts: dict[object, object],
) -> dict[str, int]:
    ordered_counts: dict[str, int] = {}
    for severity, _threshold in PROFILE_BOTTLENECK_SEVERITY_LEVELS:
        count = _display_count_value(counts.get(severity))
        if count is not None and count > 0:
            ordered_counts[severity] = count
    for severity in sorted(
        key for key in counts if isinstance(key, str) and key not in ordered_counts
    ):
        count = _display_count_value(counts.get(severity))
        if count is not None and count > 0:
            ordered_counts[severity] = count
    return ordered_counts


def _add_profile_bottleneck_candidates(summary: dict) -> None:
    ranked_candidates = _ranked_profile_bottleneck_candidates(summary)
    if not ranked_candidates:
        return
    candidates = ranked_candidates[:PROFILE_BOTTLENECK_CANDIDATE_LIMIT]
    omitted_count = len(ranked_candidates) - len(candidates)
    summary["profile_bottleneck_candidate_count"] = len(ranked_candidates)
    summary["profile_bottleneck_candidate_returned_count"] = len(candidates)
    summary["profile_bottleneck_candidate_limit"] = PROFILE_BOTTLENECK_CANDIDATE_LIMIT
    if omitted_count > 0:
        summary["profile_bottleneck_candidate_omitted_count"] = omitted_count
    summary["profile_bottleneck_top_candidate"] = candidates[0]
    summary["profile_bottleneck_severity_thresholds"] = _profile_bottleneck_severity_thresholds()
    summary["profile_bottleneck_severity_counts"] = _profile_bottleneck_severity_counts(
        ranked_candidates
    )
    category_summary = _profile_bottleneck_category_summary(ranked_candidates)
    top_category = _profile_bottleneck_top_category(category_summary)
    if top_category is not None:
        summary["profile_bottleneck_top_category"] = top_category
    summary["profile_bottleneck_category_summary"] = category_summary
    summary["profile_bottleneck_candidates"] = candidates


def _summary_choice(raw: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = raw.strip()
    if value not in allowed:
        raise ValueError(f"{name} has unsupported value: {value}")
    return value


def _compact_run(row: dict) -> dict:
    compact, _omissions = _compact_run_with_omissions(row)
    return compact


def _compact_run_with_omissions(row: dict) -> tuple[dict, list[dict[str, str]]]:
    compact = {}
    omissions: list[dict[str, str]] = []

    def omit(field: str, reason: str) -> None:
        omissions.append({"field": field, "reason": reason})

    for field in BEST_RUN_FIELDS:
        if field not in row:
            continue
        value = row[field]
        if field in BEST_RUN_TEXT_FIELDS:
            if field == "dataset_mode":
                value = _summary_choice(value, DATASET_MODE_CHOICES, field)
            elif field == "profile_model_status":
                try:
                    value = _summary_choice(value, PROFILE_MODEL_STATUS_CHOICES, field)
                except ValueError:
                    omit(field, "invalid_choice")
                    continue
            elif field == "scheduler_last_error":
                failures = _display_count_value(row.get("scheduler_step_failures"))
                if failures is None or failures <= 0:
                    if isinstance(value, str) and value.strip():
                        omit(field, "inactive_context")
                    elif not isinstance(value, str) and value is not None:
                        omit(field, "invalid_text")
                    continue
                if not isinstance(value, str) or not value.strip():
                    omit(field, "empty_text" if isinstance(value, str) else "invalid_text")
                    continue
                value = value.strip()
            elif not isinstance(value, str):
                omit(field, "invalid_text")
                continue
            elif not value.strip():
                continue
            else:
                value = value.strip()
        elif field in BEST_RUN_BOOL_FIELDS:
            if not isinstance(value, bool):
                omit(field, "invalid_boolean")
                continue
        elif field == "seed":
            try:
                value = _int_setting(value, field)
            except ValueError:
                omit(field, "invalid_integer")
                continue
        elif field in BEST_RUN_INTEGER_FIELDS:
            try:
                if field in BEST_RUN_POSITIVE_INTEGER_FIELDS:
                    value = _positive_int_setting(value, field)
                else:
                    value = _non_negative_int_setting(value, field)
            except ValueError:
                omit(field, "invalid_integer")
                continue
            if field in BEST_RUN_POSITIVE_ONLY_INTEGER_FIELDS and value <= 0:
                continue
        else:
            numeric_value = _finite_summary_value(value)
            if numeric_value is None:
                omit(field, "non_numeric_or_non_finite")
                continue
            if numeric_value < 0.0:
                omit(field, "negative")
                continue
            max_value = _summary_metric_max_value(field)
            if max_value is not None and numeric_value > max_value:
                omit(field, "above_max")
                continue
        compact[field] = value
    return compact, omissions


def _finite_metric_value(row: dict, field: str, *, min_value: Optional[float] = None) -> Optional[float]:
    if field not in row:
        return None
    value = _finite_summary_value(row[field])
    if value is None:
        return None
    if min_value is not None and value < min_value:
        return None
    return value


def _positive_sample_count_value(raw: object) -> Optional[float]:
    value = _finite_summary_value(raw)
    if value is None or value <= 0.0:
        return None
    if not value.is_integer():
        return None
    return value


def _summary_metric_max_value(field: str) -> Optional[float]:
    return 100.0 if field in BOUNDED_PERCENT_SUMMARY_FIELDS else None


def _summary_metric_min_value(field: str) -> float:
    return 1.0 if field == "grad_accum" else 0.0


def _best_finite_row(
    rows: list[dict],
    field: str,
    *,
    prefer_high: bool,
    sample_count_field: Optional[str] = None,
    min_value: Optional[float] = None,
) -> Optional[dict]:
    candidates = []
    for row in rows:
        if sample_count_field is not None and sample_count_field in row:
            sample_count = _positive_sample_count_value(row[sample_count_field])
            if sample_count is None:
                continue
        value = _finite_metric_value(row, field, min_value=min_value)
        if value is None:
            continue
        candidates.append((value, row))
    if not candidates:
        return None
    selector = max if prefer_high else min
    return selector(candidates, key=lambda item: item[0])[1]


def _summary_field_has_signal(rows: list[dict], field: str) -> bool:
    min_value = _summary_metric_min_value(field)
    max_value = _summary_metric_max_value(field)
    integer = field in SUMMARY_INTEGER_FIELDS
    for row in rows:
        if field not in row:
            continue
        value = _finite_summary_value(row[field])
        if value is None:
            return True
        if value > 0.0:
            return True
        if value < min_value:
            return True
        if max_value is not None and value > max_value:
            return True
        if integer and not value.is_integer():
            return True
    return False


def summary_fields_for_rows(rows: list[dict]) -> tuple[str, ...]:
    present_fields = set()
    for row in rows:
        present_fields.update(row.keys())
    batch_fields = tuple(field for field in BATCH_SUMMARY_FIELDS if field in present_fields)
    setup_fields = tuple(field for field in SETUP_SUMMARY_FIELDS if field in present_fields)
    config_fields = tuple(field for field in CONFIG_SUMMARY_FIELDS if field in present_fields)
    workload_fields = tuple(field for field in WORKLOAD_SUMMARY_FIELDS if field in present_fields)
    diagnostic_fields = tuple(
        field
        for field in DIAGNOSTIC_SUMMARY_FIELDS
        if field in present_fields and _summary_field_has_signal(rows, field)
    )
    profile_fields = tuple(field for field in PROFILE_SUMMARY_FIELDS if field in present_fields)
    memory_fields = tuple(field for field in DEVICE_MEMORY_SUMMARY_FIELDS if field in present_fields)
    return (
        BASE_SUMMARY_FIELDS
        + setup_fields
        + config_fields
        + batch_fields
        + workload_fields
        + diagnostic_fields
        + profile_fields
        + memory_fields
    )


def count_profiled_rows(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if any(field in row for field in PROFILE_SUMMARY_FIELDS)
    )


def _profile_model_status_counts(rows: list[dict]) -> tuple[dict[str, int], int]:
    status_totals: dict[str, int] = {}
    invalid_count = 0
    for row in rows:
        if "profile_model_status" not in row:
            continue
        raw = row["profile_model_status"]
        if not isinstance(raw, str) or not raw.strip():
            invalid_count += 1
            continue
        status = raw.strip()
        if status not in PROFILE_MODEL_STATUS_CHOICES:
            invalid_count += 1
            continue
        if status == "not_requested":
            continue
        status_totals[status] = status_totals.get(status, 0) + 1
    ordered_counts = {
        status: status_totals[status]
        for status in PROFILE_MODEL_STATUS_ORDER
        if status in status_totals
    }
    return ordered_counts, invalid_count


def _positive_diagnostic_count(raw: object) -> Optional[int]:
    value = _finite_summary_value(raw)
    if value is None or value <= 0.0 or not value.is_integer():
        return None
    return int(value)


def _summary_metric_diagnostic(
    field: str,
    stats: dict[str, float],
    *,
    field_present: bool,
) -> Optional[dict[str, object]]:
    diagnostic: dict[str, object] = {"field": field}
    missing_count = _positive_diagnostic_count(stats.get("missing_count"))
    if missing_count is not None and (field_present or field in CRITICAL_SUMMARY_MISSING_FIELDS):
        diagnostic["missing_count"] = missing_count
    for key in ("non_finite_count", "invalid_count"):
        count = _positive_diagnostic_count(stats.get(key))
        if count is not None:
            diagnostic[key] = count
    return diagnostic if len(diagnostic) > 1 else None


def _add_summary_diagnostic_totals(summary: dict, diagnostics: list[dict[str, object]]) -> None:
    if not diagnostics:
        return
    missing_field_count = sum(1 for diagnostic in diagnostics if "missing_count" in diagnostic)
    non_finite_field_count = sum(1 for diagnostic in diagnostics if "non_finite_count" in diagnostic)
    invalid_field_count = sum(1 for diagnostic in diagnostics if "invalid_count" in diagnostic)
    summary["summary_diagnostic_field_count"] = len(diagnostics)
    if missing_field_count > 0:
        summary["summary_missing_field_count"] = missing_field_count
    if non_finite_field_count > 0:
        summary["summary_non_finite_field_count"] = non_finite_field_count
    if invalid_field_count > 0:
        summary["summary_invalid_field_count"] = invalid_field_count
    summary["summary_diagnostic_fields"] = diagnostics


def summarize_metric(
    rows: list[dict],
    field: str,
    *,
    missing_as_zero: bool = True,
    min_value: Optional[float] = 0.0,
    max_value: Optional[float] = None,
    integer: bool = False,
) -> dict[str, float]:
    values = []
    sample_count = 0
    missing_count = 0
    non_finite_count = 0
    invalid_count = 0
    for row in rows:
        if field in row:
            value = _finite_summary_value(row[field])
            if value is None:
                non_finite_count += 1
                continue
            if integer and not value.is_integer():
                invalid_count += 1
                continue
            if min_value is not None and value < min_value:
                invalid_count += 1
                continue
            if max_value is not None and value > max_value:
                invalid_count += 1
                continue
            values.append(value)
            sample_count += 1
        elif missing_as_zero:
            values.append(0.0)
            missing_count += 1
    result: dict[str, float] = {}
    if not values:
        result.update({"mean": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0})
        if missing_count > 0 or non_finite_count > 0 or invalid_count > 0:
            result["sample_count"] = float(sample_count)
        if missing_count > 0:
            result["missing_count"] = float(missing_count)
        if non_finite_count > 0:
            result["non_finite_count"] = float(non_finite_count)
        if invalid_count > 0:
            result["invalid_count"] = float(invalid_count)
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
    if invalid_count > 0:
        result["invalid_count"] = float(invalid_count)
    return result


def summarize_results(rows: list[dict]) -> dict:
    summary_rows = [_summary_row(row) for row in rows]
    present_fields = set()
    for row in summary_rows:
        present_fields.update(row.keys())
    summary: dict = {
        "runs": len(summary_rows),
        "best_reported": None,
        "best_end_to_end": None,
    }
    summary_diagnostics: list[dict[str, object]] = []
    profiled_runs = count_profiled_rows(summary_rows)
    if profiled_runs > 0:
        summary["profiled_runs"] = profiled_runs
    status_counts, status_invalid_count = _profile_model_status_counts(summary_rows)
    if status_counts:
        summary["profile_model_status_counts"] = status_counts
    if status_invalid_count > 0:
        summary["profile_model_status_invalid_count"] = status_invalid_count
        summary_diagnostics.append({
            "field": "profile_model_status",
            "invalid_count": status_invalid_count,
        })
    for field in summary_fields_for_rows(summary_rows):
        missing_as_zero = field in BASE_SUMMARY_FIELDS
        metric_stats = summarize_metric(
            summary_rows,
            field,
            missing_as_zero=missing_as_zero,
            min_value=_summary_metric_min_value(field),
            max_value=_summary_metric_max_value(field),
            integer=field in SUMMARY_INTEGER_FIELDS,
        )
        diagnostic = _summary_metric_diagnostic(
            field,
            metric_stats,
            field_present=field in present_fields,
        )
        if diagnostic is not None:
            summary_diagnostics.append(diagnostic)
        for stat_name, value in metric_stats.items():
            summary[f"{stat_name}_{field}"] = value
    _add_summary_diagnostic_totals(summary, summary_diagnostics)
    _add_profile_bottleneck_candidates(summary)

    if summary_rows:
        best_reported = _best_finite_row(
            summary_rows,
            "reported_samples_per_sec",
            prefer_high=True,
            min_value=0.0,
        )
        best_end_to_end = _best_finite_row(
            summary_rows,
            "end_to_end_wall_time_s",
            prefer_high=False,
            min_value=0.0,
        )
        if best_reported is not None:
            compact_reported, reported_omissions = _compact_run_with_omissions(best_reported)
            summary["best_reported"] = compact_reported
            if reported_omissions:
                summary["best_reported_omitted_fields"] = reported_omissions
        else:
            summary["best_reported"] = None
        if best_end_to_end is not None:
            compact_end_to_end, end_to_end_omissions = _compact_run_with_omissions(best_end_to_end)
            summary["best_end_to_end"] = compact_end_to_end
            if end_to_end_omissions:
                summary["best_end_to_end_omitted_fields"] = end_to_end_omissions
        else:
            summary["best_end_to_end"] = None
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
    if raw is None:
        return raw
    if isinstance(raw, str):
        if not [item.strip() for item in raw.split(",") if item.strip()]:
            raise ValueError("profile_model_include must contain at least one non-empty pattern")
        return raw
    if not isinstance(raw, Sequence):
        raise ValueError("profile_model_include must be a string, sequence of strings, or None")
    saw_entry = False
    has_pattern = False
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("profile_model_include entries must be strings")
        saw_entry = True
        if item.strip():
            has_pattern = True
    if not saw_entry or not has_pattern:
        raise ValueError("profile_model_include must contain at least one non-empty pattern")
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


def _format_profile_bottleneck_severity_counts(summary: dict[str, Any]) -> str:
    counts = summary.get("profile_bottleneck_severity_counts")
    if not isinstance(counts, dict):
        return ""

    parts = []
    seen: set[str] = set()
    for severity, _threshold in PROFILE_BOTTLENECK_SEVERITY_LEVELS:
        count = _display_count_value(counts.get(severity))
        if count is not None and count > 0:
            parts.append(f"{severity}={count}")
            seen.add(severity)

    for severity in sorted(
        key for key in counts if isinstance(key, str) and key not in seen
    ):
        count = _display_count_value(counts.get(severity))
        if count is not None and count > 0:
            parts.append(f"{severity}={count}")

    if not parts:
        return ""
    return f"severity_counts({','.join(parts)})"


def _format_profile_bottleneck_category_pressure(
    category_name: str,
    entry: dict[str, object],
    *,
    include_count: bool,
) -> str:
    top_name = entry.get("top_candidate")
    max_score = _non_negative_display_value(entry.get("max_score"))
    if not isinstance(top_name, str) or not top_name or max_score is None:
        return ""

    entry_suffix = "%" if entry.get("score_unit") == "profile_pct" else ""
    top_severity = entry.get("top_severity")
    severity_suffix = (
        f"[{top_severity}]"
        if isinstance(top_severity, str) and top_severity
        else ""
    )
    text = f"{category_name}:{top_name}={max_score:.1f}{entry_suffix}{severity_suffix}"

    if include_count:
        count = _display_count_value(entry.get("count"))
        if count is None:
            return ""
        text = f"{text}/{count}"

    total_score = _non_negative_display_value(entry.get("total_score"))
    if total_score is not None:
        text = f"{text};sum={total_score:.1f}{entry_suffix}"
    return text


def _format_profile_bottleneck_summary(summary: dict[str, Any]) -> str:
    top_candidate = _dict_value(summary.get("profile_bottleneck_top_candidate"))
    name = top_candidate.get("name")
    score = _non_negative_display_value(top_candidate.get("score"))
    if not isinstance(name, str) or not name or score is None:
        return ""
    score_suffix = "%" if top_candidate.get("score_unit") == "profile_pct" else ""
    parts = [f"#{_format_count(top_candidate.get('rank'))} {name}={score:.1f}{score_suffix}"]

    category = top_candidate.get("category")
    if isinstance(category, str) and category:
        parts.append(f"category={category}")
    severity = top_candidate.get("severity")
    if isinstance(severity, str) and severity:
        parts.append(f"severity={severity}")
    next_step = top_candidate.get("next_step")
    if isinstance(next_step, str) and next_step:
        parts.append(f"next={next_step}")
    severity_counts = _format_profile_bottleneck_severity_counts(summary)
    if severity_counts:
        parts.append(severity_counts)

    category_summary = summary.get("profile_bottleneck_category_summary")
    category_parts = []
    if isinstance(category_summary, dict):
        for category_name, entry in _ranked_profile_bottleneck_category_items(category_summary):
            category_part = _format_profile_bottleneck_category_pressure(
                category_name,
                entry,
                include_count=True,
            )
            if category_part:
                category_parts.append(category_part)
    if category_parts:
        parts.append(f"categories={','.join(category_parts)}")
    return " ".join(parts)


def _has_positive_display_value(raw: object) -> bool:
    value = _finite_display_value(raw)
    return value is not None and value > 0.0


def _non_negative_display_value(raw: object) -> Optional[float]:
    value = _finite_display_value(raw)
    if value is None or value < 0.0:
        return None
    return value


def _format_non_negative_metric_value(
    raw: object,
    *,
    precision: int,
    scale: float = 1.0,
    suffix: str = "",
) -> Optional[str]:
    value = _non_negative_display_value(raw)
    if value is None:
        return None
    return f"{value * scale:.{precision}f}{suffix}"


def _profile_row_name(row: dict[str, Any]) -> str:
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return "<unnamed>"


def _dict_value(raw: object) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _list_value(raw: object) -> list[object]:
    return raw if isinstance(raw, list) else []


def _display_count_value(raw: object) -> Optional[int]:
    if isinstance(raw, str):
        return None
    try:
        return non_negative_int_arg(raw)
    except argparse.ArgumentTypeError:
        return None


def _format_profile_open_timer_summary(profile: dict[str, Any]) -> str:
    parts = []
    phase_count = _display_count_value(profile.get("profile_open_phase_count"))
    if phase_count is not None and phase_count > 0:
        phase_names = [
            name.strip()
            for name in _list_value(profile.get("profile_open_phases"))
            if isinstance(name, str) and name.strip()
        ]
        suffix = f":{','.join(phase_names[:4])}" if phase_names else ""
        parts.append(f"phases={phase_count}{suffix}")

    detail_count = _display_count_value(profile.get("profile_open_detail_count"))
    if detail_count is not None and detail_count > 0:
        detail_names = []
        for row in _list_value(profile.get("profile_open_details")):
            if not isinstance(row, dict):
                continue
            parent = row.get("parent")
            name = row.get("name")
            count = _display_count_value(row.get("count"))
            if not isinstance(parent, str) or not parent.strip():
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            count_suffix = f"x{count}" if count is not None and count > 1 else ""
            detail_names.append(f"{parent.strip()}.{name.strip()}{count_suffix}")
        suffix = f":{','.join(detail_names[:4])}" if detail_names else ""
        parts.append(f"details={detail_count}{suffix}")
    return " ".join(parts)


def _format_setup_breakdown(metrics: dict[str, Any]) -> str:
    parts = []
    for field, label in (
        ("dataset_setup_time_s", "dataset"),
        ("loader_setup_time_s", "loader"),
        ("model_setup_time_s", "model"),
        ("compile_init_time_s", "compile"),
    ):
        value = _non_negative_display_value(metrics.get(field))
        if value is not None and value > 0.0:
            parts.append(f"{label}={value:.2f}s")
    return f"init({','.join(parts)})" if parts else ""


def _format_profile_model_hook_summary(metrics: dict[str, Any]) -> str:
    requested = metrics.get("profile_model_requested")
    status_present = "profile_model_status" in metrics
    status_raw = metrics.get("profile_model_status")
    status = status_raw.strip() if isinstance(status_raw, str) else ""
    if requested is not True and status != "hook_failures":
        return ""
    status_valid = bool(status) and status in PROFILE_MODEL_STATUS_CHOICES
    status_invalid = status_present and not status_valid

    parts = []
    if status_invalid:
        parts.append("status=invalid")
    elif status and status != "not_requested":
        parts.append(f"status={status}")
    failure_count: Optional[int] = None
    for field, label in (
        ("profile_model_modules_selected", "modules"),
        ("profile_model_hook_count", "hooks"),
        ("profile_model_hook_failures", "failures"),
    ):
        count = _display_count_value(metrics.get(field))
        if count is not None:
            if label == "failures":
                failure_count = count
            parts.append(f"{label}={count}")
    last_error = metrics.get("profile_model_hook_last_error")
    if isinstance(last_error, str) and last_error.strip() and failure_count != 0:
        parts.append(f"error={last_error.strip()}")
    return " ".join(parts)


def _format_scheduler_summary(metrics: dict[str, Any]) -> str:
    failures = _display_count_value(metrics.get("scheduler_step_failures"))
    if failures is None or failures <= 0:
        return ""
    parts = [f"failures={failures}"]
    last_error = metrics.get("scheduler_last_error")
    if isinstance(last_error, str) and last_error.strip():
        parts.append(f"error={last_error.strip()}")
    return " ".join(parts)


def _profile_child_rows(profile: dict[str, Any], section: str, group: str) -> list[object]:
    sections = _dict_value(profile.get(section))
    group_profile = _dict_value(sections.get(group))
    return _list_value(group_profile.get("top_children"))


def _format_profile_event_timing(
    row: dict[str, Any],
    *,
    precision: int = 1,
    include_p95: bool = False,
) -> str:
    avg_text = _format_metric_value(row.get("avg_ms"), precision=precision, suffix="ms")
    if avg_text == "n/a":
        return avg_text
    pct_text = _format_non_negative_metric_value(
        row.get("avg_pct_of_parent"),
        precision=1,
        suffix="%",
    )
    if pct_text is None:
        parts = [avg_text]
    else:
        parts = [f"{avg_text}@{pct_text}"]
    if include_p95:
        p95_text = _format_non_negative_metric_value(
            row.get("p95_ms"),
            precision=precision,
            suffix="ms",
        )
        if p95_text is not None:
            parts.append(f"p95={p95_text}")
        for field, label in (
            ("p99_ms", "p99"),
            ("std_ms", "std"),
        ):
            value_text = _format_non_negative_metric_value(
                row.get(field),
                precision=precision,
                suffix="ms",
            )
            if value_text is not None:
                parts.append(f"{label}={value_text}")
    parts.extend(_profile_count_fields(row))
    return " ".join(parts)


def _format_profile_event_group_summary(profile: dict[str, Any], group: str) -> str:
    events = _dict_value(profile.get("phase_events"))
    event_group = _dict_value(events.get(group))
    if not event_group:
        return ""
    parts = []
    span_text = _format_non_negative_metric_value(
        event_group.get("span_avg_ms"),
        precision=2,
        suffix="ms",
    )
    if span_text is not None:
        span_pct_text = _format_non_negative_metric_value(
            event_group.get("span_pct_of_parent"),
            precision=1,
            suffix="%",
        )
        suffix = f"@{span_pct_text}" if span_pct_text is not None else ""
        parts.append(f"span={span_text}{suffix}")
    earliest_text = _format_non_negative_metric_value(
        event_group.get("earliest_avg_ms"),
        precision=2,
        suffix="ms",
    )
    latest_text = _format_non_negative_metric_value(
        event_group.get("latest_avg_ms"),
        precision=2,
        suffix="ms",
    )
    if earliest_text is not None and latest_text is not None:
        parts.append(f"range={earliest_text}-{latest_text}")
    earliest_pct_text = _format_non_negative_metric_value(
        event_group.get("earliest_pct_of_parent"),
        precision=1,
        suffix="%",
    )
    latest_pct_text = _format_non_negative_metric_value(
        event_group.get("latest_pct_of_parent"),
        precision=1,
        suffix="%",
    )
    if earliest_pct_text is not None and latest_pct_text is not None:
        parts.append(f"range_pct={earliest_pct_text}-{latest_pct_text}")
    return " ".join(parts)


def _profile_count_fields(row: dict[str, Any]) -> list[str]:
    parts = []
    for field, label in (
        ("calls", "calls"),
        ("sample_count", "samples"),
        ("window_sample_count", "window"),
    ):
        if field not in row:
            continue
        count = _display_count_value(row.get(field))
        if count is not None:
            parts.append(f"{label}={count}")
    return parts


def _format_profile_breakdown_child_timing(row: dict[str, Any]) -> str:
    parts = [_format_metric_value(row.get("pct_of_parent"), precision=1, suffix="%")]
    avg_text = _format_non_negative_metric_value(row.get("avg_ms"), precision=2, suffix="ms")
    if avg_text is not None:
        parts.append(f"avg={avg_text}")
    for field, label in (
        ("p95_ms", "p95"),
        ("p99_ms", "p99"),
        ("std_ms", "std"),
    ):
        value_text = _format_non_negative_metric_value(row.get(field), precision=2, suffix="ms")
        if value_text is not None:
            parts.append(f"{label}={value_text}")
    parts.extend(_profile_count_fields(row))
    return " ".join(parts)


def _format_profile_phase_timing(row: dict[str, Any]) -> str:
    parts = [_format_metric_value(row.get("pct"), precision=1, suffix="%")]
    avg_text = _format_non_negative_metric_value(row.get("avg_ms"), precision=2, suffix="ms")
    if avg_text is not None:
        parts.append(f"avg={avg_text}")
    for field, label in (
        ("p95_ms", "p95"),
        ("p99_ms", "p99"),
        ("std_ms", "std"),
    ):
        value_text = _format_non_negative_metric_value(row.get(field), precision=2, suffix="ms")
        if value_text is not None:
            parts.append(f"{label}={value_text}")
    parts.extend(_profile_count_fields(row))
    return " ".join(parts)


def _profile_breakdown(profile: dict[str, Any], group: str) -> dict[str, Any]:
    breakdowns = _dict_value(profile.get("phase_breakdowns"))
    return _dict_value(breakdowns.get(group))


def _format_profile_breakdown_summary(profile: dict[str, Any], group: str) -> str:
    breakdown = _profile_breakdown(profile, group)
    if not breakdown:
        return ""
    parts = []
    coverage_text = _format_non_negative_metric_value(
        breakdown.get("coverage_pct"),
        precision=1,
        suffix="%",
    )
    if coverage_text is not None:
        parts.append(f"coverage={coverage_text}")
    tracked_text = _format_non_negative_metric_value(
        breakdown.get("tracked_s"),
        precision=2,
        scale=1e3,
        suffix="ms",
    )
    if tracked_text is not None:
        parts.append(f"tracked={tracked_text}")
    untracked_text = _format_non_negative_metric_value(
        breakdown.get("untracked_s"),
        precision=2,
        scale=1e3,
        suffix="ms",
    )
    if untracked_text is not None:
        parts.append(f"untracked={untracked_text}")
    if _has_positive_display_value(breakdown.get("overtracked_s")):
        overtracked_text = _format_non_negative_metric_value(
            breakdown.get("overtracked_s"),
            precision=2,
            scale=1e3,
            suffix="ms",
        )
        if overtracked_text is not None:
            parts.append(f"overtracked={overtracked_text}")
    return " ".join(parts)


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
    for field in (
        "compile",
        "collect_profile",
        "meter_fast_mode",
        "profile_sync",
        "profile_distribution",
        "profile_model",
    ):
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
    dataset_setup_time_s = _non_negative_finite_float_setting(
        time.perf_counter() - dataset_start,
        "dataset_setup_time_s",
    )

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
    loader_setup_time_s = _non_negative_finite_float_setting(
        time.perf_counter() - loader_start,
        "loader_setup_time_s",
    )

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
        meter_fast_mode=getattr(args, "meter_fast_mode", False),
    )
    model_setup_time_s = _non_negative_finite_float_setting(
        time.perf_counter() - model_start,
        "model_setup_time_s",
    )
    setup_time_s = _non_negative_finite_float_setting(
        time.perf_counter() - setup_start,
        "setup_time_s",
    )

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
    wall = _non_negative_finite_float_setting(time.perf_counter() - start, "wall_time_s")
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
    parser.add_argument("--meter-fast-mode", action="store_true", help="Use lighter throughput meters without tail/window stats.")
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
        setup_breakdown = _format_setup_breakdown(metrics)
        setup_breakdown_text = f"{setup_breakdown} " if setup_breakdown else ""
        print(
            f"Run {run_index}: "
            f"wall={_format_metric_value(metrics.get('wall_time_s'), precision=2, suffix='s')} "
            f"setup={_format_metric_value(metrics.get('setup_time_s'), precision=2, suffix='s')} "
            f"{setup_breakdown_text}"
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
        profile_model_summary = _format_profile_model_hook_summary(metrics)
        if profile_model_summary:
            print(f"  profile_model: {profile_model_summary}")
        scheduler_summary = _format_scheduler_summary(metrics)
        if scheduler_summary:
            print(f"  scheduler: {scheduler_summary}")
        profile = _dict_value(metrics.get("profile"))
        if profile:
            open_timer_summary = _format_profile_open_timer_summary(profile)
            if open_timer_summary:
                print(f"  open_timers: {open_timer_summary}")
            top_phases = ", ".join(
                f"{_profile_row_name(row)}={_format_profile_phase_timing(row)}"
                for row in _list_value(profile.get("top_phases"))[:4]
                if isinstance(row, dict)
            )
            print(f"  phases: {top_phases}")
            forward = _profile_child_rows(profile, "phase_breakdowns", "forward")
            if forward:
                forward_summary = _format_profile_breakdown_summary(profile, "forward")
                top_forward = ", ".join(
                    f"{_profile_row_name(row)}="
                    f"{_format_profile_breakdown_child_timing(row)}"
                    for row in forward[:4]
                    if isinstance(row, dict)
                )
                suffix = f" ({forward_summary})" if forward_summary else ""
                print(f"  forward: {top_forward}{suffix}")
            backward = _profile_child_rows(profile, "phase_events", "backward_grad_ready")
            if backward:
                backward_summary = _format_profile_event_group_summary(profile, "backward_grad_ready")
                if backward_summary:
                    print(f"  backward_grad_ready_summary: {backward_summary}")
                top_backward = ", ".join(
                    f"{_profile_row_name(row)}="
                    f"{_format_profile_event_timing(row, include_p95=True)}"
                    for row in backward[:4]
                    if isinstance(row, dict)
                )
                print(f"  backward_grad_ready: {top_backward}")
            optimizer = _profile_child_rows(profile, "phase_breakdowns", "optimizer")
            if optimizer:
                optimizer_summary = _format_profile_breakdown_summary(profile, "optimizer")
                top_optimizer = ", ".join(
                    f"{_profile_row_name(row)}="
                    f"{_format_profile_breakdown_child_timing(row)}"
                    for row in optimizer[:4]
                    if isinstance(row, dict)
                )
                suffix = f" ({optimizer_summary})" if optimizer_summary else ""
                print(f"  optimizer: {top_optimizer}{suffix}")

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

    bottleneck_summary = _format_profile_bottleneck_summary(aggregate)
    if bottleneck_summary:
        print(f"Bottleneck: {bottleneck_summary}")
    print("Aggregate:", dumps_json(aggregate))


if __name__ == "__main__":
    main()
