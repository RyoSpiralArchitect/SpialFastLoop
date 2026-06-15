from __future__ import annotations

import argparse
import io
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bench_parallel_transactions as bpt
from scripts.bench_parallel_transactions import (
    BenchmarkResult,
    SyntheticTransactionDataset,
    _best_finite_row,
    _format_count,
    _format_metric_value,
    _format_profile_breakdown_child_timing,
    _format_profile_breakdown_summary,
    _format_profile_event_group_summary,
    _format_profile_event_timing,
    _format_profile_model_hook_summary,
    _format_profile_open_timer_summary,
    _format_profile_phase_timing,
    _format_scheduler_summary,
    _format_setup_breakdown,
    _has_positive_display_value,
    _profile_count_fields,
    _profile_row_name,
    build_model,
    device_arg,
    non_negative_int_arg,
    parse_args,
    positive_float_arg,
    positive_int_arg,
    run_once,
    summarize_metric,
    summarize_results,
    validate_benchmark_args,
)
from scripts.json_utils import dump_json, dumps_json


class _FailingIndex:
    def __index__(self) -> int:
        raise RuntimeError("index conversion failed")


class _FailingFloat:
    def __float__(self) -> float:
        raise RuntimeError("float conversion failed")


def test_materialized_transaction_dataset_matches_shape_and_is_stable() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=True)

    features_a, target_a = dataset[2]
    features_b, target_b = dataset[2]

    assert features_a.shape == (4,)
    assert target_a.shape == ()
    assert torch.equal(features_a, features_b)
    assert torch.equal(target_a, target_b)
    assert dataset.materialized_bytes == (8 * 4 * 4) + (8 * 8)


def test_generated_transaction_dataset_is_index_deterministic() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=False)

    features_a, target_a = dataset[2]
    features_b, target_b = dataset[2]

    assert torch.equal(features_a, features_b)
    assert torch.equal(target_a, target_b)
    assert dataset.materialized_bytes == 0


def test_benchmark_result_as_dict_preserves_authoritative_run_and_wall_time() -> None:
    result = BenchmarkResult(
        wall_time_s=1.25,
        trainer_metrics={
            "run": 99,
            "wall_time_s": 99.0,
            "samples_per_sec": 10.0,
        },
        run_index=3,
    )

    payload = result.as_dict()

    assert payload["run"] == 3
    assert payload["wall_time_s"] == pytest.approx(1.25)
    assert payload["samples_per_sec"] == pytest.approx(10.0)


def test_transaction_dataset_ignores_global_default_device() -> None:
    if not hasattr(torch, "set_default_device"):
        pytest.skip("torch.set_default_device is not available")
    previous_device = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    torch.set_default_device("meta")
    try:
        materialized = SyntheticTransactionDataset(4, 2, 2, seed=123, materialized=True)
        generated = SyntheticTransactionDataset(4, 2, 2, seed=123, materialized=False)

        materialized_features, materialized_target = materialized[0]
        generated_features, generated_target = generated[0]

        assert materialized_features.device.type == "cpu"
        assert materialized_target.device.type == "cpu"
        assert generated_features.device.type == "cpu"
        assert generated_target.device.type == "cpu"
    finally:
        torch.set_default_device(previous_device)


def test_transaction_dataset_rejects_invalid_shapes() -> None:
    for kwargs in (
        {"size": 0, "features": 4, "classes": 3},
        {"size": -1, "features": 4, "classes": 3},
        {"size": True, "features": 4, "classes": 3},
        {"size": 8, "features": 0, "classes": 3},
        {"size": 8, "features": True, "classes": 3},
        {"size": 8, "features": 4, "classes": 0},
        {"size": 8, "features": 4, "classes": True},
    ):
        with pytest.raises(ValueError):
            SyntheticTransactionDataset(**kwargs)


def test_transaction_dataset_rejects_invalid_seed_and_materialized() -> None:
    for kwargs in (
        {"seed": True},
        {"seed": 1.5},
        {"seed": "1"},
        {"materialized": 1},
        {"materialized": "true"},
    ):
        with pytest.raises(ValueError):
            SyntheticTransactionDataset(8, 4, 3, **kwargs)  # type: ignore[arg-type]


def test_transaction_dataset_bounds_indices_consistently() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=True)

    last_features, last_target = dataset[-1]
    direct_features, direct_target = dataset[7]

    assert torch.equal(last_features, direct_features)
    assert torch.equal(last_target, direct_target)

    for index in (-9, 8):
        with pytest.raises(IndexError):
            dataset[index]


@pytest.mark.parametrize(
    ("features", "classes", "match"),
    [
        (0, 2, "features"),
        (-1, 2, "features"),
        (1.5, 2, "features"),
        (True, 2, "features"),
        (4, 0, "classes"),
        (4, 1.5, "classes"),
        (4, True, "classes"),
    ],
)
def test_build_model_rejects_invalid_shape_values(
    features: object,
    classes: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_model(features, classes)  # type: ignore[arg-type]


def test_summarize_metric_reports_distribution() -> None:
    rows = [
        {"samples_per_sec": 100.0},
        {"samples_per_sec": 300.0},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(200.0)
    assert stats["min"] == pytest.approx(100.0)
    assert stats["max"] == pytest.approx(300.0)
    assert stats["stddev"] == pytest.approx(100.0)


def test_summarize_metric_reports_missing_zero_imputation() -> None:
    rows = [
        {"samples_per_sec": 100.0},
        {},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(50.0)
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(100.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["missing_count"] == pytest.approx(1.0)


def test_summarize_metric_skips_non_finite_values() -> None:
    rows = [
        {"samples_per_sec": 100.0},
        {"samples_per_sec": float("nan")},
        {"samples_per_sec": float("inf")},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(100.0)
    assert stats["min"] == pytest.approx(100.0)
    assert stats["max"] == pytest.approx(100.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["non_finite_count"] == pytest.approx(2.0)


def test_summarize_metric_skips_negative_values() -> None:
    rows = [
        {"samples_per_sec": 100.0},
        {"samples_per_sec": -20.0},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(100.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["invalid_count"] == pytest.approx(1.0)
    assert "non_finite_count" not in stats


def test_summarize_metric_skips_values_above_maximum() -> None:
    rows = [
        {"profile_forward_pct": 40.0},
        {"profile_forward_pct": 140.0},
    ]

    stats = summarize_metric(rows, "profile_forward_pct", max_value=100.0)

    assert stats["mean"] == pytest.approx(40.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["invalid_count"] == pytest.approx(1.0)


def test_summarize_metric_skips_fractional_integer_values() -> None:
    rows = [
        {"steps": 4},
        {"steps": 2.0},
        {"steps": 1.5},
    ]

    stats = summarize_metric(rows, "steps", integer=True)

    assert stats["mean"] == pytest.approx(3.0)
    assert stats["sample_count"] == pytest.approx(2.0)
    assert stats["invalid_count"] == pytest.approx(1.0)


def test_summarize_metric_skips_bool_values() -> None:
    rows = [
        {"samples_per_sec": 100.0},
        {"samples_per_sec": True},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(100.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["non_finite_count"] == pytest.approx(1.0)


def test_summarize_metric_skips_numeric_strings() -> None:
    rows = [
        {"samples_per_sec": 100.0},
        {"samples_per_sec": "300.0"},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(100.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["non_finite_count"] == pytest.approx(1.0)


def test_summarize_metric_skips_failed_float_like_values() -> None:
    class FailingFloat:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

    rows = [
        {"samples_per_sec": 100.0},
        {"samples_per_sec": FailingFloat()},
    ]

    stats = summarize_metric(rows, "samples_per_sec")

    assert stats["mean"] == pytest.approx(100.0)
    assert stats["sample_count"] == pytest.approx(1.0)
    assert stats["non_finite_count"] == pytest.approx(1.0)


def test_dump_json_converts_non_finite_values_to_null() -> None:
    buffer = io.StringIO()

    dump_json(
        [
            {
                "loss": float("nan"),
                "profile": {"pct": float("inf")},
                "history": (1.0, float("-inf")),
            },
        ],
        buffer,
    )

    payload = json.loads(buffer.getvalue())

    assert payload == [
        {
            "loss": None,
            "profile": {"pct": None},
            "history": [1.0, None],
        },
    ]


def test_dumps_json_normalizes_artifact_payload_values() -> None:
    payload = {
        "scalar_tensor": torch.tensor(float("nan")),
        "vector_tensor": torch.tensor([1.0, float("inf")]),
        "path": Path("artifacts") / "summary.json",
        "tags": {"beta", "alpha"},
        ("tuple", "key"): "value",
    }

    normalized = json.loads(dumps_json(payload))

    assert normalized["scalar_tensor"] is None
    assert normalized["vector_tensor"] == [1.0, None]
    assert normalized["path"] == "artifacts/summary.json"
    assert normalized["tags"] == ["alpha", "beta"]
    assert normalized["['tuple', 'key']"] == "value"


def test_dumps_json_falls_back_for_overflowing_float_like_values() -> None:
    class OverflowingFloat:
        def __float__(self) -> float:
            raise OverflowError("too large")

        def __str__(self) -> str:
            return "overflowing"

    payload = {
        "value": OverflowingFloat(),
        OverflowingFloat(): "key",
    }

    normalized = json.loads(dumps_json(payload))

    assert normalized["value"] == "overflowing"
    assert normalized["overflowing"] == "key"


def test_dumps_json_falls_back_for_failed_float_like_values() -> None:
    class FailingFloat:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

        def __str__(self) -> str:
            return "failed-float"

    payload = {
        "value": FailingFloat(),
        FailingFloat(): "key",
    }

    normalized = json.loads(dumps_json(payload))

    assert normalized["value"] == "failed-float"
    assert normalized["failed-float"] == "key"


def test_dumps_json_sorts_sets_with_malformed_repr_values() -> None:
    class MalformedRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

        def __str__(self) -> str:
            return "malformed-repr"

    normalized = json.loads(dumps_json({"items": {MalformedRepr()}}))

    assert normalized["items"] == ["malformed-repr"]


def test_dumps_json_falls_back_for_unrepresentable_values() -> None:
    class Unrepresentable:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

        def __str__(self) -> str:
            raise RuntimeError("str failed")

    payload = {
        "value": Unrepresentable(),
        Unrepresentable(): "key",
    }

    normalized = json.loads(dumps_json(payload))
    fallback_keys = [key for key in normalized if key != "value"]

    assert "Unrepresentable" in normalized["value"]
    assert fallback_keys
    assert "Unrepresentable" in fallback_keys[0]
    assert normalized[fallback_keys[0]] == "key"


def test_dumps_json_falls_back_for_unreadable_tensor_values() -> None:
    normalized = json.loads(dumps_json({"tensor": torch.empty(2, device="meta")}))

    assert "tensor" in normalized["tensor"]
    assert "meta" in normalized["tensor"]


def test_format_setup_breakdown_includes_positive_setup_parts() -> None:
    assert _format_setup_breakdown({
        "dataset_setup_time_s": 0.05,
        "loader_setup_time_s": 0.07,
        "model_setup_time_s": 0.13,
        "compile_init_time_s": 0.02,
    }) == "init(dataset=0.05s,loader=0.07s,model=0.13s,compile=0.02s)"


def test_format_setup_breakdown_omits_zero_and_invalid_parts() -> None:
    assert _format_setup_breakdown({
        "dataset_setup_time_s": 0.0,
        "loader_setup_time_s": -0.01,
        "model_setup_time_s": "slow",
        "compile_init_time_s": False,
    }) == ""


def test_dumps_json_falls_back_for_failed_pathlike_values() -> None:
    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError("path failed")

        def __str__(self) -> str:
            return "failed-path"

    payload = {
        "path": FailingPath(),
        FailingPath(): "key",
    }

    normalized = json.loads(dumps_json(payload))

    assert normalized["path"] == "failed-path"
    assert normalized["failed-path"] == "key"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({1: "int", "1": "str"}, "unique after normalization at \\$"),
        ({"": "blank"}, "non-empty after normalization at \\$"),
        ({"nested": {1: "int", "1": "str"}}, "unique after normalization at \\$.nested"),
        ({"items": [{1: "int", "1": "str"}]}, "unique after normalization at \\$.items"),
    ],
)
def test_dumps_json_rejects_ambiguous_artifact_keys(payload: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dumps_json(payload)


def test_dump_json_rejects_ambiguous_artifact_keys_before_writing() -> None:
    buffer = io.StringIO()

    with pytest.raises(ValueError, match="unique after normalization"):
        dump_json({1: "int", "1": "str"}, buffer)

    assert buffer.getvalue() == ""


def test_summarize_results_reports_best_runs_and_fallbacks() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "samples_per_sec": 90.0,
            "steady_samples_per_sec": 110.0,
            "wall_time_s": 2.0,
            "setup_time_s": 1.0,
            "profile_flat_metric_invalid_count": 2,
            "profile_open_phase_count": 1,
            "profile_open_detail_count": 2,
            "profile_model_requested": True,
            "profile_model_enabled": True,
            "profile_model_status": "hook_failures",
            "profile_model_modules_selected": 1,
            "profile_model_hook_count": 3,
            "profile_model_hook_failures": 1,
            "profile_model_hook_last_error": "RuntimeError: hook boom",
            "profile_forward_backward_pct": 40.0,
            "profile_forward_backward_time_s": 0.20,
            "profile_forward_pct": 15.0,
            "profile_forward_p50_ms": 1.0,
            "profile_forward_p95_ms": 2.0,
            "profile_forward_p99_ms": 3.0,
            "profile_forward_std_ms": 0.2,
            "profile_forward_min_ms": 0.8,
            "profile_forward_max_ms": 3.5,
            "profile_forward_sample_count": 2,
            "profile_forward_window_sample_count": 2,
            "profile_forward_child_count": 1,
            "profile_forward_tracked_time_s": 0.05,
            "profile_forward_untracked_time_s": 0.01,
            "profile_forward_overtracked_time_s": 0.0,
            "profile_forward_coverage_pct": 80.0,
            "profile_forward_untracked_pct": 20.0,
            "profile_forward_overtracked_pct_of_parent": 0.0,
            "profile_forward_top_time_s": 0.04,
            "profile_forward_top_pct_of_parent": 40.0,
            "profile_forward_top_avg_ms": 4.0,
            "profile_forward_top_p95_ms": 4.5,
            "profile_forward_top_calls": 2,
            "profile_loss_pct": 5.0,
            "profile_loss_reduce_pct": 2.0,
            "profile_backward_pct": 25.0,
            "profile_backward_grad_ready_child_count": 1,
            "profile_backward_grad_ready_parent_avg_ms": 12.0,
            "profile_backward_grad_ready_top_avg_ms": 4.0,
            "profile_backward_grad_ready_top_pct": 30.0,
            "profile_backward_grad_ready_top_p95_ms": 4.5,
            "profile_backward_grad_ready_top_calls": 2,
            "profile_optimizer_pct": 10.0,
            "profile_optimizer_p95_ms": 4.0,
            "profile_optimizer_child_count": 1,
            "profile_optimizer_tracked_time_s": 0.03,
            "profile_optimizer_untracked_time_s": 0.01,
            "profile_optimizer_overtracked_time_s": 0.0,
            "profile_optimizer_coverage_pct": 75.0,
            "profile_optimizer_untracked_pct": 25.0,
            "profile_optimizer_overtracked_pct_of_parent": 0.0,
            "profile_optimizer_top_time_s": 0.02,
            "profile_optimizer_top_pct_of_parent": 20.0,
            "profile_optimizer_top_avg_ms": 2.0,
            "profile_optimizer_top_p95_ms": 2.5,
            "profile_optimizer_top_calls": 2,
            "profile_user_metrics_pct": 4.0,
            "profile_postprocess_pct": 6.0,
            "profile_collect_output_pct": 3.0,
            "profile_metrics_pct": 1.0,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "device": "cpu",
            "amp": False,
            "compile_requested": True,
            "compiled": False,
            "compile_init_time_s": 0.02,
            "compile_fallback_reason": "cpu_device",
            "transactions": 64,
            "batch_size": 4,
            "num_workers": 0,
            "world_size": 1,
            "rank": 0,
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 160.0,
            "steady_samples_per_sec": 200.0,
            "p99_s": 0.010,
            "std_batch_s": 0.001,
            "avg_batch_s": 0.004,
            "last_batch_s": 0.003,
            "min_batch_s": 0.002,
            "max_batch_s": 0.007,
            "batches": 3,
            "best_samples_per_sec": 250.0,
            "headroom_ratio": 1.25,
            "ema_samples_per_sec": 220.0,
            "window_samples_per_sec": 210.0,
            "window_time_s": 0.040,
            "window_batches": 2,
            "window_samples": 8,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "dataset_setup_time_s": 0.05,
            "loader_setup_time_s": 0.07,
            "model_setup_time_s": 0.13,
            "dataset_materialized_bytes": 4096,
            "end_to_end_wall_time_s": 1.25,
            "steps": 3,
            "samples": 12,
            "optimizer_steps": 2,
            "grad_accum": 2,
            "partial_optimizer_steps": 1,
            "grad_accum_tail_steps": 1,
            "warmup_steps": 1,
            "warmup_samples": 4,
            "warmup_optimizer_steps": 0,
            "warmup_samples_per_sec": 80.0,
            "warmup_total_time_s": 0.05,
            "warmup_p99_s": 0.05,
            "cold_start_steps": 1,
            "cold_start_samples_per_sec": 80.0,
            "steady_steps": 2,
            "steady_samples": 8,
            "steady_optimizer_steps": 2,
            "steady_total_time_s": 0.08,
            "steady_p99_s": 0.04,
            "profile_flat_metric_invalid_count": 0,
            "profile_open_phase_count": 0,
            "profile_open_detail_count": 0,
            "profile_model_requested": True,
            "profile_model_enabled": True,
            "profile_model_status": "ok",
            "profile_model_modules_selected": 2,
            "profile_model_hook_count": 4,
            "profile_model_hook_failures": 0,
            "profile_model_hook_last_error": "",
            "profile_forward_backward_pct": 60.0,
            "profile_forward_backward_time_s": 0.30,
            "profile_forward_pct": 20.0,
            "profile_forward_p50_ms": 2.0,
            "profile_forward_p95_ms": 4.0,
            "profile_forward_p99_ms": 6.0,
            "profile_forward_std_ms": 0.4,
            "profile_forward_min_ms": 1.5,
            "profile_forward_max_ms": 7.0,
            "profile_forward_sample_count": 3,
            "profile_forward_window_sample_count": 3,
            "profile_forward_child_count": 2,
            "profile_forward_tracked_time_s": 0.08,
            "profile_forward_untracked_time_s": 0.02,
            "profile_forward_overtracked_time_s": 0.01,
            "profile_forward_coverage_pct": 80.0,
            "profile_forward_untracked_pct": 20.0,
            "profile_forward_overtracked_pct_of_parent": 10.0,
            "profile_forward_top_time_s": 0.06,
            "profile_forward_top_pct_of_parent": 60.0,
            "profile_forward_top_avg_ms": 6.0,
            "profile_forward_top_p95_ms": 6.5,
            "profile_forward_top_calls": 3,
            "profile_loss_pct": 7.0,
            "profile_loss_reduce_pct": 3.0,
            "profile_backward_pct": 40.0,
            "profile_backward_grad_ready_child_count": 2,
            "profile_backward_grad_ready_parent_avg_ms": 14.0,
            "profile_backward_grad_ready_top_avg_ms": 6.0,
            "profile_backward_grad_ready_top_pct": 50.0,
            "profile_backward_grad_ready_top_p95_ms": 6.5,
            "profile_backward_grad_ready_top_calls": 3,
            "profile_optimizer_pct": 15.0,
            "profile_optimizer_p95_ms": 8.0,
            "profile_optimizer_child_count": 2,
            "profile_optimizer_tracked_time_s": 0.04,
            "profile_optimizer_untracked_time_s": 0.02,
            "profile_optimizer_overtracked_time_s": 0.01,
            "profile_optimizer_coverage_pct": 80.0,
            "profile_optimizer_untracked_pct": 20.0,
            "profile_optimizer_overtracked_pct_of_parent": 10.0,
            "profile_optimizer_top_time_s": 0.03,
            "profile_optimizer_top_pct_of_parent": 30.0,
            "profile_optimizer_top_avg_ms": 3.0,
            "profile_optimizer_top_p95_ms": 3.5,
            "profile_optimizer_top_calls": 3,
            "profile_user_metrics_pct": 8.0,
            "profile_postprocess_pct": 10.0,
            "profile_collect_output_pct": 5.0,
            "profile_metrics_pct": 2.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["runs"] == 2
    assert summary["mean_reported_samples_per_sec"] == pytest.approx(155.0)
    assert summary["min_end_to_end_wall_time_s"] == pytest.approx(1.25)
    assert summary["max_end_to_end_wall_time_s"] == pytest.approx(2.0)
    assert summary["stddev_wall_time_s"] == pytest.approx(0.5)
    assert summary["mean_p99_s"] == pytest.approx(0.010)
    assert summary["mean_std_batch_s"] == pytest.approx(0.001)
    assert summary["mean_avg_batch_s"] == pytest.approx(0.004)
    assert summary["mean_last_batch_s"] == pytest.approx(0.003)
    assert summary["mean_min_batch_s"] == pytest.approx(0.002)
    assert summary["mean_max_batch_s"] == pytest.approx(0.007)
    assert summary["mean_batches"] == pytest.approx(3.0)
    assert summary["mean_best_samples_per_sec"] == pytest.approx(250.0)
    assert summary["mean_headroom_ratio"] == pytest.approx(1.25)
    assert summary["mean_dataset_setup_time_s"] == pytest.approx(0.05)
    assert summary["sample_count_dataset_setup_time_s"] == pytest.approx(1.0)
    assert summary["mean_loader_setup_time_s"] == pytest.approx(0.07)
    assert summary["mean_model_setup_time_s"] == pytest.approx(0.13)
    assert summary["mean_compile_init_time_s"] == pytest.approx(0.02)
    assert summary["mean_transactions"] == pytest.approx(64.0)
    assert summary["mean_batch_size"] == pytest.approx(4.0)
    assert summary["mean_num_workers"] == pytest.approx(0.0)
    assert summary["mean_world_size"] == pytest.approx(1.0)
    assert summary["mean_dataset_materialized_bytes"] == pytest.approx(4096.0)
    assert summary["sample_count_dataset_materialized_bytes"] == pytest.approx(1.0)
    assert summary["mean_ema_samples_per_sec"] == pytest.approx(220.0)
    assert summary["mean_window_samples_per_sec"] == pytest.approx(210.0)
    assert summary["mean_window_time_s"] == pytest.approx(0.040)
    assert summary["mean_window_batches"] == pytest.approx(2.0)
    assert summary["mean_window_samples"] == pytest.approx(8.0)
    assert summary["mean_steps"] == pytest.approx(3.0)
    assert summary["mean_samples"] == pytest.approx(12.0)
    assert summary["mean_optimizer_steps"] == pytest.approx(2.0)
    assert summary["mean_grad_accum"] == pytest.approx(2.0)
    assert summary["mean_partial_optimizer_steps"] == pytest.approx(1.0)
    assert summary["mean_grad_accum_tail_steps"] == pytest.approx(1.0)
    assert summary["mean_warmup_steps"] == pytest.approx(1.0)
    assert summary["mean_warmup_samples_per_sec"] == pytest.approx(80.0)
    assert summary["mean_cold_start_samples_per_sec"] == pytest.approx(80.0)
    assert summary["mean_steady_steps"] == pytest.approx(2.0)
    assert summary["mean_steady_p99_s"] == pytest.approx(0.04)
    assert summary["mean_profile_flat_metric_invalid_count"] == pytest.approx(1.0)
    assert summary["max_profile_flat_metric_invalid_count"] == pytest.approx(2.0)
    assert summary["mean_profile_open_phase_count"] == pytest.approx(0.5)
    assert summary["max_profile_open_detail_count"] == pytest.approx(2.0)
    assert summary["mean_profile_model_modules_selected"] == pytest.approx(1.5)
    assert summary["max_profile_model_hook_count"] == pytest.approx(4.0)
    assert summary["mean_profile_model_hook_failures"] == pytest.approx(0.5)
    assert summary["mean_profile_forward_backward_pct"] == pytest.approx(50.0)
    assert summary["mean_profile_forward_p50_ms"] == pytest.approx(1.5)
    assert summary["mean_profile_forward_p95_ms"] == pytest.approx(3.0)
    assert summary["mean_profile_forward_p99_ms"] == pytest.approx(4.5)
    assert summary["mean_profile_forward_std_ms"] == pytest.approx(0.3)
    assert summary["mean_profile_forward_min_ms"] == pytest.approx(1.15)
    assert summary["mean_profile_forward_max_ms"] == pytest.approx(5.25)
    assert summary["mean_profile_forward_sample_count"] == pytest.approx(2.5)
    assert summary["mean_profile_forward_window_sample_count"] == pytest.approx(2.5)
    assert summary["mean_profile_forward_child_count"] == pytest.approx(1.5)
    assert summary["mean_profile_forward_tracked_time_s"] == pytest.approx(0.065)
    assert summary["mean_profile_forward_untracked_time_s"] == pytest.approx(0.015)
    assert summary["mean_profile_forward_overtracked_time_s"] == pytest.approx(0.005)
    assert summary["mean_profile_forward_coverage_pct"] == pytest.approx(80.0)
    assert summary["mean_profile_forward_untracked_pct"] == pytest.approx(20.0)
    assert summary["mean_profile_forward_overtracked_pct_of_parent"] == pytest.approx(5.0)
    assert summary["mean_profile_forward_top_time_s"] == pytest.approx(0.05)
    assert summary["mean_profile_forward_top_pct_of_parent"] == pytest.approx(50.0)
    assert summary["mean_profile_forward_top_avg_ms"] == pytest.approx(5.0)
    assert summary["mean_profile_forward_top_p95_ms"] == pytest.approx(5.5)
    assert summary["mean_profile_forward_top_calls"] == pytest.approx(2.5)
    assert summary["mean_profile_loss_pct"] == pytest.approx(6.0)
    assert summary["mean_profile_loss_reduce_pct"] == pytest.approx(2.5)
    assert summary["mean_profile_user_metrics_pct"] == pytest.approx(6.0)
    assert summary["mean_profile_postprocess_pct"] == pytest.approx(8.0)
    assert summary["mean_profile_collect_output_pct"] == pytest.approx(4.0)
    assert summary["mean_profile_metrics_pct"] == pytest.approx(1.5)
    assert summary["max_profile_backward_pct"] == pytest.approx(40.0)
    assert summary["mean_profile_backward_grad_ready_child_count"] == pytest.approx(1.5)
    assert summary["mean_profile_backward_grad_ready_parent_avg_ms"] == pytest.approx(13.0)
    assert summary["mean_profile_backward_grad_ready_top_avg_ms"] == pytest.approx(5.0)
    assert summary["mean_profile_backward_grad_ready_top_pct"] == pytest.approx(40.0)
    assert summary["mean_profile_backward_grad_ready_top_p95_ms"] == pytest.approx(5.5)
    assert summary["mean_profile_backward_grad_ready_top_calls"] == pytest.approx(2.5)
    assert summary["mean_profile_optimizer_child_count"] == pytest.approx(1.5)
    assert summary["mean_profile_optimizer_p95_ms"] == pytest.approx(6.0)
    assert summary["mean_profile_optimizer_tracked_time_s"] == pytest.approx(0.035)
    assert summary["mean_profile_optimizer_untracked_time_s"] == pytest.approx(0.015)
    assert summary["mean_profile_optimizer_overtracked_time_s"] == pytest.approx(0.005)
    assert summary["mean_profile_optimizer_coverage_pct"] == pytest.approx(77.5)
    assert summary["mean_profile_optimizer_untracked_pct"] == pytest.approx(22.5)
    assert summary["mean_profile_optimizer_overtracked_pct_of_parent"] == pytest.approx(5.0)
    assert summary["mean_profile_optimizer_top_time_s"] == pytest.approx(0.025)
    assert summary["mean_profile_optimizer_top_pct_of_parent"] == pytest.approx(25.0)
    assert summary["mean_profile_optimizer_top_avg_ms"] == pytest.approx(2.5)
    assert summary["mean_profile_optimizer_top_p95_ms"] == pytest.approx(3.0)
    assert summary["mean_profile_optimizer_top_calls"] == pytest.approx(2.5)
    assert summary["profiled_runs"] == 2
    assert summary["profile_model_status_counts"] == {"hook_failures": 1, "ok": 1}
    assert summary["best_reported"]["run"] == 1
    assert summary["best_reported"]["profile_flat_metric_invalid_count"] == pytest.approx(0.0)
    assert summary["best_reported"]["profile_open_phase_count"] == 0
    assert summary["best_reported"]["profile_open_detail_count"] == 0
    assert summary["best_reported"]["profile_model_requested"] is True
    assert summary["best_reported"]["profile_model_enabled"] is True
    assert summary["best_reported"]["profile_model_status"] == "ok"
    assert summary["best_reported"]["profile_model_modules_selected"] == 2
    assert summary["best_reported"]["profile_model_hook_count"] == 4
    assert summary["best_reported"]["profile_model_hook_failures"] == 0
    assert "profile_model_hook_last_error" not in summary["best_reported"]
    assert summary["best_reported"]["profile_forward_backward_pct"] == pytest.approx(60.0)
    assert summary["best_reported"]["profile_forward_p50_ms"] == pytest.approx(2.0)
    assert summary["best_reported"]["profile_forward_p95_ms"] == pytest.approx(4.0)
    assert summary["best_reported"]["profile_forward_p99_ms"] == pytest.approx(6.0)
    assert summary["best_reported"]["profile_forward_std_ms"] == pytest.approx(0.4)
    assert summary["best_reported"]["profile_forward_min_ms"] == pytest.approx(1.5)
    assert summary["best_reported"]["profile_forward_max_ms"] == pytest.approx(7.0)
    assert summary["best_reported"]["profile_forward_sample_count"] == 3
    assert summary["best_reported"]["profile_forward_window_sample_count"] == 3
    assert summary["best_reported"]["profile_forward_child_count"] == 2
    assert summary["best_reported"]["profile_forward_tracked_time_s"] == pytest.approx(0.08)
    assert summary["best_reported"]["profile_forward_untracked_time_s"] == pytest.approx(0.02)
    assert summary["best_reported"]["profile_forward_overtracked_time_s"] == pytest.approx(0.01)
    assert summary["best_reported"]["profile_forward_coverage_pct"] == pytest.approx(80.0)
    assert summary["best_reported"]["profile_forward_untracked_pct"] == pytest.approx(20.0)
    assert summary["best_reported"]["profile_forward_overtracked_pct_of_parent"] == pytest.approx(10.0)
    assert summary["best_reported"]["profile_forward_top_time_s"] == pytest.approx(0.06)
    assert summary["best_reported"]["profile_forward_top_pct_of_parent"] == pytest.approx(60.0)
    assert summary["best_reported"]["profile_forward_top_avg_ms"] == pytest.approx(6.0)
    assert summary["best_reported"]["profile_forward_top_p95_ms"] == pytest.approx(6.5)
    assert summary["best_reported"]["profile_forward_top_calls"] == 3
    assert summary["best_reported"]["profile_loss_pct"] == pytest.approx(7.0)
    assert summary["best_reported"]["profile_backward_grad_ready_child_count"] == 2
    assert summary["best_reported"]["profile_backward_grad_ready_parent_avg_ms"] == pytest.approx(14.0)
    assert summary["best_reported"]["profile_backward_grad_ready_top_avg_ms"] == pytest.approx(6.0)
    assert summary["best_reported"]["profile_backward_grad_ready_top_pct"] == pytest.approx(50.0)
    assert summary["best_reported"]["profile_backward_grad_ready_top_p95_ms"] == pytest.approx(6.5)
    assert summary["best_reported"]["profile_backward_grad_ready_top_calls"] == 3
    assert summary["best_reported"]["profile_optimizer_child_count"] == 2
    assert summary["best_reported"]["profile_optimizer_p95_ms"] == pytest.approx(8.0)
    assert summary["best_reported"]["profile_optimizer_tracked_time_s"] == pytest.approx(0.04)
    assert summary["best_reported"]["profile_optimizer_untracked_time_s"] == pytest.approx(0.02)
    assert summary["best_reported"]["profile_optimizer_overtracked_time_s"] == pytest.approx(0.01)
    assert summary["best_reported"]["profile_optimizer_coverage_pct"] == pytest.approx(80.0)
    assert summary["best_reported"]["profile_optimizer_untracked_pct"] == pytest.approx(20.0)
    assert summary["best_reported"]["profile_optimizer_overtracked_pct_of_parent"] == pytest.approx(10.0)
    assert summary["best_reported"]["profile_optimizer_top_time_s"] == pytest.approx(0.03)
    assert summary["best_reported"]["profile_optimizer_top_pct_of_parent"] == pytest.approx(30.0)
    assert summary["best_reported"]["profile_optimizer_top_avg_ms"] == pytest.approx(3.0)
    assert summary["best_reported"]["profile_optimizer_top_p95_ms"] == pytest.approx(3.5)
    assert summary["best_reported"]["profile_optimizer_top_calls"] == 3
    assert summary["best_reported"]["profile_postprocess_pct"] == pytest.approx(10.0)
    assert summary["best_reported"]["profile_collect_output_pct"] == pytest.approx(5.0)
    assert summary["best_reported"]["device"] == "cpu"
    assert summary["best_reported"]["amp"] is False
    assert summary["best_reported"]["compile_requested"] is True
    assert summary["best_reported"]["compiled"] is False
    assert summary["best_reported"]["compile_init_time_s"] == pytest.approx(0.02)
    assert summary["best_reported"]["compile_fallback_reason"] == "cpu_device"
    assert summary["best_reported"]["transactions"] == 64
    assert summary["best_reported"]["batch_size"] == 4
    assert summary["best_reported"]["num_workers"] == 0
    assert summary["best_reported"]["world_size"] == 1
    assert summary["best_reported"]["rank"] == 0
    assert summary["best_reported"]["avg_batch_s"] == pytest.approx(0.004)
    assert summary["best_reported"]["last_batch_s"] == pytest.approx(0.003)
    assert summary["best_reported"]["min_batch_s"] == pytest.approx(0.002)
    assert summary["best_reported"]["max_batch_s"] == pytest.approx(0.007)
    assert summary["best_reported"]["batches"] == 3
    assert summary["best_reported"]["ema_samples_per_sec"] == pytest.approx(220.0)
    assert summary["best_reported"]["window_samples_per_sec"] == pytest.approx(210.0)
    assert summary["best_reported"]["window_time_s"] == pytest.approx(0.040)
    assert summary["best_reported"]["window_batches"] == 2
    assert summary["best_reported"]["window_samples"] == 8
    assert summary["best_reported"]["dataset_setup_time_s"] == pytest.approx(0.05)
    assert summary["best_reported"]["loader_setup_time_s"] == pytest.approx(0.07)
    assert summary["best_reported"]["model_setup_time_s"] == pytest.approx(0.13)
    assert summary["best_reported"]["dataset_materialized_bytes"] == 4096
    assert summary["best_reported"]["steps"] == 3
    assert summary["best_reported"]["warmup_steps"] == 1
    assert summary["best_reported"]["steady_steps"] == 2
    assert summary["best_reported"]["steady_p99_s"] == pytest.approx(0.04)
    assert summary["best_end_to_end"]["run"] == 1


def test_summarize_results_preserves_backward_readiness_span_metrics() -> None:
    rows = [
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 90.0,
            "end_to_end_wall_time_s": 1.0,
            "profile_backward_grad_ready_earliest_avg_ms": 2.0,
            "profile_backward_grad_ready_latest_avg_ms": 8.0,
            "profile_backward_grad_ready_span_avg_ms": 6.0,
            "profile_backward_grad_ready_earliest_pct": 20.0,
            "profile_backward_grad_ready_latest_pct": 80.0,
            "profile_backward_grad_ready_span_pct": 60.0,
        },
        {
            "run": 1,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 180.0,
            "end_to_end_wall_time_s": 0.5,
            "profile_backward_grad_ready_earliest_avg_ms": 4.0,
            "profile_backward_grad_ready_latest_avg_ms": 10.0,
            "profile_backward_grad_ready_span_avg_ms": 6.0,
            "profile_backward_grad_ready_earliest_pct": 40.0,
            "profile_backward_grad_ready_latest_pct": 100.0,
            "profile_backward_grad_ready_span_pct": 60.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_profile_backward_grad_ready_earliest_avg_ms"] == pytest.approx(3.0)
    assert summary["mean_profile_backward_grad_ready_latest_avg_ms"] == pytest.approx(9.0)
    assert summary["mean_profile_backward_grad_ready_span_avg_ms"] == pytest.approx(6.0)
    assert summary["mean_profile_backward_grad_ready_earliest_pct"] == pytest.approx(30.0)
    assert summary["mean_profile_backward_grad_ready_latest_pct"] == pytest.approx(90.0)
    assert summary["mean_profile_backward_grad_ready_span_pct"] == pytest.approx(60.0)
    assert summary["best_reported"]["profile_backward_grad_ready_span_avg_ms"] == pytest.approx(6.0)
    assert summary["best_reported"]["profile_backward_grad_ready_latest_pct"] == pytest.approx(100.0)
    json.dumps(summary, allow_nan=False)


def test_summarize_results_ranks_profile_bottleneck_candidates() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 90.0,
            "end_to_end_wall_time_s": 1.0,
            "profile_forward_pct": 20.0,
            "profile_forward_avg_ms": 2.0,
            "profile_forward_untracked_pct": 25.0,
            "profile_forward_coverage_pct": 75.0,
            "profile_forward_top_pct_of_parent": 50.0,
            "profile_forward_top_avg_ms": 5.0,
            "profile_backward_pct": 50.0,
            "profile_backward_grad_ready_span_pct": 80.0,
            "profile_backward_grad_ready_span_avg_ms": 8.0,
            "profile_backward_grad_ready_top_pct": 30.0,
            "profile_backward_grad_ready_top_avg_ms": 3.0,
            "profile_optimizer_pct": 10.0,
            "profile_optimizer_untracked_pct": 20.0,
            "profile_optimizer_coverage_pct": 80.0,
            "profile_optimizer_top_pct_of_parent": 90.0,
            "profile_optimizer_top_avg_ms": 3.0,
        },
    ])

    candidates = summary["profile_bottleneck_candidates"]

    assert summary["profile_bottleneck_candidate_count"] == 9
    assert summary["profile_bottleneck_candidate_returned_count"] == 8
    assert summary["profile_bottleneck_candidate_limit"] == 8
    assert summary["profile_bottleneck_candidate_omitted_count"] == 1
    assert summary["profile_bottleneck_severity_thresholds"] == {
        "score_unit": "profile_pct",
        "levels": [
            {"severity": "high", "min_score": pytest.approx(25.0)},
            {"severity": "medium", "min_score": pytest.approx(10.0)},
            {"severity": "low", "min_score": pytest.approx(0.0)},
        ],
    }
    assert summary["profile_bottleneck_severity_counts"] == {
        "high": 2,
        "medium": 4,
        "low": 3,
    }
    assert len(candidates) == 8
    assert summary["profile_bottleneck_top_candidate"] == candidates[0]
    category_summary = summary["profile_bottleneck_category_summary"]
    assert summary["profile_bottleneck_category_order"] == [
        "phase_share",
        "readiness_span",
        "child_hotspot",
        "coverage_gap",
    ]
    assert summary["profile_bottleneck_top_category"] == {
        "category": "phase_share",
        "count": 3,
        "max_score": pytest.approx(50.0),
        "total_score": pytest.approx(80.0),
        "mean_score": pytest.approx(80.0 / 3.0),
        "pressure_score": pytest.approx(80.0),
        "pressure_score_unit": "profile_pct",
        "returned_count": 3,
        "omitted_count": 0,
        "score_unit": "profile_pct",
        "top_candidate": "backward_phase",
        "top_rank": 1,
        "top_severity": "high",
        "top_candidate_returned": True,
        "top_score": pytest.approx(50.0),
        "top_score_unit": "profile_pct",
        "top_metric": "profile_backward_pct",
        "top_value": pytest.approx(50.0),
        "top_unit": "profile_pct",
        "top_score_basis": "direct_metric",
        "top_score_formula": "score=value",
        "top_label": "backward phase",
        "top_reason": "backward owns a large share of profiled loop time",
        "top_next_step": "inspect gradient-ready span and backward top-child metrics",
        "severity_counts": {"high": 1, "medium": 2},
        "pressure_rank": 1,
    }
    assert category_summary["phase_share"] == {
        "count": 3,
        "max_score": pytest.approx(50.0),
        "total_score": pytest.approx(80.0),
        "mean_score": pytest.approx(80.0 / 3.0),
        "pressure_score": pytest.approx(80.0),
        "pressure_score_unit": "profile_pct",
        "returned_count": 3,
        "omitted_count": 0,
        "score_unit": "profile_pct",
        "top_candidate": "backward_phase",
        "top_rank": 1,
        "top_severity": "high",
        "top_candidate_returned": True,
        "top_score": pytest.approx(50.0),
        "top_score_unit": "profile_pct",
        "top_metric": "profile_backward_pct",
        "top_value": pytest.approx(50.0),
        "top_unit": "profile_pct",
        "top_score_basis": "direct_metric",
        "top_score_formula": "score=value",
        "top_label": "backward phase",
        "top_reason": "backward owns a large share of profiled loop time",
        "top_next_step": "inspect gradient-ready span and backward top-child metrics",
        "severity_counts": {"high": 1, "medium": 2},
        "pressure_rank": 1,
    }
    assert category_summary["child_hotspot"] == {
        "count": 3,
        "max_score": pytest.approx(15.0),
        "total_score": pytest.approx(34.0),
        "mean_score": pytest.approx(34.0 / 3.0),
        "pressure_score": pytest.approx(34.0),
        "pressure_score_unit": "profile_pct",
        "returned_count": 3,
        "omitted_count": 0,
        "score_unit": "profile_pct",
        "top_candidate": "backward_ready_top_child",
        "top_rank": 4,
        "top_severity": "medium",
        "top_candidate_returned": True,
        "top_score": pytest.approx(15.0),
        "top_score_unit": "profile_pct",
        "top_metric": "profile_backward_grad_ready_top_pct",
        "top_value": pytest.approx(30.0),
        "top_unit": "pct_of_parent",
        "top_score_basis": "parent_metric_weighted",
        "top_score_formula": "score=parent_value*value/100",
        "top_parent_metric": "profile_backward_pct",
        "top_parent_value": pytest.approx(50.0),
        "top_label": "backward ready top child",
        "top_reason": "one module dominates gradient-ready timing",
        "top_next_step": "focus backward inspection on the slowest ready module",
        "severity_counts": {"medium": 2, "low": 1},
        "pressure_rank": 3,
    }
    assert category_summary["coverage_gap"] == {
        "count": 2,
        "max_score": pytest.approx(5.0),
        "total_score": pytest.approx(7.0),
        "mean_score": pytest.approx(3.5),
        "pressure_score": pytest.approx(7.0),
        "pressure_score_unit": "profile_pct",
        "returned_count": 1,
        "omitted_count": 1,
        "score_unit": "profile_pct",
        "top_candidate": "forward_untracked",
        "top_rank": 8,
        "top_severity": "low",
        "top_candidate_returned": True,
        "top_score": pytest.approx(5.0),
        "top_score_unit": "profile_pct",
        "top_metric": "profile_forward_untracked_pct",
        "top_value": pytest.approx(25.0),
        "top_unit": "pct_of_parent",
        "top_score_basis": "parent_metric_weighted",
        "top_score_formula": "score=parent_value*value/100",
        "top_parent_metric": "profile_forward_pct",
        "top_parent_value": pytest.approx(20.0),
        "top_label": "forward untracked time",
        "top_reason": "a large share of forward time is outside child timers",
        "top_next_step": "increase model profiling coverage or include narrower module filters",
        "severity_counts": {"low": 2},
        "pressure_rank": 4,
    }
    assert [candidate["name"] for candidate in candidates[:4]] == [
        "backward_phase",
        "backward_readiness_span",
        "forward_phase",
        "backward_ready_top_child",
    ]
    assert candidates == sorted(
        candidates,
        key=lambda candidate: (
            -candidate["score"],
            -candidate["value"],
            candidate["name"],
        ),
    )
    assert [candidate["rank"] for candidate in candidates] == list(range(1, 9))
    backward_phase = candidates[0]
    assert backward_phase["metric"] == "profile_backward_pct"
    assert backward_phase["label"] == "backward phase"
    assert backward_phase["category"] == "phase_share"
    assert backward_phase["reason"] == "backward owns a large share of profiled loop time"
    assert backward_phase["next_step"] == "inspect gradient-ready span and backward top-child metrics"
    assert backward_phase["score"] == pytest.approx(50.0)
    assert backward_phase["score_unit"] == "profile_pct"
    assert backward_phase["score_basis"] == "direct_metric"
    assert backward_phase["score_formula"] == "score=value"
    assert backward_phase["severity"] == "high"
    backward_span = candidates[1]
    assert backward_span["score"] == pytest.approx(40.0)
    assert backward_span["severity"] == "high"
    assert backward_span["parent_metric"] == "profile_backward_pct"
    assert backward_span["parent_value"] == pytest.approx(50.0)
    assert backward_span["score_basis"] == "parent_metric_weighted"
    assert backward_span["score_formula"] == "score=parent_value*value/100"
    assert backward_span["span_avg_ms"] == pytest.approx(8.0)
    assert candidates[2]["severity"] == "medium"
    forward_top = next(
        candidate
        for candidate in candidates
        if candidate["name"] == "forward_top_child"
    )
    assert forward_top["score"] == pytest.approx(10.0)
    assert forward_top["severity"] == "medium"
    assert forward_top["avg_ms"] == pytest.approx(5.0)
    json.dumps(summary, allow_nan=False)


def test_summarize_results_marks_omitted_profile_bottleneck_category_top(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bpt, "PROFILE_BOTTLENECK_CANDIDATE_LIMIT", 1)

    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 90.0,
            "end_to_end_wall_time_s": 1.0,
            "profile_forward_pct": 50.0,
            "profile_forward_top_pct_of_parent": 20.0,
            "profile_forward_top_avg_ms": 2.0,
        },
    ])

    category = summary["profile_bottleneck_category_summary"]["child_hotspot"]

    assert summary["profile_bottleneck_candidate_count"] == 2
    assert summary["profile_bottleneck_candidate_returned_count"] == 1
    assert summary["profile_bottleneck_candidate_omitted_count"] == 1
    assert category["top_candidate"] == "forward_top_child"
    assert category["top_rank"] == 2
    assert category["top_candidate_returned"] is False
    assert category["returned_count"] == 0
    assert category["omitted_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_add_profile_bottleneck_candidates_clears_stale_fields() -> None:
    summary = {
        "profile_forward_pct": 25.0,
        "profile_bottleneck_candidate_omitted_count": 7,
        "profile_bottleneck_candidates": [{"name": "stale"}],
        "profile_bottleneck_top_candidate": {"name": "stale"},
    }

    bpt._add_profile_bottleneck_candidates(summary)

    assert summary["profile_bottleneck_candidate_count"] == 1
    assert summary["profile_bottleneck_candidates"][0]["name"] == "forward_phase"
    assert "profile_bottleneck_candidate_omitted_count" not in summary

    summary.pop("profile_forward_pct")
    bpt._add_profile_bottleneck_candidates(summary)

    for field in bpt.PROFILE_BOTTLENECK_COUNT_FIELDS + bpt.PROFILE_BOTTLENECK_OBJECT_FIELDS:
        assert field not in summary


def test_summarize_results_preserves_best_run_profile_bottleneck_context() -> None:
    category_summary = {
        "phase_share": {
            "count": 1,
            "max_score": 42.0,
            "total_score": 42.0,
            "mean_score": 42.0,
            "pressure_score": 42.0,
            "pressure_score_unit": "profile_pct",
            "returned_count": 1,
            "omitted_count": 0,
            "score_unit": "profile_pct",
            "top_candidate": "forward_phase",
            "top_rank": 1,
            "top_candidate_returned": True,
            "top_score": 42.0,
            "top_score_unit": "profile_pct",
            "top_metric": "profile_forward_pct",
            "top_value": 42.0,
            "top_unit": "profile_pct",
            "top_score_basis": "direct_metric",
            "top_score_formula": "score=value",
            "top_label": "forward phase",
            "top_reason": "forward owns a large share of profiled loop time",
            "top_next_step": "inspect forward top-child and tail metrics",
            "top_severity": "high",
            "severity_counts": {"high": 1},
            "pressure_rank": 1,
        },
    }
    top_candidate = {
        "name": "forward_phase",
        "metric": "profile_forward_pct",
        "value": 42.0,
        "unit": "profile_pct",
        "score": 42.0,
        "score_unit": "profile_pct",
        "rank": 1,
        "severity": "high",
    }

    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "end_to_end_wall_time_s": 1.0,
            "profile_bottleneck_candidate_count": 2,
            "profile_bottleneck_candidate_returned_count": 1,
            "profile_bottleneck_candidate_limit": 1,
            "profile_bottleneck_candidate_omitted_count": 1,
            "profile_bottleneck_candidates": [top_candidate],
            "profile_bottleneck_top_candidate": top_candidate,
            "profile_bottleneck_top_category": {
                **category_summary["phase_share"],
                "category": "phase_share",
            },
            "profile_bottleneck_category_order": ["phase_share"],
            "profile_bottleneck_severity_counts": {"high": 1},
            "profile_bottleneck_severity_thresholds": {
                "score_unit": "profile_pct",
                "levels": [{"severity": "high", "min_score": 25.0}],
            },
            "profile_bottleneck_category_summary": category_summary,
        },
    ])

    best_reported = summary["best_reported"]

    assert best_reported["profile_bottleneck_candidate_count"] == 2
    assert best_reported["profile_bottleneck_candidate_returned_count"] == 1
    assert best_reported["profile_bottleneck_candidate_limit"] == 1
    assert best_reported["profile_bottleneck_candidate_omitted_count"] == 1
    assert best_reported["profile_bottleneck_candidates"] == [top_candidate]
    assert best_reported["profile_bottleneck_top_candidate"] == top_candidate
    assert best_reported["profile_bottleneck_top_category"]["category"] == "phase_share"
    assert best_reported["profile_bottleneck_category_order"] == ["phase_share"]
    assert best_reported["profile_bottleneck_severity_counts"] == {"high": 1}
    assert best_reported["profile_bottleneck_severity_thresholds"]["score_unit"] == "profile_pct"
    assert best_reported["profile_bottleneck_category_summary"] == category_summary
    json.dumps(summary, allow_nan=False)


def test_summarize_results_skips_invalid_profile_bottleneck_candidates() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 90.0,
            "end_to_end_wall_time_s": 1.0,
            "profile_forward_pct": float("inf"),
            "profile_forward_untracked_pct": 50.0,
            "profile_forward_top_pct_of_parent": 50.0,
            "profile_backward_pct": -1.0,
            "profile_backward_grad_ready_span_pct": 50.0,
            "profile_optimizer_pct": 120.0,
        },
    ])

    assert "profile_bottleneck_candidate_count" not in summary
    assert "profile_bottleneck_category_order" not in summary
    assert "profile_bottleneck_category_summary" not in summary
    assert "profile_bottleneck_severity_thresholds" not in summary
    assert "profile_bottleneck_severity_counts" not in summary
    assert "profile_bottleneck_top_category" not in summary
    assert "profile_bottleneck_top_candidate" not in summary
    assert "profile_bottleneck_candidates" not in summary
    json.dumps(summary, allow_nan=False)


def test_summarize_results_preserves_best_run_profile_model_failure_context() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "wall_time_s": 1.0,
            "profile_model_requested": True,
            "profile_model_enabled": True,
            "profile_model_status": "hook_failures",
            "profile_model_modules_selected": 1,
            "profile_model_hook_count": 2,
            "profile_model_hook_failures": 1,
            "profile_model_hook_last_error": "RuntimeError: forward hook boom",
        },
    ])

    best_reported = summary["best_reported"]
    assert best_reported["profile_model_requested"] is True
    assert best_reported["profile_model_enabled"] is True
    assert best_reported["profile_model_status"] == "hook_failures"
    assert best_reported["profile_model_hook_last_error"] == "RuntimeError: forward hook boom"
    json.dumps(summary, allow_nan=False)


def test_summarize_results_preserves_top_profile_distribution_metrics() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 180.0,
            "wall_time_s": 1.2,
            "profile_forward_top_p50_ms": 2.0,
            "profile_forward_top_p95_ms": 4.0,
            "profile_forward_top_p99_ms": 6.0,
            "profile_forward_top_std_ms": 0.5,
            "profile_forward_top_min_ms": 1.0,
            "profile_forward_top_max_ms": 7.0,
            "profile_forward_top_sample_count": 3,
            "profile_forward_top_window_sample_count": 2,
            "profile_backward_grad_ready_top_p99_ms": 5.0,
            "profile_backward_grad_ready_top_std_ms": 0.4,
            "profile_backward_grad_ready_top_sample_count": 3,
            "profile_optimizer_top_p99_ms": 3.0,
            "profile_optimizer_top_std_ms": 0.3,
            "profile_optimizer_top_window_sample_count": 2,
        },
        {
            "run": 1,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "wall_time_s": 1.0,
            "profile_forward_top_p50_ms": 3.0,
            "profile_forward_top_p95_ms": 5.0,
            "profile_forward_top_p99_ms": 9.0,
            "profile_forward_top_std_ms": 0.7,
            "profile_forward_top_min_ms": 2.0,
            "profile_forward_top_max_ms": 10.0,
            "profile_forward_top_sample_count": 4,
            "profile_forward_top_window_sample_count": 3,
            "profile_backward_grad_ready_top_p99_ms": 8.0,
            "profile_backward_grad_ready_top_std_ms": 0.6,
            "profile_backward_grad_ready_top_sample_count": 4,
            "profile_optimizer_top_p99_ms": 4.0,
            "profile_optimizer_top_std_ms": 0.5,
            "profile_optimizer_top_window_sample_count": 3,
        },
    ])

    assert summary["mean_profile_forward_top_p50_ms"] == pytest.approx(2.5)
    assert summary["mean_profile_forward_top_p99_ms"] == pytest.approx(7.5)
    assert summary["mean_profile_forward_top_std_ms"] == pytest.approx(0.6)
    assert summary["mean_profile_forward_top_min_ms"] == pytest.approx(1.5)
    assert summary["max_profile_forward_top_max_ms"] == pytest.approx(10.0)
    assert summary["mean_profile_forward_top_sample_count"] == pytest.approx(3.5)
    assert summary["mean_profile_forward_top_window_sample_count"] == pytest.approx(2.5)
    assert summary["mean_profile_backward_grad_ready_top_p99_ms"] == pytest.approx(6.5)
    assert summary["mean_profile_backward_grad_ready_top_std_ms"] == pytest.approx(0.5)
    assert summary["mean_profile_backward_grad_ready_top_sample_count"] == pytest.approx(3.5)
    assert summary["mean_profile_optimizer_top_p99_ms"] == pytest.approx(3.5)
    assert summary["mean_profile_optimizer_top_std_ms"] == pytest.approx(0.4)
    assert summary["mean_profile_optimizer_top_window_sample_count"] == pytest.approx(2.5)
    assert summary["best_reported"]["profile_forward_top_p99_ms"] == pytest.approx(9.0)
    assert summary["best_reported"]["profile_forward_top_sample_count"] == 4
    assert summary["best_reported"]["profile_backward_grad_ready_top_p99_ms"] == pytest.approx(8.0)
    assert summary["best_reported"]["profile_optimizer_top_window_sample_count"] == 3
    json.dumps(summary, allow_nan=False)


def test_summarize_results_omits_zero_only_scheduler_diagnostics() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "wall_time_s": 1.0,
            "scheduler_step_failures": 0,
            "scheduler_last_error": "RuntimeError: stale scheduler error",
        },
    ])

    assert "mean_scheduler_step_failures" not in summary
    assert "max_scheduler_step_failures" not in summary
    assert "scheduler_step_failures" not in summary["best_reported"]
    assert "scheduler_last_error" not in summary["best_reported"]
    assert summary["best_reported_omitted_fields"] == [
        {"field": "scheduler_last_error", "reason": "inactive_context"},
    ]
    json.dumps(summary, allow_nan=False)


def test_summarize_results_preserves_scheduler_failure_context() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 180.0,
            "wall_time_s": 1.1,
            "scheduler_step_failures": 0,
            "scheduler_last_error": "",
        },
        {
            "run": 1,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "wall_time_s": 1.0,
            "scheduler_step_failures": 2,
            "scheduler_last_error": "RuntimeError: scheduler boom",
        },
    ])

    assert summary["mean_scheduler_step_failures"] == pytest.approx(1.0)
    assert summary["max_scheduler_step_failures"] == pytest.approx(2.0)
    assert summary["best_reported"]["scheduler_step_failures"] == 2
    assert summary["best_reported"]["scheduler_last_error"] == "RuntimeError: scheduler boom"
    json.dumps(summary, allow_nan=False)


def test_summarize_results_omits_not_requested_profile_model_status_counts() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "wall_time_s": 1.0,
            "profile_model_requested": False,
            "profile_model_enabled": False,
            "profile_model_status": "not_requested",
            "profile_model_modules_selected": 0,
            "profile_model_hook_count": 0,
            "profile_model_hook_failures": 0,
        },
    ])

    assert "profile_model_status_counts" not in summary
    assert "profile_model_status_invalid_count" not in summary
    assert "profiled_runs" not in summary
    assert "mean_profile_model_modules_selected" not in summary
    assert "mean_profile_model_hook_count" not in summary
    assert "mean_profile_model_hook_failures" not in summary
    assert "profile_model_requested" not in summary["best_reported"]
    assert "profile_model_enabled" not in summary["best_reported"]
    assert "profile_model_status" not in summary["best_reported"]
    assert "profile_model_modules_selected" not in summary["best_reported"]
    assert "profile_model_hook_count" not in summary["best_reported"]
    assert "profile_model_hook_failures" not in summary["best_reported"]
    json.dumps(summary, allow_nan=False)


def test_summarize_results_surfaces_summary_diagnostic_fields() -> None:
    summary = summarize_results([
        {
            "run": 0,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 90.0,
            "steady_samples_per_sec": 100.0,
            "end_to_end_wall_time_s": 1.1,
            "setup_time_s": 0.1,
            "wall_time_s": 1.0,
            "cold_start_time_s": 0.2,
            "grad_accum": 2,
            "profile_forward_backward_pct": 50.0,
            "profile_model_status": "unknown_status",
        },
        {
            "run": 1,
            "dataset_mode": "generated",
            "reported_samples_per_sec": float("nan"),
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 110.0,
            "end_to_end_wall_time_s": 0.9,
            "setup_time_s": 0.1,
            "wall_time_s": 0.8,
            "grad_accum": 0,
            "profile_forward_backward_pct": 125.0,
        },
    ])

    diagnostics = {
        diagnostic["field"]: diagnostic
        for diagnostic in summary["summary_diagnostic_fields"]
    }
    assert summary["summary_diagnostic_field_count"] == 5
    assert summary["summary_missing_field_count"] == 1
    assert summary["summary_non_finite_field_count"] == 1
    assert summary["summary_invalid_field_count"] == 3
    assert diagnostics["profile_model_status"]["invalid_count"] == 1
    assert diagnostics["reported_samples_per_sec"]["non_finite_count"] == 1
    assert diagnostics["cold_start_time_s"]["missing_count"] == 1
    assert diagnostics["grad_accum"]["invalid_count"] == 1
    assert diagnostics["profile_forward_backward_pct"]["invalid_count"] == 1
    assert "samples_per_sec" not in diagnostics
    json.dumps(summary, allow_nan=False)


def test_summarize_results_includes_device_memory_metrics_when_present() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 90.0,
            "steady_samples_per_sec": 100.0,
            "wall_time_s": 1.0,
            "end_to_end_wall_time_s": 1.2,
            "cuda_current_mem_bytes": 1024,
            "cuda_max_mem_bytes": 2048,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 90.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 90.0,
            "wall_time_s": 1.5,
            "end_to_end_wall_time_s": 1.7,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_cuda_current_mem_bytes"] == pytest.approx(1024.0)
    assert summary["sample_count_cuda_current_mem_bytes"] == pytest.approx(1.0)
    assert summary["mean_cuda_max_mem_bytes"] == pytest.approx(2048.0)
    assert summary["best_reported"]["cuda_max_mem_bytes"] == 2048
    assert "mean_mps_current_mem_bytes" not in summary


def test_summarize_results_handles_empty_input() -> None:
    summary = summarize_results([])

    assert summary["runs"] == 0
    assert summary["best_reported"] is None
    assert summary["best_end_to_end"] is None
    assert summary["mean_wall_time_s"] == pytest.approx(0.0)
    assert summary["stddev_wall_time_s"] == pytest.approx(0.0)
    assert "mean_profile_forward_backward_pct" not in summary


def test_summarize_results_keeps_missing_base_metrics_out_of_best_runs() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "setup_time_s": 0.25,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_reported_samples_per_sec"] == pytest.approx(0.0)
    assert summary["sample_count_reported_samples_per_sec"] == pytest.approx(0.0)
    assert summary["missing_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert summary["mean_end_to_end_wall_time_s"] == pytest.approx(0.0)
    assert summary["sample_count_end_to_end_wall_time_s"] == pytest.approx(0.0)
    assert summary["missing_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["best_reported"] is None
    assert summary["best_end_to_end"] is None


def test_summarize_results_skips_profile_fields_when_absent() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_reported_samples_per_sec"] == pytest.approx(100.0)
    assert "mean_profile_forward_backward_pct" not in summary
    assert "mean_profile_backward_grad_ready_top_pct" not in summary
    assert "profile_forward_backward_pct" not in summary["best_reported"]
    assert "profile_backward_grad_ready_top_pct" not in summary["best_reported"]
    assert "profiled_runs" not in summary


def test_summarize_results_ignores_missing_rows_for_profile_aggregates() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "profile_forward_backward_pct": 60.0,
            "profile_backward_pct": 35.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["runs"] == 2
    assert summary["profiled_runs"] == 1
    assert summary["mean_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert summary["min_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert summary["stddev_profile_forward_backward_pct"] == pytest.approx(0.0)
    assert summary["sample_count_profile_forward_backward_pct"] == pytest.approx(1.0)


def test_summarize_results_skips_non_finite_profile_values() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "profile_forward_backward_pct": float("nan"),
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "profile_forward_backward_pct": 60.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["profiled_runs"] == 2
    assert summary["mean_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert summary["sample_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    assert summary["non_finite_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)


def test_summarize_results_skips_non_finite_best_rank_values() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": float("nan"),
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": float("inf"),
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "end_to_end_wall_time_s": 0.9,
        },
    ]

    summary = summarize_results(rows)

    assert summary["best_reported"]["run"] == 1
    assert summary["best_end_to_end"]["run"] == 1
    assert summary["sample_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert summary["non_finite_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert summary["sample_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["non_finite_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)


def test_summarize_results_skips_negative_best_rank_values() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": -10.0,
            "samples_per_sec": -8.0,
            "steady_samples_per_sec": -10.0,
            "wall_time_s": -1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": -1.0,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "end_to_end_wall_time_s": 0.9,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_reported_samples_per_sec"] == pytest.approx(120.0)
    assert summary["sample_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert summary["invalid_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert summary["mean_end_to_end_wall_time_s"] == pytest.approx(0.9)
    assert summary["sample_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["invalid_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["best_reported"]["run"] == 1
    assert summary["best_end_to_end"]["run"] == 1
    json.dumps(summary, allow_nan=False)


def test_summarize_results_skips_out_of_range_percentages() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
            "profile_forward_backward_pct": 125.0,
            "profile_forward_top_pct_of_parent": 125.0,
            "profile_loss_pct": -1.0,
            "profile_backward_pct": 60.0,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 180.0,
            "steady_samples_per_sec": 200.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "end_to_end_wall_time_s": 1.10,
            "profile_forward_backward_pct": 50.0,
            "profile_forward_top_pct_of_parent": 75.0,
            "profile_loss_pct": 8.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_profile_forward_backward_pct"] == pytest.approx(50.0)
    assert summary["sample_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    assert summary["invalid_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    assert summary["mean_profile_forward_top_pct_of_parent"] == pytest.approx(100.0)
    assert "invalid_count_profile_forward_top_pct_of_parent" not in summary
    assert summary["mean_profile_loss_pct"] == pytest.approx(8.0)
    assert summary["invalid_count_profile_loss_pct"] == pytest.approx(1.0)
    assert summary["best_reported"]["run"] == 0
    assert "profile_forward_backward_pct" not in summary["best_reported"]
    assert "profile_loss_pct" not in summary["best_reported"]
    assert summary["best_reported"]["profile_forward_top_pct_of_parent"] == pytest.approx(125.0)
    assert summary["best_reported"]["profile_backward_pct"] == pytest.approx(60.0)
    omitted = {
        row["field"]: row["reason"]
        for row in summary["best_reported_omitted_fields"]
    }
    assert omitted["profile_forward_backward_pct"] == "above_max"
    assert omitted["profile_loss_pct"] == "negative"
    json.dumps(summary, allow_nan=False)


def test_summarize_results_omits_non_finite_best_run_fields() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
            "p99_s": True,
            "steps": "many",
            "optimizer_steps": "2",
            "profile_forward_backward_pct": float("nan"),
            "profile_backward_pct": float("inf"),
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 180.0,
            "steady_samples_per_sec": 200.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "end_to_end_wall_time_s": 1.10,
            "profile_forward_backward_pct": 50.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["best_reported"]["run"] == 0
    assert "p99_s" not in summary["best_reported"]
    assert "steps" not in summary["best_reported"]
    assert "optimizer_steps" not in summary["best_reported"]
    assert "profile_forward_backward_pct" not in summary["best_reported"]
    assert "profile_backward_pct" not in summary["best_reported"]
    assert {
        row["field"]: row["reason"]
        for row in summary["best_reported_omitted_fields"]
    } == {
        "p99_s": "non_numeric_or_non_finite",
        "steps": "invalid_integer",
        "optimizer_steps": "invalid_integer",
        "profile_forward_backward_pct": "non_numeric_or_non_finite",
        "profile_backward_pct": "non_numeric_or_non_finite",
    }
    assert "best_end_to_end_omitted_fields" not in summary
    json.dumps(summary, allow_nan=False)


def test_summarize_results_omits_invalid_best_run_identity_and_counts() -> None:
    rows = [
        {
            "run": 0.5,
            "seed": 10.5,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": -1.0,
            "setup_time_s": -0.25,
            "end_to_end_wall_time_s": 1.25,
            "steps": 2.0,
            "samples": -12,
            "grad_accum": 0,
            "profile_flat_metric_invalid_count": -1.0,
            "profile_open_phase_count": -1.0,
            "profile_open_detail_count": 0.5,
            "profile_model_requested": 1,
            "profile_model_enabled": "true",
            "profile_model_status": "missing",
            "profile_model_modules_selected": -1.0,
            "profile_model_hook_count": 0.5,
            "profile_model_hook_failures": True,
            "profile_model_hook_last_error": "",
            "profile_forward_backward_pct": 125.0,
            "cuda_max_mem_bytes": 2048.5,
        },
    ]

    summary = summarize_results(rows)
    best_reported = summary["best_reported"]

    assert best_reported["dataset_mode"] == "generated"
    assert best_reported["reported_samples_per_sec"] == pytest.approx(300.0)
    assert "run" not in best_reported
    assert "seed" not in best_reported
    assert "wall_time_s" not in best_reported
    assert "setup_time_s" not in best_reported
    assert "steps" not in best_reported
    assert "samples" not in best_reported
    assert "grad_accum" not in best_reported
    assert "profile_flat_metric_invalid_count" not in best_reported
    assert "profile_open_phase_count" not in best_reported
    assert "profile_open_detail_count" not in best_reported
    assert "profile_model_requested" not in best_reported
    assert "profile_model_enabled" not in best_reported
    assert "profile_model_status" not in best_reported
    assert "profile_model_modules_selected" not in best_reported
    assert "profile_model_hook_count" not in best_reported
    assert "profile_model_hook_failures" not in best_reported
    assert "profile_model_hook_last_error" not in best_reported
    assert "profile_model_status_counts" not in best_reported
    assert summary["profile_model_status_invalid_count"] == 1
    assert "profile_forward_backward_pct" not in best_reported
    assert "cuda_max_mem_bytes" not in best_reported
    omitted = {
        row["field"]: row["reason"]
        for row in summary["best_reported_omitted_fields"]
    }
    assert omitted["run"] == "invalid_integer"
    assert omitted["seed"] == "invalid_integer"
    assert omitted["wall_time_s"] == "negative"
    assert omitted["setup_time_s"] == "negative"
    assert omitted["grad_accum"] == "invalid_integer"
    assert omitted["profile_model_requested"] == "invalid_boolean"
    assert omitted["profile_model_enabled"] == "invalid_boolean"
    assert omitted["profile_model_status"] == "invalid_choice"
    assert omitted["profile_forward_backward_pct"] == "above_max"
    assert omitted["cuda_max_mem_bytes"] == "invalid_integer"
    json.dumps(summary, allow_nan=False)


def test_summarize_results_omits_fractional_profile_invalid_count_from_best_run() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
            "profile_flat_metric_invalid_count": 0.5,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_profile_flat_metric_invalid_count"] == pytest.approx(0.0)
    assert summary["sample_count_profile_flat_metric_invalid_count"] == pytest.approx(0.0)
    assert summary["invalid_count_profile_flat_metric_invalid_count"] == pytest.approx(1.0)
    assert "profile_flat_metric_invalid_count" not in summary["best_reported"]
    assert summary["best_reported_omitted_fields"] == [
        {"field": "profile_flat_metric_invalid_count", "reason": "invalid_integer"},
    ]
    json.dumps(summary, allow_nan=False)


def test_summarize_results_skips_zero_grad_accum_in_aggregates() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
            "grad_accum": 0,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 180.0,
            "steady_samples_per_sec": 200.0,
            "wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "end_to_end_wall_time_s": 1.10,
            "grad_accum": 2,
        },
    ]

    summary = summarize_results(rows)

    assert summary["mean_grad_accum"] == pytest.approx(2.0)
    assert summary["sample_count_grad_accum"] == pytest.approx(1.0)
    assert summary["invalid_count_grad_accum"] == pytest.approx(1.0)
    assert "grad_accum" not in summary["best_reported"]
    assert summary["best_reported_omitted_fields"] == [
        {"field": "grad_accum", "reason": "invalid_integer"},
    ]
    json.dumps(summary, allow_nan=False)


def test_summarize_results_normalizes_best_run_dataset_mode() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": " materialized ",
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
        },
    ]

    summary = summarize_results(rows)

    assert summary["best_reported"]["dataset_mode"] == "materialized"
    assert summary["best_end_to_end"]["dataset_mode"] == "materialized"


@pytest.mark.parametrize(
    "dataset_mode",
    [None, True, "", "   ", "cached"],
)
def test_summarize_results_rejects_invalid_best_run_dataset_mode(
    dataset_mode: object,
) -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "dataset_mode": dataset_mode,
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 280.0,
            "steady_samples_per_sec": 300.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
        },
    ]

    with pytest.raises(ValueError, match="dataset_mode"):
        summarize_results(rows)


def test_best_finite_row_requires_positive_sample_count() -> None:
    rows = [
        {"run": 0, "mean_score": 300.0, "sample_count_score": float("nan")},
        {"run": 1, "mean_score": 250.0, "sample_count_score": -1.0},
        {"run": 2, "mean_score": 200.0, "sample_count_score": 0.0},
        {"run": 3, "mean_score": 150.0, "sample_count_score": 2.0},
    ]

    best = _best_finite_row(
        rows,
        "mean_score",
        prefer_high=True,
        sample_count_field="sample_count_score",
    )

    assert best == rows[3]


def test_best_finite_row_rejects_fractional_sample_count() -> None:
    rows = [
        {"run": 0, "mean_score": 300.0, "sample_count_score": 0.5},
        {"run": 1, "mean_score": 250.0, "sample_count_score": 1.0},
    ]

    best = _best_finite_row(
        rows,
        "mean_score",
        prefer_high=True,
        sample_count_field="sample_count_score",
    )

    assert best == rows[1]


def test_best_finite_row_rejects_bool_rank_values() -> None:
    rows = [
        {"run": 0, "mean_score": True},
        {"run": 1, "mean_score": 0.5},
    ]

    best = _best_finite_row(rows, "mean_score", prefer_high=True)

    assert best == rows[1]


def test_best_finite_row_rejects_numeric_string_rank_values() -> None:
    rows = [
        {"run": 0, "mean_score": "300.0"},
        {"run": 1, "mean_score": 0.5},
    ]

    best = _best_finite_row(rows, "mean_score", prefer_high=True)

    assert best == rows[1]


def test_best_finite_row_keeps_rows_without_sample_count_field() -> None:
    rows = [
        {"run": 0, "mean_score": 100.0},
        {"run": 1, "mean_score": 150.0, "sample_count_score": 0.0},
    ]

    best = _best_finite_row(
        rows,
        "mean_score",
        prefer_high=True,
        sample_count_field="sample_count_score",
    )

    assert best == rows[0]


def test_best_finite_row_skips_values_below_minimum() -> None:
    rows = [
        {"run": 0, "mean_wall_time_s": -1.0},
        {"run": 1, "mean_wall_time_s": 0.5},
    ]

    best = _best_finite_row(
        rows,
        "mean_wall_time_s",
        prefer_high=False,
        min_value=0.0,
    )

    assert best == rows[1]


def test_benchmark_arg_types_reject_empty_or_invalid_runs() -> None:
    assert positive_int_arg("1") == 1
    assert non_negative_int_arg("0") == 0
    assert positive_float_arg("0.25") == pytest.approx(0.25)

    for parser in (positive_int_arg, positive_float_arg):
        with pytest.raises(argparse.ArgumentTypeError):
            parser("0")
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int_arg("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int_arg("1.5")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float_arg("nan")


def test_benchmark_arg_types_accept_strict_direct_values() -> None:
    assert positive_int_arg(1) == 1
    assert non_negative_int_arg(0) == 0
    assert positive_float_arg(0.25) == pytest.approx(0.25)
    assert device_arg("auto") == "auto"
    assert device_arg("cpu") == "cpu"


@pytest.mark.parametrize(
    ("parser", "raw"),
    [
        (positive_int_arg, 1.0),
        (positive_int_arg, 1.5),
        (positive_int_arg, True),
        (non_negative_int_arg, 1.0),
        (non_negative_int_arg, True),
        (positive_float_arg, True),
    ],
)
def test_benchmark_arg_types_reject_ambiguous_direct_values(
    parser: Callable[[object], object],
    raw: object,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parser(raw)


@pytest.mark.parametrize(
    ("parser", "raw"),
    [
        (positive_int_arg, _FailingIndex()),
        (non_negative_int_arg, _FailingIndex()),
        (positive_float_arg, _FailingFloat()),
    ],
)
def test_benchmark_arg_types_wrap_malformed_numeric_values(
    parser: Callable[[object], object],
    raw: object,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parser(raw)


@pytest.mark.parametrize("raw", ["gpu", "cuda:0", "", True, None])
def test_device_arg_rejects_unsupported_values(raw: object) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        device_arg(raw)


def test_display_formatters_hide_malformed_values() -> None:
    class FailingFloat:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

    assert _format_metric_value(1.234, precision=1, suffix="/s") == "1.2/s"
    assert _format_metric_value(0.00125, precision=2, scale=1e3, suffix="ms") == "1.25ms"

    for raw in (None, "fast", "1.0", True, float("nan"), float("inf"), FailingFloat()):
        assert _format_metric_value(raw, precision=1, suffix="/s") == "n/a"

    assert _format_count(0) == "0"
    for raw in (True, 1.5, -1, "many", "2"):
        assert _format_count(raw) == "n/a"

    assert _has_positive_display_value(2)
    for raw in (0, None, True, "2", float("nan"), float("inf"), FailingFloat()):
        assert not _has_positive_display_value(raw)


def test_format_scheduler_summary_reports_failures_and_last_error() -> None:
    assert _format_scheduler_summary({
        "scheduler_step_failures": 2,
        "scheduler_last_error": " RuntimeError: scheduler boom ",
    }) == "failures=2 error=RuntimeError: scheduler boom"

    assert _format_scheduler_summary({
        "scheduler_step_failures": 1,
        "scheduler_last_error": "",
    }) == "failures=1"


@pytest.mark.parametrize("raw", [0, None, True, "2", -1, 1.5, float("nan")])
def test_format_scheduler_summary_omits_zero_or_malformed_failures(raw: object) -> None:
    assert _format_scheduler_summary({
        "scheduler_step_failures": raw,
        "scheduler_last_error": "RuntimeError: scheduler boom",
    }) == ""


def test_format_profile_breakdown_summary_includes_overtracked_when_positive() -> None:
    profile = {
        "phase_breakdowns": {
            "forward": {
                "tracked_s": 0.07,
                "untracked_s": 0.03,
                "overtracked_s": 0.02,
                "coverage_pct": 70.0,
            },
            "optimizer": {
                "tracked_s": 0.04,
                "untracked_s": 0.01,
                "overtracked_s": 0.0,
                "coverage_pct": 80.0,
            },
        },
    }

    assert _format_profile_breakdown_summary(
        profile,
        "forward",
    ) == "coverage=70.0% tracked=70.00ms untracked=30.00ms overtracked=20.00ms"
    assert _format_profile_breakdown_summary(
        profile,
        "optimizer",
    ) == "coverage=80.0% tracked=40.00ms untracked=10.00ms"
    assert _format_profile_breakdown_summary(profile, "loss") == ""


def test_format_profile_breakdown_summary_omits_malformed_values() -> None:
    profile = {
        "phase_breakdowns": {
            "forward": {
                "tracked_s": -0.07,
                "untracked_s": "0.03",
                "overtracked_s": True,
            },
            "optimizer": {
                "untracked_s": 0.01,
            },
            "loss": {
                "top_children": [{"name": "loss"}],
            },
        },
    }

    assert _format_profile_breakdown_summary(profile, "forward") == ""
    assert _format_profile_breakdown_summary(profile, "optimizer") == "untracked=10.00ms"
    assert _format_profile_breakdown_summary(profile, "loss") == ""


def test_format_profile_breakdown_child_timing_includes_avg_and_tail() -> None:
    formatted = _format_profile_breakdown_child_timing({
        "pct_of_parent": 42.5,
        "avg_ms": 1.25,
        "p95_ms": 2.5,
        "p99_ms": 3.5,
        "std_ms": 0.25,
        "calls": 3,
        "sample_count": 3,
        "window_sample_count": 2,
    })

    assert formatted == "42.5% avg=1.25ms p95=2.50ms p99=3.50ms std=0.25ms calls=3 samples=3 window=2"


def test_format_profile_breakdown_child_timing_omits_malformed_tail() -> None:
    formatted = _format_profile_breakdown_child_timing({
        "pct_of_parent": "slow",
        "avg_ms": -1.0,
        "p95_ms": True,
        "p99_ms": "fast",
        "std_ms": float("nan"),
    })

    assert formatted == "n/a"


def test_format_profile_event_timing_includes_parent_position() -> None:
    assert _format_profile_event_timing({
        "avg_ms": 4.25,
        "avg_pct_of_parent": 42.5,
    }) == "4.2ms@42.5%"
    assert _format_profile_event_timing({"avg_ms": 4.25}) == "4.2ms"
    assert _format_profile_event_timing({
        "avg_ms": "slow",
        "avg_pct_of_parent": 42.5,
    }) == "n/a"


def test_format_profile_event_timing_can_include_tail() -> None:
    assert _format_profile_event_timing({
        "avg_ms": 4.25,
        "avg_pct_of_parent": 42.5,
        "p95_ms": 6.5,
        "p99_ms": 7.5,
        "std_ms": 0.5,
        "calls": 4,
        "sample_count": 4,
        "window_sample_count": 3,
    }, include_p95=True) == "4.2ms@42.5% p95=6.5ms p99=7.5ms std=0.5ms calls=4 samples=4 window=3"
    assert _format_profile_event_timing({
        "avg_ms": 4.25,
        "p95_ms": 6.5,
        "p99_ms": 7.5,
        "std_ms": 0.5,
    }, include_p95=True) == "4.2ms p95=6.5ms p99=7.5ms std=0.5ms"


def test_format_profile_event_group_summary_includes_readiness_span() -> None:
    profile = {
        "phase_events": {
            "backward_grad_ready": {
                "earliest_avg_ms": 1.25,
                "latest_avg_ms": 4.75,
                "span_avg_ms": 3.50,
                "earliest_pct_of_parent": 12.5,
                "latest_pct_of_parent": 47.5,
                "span_pct_of_parent": 35.0,
            },
        },
    }

    assert _format_profile_event_group_summary(
        profile,
        "backward_grad_ready",
    ) == "span=3.50ms@35.0% range=1.25ms-4.75ms range_pct=12.5%-47.5%"


def test_format_profile_event_group_summary_omits_malformed_values() -> None:
    profile = {
        "phase_events": {
            "backward_grad_ready": {
                "earliest_avg_ms": "early",
                "latest_avg_ms": 4.75,
                "span_avg_ms": -1.0,
                "span_pct_of_parent": True,
            },
        },
    }

    assert _format_profile_event_group_summary(profile, "backward_grad_ready") == ""
    assert _format_profile_event_group_summary(profile, "forward") == ""


def test_format_profile_phase_timing_includes_tail_latency() -> None:
    formatted = _format_profile_phase_timing({
        "pct": 42.5,
        "avg_ms": 1.25,
        "p95_ms": 2.5,
        "p99_ms": 3.5,
        "std_ms": 0.25,
        "calls": 3,
        "sample_count": 3,
        "window_sample_count": 2,
    })

    assert formatted == "42.5% avg=1.25ms p95=2.50ms p99=3.50ms std=0.25ms calls=3 samples=3 window=2"


def test_profile_count_fields_omit_malformed_values() -> None:
    assert _profile_count_fields({
        "calls": 3,
        "sample_count": 3,
        "window_sample_count": 2,
    }) == ["calls=3", "samples=3", "window=2"]
    assert _profile_count_fields({
        "calls": True,
        "sample_count": "3",
        "window_sample_count": -1,
    }) == []


def test_format_profile_phase_timing_omits_malformed_tail_latency() -> None:
    formatted = _format_profile_phase_timing({
        "pct": "slow",
        "avg_ms": -1.0,
        "p95_ms": True,
        "p99_ms": "fast",
        "std_ms": float("nan"),
    })

    assert formatted == "n/a"


def test_format_profile_open_timer_summary_reports_open_work() -> None:
    profile = {
        "profile_open_phase_count": 2,
        "profile_open_detail_count": 3,
        "profile_open_phases": ["forward", " backward ", ""],
        "profile_open_details": [
            {"parent": "forward", "name": "model.0", "count": 2},
            {"parent": "optimizer", "name": "step", "count": 1},
        ],
    }

    assert _format_profile_open_timer_summary(profile) == (
        "phases=2:forward,backward details=3:forward.model.0x2,optimizer.step"
    )


def test_format_profile_open_timer_summary_omits_zero_and_malformed_values() -> None:
    assert _format_profile_open_timer_summary({
        "profile_open_phase_count": 0,
        "profile_open_detail_count": 0,
    }) == ""
    assert _format_profile_open_timer_summary({
        "profile_open_phase_count": "1",
        "profile_open_detail_count": True,
        "profile_open_phases": ["forward"],
        "profile_open_details": [
            {"parent": "forward", "name": "model.0", "count": 2},
        ],
    }) == ""
    assert _format_profile_open_timer_summary({
        "profile_open_detail_count": 1,
        "profile_open_details": [
            {"parent": "", "name": "model.0", "count": 2},
            {"parent": "forward", "name": True, "count": 2},
        ],
    }) == "details=1"


def test_format_profile_model_hook_summary_reports_requested_counts() -> None:
    assert _format_profile_model_hook_summary({
        "profile_model_requested": True,
        "profile_model_status": "ok",
        "profile_model_modules_selected": 2,
        "profile_model_hook_count": 7,
        "profile_model_hook_failures": 0,
    }) == "status=ok modules=2 hooks=7 failures=0"


def test_format_profile_model_hook_summary_reports_failures_with_error() -> None:
    assert _format_profile_model_hook_summary({
        "profile_model_requested": True,
        "profile_model_status": "hook_failures",
        "profile_model_modules_selected": 2,
        "profile_model_hook_count": 4,
        "profile_model_hook_failures": 1,
        "profile_model_hook_last_error": "RuntimeError: hook boom",
    }) == "status=hook_failures modules=2 hooks=4 failures=1 error=RuntimeError: hook boom"


def test_format_profile_model_hook_summary_hides_not_requested_and_bad_counts() -> None:
    assert _format_profile_model_hook_summary({
        "profile_model_requested": False,
        "profile_model_status": "not_requested",
        "profile_model_modules_selected": 2,
    }) == ""
    assert _format_profile_model_hook_summary({
        "profile_model_requested": True,
        "profile_model_status": "no_matching_modules",
        "profile_model_modules_selected": -1,
        "profile_model_hook_count": 0.5,
        "profile_model_hook_failures": True,
    }) == "status=no_matching_modules"


def test_format_profile_model_hook_summary_marks_invalid_status() -> None:
    formatted = _format_profile_model_hook_summary({
        "profile_model_requested": True,
        "profile_model_status": "missing",
        "profile_model_modules_selected": 2,
        "profile_model_hook_count": 4,
        "profile_model_hook_failures": 0,
    })

    assert formatted == "status=invalid modules=2 hooks=4 failures=0"
    assert "missing" not in formatted


def test_format_profile_model_hook_summary_marks_malformed_status() -> None:
    formatted = _format_profile_model_hook_summary({
        "profile_model_requested": True,
        "profile_model_status": True,
        "profile_model_modules_selected": 2,
    })

    assert formatted == "status=invalid modules=2"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"name": "forward"}, "forward"),
        ({"name": ""}, "<unnamed>"),
        ({"name": "   "}, "<unnamed>"),
        ({"name": True}, "<unnamed>"),
        ({"name": 1}, "<unnamed>"),
        ({"name": ["forward"]}, "<unnamed>"),
        ({}, "<unnamed>"),
    ],
)
def test_profile_row_name_hides_malformed_names(row: dict[str, object], expected: str) -> None:
    assert _profile_row_name(row) == expected


def test_validate_benchmark_args_rejects_warmup_larger_than_steps() -> None:
    with pytest.raises(ValueError, match="warmup-steps"):
        validate_benchmark_args(Namespace(warmup_steps=3, steps=2))

    validate_benchmark_args(Namespace(warmup_steps=2, steps=2))


@pytest.mark.parametrize(
    ("steps", "warmup_steps", "match"),
    [
        (0, 0, "steps"),
        (1.5, 0, "steps"),
        ("2", 0, "steps"),
        (True, 0, "steps"),
        (2, -1, "warmup_steps"),
        (2, 0.5, "warmup_steps"),
        (2, "1", "warmup_steps"),
        (2, True, "warmup_steps"),
        (2, 0, "device"),
    ],
)
def test_validate_benchmark_args_rejects_invalid_direct_values(
    steps: object,
    warmup_steps: object,
    match: str,
) -> None:
    device = "gpu" if match == "device" else "cpu"
    with pytest.raises(ValueError, match=match):
        validate_benchmark_args(Namespace(warmup_steps=warmup_steps, steps=steps, device=device))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("transactions", 0, "transactions"),
        ("feature_dim", 1.5, "feature_dim"),
        ("num_classes", True, "num_classes"),
        ("batch_size", 0, "batch_size"),
        ("grad_accum", "2", "grad_accum"),
        ("prefetch_factor", -1, "prefetch_factor"),
        ("runs", 0, "runs"),
        ("profile_window", 0, "profile_window"),
        ("profile_model_depth", 0, "profile_model_depth"),
        ("profile_model_max_modules", 0, "profile_model_max_modules"),
        ("workers", -1, "workers"),
        ("log_interval", 1.5, "log_interval"),
        ("seed", True, "seed"),
        ("learning_rate", "0.001", "learning_rate"),
        ("learning_rate", float("nan"), "learning_rate"),
        ("compile", 1, "compile"),
        ("collect_profile", "yes", "collect_profile"),
        ("meter_fast_mode", "yes", "meter_fast_mode"),
        ("profile_sync", 0, "profile_sync"),
        ("profile_distribution", "true", "profile_distribution"),
        ("profile_model", 1, "profile_model"),
        ("profile_model_include", "", "profile_model_include"),
        ("profile_model_include", "   ", "profile_model_include"),
        ("profile_model_include", ",,", "profile_model_include"),
        ("profile_model_include", [], "profile_model_include"),
        ("profile_model_include", ["", "  "], "profile_model_include"),
        ("profile_model_include", 1, "profile_model_include"),
        ("profile_model_include", ["0", 2], "profile_model_include"),
        ("json_out", "", "json_out"),
        ("json_out", "   ", "json_out"),
        ("json_out", True, "json_out"),
        ("json_out", 1, "json_out"),
        ("summary_out", "", "summary_out"),
        ("summary_out", "   ", "summary_out"),
        ("summary_out", True, "summary_out"),
        ("dataset_mode", "cached", "dataset_mode"),
    ],
)
def test_validate_benchmark_args_rejects_invalid_optional_direct_values(
    field: str,
    value: object,
    match: str,
) -> None:
    args = Namespace(warmup_steps=0, steps=1, device="cpu")
    setattr(args, field, value)

    with pytest.raises(ValueError, match=match):
        validate_benchmark_args(args)


@pytest.mark.parametrize("profile_model_include", [None, "0,2", "model.0,model.2", ["0", "2"], ["model.0", "model.2"]])
def test_validate_benchmark_args_accepts_profile_model_include_forms(
    profile_model_include: object,
) -> None:
    validate_benchmark_args(
        Namespace(
            warmup_steps=0,
            steps=1,
            device="cpu",
            profile_model_include=profile_model_include,
        ),
    )


def test_validate_benchmark_args_accepts_output_pathlike_values(tmp_path: Path) -> None:
    validate_benchmark_args(
        Namespace(
            warmup_steps=0,
            steps=1,
            device="cpu",
            json_out=tmp_path / "rows.json",
            summary_out=str(tmp_path / "summary.json"),
        ),
    )


def test_validate_benchmark_args_rejects_malformed_output_pathlike() -> None:
    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError("path failed")

    args = Namespace(warmup_steps=0, steps=1, device="cpu", json_out=FailingPath())

    with pytest.raises(ValueError, match="json_out"):
        validate_benchmark_args(args)


def test_parse_args_rejects_zero_profile_model_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_parallel_transactions.py", "--profile-model-depth", "0"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_rejects_fractional_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_parallel_transactions.py", "--seed", "1.5"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_rejects_invalid_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_parallel_transactions.py", "--device", "gpu"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_accepts_meter_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_parallel_transactions.py", "--meter-fast-mode"])

    args = parse_args()

    assert args.meter_fast_mode is True


def test_main_prints_setup_breakdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = Namespace(
        profile_model=False,
        collect_profile=False,
        runs=1,
        json_out=None,
        summary_out=None,
    )

    def fake_run_once(_args: Namespace, run_index: int) -> BenchmarkResult:
        return BenchmarkResult(
            wall_time_s=0.50,
            trainer_metrics={
                "run": run_index,
                "reported_samples_per_sec": 200.0,
                "samples_per_sec": 180.0,
                "p99_s": 0.010,
                "std_batch_s": 0.002,
                "avg_loss": 0.1234,
                "setup_time_s": 0.27,
                "dataset_setup_time_s": 0.05,
                "loader_setup_time_s": 0.07,
                "model_setup_time_s": 0.13,
                "compile_init_time_s": 0.02,
                "end_to_end_wall_time_s": 0.77,
            },
            run_index=run_index,
        )

    monkeypatch.setattr(bpt, "parse_args", lambda: args)
    monkeypatch.setattr(bpt, "run_once", fake_run_once)

    bpt.main()

    assert (
        "setup=0.27s "
        "init(dataset=0.05s,loader=0.07s,model=0.13s,compile=0.02s) "
        "e2e=0.77s"
    ) in capsys.readouterr().out


def test_main_prints_backward_event_parent_position(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = Namespace(
        profile_model=False,
        collect_profile=True,
        runs=1,
        json_out=None,
        summary_out=None,
    )

    def fake_run_once(_args: Namespace, run_index: int) -> BenchmarkResult:
        return BenchmarkResult(
            wall_time_s=0.50,
            trainer_metrics={
                "run": run_index,
                "reported_samples_per_sec": 200.0,
                "samples_per_sec": 180.0,
                "p99_s": 0.010,
                "std_batch_s": 0.002,
                "avg_loss": 0.1234,
                "setup_time_s": 0.0,
                "end_to_end_wall_time_s": 0.50,
                "profile_forward_pct": 42.5,
                "profile_forward_top_pct_of_parent": 70.0,
                "profile_backward_pct": 30.0,
                "profile_backward_grad_ready_span_pct": 23.0,
                "profile_optimizer_pct": 12.5,
                "profile": {
                    "top_phases": [
                        {
                            "name": "forward",
                            "pct": 42.5,
                            "avg_ms": 1.25,
                            "p95_ms": 2.5,
                            "p99_ms": 3.5,
                            "std_ms": 0.25,
                            "calls": 4,
                            "sample_count": 4,
                            "window_sample_count": 3,
                        },
                    ],
                    "phase_breakdowns": {
                        "forward": {
                            "tracked_s": 0.07,
                            "untracked_s": 0.03,
                            "top_children": [
                                {
                                    "name": "model.0",
                                    "pct_of_parent": 70.0,
                                    "avg_ms": 1.25,
                                    "p95_ms": 2.5,
                                    "calls": 2,
                                    "sample_count": 2,
                                    "window_sample_count": 2,
                                },
                            ],
                        },
                        "optimizer": {
                            "tracked_s": 0.04,
                            "top_children": [
                                {
                                    "name": "optimizer.step",
                                    "pct_of_parent": 80.0,
                                    "avg_ms": 3.25,
                                    "p95_ms": 4.5,
                                    "calls": 1,
                                },
                            ],
                        },
                    },
                    "phase_events": {
                        "backward_grad_ready": {
                            "earliest_avg_ms": 1.2,
                            "latest_avg_ms": 3.5,
                            "span_avg_ms": 2.3,
                            "earliest_pct_of_parent": 12.0,
                            "latest_pct_of_parent": 35.0,
                            "span_pct_of_parent": 23.0,
                            "top_children": [
                                {
                                    "name": "model.0",
                                    "avg_ms": 3.5,
                                    "avg_pct_of_parent": 35.0,
                                    "p95_ms": 4.5,
                                    "calls": 2,
                                    "sample_count": 2,
                                    "window_sample_count": 2,
                                },
                                {"name": "model.2", "avg_ms": 1.2, "calls": 1},
                            ],
                        },
                    },
                },
            },
            run_index=run_index,
        )

    monkeypatch.setattr(bpt, "parse_args", lambda: args)
    monkeypatch.setattr(bpt, "run_once", fake_run_once)

    bpt.main()

    output = capsys.readouterr().out
    assert "phases: forward=42.5% avg=1.25ms p95=2.50ms p99=3.50ms std=0.25ms calls=4 samples=4 window=3" in output
    assert "forward: model.0=70.0% avg=1.25ms p95=2.50ms calls=2 samples=2 window=2" in output
    assert "backward_grad_ready_summary: span=2.30ms@23.0% range=1.20ms-3.50ms range_pct=12.0%-35.0%" in output
    assert "backward_grad_ready: model.0=3.5ms@35.0% p95=4.5ms calls=2 samples=2 window=2, model.2=1.2ms calls=1" in output
    assert "optimizer: optimizer.step=80.0% avg=3.25ms p95=4.50ms calls=1" in output
    assert (
        "  bottleneck: #1 forward_phase=42.5% category=phase_share severity=high "
        "next=inspect forward top-child and tail metrics"
    ) in output
    assert (
        "Bottleneck: #1 forward_phase=42.5% category=phase_share severity=high "
        "next=inspect forward top-child and tail metrics"
    ) in output
    assert (
        "severity_counts(high=3,medium=1,low=1) "
        "categories=#1 phase_share:forward_phase=42.5%[high]/3;sum=85.0%,"
        "#2 child_hotspot:forward_top_child=29.8%[high]/1;sum=29.8%,"
        "#3 readiness_span:backward_readiness_span=6.9%[low]/1;sum=6.9%"
    ) in output


def test_main_writes_run_level_profile_bottleneck_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    json_out = tmp_path / "rows.json"
    args = Namespace(
        profile_model=False,
        collect_profile=True,
        runs=1,
        json_out=json_out,
        summary_out=None,
    )

    def fake_run_once(_args: Namespace, run_index: int) -> BenchmarkResult:
        return BenchmarkResult(
            wall_time_s=0.50,
            trainer_metrics={
                "run": run_index,
                "reported_samples_per_sec": 200.0,
                "samples_per_sec": 180.0,
                "p99_s": 0.010,
                "std_batch_s": 0.002,
                "avg_loss": 0.1234,
                "setup_time_s": 0.0,
                "end_to_end_wall_time_s": 0.50,
                "profile_forward_pct": 55.0,
                "profile_forward_top_pct_of_parent": 50.0,
                "profile_backward_pct": 20.0,
            },
            run_index=run_index,
        )

    monkeypatch.setattr(bpt, "parse_args", lambda: args)
    monkeypatch.setattr(bpt, "run_once", fake_run_once)

    bpt.main()

    rows = json.loads(json_out.read_text())
    assert rows[0]["profile_bottleneck_top_candidate"]["name"] == "forward_phase"
    assert rows[0]["profile_bottleneck_candidates"][1]["name"] == "forward_top_child"
    assert rows[0]["profile_bottleneck_severity_counts"] == {"high": 2, "medium": 1}


def test_format_profile_bottleneck_category_pressure_includes_omitted_count() -> None:
    text = bpt._format_profile_bottleneck_category_pressure(
        "coverage_gap",
        {
            "count": 3,
            "max_score": 12.5,
            "total_score": 20.0,
            "omitted_count": 2,
            "score_unit": "profile_pct",
            "top_candidate": "forward_untracked",
            "top_candidate_returned": False,
            "top_severity": "medium",
            "pressure_rank": 4,
        },
        include_count=True,
    )

    assert text == "#4 coverage_gap:forward_untracked=12.5%[medium]/3;sum=20.0%;omitted=2;top_omitted"


def test_transaction_benchmark_records_run_seed() -> None:
    class Args:
        transactions = 64
        feature_dim = 8
        num_classes = 3
        seed = 100
        dataset_mode = "materialized"
        batch_size = 16
        device = "cpu"
        workers = 0
        prefetch_factor = 2
        learning_rate = 3e-4
        meter_fast_mode = True
        compile = False
        grad_accum = 2
        log_interval = 0
        steps = 2
        collect_profile = False
        profile_sync = False
        profile_distribution = True
        profile_window = 16
        profile_model = False
        profile_model_depth = 1
        profile_model_max_modules = 8
        profile_model_include = None
        warmup_steps = 0

    result = run_once(Args, 3).as_dict()

    assert result["seed"] == 103
    assert result["dataset_mode"] == "materialized"
    assert result["dataset_materialized_bytes"] == (64 * 8 * 4) + (64 * 8)
    assert result["distribution_tracked"] is False
    assert result["window_tracked"] is False


@pytest.mark.parametrize(
    "timer_name",
    [
        "dataset_setup_time_s",
        "loader_setup_time_s",
        "model_setup_time_s",
        "setup_time_s",
        "wall_time_s",
    ],
)
def test_transaction_benchmark_validates_timer_outputs(
    monkeypatch: pytest.MonkeyPatch,
    timer_name: str,
) -> None:
    def validate_timer(value: object, name: str) -> float:
        if name == timer_name:
            raise ValueError(f"{timer_name} boom")
        return float(value)

    class DummyTrainer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def train_one_epoch(self, *_args: object, **_kwargs: object) -> dict[str, float]:
            return {}

    monkeypatch.setattr(bpt, "_non_negative_finite_float_setting", validate_timer, raising=False)
    monkeypatch.setattr(bpt, "FastTrainer", DummyTrainer)
    args = Namespace(
        transactions=8,
        feature_dim=4,
        num_classes=2,
        seed=100,
        dataset_mode="generated",
        batch_size=4,
        device="cpu",
        workers=0,
        prefetch_factor=2,
        learning_rate=3e-4,
        compile=False,
        grad_accum=1,
        log_interval=0,
        steps=1,
        collect_profile=False,
        profile_sync=False,
        profile_distribution=True,
        profile_window=16,
        profile_model=False,
        profile_model_depth=1,
        profile_model_max_modules=8,
        profile_model_include=None,
        warmup_steps=0,
    )

    with pytest.raises(ValueError, match=f"{timer_name} boom"):
        run_once(args, 0)


def test_transaction_benchmark_rejects_invalid_direct_seed() -> None:
    class Args:
        transactions = 64
        feature_dim = 8
        num_classes = 3
        seed = "100"
        dataset_mode = "materialized"
        batch_size = 16
        device = "cpu"
        workers = 0
        prefetch_factor = 2
        learning_rate = 3e-4
        compile = False
        grad_accum = 2
        log_interval = 0
        steps = 2
        collect_profile = False
        profile_sync = False
        profile_distribution = True
        profile_window = 16
        profile_model = False
        profile_model_depth = 1
        profile_model_max_modules = 8
        profile_model_include = None
        warmup_steps = 0

    with pytest.raises(ValueError, match="seed"):
        run_once(Args, 0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("learning_rate", "0.001", "learning_rate"),
        ("dataset_mode", "cached", "dataset_mode"),
        ("compile", 1, "compile"),
    ],
)
def test_transaction_benchmark_rejects_invalid_direct_config(
    field: str,
    value: object,
    match: str,
) -> None:
    args = Namespace(
        transactions=64,
        feature_dim=8,
        num_classes=3,
        seed=100,
        dataset_mode="materialized",
        batch_size=16,
        device="cpu",
        workers=0,
        prefetch_factor=2,
        learning_rate=3e-4,
        compile=False,
        grad_accum=2,
        log_interval=0,
        steps=2,
        collect_profile=False,
        profile_sync=False,
        profile_distribution=True,
        profile_window=16,
        profile_model=False,
        profile_model_depth=1,
        profile_model_max_modules=8,
        profile_model_include=None,
        warmup_steps=0,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=match):
        run_once(args, 0)
