from __future__ import annotations

import argparse
import io
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_parallel_transactions import (
    SyntheticTransactionDataset,
    _best_finite_row,
    _format_count,
    _format_metric_value,
    _has_positive_display_value,
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
from scripts.json_utils import dump_json


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
            "profile_forward_backward_pct": 40.0,
            "profile_forward_backward_time_s": 0.20,
            "profile_forward_pct": 15.0,
            "profile_loss_pct": 5.0,
            "profile_loss_reduce_pct": 2.0,
            "profile_backward_pct": 25.0,
            "profile_optimizer_pct": 10.0,
            "profile_metrics_pct": 1.0,
        },
        {
            "run": 1,
            "seed": 11,
            "dataset_mode": "generated",
            "reported_samples_per_sec": 200.0,
            "samples_per_sec": 160.0,
            "steady_samples_per_sec": 200.0,
            "wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "end_to_end_wall_time_s": 1.25,
            "profile_forward_backward_pct": 60.0,
            "profile_forward_backward_time_s": 0.30,
            "profile_forward_pct": 20.0,
            "profile_loss_pct": 7.0,
            "profile_loss_reduce_pct": 3.0,
            "profile_backward_pct": 40.0,
            "profile_optimizer_pct": 15.0,
            "profile_metrics_pct": 2.0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["runs"] == 2
    assert summary["mean_reported_samples_per_sec"] == pytest.approx(155.0)
    assert summary["min_end_to_end_wall_time_s"] == pytest.approx(1.25)
    assert summary["max_end_to_end_wall_time_s"] == pytest.approx(2.0)
    assert summary["stddev_wall_time_s"] == pytest.approx(0.5)
    assert summary["mean_profile_forward_backward_pct"] == pytest.approx(50.0)
    assert summary["mean_profile_loss_pct"] == pytest.approx(6.0)
    assert summary["mean_profile_loss_reduce_pct"] == pytest.approx(2.5)
    assert summary["mean_profile_metrics_pct"] == pytest.approx(1.5)
    assert summary["max_profile_backward_pct"] == pytest.approx(40.0)
    assert summary["profiled_runs"] == 2
    assert summary["best_reported"]["run"] == 1
    assert summary["best_reported"]["profile_forward_backward_pct"] == pytest.approx(60.0)
    assert summary["best_reported"]["profile_loss_pct"] == pytest.approx(7.0)
    assert summary["best_end_to_end"]["run"] == 1


def test_summarize_results_handles_empty_input() -> None:
    summary = summarize_results([])

    assert summary["runs"] == 0
    assert summary["best_reported"] is None
    assert summary["best_end_to_end"] is None
    assert summary["mean_wall_time_s"] == pytest.approx(0.0)
    assert summary["stddev_wall_time_s"] == pytest.approx(0.0)
    assert "mean_profile_forward_backward_pct" not in summary


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
    assert "profile_forward_backward_pct" not in summary["best_reported"]
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
    assert "profile_forward_backward_pct" not in summary["best_reported"]
    assert "profile_backward_pct" not in summary["best_reported"]
    json.dumps(summary, allow_nan=False)


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


@pytest.mark.parametrize("raw", ["gpu", "cuda:0", "", True, None])
def test_device_arg_rejects_unsupported_values(raw: object) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        device_arg(raw)


def test_display_formatters_hide_malformed_values() -> None:
    assert _format_metric_value(1.234, precision=1, suffix="/s") == "1.2/s"
    assert _format_metric_value(0.00125, precision=2, scale=1e3, suffix="ms") == "1.25ms"

    for raw in (None, "fast", True, float("nan"), float("inf")):
        assert _format_metric_value(raw, precision=1, suffix="/s") == "n/a"

    assert _format_count("2") == "2"
    assert _format_count(0) == "0"
    for raw in (True, 1.5, -1, "many"):
        assert _format_count(raw) == "n/a"

    assert _has_positive_display_value("2")
    for raw in (0, None, True, float("nan"), float("inf")):
        assert not _has_positive_display_value(raw)


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
        ("profile_sync", 0, "profile_sync"),
        ("profile_distribution", "true", "profile_distribution"),
        ("profile_model", 1, "profile_model"),
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
