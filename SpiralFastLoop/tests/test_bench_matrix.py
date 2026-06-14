from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_matrix import (
    _compile_requested,
    _format_run_row,
    _format_summary_row,
    _measured_summary_value,
    _parse_csv_choices,
    _parse_worker_counts,
    parse_args,
    summarize_rows,
)


def test_parse_csv_choices_trims_and_validates_values() -> None:
    assert _parse_csv_choices(" generated,materialized ", {"generated", "materialized"}, name="modes") == [
        "generated",
        "materialized",
    ]
    assert _parse_csv_choices("generated,,", {"generated"}, name="modes") == ["generated"]

    with pytest.raises(ValueError):
        _parse_csv_choices("", {"generated"}, name="modes")
    with pytest.raises(ValueError):
        _parse_csv_choices("generated,other", {"generated"}, name="modes")
    with pytest.raises(ValueError, match="comma-separated string"):
        _parse_csv_choices(True, {"generated"}, name="modes")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_csv_choices("generated,generated", {"generated"}, name="modes")


def test_parse_worker_counts_rejects_empty_or_negative_values() -> None:
    assert _parse_worker_counts("0, 2") == [0, 2]
    assert _parse_worker_counts("0,,") == [0]

    with pytest.raises(ValueError):
        _parse_worker_counts("")
    with pytest.raises(ValueError):
        _parse_worker_counts("-1")
    with pytest.raises(ValueError):
        _parse_worker_counts("cpu")
    with pytest.raises(ValueError, match="comma-separated string"):
        _parse_worker_counts(None)
    with pytest.raises(ValueError, match="duplicate"):
        _parse_worker_counts("0,0")


def test_compile_requested_maps_modes() -> None:
    assert _compile_requested("compile") is True
    assert _compile_requested("no-compile") is False

    with pytest.raises(ValueError):
        _compile_requested("sometimes")


def test_parse_args_rejects_zero_profile_model_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_matrix.py", "--profile-model-depth", "0"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_rejects_fractional_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_matrix.py", "--seed", "1.5"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_rejects_invalid_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_matrix.py", "--device", "gpu"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_format_summary_row_includes_profile_suffix_when_available() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_forward_backward_pct": 62.5,
        "mean_profile_loss_pct": 8.5,
        "mean_profile_optimizer_pct": 12.0,
    }

    formatted = _format_summary_row(row)

    assert "fwd+bwd=62.5%" in formatted
    assert "loss=8.5%" in formatted
    assert "opt=12.0%" in formatted


def test_format_summary_row_omits_profile_suffix_when_absent() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
    }

    formatted = _format_summary_row(row)

    assert "fwd+bwd" not in formatted
    assert "loss=" not in formatted
    assert "opt=" not in formatted


def test_format_summary_row_marks_unmeasured_base_fields() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 0.0,
        "sample_count_reported_samples_per_sec": 0.0,
        "non_finite_count_reported_samples_per_sec": 1.0,
        "mean_end_to_end_wall_time_s": 0.0,
        "sample_count_end_to_end_wall_time_s": 0.0,
        "non_finite_count_end_to_end_wall_time_s": 1.0,
    }

    formatted = _format_summary_row(row)

    assert "reported=n/a" in formatted
    assert "e2e=n/a" in formatted
    assert "reported=0.0/s" not in formatted
    assert "e2e=0.00s" not in formatted


def test_measured_summary_value_rejects_bool_and_string_values() -> None:
    assert _measured_summary_value(
        {"mean_reported_samples_per_sec": True},
        "mean_reported_samples_per_sec",
    ) is None
    assert _measured_summary_value(
        {
            "mean_reported_samples_per_sec": 100.0,
            "sample_count_reported_samples_per_sec": True,
        },
        "mean_reported_samples_per_sec",
    ) is None
    assert _measured_summary_value(
        {"mean_reported_samples_per_sec": "100.0"},
        "mean_reported_samples_per_sec",
    ) is None
    assert _measured_summary_value(
        {
            "mean_reported_samples_per_sec": 100.0,
            "sample_count_reported_samples_per_sec": "1",
        },
        "mean_reported_samples_per_sec",
    ) is None


def test_format_summary_row_keeps_zero_when_it_was_measured() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 0.0,
        "sample_count_reported_samples_per_sec": 1.0,
        "mean_end_to_end_wall_time_s": 0.0,
        "sample_count_end_to_end_wall_time_s": 1.0,
    }

    formatted = _format_summary_row(row)

    assert "reported=0.0/s" in formatted
    assert "e2e=0.00s" in formatted


def test_format_summary_row_omits_unmeasured_profile_suffix() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_forward_backward_pct": 0.0,
        "sample_count_profile_forward_backward_pct": 0.0,
        "non_finite_count_profile_forward_backward_pct": 1.0,
        "mean_profile_loss_pct": 0.0,
        "sample_count_profile_loss_pct": 0.0,
        "non_finite_count_profile_loss_pct": 1.0,
    }

    formatted = _format_summary_row(row)

    assert "fwd+bwd" not in formatted
    assert "loss=" not in formatted
    assert "opt=" not in formatted


def test_format_summary_row_includes_only_measured_profile_parts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_forward_backward_pct": 62.5,
        "sample_count_profile_forward_backward_pct": 2.0,
        "mean_profile_loss_pct": 0.0,
        "sample_count_profile_loss_pct": 0.0,
        "mean_profile_optimizer_pct": 12.0,
    }

    formatted = _format_summary_row(row)

    assert "fwd+bwd=62.5%" in formatted
    assert "loss=" not in formatted
    assert "opt=12.0%" in formatted


def test_format_run_row_uses_reported_and_e2e_metrics() -> None:
    row = _format_run_row(
        "generated",
        "no-compile",
        2,
        3,
        {
            "reported_samples_per_sec": 123.45,
            "samples_per_sec": 99.0,
            "end_to_end_wall_time_s": 1.234,
            "wall_time_s": 0.5,
        },
    )

    assert "generated no-compile workers=2" in row
    assert "run=3" in row
    assert "steady=123.5/s" in row
    assert "e2e=1.23s" in row


def test_format_run_row_falls_back_to_total_metrics() -> None:
    row = _format_run_row(
        "materialized",
        "compile",
        0,
        0,
        {
            "samples_per_sec": 80.0,
            "wall_time_s": 2.0,
        },
    )

    assert "steady=80.0/s" in row
    assert "e2e=2.00s" in row


def test_format_run_row_marks_malformed_metrics_as_na() -> None:
    row = _format_run_row(
        "generated",
        "no-compile",
        0,
        0,
        {
            "reported_samples_per_sec": float("nan"),
            "samples_per_sec": True,
            "end_to_end_wall_time_s": None,
            "wall_time_s": "slow",
        },
    )

    assert "steady=n/a" in row
    assert "e2e=n/a" in row


def test_summarize_rows_groups_configs_and_ranks_best() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "p99_s": 0.020,
            "std_batch_s": 0.002,
            "best_samples_per_sec": 140.0,
            "headroom_ratio": 1.4,
            "end_to_end_wall_time_s": 3.0,
            "setup_time_s": 1.0,
            "wall_time_s": 2.0,
            "cold_start_time_s": 0.5,
            "steps": 3,
            "samples": 12,
            "optimizer_steps": 2,
            "grad_accum": 2,
            "partial_optimizer_steps": 1,
            "grad_accum_tail_steps": 1,
            "warmup_steps": 1,
            "warmup_samples": 4,
            "warmup_optimizer_steps": 0,
            "warmup_samples_per_sec": 70.0,
            "warmup_total_time_s": 0.06,
            "warmup_p99_s": 0.06,
            "cold_start_steps": 1,
            "cold_start_samples_per_sec": 70.0,
            "steady_steps": 2,
            "steady_samples": 8,
            "steady_optimizer_steps": 2,
            "steady_total_time_s": 0.09,
            "steady_p99_s": 0.05,
            "dataset_materialized_bytes": 0,
            "profile_flat_metric_invalid_count": 2.0,
            "profile_forward_backward_pct": 40.0,
            "profile_forward_pct": 15.0,
            "profile_loss_pct": 4.0,
            "profile_loss_reduce_pct": 1.0,
            "profile_backward_pct": 25.0,
            "profile_user_metrics_pct": 2.0,
            "profile_postprocess_pct": 4.0,
            "profile_collect_output_pct": 1.0,
            "profile_metrics_pct": 0.5,
        },
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 240.0,
            "steady_samples_per_sec": 300.0,
            "p99_s": 0.010,
            "std_batch_s": 0.001,
            "best_samples_per_sec": 360.0,
            "headroom_ratio": 1.2,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "cold_start_time_s": 0.1,
            "steps": 3,
            "samples": 12,
            "optimizer_steps": 2,
            "grad_accum": 2,
            "partial_optimizer_steps": 1,
            "grad_accum_tail_steps": 1,
            "warmup_steps": 1,
            "warmup_samples": 4,
            "warmup_optimizer_steps": 0,
            "warmup_samples_per_sec": 90.0,
            "warmup_total_time_s": 0.04,
            "warmup_p99_s": 0.04,
            "cold_start_steps": 1,
            "cold_start_samples_per_sec": 90.0,
            "steady_steps": 2,
            "steady_samples": 8,
            "steady_optimizer_steps": 2,
            "steady_total_time_s": 0.06,
            "steady_p99_s": 0.03,
            "dataset_materialized_bytes": 0,
            "profile_flat_metric_invalid_count": 0.0,
            "profile_forward_backward_pct": 60.0,
            "profile_forward_pct": 20.0,
            "profile_loss_pct": 8.0,
            "profile_loss_reduce_pct": 3.0,
            "profile_backward_pct": 40.0,
            "profile_user_metrics_pct": 6.0,
            "profile_postprocess_pct": 8.0,
            "profile_collect_output_pct": 3.0,
            "profile_metrics_pct": 1.5,
            "cuda_current_mem_bytes": 1024,
            "cuda_max_mem_bytes": 2048,
        },
        {
            "matrix_dataset_mode": "materialized",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 250.0,
            "samples_per_sec": 180.0,
            "steady_samples_per_sec": 250.0,
            "p99_s": 0.005,
            "std_batch_s": 0.0005,
            "best_samples_per_sec": 320.0,
            "headroom_ratio": 1.28,
            "end_to_end_wall_time_s": 0.8,
            "setup_time_s": 0.2,
            "wall_time_s": 0.6,
            "cold_start_time_s": 0.05,
            "steps": 3,
            "samples": 12,
            "optimizer_steps": 2,
            "grad_accum": 2,
            "partial_optimizer_steps": 1,
            "grad_accum_tail_steps": 1,
            "warmup_steps": 1,
            "warmup_samples": 4,
            "warmup_optimizer_steps": 0,
            "warmup_samples_per_sec": 100.0,
            "warmup_total_time_s": 0.03,
            "warmup_p99_s": 0.03,
            "cold_start_steps": 1,
            "cold_start_samples_per_sec": 100.0,
            "steady_steps": 2,
            "steady_samples": 8,
            "steady_optimizer_steps": 2,
            "steady_total_time_s": 0.04,
            "steady_p99_s": 0.02,
            "dataset_materialized_bytes": 1024,
            "profile_flat_metric_invalid_count": 1.0,
            "profile_forward_backward_pct": 55.0,
            "profile_forward_pct": 25.0,
            "profile_loss_pct": 6.0,
            "profile_loss_reduce_pct": 2.0,
            "profile_backward_pct": 30.0,
            "profile_metrics_pct": 1.0,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["runs"] == 3
    assert summary["config_count"] == 2
    assert summary["best_reported"]["dataset_mode"] == "materialized"
    assert summary["best_end_to_end"]["dataset_mode"] == "materialized"
    generated = summary["groups"][0]
    assert generated["dataset_mode"] == "generated"
    assert generated["runs"] == 2
    assert generated["mean_reported_samples_per_sec"] == pytest.approx(200.0)
    assert generated["min_reported_samples_per_sec"] == pytest.approx(100.0)
    assert generated["max_reported_samples_per_sec"] == pytest.approx(300.0)
    assert generated["stddev_reported_samples_per_sec"] == pytest.approx(100.0)
    assert generated["mean_end_to_end_wall_time_s"] == pytest.approx(2.0)
    assert generated["stddev_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert generated["mean_p99_s"] == pytest.approx(0.015)
    assert generated["mean_std_batch_s"] == pytest.approx(0.0015)
    assert generated["mean_best_samples_per_sec"] == pytest.approx(250.0)
    assert generated["mean_headroom_ratio"] == pytest.approx(1.3)
    assert generated["mean_steps"] == pytest.approx(3.0)
    assert generated["mean_samples"] == pytest.approx(12.0)
    assert generated["mean_optimizer_steps"] == pytest.approx(2.0)
    assert generated["mean_grad_accum"] == pytest.approx(2.0)
    assert generated["mean_partial_optimizer_steps"] == pytest.approx(1.0)
    assert generated["mean_grad_accum_tail_steps"] == pytest.approx(1.0)
    assert generated["mean_warmup_steps"] == pytest.approx(1.0)
    assert generated["mean_warmup_samples_per_sec"] == pytest.approx(80.0)
    assert generated["mean_cold_start_samples_per_sec"] == pytest.approx(80.0)
    assert generated["mean_steady_steps"] == pytest.approx(2.0)
    assert generated["mean_steady_p99_s"] == pytest.approx(0.04)
    assert generated["mean_profile_flat_metric_invalid_count"] == pytest.approx(1.0)
    assert generated["max_profile_flat_metric_invalid_count"] == pytest.approx(2.0)
    assert generated["mean_profile_forward_backward_pct"] == pytest.approx(50.0)
    assert generated["mean_profile_loss_pct"] == pytest.approx(6.0)
    assert generated["mean_profile_loss_reduce_pct"] == pytest.approx(2.0)
    assert generated["mean_profile_user_metrics_pct"] == pytest.approx(4.0)
    assert generated["mean_profile_postprocess_pct"] == pytest.approx(6.0)
    assert generated["mean_profile_collect_output_pct"] == pytest.approx(2.0)
    assert generated["mean_profile_metrics_pct"] == pytest.approx(1.0)
    assert generated["max_profile_backward_pct"] == pytest.approx(40.0)
    assert generated["mean_cuda_current_mem_bytes"] == pytest.approx(1024.0)
    assert generated["sample_count_cuda_current_mem_bytes"] == pytest.approx(1.0)
    assert generated["mean_cuda_max_mem_bytes"] == pytest.approx(2048.0)
    assert generated["profiled_runs"] == 2
    assert summary["best_reported"]["mean_profile_flat_metric_invalid_count"] == pytest.approx(1.0)
    assert summary["best_reported"]["mean_profile_forward_backward_pct"] == pytest.approx(55.0)
    assert summary["best_reported"]["mean_profile_loss_pct"] == pytest.approx(6.0)
    assert summary["best_reported"]["mean_steps"] == pytest.approx(3.0)
    assert summary["best_reported"]["mean_steady_p99_s"] == pytest.approx(0.02)


def test_summarize_rows_skips_profile_fields_when_absent() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "dataset_materialized_bytes": 0,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["mean_reported_samples_per_sec"] == pytest.approx(100.0)
    assert "mean_profile_forward_backward_pct" not in group
    assert "mean_profile_forward_backward_pct" not in summary["best_reported"]
    assert "profiled_runs" not in group


def test_summarize_rows_ignores_missing_rows_for_profile_aggregates() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "dataset_materialized_bytes": 0,
        },
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "end_to_end_wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "wall_time_s": 0.70,
            "dataset_materialized_bytes": 0,
            "profile_forward_backward_pct": 60.0,
            "profile_backward_pct": 35.0,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["runs"] == 2
    assert group["profiled_runs"] == 1
    assert group["mean_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert group["min_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert group["stddev_profile_forward_backward_pct"] == pytest.approx(0.0)
    assert group["sample_count_profile_forward_backward_pct"] == pytest.approx(1.0)


def test_summarize_rows_skips_non_finite_profile_values() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "dataset_materialized_bytes": 0,
            "profile_forward_backward_pct": float("inf"),
        },
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "end_to_end_wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "wall_time_s": 0.70,
            "dataset_materialized_bytes": 0,
            "profile_forward_backward_pct": 60.0,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["profiled_runs"] == 2
    assert group["mean_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert group["sample_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    assert group["non_finite_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)


def test_summarize_rows_skips_groups_with_no_finite_best_rank_values() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": float("nan"),
            "samples_per_sec": float("nan"),
            "steady_samples_per_sec": float("nan"),
            "end_to_end_wall_time_s": float("inf"),
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "dataset_materialized_bytes": 0,
        },
        {
            "matrix_dataset_mode": "materialized",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 120.0,
            "samples_per_sec": 95.0,
            "steady_samples_per_sec": 120.0,
            "end_to_end_wall_time_s": 0.9,
            "setup_time_s": 0.20,
            "wall_time_s": 0.70,
            "dataset_materialized_bytes": 1024,
        },
    ]

    summary = summarize_rows(rows)
    generated = summary["groups"][0]

    assert generated["sample_count_reported_samples_per_sec"] == pytest.approx(0.0)
    assert generated["non_finite_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert generated["sample_count_end_to_end_wall_time_s"] == pytest.approx(0.0)
    assert generated["non_finite_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["best_reported"]["dataset_mode"] == "materialized"
    assert summary["best_end_to_end"]["dataset_mode"] == "materialized"
    json.dumps(summary, allow_nan=False)


def test_summarize_rows_accepts_string_integer_metadata() -> None:
    rows = [
        {
            "matrix_dataset_mode": "materialized",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": "2",
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "dataset_materialized_bytes": "1024",
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["workers"] == 2
    assert group["dataset_materialized_bytes"] == 1024


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("matrix_workers", 1.5, "matrix_workers"),
        ("matrix_workers", True, "matrix_workers"),
        ("matrix_workers", -1, "matrix_workers"),
        ("dataset_materialized_bytes", 1.5, "dataset_materialized_bytes"),
        ("dataset_materialized_bytes", True, "dataset_materialized_bytes"),
        ("dataset_materialized_bytes", -1, "dataset_materialized_bytes"),
    ],
)
def test_summarize_rows_rejects_ambiguous_integer_metadata(
    field: str,
    value: object,
    match: str,
) -> None:
    row = {
        "matrix_dataset_mode": "generated",
        "matrix_compile_mode": "no-compile",
        "matrix_workers": 0,
        "reported_samples_per_sec": 100.0,
        "samples_per_sec": 80.0,
        "steady_samples_per_sec": 100.0,
        "end_to_end_wall_time_s": 1.0,
        "setup_time_s": 0.25,
        "wall_time_s": 0.75,
        "dataset_materialized_bytes": 0,
    }
    row[field] = value

    with pytest.raises(ValueError, match=match):
        summarize_rows([row])


def test_summarize_rows_handles_empty_input() -> None:
    summary = summarize_rows([])

    assert summary["runs"] == 0
    assert summary["config_count"] == 0
    assert summary["groups"] == []
    assert summary["best_reported"] is None
    assert summary["best_end_to_end"] is None
