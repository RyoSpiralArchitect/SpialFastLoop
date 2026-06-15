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


@pytest.mark.parametrize(
    "argv",
    [
        ["bench_matrix.py", "--dataset-modes", "generated,cached"],
        ["bench_matrix.py", "--compile-modes", "compile,maybe"],
        ["bench_matrix.py", "--worker-counts", "0,-1"],
    ],
)
def test_parse_args_rejects_invalid_matrix_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

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


def test_format_summary_row_includes_setup_breakdown_when_available() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_setup_time_s": 0.25,
        "mean_dataset_setup_time_s": 0.05,
        "mean_loader_setup_time_s": 0.07,
        "mean_model_setup_time_s": 0.13,
        "mean_compile_init_time_s": 0.02,
    }

    formatted = _format_summary_row(row)

    assert "setup=0.25s" in formatted
    assert "init(dataset=0.05s,loader=0.07s,model=0.13s,compile=0.02s)" in formatted


def test_format_summary_row_includes_open_timer_counts_when_positive() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_open_phase_count": 0.5,
        "sample_count_profile_open_phase_count": 2.0,
        "mean_profile_open_detail_count": 2.0,
        "sample_count_profile_open_detail_count": 2.0,
    }

    formatted = _format_summary_row(row)

    assert "open(phases=0.5,details=2.0)" in formatted


def test_format_summary_row_includes_profile_model_hook_counts_when_positive() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_model_modules_selected": 2.0,
        "sample_count_profile_model_modules_selected": 2.0,
        "mean_profile_model_hook_count": 4.0,
        "sample_count_profile_model_hook_count": 2.0,
        "mean_profile_model_hook_failures": 0.5,
        "sample_count_profile_model_hook_failures": 2.0,
    }

    formatted = _format_summary_row(row)

    assert "model(modules=2.0,hooks=4.0,failures=0.5)" in formatted


def test_format_summary_row_includes_profile_model_status_counts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "profile_model_status_counts": {
            "hook_failures": 1,
            "ok": 2,
        },
        "profile_model_status_invalid_count": 1,
    }

    formatted = _format_summary_row(row)

    assert "status(hook_failures=1,ok=2,invalid=1)" in formatted


def test_format_summary_row_includes_backward_ready_position() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_backward_grad_ready_top_pct": 42.5,
        "sample_count_profile_backward_grad_ready_top_pct": 2.0,
        "mean_profile_backward_grad_ready_top_avg_ms": 3.25,
        "sample_count_profile_backward_grad_ready_top_avg_ms": 2.0,
    }

    formatted = _format_summary_row(row)

    assert "bwd_ready=42.5%@3.25ms" in formatted


def test_format_summary_row_includes_scheduler_failures_when_positive() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_scheduler_step_failures": 1.5,
        "sample_count_scheduler_step_failures": 2.0,
    }

    formatted = _format_summary_row(row)

    assert "scheduler(failures=1.5)" in formatted


def test_format_summary_row_omits_zero_scheduler_failures() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_scheduler_step_failures": 0.0,
        "sample_count_scheduler_step_failures": 2.0,
    }

    formatted = _format_summary_row(row)

    assert "scheduler(" not in formatted


def test_format_summary_row_omits_not_requested_profile_model_status_counts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "profile_model_status_counts": {
            "not_requested": 2,
        },
    }

    formatted = _format_summary_row(row)

    assert "status(" not in formatted


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


def test_format_summary_row_omits_zero_or_unmeasured_open_timer_counts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_open_phase_count": 0.0,
        "sample_count_profile_open_phase_count": 2.0,
        "mean_profile_open_detail_count": 3.0,
        "sample_count_profile_open_detail_count": 0.0,
    }

    formatted = _format_summary_row(row)

    assert "open(" not in formatted


def test_format_summary_row_omits_zero_or_unmeasured_profile_model_counts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_model_modules_selected": 0.0,
        "sample_count_profile_model_modules_selected": 2.0,
        "mean_profile_model_hook_count": 4.0,
        "sample_count_profile_model_hook_count": 0.0,
        "mean_profile_model_hook_failures": 0.0,
        "sample_count_profile_model_hook_failures": 2.0,
    }

    formatted = _format_summary_row(row)

    assert "model(" not in formatted


def test_format_summary_row_omits_malformed_profile_model_status_counts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "profile_model_status_counts": {
            "ok": "2",
            "hook_failures": True,
        },
        "profile_model_status_invalid_count": "1",
    }

    formatted = _format_summary_row(row)

    assert "status(" not in formatted


def test_format_run_row_marks_invalid_profile_model_status() -> None:
    formatted = _format_run_row(
        "generated",
        "no-compile",
        0,
        0,
        {
            "reported_samples_per_sec": 200.0,
            "end_to_end_wall_time_s": 1.25,
            "profile_model_requested": True,
            "profile_model_status": "missing",
            "profile_model_modules_selected": 2,
            "profile_model_hook_count": 4,
            "profile_model_hook_failures": 0,
        },
    )

    assert "profile_model(status=invalid modules=2 hooks=4 failures=0)" in formatted
    assert "missing" not in formatted


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


def test_measured_summary_value_rejects_negative_values() -> None:
    assert _measured_summary_value(
        {"mean_reported_samples_per_sec": -1.0},
        "mean_reported_samples_per_sec",
    ) is None


def test_measured_summary_value_rejects_fractional_sample_count() -> None:
    assert _measured_summary_value(
        {
            "mean_reported_samples_per_sec": 100.0,
            "sample_count_reported_samples_per_sec": 0.5,
        },
        "mean_reported_samples_per_sec",
    ) is None


def test_measured_summary_value_rejects_out_of_range_percentages() -> None:
    assert _measured_summary_value(
        {"mean_profile_forward_backward_pct": 125.0},
        "mean_profile_forward_backward_pct",
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


def test_format_summary_row_omits_out_of_range_profile_parts() -> None:
    row = {
        "dataset_mode": "generated",
        "compile_mode": "no-compile",
        "workers": 0,
        "mean_reported_samples_per_sec": 200.0,
        "mean_end_to_end_wall_time_s": 1.25,
        "mean_profile_forward_backward_pct": 125.0,
        "mean_profile_loss_pct": 8.0,
    }

    formatted = _format_summary_row(row)

    assert "fwd+bwd" not in formatted
    assert "loss=8.0%" in formatted


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
    assert "profile_model(" not in row


def test_format_run_row_includes_setup_breakdown_when_available() -> None:
    row = _format_run_row(
        "generated",
        "no-compile",
        0,
        0,
        {
            "reported_samples_per_sec": 123.45,
            "end_to_end_wall_time_s": 1.234,
            "setup_time_s": 0.27,
            "dataset_setup_time_s": 0.05,
            "loader_setup_time_s": 0.07,
            "model_setup_time_s": 0.13,
            "compile_init_time_s": 0.02,
        },
    )

    assert "setup=0.27s" in row
    assert "init(dataset=0.05s,loader=0.07s,model=0.13s,compile=0.02s)" in row
    assert "init(" in row.split("e2e=")[0]


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


def test_format_run_row_includes_profile_model_hook_summary_when_requested() -> None:
    row = _format_run_row(
        "generated",
        "no-compile",
        0,
        0,
        {
            "reported_samples_per_sec": 123.45,
            "end_to_end_wall_time_s": 1.234,
            "profile_model_requested": True,
            "profile_model_status": "no_matching_modules",
            "profile_model_modules_selected": 0,
            "profile_model_hook_count": 0,
            "profile_model_hook_failures": 0,
        },
    )

    assert "profile_model(status=no_matching_modules modules=0 hooks=0 failures=0)" in row


def test_format_run_row_includes_scheduler_failure_summary() -> None:
    row = _format_run_row(
        "generated",
        "no-compile",
        0,
        0,
        {
            "reported_samples_per_sec": 123.45,
            "end_to_end_wall_time_s": 1.234,
            "scheduler_step_failures": 2,
            "scheduler_last_error": "RuntimeError: scheduler boom",
        },
    )

    assert "scheduler(failures=2 error=RuntimeError: scheduler boom)" in row


def test_format_run_row_marks_malformed_metrics_as_na() -> None:
    row = _format_run_row(
        "generated",
        "no-compile",
        0,
        0,
        {
            "reported_samples_per_sec": "123.4",
            "samples_per_sec": True,
            "end_to_end_wall_time_s": "1.0",
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
            "transactions": 64,
            "batch_size": 4,
            "num_workers": 0,
            "world_size": 1,
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "p99_s": 0.020,
            "std_batch_s": 0.002,
            "avg_batch_s": 0.006,
            "last_batch_s": 0.007,
            "min_batch_s": 0.004,
            "max_batch_s": 0.009,
            "batches": 3,
            "best_samples_per_sec": 140.0,
            "headroom_ratio": 1.4,
            "ema_samples_per_sec": 110.0,
            "window_samples_per_sec": 105.0,
            "window_time_s": 0.080,
            "window_batches": 2,
            "window_samples": 8,
            "end_to_end_wall_time_s": 3.0,
            "setup_time_s": 1.0,
            "dataset_setup_time_s": 0.20,
            "loader_setup_time_s": 0.30,
            "model_setup_time_s": 0.50,
            "compile_init_time_s": 0.04,
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
            "profile_open_phase_count": 2,
            "profile_open_detail_count": 3,
            "profile_model_status": "hook_failures",
            "profile_model_modules_selected": 1,
            "profile_model_hook_count": 3,
            "profile_model_hook_failures": 1,
            "profile_forward_backward_pct": 40.0,
            "profile_forward_pct": 15.0,
            "profile_loss_pct": 4.0,
            "profile_loss_reduce_pct": 1.0,
            "profile_backward_pct": 25.0,
            "profile_backward_grad_ready_child_count": 1,
            "profile_backward_grad_ready_parent_avg_ms": 10.0,
            "profile_backward_grad_ready_top_avg_ms": 3.0,
            "profile_backward_grad_ready_top_pct": 30.0,
            "profile_backward_grad_ready_top_calls": 1,
            "profile_user_metrics_pct": 2.0,
            "profile_postprocess_pct": 4.0,
            "profile_collect_output_pct": 1.0,
            "profile_metrics_pct": 0.5,
        },
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "transactions": 64,
            "batch_size": 4,
            "num_workers": 0,
            "world_size": 1,
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 240.0,
            "steady_samples_per_sec": 300.0,
            "p99_s": 0.010,
            "std_batch_s": 0.001,
            "avg_batch_s": 0.004,
            "last_batch_s": 0.003,
            "min_batch_s": 0.002,
            "max_batch_s": 0.006,
            "batches": 3,
            "best_samples_per_sec": 360.0,
            "headroom_ratio": 1.2,
            "ema_samples_per_sec": 320.0,
            "window_samples_per_sec": 310.0,
            "window_time_s": 0.040,
            "window_batches": 2,
            "window_samples": 8,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "dataset_setup_time_s": 0.05,
            "loader_setup_time_s": 0.07,
            "model_setup_time_s": 0.13,
            "compile_init_time_s": 0.02,
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
            "profile_open_phase_count": 0,
            "profile_open_detail_count": 0,
            "profile_model_status": "ok",
            "profile_model_modules_selected": 2,
            "profile_model_hook_count": 4,
            "profile_model_hook_failures": 0,
            "profile_forward_backward_pct": 60.0,
            "profile_forward_pct": 20.0,
            "profile_loss_pct": 8.0,
            "profile_loss_reduce_pct": 3.0,
            "profile_backward_pct": 40.0,
            "profile_backward_grad_ready_child_count": 3,
            "profile_backward_grad_ready_parent_avg_ms": 14.0,
            "profile_backward_grad_ready_top_avg_ms": 7.0,
            "profile_backward_grad_ready_top_pct": 50.0,
            "profile_backward_grad_ready_top_calls": 3,
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
            "profile_open_phase_count": 0,
            "profile_open_detail_count": 0,
            "profile_model_status": "ok",
            "profile_model_modules_selected": 2,
            "profile_model_hook_count": 4,
            "profile_model_hook_failures": 0,
            "profile_forward_backward_pct": 55.0,
            "profile_forward_pct": 25.0,
            "profile_loss_pct": 6.0,
            "profile_loss_reduce_pct": 2.0,
            "profile_backward_pct": 30.0,
            "profile_backward_grad_ready_child_count": 2,
            "profile_backward_grad_ready_parent_avg_ms": 12.0,
            "profile_backward_grad_ready_top_avg_ms": 4.0,
            "profile_backward_grad_ready_top_pct": 45.0,
            "profile_backward_grad_ready_top_calls": 2,
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
    assert generated["mean_dataset_setup_time_s"] == pytest.approx(0.125)
    assert generated["mean_loader_setup_time_s"] == pytest.approx(0.185)
    assert generated["mean_model_setup_time_s"] == pytest.approx(0.315)
    assert generated["mean_compile_init_time_s"] == pytest.approx(0.03)
    assert generated["mean_transactions"] == pytest.approx(64.0)
    assert generated["mean_batch_size"] == pytest.approx(4.0)
    assert generated["mean_num_workers"] == pytest.approx(0.0)
    assert generated["mean_world_size"] == pytest.approx(1.0)
    assert generated["mean_dataset_materialized_bytes"] == pytest.approx(0.0)
    assert generated["mean_p99_s"] == pytest.approx(0.015)
    assert generated["mean_std_batch_s"] == pytest.approx(0.0015)
    assert generated["mean_avg_batch_s"] == pytest.approx(0.005)
    assert generated["mean_last_batch_s"] == pytest.approx(0.005)
    assert generated["mean_min_batch_s"] == pytest.approx(0.003)
    assert generated["mean_max_batch_s"] == pytest.approx(0.0075)
    assert generated["mean_batches"] == pytest.approx(3.0)
    assert generated["mean_best_samples_per_sec"] == pytest.approx(250.0)
    assert generated["mean_headroom_ratio"] == pytest.approx(1.3)
    assert generated["mean_ema_samples_per_sec"] == pytest.approx(215.0)
    assert generated["mean_window_samples_per_sec"] == pytest.approx(207.5)
    assert generated["mean_window_time_s"] == pytest.approx(0.06)
    assert generated["mean_window_batches"] == pytest.approx(2.0)
    assert generated["mean_window_samples"] == pytest.approx(8.0)
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
    assert generated["mean_profile_open_phase_count"] == pytest.approx(1.0)
    assert generated["max_profile_open_detail_count"] == pytest.approx(3.0)
    assert generated["mean_profile_model_modules_selected"] == pytest.approx(1.5)
    assert generated["max_profile_model_hook_count"] == pytest.approx(4.0)
    assert generated["mean_profile_model_hook_failures"] == pytest.approx(0.5)
    assert generated["profile_model_status_counts"] == {"hook_failures": 1, "ok": 1}
    assert generated["mean_profile_forward_backward_pct"] == pytest.approx(50.0)
    assert generated["mean_profile_loss_pct"] == pytest.approx(6.0)
    assert generated["mean_profile_loss_reduce_pct"] == pytest.approx(2.0)
    assert generated["mean_profile_user_metrics_pct"] == pytest.approx(4.0)
    assert generated["mean_profile_postprocess_pct"] == pytest.approx(6.0)
    assert generated["mean_profile_collect_output_pct"] == pytest.approx(2.0)
    assert generated["mean_profile_metrics_pct"] == pytest.approx(1.0)
    assert generated["max_profile_backward_pct"] == pytest.approx(40.0)
    assert generated["mean_profile_backward_grad_ready_child_count"] == pytest.approx(2.0)
    assert generated["mean_profile_backward_grad_ready_parent_avg_ms"] == pytest.approx(12.0)
    assert generated["mean_profile_backward_grad_ready_top_avg_ms"] == pytest.approx(5.0)
    assert generated["mean_profile_backward_grad_ready_top_pct"] == pytest.approx(40.0)
    assert generated["mean_profile_backward_grad_ready_top_calls"] == pytest.approx(2.0)
    assert generated["mean_cuda_current_mem_bytes"] == pytest.approx(1024.0)
    assert generated["sample_count_cuda_current_mem_bytes"] == pytest.approx(1.0)
    assert generated["mean_cuda_max_mem_bytes"] == pytest.approx(2048.0)
    assert generated["profiled_runs"] == 2
    assert summary["best_reported"]["mean_profile_flat_metric_invalid_count"] == pytest.approx(1.0)
    assert summary["best_reported"]["mean_profile_open_phase_count"] == pytest.approx(0.0)
    assert summary["best_reported"]["mean_profile_open_detail_count"] == pytest.approx(0.0)
    assert summary["best_reported"]["mean_profile_model_modules_selected"] == pytest.approx(2.0)
    assert summary["best_reported"]["mean_profile_model_hook_count"] == pytest.approx(4.0)
    assert summary["best_reported"]["mean_profile_model_hook_failures"] == pytest.approx(0.0)
    assert summary["best_reported"]["profile_model_status_counts"] == {"ok": 1}
    assert summary["best_reported"]["mean_profile_forward_backward_pct"] == pytest.approx(55.0)
    assert summary["best_reported"]["mean_profile_loss_pct"] == pytest.approx(6.0)
    assert summary["best_reported"]["mean_profile_backward_grad_ready_top_pct"] == pytest.approx(45.0)
    assert summary["best_reported"]["mean_profile_backward_grad_ready_top_avg_ms"] == pytest.approx(4.0)
    assert summary["best_reported"]["mean_profile_backward_grad_ready_top_calls"] == pytest.approx(2.0)
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


def test_summarize_rows_omits_not_requested_profile_model_fields() -> None:
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
            "profile_model_requested": False,
            "profile_model_enabled": False,
            "profile_model_status": "not_requested",
            "profile_model_modules_selected": 0,
            "profile_model_hook_count": 0,
            "profile_model_hook_failures": 0,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert "profiled_runs" not in group
    assert "profile_model_status_counts" not in group
    assert "mean_profile_model_modules_selected" not in group
    assert "mean_profile_model_hook_count" not in group
    assert "mean_profile_model_hook_failures" not in group
    assert "profile_model_status_counts" not in summary["best_reported"]
    assert "mean_profile_model_modules_selected" not in summary["best_reported"]


def test_summarize_rows_omits_zero_only_scheduler_diagnostics() -> None:
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
            "scheduler_step_failures": 0,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert "mean_scheduler_step_failures" not in group
    assert "mean_scheduler_step_failures" not in summary["best_reported"]
    json.dumps(summary, allow_nan=False)


def test_summarize_rows_includes_scheduler_failures_when_positive() -> None:
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
            "scheduler_step_failures": 0,
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
            "scheduler_step_failures": 2,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["mean_scheduler_step_failures"] == pytest.approx(1.0)
    assert group["max_scheduler_step_failures"] == pytest.approx(2.0)
    assert summary["best_reported"]["mean_scheduler_step_failures"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)


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


def test_summarize_rows_skips_fractional_profile_invalid_count() -> None:
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
            "profile_flat_metric_invalid_count": 0.5,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["mean_profile_flat_metric_invalid_count"] == pytest.approx(0.0)
    assert group["sample_count_profile_flat_metric_invalid_count"] == pytest.approx(0.0)
    assert group["invalid_count_profile_flat_metric_invalid_count"] == pytest.approx(1.0)
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


def test_summarize_rows_skips_groups_with_negative_best_rank_values() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": -10.0,
            "samples_per_sec": -8.0,
            "steady_samples_per_sec": -10.0,
            "end_to_end_wall_time_s": -1.0,
            "setup_time_s": 0.25,
            "wall_time_s": -1.0,
            "dataset_materialized_bytes": 0,
        },
        {
            "matrix_dataset_mode": "materialized",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 120.0,
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
    assert generated["invalid_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert generated["sample_count_end_to_end_wall_time_s"] == pytest.approx(0.0)
    assert generated["invalid_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["best_reported"]["dataset_mode"] == "materialized"
    assert summary["best_end_to_end"]["dataset_mode"] == "materialized"
    json.dumps(summary, allow_nan=False)


def test_summarize_rows_skips_out_of_range_profile_percentages() -> None:
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
            "profile_forward_backward_pct": 125.0,
            "profile_loss_pct": -1.0,
            "profile_backward_pct": 40.0,
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
            "profile_loss_pct": 8.0,
        },
    ]

    summary = summarize_rows(rows)
    group = summary["groups"][0]

    assert group["mean_profile_forward_backward_pct"] == pytest.approx(60.0)
    assert group["sample_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    assert group["invalid_count_profile_forward_backward_pct"] == pytest.approx(1.0)
    assert group["mean_profile_loss_pct"] == pytest.approx(8.0)
    assert group["invalid_count_profile_loss_pct"] == pytest.approx(1.0)
    assert group["mean_profile_backward_pct"] == pytest.approx(40.0)
    json.dumps(summary, allow_nan=False)


def test_summarize_rows_skips_groups_with_only_missing_best_rank_values() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "setup_time_s": 0.25,
            "dataset_materialized_bytes": 0,
        },
        {
            "matrix_dataset_mode": "materialized",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 120.0,
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
    assert generated["missing_count_reported_samples_per_sec"] == pytest.approx(1.0)
    assert generated["sample_count_end_to_end_wall_time_s"] == pytest.approx(0.0)
    assert generated["missing_count_end_to_end_wall_time_s"] == pytest.approx(1.0)
    assert summary["best_reported"]["dataset_mode"] == "materialized"
    assert summary["best_end_to_end"]["dataset_mode"] == "materialized"
    json.dumps(summary, allow_nan=False)


def test_summarize_rows_accepts_string_integer_metadata() -> None:
    rows = [
        {
            "matrix_dataset_mode": " materialized ",
            "matrix_compile_mode": " no-compile ",
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
    assert group["dataset_mode"] == "materialized"
    assert group["compile_mode"] == "no-compile"


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


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("matrix_dataset_mode", None, "matrix_dataset_mode"),
        ("matrix_dataset_mode", True, "matrix_dataset_mode"),
        ("matrix_dataset_mode", "", "matrix_dataset_mode"),
        ("matrix_dataset_mode", "archive", "matrix_dataset_mode"),
        ("matrix_compile_mode", None, "matrix_compile_mode"),
        ("matrix_compile_mode", True, "matrix_compile_mode"),
        ("matrix_compile_mode", "   ", "matrix_compile_mode"),
        ("matrix_compile_mode", "maybe", "matrix_compile_mode"),
    ],
)
def test_summarize_rows_rejects_invalid_group_metadata(
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


@pytest.mark.parametrize(
    ("missing_field", "match"),
    [
        ("matrix_dataset_mode", "matrix_dataset_mode"),
        ("matrix_compile_mode", "matrix_compile_mode"),
        ("matrix_workers", "matrix_workers"),
    ],
)
def test_summarize_rows_rejects_missing_group_metadata(
    missing_field: str,
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
    del row[missing_field]

    with pytest.raises(ValueError, match=match):
        summarize_rows([row])


def test_summarize_rows_handles_empty_input() -> None:
    summary = summarize_rows([])

    assert summary["runs"] == 0
    assert summary["config_count"] == 0
    assert summary["groups"] == []
    assert summary["best_reported"] is None
    assert summary["best_end_to_end"] is None
