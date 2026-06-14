from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_matrix import (
    _compile_requested,
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

    with pytest.raises(ValueError):
        _parse_csv_choices("", {"generated"}, name="modes")
    with pytest.raises(ValueError):
        _parse_csv_choices("generated,other", {"generated"}, name="modes")


def test_parse_worker_counts_rejects_empty_or_negative_values() -> None:
    assert _parse_worker_counts("0, 2") == [0, 2]

    with pytest.raises(ValueError):
        _parse_worker_counts("")
    with pytest.raises(ValueError):
        _parse_worker_counts("-1")
    with pytest.raises(ValueError):
        _parse_worker_counts("cpu")


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


def test_summarize_rows_groups_configs_and_ranks_best() -> None:
    rows = [
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 100.0,
            "samples_per_sec": 80.0,
            "steady_samples_per_sec": 100.0,
            "end_to_end_wall_time_s": 3.0,
            "setup_time_s": 1.0,
            "wall_time_s": 2.0,
            "cold_start_time_s": 0.5,
            "dataset_materialized_bytes": 0,
        },
        {
            "matrix_dataset_mode": "generated",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 300.0,
            "samples_per_sec": 240.0,
            "steady_samples_per_sec": 300.0,
            "end_to_end_wall_time_s": 1.0,
            "setup_time_s": 0.25,
            "wall_time_s": 0.75,
            "cold_start_time_s": 0.1,
            "dataset_materialized_bytes": 0,
        },
        {
            "matrix_dataset_mode": "materialized",
            "matrix_compile_mode": "no-compile",
            "matrix_workers": 0,
            "reported_samples_per_sec": 250.0,
            "samples_per_sec": 180.0,
            "steady_samples_per_sec": 250.0,
            "end_to_end_wall_time_s": 0.8,
            "setup_time_s": 0.2,
            "wall_time_s": 0.6,
            "cold_start_time_s": 0.05,
            "dataset_materialized_bytes": 1024,
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


def test_summarize_rows_handles_empty_input() -> None:
    summary = summarize_rows([])

    assert summary["runs"] == 0
    assert summary["config_count"] == 0
    assert summary["groups"] == []
    assert summary["best_reported"] is None
    assert summary["best_end_to_end"] is None
