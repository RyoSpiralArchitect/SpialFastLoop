#!/usr/bin/env python3
"""Run a small matrix of transactional benchmark configurations."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_parallel_transactions import (
    BASE_SUMMARY_FIELDS,
    PROFILE_MODEL_STATUS_ORDER,
    SUMMARY_INTEGER_FIELDS,
    _best_finite_row,
    _finite_summary_value,
    _format_metric_value,
    _format_profile_model_hook_summary,
    _format_scheduler_summary,
    _format_setup_breakdown,
    _int_arg,
    _positive_sample_count_value,
    _profile_model_status_counts,
    _summary_metric_max_value,
    _summary_metric_min_value,
    _summary_row,
    count_profiled_rows,
    device_arg,
    non_negative_int_arg,
    positive_float_arg,
    positive_int_arg,
    run_once,
    summarize_metric,
    summary_fields_for_rows,
    validate_benchmark_args,
)
from json_utils import dump_json


def _parse_csv_choices(raw: object, allowed: set[str], *, name: str) -> list[str]:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a comma-separated string")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one value")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"{name} includes unsupported values: {', '.join(invalid)}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not include duplicate values")
    return values


def _parse_worker_counts(raw: object) -> list[int]:
    if not isinstance(raw, str):
        raise ValueError("worker counts must be a comma-separated string")
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = non_negative_int_arg(text)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"worker counts {exc}") from exc
        values.append(value)
    if not values:
        raise ValueError("worker counts must include at least one value")
    if len(set(values)) != len(values):
        raise ValueError("worker counts must not include duplicate values")
    return values


def _compile_requested(mode: str) -> bool:
    if mode == "compile":
        return True
    if mode == "no-compile":
        return False
    raise ValueError(f"unsupported compile mode: {mode}")


def _non_negative_summary_int(raw: object, name: str) -> int:
    try:
        return non_negative_int_arg(raw)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"{name} {exc}") from exc


def _summary_choice(raw: object, allowed: set[str], name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = raw.strip()
    if value not in allowed:
        raise ValueError(f"{name} has unsupported value: {value}")
    return value


def _required_row_value(row: dict, field: str) -> object:
    if field not in row:
        raise ValueError(f"{field} is required")
    return row[field]


def _group_key(row: dict) -> tuple[str, str, int]:
    return (
        _summary_choice(
            _required_row_value(row, "matrix_dataset_mode"),
            {"generated", "materialized"},
            "matrix_dataset_mode",
        ),
        _summary_choice(
            _required_row_value(row, "matrix_compile_mode"),
            {"compile", "no-compile"},
            "matrix_compile_mode",
        ),
        _non_negative_summary_int(_required_row_value(row, "matrix_workers"), "matrix_workers"),
    )


def summarize_rows(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        normalized_row = _summary_row(row)
        groups.setdefault(_group_key(normalized_row), []).append(normalized_row)

    summaries = []
    for (dataset_mode, compile_mode, workers), group_rows in sorted(groups.items()):
        summary = {
            "dataset_mode": dataset_mode,
            "compile_mode": compile_mode,
            "workers": workers,
            "runs": len(group_rows),
            "dataset_materialized_bytes": max(
                _non_negative_summary_int(
                    row.get("dataset_materialized_bytes", 0),
                    "dataset_materialized_bytes",
                )
                for row in group_rows
            ),
        }
        profiled_runs = count_profiled_rows(group_rows)
        if profiled_runs > 0:
            summary["profiled_runs"] = profiled_runs
        status_counts, status_invalid_count = _profile_model_status_counts(group_rows)
        if status_counts:
            summary["profile_model_status_counts"] = status_counts
        if status_invalid_count > 0:
            summary["profile_model_status_invalid_count"] = status_invalid_count
        for field in summary_fields_for_rows(group_rows):
            missing_as_zero = field in BASE_SUMMARY_FIELDS
            for stat_name, value in summarize_metric(
                group_rows,
                field,
                missing_as_zero=missing_as_zero,
                min_value=_summary_metric_min_value(field),
                max_value=_summary_metric_max_value(field),
                integer=field in SUMMARY_INTEGER_FIELDS,
            ).items():
                summary[f"{stat_name}_{field}"] = value
        summaries.append(summary)

    best_reported = None
    best_end_to_end = None
    if summaries:
        best_reported = _best_finite_row(
            summaries,
            "mean_reported_samples_per_sec",
            prefer_high=True,
            sample_count_field="sample_count_reported_samples_per_sec",
            min_value=0.0,
        )
        best_end_to_end = _best_finite_row(
            summaries,
            "mean_end_to_end_wall_time_s",
            prefer_high=False,
            sample_count_field="sample_count_end_to_end_wall_time_s",
            min_value=0.0,
        )

    return {
        "runs": len(rows),
        "config_count": len(summaries),
        "groups": summaries,
        "best_reported": best_reported,
        "best_end_to_end": best_end_to_end,
    }


def _measured_summary_value(row: dict, mean_field: str) -> Optional[float]:
    if mean_field not in row:
        return None
    value = _finite_summary_value(row[mean_field])
    if value is None:
        return None
    if value < 0.0:
        return None

    metric_name = mean_field[len("mean_"):] if mean_field.startswith("mean_") else mean_field
    max_value = _summary_metric_max_value(metric_name)
    if max_value is not None and value > max_value:
        return None
    sample_count_field = f"sample_count_{metric_name}"
    if sample_count_field in row:
        sample_count = _positive_sample_count_value(row[sample_count_field])
        if sample_count is None:
            return None
    return value


def _format_summary_row(row: dict) -> str:
    reported_samples_per_sec = _measured_summary_value(row, "mean_reported_samples_per_sec")
    reported_text = (
        f"{reported_samples_per_sec:.1f}/s"
        if reported_samples_per_sec is not None
        else "n/a"
    )
    end_to_end_wall_time_s = _measured_summary_value(row, "mean_end_to_end_wall_time_s")
    end_to_end_text = (
        f"{end_to_end_wall_time_s:.2f}s"
        if end_to_end_wall_time_s is not None
        else "n/a"
    )
    setup_time_s = _measured_summary_value(row, "mean_setup_time_s")
    setup_text = f" setup={setup_time_s:.2f}s" if setup_time_s is not None else ""
    setup_breakdown = _format_setup_breakdown({
        "dataset_setup_time_s": _measured_summary_value(row, "mean_dataset_setup_time_s"),
        "loader_setup_time_s": _measured_summary_value(row, "mean_loader_setup_time_s"),
        "model_setup_time_s": _measured_summary_value(row, "mean_model_setup_time_s"),
        "compile_init_time_s": _measured_summary_value(row, "mean_compile_init_time_s"),
    })
    setup_breakdown_text = f" {setup_breakdown}" if setup_breakdown else ""
    profile_parts = []
    forward_backward_pct = _measured_summary_value(row, "mean_profile_forward_backward_pct")
    if forward_backward_pct is not None:
        profile_parts.append(f"fwd+bwd={forward_backward_pct:.1f}%")
    loss_pct = _measured_summary_value(row, "mean_profile_loss_pct")
    if loss_pct is not None:
        profile_parts.append(f"loss={loss_pct:.1f}%")
    optimizer_pct = _measured_summary_value(row, "mean_profile_optimizer_pct")
    if optimizer_pct is not None:
        profile_parts.append(f"opt={optimizer_pct:.1f}%")
    backward_ready_pct = _measured_summary_value(row, "mean_profile_backward_grad_ready_top_pct")
    backward_ready_avg_ms = _measured_summary_value(row, "mean_profile_backward_grad_ready_top_avg_ms")
    if backward_ready_pct is not None:
        ready_text = f"bwd_ready={backward_ready_pct:.1f}%"
        if backward_ready_avg_ms is not None:
            ready_text = f"{ready_text}@{backward_ready_avg_ms:.2f}ms"
        profile_parts.append(ready_text)
    open_phase_count = _measured_summary_value(row, "mean_profile_open_phase_count")
    open_detail_count = _measured_summary_value(row, "mean_profile_open_detail_count")
    open_parts = []
    if open_phase_count is not None and open_phase_count > 0.0:
        open_parts.append(f"phases={open_phase_count:.1f}")
    if open_detail_count is not None and open_detail_count > 0.0:
        open_parts.append(f"details={open_detail_count:.1f}")
    if open_parts:
        profile_parts.append(f"open({','.join(open_parts)})")
    profile_model_modules = _measured_summary_value(row, "mean_profile_model_modules_selected")
    profile_model_hooks = _measured_summary_value(row, "mean_profile_model_hook_count")
    profile_model_failures = _measured_summary_value(row, "mean_profile_model_hook_failures")
    model_parts = []
    if profile_model_modules is not None and profile_model_modules > 0.0:
        model_parts.append(f"modules={profile_model_modules:.1f}")
    if profile_model_hooks is not None and profile_model_hooks > 0.0:
        model_parts.append(f"hooks={profile_model_hooks:.1f}")
    if profile_model_failures is not None and profile_model_failures > 0.0:
        model_parts.append(f"failures={profile_model_failures:.1f}")
    if model_parts:
        profile_parts.append(f"model({','.join(model_parts)})")
    status_counts = row.get("profile_model_status_counts")
    status_parts = []
    if isinstance(status_counts, dict):
        for status in PROFILE_MODEL_STATUS_ORDER:
            if status == "not_requested":
                continue
            raw_count = status_counts.get(status)
            if isinstance(raw_count, (bool, str)):
                continue
            try:
                count = non_negative_int_arg(raw_count)
            except argparse.ArgumentTypeError:
                continue
            if count > 0:
                status_parts.append(f"{status}={count}")
    raw_status_invalid_count = row.get("profile_model_status_invalid_count")
    if isinstance(raw_status_invalid_count, (bool, str)):
        status_invalid_count = 0
    else:
        try:
            status_invalid_count = non_negative_int_arg(raw_status_invalid_count)
        except argparse.ArgumentTypeError:
            status_invalid_count = 0
    if status_invalid_count > 0:
        status_parts.append(f"invalid={status_invalid_count}")
    if status_parts:
        profile_parts.append(f"status({','.join(status_parts)})")
    scheduler_failures = _measured_summary_value(row, "mean_scheduler_step_failures")
    if scheduler_failures is not None and scheduler_failures > 0.0:
        profile_parts.append(f"scheduler(failures={scheduler_failures:.1f})")
    profile_suffix = f" {' '.join(profile_parts)}" if profile_parts else ""
    return (
        f"{row['dataset_mode']} {row['compile_mode']} workers={row['workers']} "
        f"reported={reported_text} "
        f"e2e={end_to_end_text}"
        f"{setup_text}"
        f"{setup_breakdown_text}"
        f"{profile_suffix}"
    )


def _format_run_row(dataset_mode: str, compile_mode: str, workers: int, run_index: int, result: dict) -> str:
    steady_text = _format_metric_value(
        result.get("reported_samples_per_sec", result.get("samples_per_sec")),
        precision=1,
        suffix="/s",
    )
    e2e_text = _format_metric_value(
        result.get("end_to_end_wall_time_s", result.get("wall_time_s")),
        precision=2,
        suffix="s",
    )
    setup_parts = []
    if "setup_time_s" in result:
        setup_parts.append(
            f"setup={_format_metric_value(result.get('setup_time_s'), precision=2, suffix='s')}"
        )
    setup_breakdown = _format_setup_breakdown(result)
    if setup_breakdown:
        setup_parts.append(setup_breakdown)
    setup_prefix = f"{' '.join(setup_parts)} " if setup_parts else ""
    profile_model_summary = _format_profile_model_hook_summary(result)
    profile_suffix = f" profile_model({profile_model_summary})" if profile_model_summary else ""
    scheduler_summary = _format_scheduler_summary(result)
    scheduler_suffix = f" scheduler({scheduler_summary})" if scheduler_summary else ""
    return (
        f"{dataset_mode:>12} {compile_mode:>10} workers={workers:<2} "
        f"run={run_index:<2} "
        f"steady={steady_text} "
        f"{setup_prefix}"
        f"e2e={e2e_text}"
        f"{profile_suffix}"
        f"{scheduler_suffix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=positive_int_arg, default=4096)
    parser.add_argument("--feature-dim", type=positive_int_arg, default=64)
    parser.add_argument("--num-classes", type=positive_int_arg, default=8)
    parser.add_argument("--batch-size", type=positive_int_arg, default=128)
    parser.add_argument("--grad-accum", type=positive_int_arg, default=2)
    parser.add_argument("--steps", type=positive_int_arg, default=16)
    parser.add_argument("--warmup-steps", type=non_negative_int_arg, default=2)
    parser.add_argument("--runs", type=positive_int_arg, default=1)
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4)
    parser.add_argument("--seed", type=_int_arg, default=1234)
    parser.add_argument("--device", type=device_arg, default="auto")
    parser.add_argument("--prefetch-factor", type=positive_int_arg, default=4)
    parser.add_argument("--log-interval", type=non_negative_int_arg, default=0)
    parser.add_argument("--dataset-modes", type=str, default="generated,materialized")
    parser.add_argument("--compile-modes", type=str, default="no-compile")
    parser.add_argument("--worker-counts", type=str, default="0")
    parser.add_argument("--collect-profile", action="store_true")
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--no-profile-distribution", dest="profile_distribution", action="store_false")
    parser.add_argument("--profile-window", type=positive_int_arg, default=512)
    parser.add_argument("--profile-model", action="store_true")
    parser.add_argument("--profile-model-depth", type=positive_int_arg, default=1)
    parser.add_argument("--profile-model-max-modules", type=positive_int_arg, default=64)
    parser.add_argument("--profile-model-include", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--summary-out", type=str, default=None)
    args = parser.parse_args()
    try:
        validate_benchmark_args(args)
        _parse_csv_choices(
            args.dataset_modes,
            {"generated", "materialized"},
            name="dataset modes",
        )
        _parse_csv_choices(
            args.compile_modes,
            {"compile", "no-compile"},
            name="compile modes",
        )
        _parse_worker_counts(args.worker_counts)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _run_args(args: argparse.Namespace, dataset_mode: str, compile_mode: str, workers: int) -> Namespace:
    return Namespace(
        transactions=args.transactions,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        workers=workers,
        prefetch_factor=args.prefetch_factor,
        device=args.device,
        steps=args.steps,
        log_interval=args.log_interval,
        compile=_compile_requested(compile_mode),
        warmup_steps=args.warmup_steps,
        runs=args.runs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        collect_profile=args.collect_profile or args.profile_model,
        profile_sync=args.profile_sync,
        profile_distribution=args.profile_distribution,
        profile_window=args.profile_window,
        profile_model=args.profile_model,
        profile_model_depth=args.profile_model_depth,
        profile_model_max_modules=args.profile_model_max_modules,
        profile_model_include=args.profile_model_include,
        dataset_mode=dataset_mode,
    )


def main() -> None:
    args = parse_args()
    dataset_modes = _parse_csv_choices(
        args.dataset_modes,
        {"generated", "materialized"},
        name="dataset modes",
    )
    compile_modes = _parse_csv_choices(
        args.compile_modes,
        {"compile", "no-compile"},
        name="compile modes",
    )
    worker_counts = _parse_worker_counts(args.worker_counts)

    rows = []
    for dataset_mode in dataset_modes:
        for compile_mode in compile_modes:
            for workers in worker_counts:
                run_args = _run_args(args, dataset_mode, compile_mode, workers)
                for run_index in range(args.runs):
                    result = run_once(run_args, run_index).as_dict()
                    result.update({
                        "matrix_dataset_mode": dataset_mode,
                        "matrix_compile_mode": compile_mode,
                        "matrix_workers": workers,
                    })
                    rows.append(result)
                    print(_format_run_row(dataset_mode, compile_mode, workers, run_index, result))

    summary = summarize_rows(rows)
    if summary["best_reported"]:
        print("Best steady:", _format_summary_row(summary["best_reported"]))
    if summary["best_end_to_end"]:
        print("Best end-to-end:", _format_summary_row(summary["best_end_to_end"]))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            dump_json(rows, handle)
        print(f"Wrote matrix results to {out_path}")
    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            dump_json(summary, handle)
        print(f"Wrote matrix summary to {out_path}")


if __name__ == "__main__":
    main()
