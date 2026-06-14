#!/usr/bin/env python3
"""Run a small matrix of transactional benchmark configurations."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_parallel_transactions import run_once


def _parse_csv_choices(raw: str, allowed: set[str], *, name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one value")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"{name} includes unsupported values: {', '.join(invalid)}")
    return values


def _parse_worker_counts(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value < 0:
            raise ValueError("worker counts must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("worker counts must include at least one value")
    return values


def _compile_requested(mode: str) -> bool:
    if mode == "compile":
        return True
    if mode == "no-compile":
        return False
    raise ValueError(f"unsupported compile mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=int, default=4096)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--dataset-modes", type=str, default="generated,materialized")
    parser.add_argument("--compile-modes", type=str, default="no-compile")
    parser.add_argument("--worker-counts", type=str, default="0")
    parser.add_argument("--collect-profile", action="store_true")
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--no-profile-distribution", dest="profile_distribution", action="store_false")
    parser.add_argument("--profile-window", type=int, default=512)
    parser.add_argument("--profile-model", action="store_true")
    parser.add_argument("--profile-model-depth", type=int, default=1)
    parser.add_argument("--profile-model-max-modules", type=int, default=64)
    parser.add_argument("--profile-model-include", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


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
                    print(
                        f"{dataset_mode:>12} {compile_mode:>10} workers={workers:<2} "
                        f"run={run_index:<2} "
                        f"steady={result.get('reported_samples_per_sec', 0.0):.1f}/s "
                        f"e2e={result.get('end_to_end_wall_time_s', result['wall_time_s']):.2f}s"
                    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"Wrote matrix results to {out_path}")


if __name__ == "__main__":
    main()
