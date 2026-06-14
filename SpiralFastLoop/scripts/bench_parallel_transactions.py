#!/usr/bin/env python3
"""Benchmark FastTrainer under heavy transactional and parallel workloads."""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset


@dataclass
class BenchmarkResult:
    wall_time_s: float
    trainer_metrics: dict
    run_index: int

    def as_dict(self) -> dict:
        payload = {"wall_time_s": self.wall_time_s, "run": self.run_index}
        payload.update(self.trainer_metrics)
        return payload


class SyntheticTransactionDataset(Dataset):
    """Synthetic tabular dataset that simulates transactional workloads."""

    def __init__(self, size: int, features: int, classes: int, *, seed: int = 17) -> None:
        self.size = size
        self.features = features
        self.classes = classes
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        features = torch.randn(self.features, generator=generator)
        target = torch.randint(0, self.classes, (1,), generator=generator).squeeze(0)
        return features, target


def build_model(features: int, classes: int) -> nn.Module:
    hidden = max(32, features * 2)
    return nn.Sequential(
        nn.Linear(features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    )


def run_once(args, run_index: int) -> BenchmarkResult:
    dataset = SyntheticTransactionDataset(
        size=args.transactions,
        features=args.feature_dim,
        classes=args.num_classes,
        seed=args.seed + run_index,
    )

    loader = dataloader_from_dataset(
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent=args.workers > 0,
        shuffle=True,
    )

    model = build_model(args.feature_dim, args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = FastTrainer(
        model,
        optimizer,
        scheduler=None,
        device=args.device,
        grad_accum=args.grad_accum,
        log_interval=max(1, args.steps // 5),
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
    wall = time.perf_counter() - start
    metrics.update({
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "transactions": args.transactions,
    })
    return BenchmarkResult(wall_time_s=wall, trainer_metrics=metrics, run_index=run_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=int, default=100_000, help="Total synthetic transactions to sample.")
    parser.add_argument("--feature-dim", type=int, default=128, help="Width of each synthetic transaction vector.")
    parser.add_argument("--num-classes", type=int, default=32, help="Number of synthetic classification targets.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for the benchmark dataloader.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation factor.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader worker processes.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor passed to the dataloader.")
    parser.add_argument("--device", type=str, default="auto", help="Device override (auto/cuda/mps/cpu).")
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps per run.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Measure the first N steps separately and exclude them from steady-state throughput.",
    )
    parser.add_argument("--runs", type=int, default=3, help="How many repeated runs to execute.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the synthetic model.")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for synthetic data.")
    parser.add_argument("--collect-profile", action="store_true", help="Collect train-loop phase timings.")
    parser.add_argument("--profile-sync", action="store_true", help="Synchronize accelerator around profiled phases.")
    parser.add_argument(
        "--no-profile-distribution",
        dest="profile_distribution",
        action="store_false",
        help="Skip p50/p95/p99/std samples and collect phase totals only.",
    )
    parser.add_argument("--profile-window", type=int, default=512, help="Per-phase sample window size.")
    parser.add_argument("--profile-model", action="store_true", help="Collect module-level forward/backward drilldowns.")
    parser.add_argument("--profile-model-depth", type=int, default=1, help="Exact module depth to profile.")
    parser.add_argument("--profile-model-max-modules", type=int, default=64, help="Maximum modules to hook.")
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
    return parser.parse_args()


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
            f"Run {run_index}: wall={metrics['wall_time_s']:.2f}s "
            f"thr={metrics.get('reported_samples_per_sec', metrics.get('samples_per_sec', 0.0)):.1f}/s "
            f"total_thr={metrics.get('samples_per_sec', 0.0):.1f}/s "
            f"p99_batch={metrics.get('p99_s', 0.0) * 1e3:.2f}ms "
            f"std_batch={metrics.get('std_batch_s', 0.0) * 1e3:.2f}ms "
            f"avg_loss={metrics.get('avg_loss', 0.0):.4f}"
        )
        if metrics.get("warmup_steps", 0) > 0:
            print(
                f"  cold_start: steps={metrics.get('cold_start_steps', 0)} "
                f"time={metrics.get('cold_start_time_s', 0.0):.2f}s "
                f"thr={metrics.get('cold_start_samples_per_sec', 0.0):.1f}/s"
            )
        if metrics.get("steady_steps", 0) > 0:
            print(
                f"  steady: steps={metrics.get('steady_steps', 0)} "
                f"thr={metrics.get('steady_samples_per_sec', 0.0):.1f}/s "
                f"p99_batch={metrics.get('steady_p99_s', 0.0) * 1e3:.2f}ms"
            )
        profile = metrics.get("profile")
        if profile:
            top_phases = ", ".join(
                f"{row['name']}={row.get('pct', 0.0):.1f}%"
                for row in profile.get("top_phases", [])[:4]
            )
            print(f"  phases: {top_phases}")
            forward = profile.get("phase_breakdowns", {}).get("forward", {}).get("top_children", [])
            if forward:
                top_forward = ", ".join(
                    f"{row['name']}={row.get('pct_of_parent', 0.0):.1f}%"
                    for row in forward[:4]
                )
                print(f"  forward: {top_forward}")
            backward = profile.get("phase_events", {}).get("backward_grad_ready", {}).get("top_children", [])
            if backward:
                top_backward = ", ".join(
                    f"{row['name']}={row.get('avg_ms', 0.0):.1f}ms"
                    for row in backward[:4]
                )
                print(f"  backward_grad_ready: {top_backward}")
            optimizer = profile.get("phase_breakdowns", {}).get("optimizer", {}).get("top_children", [])
            if optimizer:
                top_optimizer = ", ".join(
                    f"{row['name']}={row.get('pct_of_parent', 0.0):.1f}%"
                    for row in optimizer[:4]
                )
                print(f"  optimizer: {top_optimizer}")

    if args.json_out:
        payload = [result.as_dict() for result in results]
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote results to {out_path}")

    aggregate = {
        "runs": args.runs,
        "mean_wall_time_s": sum(r.wall_time_s for r in results) / max(1, len(results)),
        "mean_samples_per_sec": sum(r.trainer_metrics.get("samples_per_sec", 0.0) for r in results)
        / max(1, len(results)),
        "mean_reported_samples_per_sec": sum(
            r.trainer_metrics.get("reported_samples_per_sec", r.trainer_metrics.get("samples_per_sec", 0.0))
            for r in results
        )
        / max(1, len(results)),
        "mean_steady_samples_per_sec": sum(r.trainer_metrics.get("steady_samples_per_sec", 0.0) for r in results)
        / max(1, len(results)),
        "mean_cold_start_time_s": sum(r.trainer_metrics.get("cold_start_time_s", 0.0) for r in results)
        / max(1, len(results)),
    }
    print("Aggregate:", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
