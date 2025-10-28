#!/usr/bin/env python3
"""Benchmark FastTrainer under heavy transactional and parallel workloads."""

import argparse
import json
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
        persistent=True,
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
    metrics = trainer.train_one_epoch(loader, criterion, steps=args.steps)
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
    parser.add_argument("--runs", type=int, default=3, help="How many repeated runs to execute.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the synthetic model.")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for synthetic data.")
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to dump the benchmark results as JSON for dashboards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for run_index in range(args.runs):
        result = run_once(args, run_index)
        results.append(result)
        metrics = result.as_dict()
        print(
            f"Run {run_index}: wall={metrics['wall_time_s']:.2f}s "
            f"thr={metrics.get('samples_per_sec', 0.0):.1f}/s "
            f"avg_loss={metrics.get('avg_loss', 0.0):.4f}"
        )

    if args.json_out:
        payload = [result.as_dict() for result in results]
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote results to {args.json_out}")

    aggregate = {
        "runs": args.runs,
        "mean_wall_time_s": sum(r.wall_time_s for r in results) / max(1, len(results)),
        "mean_samples_per_sec": sum(r.trainer_metrics.get("samples_per_sec", 0.0) for r in results)
        / max(1, len(results)),
    }
    print("Aggregate:", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
