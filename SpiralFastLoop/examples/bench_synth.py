"""
Synthetic benchmark to showcase SpiralFastLoop throughput and latency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset
from scripts.bench_parallel_transactions import (
    device_arg,
    non_negative_int_arg,
    positive_float_arg,
    positive_int_arg,
    validate_benchmark_args,
)
from scripts.json_utils import dumps_json


def best_device(requested: str = "auto") -> str:
    try:
        requested = device_arg(requested)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"device {exc}") from exc
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=positive_int_arg, default=200_000)
    parser.add_argument("--feature-dim", type=positive_int_arg, default=1024)
    parser.add_argument("--hidden-dim", type=positive_int_arg, default=2048)
    parser.add_argument("--classes", type=positive_int_arg, default=64)
    parser.add_argument("--batch-size", type=positive_int_arg, default=256)
    parser.add_argument("--steps", type=positive_int_arg, default=300)
    parser.add_argument("--warmup-steps", type=non_negative_int_arg, default=0)
    parser.add_argument("--grad-accum", type=positive_int_arg, default=2)
    parser.add_argument("--workers", type=non_negative_int_arg, default=0)
    parser.add_argument("--log-interval", type=non_negative_int_arg, default=50)
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4)
    parser.add_argument("--device", type=device_arg, default="auto")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--collect-profile", action="store_true")
    args = parser.parse_args()
    try:
        validate_benchmark_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    device = best_device(args.device)
    inputs = torch.randn(args.samples, args.feature_dim)
    weights = torch.randn(args.feature_dim, args.classes)
    targets = (inputs @ weights).argmax(dim=1)
    dataset = TensorDataset(inputs, targets)

    loader = dataloader_from_dataset(
        dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.workers,
        persistent=args.workers > 0,
    )

    model = nn.Sequential(
        nn.Linear(args.feature_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.classes),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=(device == "cuda"))
    criterion = nn.CrossEntropyLoss()

    trainer = FastTrainer(
        model,
        optimizer,
        device=device,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        use_compile=args.compile,
    )
    metrics = trainer.train_one_epoch(
        loader,
        criterion,
        steps=args.steps,
        collect_profile=args.collect_profile,
        warmup_steps=args.warmup_steps,
    )
    print(dumps_json({
        "device": device,
        "config": {
            "samples": args.samples,
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "classes": args.classes,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "grad_accum": args.grad_accum,
            "compile": args.compile,
        },
        "metrics": metrics,
    }))


if __name__ == "__main__":
    main()
