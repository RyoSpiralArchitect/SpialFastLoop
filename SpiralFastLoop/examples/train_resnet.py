"""
Minimal example training on CIFAR-10, with a synthetic fallback when offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


DATASET_CHOICES = ("auto", "fake", "cifar10")


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


def _fake_dataset(samples: int, feature_dim: int, classes: int) -> tuple[TensorDataset, int, int, bool, str]:
    inputs = torch.randn(samples, feature_dim)
    weights = torch.randn(feature_dim, classes)
    targets = (inputs @ weights).argmax(dim=1)
    return TensorDataset(inputs, targets), classes, feature_dim, False, "fake"


def get_dataset(args: argparse.Namespace) -> tuple[Any, int, int, bool, str]:
    if args.dataset == "fake":
        return _fake_dataset(args.samples, args.feature_dim, args.classes)

    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=args.download,
            transform=transform,
        )
        return dataset, 10, 3 * 32 * 32, True, "cifar10"
    except Exception:
        if args.dataset == "cifar10":
            raise
        return _fake_dataset(args.samples, args.feature_dim, args.classes)


def build_model(num_classes: int, input_dim: int, image_data: bool) -> nn.Module:
    if image_data:
        try:
            from torchvision import models

            return models.resnet18(num_classes=num_classes)
        except Exception:
            pass
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="auto")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--samples", type=positive_int_arg, default=20_000)
    parser.add_argument("--feature-dim", type=positive_int_arg, default=128)
    parser.add_argument("--classes", type=positive_int_arg, default=5)
    parser.add_argument("--batch-size", type=positive_int_arg, default=256)
    parser.add_argument("--steps", type=positive_int_arg, default=200)
    parser.add_argument("--warmup-steps", type=non_negative_int_arg, default=0)
    parser.add_argument("--grad-accum", type=positive_int_arg, default=2)
    parser.add_argument("--workers", type=non_negative_int_arg, default=0)
    parser.add_argument("--log-interval", type=non_negative_int_arg, default=20)
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4)
    parser.add_argument("--device", type=device_arg, default="auto")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--meter-fast-mode", action="store_true")
    parser.add_argument("--collect-profile", action="store_true")
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument(
        "--no-profile-distribution",
        dest="profile_distribution",
        action="store_false",
    )
    parser.add_argument("--profile-window", type=positive_int_arg, default=512)
    args = parser.parse_args()
    try:
        validate_benchmark_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    device = best_device(args.device)
    dataset, num_classes, input_dim, image_data, dataset_source = get_dataset(args)
    loader = dataloader_from_dataset(
        dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.workers,
        persistent=args.workers > 0,
    )
    model = build_model(num_classes, input_dim, image_data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainer = FastTrainer(
        model,
        optimizer,
        device=device,
        grad_accum=args.grad_accum,
        log_interval=args.log_interval,
        use_compile=args.compile,
        meter_fast_mode=args.meter_fast_mode,
    )
    metrics = trainer.train_one_epoch(
        loader,
        criterion,
        steps=args.steps,
        collect_profile=args.collect_profile,
        profile_sync=args.profile_sync,
        profile_distribution=args.profile_distribution,
        profile_window=args.profile_window,
        warmup_steps=args.warmup_steps,
    )
    print(dumps_json({
        "device": device,
        "dataset": dataset_source,
        "config": {
            "requested_dataset": args.dataset,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "grad_accum": args.grad_accum,
            "compile": args.compile,
            "meter_fast_mode": args.meter_fast_mode,
            "collect_profile": args.collect_profile,
            "profile_sync": args.profile_sync,
            "profile_distribution": args.profile_distribution,
            "profile_window": args.profile_window,
        },
        "metrics": metrics,
    }))


if __name__ == "__main__":
    main()
