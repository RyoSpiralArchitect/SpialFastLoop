#!/usr/bin/env python3
"""Profile ResNet forward blocks and backward gradient-ready timing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset, get_best_device


def _build_fake_dataset(size: int, image_size: int, classes: int) -> TensorDataset:
    inputs = torch.randn(size, 3, image_size, image_size)
    targets = torch.randint(0, classes, (size,))
    return TensorDataset(inputs, targets)


def _build_dataset(args: argparse.Namespace):
    if args.dataset == "fake":
        return _build_fake_dataset(args.dataset_size, args.image_size, args.num_classes), args.num_classes

    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on optional torchvision
        raise RuntimeError("torchvision is required for --dataset cifar10.") from exc

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=args.download,
        transform=transform,
    )
    return dataset, 10


def _build_resnet18(num_classes: int) -> nn.Module:
    try:
        from torchvision import models
    except Exception as exc:  # pragma: no cover - depends on optional torchvision
        raise RuntimeError("torchvision is required for ResNet18 profiling.") from exc
    return models.resnet18(num_classes=num_classes)


def _top_rows(profile: dict[str, Any], group: str, key: str, limit: int) -> list[dict[str, Any]]:
    if key == "phase":
        return list(profile.get("top_phases", []))[:limit]
    if key in {"forward", "breakdown"}:
        return list(profile.get("phase_breakdowns", {}).get(group, {}).get("top_children", []))[:limit]
    if key == "backward":
        return list(profile.get("phase_events", {}).get(group, {}).get("top_children", []))[:limit]
    return []


def _print_summary(metrics: dict[str, Any], topk: int) -> None:
    profile = metrics.get("profile", {})
    print(
        f"samples_per_sec={metrics.get('reported_samples_per_sec', metrics.get('samples_per_sec', 0.0)):.1f} "
        f"total={metrics.get('samples_per_sec', 0.0):.1f}"
    )
    if metrics.get("warmup_steps", 0) > 0:
        print(
            f"cold_start_steps={metrics.get('cold_start_steps', 0)} "
            f"cold_start_time_s={metrics.get('cold_start_time_s', 0.0):.2f} "
            f"cold_start_samples_per_sec={metrics.get('cold_start_samples_per_sec', 0.0):.1f}"
        )
    if metrics.get("steady_steps", 0) > 0:
        print(
            f"steady_steps={metrics.get('steady_steps', 0)} "
            f"steady_samples_per_sec={metrics.get('steady_samples_per_sec', 0.0):.1f} "
            f"steady_p99_ms={metrics.get('steady_p99_s', 0.0) * 1e3:.2f}"
        )
    print(
        f"batch_latency_p99_ms={metrics.get('p99_s', 0.0) * 1e3:.2f} "
        f"batch_latency_std_ms={metrics.get('std_batch_s', 0.0) * 1e3:.2f}"
    )
    print(f"steps={metrics.get('steps', 0)} samples={metrics.get('samples', 0)}")

    phases = _top_rows(profile, "", "phase", topk)
    if phases:
        print("top phases:")
        for row in phases:
            print(f"  {row['name']}: {row.get('pct', 0.0):.1f}% avg={row.get('avg_ms', 0.0):.2f}ms")

    forward = _top_rows(profile, "forward", "forward", topk)
    if forward:
        print("forward drilldown:")
        for row in forward:
            print(
                f"  {row['name']}: {row.get('pct_of_parent', 0.0):.1f}% "
                f"avg={row.get('avg_ms', 0.0):.2f}ms p95={row.get('p95_ms', 0.0):.2f}ms"
            )

    backward = _top_rows(profile, "backward_grad_ready", "backward", topk)
    if backward:
        print("backward grad-ready:")
        for row in backward:
            print(f"  {row['name']}: avg={row.get('avg_ms', 0.0):.2f}ms p95={row.get('p95_ms', 0.0):.2f}ms")

    optimizer = _top_rows(profile, "optimizer", "breakdown", topk)
    if optimizer:
        print("optimizer drilldown:")
        for row in optimizer:
            print(f"  {row['name']}: {row.get('pct_of_parent', 0.0):.1f}% avg={row.get('avg_ms', 0.0):.2f}ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fake", "cifar10"], default="fake")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile for lower cold-start cost.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--profile-window", type=int, default=256)
    parser.add_argument("--profile-model-depth", type=int, default=2)
    parser.add_argument("--profile-model-max-modules", type=int, default=16)
    parser.add_argument("--profile-model-include", default="layer1,layer4")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device() if args.device == "auto" else args.device
    dataset, num_classes = _build_dataset(args)
    loader = dataloader_from_dataset(
        dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent=args.workers > 0,
        shuffle=True,
        drop_last=True,
    )

    model = _build_resnet18(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = FastTrainer(
        model,
        optimizer,
        device=device,
        use_compile=args.compile,
        grad_accum=args.grad_accum,
        log_interval=max(args.steps + 1, 1),
    )

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=args.steps,
        collect_profile=True,
        profile_sync=args.profile_sync,
        profile_window=args.profile_window,
        profile_model=True,
        profile_model_depth=args.profile_model_depth,
        profile_model_max_modules=args.profile_model_max_modules,
        profile_model_include=args.profile_model_include,
        warmup_steps=args.warmup_steps,
    )

    payload = {
        "device": device,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "compile": args.compile,
        "profile_model_include": args.profile_model_include,
        "profile_model_depth": args.profile_model_depth,
        "metrics": metrics,
    }
    _print_summary(metrics, args.topk)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
