#!/usr/bin/env python3
"""Profile ResNet forward blocks and backward gradient-ready timing."""

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
from spiralfastloop.utils import (
    _bool_setting,
    _positive_int_setting,
    dataloader_from_dataset,
    get_best_device,
)
from scripts.bench_parallel_transactions import (
    _dict_value,
    _format_count,
    _format_metric_value,
    _format_non_negative_metric_value,
    _format_profile_open_timer_summary,
    _format_profile_breakdown_child_timing,
    _format_profile_breakdown_summary,
    _format_profile_event_timing,
    _has_positive_display_value,
    _list_value,
    _optional_path_setting,
    _path_setting,
    _profile_child_rows,
    _profile_row_name,
    device_arg,
    non_negative_int_arg,
    positive_float_arg,
    positive_int_arg,
    validate_benchmark_args,
)
from scripts.json_utils import dumps_json

DATASET_CHOICES = ("fake", "cifar10")


def _build_fake_dataset(size: int, image_size: int, classes: int) -> TensorDataset:
    size = _positive_int_setting(size, "size")
    image_size = _positive_int_setting(image_size, "image_size")
    classes = _positive_int_setting(classes, "classes")
    inputs = torch.randn(size, 3, image_size, image_size)
    targets = torch.randint(0, classes, (size,))
    return TensorDataset(inputs, targets)


def _dataset_arg(raw: object) -> str:
    if not isinstance(raw, str) or raw not in DATASET_CHOICES:
        raise ValueError("dataset must be one of fake, cifar10")
    return raw


def _build_dataset(args: argparse.Namespace):
    dataset_name = _dataset_arg(args.dataset)
    if dataset_name == "fake":
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
    num_classes = _positive_int_setting(num_classes, "num_classes")
    try:
        from torchvision import models
    except Exception as exc:  # pragma: no cover - depends on optional torchvision
        raise RuntimeError("torchvision is required for ResNet18 profiling.") from exc
    return models.resnet18(num_classes=num_classes)


def _profile_rows(rows: Any, limit: int) -> list[dict[str, Any]]:
    return [row for row in _list_value(rows) if isinstance(row, dict)][:limit]


def _top_rows(profile: dict[str, Any], group: str, key: str, limit: int) -> list[dict[str, Any]]:
    if key == "phase":
        return _profile_rows(profile.get("top_phases", []), limit)
    if key in {"forward", "breakdown"}:
        return _profile_rows(_profile_child_rows(profile, "phase_breakdowns", group), limit)
    if key == "backward":
        return _profile_rows(_profile_child_rows(profile, "phase_events", group), limit)
    return []


def _format_phase_timing(row: dict[str, Any]) -> str:
    pct_text = _format_non_negative_metric_value(row.get("pct"), precision=1, suffix="%")
    parts = [pct_text if pct_text is not None else "n/a"]
    avg_text = _format_non_negative_metric_value(row.get("avg_ms"), precision=2, suffix="ms")
    parts.append(f"avg={avg_text if avg_text is not None else 'n/a'}")
    for field, label in (
        ("p95_ms", "p95"),
        ("p99_ms", "p99"),
        ("std_ms", "std"),
    ):
        if field not in row:
            continue
        value_text = _format_non_negative_metric_value(row.get(field), precision=2, suffix="ms")
        parts.append(f"{label}={value_text if value_text is not None else 'n/a'}")
    return " ".join(parts)


def _print_summary(metrics: dict[str, Any], topk: int) -> None:
    topk = _positive_int_setting(topk, "topk")
    profile = _dict_value(metrics.get("profile"))
    print(
        f"samples_per_sec="
        f"{_format_metric_value(metrics.get('reported_samples_per_sec', metrics.get('samples_per_sec')), precision=1)} "
        f"total={_format_metric_value(metrics.get('samples_per_sec'), precision=1)}"
    )
    if _has_positive_display_value(metrics.get("warmup_steps")):
        print(
            f"cold_start_steps={_format_count(metrics.get('cold_start_steps'))} "
            f"cold_start_time_s={_format_metric_value(metrics.get('cold_start_time_s'), precision=2)} "
            f"cold_start_samples_per_sec="
            f"{_format_metric_value(metrics.get('cold_start_samples_per_sec'), precision=1)}"
        )
    if _has_positive_display_value(metrics.get("steady_steps")):
        print(
            f"steady_steps={_format_count(metrics.get('steady_steps'))} "
            f"steady_samples_per_sec={_format_metric_value(metrics.get('steady_samples_per_sec'), precision=1)} "
            f"steady_p99_ms={_format_metric_value(metrics.get('steady_p99_s'), precision=2, scale=1e3)}"
        )
    print(
        f"batch_latency_p99_ms={_format_metric_value(metrics.get('p99_s'), precision=2, scale=1e3)} "
        f"batch_latency_std_ms={_format_metric_value(metrics.get('std_batch_s'), precision=2, scale=1e3)}"
    )
    print(f"steps={_format_count(metrics.get('steps'))} samples={_format_count(metrics.get('samples'))}")

    open_timer_summary = _format_profile_open_timer_summary(profile)
    if open_timer_summary:
        print(f"open timers: {open_timer_summary}")

    phases = _top_rows(profile, "", "phase", topk)
    if phases:
        print("top phases:")
        for row in phases:
            print(
                f"  {_profile_row_name(row)}: "
                f"{_format_phase_timing(row)}"
            )

    forward = _top_rows(profile, "forward", "forward", topk)
    if forward:
        summary = _format_profile_breakdown_summary(profile, "forward")
        print(f"forward drilldown: {summary}" if summary else "forward drilldown:")
        for row in forward:
            print(
                f"  {_profile_row_name(row)}: "
                f"{_format_profile_breakdown_child_timing(row)}"
            )

    backward = _top_rows(profile, "backward_grad_ready", "backward", topk)
    if backward:
        print("backward grad-ready:")
        for row in backward:
            print(
                f"  {_profile_row_name(row)}: "
                f"avg={_format_profile_event_timing(row, precision=2, include_p95=True)}"
            )

    optimizer = _top_rows(profile, "optimizer", "breakdown", topk)
    if optimizer:
        summary = _format_profile_breakdown_summary(profile, "optimizer")
        print(f"optimizer drilldown: {summary}" if summary else "optimizer drilldown:")
        for row in optimizer:
            print(
                f"  {_profile_row_name(row)}: "
                f"{_format_profile_breakdown_child_timing(row)}"
            )


def validate_resnet_profile_args(args: argparse.Namespace) -> None:
    validate_benchmark_args(args)
    dataset_name = _dataset_arg(args.dataset)
    dataset_size = _positive_int_setting(args.dataset_size, "dataset_size")
    batch_size = _positive_int_setting(args.batch_size, "batch_size")
    if hasattr(args, "image_size"):
        _positive_int_setting(args.image_size, "image_size")
    if hasattr(args, "topk"):
        _positive_int_setting(args.topk, "topk")
    if hasattr(args, "download"):
        _bool_setting(args.download, "download")
    if hasattr(args, "data_root"):
        _path_setting(args.data_root, "data_root")
    if hasattr(args, "json_out"):
        _optional_path_setting(args.json_out, "json_out")
    if dataset_name == "fake" and dataset_size < batch_size:
        raise ValueError("dataset-size must be at least batch-size when using --dataset fake.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="fake")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dataset-size", type=positive_int_arg, default=4096)
    parser.add_argument("--image-size", type=positive_int_arg, default=32)
    parser.add_argument("--num-classes", type=positive_int_arg, default=10)
    parser.add_argument("--device", type=device_arg, default="auto")
    parser.add_argument("--batch-size", type=positive_int_arg, default=256)
    parser.add_argument("--grad-accum", type=positive_int_arg, default=2)
    parser.add_argument("--workers", type=non_negative_int_arg, default=2)
    parser.add_argument("--prefetch-factor", type=positive_int_arg, default=2)
    parser.add_argument("--steps", type=positive_int_arg, default=12)
    parser.add_argument("--warmup-steps", type=non_negative_int_arg, default=0)
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable torch.compile for lower cold-start cost.",
    )
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4)
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--profile-window", type=positive_int_arg, default=256)
    parser.add_argument("--profile-model-depth", type=positive_int_arg, default=2)
    parser.add_argument("--profile-model-max-modules", type=positive_int_arg, default=16)
    parser.add_argument("--profile-model-include", default="layer1,layer4")
    parser.add_argument("--topk", type=positive_int_arg, default=6)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    try:
        validate_resnet_profile_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


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
        out_path.write_text(dumps_json(payload), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
