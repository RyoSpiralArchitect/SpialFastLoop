#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer, recommended_dataloader
from spiralfastloop.utils import (
    ThroughputMeter,
    _bool_setting,
    _finite_float_setting,
    _non_negative_int_setting,
    _non_negative_finite_float_setting,
    _optional_positive_int_setting,
    _positive_int_setting,
)
from scripts.bench_parallel_transactions import (
    device_arg,
    non_negative_int_arg,
    positive_float_arg,
    positive_int_arg,
    validate_benchmark_args,
)
from scripts.json_utils import dumps_json


class Synth(Dataset):
    def __init__(self, n: int = 50_000, d: int = 128, classes: int = 10) -> None:
        n = _positive_int_setting(n, "n")
        d = _positive_int_setting(d, "d")
        classes = _positive_int_setting(classes, "classes")
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, classes, (n,))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class MLP(nn.Module):
    def __init__(self, d: int = 128, classes: int = 10) -> None:
        super().__init__()
        d = _positive_int_setting(d, "d")
        classes = _positive_int_setting(classes, "classes")
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def best_device(requested: str = "auto") -> torch.device:
    try:
        requested = device_arg(requested)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"device {exc}") from exc
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _positive_float_setting(value: object, name: str) -> float:
    if isinstance(value, str):
        raise ValueError(f"{name} must be a positive finite number")
    normalized = _finite_float_setting(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return normalized


def _optimizer_params_on_cuda(parameters: list[torch.Tensor]) -> bool:
    return any(parameter.device.type == "cuda" for parameter in parameters)


def adamw(parameters, learning_rate: float, *, fused: Optional[bool] = None) -> torch.optim.AdamW:
    learning_rate = _positive_float_setting(learning_rate, "learning_rate")
    params = list(parameters)
    if fused is None:
        fused_value = _optimizer_params_on_cuda(params)
    elif isinstance(fused, bool):
        fused_value = fused
    else:
        raise ValueError("fused must be a boolean or None")
    return torch.optim.AdamW(params, lr=learning_rate, fused=fused_value)


def plain_loop(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    epochs: int = 1,
    steps: Optional[int] = None,
    grad_accum: int = 1,
    warmup_steps: int = 0,
    meter_fast_mode: bool = False,
) -> dict[str, float]:
    epochs = _positive_int_setting(epochs, "epochs")
    step_limit = _optional_positive_int_setting(steps, "steps")
    grad_accum = _positive_int_setting(grad_accum, "grad_accum")
    warmup_step_limit = _non_negative_int_setting(warmup_steps, "warmup_steps")
    meter_fast_mode_value = _bool_setting(meter_fast_mode, "meter_fast_mode")
    if step_limit is not None and warmup_step_limit > step_limit:
        raise ValueError("warmup_steps must be less than or equal to steps")
    model.to(device).train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    meter = ThroughputMeter(fast_mode=meter_fast_mode_value)
    warmup_meter = ThroughputMeter(fast_mode=meter_fast_mode_value)
    steady_meter = ThroughputMeter(fast_mode=meter_fast_mode_value)
    step_count = 0
    samples = 0
    loss_acc = 0.0
    warmup_loss_acc = 0.0
    steady_loss_acc = 0.0
    warmup_step_count = 0
    steady_step_count = 0
    warmup_samples = 0
    steady_samples = 0
    pending_accum_steps = 0
    optimizer_steps = 0
    partial_optimizer_steps = 0
    grad_accum_tail_steps = 0
    warmup_optimizer_steps = 0
    steady_optimizer_steps = 0

    def rescale_accumulated_gradients(factor: float) -> None:
        if factor == 1.0:
            return
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.detach().mul_(factor)

    def run_optimizer_step(accumulated_steps: int) -> None:
        nonlocal optimizer_steps
        nonlocal partial_optimizer_steps
        nonlocal grad_accum_tail_steps
        nonlocal warmup_optimizer_steps
        nonlocal steady_optimizer_steps
        if accumulated_steps <= 0:
            return
        if accumulated_steps < grad_accum:
            partial_optimizer_steps += 1
            grad_accum_tail_steps = accumulated_steps
            rescale_accumulated_gradients(grad_accum / accumulated_steps)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if warmup_step_limit > 0 and step_count <= warmup_step_limit:
            warmup_optimizer_steps += 1
        else:
            steady_optimizer_steps += 1

    for _ in range(epochs):
        data_iter = iter(loader)
        while step_limit is None or step_count < step_limit:
            batch_started = time.perf_counter()
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            (loss / grad_accum).backward()
            step_count += 1
            batch_size = int(inputs.shape[0])
            samples += batch_size
            loss_value = float(loss.detach().cpu())
            loss_acc += loss_value
            pending_accum_steps += 1
            reached_accum_boundary = pending_accum_steps >= grad_accum
            reached_requested_steps = step_limit is not None and step_count >= step_limit
            if reached_accum_boundary or reached_requested_steps:
                run_optimizer_step(pending_accum_steps)
                pending_accum_steps = 0
            batch_duration_s = _non_negative_finite_float_setting(
                time.perf_counter() - batch_started,
                "batch_duration_s",
            )
            meter.record(batch_duration_s, batch_size)
            if warmup_step_limit > 0 and step_count <= warmup_step_limit:
                warmup_meter.record(batch_duration_s, batch_size)
                warmup_step_count += 1
                warmup_samples += batch_size
                warmup_loss_acc += loss_value
            else:
                steady_meter.record(batch_duration_s, batch_size)
                steady_step_count += 1
                steady_samples += batch_size
                steady_loss_acc += loss_value
        if step_limit is not None and step_count >= step_limit:
            break
    if pending_accum_steps > 0:
        run_optimizer_step(pending_accum_steps)
    elapsed = _non_negative_finite_float_setting(time.perf_counter() - start, "elapsed_sec")
    metrics = dict(meter.summary())
    warmup_summary = warmup_meter.summary()
    steady_summary = steady_meter.summary()
    for key, value in warmup_summary.items():
        metrics[f"warmup_{key}"] = value
    for key, value in steady_summary.items():
        metrics[f"steady_{key}"] = value
    metrics.update(
        {
            "avg_loss_per_step": loss_acc / max(1, step_count),
            "warmup_avg_loss_per_step": warmup_loss_acc / max(1, warmup_step_count),
            "steady_avg_loss_per_step": steady_loss_acc / max(1, steady_step_count),
            "elapsed_sec": elapsed,
            "steps": step_count,
            "samples": samples,
            "warmup_steps": warmup_step_count,
            "steady_steps": steady_step_count,
            "warmup_samples": warmup_samples,
            "steady_samples": steady_samples,
            "optimizer_steps": optimizer_steps,
            "grad_accum": grad_accum,
            "partial_optimizer_steps": partial_optimizer_steps,
            "grad_accum_tail_steps": grad_accum_tail_steps,
            "warmup_optimizer_steps": warmup_optimizer_steps,
            "steady_optimizer_steps": steady_optimizer_steps,
            "cold_start_steps": warmup_step_count,
            "cold_start_time_s": warmup_summary["total_time_s"],
            "cold_start_samples_per_sec": warmup_summary["samples_per_sec"],
            "warmup_elapsed_sec": warmup_summary["total_time_s"],
            "steady_elapsed_sec": steady_summary["total_time_s"],
            "reported_samples_per_sec": (
                steady_summary["samples_per_sec"] if steady_step_count > 0 else metrics["samples_per_sec"]
            ),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a plain PyTorch loop with SpiralFastLoop.")
    parser.add_argument("--samples", type=positive_int_arg, default=50_000)
    parser.add_argument("--feature-dim", type=positive_int_arg, default=128)
    parser.add_argument("--classes", type=positive_int_arg, default=10)
    parser.add_argument("--batch-size", type=positive_int_arg, default=256)
    parser.add_argument("--steps", type=positive_int_arg, default=200)
    parser.add_argument("--warmup-steps", type=non_negative_int_arg, default=0)
    parser.add_argument("--grad-accum", type=positive_int_arg, default=2)
    parser.add_argument("--workers", type=non_negative_int_arg, default=2)
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4)
    parser.add_argument("--device", type=device_arg, default="auto")
    parser.add_argument("--log-interval", type=non_negative_int_arg, default=0)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--meter-fast-mode", action="store_true", help="Use lighter throughput meters without tail/window stats.")
    parser.add_argument("--collect-profile", action="store_true")
    args = parser.parse_args()
    try:
        validate_benchmark_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    dataset = Synth(n=args.samples, d=args.feature_dim, classes=args.classes)
    device = best_device(args.device)

    baseline_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    fused_optimizer = device.type == "cuda"
    baseline_model = MLP(d=args.feature_dim, classes=args.classes).to(device)
    baseline_optimizer = adamw(
        baseline_model.parameters(),
        args.learning_rate,
        fused=fused_optimizer,
    )
    criterion = nn.CrossEntropyLoss()
    baseline = plain_loop(
        baseline_loader,
        baseline_model,
        baseline_optimizer,
        criterion,
        device,
        epochs=1,
        steps=args.steps,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps,
        meter_fast_mode=args.meter_fast_mode,
    )

    loader = recommended_dataloader(
        dataset,
        batch_size=args.batch_size,
        device=str(device),
        num_workers=args.workers,
        persistent=args.workers > 0,
    )
    model = MLP(d=args.feature_dim, classes=args.classes).to(device)
    optimizer = adamw(model.parameters(), args.learning_rate, fused=fused_optimizer)
    trainer = FastTrainer(
        model,
        optimizer,
        device=str(device),
        use_compile=args.compile,
        grad_accum=args.grad_accum,
        channels_last=False,
        log_interval=args.log_interval,
        meter_fast_mode=args.meter_fast_mode,
    )
    fast = trainer.train_one_epoch(
        loader,
        criterion,
        steps=args.steps,
        collect_profile=args.collect_profile,
        warmup_steps=args.warmup_steps,
    )

    print(dumps_json({"device": str(device), "baseline": baseline, "spiralfastloop": fast}))


if __name__ == "__main__":
    main()
