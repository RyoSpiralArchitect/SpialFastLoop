#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer, recommended_dataloader
from scripts.bench_parallel_transactions import (
    non_negative_int_arg,
    positive_float_arg,
    positive_int_arg,
    validate_benchmark_args,
)
from scripts.json_utils import dumps_json


class Synth(Dataset):
    def __init__(self, n: int = 50_000, d: int = 128, classes: int = 10) -> None:
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, classes, (n,))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class MLP(nn.Module):
    def __init__(self, d: int = 128, classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def best_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def adamw(parameters, learning_rate: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=learning_rate, fused=torch.cuda.is_available())


def plain_loop(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    epochs: int = 1,
) -> dict[str, float]:
    model.to(device).train()
    start = time.perf_counter()
    steps = 0
    samples = 0
    loss_acc = 0.0
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            samples += int(inputs.shape[0])
            loss_acc += float(loss.detach().cpu())
    elapsed = time.perf_counter() - start
    return {
        "samples_per_sec": samples / max(1e-9, elapsed),
        "avg_loss_per_step": loss_acc / max(1, steps),
        "elapsed_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a plain PyTorch loop with SpiralFastLoop.")
    parser.add_argument("--samples", type=positive_int_arg, default=50_000)
    parser.add_argument("--feature-dim", type=positive_int_arg, default=128)
    parser.add_argument("--classes", type=positive_int_arg, default=10)
    parser.add_argument("--batch-size", type=positive_int_arg, default=256)
    parser.add_argument("--steps", type=positive_int_arg, default=200)
    parser.add_argument("--warmup-steps", type=non_negative_int_arg, default=0)
    parser.add_argument("--workers", type=non_negative_int_arg, default=2)
    parser.add_argument("--learning-rate", type=positive_float_arg, default=3e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-interval", type=non_negative_int_arg, default=0)
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
    dataset = Synth(n=args.samples, d=args.feature_dim, classes=args.classes)
    device = best_device(args.device)

    baseline_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    baseline_model = MLP(d=args.feature_dim, classes=args.classes)
    baseline_optimizer = adamw(baseline_model.parameters(), args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    baseline = plain_loop(baseline_loader, baseline_model, baseline_optimizer, criterion, device, epochs=1)

    loader = recommended_dataloader(
        dataset,
        batch_size=args.batch_size,
        device=str(device),
        num_workers=args.workers,
        persistent=args.workers > 0,
    )
    model = MLP(d=args.feature_dim, classes=args.classes)
    optimizer = adamw(model.parameters(), args.learning_rate)
    trainer = FastTrainer(
        model,
        optimizer,
        device=str(device),
        use_compile=args.compile,
        grad_accum=2,
        channels_last=False,
        log_interval=args.log_interval,
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
