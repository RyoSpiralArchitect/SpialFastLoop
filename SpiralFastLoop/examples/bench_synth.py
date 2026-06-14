"""
Synthetic benchmark to showcase SpiralFastLoop throughput and latency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = best_device()
    samples = 200_000
    feature_dim = 1024
    classes = 64
    inputs = torch.randn(samples, feature_dim)
    weights = torch.randn(feature_dim, classes)
    targets = (inputs @ weights).argmax(dim=1)
    dataset = TensorDataset(inputs, targets)

    loader = dataloader_from_dataset(dataset, batch_size=256, device=device)

    model = nn.Sequential(
        nn.Linear(feature_dim, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, classes),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(device == "cuda"))
    criterion = nn.CrossEntropyLoss()

    trainer = FastTrainer(model, optimizer, grad_accum=2, log_interval=50, compile_mode="reduce-overhead")
    metrics = trainer.train_one_epoch(loader, criterion, steps=300)
    print("BENCH:", metrics)


if __name__ == "__main__":
    main()
