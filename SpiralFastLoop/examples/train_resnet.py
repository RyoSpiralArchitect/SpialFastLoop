"""
Minimal example training on CIFAR-10, with a synthetic fallback when offline.
"""

from __future__ import annotations

from typing import Any

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


def get_dataset() -> tuple[Any, int, int, bool]:
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        return dataset, 10, 3 * 32 * 32, True
    except Exception:
        samples = 20_000
        feature_dim = 128
        classes = 5
        inputs = torch.randn(samples, feature_dim)
        weights = torch.randn(feature_dim, classes)
        targets = (inputs @ weights).argmax(dim=1)
        return TensorDataset(inputs, targets), classes, feature_dim, False


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


def main() -> None:
    device = best_device()
    dataset, num_classes, input_dim, image_data = get_dataset()
    loader = dataloader_from_dataset(dataset, batch_size=256, device=device)
    model = build_model(num_classes, input_dim, image_data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = FastTrainer(model, optimizer, grad_accum=2, log_interval=20, compile_mode="reduce-overhead")
    metrics = trainer.train_one_epoch(loader, criterion, steps=200)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
