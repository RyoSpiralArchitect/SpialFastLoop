\
"""
Minimal example training on CIFAR-10 (or synthetic fallback if offline).
"""
import os, time
import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms, models

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset

def get_dataset():
    try:
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
        num_classes = 10
        return ds, num_classes
    except Exception:
        # Fallback: synthetic classification
        N, D, C = 20000, 128, 5
        X = torch.randn(N, D)
        W = torch.randn(D, C)
        y = (X @ W).argmax(dim=1)
        ds = TensorDataset(X, y)
        num_classes = C
        return ds, num_classes

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ds, num_classes = get_dataset()
    loader = dataloader_from_dataset(ds, batch_size=256, device=device)

    try:
        model = models.resnet18(num_classes=num_classes)
    except Exception:
        # ultra-minimal MLP fallback
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, num_classes))

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()

    trainer = FastTrainer(model, opt, grad_accum=2, log_interval=20, compile_mode="reduce-overhead")
    metrics = trainer.train_one_epoch(loader, crit, steps=200)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
