"""
Synthetic benchmark to showcase SpiralFastLoop throughput and latency.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    N, D, C = 200_000, 1024, 64
    X = torch.randn(N, D)
    W = torch.randn(D, C)
    y = (X @ W).argmax(dim=1)
    ds = TensorDataset(X, y)

    loader = dataloader_from_dataset(ds, batch_size=256, device=device)

    model = nn.Sequential(
        nn.Linear(D, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, C)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(device == "cuda"))
    crit = nn.CrossEntropyLoss()

    trainer = FastTrainer(model, opt, grad_accum=2, log_interval=50, compile_mode="reduce-overhead")
    m = trainer.train_one_epoch(loader, crit, steps=300)
    print("BENCH:", m)


if __name__ == "__main__":
    main()
