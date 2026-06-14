#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer, recommended_dataloader

class Synth(Dataset):
    def __init__(self, n=50_000, d=128, C=10):
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, C, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, d=128, C=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(),
            nn.Linear(512, C)
        )
    def forward(self, x): return self.net(x)

def plain_loop(loader, model, opt, crit, device, epochs=1):
    model.to(device).train()
    t0 = time.perf_counter()
    steps=0; samples=0; loss_acc=0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step(); opt.zero_grad()
            steps+=1; samples+=xb.shape[0]; loss_acc+=float(loss.detach().cpu())
    dt = time.perf_counter()-t0
    return {"samples_per_sec": samples/max(1e-9,dt), "avg_loss_per_step": loss_acc/max(1,steps), "elapsed_sec": dt}

def parse_args():
    parser = argparse.ArgumentParser(description="Compare a plain PyTorch loop with SpiralFastLoop.")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--collect-profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = Synth(n=args.samples, d=args.feature_dim, C=args.classes)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")
    # baseline
    loader0 = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    m0 = MLP(d=args.feature_dim, C=args.classes); o0 = torch.optim.AdamW(m0.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    base = plain_loop(loader0, m0, o0, crit, device, epochs=1)

    # SpiralFastLoop
    loader1 = recommended_dataloader(
        ds,
        batch_size=args.batch_size,
        device=str(device),
        num_workers=args.workers,
        persistent=args.workers > 0,
    )
    m1 = MLP(d=args.feature_dim, C=args.classes); o1 = torch.optim.AdamW(m1.parameters(), lr=3e-4, fused=torch.cuda.is_available())
    trainer = FastTrainer(
        m1,
        o1,
        device=str(device),
        use_compile=args.compile,
        grad_accum=2,
        channels_last=False,
        log_interval=args.log_interval,
    )
    fast = trainer.train_one_epoch(
        loader1,
        crit,
        steps=args.steps,
        collect_profile=args.collect_profile,
        warmup_steps=args.warmup_steps,
    )

    out = {"device": str(device), "baseline": base, "spiralfastloop": fast}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
