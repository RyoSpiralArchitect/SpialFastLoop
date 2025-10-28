# SpiralFastLoop

> Fast, pragmatic training loop template for PyTorch — **CUDA / MPS / CPU**.

**Status:** v0.1.1 • License: Apache-2.0 • Python ≥ 3.9 • PyTorch ≥ 2.1

---

## Overview

**SpiralFastLoop** delivers a high-throughput, batteries-included training loop for PyTorch. It wraps repeatable performance
tweaks—automatic device selection, mixed precision, gradient accumulation, and optional `torch.compile`—in a concise API so you
can focus on modeling instead of plumbing. When you need more novelty, wire in the optional **Surprise→Repair (Surprisal
Sandwich)** hook to blend procedurally perturbed samples back into the batch.

### Feature highlights

- **Auto device** fallback: CUDA → MPS → CPU
- **AMP aware** with `GradScaler` on CUDA and bf16/fp16 on supported backends
- **Gradient accumulation** with careful `.item()` avoidance and `zero_grad(set_to_none=True)`
- **Data transfer tweaks** for non-blocking host→device copies
- **Optional trigger hook** to inject hard samples (e.g., Surprisal Sandwich)

## Installation

```bash
git clone https://github.com/your-user/SpiralFastLoop.git
cd SpiralFastLoop
pip install -e .
```

> Tip: create a virtual environment (`python -m venv .venv && source .venv/bin/activate`) before installing editable deps.

## Quickstart

```python
from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

X = torch.randn(50_000, 128)
y = torch.randint(0, 10, (50_000,))
ds = TensorDataset(X, y)
loader = dataloader_from_dataset(ds, batch_size=256, device="auto")

model = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 10))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
crit = nn.CrossEntropyLoss()

trainer = FastTrainer(model, opt, grad_accum=2, log_interval=50)
metrics = trainer.train_one_epoch(loader, crit, steps=200)
print(metrics)
```

## Surprise→Repair (Surprisal Sandwich)

**Goal:** inject mid-sequence *surprise* by soft-penalising the most likely tokens, then **repair** fluency toward the end.
Use the provided trigger mixins to curate or synthesize high-variance samples once the per-sample loss spikes.

- Middle window (≈45–70% of new tokens): apply an **AntiTopK** penalty (−α)
- Tail window (final 30%): optional **Coherence boost** using a lightweight LM (μ)
- Hook into training via `trigger_hook` with a per-sample criterion (`reduction="none"`)

Run the demo:

```bash
python examples/sr_generate_demo.py
```

## Benchmarks (example figures)

| Device          | Baseline (plain loop) | SpiralFastLoop | Speedup |
| --------------: | --------------------: | -------------: | ------: |
| GTX 1650 (CUDA) | 450 samples/s         | 610 samples/s  | 1.35×   |
| M4 (MPS)        | 520 samples/s         | 780 samples/s  | 1.50×   |

> Notes: batch=256; synthetic MLP; PyTorch 2.3; CUDA driver XX; macOS 14.x. Re-run `python examples/bench_synth.py` with your
> hardware to update these numbers.

## Local development

### Tooling

The project ships a [`.pre-commit-config.yaml`](./.pre-commit-config.yaml) that runs `black`, `isort`, and `flake8` for a
consistent style baseline.

```bash
pip install -e .[extras]  # optional extras
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Useful scripts

- `python examples/train_resnet.py` — CIFAR-10 training (falls back to synthetic data offline)
- `python examples/bench_synth.py` — synthetic throughput benchmark
- `python examples/sr_generate_demo.py` — Surprisal Sandwich text sampling demo

## Trigger hook API

Enable per-sample control by passing a criterion with `reduction='none'` and registering a `trigger_hook`:

```python
from spiralfastloop.extras.trigger_mix import LossStdTrigger, LossStdConfig


def my_provider(k, device, ctx):
    # Return k extra hard samples (inputs, targets)
    loss_vec = ctx["loss_vec"]
    inputs, targets = ctx["inputs"], ctx["targets"]
    idx = loss_vec.topk(min(k, loss_vec.numel())).indices
    return inputs[idx], targets[idx]


trigger = LossStdTrigger(my_provider, LossStdConfig(std_threshold=0.15, inject_ratio=0.08))
trainer = FastTrainer(model, opt, trigger_hook=trigger)
```

> **Batch structure requirements:** The tensors (or nested structures) returned by the trigger must mirror the original batch
> exactly. SpiralFastLoop concatenates the original and injected batches element-wise before recomputing the forward pass. If you
> provide optional sample weights, supply a 1D tensor that matches the concatenated batch length and sums to a positive value.

### Trigger tolerances

`spiralfastloop.extras.trigger_mix` exposes two module-level tolerances that keep its floating-point bookkeeping predictable:

- `FRACTION_NORMALIZATION_EPS = 1e-12` drops rounding residue when fractional sample budget credits are accumulated.
- `COEFVAR_STABILIZER = 1e-8` keeps coefficient-of-variation checks finite even when the mean per-sample loss is near zero.

Override them if you operate with vastly different loss magnitudes.

## License

Apache 2.0 License (see [`LICENSE`](./LICENSE)).

---

Made with 🌀 by Ryō ∴ SpiralArchitect and SpiralReality — *Full-stack AI Architect / Research Engineer*.

---

## Legal / Credits

- © 2025 Ryō. Code licensed under **Apache 2.0** (see LICENSE). See **COPYRIGHT** and **TRADEMARKS.md** for name/branding terms.
- This project may interact with third-party models/libraries; see **NOTICE** for their licenses.
- How to cite: see **CITATION.cff**.
