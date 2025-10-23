# SpiralFastLoop

> Fast, pragmatic training loop template for PyTorch — **CUDA / MPS / CPU**.

**Status:** v0.1.1 • License: Apache-2.0 • Python ≥ 3.9 • PyTorch ≥ 2.1


**SpiralFastLoop** is a fast, practical PyTorch training loop template focused on *throughput, stability, and simplicity*.  
It also ships an optional **Surprise→Repair (Surprisal Sandwich)** mechanism to inject *novelty* during training to counteract gradient "over-smoothing" caused by large effective batch sizes (via gradient accumulation).

## ✨ Features
- **Auto device**: CUDA / MPS / CPU
- **AMP**: auto-select bf16/fp16 (CUDA/MPS) with GradScaler on CUDA
- **Gradient Accumulation**: stable big-batch effect on small VRAM
- **Data transfer tweaks**: non_blocking transfers; pin_memory recommended
- **`torch.compile` (best-effort)**: reduces Python overhead
- **Sync reduction**: `.item()` minimized; `zero_grad(set_to_none=True)`
- **Trigger hook (optional)**: per-sample loss driven injection (e.g., "Surprise→Repair" text augmentation)

## Install (local)
```bash
pip install -e .
```

## Quickstart
```python
from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset
import torch, torch.nn as nn
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
**Goal:** Inject *surprise* mid-sentence by penalizing the most likely tokens, then **repair** coherence near the end.
Use it to create *novel but coherent* samples and mix them into training (loss-std triggered) to avoid over-smoothed gradients.

- Middle window (e.g., 45–70% of new tokens): **AntiTopK** penalty (−α)
- Tail (final 30%): optional **Coherence boost** via a tiny LM (μ)
- Hook into training via `trigger_hook` + per-sample loss

HF demo:
```bash
python examples/sr_generate_demo.py
```

## Benchmarks (example — fill with your measurements)
| Device | Baseline (plain loop) | SpiralFastLoop | Speedup |
|-------:|-----------------------:|---------------:|--------:|
| GTX 1650 (CUDA) | 450 samples/s | 610 samples/s | 1.35× |
| M4 (MPS)    | 520 samples/s | 780 samples/s | 1.50× |

> Notes: batch=256; synthetic MLP; PyTorch 2.3; CUDA driver XX; macOS 14.x.

Run local synthetic bench:
```bash
python examples/bench_synth.py
```

## Examples
- `examples/train_resnet.py` — CIFAR-10 (falls back to synthetic offline)
- `examples/bench_synth.py` — synthetic speed test
- `examples/sr_generate_demo.py` — Surprisal Sandwich generation (HF)

## Trigger hook API
To enable per-sample control, pass a criterion with `reduction='none'` and a `trigger_hook`:

```python
from spiralfastloop.extras.trigger_mix import LossStdTrigger, LossStdConfig

def my_provider(k, device, ctx):
    # Return k extra hard samples (inputs, targets)
    # Example: reuse batch subset with highest loss, or generate on-the-fly.
    loss_vec = ctx["loss_vec"]
    inputs, targets = ctx["inputs"], ctx["targets"]
    idx = loss_vec.topk(min(k, loss_vec.numel())).indices
    return inputs[idx], targets[idx]

trigger = LossStdTrigger(my_provider, LossStdConfig(std_threshold=0.15, inject_ratio=0.08))
trainer = FastTrainer(model, opt, trigger_hook=trigger)
```

> **Batch structure requirements:** The tensors (or nested structures of tensors) returned by the trigger must mirror the original batch exactly (matching keys for dicts and positional elements for tuples/lists). SpiralFastLoop concatenates the original and injected batches element-wise before recomputing the forward pass. If you provide optional sample weights, supply a 1D tensor that matches the concatenated batch length and sums to a positive value.

## License
Apache 2.0 License (see `LICENSE`).

---

Made with 🌀 by Ryō ∴ SpiralArchitect and SpiralReality — *Full-stack AI Architect / Research Engineer*.


---

## Legal / Credits
- © 2025 Ryō. Code licensed under **Apache 2.0** (see LICENSE). See **COPYRIGHT** and **TRADEMARKS.md** for name/branding terms.
- This project may interact with third-party models/libraries; see **NOTICE** for their licenses.
- How to cite: see **CITATION.cff**.
