# SpiralFastLoop

Fast, pragmatic training loop utilities for PyTorch on CUDA, MPS, and CPU.

SpiralFastLoop focuses on throughput, stable accumulation, simple device setup,
and opt-in profiling that makes training-loop bottlenecks visible without
rewriting a project around a heavyweight framework.

## Features

- Auto device selection for CUDA, MPS, and CPU
- AMP policy helpers for accelerator-specific dtype choices
- Gradient accumulation with low-overhead metrics
- Optional `torch.compile` with `use_compile=False` for short smoke runs
- Phase profiling for data wait, transfer, forward, loss, backward, optimizer,
  and selected module drilldowns
- Trigger hooks for per-sample-loss driven hard-sample injection

## Install

```bash
pip install -e .
```

For development checks:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from spiralfastloop import FastTrainer
from spiralfastloop.utils import dataloader_from_dataset

inputs = torch.randn(50_000, 128)
targets = torch.randint(0, 10, (50_000,))
dataset = TensorDataset(inputs, targets)
loader = dataloader_from_dataset(dataset, batch_size=256, device="auto")

model = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

trainer = FastTrainer(model, optimizer, grad_accum=2, log_interval=50)
metrics = trainer.train_one_epoch(loader, criterion, steps=200)
print(metrics["reported_samples_per_sec"])
```

## Profiling

```python
metrics = trainer.train_one_epoch(
    loader,
    criterion,
    steps=200,
    collect_profile=True,
    profile_model=True,
    profile_model_depth=1,
    warmup_steps=4,
)

print(metrics["profile"]["top_phases"][:3])
print(metrics["profile"]["phase_breakdowns"]["forward"]["top_children"][:3])
print(metrics["profile"]["phase_events"]["backward_grad_ready"]["top_children"][:3])
```

Benchmark scripts default to `--log-interval 0` to avoid extra synchronization
in timing runs. Use `--no-compile` when short MPS or CPU smoke tests are
dominated by compile startup.

## Transactional Benchmark

```bash
python scripts/bench_parallel_transactions.py \
  --device mps --steps 40 --runs 1 --workers 2 \
  --dataset-mode materialized \
  --warmup-steps 4 --no-compile \
  --collect-profile --profile-model --profile-model-depth 1 \
  --profile-model-include 0,2 \
  --json-out reports/bench_parallel_profile.json
```

The transactional benchmark reports training-only `wall_time_s`, separate
`setup_time_s`, combined `end_to_end_wall_time_s`, and
`dataset_materialized_bytes` so materialized-data runs remain transparent.

```bash
python scripts/bench_matrix.py \
  --device cpu --steps 16 --runs 1 --worker-counts 0 \
  --dataset-modes generated,materialized \
  --compile-modes no-compile \
  --json-out reports/bench_matrix.json
```

## License

Apache-2.0. See `LICENSE`.
