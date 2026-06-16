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
- Phase profiling for train, evaluate, and predict loops, including data wait,
  transfer, forward, loss, postprocess, metrics, optimizer, and selected module
  drilldowns
- Matrix benchmark diagnostics for cold/post-cold jitter, setup pressure, and
  ranked bottleneck candidates
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

For the shortest dataset-first path, let the trainer build the loader and still
pass training options directly:

```python
metrics = trainer.fit(
    dataset,
    criterion,
    batch_size=256,
    steps=200,
    collect_profile=True,
    warmup_steps=4,
)
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

Evaluation uses the same lightweight phase profiler:

```python
eval_metrics = trainer.evaluate(
    loader,
    criterion,
    steps=50,
    collect_profile=True,
)
print(eval_metrics["profile"]["top_phases"][:3])
```

Prediction keeps the default list return value. Ask for metrics explicitly when
you want inference timings:

```python
predictions, pred_metrics = trainer.predict(
    loader,
    steps=50,
    collect_profile=True,
)
print(pred_metrics["profile"]["top_phases"][:3])
```

If prediction inputs do not expose an inferable batch dimension, the returned
metrics report `unmeasured_steps` and `batch_size_inference_failures` instead of
silently presenting those steps as measured throughput.
Train, evaluation, and prediction metrics all expose `reported_samples_per_sec`;
CUDA/MPS runs also include device memory fields such as `cuda_max_mem_bytes` or
`mps_current_mem_bytes` when the backend exposes them.

Benchmark scripts default to `--log-interval 0` to avoid extra synchronization
in timing runs. Use `--no-compile` when short MPS or CPU smoke tests are
dominated by compile startup. Use `--meter-fast-mode` to skip throughput
tail/window bookkeeping, and `--no-profile-distribution` when profile totals
are enough without per-phase p50/p95/p99/std samples.

Model profiling also records backward gradient-ready positions. Summary fields
such as `profile_backward_grad_ready_top_pct`,
`profile_backward_grad_ready_top_p95_pct_of_parent`, and
`profile_backward_grad_ready_top_p99_pct_of_parent` show whether a selected
module is merely slow on average or lands late in the backward tail. Matrix
output condenses that signal into `bwd_ready_tail(...)`,
`bwd_ready_shape(...)`, and the `backward_ready_tail_position` bottleneck
candidate.

## Example Smoke Runs

The examples emit strict JSON so they can be used as tiny CI or notebook smoke
runs before scaling up the benchmark size.

```bash
PYTHONNOUSERSITE=1 python3 examples/bench_synth.py \
  --device cpu --samples 8 --feature-dim 4 --hidden-dim 8 --classes 2 \
  --batch-size 4 --steps 1 --grad-accum 1 --workers 0 \
  --log-interval 0 --no-compile \
  --meter-fast-mode --collect-profile --no-profile-distribution
```

```bash
PYTHONNOUSERSITE=1 python3 examples/train_resnet.py \
  --dataset fake --device cpu --samples 8 --feature-dim 4 --classes 2 \
  --batch-size 4 --steps 1 --grad-accum 1 --workers 0 \
  --log-interval 0 --no-compile \
  --meter-fast-mode --collect-profile --no-profile-distribution
```

Both commands print top-level `device`, `config`, and `metrics` fields; the
ResNet example also reports the resolved `dataset`. `PYTHONNOUSERSITE=1` keeps
local Python startup hooks from writing extra text to JSON stdout. Increase
`--samples`, `--steps`, and model sizes for throughput runs, add
`--collect-profile --profile-sync --profile-window 512` for synchronized
phase timings with per-phase tail samples, or use
`--dataset cifar10 --download` when CIFAR-10 should be fetched explicitly.

## Transactional Benchmark

```bash
python3 scripts/bench_parallel_transactions.py \
  --device mps --steps 40 --runs 1 --workers 2 \
  --dataset-mode materialized \
  --warmup-steps 4 --no-compile \
  --collect-profile --profile-model --profile-model-depth 1 \
  --profile-model-include 0,2 \
  --json-out reports/bench_parallel_profile.json \
  --summary-out reports/bench_parallel_profile_summary.json
```

The transactional benchmark reports training-only `wall_time_s`, separate
`setup_time_s`, combined `end_to_end_wall_time_s`, and an `init(...)`
breakdown for dataset, loader, model, and compile setup. It also records
`dataset_materialized_bytes` so materialized-data runs remain transparent.
Use `--summary-out` to capture mean/min/max/stddev timing, setup, throughput,
and run-context stats across repeated runs. Aggregate summaries also include
`summary_diagnostic_fields` when metrics contain missing, non-finite, or
out-of-range values, plus best-run omitted-field lists when a selected run has
malformed values that are left out of `best_reported` or `best_end_to_end`.
With `--profile-model`, backward grad-ready rows print `avg_ms@pct`, where
`pct` is the average event position within the parent backward phase. Aggregate
summaries also expose backward grad-ready top timing and position fields for
matrix comparisons, plus earliest/latest/span readiness fields that show how
wide the backward readiness window is. Backward ready p95/p99 parent-position
fields make tail pressure visible when average timing is not enough. Forward
and optimizer drilldowns include
`coverage_pct`, `untracked_pct`, and `overtracked_pct_of_parent` so module-level
hooks can be judged against their parent phase. Human profile summaries include
`calls`, timing `samples`, and bounded `window` counts when distribution samples
are available. Include filters accept either module names such as `layer1` or
printed labels such as `model.layer1`.

```bash
python3 scripts/bench_matrix.py \
  --device cpu --steps 20 --runs 5 --worker-counts 0 \
  --dataset-modes generated,materialized \
  --compile-modes no-compile \
  --collect-profile --profile-model --profile-model-depth 1 \
  --json-out reports/bench_matrix.json \
  --summary-out reports/bench_matrix_summary.json
```
Matrix summaries include per-config means, min/max values, stddev, setup
breakdowns, best steady throughput, and best end-to-end configuration. Repeated
runs split the earliest run from post-cold runs with `matrix_cold_run_*`,
`matrix_post_cold_*`, and `matrix_cold_run_vs_post_cold_*` fields; console
rows show compact `cold(...)` and `post_jitter(...)` fragments so one-time setup
or first-run spikes do not masquerade as steady jitter. Profile summaries
preserve phase fields such as `user_metrics`, `postprocess`, and
`collect_output` when a run emits them. Per-config groups also carry
`summary_diagnostic_fields` for missing, non-finite, or out-of-range inputs and
ranked bottleneck pressure categories for phase share, child hotspots, coverage
gaps, readiness span, and readiness tail.

## License

Apache-2.0. See `LICENSE`.
