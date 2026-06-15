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
- **Phase profiling**: opt-in timings for data wait, transfer, forward, loss, backward, optimizer, and module-level model drilldowns
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

## Phase profiling
Use `collect_profile=True` when you need a thicker read on training-loop bottlenecks.

```python
metrics = trainer.train_one_epoch(
    loader,
    crit,
    steps=200,
    collect_profile=True,
    profile_model=True,
    profile_model_depth=1,
)

print(metrics["profile"]["top_phases"][:3])
print(metrics["profile"]["phase_breakdowns"]["forward"]["top_children"][:3])
print(metrics["profile"]["phase_breakdowns"]["optimizer"]["top_children"][:3])
print(metrics["profile"]["phase_events"]["backward_grad_ready"]["top_children"][:3])
```

`profile_model=True` adds per-module forward timings under
`phase_breakdowns.forward` and backward gradient-ready timings under
`phase_events.backward_grad_ready`. Use `profile_model_include="layer1,layer4"`
with `profile_model_depth=2` to drill into selected blocks without hooking the
entire model. Benchmark summaries also flatten the forward drilldown into
fields such as `profile_forward_child_count`, `profile_forward_top_avg_ms`,
`profile_forward_top_pct_of_parent`, and `profile_forward_top_p95_ms`, plus
backward gradient-ready fields such as `profile_backward_grad_ready_top_avg_ms` and
`profile_backward_grad_ready_top_pct`. Optimizer drilldowns are available under
`phase_breakdowns.optimizer` and flattened as `profile_optimizer_top_avg_ms`,
`profile_optimizer_top_pct_of_parent`, `profile_optimizer_top_p95_ms`, and
related tracked/untracked fields. Benchmark console drilldown lines include
top-child `avg` and `p95` timings when distribution tracking is enabled.
The top-level phases include `data_wait`, so loader stalls can be separated from
compute time, and their distribution windows are flattened as fields like
`profile_forward_p95_ms`, `profile_forward_p99_ms`, and
`profile_forward_std_ms` when distribution tracking is enabled. Benchmark
output surfaces those tails in the `phases:` line, and matrix output also
surfaces compact summaries such as `fwd_tail(p95=...)`, `bwd_tail(p99=...)`,
and `opt_tail(std=...)`. Aggregate benchmark summaries also include ranked
`profile_bottleneck_candidates` with the strongest phase, top-child, untracked,
and backward-readiness-span signals normalized into profile-percentage scores;
each candidate carries a `high`/`medium`/`low` severity label, reason,
next-step hint, rank, and returned/omitted candidate counts when the list is
capped. Summaries also expose `profile_bottleneck_top_candidate`,
`profile_bottleneck_severity_counts`, and
`profile_bottleneck_category_summary` for dashboards that compare phase-share,
coverage-gap, child-hotspot, and readiness-span pressure directly. Matrix
summaries surface the same signal as compact `hotspot=...` and `pressure(...)`
fragments, including the top hotspot severity, category-level severity suffixes
such as `[high]`, and `severity_counts(...)`. The transactional benchmark
prints the same severity-aware `Bottleneck:` line before the aggregate JSON when
candidates are available.
The ResNet drilldown script prints the same top-phase tail fields. Throughput
summaries include `p99_s` and `std_batch_s`. Set `profile_sync=True` only when
you need stricter accelerator timings; it synchronizes around profiled regions
and slows the run down. Use `--no-profile-distribution` in benchmark scripts to
keep totals while skipping percentile windows when profiler overhead matters.

For cold-start versus steady-state reads, pass `warmup_steps=N`. The first `N`
completed training steps are still executed and measured, but they are reported
separately as `cold_start_time_s`, `warmup_samples_per_sec`, and
`warmup_avg_loss`. Post-warmup values are exposed as `steady_samples_per_sec`,
`steady_p99_s`, and `steady_avg_loss`; `reported_samples_per_sec` uses steady
throughput when steady steps exist. `compile_init_time_s` captures the immediate
`torch.compile` wrapper setup cost, while lazy first-forward compilation shows up
inside the warmup/cold-start window.

For short MPS or smoke-test runs where compile startup dominates, pass
`--no-compile` in the benchmark scripts or `FastTrainer(..., use_compile=False)`.
Benchmark scripts also default to `--log-interval 0` so step logs do not force
extra host/device synchronization; pass a positive `--log-interval` when you want
live step output.

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

## Benchmarks
| Device | Baseline (plain loop) | SpiralFastLoop | Speedup |
|-------:|-----------------------:|---------------:|--------:|
| GTX 1650 (CUDA) | 450 samples/s | 610 samples/s | 1.35× |
| M4 (MPS)    | 520 samples/s | 780 samples/s | 1.50× |

> Notes: batch=256; synthetic MLP; PyTorch 2.3; CUDA driver XX; macOS 14.x.

Run local synthetic bench:
```bash
python examples/bench_synth.py
```

Profile the transactional benchmark:
```bash
python scripts/bench_parallel_transactions.py \
  --device mps --steps 40 --runs 1 --workers 2 \
  --dataset-mode materialized \
  --warmup-steps 4 --no-compile \
  --collect-profile --profile-model --profile-model-depth 1 \
  --profile-model-include 0,2 \
  --json-out reports/bench_parallel_profile.json \
  --summary-out reports/bench_parallel_profile_summary.json
```

The transactional benchmark reports training-only `wall_time_s`, separate
`setup_time_s`, combined `end_to_end_wall_time_s`, and
`dataset_materialized_bytes` so materialized-data runs remain transparent.
Use `--summary-out` to capture mean/min/max/stddev timing and throughput
stats across repeated runs.

Run a compact comparison matrix:
```bash
python scripts/bench_matrix.py \
  --device cpu --steps 16 --runs 1 --worker-counts 0 \
  --dataset-modes generated,materialized \
  --compile-modes no-compile \
  --json-out reports/bench_matrix.json \
  --summary-out reports/bench_matrix_summary.json
```
Matrix summaries include per-config means, min/max values, stddev, best steady
throughput, and best end-to-end configuration.

Drill into ResNet blocks:
```bash
python scripts/profile_resnet_drilldown.py \
  --device mps --dataset fake --steps 12 \
  --warmup-steps 2 \
  --profile-model-include layer1,layer4 --profile-model-depth 2 \
  --json-out reports/resnet_layer14_drilldown.json
```

## Examples
- `examples/train_resnet.py` — CIFAR-10 (falls back to synthetic offline)
- `examples/bench_synth.py` — synthetic speed test
- `examples/sr_generate_demo.py` — Surprisal Sandwich generation (HF)
- `scripts/bench_parallel_transactions.py` — synthetic transactional benchmark with optional phase profiling
- `scripts/profile_resnet_drilldown.py` — ResNet block-level forward/backward profiling

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

### Trigger tolerances

`spiralfastloop.extras.trigger_mix` exposes two module-level tolerances that keep
its floating-point bookkeeping predictable:

- `FRACTION_NORMALIZATION_EPS = 1e-12` drops rounding residue when fractional
  sample budget credits are accumulated across steps. The value is unit-less and
  roughly corresponds to "less than a trillionth of a sample".
- `COEFVAR_STABILIZER = 1e-8` ensures the coefficient-of-variation check stays
  finite even when the mean per-sample loss is numerically close to zero. This
  is effectively "0.00000001 loss" and is safe for typical cross-entropy scales.

Override them if you operate with vastly different loss magnitudes.

## License
Apache 2.0 License (see `LICENSE`).

---

Made with 🌀 by Ryō ∴ SpiralArchitect and SpiralReality — *Full-stack AI Architect / Research Engineer*.


---

## Legal / Credits
- © 2025 Ryō. Code licensed under **Apache 2.0** (see LICENSE). See **COPYRIGHT** and **TRADEMARKS.md** for name/branding terms.
- This project may interact with third-party models/libraries; see **NOTICE** for their licenses.
- How to cite: see **CITATION.cff**.
