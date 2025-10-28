# Decimal Migration Assessment

This note inventories floating-point usage across the SpiralFastLoop codebase
and proposes an incremental plan for evaluating a potential migration toward
`decimal.Decimal` backed numerics.

## 1. Inventory of floating-point touchpoints

| Area | Location | Notes |
| --- | --- | --- |
| AMP policy selection | `spiralfastloop/utils.py` – `get_amp_policy` returns floating-point PyTorch dtypes (`torch.float16`, `torch.float32`, `torch.bfloat16`). | Dtype constants must remain Tensor dtypes; cannot be replaced with `Decimal` without rewriting tensor operations. |
| Throughput metering | `spiralfastloop/utils.py` – `ThroughputMeter` stores elapsed time and throughput as Python floats. | Values originate from `time.perf_counter()` and must stay floats to interoperate with Python’s timing APIs. |
| FastTrainer accumulators | `spiralfastloop/engine.py` – epoch accumulators build `torch.float64` tensors for loss and weights. | Backed by Tensor arithmetic – migrating to `Decimal` would bypass GPU acceleration. |
| Trigger configuration | `spiralfastloop/extras/trigger_mix.py` – `LossStdConfig` uses floats for ratios, epsilons, and weights. | These parameters feed Tensor computations and probability thresholds. |
| Surprisal Sandwich extras | `spiralfastloop/extras/surprisal_sandwich.py` – window fractions (`start_frac`, `end_frac`), penalties (`alpha`, `mu`). | All integrated with PyTorch operations and scheduler thresholds. |
| Benchmarks and examples | Scripts under `examples/` and `scripts/` default to float hyperparameters. | Provide user-facing defaults; swapping to `Decimal` would require repeated conversions back to floats. |

## 2. API compatibility considerations

1. **PyTorch integration:** PyTorch expects native float types (`float`, `torch.dtype`). `Decimal`
   values are not directly supported in tensor constructors, CUDA kernels, or autograd.
2. **Third-party interfaces:** Dataloader worker counts, logging, and JSON/CSV telemetry rely on
   standard floats. Introducing Decimals would require coercion when serializing or interfacing
   with libraries such as `json`, `argparse`, and `torchmetrics`.
3. **Trigger ecosystem:** The trigger hook contract communicates numeric thresholds and weights to
   downstream providers. Requiring `Decimal` would be a backwards-incompatible change for any
   existing integrations.

## 3. Performance implications

* Python `Decimal` is implemented in software and is substantially slower than native IEEE 754
  operations. Replacing the trigger budget arithmetic or throughput meters with `Decimal` would
  introduce measurable CPU overhead in the hottest control-path code.
* GPU kernels cannot consume `Decimal` inputs. Any conversion back to float tensors would forfeit
  the extra precision while still paying the conversion cost.
* Maintaining dual pathways (float and `Decimal`) would multiply the testing and maintenance
  surface for minimal gain because the numerical issues we guard against (rounding residue,
  coefficient-of-variation stability) are already mitigated with explicit epsilons.

## 4. Recommended phased approach

1. **Documentation of constraints (complete):** Track the float-based decisions and tolerances in
   code comments and developer documentation.
2. **Targeted experiments:** If a specific downstream integration demands arbitrary precision,
   wrap the small scalar calculations (e.g., trigger budget bookkeeping) behind an adapter object
   that can optionally use `Decimal` while continuing to emit float values toward PyTorch.
3. **Performance micro-benchmarks:** Extend the new benchmarking harness to time both float- and
   `Decimal`-backed adapters in isolation before committing to broader changes.
4. **Compatibility review:** Should experiments prove viable, design an opt-in configuration flag
   that exposes Decimal arithmetic without altering the existing float-based API defaults.
5. **User communication:** Publish release notes and migration guidance emphasizing that Decimal
   mode is optional, experimental, and carries a performance trade-off compared with the default
   float implementation.

## 5. Decision

Given the tight coupling between SpiralFastLoop and PyTorch tensors, the default runtime should
remain float-based. Decimal support, if ever added, must be opt-in and carefully sandboxed around
scalar control logic to avoid compromising GPU acceleration or API ergonomics.
