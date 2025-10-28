# Normalization Metrics Pipeline

This document describes the new normalization telemetry that monitors how often
fractional-budget bookkeeping is renormalized inside the trigger subsystem.

## 1. What is being measured?

* **Normalization frequency** – how often the fractional buffer is coerced to
  zero because the remainder is below `FRACTION_NORMALIZATION_EPS`.
* **Average magnitude** – mean absolute value before and after normalization so
  we can track whether rounding residue is growing.
* **Context tags** – every normalization call annotates the reason (`budget_buffer`,
  `fractional_credit`, `carryover_credit`).

All data flows through `spiralfastloop.metrics.NormalizationMetricsCollector`,
which exposes:

* `record(before, after, context)` – log an event.
* `summary()` – fast aggregates for alerts.
* `to_timeseries()` – rolling window suitable for dashboards.
* `export_csv(path)` – write the window to disk for scheduled jobs.

## 2. Integration points

* `LossStdTrigger` now accepts an optional `normalization_metrics` argument. The
  default points to the shared `GLOBAL_NORMALIZATION_METRICS` collector, so
  applications get telemetry without changing their wiring.
* Each internal call to `_drop_rounding_noise` emits a context-aware event. This
  includes the main buffer cleanup, fractional credit accumulation, and the
  carry-over pass when the requested injection is clipped by the remaining
  budget.

## 3. Dashboard & reporting workflow

1. Instantiate a collector (or reuse the global singleton) in the training app.
2. Periodically call `collector.export_csv("/tmp/trigger_normalization.csv")`.
3. Point your BI stack (Grafana, Metabase, Google Sheets) at the CSV to chart
   frequency and magnitudes over time.
4. For richer telemetry stacks, adapt `collector.to_timeseries()` into your
   metrics ingestion pipeline (Prometheus push gateway, OpenTelemetry custom
   metrics, etc.).

Example snippet:

```python
from spiralfastloop.extras.trigger_mix import LossStdTrigger
from spiralfastloop.metrics import NormalizationMetricsCollector

collector = NormalizationMetricsCollector(history_limit=2048)
trigger = LossStdTrigger(provider=my_provider, normalization_metrics=collector)

# Later in a monitoring task
print(collector.report())
collector.export_csv("trigger_normalization.csv")
```

## 4. Alerting guardrails

* Alert when the zeroed fraction drops sharply – this may indicate that the
  fractional buffer is consistently accumulating large residues, signalling a
  potential mis-tuned epsilon.
* Track the mean absolute **after** values. Non-zero drift implies that the
  rounding noise threshold is rarely activated and could be tightened.
* Correlate spikes with benchmark results from `scripts/bench_parallel_transactions.py`
  to spot regressions in throughput linked to aggressive trigger injections.

## 5. Future extensions

* Emit Prometheus gauges via an optional helper so the collector can push metrics
  straight into time-series databases.
* Capture per-step identifiers to tie normalization behaviour back to specific
  gradient accumulation phases.
* Combine with trigger budget telemetry (spent vs. total) for a holistic view of
  injection pressure.
