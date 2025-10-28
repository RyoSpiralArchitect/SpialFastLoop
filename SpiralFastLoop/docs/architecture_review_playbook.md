# Architecture Review Playbook

To keep SpiralFastLoop’s API boundaries sharp and responsibilities well
separated, run a focused architecture review on a regular cadence.

## 1. Cadence and participants

* **Frequency:** Every six weeks, with ad-hoc sessions when major trigger/model
  changes land.
* **Participants:** Core maintainers (loop, trigger, utils), representative
  downstream integrators, and an observability owner.
* **Inputs:** Changelog since last review, outstanding design docs, performance
  telemetry (including normalization metrics and benchmarks).

## 2. Agenda template

1. **Boundary health check**
   * Verify `spiralfastloop.engine` only exposes high-level training APIs and
     does not leak helper utilities meant for internal use.
   * Confirm extras packages (`extras.trigger_mix`, `extras.surprisal_sandwich`)
     remain optional and do not introduce hard dependencies into the core loop.
2. **Responsibility audit**
   * Review recent additions for violations of single-responsibility (e.g.,
     trainer code accumulating logging responsibilities that belong in a
     separate module).
   * Track any data-model changes that would alter trigger hook contracts.
3. **Performance regression scan**
   * Inspect results from `scripts/bench_parallel_transactions.py` to ensure
     throughput hasn’t regressed beyond agreed tolerances.
   * Cross-reference GPU vs. CPU behaviour and identify optimizations to queue.
4. **Action item triage**
   * Capture follow-ups in the engineering tracker with owners and due dates.
   * Decide whether any proposals require a formal design review RFC.

## 3. Checklists

* [ ] Core loop still depends only on PyTorch and stdlib.
* [ ] Trigger hook API unchanged or documented if extended.
* [ ] AMP policy defaults validated against supported PyTorch releases.
* [ ] Telemetry sinks (normalization metrics, benchmarks) producing usable
      reports.
* [ ] Documentation updated for any architectural shifts.

## 4. Outputs

* Meeting notes summarizing decisions and follow-up tasks.
* Updated backlog entries for deeper refactors or performance work.
* Optional architecture scorecard capturing API hygiene, performance, and
  observability readiness.

## 5. Continuous improvement

Use retrospective slots inside the review meeting to refine this playbook –
collect feedback on agenda fit, missing stakeholders, or metrics that would help
expose architectural drift earlier.
