import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
import spiralfastloop.engine as engine
import spiralfastloop.utils as utils_mod
from spiralfastloop.engine import (
    TriggerResult,
    _add_profile_phase_metrics,
    _infer_batch_size,
    _try_infer_batch_size_with_reason,
)
from spiralfastloop.utils import PhaseProfiler


class _FailingFloat:
    def __float__(self) -> float:
        raise RuntimeError("float conversion failed")


class BrokenStrError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("string conversion failed")


def _make_supervised_components() -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    nn.Module,
    torch.optim.Optimizer,
]:
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    return loader, model, optimizer


def _capture_engine_phase_profilers(monkeypatch: pytest.MonkeyPatch) -> list[PhaseProfiler]:
    profilers: list[PhaseProfiler] = []

    class CapturingPhaseProfiler(PhaseProfiler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            profilers.append(self)

    monkeypatch.setattr(engine, "PhaseProfiler", CapturingPhaseProfiler)
    return profilers


class _FailingLoader:
    def __iter__(self) -> "_FailingLoader":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("loader boom")


class _CapturingLogger:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, object], str]] = []

    def log_metrics(
        self,
        stage: str,
        metrics: dict[str, object],
        *,
        mode: str = "step",
        **_: object,
    ) -> None:
        self.rows.append((stage, metrics, mode))


def _single_error_metrics(logger: _CapturingLogger, stage: str) -> dict[str, object]:
    assert len(logger.rows) == 1
    row_stage, metrics, mode = logger.rows[0]
    assert row_stage == stage
    assert mode == "error"
    return metrics


def _make_logged_cpu_trainer(
    logger: _CapturingLogger,
    *,
    trigger_hook: object = None,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    torch.optim.Optimizer,
    FastTrainer,
]:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        trigger_hook=trigger_hook,  # type: ignore[arg-type]
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )
    return loader, optimizer, trainer


def _assert_train_failure_metrics(
    logger: _CapturingLogger,
    *,
    stage: str,
    last_error: str,
    optimizer_steps: int = 0,
    exact_error: bool = True,
) -> dict[str, object]:
    metrics = _single_error_metrics(logger, "train")
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == optimizer_steps
    assert metrics["samples"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == stage
    if exact_error:
        assert metrics["train_failure_last_error"] == last_error
    else:
        assert last_error in metrics["train_failure_last_error"]
    return metrics


def _assert_eval_failure_metrics(
    logger: _CapturingLogger,
    *,
    stage: str,
    last_error: str,
) -> dict[str, object]:
    metrics = _single_error_metrics(logger, "eval")
    assert metrics["steps"] == 1
    assert metrics["measured_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["eval_failed"] is True
    assert metrics["eval_failure_stage"] == stage
    assert metrics["eval_failure_last_error"] == last_error
    return metrics


def _assert_predict_failure_metrics(
    logger: _CapturingLogger,
    *,
    stage: str,
    last_error: str,
) -> dict[str, object]:
    metrics = _single_error_metrics(logger, "predict")
    assert metrics["steps"] == 1
    assert metrics["measured_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["predict_failed"] is True
    assert metrics["predict_failure_stage"] == stage
    assert metrics["predict_failure_last_error"] == last_error
    return metrics


def _fail_first_meter_record(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    original_meter = engine.ThroughputMeter
    meters: list[object] = []

    class FailingRecordMeter(original_meter):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            meters.append(self)

        def record(self, duration: float, batch_size: int) -> None:
            if self is meters[0]:
                raise RuntimeError(message)
            super().record(duration, batch_size)

    monkeypatch.setattr(engine, "ThroughputMeter", FailingRecordMeter)


def _fail_batch_duration_validation(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    original_validator = engine._non_negative_finite_float_setting

    def validate(value: object, name: str) -> float:
        if name == "batch_duration_s":
            raise ValueError(message)
        return original_validator(value, name)

    monkeypatch.setattr(engine, "_non_negative_finite_float_setting", validate)


@pytest.mark.parametrize(
    ("batch", "reason", "match"),
    [
        (torch.tensor(1.0), "tensor_scalar", "scalar tensor"),
        (torch.empty(0, 4), "tensor_empty", "non-zero"),
        ({}, "mapping_empty", "mapping input"),
        ({"x": torch.randn(2, 4), "y": torch.randn(3, 4)}, "mapping_inconsistent", "Inconsistent"),
        ({"x": torch.randn(2, 4), "y": torch.tensor(1.0)}, "mapping_inconsistent", "Inconsistent"),
        ([], "sequence_empty", "Sequence batch dimension"),
        ((torch.randn(2, 4), torch.randn(3, 4)), "sequence_inconsistent", "Inconsistent"),
        ((torch.randn(2, 4), object()), "sequence_inconsistent", "Inconsistent"),
        (None, "none", "None"),
        (object(), "unsupported_type", "Unsupported batch structure"),
    ],
)
def test_batch_size_inference_reports_failure_reasons(
    batch: object,
    reason: str,
    match: str,
) -> None:
    batch_size, failure_reason = _try_infer_batch_size_with_reason(batch)

    assert batch_size is None
    assert failure_reason == reason
    with pytest.raises((TypeError, ValueError), match=match):
        _infer_batch_size(batch)


def test_batch_size_inference_preserves_sample_sequence_length_fallback() -> None:
    batch_size, failure_reason = _try_infer_batch_size_with_reason([object(), object(), object()])

    assert batch_size == 3
    assert failure_reason == ""
    assert _infer_batch_size([object(), object(), object()]) == 3


@pytest.mark.parametrize("window", [0, -1, 1.5, "8", True])
def test_phase_profiler_rejects_invalid_window_values(window: object) -> None:
    with pytest.raises(ValueError, match="window"):
        PhaseProfiler(enabled=True, window=window)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"enabled": 1}, "enabled"),
        ({"enabled": "true"}, "enabled"),
        ({"sync": 0}, "sync"),
        ({"sync": "false"}, "sync"),
        ({"track_distribution": 1}, "track_distribution"),
        ({"track_distribution": "false"}, "track_distribution"),
    ],
)
def test_phase_profiler_rejects_invalid_boolean_settings(
    kwargs: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        PhaseProfiler(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("device", [None, True, 1, "", "   ", object()])
def test_phase_profiler_rejects_invalid_device_settings(device: object) -> None:
    with pytest.raises(ValueError, match="device"):
        PhaseProfiler(device=device)  # type: ignore[arg-type]


def test_phase_profiler_normalizes_device_setting() -> None:
    profiler = PhaseProfiler(device=" cpu ")

    assert profiler.device == "cpu"


def test_phase_profiler_respects_exact_distribution_window() -> None:
    profiler = PhaseProfiler(enabled=True, window=1)

    profiler._record("forward", 0.001)
    profiler._record("forward", 0.003)
    profile = profiler.summary()

    forward = profile["phases"]["forward"]
    assert forward["calls"] == 2
    assert forward["sample_count"] == 1
    assert forward["p50_ms"] == pytest.approx(3.0)


def test_phase_profiler_reports_open_timers_in_summary() -> None:
    profiler = PhaseProfiler(enabled=True)

    profiler.start("forward")
    profiler.start_detail("forward", "model.0")
    profiler.start_detail("forward", "model.0")

    profile = profiler.summary()

    assert profile["profile_open_phase_count"] == 1
    assert profile["profile_open_detail_count"] == 2
    assert profile["profile_open_phases"] == ["forward"]
    assert profile["profile_open_details"] == [
        {"parent": "forward", "name": "model.0", "count": 2},
    ]


def test_phase_profiler_reports_zero_open_timers_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([1.0, 1.01, 1.02, 1.03])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start("forward")
    profiler.stop("forward")
    profiler.start_detail("forward", "model.0")
    profiler.stop_detail("forward", "model.0")

    profile = profiler.summary()

    assert profile["profile_open_phase_count"] == 0
    assert profile["profile_open_detail_count"] == 0
    assert profile["profile_open_phases"] == []
    assert profile["profile_open_details"] == []


def test_phase_profiler_reports_non_negative_untracked_breakdown_time() -> None:
    profiler = PhaseProfiler(enabled=True)

    profiler._record("forward", 0.10)
    profiler._record_detail("forward", "model.0", 0.03)
    profiler._record_detail("forward", "model.2", 0.04)

    breakdown = profiler.summary()["phase_breakdowns"]["forward"]

    assert breakdown["parent_total_s"] == pytest.approx(0.10)
    assert breakdown["tracked_s"] == pytest.approx(0.07)
    assert breakdown["untracked_s"] == pytest.approx(0.03)
    assert breakdown["overtracked_s"] == pytest.approx(0.0)


def test_phase_profiler_reports_overtracked_breakdown_time() -> None:
    profiler = PhaseProfiler(enabled=True)

    profiler._record("forward", 0.10)
    profiler._record_detail("forward", "model.0", 0.08)
    profiler._record_detail("forward", "model.2", 0.05)

    breakdown = profiler.summary()["phase_breakdowns"]["forward"]

    assert breakdown["parent_total_s"] == pytest.approx(0.10)
    assert breakdown["tracked_s"] == pytest.approx(0.13)
    assert breakdown["untracked_s"] == pytest.approx(0.0)
    assert breakdown["overtracked_s"] == pytest.approx(0.03)


@pytest.mark.parametrize("seconds", [-0.1, float("nan"), float("inf"), True, object()])
def test_phase_profiler_rejects_invalid_phase_durations(seconds: object) -> None:
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match="seconds"):
        profiler._record("forward", seconds)  # type: ignore[arg-type]

    assert profiler.summary()["phases"] == {}


@pytest.mark.parametrize("seconds", [-0.1, float("nan"), float("-inf"), True, object()])
def test_phase_profiler_rejects_invalid_detail_durations(seconds: object) -> None:
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match="seconds"):
        profiler._record_detail("optimizer", "step", seconds)  # type: ignore[arg-type]

    profile = profiler.summary()
    assert profile["phase_breakdowns"] == {}


@pytest.mark.parametrize("seconds", [-0.1, float("nan"), float("inf"), True, object()])
def test_phase_profiler_rejects_invalid_event_durations(seconds: object) -> None:
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match="seconds"):
        profiler._record_event("backward_grad_ready", "model.0", seconds)  # type: ignore[arg-type]

    profile = profiler.summary()
    assert profile["phase_events"] == {}


def test_phase_profiler_stop_rejects_invalid_elapsed_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start("forward")
    profile_before = profiler.summary()

    with pytest.raises(ValueError, match="seconds"):
        profiler.stop("forward")

    assert profiler._starts == {"forward": 2.0}
    assert profiler.summary() == profile_before


def test_phase_profiler_stop_detail_rejects_invalid_elapsed_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start_detail("forward", "model.0")
    profile_before = profiler.summary()

    with pytest.raises(ValueError, match="seconds"):
        profiler.stop_detail("forward", "model.0")

    assert profiler._detail_starts == {("forward", "model.0"): [2.0]}
    assert profiler.summary() == profile_before


def test_phase_profiler_event_rejects_invalid_elapsed_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start("backward")
    profile_before = profiler.summary()

    with pytest.raises(ValueError, match="seconds"):
        profiler.record_event_since_start("backward", "backward_grad_ready", "model.0")

    assert profiler._starts == {"backward": 2.0}
    assert profiler.summary() == profile_before


def test_phase_profiler_stop_preserves_body_exception_on_invalid_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start("forward")
    profile_before = profiler.summary()

    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            profiler.stop("forward")

    assert profiler._starts == {}
    profile_after = profiler.summary()
    assert profile_after["phases"] == profile_before["phases"]
    assert profile_after["profile_open_phase_count"] == 0
    assert profile_after["profile_open_phases"] == []


def test_phase_profiler_stop_detail_preserves_body_exception_on_invalid_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start_detail("forward", "model.0")
    profile_before = profiler.summary()

    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            profiler.stop_detail("forward", "model.0")

    assert profiler._detail_starts == {}
    profile_after = profiler.summary()
    assert profile_after["phase_breakdowns"] == profile_before["phase_breakdowns"]
    assert profile_after["profile_open_detail_count"] == 0
    assert profile_after["profile_open_details"] == []


def test_phase_profiler_event_preserves_body_exception_on_invalid_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start("backward")
    profile_before = profiler.summary()

    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            profiler.record_event_since_start("backward", "backward_grad_ready", "model.0")

    assert profiler._starts == {"backward": 2.0}
    assert profiler.summary() == profile_before


def test_phase_profiler_events_report_parent_relative_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([1.0, 1.04, 1.10])
    monkeypatch.setattr(utils_mod.time, "perf_counter", lambda: next(timestamps))
    profiler = PhaseProfiler(enabled=True)

    profiler.start("backward")
    profiler.record_event_since_start("backward", "backward_grad_ready", "model.0")
    profiler.stop("backward")

    event_group = profiler.summary()["phase_events"]["backward_grad_ready"]
    child = event_group["children"]["model.0"]
    top_child = event_group["top_children"][0]

    assert event_group["parent"] == "backward"
    assert event_group["parent_total_s"] == pytest.approx(0.10)
    assert event_group["parent_avg_ms"] == pytest.approx(100.0)
    assert child["avg_ms"] == pytest.approx(40.0)
    assert child["avg_pct_of_parent"] == pytest.approx(40.0)
    assert top_child["avg_pct_of_parent"] == pytest.approx(40.0)


@pytest.mark.parametrize("name", [None, True, "", "   ", object()])
def test_phase_profiler_rejects_invalid_phase_names_without_mutating_state(name: object) -> None:
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match="name"):
        profiler.start(name)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        profiler.stop(name)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        profiler.cancel(name)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        profiler._record(name, 0.001)  # type: ignore[arg-type]

    profile = profiler.summary()
    assert profiler._starts == {}
    assert profile["phases"] == {}


@pytest.mark.parametrize(
    ("parent", "name", "match"),
    [
        (None, "module", "parent"),
        (True, "module", "parent"),
        ("", "module", "parent"),
        ("   ", "module", "parent"),
        ("forward", None, "name"),
        ("forward", True, "name"),
        ("forward", "", "name"),
        ("forward", "   ", "name"),
    ],
)
def test_phase_profiler_rejects_invalid_detail_names_without_mutating_state(
    parent: object,
    name: object,
    match: str,
) -> None:
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match=match):
        profiler.start_detail(parent, name)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        profiler.stop_detail(parent, name)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        profiler._record_detail(parent, name, 0.001)  # type: ignore[arg-type]

    profile = profiler.summary()
    assert profiler._detail_starts == {}
    assert profile["phase_breakdowns"] == {}


@pytest.mark.parametrize(
    ("parent", "group", "name", "match"),
    [
        (None, "grad", "module", "parent"),
        ("backward", None, "module", "group"),
        ("backward", "", "module", "group"),
        ("backward", "grad", None, "name"),
        ("backward", "grad", "", "name"),
        ("backward", "grad", True, "name"),
    ],
)
def test_phase_profiler_rejects_invalid_event_names_without_mutating_state(
    parent: object,
    group: object,
    name: object,
    match: str,
) -> None:
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match=match):
        profiler.record_event_since_start(parent, group, name)  # type: ignore[arg-type]

    profile = profiler.summary()
    assert profile["phase_events"] == {}


def test_disabled_phase_profiler_ignores_invalid_names() -> None:
    profiler = PhaseProfiler(enabled=False)

    profiler.start(None)  # type: ignore[arg-type]
    profiler.stop(True)  # type: ignore[arg-type]
    profiler.cancel("")  # type: ignore[arg-type]
    profiler.start_detail(None, True)  # type: ignore[arg-type]
    profiler.stop_detail("", object())  # type: ignore[arg-type]
    profiler.record_event_since_start(None, "", True)  # type: ignore[arg-type]

    assert profiler.summary() == {}


@pytest.mark.parametrize(
    ("trainer_kwargs", "match"),
    [
        ({"grad_accum": 0}, "grad_accum"),
        ({"grad_accum": -1}, "grad_accum"),
        ({"grad_accum": 1.5}, "grad_accum"),
        ({"grad_accum": "2"}, "grad_accum"),
        ({"grad_accum": True}, "grad_accum"),
        ({"log_interval": -1}, "log_interval"),
        ({"log_interval": 1.5}, "log_interval"),
        ({"clip_grad_norm": -0.1}, "clip_grad_norm"),
        ({"clip_grad_norm": float("nan")}, "clip_grad_norm"),
    ],
)
def test_fast_trainer_rejects_invalid_numeric_settings(
    trainer_kwargs: dict[str, object],
    match: str,
) -> None:
    _loader, model, optimizer = _make_supervised_components()

    with pytest.raises(ValueError, match=match):
        FastTrainer(
            model,
            optimizer,
            device="cpu",
            use_amp=False,
            use_compile=False,
            **trainer_kwargs,
        )


@pytest.mark.parametrize(
    ("trainer_kwargs", "match"),
    [
        ({"use_amp": "false"}, "use_amp"),
        ({"use_compile": "false"}, "use_compile"),
        ({"channels_last": 1}, "channels_last"),
        ({"distributed": "true"}, "distributed"),
        ({"log_on_rank0": 1}, "log_on_rank0"),
        ({"enable_tf32": "true"}, "enable_tf32"),
        ({"cudnn_benchmark": 1}, "cudnn_benchmark"),
        ({"reduced_precision_reduction": "false"}, "reduced_precision_reduction"),
        ({"enable_flash_sdp": "false"}, "enable_flash_sdp"),
        ({"enable_mem_efficient_sdp": 0}, "enable_mem_efficient_sdp"),
        ({"enable_math_sdp": "true"}, "enable_math_sdp"),
        ({"meter_fast_mode": "true"}, "meter_fast_mode"),
    ],
)
def test_fast_trainer_rejects_invalid_boolean_settings(
    trainer_kwargs: dict[str, object],
    match: str,
) -> None:
    _loader, model, optimizer = _make_supervised_components()
    kwargs = {"device": "cpu", "use_amp": False, "use_compile": False, **trainer_kwargs}

    with pytest.raises(ValueError, match=match):
        FastTrainer(model, optimizer, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_value", [None, True, 1, object()])
def test_fast_trainer_rejects_invalid_models(model_value: object) -> None:
    _loader, model, optimizer = _make_supervised_components()

    with pytest.raises(ValueError, match="model"):
        FastTrainer(
            model_value,  # type: ignore[arg-type]
            optimizer,
            device="cpu",
            use_amp=False,
            use_compile=False,
        )


@pytest.mark.parametrize("optimizer_value", [None, True, 1, object()])
def test_fast_trainer_rejects_invalid_optimizers(optimizer_value: object) -> None:
    _loader, model, _optimizer = _make_supervised_components()

    with pytest.raises(ValueError, match="optimizer"):
        FastTrainer(
            model,
            optimizer_value,  # type: ignore[arg-type]
            device="cpu",
            use_amp=False,
            use_compile=False,
        )


@pytest.mark.parametrize(
    ("trainer_kwargs", "match"),
    [
        ({"device": True}, "device"),
        ({"device": 1}, "device"),
        ({"device": ""}, "device"),
        ({"device": "   "}, "device"),
        ({"scheduler": object()}, "scheduler"),
        ({"scheduler": type("BadScheduler", (), {"step": object()})()}, "scheduler"),
        ({"trigger_hook": True}, "trigger_hook"),
        ({"trigger_hook": object()}, "trigger_hook"),
        ({"logger": object()}, "logger"),
        ({"logger": type("BadLogger", (), {"log_metrics": object()})()}, "logger"),
        ({"distributed_backend": True}, "distributed_backend"),
        ({"distributed_backend": ""}, "distributed_backend"),
        ({"ddp_kwargs": True}, "ddp_kwargs"),
        ({"ddp_kwargs": [("find_unused_parameters", True)]}, "ddp_kwargs"),
        ({"ddp_kwargs": {1: True}}, "ddp_kwargs"),
        ({"ddp_kwargs": {"": True}}, "ddp_kwargs"),
    ],
)
def test_fast_trainer_rejects_invalid_public_objects(
    trainer_kwargs: dict[str, object],
    match: str,
) -> None:
    _loader, model, optimizer = _make_supervised_components()

    with pytest.raises(ValueError, match=match):
        FastTrainer(
            model,
            optimizer,
            use_amp=False,
            use_compile=False,
            **trainer_kwargs,
        )


def test_fast_trainer_accepts_duck_typed_logger_scheduler_and_trigger() -> None:
    _loader, model, optimizer = _make_supervised_components()

    class Scheduler:
        def __init__(self) -> None:
            self.calls = 0

        def step(self) -> None:
            self.calls += 1

    class Logger:
        def __init__(self) -> None:
            self.rows: list[tuple[str, dict[str, object]]] = []

        def log_metrics(
            self,
            stage: str,
            metrics: dict[str, object],
            **_: object,
        ) -> None:
            self.rows.append((stage, metrics))

    def trigger(_ctx: dict[str, object]) -> None:
        return None

    scheduler = Scheduler()
    logger = Logger()
    trainer = FastTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        trigger_hook=trigger,
        logger=logger,
        ddp_kwargs={"find_unused_parameters": False},
        device=" cpu ",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    assert trainer.scheduler is scheduler
    assert trainer.trigger_hook is trigger
    assert trainer.logger is logger
    assert trainer.device == "cpu"


def test_fast_trainer_rejects_backward_compile_init_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoMoveLinear(nn.Linear):
        def to(self, *args: object, **kwargs: object) -> "NoMoveLinear":
            return self

    class CompileResult:
        def __init__(self, model: nn.Module) -> None:
            self.model = model
            self.compiled = True
            self.fallback_reason = ""

    timestamps = iter([2.0, 1.0])
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(
        engine,
        "safe_compile_with_diagnostics",
        lambda model, *, mode="default": CompileResult(model),
    )
    model = NoMoveLinear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="compile_init_time_s"):
        FastTrainer(
            model,
            optimizer,
            device="mps",
            use_amp=False,
            use_compile=True,
        )


@pytest.mark.parametrize(
    ("train_kwargs", "match"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": -1}, "steps"),
        ({"steps": 1.5}, "steps"),
        ({"steps": "2"}, "steps"),
        ({"steps": True}, "steps"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"warmup_steps": 0.5}, "warmup_steps"),
        ({"steps": 1, "warmup_steps": 2}, "warmup_steps"),
        ({"profile_window": 0}, "profile_window"),
        ({"profile_window": 8.5}, "profile_window"),
        ({"profile_model_depth": 0}, "profile_model_depth"),
        ({"profile_model_max_modules": 0}, "profile_model_max_modules"),
    ],
)
def test_train_one_epoch_rejects_invalid_numeric_settings(
    train_kwargs: dict[str, object],
    match: str,
) -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match=match):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), **train_kwargs)


@pytest.mark.parametrize(
    ("train_kwargs", "match"),
    [
        ({"collect_profile": "true"}, "collect_profile"),
        ({"profile_sync": 1}, "profile_sync"),
        ({"profile_distribution": "false"}, "profile_distribution"),
        ({"profile_model": 1}, "profile_model"),
    ],
)
def test_train_one_epoch_rejects_invalid_boolean_settings(
    train_kwargs: dict[str, object],
    match: str,
) -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match=match):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), **train_kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "profile_model_include",
    [
        0,
        True,
        object(),
        ["0", 2],
        [True],
    ],
)
def test_train_one_epoch_rejects_invalid_profile_model_include(
    profile_model_include: object,
) -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match="profile_model_include"):
        trainer.train_one_epoch(
            loader,
            nn.CrossEntropyLoss(),
            steps=1,
            collect_profile=True,
            profile_model=True,
            profile_model_include=profile_model_include,  # type: ignore[arg-type]
        )


def test_eval_and_predict_reject_invalid_step_limits() -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match="steps"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=0)
    with pytest.raises(ValueError, match="steps"):
        trainer.predict(loader, steps=-1)
    with pytest.raises(ValueError, match="steps"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=1.5)
    with pytest.raises(ValueError, match="steps"):
        trainer.predict(loader, steps="2")
    with pytest.raises(ValueError, match="profile_window"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), profile_window=0)
    with pytest.raises(ValueError, match="profile_window"):
        trainer.predict(loader, profile_window=0, return_metrics=True)


@pytest.mark.parametrize(
    ("eval_kwargs", "match"),
    [
        ({"collect_profile": "true"}, "collect_profile"),
        ({"profile_sync": 1}, "profile_sync"),
        ({"profile_distribution": "false"}, "profile_distribution"),
    ],
)
def test_evaluate_rejects_invalid_boolean_settings(
    eval_kwargs: dict[str, object],
    match: str,
) -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match=match):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), **eval_kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("predict_kwargs", "match"),
    [
        ({"collect_profile": "true"}, "collect_profile"),
        ({"profile_sync": 1}, "profile_sync"),
        ({"profile_distribution": "false"}, "profile_distribution"),
        ({"return_metrics": "true"}, "return_metrics"),
    ],
)
def test_predict_rejects_invalid_boolean_settings(
    predict_kwargs: dict[str, object],
    match: str,
) -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match=match):
        trainer.predict(loader, **predict_kwargs)  # type: ignore[arg-type]


def test_predict_rejects_invalid_postprocess_before_loop() -> None:
    class CountingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.proj = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.proj(inputs)

    inputs = torch.randn(2, 4)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = CountingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match="postprocess"):
        trainer.predict(loader, postprocess=object())  # type: ignore[arg-type]

    assert model.calls == 0


def test_predict_detaches_nested_outputs_to_cpu() -> None:
    class NestedOutputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> dict[str, object]:
            logits = self.proj(inputs)
            return {
                "logits": logits,
                "nested": [logits + 1.0, (logits + 2.0,)],
                "label": "kept",
            }

    inputs = torch.randn(4, 4)
    targets = torch.zeros(4)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = NestedOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    def postprocess(outputs: dict[str, object]) -> dict[str, object]:
        logits = outputs["logits"]
        assert isinstance(logits, torch.Tensor)
        return {
            "logits": logits.detach().requires_grad_(),
            "nested": outputs["nested"],
            "label": outputs["label"],
        }

    predictions = trainer.predict(loader, steps=1, postprocess=postprocess)

    assert len(predictions) == 1
    first = predictions[0]
    assert isinstance(first, dict)
    assert first["label"] == "kept"
    tensors = [
        first["logits"],
        first["nested"][0],  # type: ignore[index]
        first["nested"][1][0],  # type: ignore[index]
    ]
    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False


def test_evaluate_collects_phase_profile_and_user_metrics() -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    def metrics_fn(outputs: torch.Tensor, batch_targets: torch.Tensor, _inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        predictions = outputs.argmax(dim=1)
        return {"accuracy": (predictions == batch_targets).float().mean()}

    metrics = trainer.evaluate(
        loader,
        nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn,
        steps=2,
        collect_profile=True,
    )

    profile = metrics["profile"]
    phases = profile["phases"]
    assert metrics["steps"] == 2
    assert metrics["measured_steps"] == 2
    assert metrics["unmeasured_steps"] == 0
    assert metrics["batch_size_inference_failures"] == 0
    assert metrics["batch_size_inference_failure_reasons"] == {}
    assert metrics["samples"] == 8
    assert metrics["reported_samples_per_sec"] == metrics["samples_per_sec"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["user_metric_valid_count"] == 2
    assert metrics["user_metric_invalid_count"] == 0
    assert metrics["user_metric_non_finite_count"] == 0
    assert metrics["user_metric_unmeasured_count"] == 0
    assert metrics["user_metric_skipped_count"] == 0
    assert metrics["metrics_fn_requested"] is True
    assert metrics["metrics_fn_calls"] == 2
    assert metrics["metrics_fn_successes"] == 2
    assert metrics["metrics_fn_failures"] == 0
    assert metrics["metrics_fn_last_error"] == ""
    assert metrics["eval_failed"] is False
    assert metrics["eval_failure_stage"] == ""
    assert metrics["eval_failure_last_error"] == ""
    for phase_name in ("data_wait", "transfer", "forward", "loss", "user_metrics", "metrics"):
        assert phase_name in phases
        assert metrics[f"profile_{phase_name}_time_s"] == pytest.approx(phases[phase_name]["total_s"])
        assert metrics[f"profile_{phase_name}_pct"] == pytest.approx(phases[phase_name]["pct"])
        assert metrics[f"profile_{phase_name}_avg_ms"] == pytest.approx(phases[phase_name]["avg_ms"])


def test_train_eval_predict_share_device_memory_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_devices: list[str] = []
    collect_devices: list[str] = []

    def fake_reset(device: str) -> None:
        reset_devices.append(device)

    def fake_collect(device: str) -> dict[str, int]:
        collect_devices.append(device)
        return {"cuda_current_mem_bytes": len(collect_devices)}

    monkeypatch.setattr(engine, "_reset_device_peak_memory_stats", fake_reset)
    monkeypatch.setattr(engine, "_collect_device_memory_metrics", fake_collect)
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    train_metrics = trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1)
    eval_metrics = trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=1)
    _predictions, predict_metrics = trainer.predict(loader, steps=1, return_metrics=True)

    assert train_metrics["cuda_current_mem_bytes"] == 1
    assert eval_metrics["cuda_current_mem_bytes"] == 2
    assert predict_metrics["cuda_current_mem_bytes"] == 3
    assert reset_devices == ["cpu", "cpu", "cpu"]
    assert collect_devices == ["cpu", "cpu", "cpu"]


def test_evaluate_skips_invalid_user_metrics_and_reports_counts() -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)
    calls = 0

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "accuracy": torch.tensor(0.25),
                "bad": torch.tensor(float("nan")),
                "invalid": object(),
            }
        return {
            "accuracy": torch.tensor(0.75),
            "bad": torch.tensor(float("inf")),
            "invalid": "not-a-number",
        }

    metrics = trainer.evaluate(
        loader,
        nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn,
        steps=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert "bad" not in metrics
    assert "invalid" not in metrics
    assert metrics["user_metric_valid_count"] == 2
    assert metrics["user_metric_invalid_count"] == 2
    assert metrics["user_metric_non_finite_count"] == 2
    assert metrics["user_metric_unmeasured_count"] == 0
    assert metrics["user_metric_skipped_count"] == 4
    assert metrics["metrics_fn_requested"] is True
    assert metrics["metrics_fn_calls"] == 2
    assert metrics["metrics_fn_successes"] == 2
    assert metrics["metrics_fn_failures"] == 0
    assert metrics["metrics_fn_last_error"] == ""


def test_evaluate_rejects_coerced_user_metrics_and_bad_names() -> None:
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> dict[object, object]:
        return {
            "valid_int_tensor": torch.tensor([1, 3], dtype=torch.int64),
            "valid_python_int": 2,
            "truthy": True,
            "numeric_string": "0.5",
            "numeric_bytes": b"0.5",
            "bool_tensor": torch.tensor([True, False]),
            "complex_tensor": torch.tensor([1.0 + 0.0j]),
            "broken_float": _FailingFloat(),
            "": 1.0,
            ("tuple", "key"): 1.0,
        }

    metrics = trainer.evaluate(
        loader,
        nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn,
        steps=1,
    )

    assert metrics["valid_int_tensor"] == pytest.approx(2.0)
    assert metrics["valid_python_int"] == pytest.approx(2.0)
    assert "truthy" not in metrics
    assert "numeric_string" not in metrics
    assert "numeric_bytes" not in metrics
    assert "bool_tensor" not in metrics
    assert "complex_tensor" not in metrics
    assert "broken_float" not in metrics
    assert metrics["user_metric_valid_count"] == 2
    assert metrics["user_metric_invalid_count"] == 8
    assert metrics["user_metric_non_finite_count"] == 0
    assert metrics["user_metric_skipped_count"] == 8


def test_evaluate_rejects_user_metrics_that_collide_with_internal_metrics() -> None:
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> dict[str, float]:
        return {
            "accuracy": 0.75,
            "samples": 999.0,
            "samples_per_sec": 999.0,
            "reported_samples_per_sec": 999.0,
            "user_metric_valid_count": 999.0,
            "profile_forward_pct": 999.0,
            "batch_size_inference_failures": 999.0,
            "cuda_max_mem_bytes": 999.0,
            "metrics_fn_calls": 999.0,
        }

    metrics = trainer.evaluate(
        loader,
        nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn,
        steps=1,
        collect_profile=True,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["samples"] == 4
    assert metrics["samples_per_sec"] != 999.0
    assert metrics["reported_samples_per_sec"] == metrics["samples_per_sec"]
    assert metrics["user_metric_valid_count"] == 1
    assert metrics["user_metric_invalid_count"] == 8
    assert metrics["user_metric_skipped_count"] == 8
    assert metrics["batch_size_inference_failures"] == 0
    assert metrics["metrics_fn_calls"] == 1
    assert "cuda_max_mem_bytes" not in metrics
    assert metrics["profile_forward_pct"] != 999.0


def test_evaluate_distributed_summary_sums_metrics_fn_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)
    trainer.dist_ctx = utils_mod.DistributedContext(
        is_initialized=True,
        rank=0,
        world_size=2,
        local_rank=0,
        backend="gloo",
    )

    def fake_distributed_sum(value: torch.Tensor) -> torch.Tensor:
        return value * 2

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> dict[str, float]:
        return {"accuracy": 0.5}

    monkeypatch.setattr(engine, "distributed_sum", fake_distributed_sum)

    metrics = trainer.evaluate(
        loader,
        nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn,
        steps=1,
    )

    assert metrics["steps"] == 2
    assert metrics["batches"] == pytest.approx(2.0)
    assert metrics["samples"] == 4
    assert type(metrics["samples"]) is int
    assert metrics["measured_steps"] == 2
    assert metrics["unmeasured_steps"] == 0
    assert metrics["samples_per_sec"] == pytest.approx(
        metrics["samples"] / metrics["total_time_s"]
    )
    assert metrics["reported_samples_per_sec"] == metrics["samples_per_sec"]
    assert metrics["metrics_fn_calls"] == 2
    assert metrics["metrics_fn_successes"] == 2
    assert metrics["metrics_fn_failures"] == 0
    assert metrics["user_metric_valid_count"] == 2
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_train_distributed_summary_sums_workload_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingScheduler:
        def step(self) -> None:
            raise RuntimeError("scheduler boom")

    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(
        model,
        optimizer,
        scheduler=FailingScheduler(),
        device="cpu",
        use_amp=False,
        use_compile=False,
        grad_accum=2,
        log_interval=999,
    )
    trainer.dist_ctx = utils_mod.DistributedContext(
        is_initialized=True,
        rank=0,
        world_size=2,
        local_rank=0,
        backend="gloo",
    )

    def fake_distributed_sum(value: torch.Tensor) -> torch.Tensor:
        return value * 2

    monkeypatch.setattr(engine, "distributed_sum", fake_distributed_sum)

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=2,
        warmup_steps=1,
    )

    assert metrics["steps"] == 4
    assert metrics["batches"] == pytest.approx(4.0)
    assert metrics["samples"] == 8
    assert type(metrics["samples"]) is int
    assert metrics["samples_per_sec"] == pytest.approx(
        metrics["samples"] / metrics["total_time_s"]
    )
    assert metrics["optimizer_steps"] == 2
    assert metrics["grad_accum"] == 2
    assert metrics["partial_optimizer_steps"] == 0
    assert metrics["grad_accum_tail_steps"] == 0
    assert metrics["scheduler_step_failures"] == 2
    assert metrics["warmup_steps"] == 2
    assert metrics["warmup_batches"] == pytest.approx(2.0)
    assert metrics["warmup_samples"] == 4
    assert metrics["warmup_samples_per_sec"] == pytest.approx(
        metrics["warmup_samples"] / metrics["warmup_total_time_s"]
    )
    assert metrics["warmup_optimizer_steps"] == 0
    assert metrics["steady_steps"] == 2
    assert metrics["steady_batches"] == pytest.approx(2.0)
    assert metrics["steady_samples"] == 4
    assert metrics["steady_samples_per_sec"] == pytest.approx(
        metrics["steady_samples"] / metrics["steady_total_time_s"]
    )
    assert metrics["reported_samples_per_sec"] == metrics["steady_samples_per_sec"]
    assert metrics["steady_optimizer_steps"] == 2
    assert metrics["cold_start_steps"] == 2


def test_evaluate_reports_non_mapping_user_metrics_as_invalid() -> None:
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=1)

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> list[tuple[str, float]]:
        return [("accuracy", 1.0)]

    metrics = trainer.evaluate(
        loader,
        nn.CrossEntropyLoss(),
        metrics_fn=metrics_fn,  # type: ignore[arg-type]
        steps=1,
    )

    assert metrics["steps"] == 1
    assert metrics["user_metric_valid_count"] == 0
    assert metrics["user_metric_invalid_count"] == 1
    assert metrics["user_metric_skipped_count"] == 1
    assert "accuracy" not in metrics


def test_evaluate_rejects_invalid_metrics_fn_before_loop() -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match="metrics_fn"):
        trainer.evaluate(
            loader,
            nn.CrossEntropyLoss(),
            metrics_fn=object(),  # type: ignore[arg-type]
        )


def test_evaluate_logs_metrics_fn_failures_before_reraising() -> None:
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError("metrics boom")

    with pytest.raises(RuntimeError, match="metrics boom"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), metrics_fn=metrics_fn, steps=1)

    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "eval"
    assert mode == "error"
    assert metrics["steps"] == 1
    assert metrics["measured_steps"] == 0
    assert metrics["metrics_fn_requested"] is True
    assert metrics["metrics_fn_calls"] == 1
    assert metrics["metrics_fn_successes"] == 0
    assert metrics["metrics_fn_failures"] == 1
    assert metrics["metrics_fn_last_error"] == "RuntimeError: metrics boom"
    assert metrics["eval_failed"] is True
    assert metrics["eval_failure_stage"] == "user_metrics"
    assert metrics["eval_failure_last_error"] == "RuntimeError: metrics boom"
    assert metrics["user_metric_valid_count"] == 0
    assert metrics["user_metric_skipped_count"] == 0


def test_evaluate_failure_metrics_handle_unstringifiable_exceptions() -> None:
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    def metrics_fn(
        _outputs: torch.Tensor,
        _batch_targets: torch.Tensor,
        _inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise BrokenStrError()

    with pytest.raises(BrokenStrError):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), metrics_fn=metrics_fn, steps=1)

    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "eval"
    assert mode == "error"
    assert metrics["metrics_fn_last_error"] == "BrokenStrError"
    assert metrics["eval_failure_last_error"] == "BrokenStrError"
    assert metrics["eval_failure_stage"] == "user_metrics"


def test_evaluate_logs_forward_failures_before_reraising() -> None:
    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 3)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("eval forward boom")

    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = FailingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="eval forward boom"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "eval"
    assert mode == "error"
    assert metrics["steps"] == 1
    assert metrics["measured_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["eval_failed"] is True
    assert metrics["eval_failure_stage"] == "forward"
    assert metrics["eval_failure_last_error"] == "RuntimeError: eval forward boom"
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "data_wait" in phases
    assert "transfer" in phases
    assert "forward" in phases


def test_evaluate_cleans_profile_phase_when_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profilers = _capture_engine_phase_profilers(monkeypatch)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="loader boom"):
        trainer.evaluate(_FailingLoader(), nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "eval"
    assert mode == "error"
    assert metrics["steps"] == 0
    assert metrics["measured_steps"] == 0
    assert metrics["eval_failed"] is True
    assert metrics["eval_failure_stage"] == "data_wait"
    assert metrics["eval_failure_last_error"] == "RuntimeError: loader boom"


def test_evaluate_logs_transfer_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_device(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("eval transfer boom")

    monkeypatch.setattr(engine, "to_device", fail_to_device)
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="eval transfer boom"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_eval_failure_metrics(
        logger,
        stage="transfer",
        last_error="RuntimeError: eval transfer boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "data_wait" in phases
    assert "transfer" in phases


def test_evaluate_logs_loss_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCriterion(nn.Module):
        def forward(self, _outputs: torch.Tensor, _targets: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("eval loss boom")

    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="eval loss boom"):
        trainer.evaluate(loader, FailingCriterion(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_eval_failure_metrics(
        logger,
        stage="loss",
        last_error="RuntimeError: eval loss boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "forward" in phases
    assert "loss" in phases


def test_evaluate_logs_metrics_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_first_meter_record(monkeypatch, "eval metrics boom")
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="eval metrics boom"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_eval_failure_metrics(
        logger,
        stage="metrics",
        last_error="RuntimeError: eval metrics boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "loss" in phases
    assert "metrics" in phases


def test_evaluate_rejects_invalid_batch_duration_before_recording_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_batch_duration_validation(monkeypatch, "eval batch duration boom")
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(ValueError, match="eval batch duration boom"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_eval_failure_metrics(
        logger,
        stage="metrics",
        last_error="ValueError: eval batch duration boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "loss" in phases
    assert "metrics" in phases


def test_evaluate_reports_scalar_tensor_inputs_as_unmeasured() -> None:
    class ScalarTensorDataset(torch.utils.data.Dataset[torch.Tensor]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.tensor(float(index))

    class ScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * self.weight

    def metrics_fn(
        outputs: torch.Tensor,
        _targets: None,
        _inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {"score": outputs + 1.0}

    loader = DataLoader(ScalarTensorDataset(), batch_size=None)
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    metrics = trainer.evaluate(loader, metrics_fn=metrics_fn)

    assert metrics["steps"] == 2
    assert metrics["measured_steps"] == 0
    assert metrics["unmeasured_steps"] == 2
    assert metrics["batch_size_inference_failures"] == 2
    assert metrics["batch_size_inference_failure_reasons"] == {"tensor_scalar": 2}
    assert metrics["batch_size_inference_tensor_scalar_failures"] == 2
    assert metrics["batch_size_inference_unsupported_type_failures"] == 0
    assert metrics["samples"] == 0
    assert metrics["samples_per_sec"] == 0.0
    assert metrics["reported_samples_per_sec"] == 0.0
    assert metrics["avg_loss"] == 0.0
    assert "score" not in metrics
    assert metrics["user_metric_valid_count"] == 0
    assert metrics["user_metric_unmeasured_count"] == 2
    assert metrics["user_metric_skipped_count"] == 2


def test_predict_can_return_metrics_and_phase_profile() -> None:
    inputs = torch.randn(6, 4)
    targets = torch.zeros(6)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    def postprocess(outputs: torch.Tensor) -> torch.Tensor:
        return outputs.softmax(dim=1)

    predictions, metrics = trainer.predict(
        loader,
        steps=2,
        postprocess=postprocess,
        collect_profile=True,
    )

    profile = metrics["profile"]
    phases = profile["phases"]
    assert len(predictions) == 2
    assert metrics["steps"] == 2
    assert metrics["measured_steps"] == 2
    assert metrics["unmeasured_steps"] == 0
    assert metrics["batch_size_inference_failures"] == 0
    assert metrics["batch_size_inference_failure_reasons"] == {}
    assert metrics["samples"] == 6
    assert metrics["reported_samples_per_sec"] == metrics["samples_per_sec"]
    assert metrics["postprocess_requested"] is True
    assert metrics["postprocess_calls"] == 2
    assert metrics["postprocess_successes"] == 2
    assert metrics["postprocess_failures"] == 0
    assert metrics["postprocess_last_error"] == ""
    assert metrics["predict_failed"] is False
    assert metrics["predict_failure_stage"] == ""
    assert metrics["predict_failure_last_error"] == ""
    for phase_name in ("data_wait", "transfer", "forward", "postprocess", "collect_output", "metrics"):
        assert phase_name in phases
        assert metrics[f"profile_{phase_name}_time_s"] == pytest.approx(phases[phase_name]["total_s"])
        assert metrics[f"profile_{phase_name}_pct"] == pytest.approx(phases[phase_name]["pct"])


def test_predict_distributed_summary_sums_counter_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)
    trainer.dist_ctx = utils_mod.DistributedContext(
        is_initialized=True,
        rank=0,
        world_size=2,
        local_rank=0,
        backend="gloo",
    )

    def fake_distributed_sum(value: torch.Tensor) -> torch.Tensor:
        return value * 2

    def postprocess(outputs: torch.Tensor) -> torch.Tensor:
        return outputs.softmax(dim=1)

    monkeypatch.setattr(engine, "distributed_sum", fake_distributed_sum)

    predictions, metrics = trainer.predict(
        loader,
        steps=1,
        postprocess=postprocess,
        return_metrics=True,
    )

    assert len(predictions) == 1
    assert metrics["steps"] == 2
    assert metrics["batches"] == pytest.approx(2.0)
    assert metrics["samples"] == 4
    assert type(metrics["samples"]) is int
    assert metrics["measured_steps"] == 2
    assert metrics["unmeasured_steps"] == 0
    assert metrics["samples_per_sec"] == pytest.approx(
        metrics["samples"] / metrics["total_time_s"]
    )
    assert metrics["reported_samples_per_sec"] == metrics["samples_per_sec"]
    assert metrics["batch_size_inference_failures"] == 0
    assert metrics["postprocess_calls"] == 2
    assert metrics["postprocess_successes"] == 2
    assert metrics["postprocess_failures"] == 0


def test_predict_logs_postprocess_failures_before_reraising() -> None:
    inputs = torch.randn(2, 4)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    def postprocess(_outputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("postprocess boom")

    with pytest.raises(RuntimeError, match="postprocess boom"):
        trainer.predict(loader, postprocess=postprocess, return_metrics=True, steps=1)

    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "predict"
    assert mode == "error"
    assert metrics["steps"] == 1
    assert metrics["measured_steps"] == 0
    assert metrics["postprocess_requested"] is True
    assert metrics["postprocess_calls"] == 1
    assert metrics["postprocess_successes"] == 0
    assert metrics["postprocess_failures"] == 1
    assert metrics["postprocess_last_error"] == "RuntimeError: postprocess boom"
    assert metrics["predict_failed"] is True
    assert metrics["predict_failure_stage"] == "postprocess"
    assert metrics["predict_failure_last_error"] == "RuntimeError: postprocess boom"


def test_predict_logs_forward_failures_before_reraising() -> None:
    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("predict forward boom")

    inputs = torch.randn(2, 4)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = FailingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="predict forward boom"):
        trainer.predict(loader, steps=1, collect_profile=True)

    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "predict"
    assert mode == "error"
    assert metrics["steps"] == 1
    assert metrics["measured_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["predict_failed"] is True
    assert metrics["predict_failure_stage"] == "forward"
    assert metrics["predict_failure_last_error"] == "RuntimeError: predict forward boom"
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "data_wait" in phases
    assert "transfer" in phases
    assert "forward" in phases


def test_predict_cleans_profile_phase_when_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profilers = _capture_engine_phase_profilers(monkeypatch)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="loader boom"):
        trainer.predict(_FailingLoader(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "predict"
    assert mode == "error"
    assert metrics["steps"] == 0
    assert metrics["measured_steps"] == 0
    assert metrics["predict_failed"] is True
    assert metrics["predict_failure_stage"] == "data_wait"
    assert metrics["predict_failure_last_error"] == "RuntimeError: loader boom"


def test_predict_logs_transfer_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_device(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("predict transfer boom")

    monkeypatch.setattr(engine, "to_device", fail_to_device)
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="predict transfer boom"):
        trainer.predict(loader, steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_predict_failure_metrics(
        logger,
        stage="transfer",
        last_error="RuntimeError: predict transfer boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "data_wait" in phases
    assert "transfer" in phases


def test_predict_logs_collect_output_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_detach_to_cpu(_outputs: object) -> None:
        raise RuntimeError("collect boom")

    monkeypatch.setattr(engine, "_detach_to_cpu", fail_detach_to_cpu)
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="collect boom"):
        trainer.predict(loader, steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_predict_failure_metrics(
        logger,
        stage="collect_output",
        last_error="RuntimeError: collect boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "forward" in phases
    assert "collect_output" in phases


def test_predict_logs_metrics_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_first_meter_record(monkeypatch, "predict metrics boom")
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="predict metrics boom"):
        trainer.predict(loader, steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_predict_failure_metrics(
        logger,
        stage="metrics",
        last_error="RuntimeError: predict metrics boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "collect_output" in phases
    assert "metrics" in phases


def test_predict_rejects_invalid_batch_duration_before_recording_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_batch_duration_validation(monkeypatch, "predict batch duration boom")
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(ValueError, match="predict batch duration boom"):
        trainer.predict(loader, steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_predict_failure_metrics(
        logger,
        stage="metrics",
        last_error="ValueError: predict batch duration boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "collect_output" in phases
    assert "metrics" in phases


def test_predict_reports_unmeasured_steps_when_batch_size_is_unknown() -> None:
    class ScalarDataset(torch.utils.data.Dataset[object]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> object:
            return object()

    class ConstantModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

        def forward(self, _inputs: object) -> torch.Tensor:
            return self.weight

    loader = DataLoader(ScalarDataset(), batch_size=None)
    model = ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    predictions, metrics = trainer.predict(loader, return_metrics=True)

    assert len(predictions) == 2
    assert metrics["steps"] == 2
    assert metrics["measured_steps"] == 0
    assert metrics["unmeasured_steps"] == 2
    assert metrics["batch_size_inference_failures"] == 2
    assert metrics["batch_size_inference_failure_reasons"] == {"unsupported_type": 2}
    assert metrics["batch_size_inference_unsupported_type_failures"] == 2
    assert metrics["batch_size_inference_tensor_scalar_failures"] == 0
    assert metrics["postprocess_requested"] is False
    assert metrics["postprocess_calls"] == 0
    assert metrics["postprocess_failures"] == 0
    assert metrics["postprocess_last_error"] == ""
    assert metrics["samples"] == 0
    assert metrics["samples_per_sec"] == 0.0
    assert metrics["reported_samples_per_sec"] == 0.0


def test_predict_reports_scalar_tensor_inputs_as_unmeasured() -> None:
    class ScalarTensorDataset(torch.utils.data.Dataset[torch.Tensor]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> torch.Tensor:
            return torch.tensor(float(index))

    class ScalarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * self.weight

    loader = DataLoader(ScalarTensorDataset(), batch_size=None)
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    predictions, metrics = trainer.predict(loader, return_metrics=True)

    assert len(predictions) == 2
    assert metrics["steps"] == 2
    assert metrics["measured_steps"] == 0
    assert metrics["unmeasured_steps"] == 2
    assert metrics["batch_size_inference_failures"] == 2
    assert metrics["batch_size_inference_failure_reasons"] == {"tensor_scalar": 2}
    assert metrics["batch_size_inference_tensor_scalar_failures"] == 2
    assert metrics["batch_size_inference_unsupported_type_failures"] == 0
    assert metrics["samples"] == 0
    assert metrics["reported_samples_per_sec"] == 0.0


def test_fit_accepts_train_profile_and_loader_options() -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    dataset = TensorDataset(inputs, targets)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    metrics = trainer.fit(
        dataset,
        nn.CrossEntropyLoss(),
        batch_size=4,
        num_workers=0,
        shuffle=False,
        steps=2,
        warmup_steps=1,
        collect_profile=True,
    )

    phases = metrics["profile"]["phases"]
    assert metrics["steps"] == 2
    assert metrics["warmup_steps"] == 1
    assert metrics["steady_steps"] == 1
    assert "forward" in phases
    assert "loss" in phases


def test_fit_rejects_invalid_profile_boolean_before_loader_build() -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    dataset = TensorDataset(inputs, targets)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match="collect_profile"):
        trainer.fit(  # type: ignore[arg-type]
            dataset,
            nn.CrossEntropyLoss(),
            batch_size=4,
            num_workers=0,
            collect_profile="true",
        )


def test_train_one_epoch_collects_phase_and_model_profile() -> None:
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=0)

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=2,
        collect_profile=True,
        profile_model=True,
        profile_model_include="0,2",
    )

    profile = metrics["profile"]
    phases = profile["phases"]
    assert metrics["steps"] == 2
    assert "p99_s" in metrics
    assert "std_batch_s" in metrics
    assert metrics["profile_model_requested"] is True
    assert metrics["profile_model_enabled"] is True
    assert metrics["profile_model_status"] == "ok"
    assert metrics["profile_model_modules_selected"] == 2
    assert metrics["profile_model_hook_count"] >= 4
    assert metrics["profile_model_hook_failures"] == 0
    assert metrics["profile_model_hook_last_error"] == ""
    assert metrics["train_failed"] is False
    assert metrics["train_failure_stage"] == ""
    assert metrics["train_failure_last_error"] == ""
    assert "data_wait" in phases
    assert "forward" in phases
    assert "loss" in phases
    assert "backward" in phases
    assert "optimizer" in phases
    assert "p99_ms" in phases["forward"]
    assert "std_ms" in phases["forward"]
    assert metrics["profile_total_s"] == pytest.approx(profile["profile_total_s"])
    assert metrics["profile_forward_time_s"] == pytest.approx(phases["forward"]["total_s"])
    assert metrics["profile_forward_pct"] == pytest.approx(phases["forward"]["pct"])
    assert metrics["profile_forward_avg_ms"] == pytest.approx(phases["forward"]["avg_ms"])
    assert metrics["profile_backward_time_s"] == pytest.approx(phases["backward"]["total_s"])
    assert metrics["profile_backward_pct"] == pytest.approx(phases["backward"]["pct"])
    assert metrics["profile_optimizer_time_s"] == pytest.approx(phases["optimizer"]["total_s"])
    assert metrics["profile_forward_backward_time_s"] == pytest.approx(
        phases["forward"]["total_s"] + phases["backward"]["total_s"]
    )
    assert metrics["profile_forward_backward_pct"] == pytest.approx(
        phases["forward"]["pct"] + phases["backward"]["pct"]
    )
    assert metrics["profile_flat_metric_invalid_count"] == 0
    assert "profile_flat_metric_invalid_fields" not in metrics

    forward_children = profile["phase_breakdowns"]["forward"]["top_children"]
    assert forward_children
    assert forward_children[0]["name"] in {"model.0", "model.2"}

    backward_events = profile["phase_events"]["backward_grad_ready"]["top_children"]
    assert backward_events
    assert backward_events[0]["name"] in {"model.0", "model.2"}
    assert backward_events[0]["calls"] >= 1

    optimizer_children = profile["phase_breakdowns"]["optimizer"]["top_children"]
    assert optimizer_children
    assert optimizer_children[0]["name"] in {"optimizer.step", "zero_grad"}


def test_train_one_epoch_cleans_profile_phase_when_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profilers = _capture_engine_phase_profilers(monkeypatch)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="loader boom"):
        trainer.train_one_epoch(_FailingLoader(), nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "train"
    assert mode == "error"
    assert metrics["steps"] == 0
    assert metrics["optimizer_steps"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == "data_wait"
    assert metrics["train_failure_last_error"] == "RuntimeError: loader boom"


def test_train_one_epoch_cleans_profile_phase_when_forward_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 3)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("forward boom")

    profilers = _capture_engine_phase_profilers(monkeypatch)
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = FailingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="forward boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    assert len(logger.rows) == 1
    stage, metrics, mode = logger.rows[0]
    assert stage == "train"
    assert mode == "error"
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == "forward"
    assert metrics["train_failure_last_error"] == "RuntimeError: forward boom"
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "data_wait" in phases
    assert "transfer" in phases
    assert "forward" in phases


def test_train_one_epoch_cleans_model_profile_detail_when_module_forward_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingLinear(nn.Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("module forward boom")

    profilers = _capture_engine_phase_profilers(monkeypatch)
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(FailingLinear(4, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(RuntimeError, match="module forward boom"):
        trainer.train_one_epoch(
            loader,
            nn.CrossEntropyLoss(),
            steps=1,
            collect_profile=True,
            profile_model=True,
            profile_model_include="0",
        )

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    assert profilers[0]._detail_starts == {}
    assert not model[0]._forward_pre_hooks
    assert not model[0]._forward_hooks


def test_train_one_epoch_cleans_profile_phase_when_loss_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCriterion(nn.Module):
        def forward(self, _outputs: torch.Tensor, _targets: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("loss boom")

    profilers = _capture_engine_phase_profilers(monkeypatch)
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="loss boom"):
        trainer.train_one_epoch(loader, FailingCriterion(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _single_error_metrics(logger, "train")
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == "loss"
    assert metrics["train_failure_last_error"] == "RuntimeError: loss boom"
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "forward" in phases
    assert "loss" in phases


def test_train_one_epoch_logs_backward_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DetachedCriterion(nn.Module):
        def forward(self, outputs: torch.Tensor, _targets: torch.Tensor) -> torch.Tensor:
            return outputs.detach().sum()

    profilers = _capture_engine_phase_profilers(monkeypatch)
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="does not require grad"):
        trainer.train_one_epoch(loader, DetachedCriterion(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _single_error_metrics(logger, "train")
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == "backward"
    assert "does not require grad" in metrics["train_failure_last_error"]
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "loss" in phases
    assert "backward" in phases


def test_train_one_epoch_logs_optimizer_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profilers = _capture_engine_phase_profilers(monkeypatch)
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    def fail_step(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("optimizer boom")

    monkeypatch.setattr(optimizer, "step", fail_step)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="optimizer boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _single_error_metrics(logger, "train")
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["samples"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == "optimizer"
    assert metrics["train_failure_last_error"] == "RuntimeError: optimizer boom"
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "backward" in phases
    assert "optimizer" in phases


def test_train_one_epoch_logs_metrics_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_first_meter_record(monkeypatch, "metrics meter boom")
    profilers = _capture_engine_phase_profilers(monkeypatch)
    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = _CapturingLogger()
    trainer = FastTrainer(
        model,
        optimizer,
        logger=logger,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(RuntimeError, match="metrics meter boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _single_error_metrics(logger, "train")
    assert metrics["steps"] == 1
    assert metrics["optimizer_steps"] == 1
    assert metrics["samples"] == 0
    assert metrics["train_failed"] is True
    assert metrics["train_failure_stage"] == "metrics"
    assert metrics["train_failure_last_error"] == "RuntimeError: metrics meter boom"
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "optimizer" in phases
    assert "metrics" in phases


def test_train_one_epoch_rejects_invalid_batch_duration_before_recording_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_batch_duration_validation(monkeypatch, "train batch duration boom")
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(ValueError, match="train batch duration boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_train_failure_metrics(
        logger,
        stage="metrics",
        last_error="ValueError: train batch duration boom",
        optimizer_steps=1,
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "optimizer" in phases
    assert "metrics" in phases


def test_train_one_epoch_logs_transfer_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_device(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("transfer boom")

    monkeypatch.setattr(engine, "to_device", fail_to_device)
    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger)

    with pytest.raises(RuntimeError, match="transfer boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_train_failure_metrics(
        logger,
        stage="transfer",
        last_error="RuntimeError: transfer boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "data_wait" in phases
    assert "transfer" in phases


def test_train_one_epoch_logs_trigger_observe_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingObserveTrigger:
        def observe(self, _ctx: dict[str, object]) -> None:
            raise RuntimeError("observe boom")

        def __call__(self, _ctx: dict[str, object]) -> None:
            return None

    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger, trigger_hook=FailingObserveTrigger())

    with pytest.raises(RuntimeError, match="observe boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_train_failure_metrics(
        logger,
        stage="trigger_observe",
        last_error="RuntimeError: observe boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "loss" in phases
    assert "trigger" not in phases


def test_train_one_epoch_logs_trigger_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def trigger(_ctx: dict[str, object]) -> None:
        raise RuntimeError("trigger boom")

    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger, trigger_hook=trigger)

    with pytest.raises(RuntimeError, match="trigger boom"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_train_failure_metrics(
        logger,
        stage="trigger",
        last_error="RuntimeError: trigger boom",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "loss" in phases
    assert "trigger" in phases


def test_train_one_epoch_logs_inject_transfer_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def trigger(_ctx: dict[str, object]) -> TriggerResult:
        return TriggerResult(
            extra_inputs={"inputs": torch.randn(1, 4)},
            extra_targets=torch.randint(0, 3, (1,)),
        )

    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger, trigger_hook=trigger)

    with pytest.raises(TypeError, match="mirror tensor structure"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_train_failure_metrics(
        logger,
        stage="inject_transfer",
        last_error="TypeError: Extra inputs must mirror tensor structure of original batch.",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "trigger" in phases
    assert "inject_transfer" in phases


def test_train_one_epoch_logs_loss_reduce_failures_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def trigger(_ctx: dict[str, object]) -> TriggerResult:
        return TriggerResult(weights=torch.ones(1))

    profilers = _capture_engine_phase_profilers(monkeypatch)
    logger = _CapturingLogger()
    loader, _optimizer, trainer = _make_logged_cpu_trainer(logger, trigger_hook=trigger)

    with pytest.raises(ValueError, match="matches the concatenated batch size"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1, collect_profile=True)

    assert len(profilers) == 1
    assert profilers[0]._starts == {}
    metrics = _assert_train_failure_metrics(
        logger,
        stage="loss_reduce",
        last_error="ValueError: Trigger weights must be a 1D tensor that matches the concatenated batch size.",
    )
    profile = metrics["profile"]
    assert isinstance(profile, dict)
    phases = profile["phases"]
    assert "trigger" in phases
    assert "loss_reduce" in phases


def test_profile_flat_metrics_skip_invalid_values() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": _FailingFloat(),
        "phases": {
            "forward": {"total_s": 0.1, "pct": float("nan"), "avg_ms": "bad"},
            "backward": {"total_s": float("inf"), "pct": 30.0, "avg_ms": True},
            "optimizer": {"total_s": True, "pct": "5.0", "avg_ms": 1.5},
            "loss": "bad-row",
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert "profile_total_s" not in metrics
    assert metrics["profile_forward_time_s"] == pytest.approx(0.1)
    assert "profile_forward_pct" not in metrics
    assert "profile_forward_avg_ms" not in metrics
    assert "profile_backward_time_s" not in metrics
    assert metrics["profile_backward_pct"] == pytest.approx(30.0)
    assert "profile_backward_avg_ms" not in metrics
    assert "profile_optimizer_time_s" not in metrics
    assert "profile_optimizer_pct" not in metrics
    assert metrics["profile_optimizer_avg_ms"] == pytest.approx(1.5)
    assert metrics["profile_forward_backward_time_s"] == pytest.approx(0.1)
    assert metrics["profile_forward_backward_pct"] == pytest.approx(30.0)
    assert metrics["profile_flat_metric_invalid_count"] == 7
    assert metrics["profile_flat_metric_invalid_fields"] == [
        "profile_total_s",
        "profile_forward_pct",
        "profile_forward_avg_ms",
        "profile_backward_time_s",
        "profile_backward_avg_ms",
        "profile_optimizer_time_s",
        "profile_optimizer_pct",
    ]


def test_profile_flat_metrics_include_open_timer_counts() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "profile_open_phase_count": 1.0,
        "profile_open_detail_count": 2,
        "phases": {},
    }

    _add_profile_phase_metrics(metrics, profile)

    assert metrics["profile_total_s"] == pytest.approx(1.0)
    assert metrics["profile_open_phase_count"] == 1
    assert metrics["profile_open_detail_count"] == 2
    assert metrics["profile_flat_metric_invalid_count"] == 0
    assert "profile_flat_metric_invalid_fields" not in metrics


def test_profile_flat_metrics_include_backward_event_position() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "phases": {
            "backward": {"total_s": 0.2, "pct": 20.0, "avg_ms": 20.0},
        },
        "phase_events": {
            "backward_grad_ready": {
                "parent": "backward",
                "parent_avg_ms": 20.0,
                "children": {
                    "model.0": {"avg_ms": 8.0},
                    "model.2": {"avg_ms": 4.0},
                },
                "top_children": [
                    {
                        "name": "model.0",
                        "avg_ms": 8.0,
                        "avg_pct_of_parent": 40.0,
                        "calls": 2,
                    },
                ],
            },
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert metrics["profile_backward_grad_ready_child_count"] == 2
    assert metrics["profile_backward_grad_ready_parent_avg_ms"] == pytest.approx(20.0)
    assert metrics["profile_backward_grad_ready_top_avg_ms"] == pytest.approx(8.0)
    assert metrics["profile_backward_grad_ready_top_pct"] == pytest.approx(40.0)
    assert metrics["profile_backward_grad_ready_top_calls"] == 2
    assert metrics["profile_flat_metric_invalid_count"] == 0
    assert "profile_flat_metric_invalid_fields" not in metrics


def test_profile_flat_metrics_tolerate_partial_backward_event_rows() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "phases": {},
        "phase_events": {
            "backward_grad_ready": {
                "children": {"model.0": {}},
                "top_children": [
                    {
                        "name": "model.0",
                        "avg_ms": 8.0,
                    },
                ],
            },
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert metrics["profile_backward_grad_ready_child_count"] == 1
    assert metrics["profile_backward_grad_ready_top_avg_ms"] == pytest.approx(8.0)
    assert "profile_backward_grad_ready_parent_avg_ms" not in metrics
    assert "profile_backward_grad_ready_top_pct" not in metrics
    assert "profile_backward_grad_ready_top_calls" not in metrics
    assert metrics["profile_flat_metric_invalid_count"] == 0
    assert "profile_flat_metric_invalid_fields" not in metrics


def test_profile_flat_metrics_reject_invalid_backward_event_values() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "phases": {},
        "phase_events": {
            "backward_grad_ready": {
                "parent_avg_ms": "slow",
                "children": {"model.0": {}},
                "top_children": [
                    {
                        "name": "model.0",
                        "avg_ms": float("nan"),
                        "avg_pct_of_parent": 125.0,
                        "calls": 1.5,
                    },
                ],
            },
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert metrics["profile_backward_grad_ready_child_count"] == 1
    assert "profile_backward_grad_ready_parent_avg_ms" not in metrics
    assert "profile_backward_grad_ready_top_avg_ms" not in metrics
    assert "profile_backward_grad_ready_top_pct" not in metrics
    assert "profile_backward_grad_ready_top_calls" not in metrics
    assert metrics["profile_flat_metric_invalid_count"] == 4
    assert metrics["profile_flat_metric_invalid_fields"] == [
        "profile_backward_grad_ready_parent_avg_ms",
        "profile_backward_grad_ready_top_avg_ms",
        "profile_backward_grad_ready_top_pct",
        "profile_backward_grad_ready_top_calls",
    ]


def test_profile_flat_metrics_reject_invalid_open_timer_counts() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "profile_open_phase_count": 0.5,
        "profile_open_detail_count": True,
        "phases": {},
    }

    _add_profile_phase_metrics(metrics, profile)

    assert "profile_open_phase_count" not in metrics
    assert "profile_open_detail_count" not in metrics
    assert metrics["profile_flat_metric_invalid_count"] == 2
    assert metrics["profile_flat_metric_invalid_fields"] == [
        "profile_open_phase_count",
        "profile_open_detail_count",
    ]


def test_profile_flat_metrics_skip_negative_and_out_of_range_values() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": -1.0,
        "phases": {
            "forward": {"total_s": -0.1, "pct": 101.0, "avg_ms": -0.5},
            "backward": {"total_s": 0.2, "pct": -5.0, "avg_ms": 0.7},
            "optimizer": {"total_s": 0.1, "pct": 50.0, "avg_ms": 0.5},
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert "profile_total_s" not in metrics
    assert "profile_forward_time_s" not in metrics
    assert "profile_forward_pct" not in metrics
    assert "profile_forward_avg_ms" not in metrics
    assert metrics["profile_backward_time_s"] == pytest.approx(0.2)
    assert "profile_backward_pct" not in metrics
    assert metrics["profile_backward_avg_ms"] == pytest.approx(0.7)
    assert metrics["profile_optimizer_pct"] == pytest.approx(50.0)
    assert metrics["profile_forward_backward_time_s"] == pytest.approx(0.2)
    assert "profile_forward_backward_pct" not in metrics
    assert metrics["profile_flat_metric_invalid_count"] == 5
    assert metrics["profile_flat_metric_invalid_fields"] == [
        "profile_total_s",
        "profile_forward_time_s",
        "profile_forward_pct",
        "profile_forward_avg_ms",
        "profile_backward_pct",
    ]


def test_profile_flat_metrics_skip_missing_phase_fields() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "phases": {
            "forward": {"pct": 10.0},
            "backward": {"total_s": 0.2, "avg_ms": 0.7},
            "optimizer": {},
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert "profile_total_s" not in metrics
    assert "profile_forward_time_s" not in metrics
    assert metrics["profile_forward_pct"] == pytest.approx(10.0)
    assert "profile_forward_avg_ms" not in metrics
    assert metrics["profile_backward_time_s"] == pytest.approx(0.2)
    assert "profile_backward_pct" not in metrics
    assert metrics["profile_backward_avg_ms"] == pytest.approx(0.7)
    assert "profile_optimizer_time_s" not in metrics
    assert "profile_optimizer_pct" not in metrics
    assert "profile_optimizer_avg_ms" not in metrics
    assert metrics["profile_forward_backward_time_s"] == pytest.approx(0.2)
    assert metrics["profile_forward_backward_pct"] == pytest.approx(10.0)
    assert metrics["profile_flat_metric_invalid_count"] == 7
    assert metrics["profile_flat_metric_invalid_fields"] == [
        "profile_total_s",
        "profile_forward_time_s",
        "profile_forward_avg_ms",
        "profile_backward_pct",
        "profile_optimizer_time_s",
        "profile_optimizer_pct",
        "profile_optimizer_avg_ms",
    ]


def test_profile_flat_metrics_omit_combined_pct_above_100() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "phases": {
            "forward": {"total_s": 0.4, "pct": 60.0, "avg_ms": 1.0},
            "backward": {"total_s": 0.3, "pct": 50.0, "avg_ms": 2.0},
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert metrics["profile_forward_pct"] == pytest.approx(60.0)
    assert metrics["profile_backward_pct"] == pytest.approx(50.0)
    assert metrics["profile_forward_backward_time_s"] == pytest.approx(0.7)
    assert "profile_forward_backward_pct" not in metrics
    assert metrics["profile_flat_metric_invalid_count"] == 1
    assert metrics["profile_flat_metric_invalid_fields"] == ["profile_forward_backward_pct"]


def test_profile_flat_metrics_omit_combined_forward_backward_when_invalid() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": 1.0,
        "phases": {
            "forward": {"total_s": float("nan"), "pct": "bad", "avg_ms": 1.0},
            "backward": {"total_s": True, "pct": float("inf"), "avg_ms": 2.0},
        },
    }

    _add_profile_phase_metrics(metrics, profile)

    assert metrics["profile_total_s"] == pytest.approx(1.0)
    assert "profile_forward_backward_time_s" not in metrics
    assert "profile_forward_backward_pct" not in metrics
    assert metrics["profile_forward_avg_ms"] == pytest.approx(1.0)
    assert metrics["profile_backward_avg_ms"] == pytest.approx(2.0)
    assert metrics["profile_flat_metric_invalid_count"] == 4
    assert metrics["profile_flat_metric_invalid_fields"] == [
        "profile_forward_time_s",
        "profile_forward_pct",
        "profile_backward_time_s",
        "profile_backward_pct",
    ]


def test_model_profile_events_keep_totals_without_distribution() -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=1,
        collect_profile=True,
        profile_distribution=False,
        profile_model=True,
        profile_model_include="0",
    )

    event = metrics["profile"]["phase_events"]["backward_grad_ready"]["children"]["model.0"]
    assert event["calls"] >= 1
    assert event["total_s"] >= 0.0
    assert event["avg_ms"] >= 0.0
    assert "p95_ms" not in event
    top_event = metrics["profile"]["phase_events"]["backward_grad_ready"]["top_children"][0]
    assert "p95_ms" not in top_event


def test_phase_profiler_top_children_omit_distribution_fields_when_disabled() -> None:
    profiler = PhaseProfiler(enabled=True, track_distribution=False)

    profiler._record("forward", 0.01)
    profiler._record_detail("forward", "model.0", 0.006)
    profiler._record_event("backward_grad_ready", "model.0", 0.004)
    profile = profiler.summary()

    assert "p95_ms" not in profile["top_phases"][0]
    assert "p95_ms" not in profile["phase_breakdowns"]["forward"]["top_children"][0]
    assert "p95_ms" not in profile["phase_events"]["backward_grad_ready"]["top_children"][0]


def test_train_one_epoch_rejects_non_callable_trigger_observe() -> None:
    class TriggerWithBadObserve:
        observe = object()

        def __call__(self, _ctx: dict[str, object]) -> None:
            return None

    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(
        model,
        optimizer,
        trigger_hook=TriggerWithBadObserve(),  # type: ignore[arg-type]
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(ValueError, match="trigger_hook.observe"):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1)


@pytest.mark.parametrize(
    ("trigger_result", "match"),
    [
        ({"extra_inputs": None}, "TriggerResult or None"),
        (TriggerResult(extra_inputs=torch.randn(1, 4)), "extra_targets"),
        (TriggerResult(weights=[1.0, 1.0]), "TriggerResult.weights"),
    ],
)
def test_train_one_epoch_rejects_malformed_trigger_results(
    trigger_result: object,
    match: str,
) -> None:
    loader, model, optimizer = _make_supervised_components()

    def trigger(_ctx: dict[str, object]) -> object:
        return trigger_result

    trainer = FastTrainer(
        model,
        optimizer,
        trigger_hook=trigger,  # type: ignore[arg-type]
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    with pytest.raises(ValueError, match=match):
        trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=1)


def test_train_one_epoch_splits_warmup_and_steady_metrics() -> None:
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 3, (12,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=3,
        warmup_steps=1,
    )

    assert metrics["steps"] == 3
    assert metrics["warmup_steps"] == 1
    assert metrics["steady_steps"] == 2
    assert metrics["cold_start_steps"] == 1
    assert metrics["warmup_samples"] == 4
    assert metrics["steady_samples"] == 8
    assert metrics["cold_start_time_s"] == metrics["warmup_total_time_s"]
    assert metrics["cold_start_samples_per_sec"] == metrics["warmup_samples_per_sec"]
    assert metrics["steady_samples_per_sec"] > 0.0
    assert metrics["reported_samples_per_sec"] == metrics["steady_samples_per_sec"]
    assert metrics["compile_init_time_s"] == 0.0
    assert metrics["compile_fallback_reason"] == "cpu_device"
    assert metrics["grad_accum"] == 1
    assert metrics["partial_optimizer_steps"] == 0
    assert metrics["grad_accum_tail_steps"] == 0
    assert metrics["scheduler_step_failures"] == 0
    assert metrics["scheduler_last_error"] == ""
    assert metrics["profile_model_requested"] is False
    assert metrics["profile_model_enabled"] is False
    assert metrics["profile_model_status"] == "not_requested"
    assert metrics["profile_model_modules_selected"] == 0
    assert metrics["profile_model_hook_count"] == 0
    assert metrics["profile_model_hook_failures"] == 0
    assert metrics["profile_model_hook_last_error"] == ""
    assert metrics["warmup_avg_loss"] > 0.0
    assert metrics["steady_avg_loss"] > 0.0


def test_train_one_epoch_records_scheduler_step_failures() -> None:
    class FailingScheduler:
        def __init__(self) -> None:
            self.calls = 0

        def step(self) -> None:
            self.calls += 1
            raise RuntimeError("scheduler boom")

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = FailingScheduler()
    trainer = FastTrainer(
        model,
        optimizer,
        scheduler=scheduler,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=999,
    )

    metrics = trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=2)

    assert metrics["steps"] == 2
    assert metrics["optimizer_steps"] == 2
    assert scheduler.calls == 2
    assert metrics["scheduler_step_failures"] == 2
    assert metrics["scheduler_last_error"] == "RuntimeError: scheduler boom"


def test_train_one_epoch_flushes_partial_grad_accumulation() -> None:
    class Scale(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * self.weight

    class MeanOutputLoss(nn.Module):
        def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return outputs.mean()

    inputs = torch.ones(3, 1)
    targets = torch.zeros(3, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)
    model = Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FastTrainer(
        model,
        optimizer,
        device="cpu",
        use_amp=False,
        use_compile=False,
        grad_accum=2,
        log_interval=999,
    )

    metrics = trainer.train_one_epoch(loader, MeanOutputLoss())

    assert metrics["steps"] == 3
    assert metrics["optimizer_steps"] == 2
    assert metrics["grad_accum"] == 2
    assert metrics["partial_optimizer_steps"] == 1
    assert metrics["grad_accum_tail_steps"] == 1
    assert metrics["steady_optimizer_steps"] == 2
    assert model.weight.item() == pytest.approx(-0.2, abs=1e-6)
    assert model.weight.grad is None


def test_train_one_epoch_reports_profile_model_hook_failures() -> None:
    class ForwardHookFailingLinear(nn.Linear):
        def register_forward_hook(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("forward hook boom")

    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(ForwardHookFailingLinear(4, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=1,
        collect_profile=True,
        profile_model=True,
        profile_model_include="0",
    )

    assert metrics["profile_model_requested"] is True
    assert metrics["profile_model_enabled"] is True
    assert metrics["profile_model_status"] == "hook_failures"
    assert metrics["profile_model_modules_selected"] == 1
    assert metrics["profile_model_hook_count"] == 2
    assert metrics["profile_model_hook_failures"] == 1
    assert metrics["profile_model_hook_last_error"] == "RuntimeError: forward hook boom"
    assert not model[0]._forward_pre_hooks
    assert not model[0]._forward_hooks


def test_train_one_epoch_reports_profile_model_include_misses() -> None:
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)

    metrics = trainer.train_one_epoch(
        loader,
        nn.CrossEntropyLoss(),
        steps=1,
        collect_profile=True,
        profile_model=True,
        profile_model_include="missing.*",
    )

    assert metrics["profile_model_requested"] is True
    assert metrics["profile_model_enabled"] is True
    assert metrics["profile_model_status"] == "no_matching_modules"
    assert metrics["profile_model_modules_selected"] == 0
    assert metrics["profile_model_hook_count"] == 0
    assert metrics["profile_model_hook_failures"] == 0
    assert metrics["profile_model_hook_last_error"] == ""


def test_train_one_epoch_can_disable_step_logs_and_compile(capsys) -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(
        model,
        optimizer,
        device="cpu",
        use_amp=False,
        use_compile=False,
        log_interval=0,
    )

    metrics = trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), steps=2)
    output = capsys.readouterr().out

    assert "[train:step]" not in output
    assert metrics["compile_requested"] is False
    assert metrics["compiled"] is False
    assert metrics["compile_init_time_s"] == 0.0
    assert metrics["compile_fallback_reason"] == "not_requested"


def test_model_profile_hooks_are_removed_after_exception() -> None:
    class RaisingLoss(nn.Module):
        def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("loss failed")

    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)

    try:
        trainer.train_one_epoch(
            loader,
            RaisingLoss(),
            steps=1,
            collect_profile=True,
            profile_model=True,
            profile_model_include="0,2",
        )
    except RuntimeError as exc:
        assert "loss failed" in str(exc)
    else:
        raise AssertionError("expected training to fail")

    assert not model[0]._forward_pre_hooks
    assert not model[0]._forward_hooks
    assert not model[2]._forward_pre_hooks
    assert not model[2]._forward_hooks


def test_model_profile_hooks_look_through_compiled_wrapper() -> None:
    class CompiledLike(nn.Module):
        def __init__(self, original: nn.Module) -> None:
            super().__init__()
            self._orig_mod = original

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self._orig_mod(inputs)

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)
    trainer.model = CompiledLike(model)
    profiler = PhaseProfiler(enabled=True)

    result = trainer._install_profile_model_hooks(profiler, include="0,2")
    handles = result.handles

    try:
        assert handles
        assert result.modules_selected == 2
        assert result.hook_failures == 0
        assert result.last_error == ""
        assert model[0]._forward_pre_hooks
        assert model[2]._forward_hooks
    finally:
        for handle in handles:
            handle.remove()


def test_model_profile_hooks_accept_sequence_include() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)
    profiler = PhaseProfiler(enabled=True)

    result = trainer._install_profile_model_hooks(profiler, include=["0", "2"])
    handles = result.handles

    try:
        assert handles
        assert result.modules_selected == 2
        assert result.hook_failures == 0
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"depth": 0}, "profile_model_depth"),
        ({"max_modules": 0}, "profile_model_max_modules"),
        ({"include": [1]}, "profile_model_include"),
    ],
)
def test_model_profile_hooks_reject_invalid_direct_settings(
    kwargs: dict[str, object],
    match: str,
) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, log_interval=999)
    profiler = PhaseProfiler(enabled=True)

    with pytest.raises(ValueError, match=match):
        trainer._install_profile_model_hooks(profiler, **kwargs)  # type: ignore[arg-type]
