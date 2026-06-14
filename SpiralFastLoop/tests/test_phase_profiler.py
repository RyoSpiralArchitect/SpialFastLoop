import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
import spiralfastloop.engine as engine
from spiralfastloop.engine import (
    _add_profile_phase_metrics,
    _infer_batch_size,
    _try_infer_batch_size_with_reason,
)
from spiralfastloop.utils import PhaseProfiler


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


@pytest.mark.parametrize(
    ("batch", "reason", "match"),
    [
        (torch.tensor(1.0), "tensor_scalar", "scalar tensor"),
        (torch.empty(0, 4), "tensor_empty", "non-zero"),
        ({}, "mapping_empty", "mapping input"),
        ({"x": torch.randn(2, 4), "y": torch.randn(3, 4)}, "mapping_inconsistent", "Inconsistent"),
        ([], "sequence_empty", "Sequence batch dimension"),
        ((torch.randn(2, 4), torch.randn(3, 4)), "sequence_inconsistent", "Inconsistent"),
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


def test_phase_profiler_respects_exact_distribution_window() -> None:
    profiler = PhaseProfiler(enabled=True, window=1)

    profiler._record("forward", 0.001)
    profiler._record("forward", 0.003)
    profile = profiler.summary()

    forward = profile["phases"]["forward"]
    assert forward["calls"] == 2
    assert forward["sample_count"] == 1
    assert forward["p50_ms"] == pytest.approx(3.0)


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
    assert metrics["user_metric_valid_count"] == 2
    assert metrics["user_metric_invalid_count"] == 7
    assert metrics["user_metric_non_finite_count"] == 0
    assert metrics["user_metric_skipped_count"] == 7


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
    class CapturingLogger:
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

    inputs = torch.randn(2, 4)
    targets = torch.randint(0, 3, (2,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = CapturingLogger()
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
    assert metrics["user_metric_valid_count"] == 0
    assert metrics["user_metric_skipped_count"] == 0


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
    for phase_name in ("data_wait", "transfer", "forward", "postprocess", "collect_output", "metrics"):
        assert phase_name in phases
        assert metrics[f"profile_{phase_name}_time_s"] == pytest.approx(phases[phase_name]["total_s"])
        assert metrics[f"profile_{phase_name}_pct"] == pytest.approx(phases[phase_name]["pct"])


def test_predict_logs_postprocess_failures_before_reraising() -> None:
    class CapturingLogger:
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

    inputs = torch.randn(2, 4)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    logger = CapturingLogger()
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


def test_profile_flat_metrics_skip_invalid_values() -> None:
    metrics: dict[str, object] = {}
    profile = {
        "profile_total_s": "slow",
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
