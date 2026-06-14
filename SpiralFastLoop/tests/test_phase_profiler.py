import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
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
    ("trainer_kwargs", "match"),
    [
        ({"grad_accum": 0}, "grad_accum"),
        ({"grad_accum": -1}, "grad_accum"),
        ({"log_interval": -1}, "log_interval"),
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
    ("train_kwargs", "match"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": -1}, "steps"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"steps": 1, "warmup_steps": 2}, "warmup_steps"),
        ({"profile_window": 0}, "profile_window"),
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


def test_eval_and_predict_reject_invalid_step_limits() -> None:
    loader, model, optimizer = _make_supervised_components()
    trainer = FastTrainer(model, optimizer, device="cpu", use_amp=False, use_compile=False, log_interval=999)

    with pytest.raises(ValueError, match="steps"):
        trainer.evaluate(loader, nn.CrossEntropyLoss(), steps=0)
    with pytest.raises(ValueError, match="steps"):
        trainer.predict(loader, steps=-1)


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
