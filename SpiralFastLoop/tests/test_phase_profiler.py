import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop import FastTrainer
from spiralfastloop.utils import PhaseProfiler


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
    assert metrics["warmup_avg_loss"] > 0.0
    assert metrics["steady_avg_loss"] > 0.0


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

    handles = trainer._install_profile_model_hooks(profiler, include="0,2")

    try:
        assert handles
        assert model[0]._forward_pre_hooks
        assert model[2]._forward_hooks
    finally:
        for handle in handles:
            handle.remove()
