from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bench


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"d": 0},
        {"classes": 0},
        {"n": True},
        {"d": 1.5},
    ],
)
def test_synth_rejects_invalid_shape_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        bench.Synth(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d": 0},
        {"classes": 0},
        {"d": True},
        {"classes": 1.5},
    ],
)
def test_mlp_rejects_invalid_shape_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        bench.MLP(**kwargs)  # type: ignore[arg-type]


def test_best_device_accepts_supported_direct_device() -> None:
    assert bench.best_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device", ["gpu", "cuda:0", "", True])
def test_best_device_rejects_invalid_direct_device(device: object) -> None:
    with pytest.raises(ValueError, match="device"):
        bench.best_device(device)  # type: ignore[arg-type]


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, float("nan"), float("inf"), True, "0.001"])
def test_adamw_rejects_invalid_direct_learning_rates(learning_rate: object) -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(ValueError):
        bench.adamw(model.parameters(), learning_rate)  # type: ignore[arg-type]


def test_adamw_defaults_fused_false_for_cpu_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyAdamW:
        def __init__(self, parameters: object, *, lr: float, fused: bool) -> None:
            captured["parameters"] = list(parameters)  # type: ignore[arg-type]
            captured["lr"] = lr
            captured["fused"] = fused

    monkeypatch.setattr(bench.torch.optim, "AdamW", DummyAdamW)
    model = nn.Linear(2, 2)

    bench.adamw(model.parameters(), 0.01)

    assert captured["fused"] is False
    assert captured["lr"] == pytest.approx(0.01)
    assert len(captured["parameters"]) == 2


@pytest.mark.parametrize("fused", [True, False])
def test_adamw_accepts_explicit_fused_setting(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
) -> None:
    captured: dict[str, object] = {}

    class DummyAdamW:
        def __init__(self, parameters: object, *, lr: float, fused: bool) -> None:
            captured["fused"] = fused

    monkeypatch.setattr(bench.torch.optim, "AdamW", DummyAdamW)
    model = nn.Linear(2, 2)

    bench.adamw(model.parameters(), 0.01, fused=fused)

    assert captured["fused"] is fused


@pytest.mark.parametrize("fused", [1, "true"])
def test_adamw_rejects_invalid_explicit_fused_setting(fused: object) -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="fused"):
        bench.adamw(model.parameters(), 0.01, fused=fused)  # type: ignore[arg-type]


@pytest.mark.parametrize("epochs", [0, -1, 1.5, True, "1"])
def test_plain_loop_rejects_invalid_direct_epochs(epochs: object) -> None:
    dataset = TensorDataset(torch.randn(2, 2), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=1)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError):
        bench.plain_loop(
            loader,
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            epochs=epochs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("steps", [0, -1, 1.5, True, "1"])
def test_plain_loop_rejects_invalid_direct_steps(steps: object) -> None:
    dataset = TensorDataset(torch.randn(2, 2), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=1)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="steps"):
        bench.plain_loop(
            loader,
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            steps=steps,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("grad_accum", [0, -1, 1.5, True, "2"])
def test_plain_loop_rejects_invalid_direct_grad_accum(grad_accum: object) -> None:
    dataset = TensorDataset(torch.randn(2, 2), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=1)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="grad_accum"):
        bench.plain_loop(
            loader,
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            grad_accum=grad_accum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("warmup_steps", [-1, 1.5, True, "1"])
def test_plain_loop_rejects_invalid_direct_warmup_steps(warmup_steps: object) -> None:
    dataset = TensorDataset(torch.randn(2, 2), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=1)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="warmup_steps"):
        bench.plain_loop(
            loader,
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            warmup_steps=warmup_steps,  # type: ignore[arg-type]
        )


def test_plain_loop_rejects_direct_warmup_larger_than_steps() -> None:
    dataset = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=1)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="warmup_steps"):
        bench.plain_loop(
            loader,
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            steps=2,
            warmup_steps=3,
        )


def test_plain_loop_respects_step_limit_and_reports_counts() -> None:
    dataset = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = bench.plain_loop(
        loader,
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        steps=2,
    )

    assert metrics["steps"] == 2
    assert metrics["samples"] == 2
    assert metrics["optimizer_steps"] == 2
    assert metrics["grad_accum"] == 1
    assert metrics["warmup_steps"] == 0
    assert metrics["steady_steps"] == 2
    assert metrics["warmup_samples"] == 0
    assert metrics["steady_samples"] == 2
    assert metrics["warmup_optimizer_steps"] == 0
    assert metrics["steady_optimizer_steps"] == 2
    assert metrics["partial_optimizer_steps"] == 0
    assert metrics["grad_accum_tail_steps"] == 0
    assert metrics["samples_per_sec"] > 0.0
    assert metrics["steady_samples_per_sec"] > 0.0
    assert metrics["reported_samples_per_sec"] == metrics["steady_samples_per_sec"]


def test_plain_loop_flushes_partial_grad_accumulation() -> None:
    dataset = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = bench.plain_loop(
        loader,
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        steps=3,
        grad_accum=2,
    )

    assert metrics["steps"] == 3
    assert metrics["samples"] == 3
    assert metrics["optimizer_steps"] == 2
    assert metrics["grad_accum"] == 2
    assert metrics["partial_optimizer_steps"] == 1
    assert metrics["grad_accum_tail_steps"] == 1


def test_plain_loop_splits_warmup_and_reported_throughput() -> None:
    dataset = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = bench.plain_loop(
        loader,
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        steps=3,
        grad_accum=2,
        warmup_steps=1,
    )

    assert metrics["steps"] == 3
    assert metrics["samples"] == 3
    assert metrics["warmup_steps"] == 1
    assert metrics["steady_steps"] == 2
    assert metrics["warmup_samples"] == 1
    assert metrics["steady_samples"] == 2
    assert metrics["optimizer_steps"] == 2
    assert metrics["warmup_optimizer_steps"] == 0
    assert metrics["steady_optimizer_steps"] == 2
    assert metrics["partial_optimizer_steps"] == 1
    assert metrics["grad_accum_tail_steps"] == 1
    assert metrics["warmup_samples_per_sec"] > 0.0
    assert metrics["steady_samples_per_sec"] > 0.0
    assert metrics["cold_start_steps"] == metrics["warmup_steps"]
    assert metrics["cold_start_time_s"] == metrics["warmup_total_time_s"]
    assert metrics["cold_start_samples_per_sec"] == metrics["warmup_samples_per_sec"]
    assert metrics["warmup_elapsed_sec"] == metrics["warmup_total_time_s"]
    assert metrics["steady_elapsed_sec"] == metrics["steady_total_time_s"]
    assert metrics["reported_samples_per_sec"] == metrics["steady_samples_per_sec"]


def test_bench_parse_args_accepts_valid_minimal_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--samples",
            "8",
            "--feature-dim",
            "4",
            "--classes",
            "2",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--warmup-steps",
            "1",
            "--grad-accum",
            "1",
            "--workers",
            "0",
            "--learning-rate",
            "0.001",
            "--log-interval",
            "0",
        ],
    )

    args = bench.parse_args()

    assert args.samples == 8
    assert args.steps == 1
    assert args.warmup_steps == 1
    assert args.grad_accum == 1
    assert args.workers == 0
    assert args.learning_rate == pytest.approx(0.001)


@pytest.mark.parametrize(
    "argv",
    [
        ["bench.py", "--samples", "0"],
        ["bench.py", "--feature-dim", "-1"],
        ["bench.py", "--classes", "0"],
        ["bench.py", "--batch-size", "1.5"],
        ["bench.py", "--steps", "0"],
        ["bench.py", "--warmup-steps", "-1"],
        ["bench.py", "--grad-accum", "0"],
        ["bench.py", "--workers", "-1"],
        ["bench.py", "--learning-rate", "nan"],
        ["bench.py", "--device", "gpu"],
        ["bench.py", "--log-interval", "-1"],
    ],
)
def test_bench_parse_args_rejects_invalid_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        bench.parse_args()

    assert exc_info.value.code == 2


def test_bench_parse_args_rejects_warmup_larger_than_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["bench.py", "--steps", "2", "--warmup-steps", "3"])

    with pytest.raises(SystemExit) as exc_info:
        bench.parse_args()

    assert exc_info.value.code == 2
