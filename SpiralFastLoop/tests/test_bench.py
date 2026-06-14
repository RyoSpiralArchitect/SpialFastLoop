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


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, float("nan"), float("inf"), True, "0.001"])
def test_adamw_rejects_invalid_direct_learning_rates(learning_rate: object) -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(ValueError):
        bench.adamw(model.parameters(), learning_rate)  # type: ignore[arg-type]


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
        ["bench.py", "--workers", "-1"],
        ["bench.py", "--learning-rate", "nan"],
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
