from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import profile_resnet_drilldown as drilldown


@pytest.mark.parametrize(
    ("size", "image_size", "classes"),
    [
        (0, 32, 2),
        (4, 0, 2),
        (4, 32, 0),
        (True, 32, 2),
        (4, 1.5, 2),
    ],
)
def test_build_fake_dataset_rejects_invalid_shape_values(
    size: object,
    image_size: object,
    classes: object,
) -> None:
    with pytest.raises(ValueError):
        drilldown._build_fake_dataset(size, image_size, classes)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_classes", [0, -1, 1.5, True, "2"])
def test_build_resnet18_rejects_invalid_num_classes_before_import(num_classes: object) -> None:
    with pytest.raises(ValueError, match="num_classes"):
        drilldown._build_resnet18(num_classes)  # type: ignore[arg-type]


def test_build_dataset_rejects_unknown_direct_dataset() -> None:
    args = Namespace(
        dataset="imagenet",
        dataset_size=8,
        image_size=32,
        num_classes=2,
    )

    with pytest.raises(ValueError, match="dataset"):
        drilldown._build_dataset(args)


@pytest.mark.parametrize(
    ("dataset_size", "batch_size", "match"),
    [
        (0, 1, "dataset_size"),
        (1.5, 1, "dataset_size"),
        ("2", 1, "dataset_size"),
        (True, 1, "dataset_size"),
        (4, 0, "batch_size"),
        (4, 1.5, "batch_size"),
        (4, "2", "batch_size"),
        (4, True, "batch_size"),
    ],
)
def test_validate_resnet_profile_args_rejects_invalid_direct_sizes(
    dataset_size: object,
    batch_size: object,
    match: str,
) -> None:
    args = Namespace(
        dataset="fake",
        dataset_size=dataset_size,
        batch_size=batch_size,
        steps=1,
        warmup_steps=0,
    )

    with pytest.raises(ValueError, match=match):
        drilldown.validate_resnet_profile_args(args)


def test_validate_resnet_profile_args_rejects_empty_fake_batches() -> None:
    args = Namespace(
        dataset="fake",
        dataset_size=4,
        batch_size=8,
        steps=1,
        warmup_steps=0,
    )

    with pytest.raises(ValueError, match="dataset-size"):
        drilldown.validate_resnet_profile_args(args)


def test_validate_resnet_profile_args_rejects_unknown_direct_dataset() -> None:
    args = Namespace(
        dataset="imagenet",
        dataset_size=8,
        batch_size=4,
        steps=1,
        warmup_steps=0,
    )

    with pytest.raises(ValueError, match="dataset"):
        drilldown.validate_resnet_profile_args(args)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("image_size", 0, "image_size"),
        ("image_size", 1.5, "image_size"),
        ("image_size", True, "image_size"),
        ("topk", 0, "topk"),
        ("topk", "2", "topk"),
        ("topk", True, "topk"),
        ("download", 1, "download"),
        ("download", "true", "download"),
    ],
)
def test_validate_resnet_profile_args_rejects_invalid_optional_direct_values(
    field: str,
    value: object,
    match: str,
) -> None:
    args = Namespace(
        dataset="fake",
        dataset_size=8,
        batch_size=4,
        steps=1,
        warmup_steps=0,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=match):
        drilldown.validate_resnet_profile_args(args)


@pytest.mark.parametrize("topk", [0, 1.5, True, "2"])
def test_print_summary_rejects_invalid_direct_topk(topk: object) -> None:
    with pytest.raises(ValueError, match="topk"):
        drilldown._print_summary({"samples_per_sec": 1.0}, topk=topk)  # type: ignore[arg-type]


def test_print_summary_formats_malformed_metrics_as_na(
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = {
        "reported_samples_per_sec": "fast",
        "samples_per_sec": float("inf"),
        "warmup_steps": 1,
        "cold_start_steps": True,
        "cold_start_time_s": None,
        "cold_start_samples_per_sec": float("nan"),
        "steady_steps": "2",
        "steady_samples_per_sec": None,
        "steady_p99_s": "slow",
        "p99_s": None,
        "std_batch_s": float("-inf"),
        "steps": True,
        "samples": "many",
        "profile": {
            "top_phases": [
                "skip-me",
                {"name": "forward", "pct": float("nan"), "avg_ms": "slow"},
                {"pct": 12.0, "avg_ms": 1.0},
            ],
            "phase_breakdowns": {
                "forward": {
                    "top_children": [
                        {"name": "conv", "pct_of_parent": None, "avg_ms": float("inf"), "p95_ms": 1.0},
                    ],
                },
                "optimizer": {
                    "top_children": [
                        {"name": "adamw", "pct_of_parent": True, "avg_ms": 0.5},
                    ],
                },
            },
            "phase_events": {
                "backward_grad_ready": {
                    "top_children": [
                        {"name": "layer4", "avg_ms": "slow", "p95_ms": float("nan")},
                    ],
                },
            },
        },
    }

    drilldown._print_summary(metrics, topk=4)

    output = capsys.readouterr().out
    assert "samples_per_sec=n/a total=n/a" in output
    assert "cold_start_steps=n/a cold_start_time_s=n/a cold_start_samples_per_sec=n/a" in output
    assert "steady_steps=2 steady_samples_per_sec=n/a steady_p99_ms=n/a" in output
    assert "batch_latency_p99_ms=n/a batch_latency_std_ms=n/a" in output
    assert "steps=n/a samples=n/a" in output
    assert "forward: n/a avg=n/a" in output
    assert "<unnamed>: 12.0% avg=1.00ms" in output
    assert "conv: n/a avg=n/a p95=1.00ms" in output
    assert "layer4: avg=n/a p95=n/a" in output
    assert "adamw: n/a avg=0.50ms" in output


def test_print_summary_ignores_malformed_profile_container(
    capsys: pytest.CaptureFixture[str],
) -> None:
    drilldown._print_summary({"samples_per_sec": 1.0, "profile": "bad-profile"}, topk=2)

    output = capsys.readouterr().out
    assert "samples_per_sec=1.0 total=1.0" in output
    assert "top phases:" not in output


def test_resnet_drilldown_parse_args_accepts_valid_fake_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_resnet_drilldown.py",
            "--dataset",
            "fake",
            "--dataset-size",
            "8",
            "--image-size",
            "32",
            "--num-classes",
            "2",
            "--batch-size",
            "4",
            "--grad-accum",
            "1",
            "--workers",
            "0",
            "--prefetch-factor",
            "2",
            "--steps",
            "1",
            "--warmup-steps",
            "1",
            "--learning-rate",
            "0.001",
            "--profile-window",
            "8",
            "--profile-model-depth",
            "1",
            "--profile-model-max-modules",
            "4",
            "--topk",
            "2",
        ],
    )

    args = drilldown.parse_args()

    assert args.dataset == "fake"
    assert args.dataset_size == 8
    assert args.batch_size == 4
    assert args.steps == 1
    assert args.warmup_steps == 1
    assert args.topk == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["profile_resnet_drilldown.py", "--dataset-size", "0"],
        ["profile_resnet_drilldown.py", "--image-size", "0"],
        ["profile_resnet_drilldown.py", "--num-classes", "0"],
        ["profile_resnet_drilldown.py", "--batch-size", "0"],
        ["profile_resnet_drilldown.py", "--grad-accum", "0"],
        ["profile_resnet_drilldown.py", "--workers", "-1"],
        ["profile_resnet_drilldown.py", "--prefetch-factor", "0"],
        ["profile_resnet_drilldown.py", "--steps", "0"],
        ["profile_resnet_drilldown.py", "--warmup-steps", "-1"],
        ["profile_resnet_drilldown.py", "--learning-rate", "nan"],
        ["profile_resnet_drilldown.py", "--device", "gpu"],
        ["profile_resnet_drilldown.py", "--profile-window", "0"],
        ["profile_resnet_drilldown.py", "--profile-model-depth", "0"],
        ["profile_resnet_drilldown.py", "--profile-model-max-modules", "0"],
        ["profile_resnet_drilldown.py", "--topk", "0"],
    ],
)
def test_resnet_drilldown_parse_args_rejects_invalid_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        drilldown.parse_args()

    assert exc_info.value.code == 2


def test_resnet_drilldown_parse_args_rejects_warmup_larger_than_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_resnet_drilldown.py", "--steps", "2", "--warmup-steps", "3"],
    )

    with pytest.raises(SystemExit) as exc_info:
        drilldown.parse_args()

    assert exc_info.value.code == 2


def test_resnet_drilldown_parse_args_rejects_empty_fake_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_resnet_drilldown.py",
            "--dataset",
            "fake",
            "--dataset-size",
            "4",
            "--batch-size",
            "8",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        drilldown.parse_args()

    assert exc_info.value.code == 2


def test_resnet_drilldown_allows_small_dataset_for_cifar10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_resnet_drilldown.py",
            "--dataset",
            "cifar10",
            "--dataset-size",
            "4",
            "--batch-size",
            "8",
        ],
    )

    args = drilldown.parse_args()

    assert args.dataset == "cifar10"
    assert args.dataset_size == 4
    assert args.batch_size == 8
