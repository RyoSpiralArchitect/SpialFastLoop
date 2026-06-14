from __future__ import annotations

import sys
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
