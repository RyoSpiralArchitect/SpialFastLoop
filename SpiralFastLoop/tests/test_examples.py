from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import bench_synth, train_resnet


def test_bench_synth_example_emits_strict_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_synth.py",
            "--samples",
            "8",
            "--feature-dim",
            "4",
            "--hidden-dim",
            "8",
            "--classes",
            "2",
            "--batch-size",
            "4",
            "--steps",
            "1",
            "--grad-accum",
            "1",
            "--workers",
            "0",
            "--log-interval",
            "0",
            "--device",
            "cpu",
            "--no-compile",
        ],
    )

    bench_synth.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu"
    assert payload["config"]["samples"] == 8
    assert payload["metrics"]["steps"] == 1


def test_train_resnet_example_fake_dataset_emits_strict_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_resnet.py",
            "--dataset",
            "fake",
            "--samples",
            "8",
            "--feature-dim",
            "4",
            "--classes",
            "2",
            "--batch-size",
            "4",
            "--steps",
            "1",
            "--grad-accum",
            "1",
            "--workers",
            "0",
            "--log-interval",
            "0",
            "--device",
            "cpu",
            "--no-compile",
        ],
    )

    train_resnet.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu"
    assert payload["dataset"] == "fake"
    assert payload["metrics"]["steps"] == 1
