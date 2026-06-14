from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest
import torch

from spiralfastloop.logging_utils import MetricsLogger


def test_metrics_logger_writes_strict_jsonl_for_non_finite_metrics(tmp_path) -> None:
    logger = logging.getLogger("spiralfastloop.test.strict_jsonl")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    jsonl_path = tmp_path / "metrics.jsonl"

    metrics_logger = MetricsLogger(logger=logger, jsonl_path=str(jsonl_path))
    metrics_logger.log_metrics(
        "train",
        {
            "loss": float("nan"),
            "vector": torch.tensor([1.0, float("inf")]),
            "nested": {
                "score": torch.tensor(float("-inf")),
                "path": Path("artifacts") / "metrics.jsonl",
                "tags": {"beta", "alpha"},
                ("tuple", "key"): "value",
            },
            ("top", "tuple"): torch.tensor(3.0),
        },
        step=1,
    )

    raw = jsonl_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    payload = json.loads(raw)

    assert payload["loss"] is None
    assert payload["vector"] == [1.0, None]
    assert payload["nested"] == {
        "score": None,
        "path": "artifacts/metrics.jsonl",
        "tags": ["alpha", "beta"],
        "['tuple', 'key']": "value",
    }
    assert payload["['top', 'tuple']"] == 3.0
    assert payload["step"] == 1


@pytest.mark.parametrize("step", [-1, 1.5, "2", True])
def test_metrics_logger_rejects_invalid_step_values(step: object) -> None:
    logger = logging.getLogger("spiralfastloop.test.invalid_step")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    metrics_logger = MetricsLogger(logger=logger)

    with pytest.raises(ValueError, match="step"):
        metrics_logger.log_metrics(
            "train",
            {"loss": 1.0},
            step=step,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("epoch", [-1, 1.5, "2", True])
def test_metrics_logger_rejects_invalid_epoch_values(epoch: object) -> None:
    logger = logging.getLogger("spiralfastloop.test.invalid_epoch")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    metrics_logger = MetricsLogger(logger=logger)

    with pytest.raises(ValueError, match="epoch"):
        metrics_logger.log_metrics(
            "train",
            {"loss": 1.0},
            epoch=epoch,  # type: ignore[arg-type]
        )


def test_metrics_logger_keeps_non_negative_step_and_epoch_in_jsonl(tmp_path) -> None:
    logger = logging.getLogger("spiralfastloop.test.step_epoch_jsonl")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    jsonl_path = tmp_path / "metrics.jsonl"
    metrics_logger = MetricsLogger(logger=logger, jsonl_path=str(jsonl_path))

    metrics_logger.log_metrics("train", {"loss": 1.0}, step=0, epoch=0)

    payload = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert payload["step"] == 0
    assert payload["epoch"] == 0


def test_metrics_logger_writes_nested_csv_values_as_json_strings(tmp_path) -> None:
    logger = logging.getLogger("spiralfastloop.test.csv_nested")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    csv_path = tmp_path / "metrics.csv"
    metrics_logger = MetricsLogger(logger=logger, csv_path=str(csv_path))

    metrics_logger.log_metrics(
        "train",
        {
            "vector": torch.tensor([1.0, float("inf")]),
            "nested": {"tags": {"beta", "alpha"}},
            "path": Path("artifacts") / "metrics.csv",
        },
        step=0,
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert json.loads(row["vector"]) == [1.0, None]
    assert json.loads(row["nested"]) == {"tags": ["alpha", "beta"]}
    assert row["path"] == "artifacts/metrics.csv"
