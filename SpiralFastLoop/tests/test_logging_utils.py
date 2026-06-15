from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import pytest
import torch

from spiralfastloop.logging_utils import MetricsLogger, default_logger


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


def test_metrics_logger_falls_back_for_malformed_metric_values_and_keys(tmp_path) -> None:
    class FailingFloat:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

        def __str__(self) -> str:
            return "failed-float"

    class MalformedRepr:
        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

        def __str__(self) -> str:
            return "malformed-repr"

    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError("path failed")

        def __str__(self) -> str:
            return "failed-path"

    class Unrepresentable:
        def __float__(self) -> float:
            raise RuntimeError("float failed")

        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

        def __str__(self) -> str:
            raise RuntimeError("str failed")

    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    metrics_logger = MetricsLogger(logger=None, jsonl_path=jsonl_path, csv_path=csv_path)

    metrics_logger.log_metrics(
        "train",
        {
            "bad_float": FailingFloat(),
            "bad_set": {MalformedRepr()},
            "bad_path": FailingPath(),
            "meta_tensor": torch.empty(2, device="meta"),
            Unrepresentable(): "key",
        },
        step=1,
    )

    payload = json.loads(jsonl_path.read_text(encoding="utf-8"))
    fallback_keys = [key for key in payload if "Unrepresentable" in key]

    assert payload["bad_float"] == "failed-float"
    assert payload["bad_set"] == ["malformed-repr"]
    assert payload["bad_path"] == "failed-path"
    assert "tensor" in payload["meta_tensor"]
    assert "meta" in payload["meta_tensor"]
    assert fallback_keys
    assert payload[fallback_keys[0]] == "key"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["bad_float"] == "failed-float"
    assert json.loads(row["bad_set"]) == ["malformed-repr"]
    assert row["bad_path"] == "failed-path"


@pytest.mark.parametrize("name", [None, True, "", "   ", object()])
def test_default_logger_rejects_invalid_names(name: object) -> None:
    with pytest.raises(ValueError, match="name"):
        default_logger(name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"logger": object()}, "logger"),
        ({"logger": type("BadLogger", (), {"log": object()})()}, "logger"),
        ({"jsonl_path": ""}, "jsonl_path"),
        ({"jsonl_path": "   "}, "jsonl_path"),
        ({"jsonl_path": object()}, "jsonl_path"),
        ({"csv_path": ""}, "csv_path"),
        ({"csv_path": "   "}, "csv_path"),
        ({"csv_path": object()}, "csv_path"),
        ({"log_level": -1}, "log_level"),
        ({"log_level": 1.5}, "log_level"),
        ({"log_level": "20"}, "log_level"),
        ({"log_level": True}, "log_level"),
        ({"is_primary": 1}, "is_primary"),
        ({"is_primary": "true"}, "is_primary"),
    ],
)
def test_metrics_logger_rejects_invalid_constructor_settings(
    kwargs: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        MetricsLogger(**kwargs)  # type: ignore[arg-type]


def test_metrics_logger_rejects_malformed_pathlike_constructor_settings() -> None:
    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError("path failed")

    with pytest.raises(ValueError, match="jsonl_path"):
        MetricsLogger(logger=None, jsonl_path=FailingPath())

    with pytest.raises(ValueError, match="csv_path"):
        MetricsLogger(logger=None, csv_path=FailingPath())


def test_metrics_logger_normalizes_pathlike_sinks_and_allows_logger_none(tmp_path) -> None:
    jsonl_path = tmp_path / "nested" / "metrics.jsonl"
    csv_path = tmp_path / "nested" / "metrics.csv"
    metrics_logger = MetricsLogger(
        logger=None,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
    )

    metrics_logger.log_metrics("train", {"loss": 1.0}, mode="epoch")

    payload = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "train"
    assert payload["mode"] == "epoch"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["loss"] == "1.0"


@pytest.mark.parametrize(
    ("stage", "mode", "metrics", "field"),
    [
        ("", "step", {"loss": 1.0}, "stage"),
        ("   ", "step", {"loss": 1.0}, "stage"),
        (True, "step", {"loss": 1.0}, "stage"),
        ("train", "", {"loss": 1.0}, "mode"),
        ("train", "   ", {"loss": 1.0}, "mode"),
        ("train", True, {"loss": 1.0}, "mode"),
        ("train", "step", object(), "metrics"),
        ("train", "step", [("loss", 1.0)], "metrics"),
    ],
)
def test_metrics_logger_rejects_invalid_payload_settings(
    stage: object,
    mode: object,
    metrics: object,
    field: str,
) -> None:
    metrics_logger = MetricsLogger(logger=None)

    with pytest.raises(ValueError, match=field):
        metrics_logger.log_metrics(
            stage,  # type: ignore[arg-type]
            metrics,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("reserved_key", ["timestamp", "stage", "mode", "step", "epoch"])
def test_metrics_logger_rejects_reserved_metric_keys_without_writing(
    tmp_path,
    reserved_key: str,
) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    metrics_logger = MetricsLogger(logger=None, jsonl_path=jsonl_path, csv_path=csv_path)

    with pytest.raises(ValueError, match="reserved payload keys"):
        metrics_logger.log_metrics("train", {reserved_key: 1.0})

    assert not jsonl_path.exists()
    assert not csv_path.exists()


def test_metrics_logger_rejects_normalized_key_collisions_without_writing(tmp_path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    metrics_logger = MetricsLogger(logger=None, jsonl_path=jsonl_path, csv_path=csv_path)

    with pytest.raises(ValueError, match="unique after normalization"):
        metrics_logger.log_metrics("train", {1: 1.0, "1": 2.0})

    assert not jsonl_path.exists()
    assert not csv_path.exists()


@pytest.mark.parametrize("metric_key", ["", "   "])
def test_metrics_logger_rejects_blank_metric_keys_without_writing(
    tmp_path,
    metric_key: str,
) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    metrics_logger = MetricsLogger(logger=None, jsonl_path=jsonl_path, csv_path=csv_path)

    with pytest.raises(ValueError, match="non-empty"):
        metrics_logger.log_metrics("train", {metric_key: 1.0})

    assert not jsonl_path.exists()
    assert not csv_path.exists()


@pytest.mark.parametrize(
    ("metrics", "match"),
    [
        ({"nested": {1: "int", "1": "str"}}, "unique after normalization at metrics.nested"),
        ({"nested": {"": 1.0}}, "non-empty after normalization at metrics.nested"),
        ({"items": [{1: "int", "1": "str"}]}, "unique after normalization at metrics.items"),
    ],
)
def test_metrics_logger_rejects_nested_metric_key_collisions_without_writing(
    tmp_path,
    metrics: dict[str, object],
    match: str,
) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    metrics_logger = MetricsLogger(logger=None, jsonl_path=jsonl_path, csv_path=csv_path)

    with pytest.raises(ValueError, match=match):
        metrics_logger.log_metrics("train", metrics)

    assert not jsonl_path.exists()
    assert not csv_path.exists()


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


def test_metrics_logger_extends_csv_fields_for_new_metrics(tmp_path) -> None:
    logger = logging.getLogger("spiralfastloop.test.csv_extend")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    csv_path = tmp_path / "metrics.csv"

    first_logger = MetricsLogger(logger=logger, csv_path=str(csv_path))
    first_logger.log_metrics("train", {"loss": 1.0}, step=0)

    second_logger = MetricsLogger(logger=logger, csv_path=str(csv_path))
    second_logger.log_metrics(
        "train",
        {"loss": 0.5, "accuracy": 0.75, "history": [1.0, float("nan")]},
        step=1,
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["loss"] == "1.0"
    assert rows[0]["accuracy"] == ""
    assert rows[0]["history"] == ""
    assert rows[1]["loss"] == "0.5"
    assert rows[1]["accuracy"] == "0.75"
    assert json.loads(rows[1]["history"]) == [1.0, None]


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ("timestamp,stage,,loss\n", "blank CSV header"),
        ("timestamp,stage,   ,loss\n", "blank CSV header"),
        ("timestamp,stage,loss,loss\n", "duplicate CSV header"),
    ],
)
def test_metrics_logger_rejects_malformed_existing_csv_headers_without_writing(
    tmp_path,
    header: str,
    match: str,
) -> None:
    csv_path = tmp_path / "metrics.csv"
    original = header + "1.0,train,step,0.5\n"
    csv_path.write_text(original, encoding="utf-8")
    metrics_logger = MetricsLogger(logger=None, csv_path=csv_path)

    with pytest.raises(ValueError, match=match):
        metrics_logger.log_metrics("train", {"loss": 0.25, "accuracy": 0.9})

    assert csv_path.read_text(encoding="utf-8") == original
