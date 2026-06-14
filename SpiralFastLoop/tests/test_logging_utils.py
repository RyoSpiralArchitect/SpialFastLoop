from __future__ import annotations

import json
import logging

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
            "nested": {"score": torch.tensor(float("-inf"))},
        },
        step=1,
    )

    raw = jsonl_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    payload = json.loads(raw)

    assert payload["loss"] is None
    assert payload["vector"] == [1.0, None]
    assert payload["nested"] == {"score": None}
    assert payload["step"] == 1
