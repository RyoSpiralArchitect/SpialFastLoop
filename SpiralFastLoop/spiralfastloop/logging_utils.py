# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .utils import _non_negative_int_setting

__all__ = ["MetricsLogger", "default_logger"]


def _json_safe_metric_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(item) for item in value]
    return value


def _normalize_metric_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe_metric_value(float(value.detach().cpu().item()))
        return _json_safe_metric_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return _json_safe_metric_value({key: _normalize_metric_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return _json_safe_metric_value([_normalize_metric_value(item) for item in value])
    if isinstance(value, (int, float, str, bool)) or value is None:
        return _json_safe_metric_value(value)
    try:
        return _json_safe_metric_value(float(value))
    except (TypeError, ValueError):
        return str(value)


def default_logger(name: str = "spiralfastloop") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class MetricsLogger:
    """Log metrics to stdlib logging and optional JSONL/CSV sinks."""

    logger: logging.Logger = field(default_factory=default_logger)
    jsonl_path: Optional[str] = None
    csv_path: Optional[str] = None
    log_level: int = logging.INFO
    is_primary: bool = True
    _csv_fields: Optional[list[str]] = field(default=None, init=False, repr=False)

    def _ensure_dir(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        if self.jsonl_path is None:
            return
        self._ensure_dir(self.jsonl_path)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    def _write_csv(self, payload: Dict[str, Any]) -> None:
        if self.csv_path is None:
            return
        self._ensure_dir(self.csv_path)
        if self._csv_fields is None:
            if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
                with open(self.csv_path, newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, None)
                    if header:
                        self._csv_fields = list(header)
            if self._csv_fields is None:
                self._csv_fields = list(payload.keys())
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields, extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(payload)

    def log_metrics(
        self,
        stage: str,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        mode: str = "step",
    ) -> None:
        if not self.is_primary:
            return
        normalized = {k: _normalize_metric_value(v) for k, v in metrics.items()}
        payload: Dict[str, Any] = {
            "timestamp": time.time(),
            "stage": stage,
            "mode": mode,
            **normalized,
        }
        if step is not None:
            step = _non_negative_int_setting(step, "step")
            payload["step"] = step
        if epoch is not None:
            epoch = _non_negative_int_setting(epoch, "epoch")
            payload["epoch"] = epoch

        if self.logger is not None:
            summary = ", ".join(f"{k}={v}" for k, v in normalized.items())
            prefix = f"[{stage}:{mode}]"
            if epoch is not None:
                prefix += f" epoch={epoch}"
            if step is not None:
                prefix += f" step={step}"
            self.logger.log(self.log_level, "%s %s", prefix, summary)
        self._write_jsonl(payload)
        self._write_csv(payload)
