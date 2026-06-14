# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Ryō

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .utils import _non_negative_int_setting

__all__ = ["MetricsLogger", "default_logger"]


def _json_safe_metric_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe_metric_value(value.detach().cpu().item())
        return _json_safe_metric_value(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {_json_safe_metric_key(key): _json_safe_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe_metric_value(item) for item in sorted(value, key=repr)]
    try:
        return _json_safe_metric_value(float(value))
    except (TypeError, ValueError):
        return str(value)


def _json_safe_metric_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    safe_key = _json_safe_metric_value(key)
    if safe_key is None:
        return "null"
    if isinstance(safe_key, (str, int, float, bool)):
        return str(safe_key)
    return str(safe_key)


def _normalize_metric_value(value: Any) -> Any:
    return _json_safe_metric_value(value)


def _csv_safe_metric_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


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
        self._ensure_csv_fields(payload)
        assert self._csv_fields is not None
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields, extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({key: _csv_safe_metric_value(value) for key, value in payload.items()})

    def _ensure_csv_fields(self, payload: Dict[str, Any]) -> None:
        if self.csv_path is None:
            return
        if self._csv_fields is None:
            self._csv_fields = self._read_csv_header()
        if self._csv_fields is None:
            self._csv_fields = list(payload.keys())
            return
        new_fields = [key for key in payload.keys() if key not in self._csv_fields]
        if not new_fields:
            return
        self._csv_fields.extend(new_fields)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return
        with open(self.csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv_header(self) -> Optional[list[str]]:
        if self.csv_path is None:
            return None
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return None
        with open(self.csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        return list(header) if header else None

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
        normalized = {_json_safe_metric_key(k): _normalize_metric_value(v) for k, v in metrics.items()}
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
