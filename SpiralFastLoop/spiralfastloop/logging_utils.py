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
from typing import Any, Dict, Optional, cast

import torch

from .utils import (
    _bool_setting,
    _non_empty_string_setting,
    _non_negative_int_setting,
)

__all__ = ["MetricsLogger", "default_logger"]


_RESERVED_PAYLOAD_KEYS = frozenset({"timestamp", "stage", "mode", "step", "epoch"})


def _fallback_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _string_fallback(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        try:
            return repr(value)
        except Exception:
            return f"<unrepresentable {_fallback_type_name(value)}>"


def _json_safe_metric_tensor(value: torch.Tensor, path: str) -> Any:
    try:
        normalized = value.detach().cpu()
        if normalized.numel() == 1:
            return _json_safe_metric_value(normalized.item(), path)
        return _json_safe_metric_value(normalized.tolist(), path)
    except Exception:
        return _string_fallback(value)


def _json_safe_metric_value(value: Any, path: str = "metrics") -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe_metric_tensor(value, path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        try:
            return os.fsdecode(value)
        except Exception:
            return _string_fallback(value)
    if isinstance(value, Mapping):
        return _normalize_metric_mapping(value, path=path)
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(item, f"{path}[]") for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe_metric_value(item, f"{path}[]") for item in sorted(value, key=_string_fallback)]
    try:
        return _json_safe_metric_value(float(value), path)
    except Exception:
        return _string_fallback(value)


def _json_safe_metric_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    safe_key = _json_safe_metric_value(key)
    if safe_key is None:
        return "null"
    if isinstance(safe_key, (str, int, float, bool)):
        return str(safe_key)
    return _string_fallback(safe_key)


def _normalize_metric_value(value: Any) -> Any:
    return _json_safe_metric_value(value)


def _normalize_metric_mapping(
    metrics: Mapping[Any, Any],
    *,
    path: str,
    reserved_key_set: frozenset[str] = frozenset(),
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    blank_keys = 0
    reserved_found: list[str] = []
    duplicate_keys: list[str] = []
    for key, value in metrics.items():
        normalized_key = _json_safe_metric_key(key)
        if not normalized_key.strip():
            blank_keys += 1
            continue
        if normalized_key in reserved_key_set:
            reserved_found.append(normalized_key)
            continue
        if normalized_key in normalized:
            duplicate_keys.append(normalized_key)
            continue
        normalized[normalized_key] = _json_safe_metric_value(value, f"{path}.{normalized_key}")
    if blank_keys:
        raise ValueError(f"metrics keys must be non-empty after normalization at {path}")
    if reserved_found:
        names = ", ".join(sorted(set(reserved_found)))
        raise ValueError(f"metrics must not contain reserved payload keys: {names}")
    if duplicate_keys:
        names = ", ".join(sorted(set(duplicate_keys)))
        raise ValueError(f"metrics keys must be unique after normalization at {path}: {names}")
    return normalized


def _normalize_metric_payload(metrics: Mapping[Any, Any]) -> Dict[str, Any]:
    return _normalize_metric_mapping(
        metrics,
        path="metrics",
        reserved_key_set=_RESERVED_PAYLOAD_KEYS,
    )


def _csv_header_fields(header: list[str], path: str) -> list[str]:
    blank_fields = [index for index, field in enumerate(header) if not field.strip()]
    if blank_fields:
        raise ValueError(f"csv_path has blank CSV header fields in {path}")
    seen = set()
    duplicate_fields = []
    for field_name in header:
        if field_name in seen:
            duplicate_fields.append(field_name)
        else:
            seen.add(field_name)
    if duplicate_fields:
        names = ", ".join(sorted(set(duplicate_fields)))
        raise ValueError(f"csv_path has duplicate CSV header fields in {path}: {names}")
    return header


def _csv_safe_metric_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _optional_path_setting(path: Any, name: str) -> Optional[str]:
    if path is None:
        return None
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"{name} must be a non-empty filesystem path or None")
    try:
        normalized = os.fspath(path)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty filesystem path or None") from exc
    if not isinstance(normalized, str) or not normalized.strip():
        raise ValueError(f"{name} must be a non-empty filesystem path or None")
    return normalized


def _optional_logger_setting(logger: Any) -> Optional[logging.Logger]:
    if logger is None:
        return None
    log = getattr(logger, "log", None)
    if not callable(log):
        raise ValueError("logger must provide a callable log() method or be None")
    return cast(logging.Logger, logger)


def default_logger(name: str = "spiralfastloop") -> logging.Logger:
    logger = logging.getLogger(_non_empty_string_setting(name, "name"))
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

    logger: Optional[logging.Logger] = field(default_factory=default_logger)
    jsonl_path: Optional[str] = None
    csv_path: Optional[str] = None
    log_level: int = logging.INFO
    is_primary: bool = True
    _csv_fields: Optional[list[str]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = _optional_logger_setting(self.logger)
        self.jsonl_path = _optional_path_setting(self.jsonl_path, "jsonl_path")
        self.csv_path = _optional_path_setting(self.csv_path, "csv_path")
        self.log_level = _non_negative_int_setting(self.log_level, "log_level")
        self.is_primary = _bool_setting(self.is_primary, "is_primary")

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
        if not header:
            return None
        return _csv_header_fields(list(header), self.csv_path)

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
        stage_value = _non_empty_string_setting(stage, "stage")
        mode_value = _non_empty_string_setting(mode, "mode")
        if not isinstance(metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        normalized = _normalize_metric_payload(metrics)
        payload: Dict[str, Any] = {
            "timestamp": time.time(),
            "stage": stage_value,
            "mode": mode_value,
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
            prefix = f"[{stage_value}:{mode_value}]"
            if epoch is not None:
                prefix += f" epoch={epoch}"
            if step is not None:
                prefix += f" step={step}"
            self.logger.log(self.log_level, "%s %s", prefix, summary)
        self._write_jsonl(payload)
        self._write_csv(payload)
