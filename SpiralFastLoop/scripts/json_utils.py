"""JSON helpers shared by benchmark and profiling scripts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, TextIO

import torch


def json_safe(value: Any, path: str = "$") -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item(), path)
        return json_safe(value.detach().cpu().tolist(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, path)
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[]") for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item, f"{path}[]") for item in sorted(value, key=repr)]
    try:
        return json_safe(float(value), path)
    except (OverflowError, TypeError, ValueError):
        return str(value)


def _json_safe_mapping(value: Mapping[Any, Any], path: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    blank_keys = 0
    duplicate_keys: list[str] = []
    for key, item in value.items():
        safe_key = _json_safe_key(key)
        if not safe_key.strip():
            blank_keys += 1
            continue
        if safe_key in normalized:
            duplicate_keys.append(safe_key)
            continue
        normalized[safe_key] = json_safe(item, f"{path}.{safe_key}")
    if blank_keys:
        raise ValueError(f"JSON object keys must be non-empty after normalization at {path}")
    if duplicate_keys:
        names = ", ".join(sorted(set(duplicate_keys)))
        raise ValueError(f"JSON object keys must be unique after normalization at {path}: {names}")
    return normalized


def _json_safe_key(key: Any) -> str:
    safe_key = json_safe(key)
    if safe_key is None:
        return "null"
    if isinstance(safe_key, (str, int, float, bool)):
        return str(safe_key)
    return str(safe_key)


def dump_json(payload: Any, handle: TextIO) -> None:
    json.dump(json_safe(payload), handle, indent=2, allow_nan=False)


def dumps_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, allow_nan=False)
