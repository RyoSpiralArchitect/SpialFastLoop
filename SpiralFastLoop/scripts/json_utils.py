"""JSON helpers shared by benchmark and profiling scripts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, TextIO

import torch


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


def _json_safe_tensor(value: torch.Tensor, path: str) -> Any:
    try:
        normalized = value.detach().cpu()
        if normalized.numel() == 1:
            return json_safe(normalized.item(), path)
        return json_safe(normalized.tolist(), path)
    except Exception:
        return _string_fallback(value)


def json_safe(value: Any, path: str = "$") -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe_tensor(value, path)
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
        return _json_safe_mapping(value, path)
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[]") for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item, f"{path}[]") for item in sorted(value, key=_string_fallback)]
    try:
        return json_safe(float(value), path)
    except Exception:
        return _string_fallback(value)


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
    return _string_fallback(safe_key)


def dump_json(payload: Any, handle: TextIO) -> None:
    json.dump(json_safe(payload), handle, indent=2, allow_nan=False)


def dumps_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, allow_nan=False)
