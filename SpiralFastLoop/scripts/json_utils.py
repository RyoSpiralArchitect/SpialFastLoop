"""JSON helpers shared by benchmark and profiling scripts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, TextIO

import torch


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return json_safe(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {_json_safe_key(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=repr)]
    try:
        return json_safe(float(value))
    except (TypeError, ValueError):
        return str(value)


def _json_safe_key(key: Any) -> Any:
    safe_key = json_safe(key)
    if safe_key is None or isinstance(safe_key, (str, int, float, bool)):
        return safe_key
    return str(safe_key)


def dump_json(payload: Any, handle: TextIO) -> None:
    json.dump(json_safe(payload), handle, indent=2, allow_nan=False)


def dumps_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, allow_nan=False)
