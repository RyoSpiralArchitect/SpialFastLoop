"""JSON helpers shared by benchmark and profiling scripts."""

from __future__ import annotations

import json
import math
from typing import Any, TextIO


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dump_json(payload: Any, handle: TextIO) -> None:
    json.dump(json_safe(payload), handle, indent=2, allow_nan=False)


def dumps_json(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, allow_nan=False)
