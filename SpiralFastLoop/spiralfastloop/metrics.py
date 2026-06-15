# SPDX-License-Identifier: Apache-2.0
"""Operational metrics helpers for monitoring training internals."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

from .utils import _finite_float_setting, _non_negative_int_setting

PathSetting = Union[str, os.PathLike[str]]


def _path_setting(path: Any, name: str) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"{name} must be a path string")
    try:
        normalized = os.fsdecode(path)
    except Exception as exc:
        raise ValueError(f"{name} must be a path string") from exc
    if not isinstance(normalized, str) or normalized.strip() == "":
        raise ValueError(f"{name} must be a non-empty path string")
    return normalized


def _optional_context_setting(context: Any) -> Optional[str]:
    if context is None or isinstance(context, str):
        return context
    raise ValueError("context must be a string or None")


@dataclass
class NormalizationEvent:
    """Snapshot of a single normalization pass."""

    timestamp: float
    before: float
    after: float
    context: Optional[str] = None

    @property
    def zeroed(self) -> bool:
        return self.after == 0.0

    @property
    def absolute_before(self) -> float:
        return abs(self.before)

    @property
    def absolute_after(self) -> float:
        return abs(self.after)


class NormalizationMetricsCollector:
    """Collect and aggregate normalization events.

    The collector keeps a rolling history of the most recent events so callers
    can export time-series data to their dashboarding systems while also
    exposing lightweight aggregate statistics for quick health checks.
    """

    def __init__(self, *, history_limit: int = 512) -> None:
        self.history_limit = _non_negative_int_setting(history_limit, "history_limit")
        self._history: List[NormalizationEvent] = []
        self.total_events = 0
        self.zeroed_events = 0
        self._sum_abs_before = 0.0
        self._sum_abs_after = 0.0

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record(
        self,
        before: float,
        after: float,
        *,
        context: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Register a normalization event."""

        before_value = _finite_float_setting(before, "before")
        after_value = _finite_float_setting(after, "after")
        context_value = _optional_context_setting(context)
        timestamp_value: Optional[float] = None
        if timestamp is not None:
            timestamp_value = _finite_float_setting(timestamp, "timestamp")

        self.total_events += 1
        if after_value == 0.0:
            self.zeroed_events += 1
        self._sum_abs_before += abs(before_value)
        self._sum_abs_after += abs(after_value)

        if self.history_limit == 0:
            return

        if timestamp_value is None:
            timestamp_value = time.time()
        event = NormalizationEvent(
            timestamp=timestamp_value,
            before=before_value,
            after=after_value,
            context=context_value,
        )
        self._history.append(event)
        if len(self._history) > self.history_limit:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def events(self) -> List[NormalizationEvent]:
        return [
            NormalizationEvent(
                timestamp=event.timestamp,
                before=event.before,
                after=event.after,
                context=event.context,
            )
            for event in self._history
        ]

    def summary(self) -> Dict[str, float]:
        avg_before = self._sum_abs_before / self.total_events if self.total_events else 0.0
        avg_after = self._sum_abs_after / self.total_events if self.total_events else 0.0
        zero_ratio = (self.zeroed_events / self.total_events) if self.total_events else 0.0
        return {
            "total_events": float(self.total_events),
            "zeroed_events": float(self.zeroed_events),
            "zero_ratio": zero_ratio,
            "avg_abs_before": avg_before,
            "avg_abs_after": avg_after,
        }

    def to_timeseries(self) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": event.timestamp,
                "before": event.before,
                "after": event.after,
                "abs_before": event.absolute_before,
                "abs_after": event.absolute_after,
                "zeroed": 1.0 if event.zeroed else 0.0,
                "context": event.context or "",
            }
            for event in self._history
        ]

    def report(self) -> str:
        if not self.total_events:
            return "No normalization events recorded."
        stats = self.summary()
        lines = [
            "Normalization Metrics",
            "---------------------",
            f"Events observed : {int(stats['total_events'])}",
            f"Zeroed fraction : {stats['zero_ratio']:.4f}",
            f"Avg |before|    : {stats['avg_abs_before']:.6e}",
            f"Avg |after|     : {stats['avg_abs_after']:.6e}",
        ]
        contexts = [event.context or "(unspecified)" for event in self._history]
        if contexts:
            unique_contexts = ", ".join(sorted(set(contexts)))
            lines.append(f"Contexts seen  : {unique_contexts}")
        return "\n".join(lines)

    def export_csv(self, path: PathSetting) -> None:
        """Persist the rolling history to a CSV file for dashboards."""

        normalized_path = _path_setting(path, "path")
        directory = os.path.dirname(normalized_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fieldnames = ["timestamp", "before", "after", "abs_before", "abs_after", "zeroed", "context"]
        with open(normalized_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.to_timeseries():
                writer.writerow(row)

    def merge(self, events: Iterable[NormalizationEvent]) -> None:
        try:
            iterator = iter(events)
        except Exception as exc:
            raise ValueError("events must be an iterable of NormalizationEvent items") from exc

        validated_events: List[NormalizationEvent] = []
        try:
            for index, event in enumerate(iterator):
                if not isinstance(event, NormalizationEvent):
                    raise ValueError("events must contain NormalizationEvent items")
                try:
                    timestamp_value = _finite_float_setting(event.timestamp, f"events[{index}].timestamp")
                    before_value = _finite_float_setting(event.before, f"events[{index}].before")
                    after_value = _finite_float_setting(event.after, f"events[{index}].after")
                    context_value = _optional_context_setting(event.context)
                except ValueError as exc:
                    raise ValueError(f"events[{index}] is invalid: {exc}") from exc
                validated_events.append(
                    NormalizationEvent(
                        timestamp=timestamp_value,
                        before=before_value,
                        after=after_value,
                        context=context_value,
                    )
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("events must be an iterable of NormalizationEvent items") from exc

        for event in validated_events:
            self.record(event.before, event.after, context=event.context, timestamp=event.timestamp)


GLOBAL_NORMALIZATION_METRICS = NormalizationMetricsCollector()

__all__ = [
    "GLOBAL_NORMALIZATION_METRICS",
    "NormalizationEvent",
    "NormalizationMetricsCollector",
]
