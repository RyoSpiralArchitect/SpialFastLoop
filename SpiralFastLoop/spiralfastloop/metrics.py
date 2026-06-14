# SPDX-License-Identifier: Apache-2.0
"""Operational metrics helpers for monitoring training internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import csv
import time

from .utils import _finite_float_setting, _non_negative_int_setting


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
            context=context,
        )
        self._history.append(event)
        if len(self._history) > self.history_limit:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def events(self) -> List[NormalizationEvent]:
        return list(self._history)

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

    def export_csv(self, path: str) -> None:
        """Persist the rolling history to a CSV file for dashboards."""

        fieldnames = ["timestamp", "before", "after", "abs_before", "abs_after", "zeroed", "context"]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.to_timeseries():
                writer.writerow(row)

    def merge(self, events: Iterable[NormalizationEvent]) -> None:
        for event in events:
            self.record(event.before, event.after, context=event.context, timestamp=event.timestamp)


GLOBAL_NORMALIZATION_METRICS = NormalizationMetricsCollector()

__all__ = [
    "GLOBAL_NORMALIZATION_METRICS",
    "NormalizationEvent",
    "NormalizationMetricsCollector",
]
