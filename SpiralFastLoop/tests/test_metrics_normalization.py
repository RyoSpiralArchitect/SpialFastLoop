import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop.metrics import (
    GLOBAL_NORMALIZATION_METRICS,
    NormalizationEvent,
    NormalizationMetricsCollector,
)


class _FailingEventIterable:
    def __iter__(self):
        yield NormalizationEvent(timestamp=5.0, before=0.2, after=0.0, context="valid")
        raise RuntimeError("iteration failed")


class _BrokenEventIterable:
    def __iter__(self):
        raise RuntimeError("iterator creation failed")


@pytest.mark.parametrize("history_limit", [-1, 1.5, "2", True])
def test_collector_rejects_invalid_history_limit(history_limit: object):
    with pytest.raises(ValueError, match="history_limit"):
        NormalizationMetricsCollector(history_limit=history_limit)  # type: ignore[arg-type]


def test_collector_tracks_events_and_summary(tmp_path):
    collector = NormalizationMetricsCollector(history_limit=4)
    collector.record(1.0, 0.0, context="buffer", timestamp=1.0)
    collector.record(-0.25, -0.25, context="credit", timestamp=2.0)
    collector.record(0.5, 0.0, context="carry", timestamp=3.0)

    summary = collector.summary()
    assert summary["total_events"] == 3.0
    assert summary["zeroed_events"] == 2.0
    assert summary["zero_ratio"] == 2.0 / 3.0
    assert summary["avg_abs_before"] == (1.0 + 0.25 + 0.5) / 3.0
    assert summary["avg_abs_after"] == (0.0 + 0.25 + 0.0) / 3.0

    timeseries = collector.to_timeseries()
    assert len(timeseries) == 3
    assert timeseries[0]["context"] == "buffer"

    out_path = tmp_path / "events.csv"
    collector.export_csv(str(out_path))
    with out_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["context"] == "buffer"


def test_collector_events_returns_snapshots_not_live_history():
    collector = NormalizationMetricsCollector(history_limit=2)
    collector.record(1.0, 0.0, context="buffer", timestamp=1.0)

    event = collector.events()[0]
    event.before = 99.0
    event.after = 88.0
    event.context = "mutated"

    timeseries = collector.to_timeseries()
    assert timeseries[0]["before"] == 1.0
    assert timeseries[0]["after"] == 0.0
    assert timeseries[0]["context"] == "buffer"


def test_collector_export_csv_creates_parent_dirs_and_accepts_pathlike(tmp_path):
    collector = NormalizationMetricsCollector(history_limit=2)
    collector.record(1.0, 0.0, context="正規化", timestamp=1.0)
    out_path = tmp_path / "artifacts" / "normalization" / "events.csv"

    collector.export_csv(out_path)

    with out_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["context"] == "正規化"


@pytest.mark.parametrize("path", [None, True, 1, "", "   ", b"events.csv"])
def test_collector_export_csv_rejects_invalid_paths(tmp_path, path: object):
    collector = NormalizationMetricsCollector(history_limit=2)
    collector.record(1.0, 0.0, context="buffer", timestamp=1.0)

    with pytest.raises(ValueError, match="path"):
        collector.export_csv(path)  # type: ignore[arg-type]

    assert list(tmp_path.rglob("*.csv")) == []


def test_collector_export_csv_rejects_malformed_pathlike(tmp_path):
    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError("path failed")

    collector = NormalizationMetricsCollector(history_limit=2)
    collector.record(1.0, 0.0, context="buffer", timestamp=1.0)

    with pytest.raises(ValueError, match="path"):
        collector.export_csv(FailingPath())

    assert list(tmp_path.rglob("*.csv")) == []


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"before": float("nan"), "after": 0.0}, "before"),
        ({"before": float("inf"), "after": 0.0}, "before"),
        ({"before": True, "after": 0.0}, "before"),
        ({"before": object(), "after": 0.0}, "before"),
        ({"before": 1.0, "after": float("nan")}, "after"),
        ({"before": 1.0, "after": float("-inf")}, "after"),
        ({"before": 1.0, "after": True}, "after"),
        ({"before": 1.0, "after": 0.0, "timestamp": float("nan")}, "timestamp"),
        ({"before": 1.0, "after": 0.0, "timestamp": True}, "timestamp"),
        ({"before": 1.0, "after": 0.0, "context": True}, "context"),
        ({"before": 1.0, "after": 0.0, "context": object()}, "context"),
    ],
)
def test_collector_rejects_invalid_events_before_mutating_state(kwargs, field):
    collector = NormalizationMetricsCollector(history_limit=4)

    with pytest.raises(ValueError, match=field):
        collector.record(**kwargs)

    assert collector.summary() == {
        "total_events": 0.0,
        "zeroed_events": 0.0,
        "zero_ratio": 0.0,
        "avg_abs_before": 0.0,
        "avg_abs_after": 0.0,
    }
    assert collector.events() == []


def test_collector_can_merge_events():
    left = NormalizationMetricsCollector(history_limit=2)
    right = NormalizationMetricsCollector(history_limit=2)
    left.record(0.2, 0.0, context="left", timestamp=5.0)
    left.record(0.1, 0.1, context="left", timestamp=6.0)
    right.merge(left.events())
    assert right.summary()["total_events"] == 2.0
    assert right.events()[0].context == "left"


@pytest.mark.parametrize(
    "events",
    [
        object(),
        "not events",
        [
            NormalizationEvent(timestamp=5.0, before=0.2, after=0.0, context="valid"),
            object(),
        ],
        [
            NormalizationEvent(timestamp=5.0, before=0.2, after=0.0, context="valid"),
            NormalizationEvent(timestamp=float("nan"), before=0.1, after=0.0, context="bad"),
        ],
        [
            NormalizationEvent(timestamp=5.0, before=0.2, after=0.0, context="valid"),
            NormalizationEvent(timestamp=6.0, before=True, after=0.0, context="bad"),
        ],
        [
            NormalizationEvent(timestamp=5.0, before=0.2, after=0.0, context="valid"),
            NormalizationEvent(timestamp=6.0, before=0.1, after=0.0, context=True),
        ],
        _FailingEventIterable(),
        _BrokenEventIterable(),
    ],
)
def test_collector_merge_rejects_invalid_events_without_mutating_state(events: object):
    collector = NormalizationMetricsCollector(history_limit=4)
    collector.record(1.0, 0.5, context="baseline", timestamp=1.0)
    before_summary = collector.summary()
    before_events = collector.events()

    with pytest.raises(ValueError, match="events"):
        collector.merge(events)  # type: ignore[arg-type]

    assert collector.summary() == before_summary
    assert collector.events() == before_events


def test_global_collector_is_shared_singleton():
    first = GLOBAL_NORMALIZATION_METRICS
    second = GLOBAL_NORMALIZATION_METRICS
    assert first is second
    before = second.summary()["total_events"]
    first.record(0.3, 0.0, context="singleton", timestamp=10.0)
    after = second.summary()["total_events"]
    assert after >= before + 1.0
