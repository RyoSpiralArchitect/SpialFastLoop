import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop.metrics import (
    GLOBAL_NORMALIZATION_METRICS,
    NormalizationMetricsCollector,
)


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


def test_collector_can_merge_events():
    left = NormalizationMetricsCollector(history_limit=2)
    right = NormalizationMetricsCollector(history_limit=2)
    left.record(0.2, 0.0, context="left", timestamp=5.0)
    left.record(0.1, 0.1, context="left", timestamp=6.0)
    right.merge(left.events())
    assert right.summary()["total_events"] == 2.0
    assert right.events()[0].context == "left"


def test_global_collector_is_shared_singleton():
    first = GLOBAL_NORMALIZATION_METRICS
    second = GLOBAL_NORMALIZATION_METRICS
    assert first is second
    before = second.summary()["total_events"]
    first.record(0.3, 0.0, context="singleton", timestamp=10.0)
    after = second.summary()["total_events"]
    assert after >= before + 1.0
