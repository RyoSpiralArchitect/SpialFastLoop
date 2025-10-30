import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop.utils import ThroughputMeter


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round(percentile * (len(ordered) - 1)))
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def test_throughput_meter_matches_percentiles_with_stream_data():
    meter = ThroughputMeter()
    durations = [0.011, 0.014, 0.009, 0.021, 0.017, 0.019, 0.016, 0.023, 0.018, 0.022]
    batch_sizes = [8, 8, 16, 16, 8, 32, 8, 8, 16, 32]

    for duration, batch_size in zip(durations, batch_sizes):
        meter.record(duration, batch_size)

    summary = meter.summary()

    total_samples = sum(batch_sizes)
    total_time = math.fsum(durations)

    assert summary["samples_per_sec"] == pytest.approx(total_samples / total_time, rel=1e-6)
    assert summary["p50_s"] == pytest.approx(_percentile(durations, 0.5), rel=0.05)
    assert summary["p95_s"] == pytest.approx(_percentile(durations, 0.95), rel=0.2)
    assert summary["total_time_s"] == pytest.approx(total_time, rel=1e-6)
    assert summary["batches"] == pytest.approx(len(durations), rel=1e-6)
    assert summary["last_duration_s"] == pytest.approx(durations[-1], rel=1e-6)


def test_throughput_meter_allows_custom_time_source():
    calls = []

    def fake_time() -> float:
        base = 10.0
        value = base + len(calls) * 0.01
        calls.append(value)
        return value

    meter = ThroughputMeter(time_fn=fake_time)
    # first tick updates internal last timestamp
    meter.tick(batch_size=16)
    meter.tick(batch_size=16)

    summary = meter.summary()
    assert summary["samples_per_sec"] > 0.0
    assert summary["batches"] == pytest.approx(2.0, rel=1e-6)


def test_throughput_meter_rejects_invalid_inputs():
    meter = ThroughputMeter()

    with pytest.raises(ValueError):
        meter.record(-0.01, 4)

    with pytest.raises(ValueError):
        meter.record(0.01, -1)

    with pytest.raises(ValueError):
        meter.record(float("nan"), 1)
