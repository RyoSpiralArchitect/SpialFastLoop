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
    assert summary["total_time_s"] == pytest.approx(total_time, rel=1e-12)
    assert summary["avg_batch_s"] == pytest.approx(total_time / len(durations), rel=1e-6)
    assert summary["batches"] == pytest.approx(len(durations))
    assert summary["samples"] == pytest.approx(total_samples)


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


def test_throughput_meter_rejects_invalid_inputs():
    meter = ThroughputMeter()

    with pytest.raises(ValueError):
        meter.record(-0.1, 8)
    with pytest.raises(ValueError):
        meter.record(float("nan"), 8)
    with pytest.raises(ValueError):
        meter.record(0.1, 0)
    with pytest.raises(ValueError):
        meter.record(0.1, -5)


def test_throughput_meter_reset_clears_state():
    meter = ThroughputMeter()
    meter.record(0.1, 8)
    meter.record(0.2, 8)

    meter.reset()

    summary = meter.summary()
    assert summary["samples_per_sec"] == 0.0
    assert summary["total_time_s"] == 0.0
    assert summary["samples"] == 0.0
    assert summary["batches"] == 0.0


def test_throughput_meter_tracks_window_and_extrema():
    meter = ThroughputMeter(smoothing=0.5, window=3)
    durations = [0.1, 0.2, 0.05, 0.15]
    for duration in durations:
        meter.record(duration, 10)

    summary = meter.summary()

    assert summary["last_batch_s"] == pytest.approx(0.15, rel=1e-6)
    assert summary["min_batch_s"] == pytest.approx(0.05, rel=1e-6)
    assert summary["max_batch_s"] == pytest.approx(0.2, rel=1e-6)
    assert summary["window_batches"] == pytest.approx(3)
    assert summary["window_samples"] == pytest.approx(30)
    assert summary["window_time_s"] == pytest.approx(0.4, rel=1e-6)
    assert summary["window_samples_per_sec"] == pytest.approx(75.0, rel=1e-6)
    assert summary["ema_samples_per_sec"] == pytest.approx(102.0833, rel=1e-4)


def test_throughput_meter_time_batch_context_records_and_handles_exceptions():
    class FakeClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            return self.value

        def advance(self, delta: float) -> None:
            self.value += delta

    clock = FakeClock()
    meter = ThroughputMeter(time_fn=clock, window=4)

    with meter.time_batch(8):
        clock.advance(0.05)

    with pytest.raises(RuntimeError):
        with meter.time_batch(4):
            clock.advance(0.03)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        with meter.time_batch(2, record_on_exception=True):
            clock.advance(0.02)
            raise RuntimeError("still boom")

    summary = meter.summary()

    assert summary["batches"] == pytest.approx(2)
    assert summary["samples"] == pytest.approx(10)
    assert summary["total_time_s"] == pytest.approx(0.07, rel=1e-12)
    assert summary["samples_per_sec"] == pytest.approx(142.857142857, rel=1e-9)
    assert summary["last_batch_s"] == pytest.approx(0.02, rel=1e-9)
    assert summary["min_batch_s"] == pytest.approx(0.02, rel=1e-9)
    assert summary["max_batch_s"] == pytest.approx(0.05, rel=1e-9)
    assert summary["window_batches"] == pytest.approx(2)
