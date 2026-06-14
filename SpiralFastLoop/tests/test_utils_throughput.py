import math
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiralfastloop.utils as utils_mod
from spiralfastloop.utils import (
    ThroughputMeter,
    _PSquareQuantile,
    autocast_ctx,
    dataloader_from_dataset,
    get_amp_policy,
    maybe_channels_last,
    safe_compile,
    safe_compile_with_diagnostics,
    to_device,
)


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
    assert summary["p99_s"] == pytest.approx(_percentile(durations, 0.99), rel=0.2)
    assert summary["std_batch_s"] > 0.0
    assert summary["total_time_s"] == pytest.approx(total_time, rel=1e-12)
    assert summary["avg_batch_s"] == pytest.approx(total_time / len(durations), rel=1e-6)
    assert summary["batches"] == pytest.approx(len(durations))
    assert summary["samples"] == pytest.approx(total_samples)
    assert summary["distribution_tracked"] is True


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


@pytest.mark.parametrize("time_fn", [True, 1, "clock", object()])
def test_throughput_meter_rejects_invalid_time_sources(time_fn: object):
    with pytest.raises(ValueError, match="time_fn"):
        ThroughputMeter(time_fn=time_fn)  # type: ignore[arg-type]


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), True, "1.0", b"1.0", object()])
def test_throughput_meter_rejects_invalid_initial_time_values(timestamp: object):
    with pytest.raises(ValueError, match="time_fn"):
        ThroughputMeter(time_fn=lambda: timestamp)  # type: ignore[arg-type, return-value]


@pytest.mark.parametrize("timestamp", [float("nan"), float("-inf"), True, "1.0", object()])
def test_throughput_meter_tick_rejects_invalid_time_values_without_mutating_state(timestamp: object):
    values = iter([0.0, timestamp])
    meter = ThroughputMeter(time_fn=lambda: next(values))  # type: ignore[arg-type, return-value]
    summary_before = meter.summary()
    last_before = meter.last

    with pytest.raises(ValueError, match="time_fn"):
        meter.tick(batch_size=4)

    assert meter.last == last_before
    assert meter.summary() == summary_before


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


@pytest.mark.parametrize("batch_size", [1.5, "2", True])
def test_throughput_meter_rejects_non_integral_batch_sizes(batch_size):
    meter = ThroughputMeter()

    with pytest.raises(ValueError, match="batch_size"):
        meter.record(0.1, batch_size)
    with pytest.raises(ValueError, match="batch_size"):
        meter.time_batch(batch_size)


@pytest.mark.parametrize("record_on_exception", [1, "false", None])
def test_throughput_meter_time_batch_rejects_invalid_record_on_exception(record_on_exception: object):
    meter = ThroughputMeter()

    with pytest.raises(ValueError, match="record_on_exception"):
        meter.time_batch(4, record_on_exception=record_on_exception)  # type: ignore[arg-type]


@pytest.mark.parametrize("window", [-1, 1.5, "2", True])
def test_throughput_meter_rejects_invalid_window_values(window):
    with pytest.raises(ValueError, match="window"):
        ThroughputMeter(window=window)


@pytest.mark.parametrize("smoothing", [0.0, -0.1, 1.1, float("nan"), True, "bad"])
def test_throughput_meter_rejects_invalid_smoothing_values(smoothing):
    with pytest.raises(ValueError, match="smoothing"):
        ThroughputMeter(smoothing=smoothing)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"track_distribution": 1}, "track_distribution"),
        ({"track_distribution": "false"}, "track_distribution"),
        ({"track_distribution": None}, "track_distribution"),
        ({"track_window": 0}, "track_window"),
        ({"track_window": "false"}, "track_window"),
        ({"fast_mode": 1}, "fast_mode"),
        ({"fast_mode": "true"}, "fast_mode"),
    ],
)
def test_throughput_meter_rejects_invalid_boolean_settings(kwargs, field):
    with pytest.raises(ValueError, match=field):
        ThroughputMeter(**kwargs)


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


def test_throughput_meter_can_skip_distribution_tracking():
    meter = ThroughputMeter(track_distribution=False, smoothing=None)
    durations = [0.1, 0.05, 0.08]
    for duration in durations:
        meter.record(duration, 4)

    summary = meter.summary()

    assert meter._median is None
    assert meter._p95 is None
    assert meter._p99 is None
    assert summary["p50_s"] == 0.0
    assert summary["p95_s"] == 0.0
    assert summary["p99_s"] == 0.0
    assert summary["min_batch_s"] == pytest.approx(min(durations))
    assert summary["max_batch_s"] == pytest.approx(max(durations))
    assert summary["ema_samples_per_sec"] == 0.0
    assert summary["distribution_tracked"] is False


def test_throughput_meter_can_skip_window_tracking():
    meter = ThroughputMeter(track_window=False, window=4)
    durations = [0.1, 0.2, 0.15]

    for duration in durations:
        meter.record(duration, 5)

    summary = meter.summary()

    assert summary["window_tracked"] is False
    assert summary["window_batches"] == 0.0
    assert summary["window_samples"] == 0.0
    assert summary["window_time_s"] == 0.0
    assert summary["window_samples_per_sec"] == 0.0


def test_throughput_meter_fast_mode_tracks_best_speed_and_headroom():
    meter = ThroughputMeter(fast_mode=True)
    meter.record(0.2, 4)  # 20 samples/s
    meter.record(0.1, 8)  # 80 samples/s (best)

    summary = meter.summary()

    assert meter.distribution_tracked is False
    assert meter.window_tracked is False
    assert summary["best_samples_per_sec"] == pytest.approx(80.0, rel=1e-6)
    assert summary["samples_per_sec"] == pytest.approx(12 / 0.3, rel=1e-6)
    assert summary["headroom_ratio"] == pytest.approx(80.0 / summary["samples_per_sec"], rel=1e-6)


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


def test_throughput_meter_time_batch_rejects_invalid_exit_time_without_mutating_state():
    values = iter([0.0, 0.0, "bad"])
    meter = ThroughputMeter(time_fn=lambda: next(values))  # type: ignore[arg-type, return-value]
    summary_before = meter.summary()
    last_before = meter.last

    with pytest.raises(ValueError, match="time_fn"):
        with meter.time_batch(4):
            pass

    assert meter.last == last_before
    assert meter.summary() == summary_before


def test_p_square_quantile_reports_inconsistent_state() -> None:
    quantile = _PSquareQuantile(0.5)
    quantile._q = [0.1, 0.2, 0.3, 0.4, 0.5]
    quantile._n = None
    quantile._np = [1.0, 2.0, 3.0, 4.0, 5.0]
    quantile._dn = [0.0, 0.25, 0.5, 0.75, 1.0]

    with pytest.raises(RuntimeError, match="state is inconsistent"):
        quantile.add(0.6)


def test_p_square_quantile_update_requires_initialized_state() -> None:
    quantile = _PSquareQuantile(0.5)

    with pytest.raises(RuntimeError, match="not initialized"):
        quantile._linear_update(2, 1)


@pytest.mark.parametrize(
    "quantile",
    [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5", b"0.5", object()],
)
def test_p_square_quantile_rejects_invalid_quantiles(quantile: object) -> None:
    with pytest.raises(ValueError, match="quantile"):
        _PSquareQuantile(quantile)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("-inf"), True, "0.1", b"0.1", object()])
def test_p_square_quantile_rejects_invalid_values_without_mutating_state(value: object) -> None:
    quantile = _PSquareQuantile(0.5)
    quantile.add(0.1)
    initial_before = list(quantile._initial)

    with pytest.raises(ValueError, match="value"):
        quantile.add(value)  # type: ignore[arg-type]

    assert quantile._initial == initial_before
    assert quantile._q is None


def test_safe_compile_rejects_non_module_compile_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1)

    def fake_compile(_: torch.nn.Module, mode: str) -> object:
        return object()

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    compiled, did_compile = safe_compile(model)

    assert compiled is model
    assert did_compile is False


def test_safe_compile_with_diagnostics_reports_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1)

    def fake_compile(_: torch.nn.Module, mode: str) -> object:
        raise RuntimeError("compile exploded")

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    result = safe_compile_with_diagnostics(model)

    assert result.model is model
    assert result.compiled is False
    assert result.fallback_reason == "RuntimeError: compile exploded"


def test_safe_compile_with_diagnostics_reports_non_module_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1)

    def fake_compile(_: torch.nn.Module, mode: str) -> object:
        return object()

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    result = safe_compile_with_diagnostics(model)

    assert result.model is model
    assert result.compiled is False
    assert result.fallback_reason == "non_module_result:object"


@pytest.mark.parametrize("use_amp", ["false", "auto ", 1, object()])
def test_get_amp_policy_rejects_invalid_amp_settings(use_amp: object) -> None:
    with pytest.raises(ValueError, match="use_amp"):
        get_amp_policy("cpu", use_amp=use_amp)  # type: ignore[arg-type]


def test_maybe_channels_last_rejects_invalid_boolean_setting() -> None:
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="channels_last"):
        maybe_channels_last(model, channels_last="true")  # type: ignore[arg-type]


@pytest.mark.parametrize("enabled", [1, "false", None])
def test_autocast_ctx_rejects_invalid_enabled_setting(enabled: object) -> None:
    with pytest.raises(ValueError, match="enabled"):
        autocast_ctx("cpu", enabled=enabled, amp_dtype=torch.float32)  # type: ignore[arg-type]


def test_to_device_preserves_nested_structures() -> None:
    Pair = namedtuple("Pair", ["left", "right"])
    batch = {
        "pair": Pair(torch.tensor([1.0]), (torch.tensor([2.0]),)),
        "defaults": defaultdict(lambda: torch.tensor([-1.0]), {"x": torch.tensor([3.0])}),
    }

    moved = to_device(batch, "cpu", non_blocking=False)

    assert isinstance(moved["pair"], Pair)
    assert isinstance(moved["pair"].right, tuple)
    assert isinstance(moved["defaults"], defaultdict)
    assert torch.equal(moved["pair"].left, torch.tensor([1.0]))
    assert torch.equal(moved["pair"].right[0], torch.tensor([2.0]))
    assert torch.equal(moved["defaults"]["x"], torch.tensor([3.0]))


@pytest.mark.parametrize("non_blocking", [1, "false", None])
def test_to_device_rejects_invalid_non_blocking_setting(non_blocking: object) -> None:
    with pytest.raises(ValueError, match="non_blocking"):
        to_device(torch.tensor([1.0]), "cpu", non_blocking=non_blocking)  # type: ignore[arg-type]


def test_dataloader_from_dataset_allows_zero_workers() -> None:
    dataset = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    loader = dataloader_from_dataset(
        dataset,
        batch_size=4,
        device="cpu",
        num_workers=0,
        prefetch_factor=4,
        persistent=True,
        shuffle=False,
    )

    first_inputs, first_targets = next(iter(loader))

    assert first_inputs.shape == (4, 2)
    assert first_targets.shape == (4,)


def test_dataloader_from_dataset_resolves_auto_device_for_pin_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    monkeypatch.setattr(utils_mod, "get_best_device", lambda: "cuda")

    loader = dataloader_from_dataset(
        dataset,
        batch_size=4,
        device="auto",
        num_workers=0,
        shuffle=False,
    )

    assert loader.pin_memory is True


@pytest.mark.parametrize(
    ("loader_kwargs", "match"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": 1.5}, "num_workers"),
        ({"prefetch_factor": 0}, "prefetch_factor"),
        ({"prefetch_factor": 1.5}, "prefetch_factor"),
        ({"seed": 7.5}, "seed"),
        ({"seed": "7"}, "seed"),
        ({"seed": False}, "seed"),
    ],
)
def test_dataloader_from_dataset_rejects_invalid_numeric_settings(
    loader_kwargs: dict[str, object],
    match: str,
) -> None:
    dataset = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    kwargs = {
        "batch_size": 4,
        "device": "cpu",
        "num_workers": 0,
        "shuffle": False,
        **loader_kwargs,
    }

    with pytest.raises(ValueError, match=match):
        dataloader_from_dataset(dataset, **kwargs)


@pytest.mark.parametrize(
    ("loader_kwargs", "match"),
    [
        ({"persistent": 1}, "persistent"),
        ({"persistent": "true"}, "persistent"),
        ({"pin_memory": 0}, "pin_memory"),
        ({"pin_memory": "false"}, "pin_memory"),
        ({"shuffle": 1}, "shuffle"),
        ({"shuffle": "false"}, "shuffle"),
        ({"distributed": 1}, "distributed"),
        ({"distributed": "true"}, "distributed"),
        ({"drop_last": 0}, "drop_last"),
        ({"drop_last": "false"}, "drop_last"),
    ],
)
def test_dataloader_from_dataset_rejects_invalid_boolean_settings(
    loader_kwargs: dict[str, object],
    match: str,
) -> None:
    dataset = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
    kwargs = {
        "batch_size": 4,
        "device": "cpu",
        "num_workers": 0,
        "shuffle": False,
        **loader_kwargs,
    }

    with pytest.raises(ValueError, match=match):
        dataloader_from_dataset(dataset, **kwargs)


def test_dataloader_from_dataset_applies_seed_to_shuffle_order() -> None:
    dataset = TensorDataset(torch.arange(16).float().unsqueeze(1), torch.arange(16))

    def order_for(seed: int) -> list[int]:
        loader = dataloader_from_dataset(
            dataset,
            batch_size=4,
            device="cpu",
            num_workers=0,
            shuffle=True,
            seed=seed,
        )
        return [int(value) for _, targets in loader for value in targets]

    first = order_for(7)
    second = order_for(7)
    different_seed = order_for(8)

    assert first == second
    assert first != different_seed


def test_dataloader_from_dataset_seed_works_with_mps_default_device() -> None:
    if not hasattr(torch, "set_default_device"):
        pytest.skip("torch.set_default_device is not available")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    dataset = TensorDataset(
        torch.arange(16, device="cpu").float().unsqueeze(1),
        torch.arange(16, device="cpu"),
    )
    previous_device = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    torch.set_default_device("mps")
    try:
        loader = dataloader_from_dataset(
            dataset,
            batch_size=4,
            device="cpu",
            num_workers=0,
            shuffle=True,
            seed=7,
        )
        order = [int(value) for _, targets in loader for value in targets]
    finally:
        torch.set_default_device(previous_device)

    assert sorted(order) == list(range(16))
