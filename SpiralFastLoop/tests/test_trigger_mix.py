from __future__ import annotations

from collections import namedtuple
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop.engine import (
    TriggerResult,
    _concatenate_batches,
    _ensure_loss_vector,
    _infer_batch_size,
)
from spiralfastloop.extras.trigger_mix import (
    COEFVAR_STABILIZER,
    FRACTION_NORMALIZATION_EPS,
    HardSampleBuffer,
    HardSampleProvider,
    LossStdConfig,
    LossStdTrigger,
)

_FIXTURES_DIR = Path(__file__).resolve().parent
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.append(str(_FIXTURES_DIR))

from fixtures import RoundingRegressionCase


def _make_provider(outputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    calls: Dict[str, list[int]] = {"requested": []}

    def provider(k: int, device: str, ctx):
        calls["requested"].append(k)
        batch_shape = (k, 2)
        inputs = torch.full(batch_shape, 1.0, device=device)
        targets = torch.zeros(k, device=device)
        if outputs is not None:
            return outputs
        return inputs, targets

    provider.calls = calls  # type: ignore[attr-defined]
    return provider


@pytest.mark.parametrize("max_samples", [-1, 1.5, "2", True])
def test_hard_sample_buffer_rejects_invalid_max_samples(max_samples: object) -> None:
    with pytest.raises(ValueError, match="max_samples"):
        HardSampleBuffer(max_samples=max_samples)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "2", True])
def test_hard_sample_buffer_rejects_invalid_top_k(top_k: object) -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    targets = torch.arange(3)
    losses = torch.ones(3)

    with pytest.raises(ValueError, match="top_k"):
        buffer.add_batch(inputs, targets, losses, top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "loss_vec, error_type, match",
    [
        (None, TypeError, "loss_vec"),
        ([1.0, 2.0, 3.0], TypeError, "loss_vec"),
        (torch.tensor([1, 2, 3]), ValueError, "floating-point"),
        (torch.tensor([True, False, True]), ValueError, "floating-point"),
        (torch.tensor([1.0, float("nan"), 3.0]), ValueError, "finite"),
        (torch.tensor([1.0, float("inf"), 3.0]), ValueError, "finite"),
    ],
)
def test_hard_sample_buffer_rejects_invalid_loss_vectors(
    loss_vec: object,
    error_type: type[Exception],
    match: str,
) -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    targets = torch.arange(3)

    with pytest.raises(error_type, match=match):
        buffer.add_batch(inputs, targets, loss_vec)  # type: ignore[arg-type]
    assert len(buffer) == 0


@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        ([], torch.arange(3)),
        (torch.arange(6, dtype=torch.float32).reshape(3, 2), []),
        (torch.arange(6, dtype=torch.float32).reshape(3, 2), torch.arange(2)),
        ({}, torch.arange(3)),
        ({"x": torch.arange(6, dtype=torch.float32).reshape(3, 2)}, {}),
        ({"nested": {}}, torch.arange(3)),
    ],
)
def test_hard_sample_buffer_rejects_misaligned_batch_structures_without_mutating(
    inputs: object,
    targets: object,
) -> None:
    buffer = HardSampleBuffer(max_samples=8)
    losses = torch.tensor([3.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="inputs and targets"):
        buffer.add_batch(inputs, targets, losses)

    assert len(buffer) == 0


@pytest.mark.parametrize("num_samples", [0, -1, 1.5, "2", True])
def test_hard_sample_buffer_rejects_invalid_sample_request(num_samples: object) -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    targets = torch.arange(3)
    losses = torch.ones(3)
    buffer.add_batch(inputs, targets, losses, top_k=2)

    with pytest.raises(ValueError, match="num_samples"):
        buffer.sample(num_samples)  # type: ignore[arg-type]


def test_hard_sample_buffer_samples_tensor_inputs_with_non_tensor_targets() -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    targets = ["easy", "medium", "hard"]
    losses = torch.tensor([1.0, 3.0, 2.0])

    buffer.add_batch(inputs, targets, losses)
    sampled_inputs, sampled_targets = buffer.sample(5)

    assert sampled_inputs.shape == (5, 2)
    assert isinstance(sampled_targets, list)
    assert len(sampled_targets) == 5
    assert set(sampled_targets).issubset(set(targets))


def test_hard_sample_buffer_samples_nested_batch_structures() -> None:
    pair = namedtuple("Pair", ["primary", "aux"])
    buffer = HardSampleBuffer(max_samples=8)
    inputs = {
        "dense": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "pair": pair(torch.arange(3, dtype=torch.float32), torch.ones((3, 1))),
    }
    targets = {
        "label": torch.arange(3),
        "weight": [torch.ones(3), torch.arange(3, dtype=torch.float32)],
    }
    losses = torch.tensor([1.0, 3.0, 2.0])

    buffer.add_batch(inputs, targets, losses)
    sampled_inputs, sampled_targets = buffer.sample(5)

    assert sampled_inputs["dense"].shape == (5, 2)
    assert sampled_inputs["pair"].primary.shape == (5,)
    assert sampled_inputs["pair"].aux.shape == (5, 1)
    assert sampled_targets["label"].shape == (5,)
    assert sampled_targets["weight"][0].shape == (5,)
    assert sampled_targets["weight"][1].shape == (5,)


def test_hard_sample_buffer_rejects_cross_batch_structure_changes_without_mutating() -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    losses = torch.tensor([1.0, 3.0, 2.0])
    buffer.add_batch(inputs, ["easy", "medium", "hard"], losses)

    with pytest.raises(ValueError, match="hard samples"):
        buffer.add_batch(inputs, torch.arange(3), losses)

    assert len(buffer) == 3
    sampled_inputs, sampled_targets = buffer.sample(5)
    assert sampled_inputs.shape == (5, 2)
    assert isinstance(sampled_targets, list)
    assert all(isinstance(target, str) for target in sampled_targets)


def test_hard_sample_buffer_rejects_internal_sample_structure_mismatch() -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    targets = [{"a": 0}, {"b": 1}, {"a": 2}]
    losses = torch.tensor([1.0, 3.0, 2.0])

    with pytest.raises(ValueError, match="hard samples"):
        buffer.add_batch(inputs, targets, losses)

    assert len(buffer) == 0


def test_hard_sample_buffer_accepts_mapping_key_order_variation() -> None:
    buffer = HardSampleBuffer(max_samples=8)
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    targets = [{"a": 0, "b": 0}, {"b": 1, "a": 1}, {"a": 2, "b": 2}]
    losses = torch.tensor([1.0, 3.0, 2.0])

    buffer.add_batch(inputs, targets, losses)
    sampled_inputs, sampled_targets = buffer.sample(5)

    assert sampled_inputs.shape == (5, 2)
    assert set(sampled_targets.keys()) == {"a", "b"}
    assert len(sampled_targets["a"]) == 5
    assert len(sampled_targets["b"]) == 5


@pytest.mark.parametrize("select_top_k", [0, -1, 1.5, "2", True])
def test_hard_sample_provider_rejects_invalid_select_top_k(select_top_k: object) -> None:
    with pytest.raises(ValueError, match="select_top_k"):
        HardSampleProvider(
            HardSampleBuffer(),
            select_top_k=select_top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("buffer", [None, True, object()])
def test_hard_sample_provider_rejects_invalid_buffers(buffer: object) -> None:
    with pytest.raises(ValueError, match="buffer"):
        HardSampleProvider(buffer)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"augmenter": True}, "augmenter"),
        ({"augmenter": object()}, "augmenter"),
        ({"fallback": True}, "fallback"),
        ({"fallback": object()}, "fallback"),
    ],
)
def test_hard_sample_provider_rejects_invalid_callables(
    kwargs: Dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        HardSampleProvider(HardSampleBuffer(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("requested", [0, -1, 1.5, "2", True])
def test_hard_sample_provider_rejects_invalid_direct_requests(requested: object) -> None:
    calls: list[int] = []

    def fallback(k: int, device: str, ctx: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
        calls.append(k)
        return torch.zeros(k, 2, device=device), torch.zeros(k, device=device)

    provider = HardSampleProvider(HardSampleBuffer(), fallback=fallback)

    with pytest.raises(ValueError, match="requested"):
        provider(requested, "cpu", {})  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.parametrize("device", [None, True, 1, "", "   ", object()])
def test_hard_sample_provider_rejects_invalid_direct_devices(device: object) -> None:
    provider = HardSampleProvider(HardSampleBuffer(), fallback=_make_provider())

    with pytest.raises(ValueError, match="device"):
        provider(1, device, {})  # type: ignore[arg-type]


def test_hard_sample_provider_rejects_invalid_direct_contexts() -> None:
    provider = HardSampleProvider(HardSampleBuffer(), fallback=_make_provider())

    with pytest.raises(ValueError, match="ctx"):
        provider(1, "cpu", object())  # type: ignore[arg-type]


def test_hard_sample_provider_observe_requires_context_for_tensor_losses() -> None:
    provider = HardSampleProvider(HardSampleBuffer())

    with pytest.raises(ValueError, match="ctx"):
        provider.observe(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inputs and targets"):
        provider.observe({"loss_vec": torch.ones(2)})


def test_hard_sample_provider_observe_rejects_non_finite_losses_without_mutating() -> None:
    buffer = HardSampleBuffer(max_samples=8)
    provider = HardSampleProvider(buffer)
    ctx = {
        "inputs": torch.zeros(2, 2),
        "targets": torch.zeros(2),
        "loss_vec": torch.tensor([1.0, float("nan")]),
    }

    with pytest.raises(ValueError, match="loss_vec"):
        provider.observe(ctx)
    assert len(buffer) == 0


@pytest.mark.parametrize(
    "loss_vec, match",
    [
        (torch.tensor([1, 2]), "floating-point"),
        (torch.tensor([True, False]), "floating-point"),
        (torch.tensor([1.0, float("inf")]), "finite"),
    ],
)
def test_hard_sample_provider_observe_rejects_invalid_loss_tensor_values(
    loss_vec: torch.Tensor,
    match: str,
) -> None:
    buffer = HardSampleBuffer(max_samples=8)
    provider = HardSampleProvider(buffer)
    ctx = {
        "inputs": torch.zeros(2, 2),
        "targets": torch.zeros(2),
        "loss_vec": loss_vec,
    }

    with pytest.raises(ValueError, match=match):
        provider.observe(ctx)
    assert len(buffer) == 0


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"std_threshold": -0.1}, "std_threshold"),
        ({"std_threshold": float("nan")}, "std_threshold"),
        ({"inject_ratio": -0.1}, "inject_ratio"),
        ({"inject_ratio": float("inf")}, "inject_ratio"),
        ({"weight_alpha": -0.1}, "weight_alpha"),
        ({"weight_alpha": True}, "weight_alpha"),
        ({"budget_frac": -0.1}, "budget_frac"),
        ({"budget_frac": float("nan")}, "budget_frac"),
        ({"pulse_every": -1}, "pulse_every"),
        ({"pulse_every": 1.5}, "pulse_every"),
        ({"pulse_every": "2"}, "pulse_every"),
        ({"pulse_every": True}, "pulse_every"),
        ({"max_injected_per_step": -1}, "max_injected_per_step"),
        ({"max_injected_per_step": 1.5}, "max_injected_per_step"),
        ({"max_injected_per_step": "2"}, "max_injected_per_step"),
        ({"max_injected_per_step": True}, "max_injected_per_step"),
    ],
)
def test_loss_std_config_rejects_invalid_numeric_settings(
    kwargs: Dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        LossStdConfig(**kwargs)  # type: ignore[arg-type]


def test_loss_std_config_allows_zero_disable_values() -> None:
    cfg = LossStdConfig(
        inject_ratio=0.0,
        budget_frac=0.0,
        pulse_every=0,
        max_injected_per_step=0,
    )

    assert cfg.inject_ratio == 0.0
    assert cfg.budget_frac == 0.0
    assert cfg.pulse_every == 0
    assert cfg.max_injected_per_step == 0


@pytest.mark.parametrize("provider", [None, True, object()])
def test_trigger_rejects_invalid_providers(provider: object) -> None:
    with pytest.raises(ValueError, match="provider"):
        LossStdTrigger(provider=provider)  # type: ignore[arg-type]


def test_trigger_rejects_invalid_config_objects() -> None:
    with pytest.raises(ValueError, match="cfg"):
        LossStdTrigger(provider=_make_provider(), cfg=object())  # type: ignore[arg-type]


def test_trigger_rejects_invalid_normalization_collectors() -> None:
    with pytest.raises(ValueError, match="normalization_metrics"):
        LossStdTrigger(
            provider=_make_provider(),
            normalization_metrics=object(),  # type: ignore[arg-type]
        )


def test_trigger_observe_rejects_non_callable_provider_observe() -> None:
    class ProviderWithBadObserve:
        observe = object()

        def __call__(
            self,
            k: int,
            device: str,
            ctx: Dict[str, object],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(k, 2, device=device), torch.zeros(k, device=device)

    trigger = LossStdTrigger(provider=ProviderWithBadObserve())

    with pytest.raises(ValueError, match="provider.observe"):
        trigger.observe({})


@pytest.mark.parametrize("step", [-1, 1.5, "2", True])
def test_trigger_rejects_invalid_step_values(step: object) -> None:
    trigger = LossStdTrigger(provider=_make_provider())
    ctx = {"loss_vec": torch.ones(4), "device": "cpu", "step": step}

    with pytest.raises(ValueError, match="step"):
        trigger(ctx)


def test_trigger_rejects_invalid_context_shapes() -> None:
    trigger = LossStdTrigger(provider=_make_provider())

    with pytest.raises(ValueError, match="ctx"):
        trigger(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="loss_vec"):
        trigger({"device": "cpu"})
    with pytest.raises(ValueError, match="device"):
        trigger({"loss_vec": torch.ones(2)})
    with pytest.raises(ValueError, match="device"):
        trigger({"loss_vec": torch.ones(2), "device": ""})


def test_trigger_rejects_non_finite_losses_without_mutating_state() -> None:
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider)
    ctx = {"loss_vec": torch.tensor([1.0, float("inf")]), "device": "cpu", "step": 1}

    with pytest.raises(ValueError, match="loss_vec"):
        trigger(ctx)

    assert provider.calls["requested"] == []
    assert trigger.total == 0
    assert trigger.spent == 0
    assert trigger._last_step is None
    assert trigger._budget_buffer == 0.0


@pytest.mark.parametrize(
    "loss_vec, match",
    [
        (torch.tensor([1, 2]), "floating-point"),
        (torch.tensor([True, False]), "floating-point"),
        (torch.tensor([1.0, float("nan")]), "finite"),
    ],
)
def test_trigger_rejects_invalid_loss_tensor_values_without_mutating_state(
    loss_vec: torch.Tensor,
    match: str,
) -> None:
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider)
    ctx = {"loss_vec": loss_vec, "device": "cpu", "step": 1}

    with pytest.raises(ValueError, match=match):
        trigger(ctx)

    assert provider.calls["requested"] == []
    assert trigger.total == 0
    assert trigger.spent == 0
    assert trigger._last_step is None


def test_trigger_provider_failures_do_not_mutate_budget_state() -> None:
    def provider(k: int, device: str, ctx: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("provider failed")

    cfg = LossStdConfig(std_threshold=10.0, inject_ratio=0.5, budget_frac=1.0)
    trigger = LossStdTrigger(provider=provider, cfg=cfg)
    trigger.total = 30
    trigger.spent = 3
    trigger._last_step = 7
    trigger._last_pulse_step = 6
    trigger._budget_buffer = 0.25
    ctx = {"loss_vec": torch.ones(4), "device": "cpu", "step": 1}

    with pytest.raises(RuntimeError, match="provider failed"):
        trigger(ctx)

    assert trigger.total == 30
    assert trigger.spent == 3
    assert trigger._last_step == 7
    assert trigger._last_pulse_step == 6
    assert trigger._budget_buffer == pytest.approx(0.25)


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, 0.0),
        (FRACTION_NORMALIZATION_EPS * 0.5, 0.0),
        (-FRACTION_NORMALIZATION_EPS * 0.5, 0.0),
        (FRACTION_NORMALIZATION_EPS, FRACTION_NORMALIZATION_EPS),
        (-FRACTION_NORMALIZATION_EPS, -FRACTION_NORMALIZATION_EPS),
        (FRACTION_NORMALIZATION_EPS * 8, FRACTION_NORMALIZATION_EPS * 8),
        (-FRACTION_NORMALIZATION_EPS * 8, -FRACTION_NORMALIZATION_EPS * 8),
        (1e6, 1e6),
        (-1e6, -1e6),
    ],
)
def test_drop_rounding_noise_handles_signed_residue(value: float, expected: float) -> None:
    trigger = LossStdTrigger(provider=_make_provider())
    result = trigger._drop_rounding_noise(value)
    if expected == 0.0:
        assert result == 0.0
    else:
        assert result == pytest.approx(expected)


def test_drop_rounding_noise_regressions(rounding_cases: Iterable[RoundingRegressionCase]) -> None:
    trigger = LossStdTrigger(provider=_make_provider())
    seen = [trigger._drop_rounding_noise(case.value) for case in rounding_cases]
    for case, observed in zip(rounding_cases, seen):
        if case.expected == 0.0:
            assert observed == 0.0
        else:
            assert observed == pytest.approx(case.expected)


def test_drop_rounding_noise_matches_previous_logic() -> None:
    trigger = LossStdTrigger(provider=_make_provider())

    def legacy_drop(val: float) -> float:
        return 0.0 if abs(val) < 1e-12 else val

    samples = [
        -1e-14,
        -FRACTION_NORMALIZATION_EPS,
        -FRACTION_NORMALIZATION_EPS * 2.3,
        -1.0,
        0.0,
        FRACTION_NORMALIZATION_EPS * 0.25,
        FRACTION_NORMALIZATION_EPS,
        FRACTION_NORMALIZATION_EPS * 3.1,
        42.0,
    ]
    for sample in samples:
        assert trigger._drop_rounding_noise(sample) == legacy_drop(sample)


def test_trigger_skips_when_variability_high() -> None:
    cfg = LossStdConfig(std_threshold=0.1, inject_ratio=0.5, pulse_every=10, budget_frac=1.0)
    trigger = LossStdTrigger(provider=_make_provider(), cfg=cfg)
    ctx = {"loss_vec": torch.tensor([0.0, 2.0]), "device": "cpu", "step": 1}
    result = trigger(ctx)
    assert result is None


def test_trigger_requests_extra_samples_on_low_std() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.5,
        weight_alpha=1.7,
        pulse_every=1000,
        budget_frac=1.0,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)
    ctx = {"loss_vec": torch.ones(4), "device": "cpu", "step": 5}
    result = trigger(ctx)
    assert isinstance(result, TriggerResult)
    assert provider.calls["requested"] == [2]
    assert result.weights.shape[0] == 6
    assert torch.allclose(result.weights[-2:], torch.full((2,), 1.7))


def test_pulse_fires_even_when_variance_high() -> None:
    cfg = LossStdConfig(
        std_threshold=0.0,
        inject_ratio=0.25,
        pulse_every=2,
        budget_frac=1.0,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)
    ctx = {"loss_vec": torch.tensor([0.0, 4.0, 8.0, 12.0]), "device": "cpu", "step": 2}
    result = trigger(ctx)
    assert isinstance(result, TriggerResult)
    assert provider.calls["requested"] == [1]
    assert result.weights.shape[0] == 5
    assert result.weights[-1].item() == pytest.approx(cfg.weight_alpha)


def test_forced_pulse_only_attempts_once_when_budget_blocked() -> None:
    cfg = LossStdConfig(
        std_threshold=0.0,
        inject_ratio=0.5,
        pulse_every=2,
        budget_frac=0.0,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"loss_vec": torch.linspace(0, 1, 6), "device": "cpu", "step": 2}
    assert trigger(ctx) is None

    ctx["step"] = 2
    assert trigger(ctx) is None
    assert provider.calls["requested"] == []


def test_pulse_only_triggers_once_per_step() -> None:
    cfg = LossStdConfig(
        std_threshold=0.0,
        inject_ratio=0.5,
        pulse_every=2,
        budget_frac=1.0,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"loss_vec": torch.linspace(0, 1, 6), "device": "cpu", "step": 2}
    first = trigger(ctx)
    assert isinstance(first, TriggerResult)
    assert provider.calls["requested"] == [3]

    ctx["loss_vec"] = torch.linspace(0, 1, 6)
    ctx["step"] = 2
    second = trigger(ctx)
    assert second is None
    assert provider.calls["requested"] == [3]


def test_budget_fraction_limits_total_injections() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.6,
        pulse_every=1000,
        budget_frac=0.1,
        max_injected_per_step=16,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"device": "cpu"}
    for idx, batch_losses in enumerate((torch.ones(5), torch.ones(5), torch.ones(5)), start=1):
        ctx["loss_vec"] = batch_losses
        ctx["step"] = idx
        trigger(ctx)

    assert trigger.spent == 2
    assert trigger.total == 15
    assert trigger.spent <= math.ceil(trigger.cfg.budget_frac * trigger.total)
    assert provider.calls["requested"] == [1, 1]


def test_budget_counters_reset_on_epoch_restart() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.5,
        pulse_every=1000,
        budget_frac=0.2,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"device": "cpu", "loss_vec": torch.ones(20)}

    ctx["step"] = 1
    first = trigger(ctx)
    assert isinstance(first, TriggerResult)
    assert provider.calls["requested"] == [4]
    assert trigger.total == 20
    assert trigger.spent == 4

    ctx["step"] = 0  # simulate a new epoch (step counter reset)
    second = trigger(ctx)
    assert isinstance(second, TriggerResult)
    assert provider.calls["requested"] == [4, 4]
    assert trigger.total == 20
    assert trigger.spent == 4


def test_budget_counters_ignore_repeated_steps() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.5,
        pulse_every=1000,
        budget_frac=1.0,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"device": "cpu", "loss_vec": torch.ones(12)}

    ctx["step"] = 5
    first = trigger(ctx)
    assert isinstance(first, TriggerResult)
    assert provider.calls["requested"] == [6]
    assert trigger.total == 12
    assert trigger.spent == 6

    ctx["step"] = 5  # repeated step should not reset counters
    second = trigger(ctx)
    assert isinstance(second, TriggerResult)
    assert provider.calls["requested"] == [6, 6]
    assert trigger.total == 24
    assert trigger.spent == 12


def test_fractional_budget_accumulates_until_whole_sample() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.6,
        pulse_every=1000,
        budget_frac=0.05,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"device": "cpu", "loss_vec": torch.ones(2)}
    for step in range(1, 4):
        ctx["step"] = step
        assert trigger(ctx) is None
        assert provider.calls["requested"] == []
    assert trigger._budget_buffer == pytest.approx(0.6, abs=1e-6)

    ctx["step"] = 4
    result = trigger(ctx)
    assert isinstance(result, TriggerResult)
    assert provider.calls["requested"] == [1]
    assert trigger.spent == 1
    assert trigger.total == 8
    assert trigger._budget_buffer == pytest.approx(0.0, abs=1e-6)

    ctx["step"] = 5
    assert trigger(ctx) is None
    assert provider.calls["requested"] == [1]


def test_fractional_carry_only_tracks_excess_credit_after_clipping() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.6,
        pulse_every=1000,
        budget_frac=0.05,
        max_injected_per_step=2,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"device": "cpu", "loss_vec": torch.ones(2)}
    for step in range(1, 4):
        ctx["step"] = step
        assert trigger(ctx) is None
    assert trigger._budget_buffer == pytest.approx(0.6, abs=1e-6)

    ctx.update({"step": 4, "loss_vec": torch.ones(30)})
    result = trigger(ctx)
    assert isinstance(result, TriggerResult)
    assert provider.calls["requested"] == [2]
    assert trigger.spent == 2
    assert trigger.total == 36
    assert trigger._budget_buffer == pytest.approx(0.4, abs=1e-6)


def test_fractional_buffer_does_not_hold_whole_units() -> None:
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.2,
        pulse_every=1000,
        budget_frac=1.0,
        max_injected_per_step=128,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"device": "cpu", "loss_vec": torch.ones(50), "step": 1}
    result = trigger(ctx)

    assert isinstance(result, TriggerResult)
    assert provider.calls["requested"] == [10]
    assert trigger.spent == 10
    assert trigger.total == 50
    assert 0.0 <= trigger._budget_buffer < 1.0


def test_pulse_resets_after_step_decrease() -> None:
    cfg = LossStdConfig(
        std_threshold=0.0,
        inject_ratio=0.5,
        pulse_every=2,
        budget_frac=1.0,
        max_injected_per_step=10,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"loss_vec": torch.linspace(0, 1, 6), "device": "cpu"}

    ctx["step"] = 2
    first = trigger(ctx)
    assert isinstance(first, TriggerResult)
    assert provider.calls["requested"] == [3]

    ctx["step"] = 1
    assert trigger(ctx) is None

    ctx["step"] = 2
    third = trigger(ctx)
    assert isinstance(third, TriggerResult)
    assert provider.calls["requested"] == [3, 3]


def test_near_zero_mean_losses_still_trigger_injection() -> None:
    provider = _make_provider()
    cfg = LossStdConfig(
        std_threshold=0.2,
        inject_ratio=0.5,
        weight_alpha=1.7,
        pulse_every=1000,
        budget_frac=1.0,
        max_injected_per_step=4,
    )
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    tiny = torch.tensor([1e-10, -1e-10], dtype=torch.float64, device="cpu")
    ctx = {"loss_vec": tiny, "device": "cpu", "step": 1}

    expected_coefvar = tiny.std(unbiased=False) / (tiny.mean().abs() + COEFVAR_STABILIZER)
    assert expected_coefvar.item() <= cfg.std_threshold

    result = trigger(ctx)
    assert isinstance(result, TriggerResult)
    assert provider.calls["requested"] == [1]
    assert result.weights.shape[0] == 3


def test_concatenate_batches_preserves_nested_structure() -> None:
    base = {"a": torch.zeros((2, 3)), "b": [torch.ones((2,))]}
    extra = {"a": torch.ones((1, 3)), "b": [torch.zeros((1,))]}
    merged = _concatenate_batches(base, extra)
    assert isinstance(merged, dict)
    assert merged["a"].shape == (3, 3)
    assert torch.allclose(merged["a"][-1], torch.ones(3))
    assert merged["b"][0].shape == (3,)


def test_concatenate_batches_mismatched_keys_raises() -> None:
    base = {"a": torch.zeros((2, 3))}
    extra = {"b": torch.zeros((1, 3))}
    with pytest.raises(KeyError):
        _concatenate_batches(base, extra)


def test_concatenate_batches_rejects_sequence_type_mismatch() -> None:
    with pytest.raises(TypeError, match="list structure"):
        _concatenate_batches([torch.zeros((2, 3))], (torch.ones((1, 3)),))


def test_concatenate_batches_rejects_namedtuple_type_mismatch() -> None:
    pair = namedtuple("Pair", ["left", "right"])
    other_pair = namedtuple("OtherPair", ["left", "right"])
    base = pair(torch.zeros((2, 3)), torch.ones((2, 1)))
    extra = other_pair(torch.ones((1, 3)), torch.zeros((1, 1)))

    with pytest.raises(TypeError, match="namedtuple structure"):
        _concatenate_batches(base, extra)


def test_infer_batch_size_handles_sequences() -> None:
    batch = (torch.zeros((4, 5)), [torch.ones((4,))], {"x": torch.zeros((4, 2))})
    size = _infer_batch_size(batch)
    assert size == 4


def test_ensure_loss_vector_handles_scalars_and_large_values() -> None:
    scalar = torch.tensor(3.14)
    vector = _ensure_loss_vector(scalar)
    assert vector.shape == (1,)
    wide = torch.ones((8, 16)) * -2.5
    collapsed = _ensure_loss_vector(wide)
    assert collapsed.shape == (8,)
    assert collapsed[0].item() == pytest.approx(-2.5)


def test_trigger_parallel_invocation_isolated_state() -> None:
    provider_calls: Dict[str, list[int]] = {"requested": []}

    def provider(k: int, device: str, ctx) -> Tuple[torch.Tensor, torch.Tensor]:
        provider_calls["requested"].append(k)
        inputs = torch.arange(k, dtype=torch.float32).unsqueeze(1)
        targets = torch.ones(k, dtype=torch.float32)
        return inputs, targets

    triggers = [
        LossStdTrigger(
            provider=provider,
            cfg=LossStdConfig(std_threshold=10.0, inject_ratio=0.5, budget_frac=1.0),
        )
        for _ in range(4)
    ]

    def run_trigger(idx: int) -> int:
        trigger = triggers[idx]
        losses = torch.full((4,), 0.01 * (idx + 1))
        ctx = {"loss_vec": losses, "device": "cpu", "step": idx + 1}
        result = trigger(ctx)
        return 0 if result is None else result.weights.shape[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_trigger, range(4)))

    assert all(r in (0, 6) for r in results)
    # Provider should have been called at most once per trigger that injected.
    assert len(provider_calls["requested"]) <= 4
