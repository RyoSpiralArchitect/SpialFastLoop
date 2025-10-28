import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
    LossStdConfig,
    LossStdTrigger,
)
from spiralfastloop.metrics import NormalizationMetricsCollector

_FIXTURES_DIR = Path(__file__).resolve().parent
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.append(str(_FIXTURES_DIR))

from fixtures import RoundingRegressionCase


def _make_provider(outputs: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    tiny = torch.tensor([1e-10, -1e-10], dtype=torch.float64)
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
