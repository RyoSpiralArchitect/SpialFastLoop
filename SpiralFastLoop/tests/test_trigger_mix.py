import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop.engine import TriggerResult
from spiralfastloop.extras.trigger_mix import LossStdConfig, LossStdTrigger


def _make_provider(outputs=None):
    calls = {"requested": []}

    def provider(k, device, ctx):
        calls["requested"].append(k)
        batch_shape = (k, 2)
        inputs = torch.full(batch_shape, 1.0, device=device)
        targets = torch.zeros(k, device=device)
        if outputs is not None:
            return outputs
        return inputs, targets

    provider.calls = calls
    return provider


def test_trigger_skips_when_variability_high():
    cfg = LossStdConfig(std_threshold=0.1, inject_ratio=0.5, pulse_every=10, budget_frac=1.0)
    trigger = LossStdTrigger(provider=_make_provider(), cfg=cfg)
    ctx = {"loss_vec": torch.tensor([0.0, 2.0]), "device": "cpu", "step": 1}
    result = trigger(ctx)
    assert result is None


def test_trigger_requests_extra_samples_on_low_std():
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


def test_pulse_fires_even_when_variance_high():
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


def test_forced_pulse_only_attempts_once_when_budget_blocked():
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


def test_pulse_only_triggers_once_per_step():
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


def test_budget_fraction_limits_total_injections():
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

    assert trigger.spent == 1
    assert trigger.total == 15
    assert trigger.spent <= trigger.cfg.budget_frac * trigger.total + 1e-6
    assert provider.calls["requested"] == [1]


def test_budget_counters_reset_on_epoch_restart():
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


def test_budget_counters_ignore_repeated_steps():
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


def test_pulse_resets_after_step_decrease():
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
