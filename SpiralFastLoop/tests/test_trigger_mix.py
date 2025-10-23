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


def test_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        LossStdConfig(inject_ratio=-0.1)
    with pytest.raises(ValueError):
        LossStdConfig(weight_alpha=0.0)
    with pytest.raises(ValueError):
        LossStdConfig(budget_frac=-0.2)
    with pytest.raises(ValueError):
        LossStdConfig(pulse_every=-5)
    with pytest.raises(ValueError):
        LossStdConfig(max_injected_per_step=-1)


def test_trigger_state_roundtrip_and_reset():
    cfg = LossStdConfig(
        std_threshold=10.0,
        inject_ratio=0.5,
        pulse_every=1000,
        budget_frac=1.0,
        max_injected_per_step=8,
    )
    provider = _make_provider()
    trigger = LossStdTrigger(provider=provider, cfg=cfg)

    ctx = {"loss_vec": torch.ones(4), "device": "cpu", "step": 1}
    trigger(ctx)

    assert trigger.spent == 2
    assert trigger.total == 4

    state = trigger.state_dict()
    assert state == {"spent": 2, "total": 4}

    resumed_provider = _make_provider()
    resumed = LossStdTrigger(provider=resumed_provider, cfg=cfg)
    resumed.load_state_dict(state)

    ctx["step"] = 2
    resumed(ctx)

    assert resumed.spent == 4
    assert resumed.total == 8
    assert resumed_provider.calls["requested"] == [2]

    trigger.reset()
    assert trigger.spent == 0
    assert trigger.total == 0
