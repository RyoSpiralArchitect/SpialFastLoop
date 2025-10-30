from __future__ import annotations

from decimal import Decimal

import pytest
import torch
from hypothesis import given, strategies as st

from spiralfastloop.extras.trigger_mix import (
    FRACTION_NORMALIZATION_EPS,
    LossStdConfig,
    LossStdTrigger,
)


def _provider(k: int, device: str, ctx):
    shape = (k, 1)
    return torch.zeros(shape, device=device), torch.zeros(k, device=device)


@given(
    st.floats(
        min_value=-FRACTION_NORMALIZATION_EPS * 0.9,
        max_value=FRACTION_NORMALIZATION_EPS * 0.9,
        allow_infinity=False,
        allow_nan=False,
    )
)
def test_drop_rounding_noise_annihilates_sub_epsilon(value: float) -> None:
    trigger = LossStdTrigger(provider=_provider)
    assert trigger._drop_rounding_noise(value) == Decimal(0)


@given(
    st.floats(
        min_value=FRACTION_NORMALIZATION_EPS * 2,
        max_value=1.0,
        allow_infinity=False,
        allow_nan=False,
    )
)
def test_drop_rounding_noise_preserves_super_epsilon(value: float) -> None:
    trigger = LossStdTrigger(provider=_provider)
    result = trigger._drop_rounding_noise(value)
    assert float(result) == pytest.approx(value)


@given(st.integers(min_value=2, max_value=32))
def test_budget_buffer_never_accumulates_full_sample(batch_size: int) -> None:
    cfg = LossStdConfig(
        std_threshold=1.0, inject_ratio=0.5, budget_frac=1.0, pulse_every=0
    )
    trigger = LossStdTrigger(provider=_provider, cfg=cfg)
    ctx = {"loss_vec": torch.ones(batch_size), "device": "cpu", "step": 1}
    trigger(ctx)
    assert 0.0 <= float(trigger._budget_buffer) < 1.0
