import math
import random

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spiralfastloop.numerics import HybridCompensatedAccumulator


def test_accumulator_tracks_total_with_compensation() -> None:
    acc = HybridCompensatedAccumulator(unit=0.01, rng=random.Random(123))
    # Sequence designed to trigger cancellation when accumulated naïvely.
    values = [1e6, -1e6, 1e-3, 2e-3, -1e-3, 3.5e-3]
    acc.extend(values)
    assert math.isclose(acc.total, sum(values), rel_tol=0.0, abs_tol=1e-15)
    assert abs(acc.compensation) < 1e-12


def test_drain_returns_quantised_amount_and_residual() -> None:
    rng = random.Random(2025)
    acc = HybridCompensatedAccumulator(unit=0.1, rng=rng)
    acc.extend([0.03, 0.04, 0.09])  # total = 0.16
    payout = acc.drain()
    assert payout in {0.1, 0.2}
    assert abs(acc.residual) < acc.unit
    assert math.isclose(payout + acc.residual, 0.16, rel_tol=0.0, abs_tol=1e-12)


def test_reset_clears_state() -> None:
    acc = HybridCompensatedAccumulator(unit=0.01, rng=random.Random(7))
    acc.add(0.25)
    acc.drain()
    acc.reset()
    assert acc.total == 0.0
    assert acc.compensation == 0.0


def test_snapshot_and_restore_round_trip() -> None:
    rng = random.Random(99)
    acc = HybridCompensatedAccumulator(unit=0.1, rng=rng)
    acc.extend([0.05, 0.07, -0.02])
    snapshot = acc.snapshot()

    restored = HybridCompensatedAccumulator(
        unit=0.1, rng=random.Random(99), initial_total=snapshot[0], initial_compensation=snapshot[1]
    )
    assert math.isclose(restored.total, acc.total, rel_tol=0.0, abs_tol=1e-15)

    restored.add(0.04)
    acc.add(0.04)
    assert math.isclose(restored.total, acc.total, rel_tol=0.0, abs_tol=1e-15)


def test_stochastic_rounding_is_unbiased_over_many_trials() -> None:
    trials = 10_000
    target = 1.237  # requires fractional rounding
    payouts = []
    for seed in range(trials):
        rng = random.Random(seed)
        acc = HybridCompensatedAccumulator(unit=0.01, rng=rng)
        acc.add(target)
        payouts.append(acc.drain())
    mean_payout = sum(payouts) / trials
    assert pytest.approx(target, abs=5e-3) == mean_payout


def test_negative_totals_round_correctly() -> None:
    rng = random.Random(314)
    acc = HybridCompensatedAccumulator(unit=0.05, rng=rng)
    acc.extend([-0.12, -0.08])  # total = -0.20
    rounded = acc.drain()
    assert rounded in {-0.2, -0.15}
    assert math.isclose(rounded + acc.residual, -0.20, rel_tol=0.0, abs_tol=1e-12)


def test_restore_rejects_non_finite() -> None:
    acc = HybridCompensatedAccumulator()
    with pytest.raises(ValueError):
        acc.restore(float("nan"), 0.0)


def test_initial_state_validation() -> None:
    with pytest.raises(ValueError):
        HybridCompensatedAccumulator(initial_total=float("inf"))
    with pytest.raises(ValueError):
        HybridCompensatedAccumulator(initial_compensation=float("nan"))
