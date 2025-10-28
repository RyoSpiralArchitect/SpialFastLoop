# SPDX-License-Identifier: MIT
"""Numerical helpers for bias-resistant floating-point accumulation.

This module exposes a small accumulator that fuses two stability techniques
commonly used in numerical finance workflows:

* **Compensated summation (Neumaier / Kahan hybrid)** keeps track of the
  rounding error that would normally be lost when repeatedly adding small
  values into a large running total.  The hybrid variant switches between the
  original Kahan update and Neumaier’s improvement depending on the magnitude
  of the incoming value so that catastrophic cancellation is minimised.
* **Stochastic rounding** converts the high-precision total into a discrete
  bucket (for example cents) without introducing systematic bias.  Instead of
  always rounding towards zero, the algorithm rounds up with probability equal
  to the fractional part of the scaled value which ensures the expected value
  matches the high-precision total.

The combination is particularly useful for aggregating monetary amounts that
arrive in arbitrary order while still emitting ledger-friendly whole units
for payouts or reporting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = [
    "HybridCompensatedAccumulator",
]


def _neumaier_step(total: float, compensation: float, value: float) -> tuple[float, float]:
    """Return the updated total and compensation using Neumaier summation."""

    t = total + value
    if abs(total) >= abs(value):
        compensation += (total - t) + value
    else:
        compensation += (value - t) + total
    return t, compensation


def _kahan_step(total: float, compensation: float, value: float) -> tuple[float, float]:
    """Return the updated total and compensation using Kahan summation."""

    y = value - compensation
    t = total + y
    compensation = (t - total) - y
    return t, compensation


def _stochastic_round(value: float, unit: float, rng: random.Random) -> float:
    """Round ``value`` to the nearest ``unit`` using stochastic rounding.

    The output is always an integer multiple of ``unit``.  If the scaled value
    already lands exactly on a discrete bucket the function is deterministic.
    Otherwise it rounds up with probability equal to the fractional part of the
    scaled value.
    """

    if not math.isfinite(value):
        raise ValueError("Value to round must be finite.")

    scaled = value / unit
    lower = math.floor(scaled)
    fraction = scaled - lower

    # Guard against representation noise (e.g., 0.9999999998) so that we
    # maintain invariants such as returning exact multiples of ``unit``.
    if fraction <= 0.0 or fraction < 1e-15:
        return lower * unit
    if fraction >= 1.0 - 1e-12:
        return (lower + 1) * unit

    if rng.random() < fraction:
        return (lower + 1) * unit
    return lower * unit


@dataclass(slots=True)
class HybridCompensatedAccumulator:
    """Running sum with compensation and stochastic rounding.

    Parameters
    ----------
    unit:
        Smallest discrete quantum that the accumulator should emit, for
        example ``0.01`` for cents.  Must be positive and finite.
    rng:
        Optional ``random.Random`` instance.  Provide one to make stochastic
        rounding deterministic under tests or distributed deployments.

    Notes
    -----
    The accumulator stores two floating-point values: a running total and the
    compensation term that captures rounding residue.  When the magnitude of
    the incoming value is large compared with the total we favour Neumaier’s
    formulation because it is more robust to cancellation.  Otherwise the
    classic Kahan step is used which carries slightly less overhead when the
    numbers are of similar scale.  Instances can be snapshotted and restored
    to support long-lived processes or distributed reducers.
    """

    unit: float = 0.01
    rng: Optional[random.Random] = None
    initial_total: float = 0.0
    initial_compensation: float = 0.0
    _rng: random.Random = field(init=False, repr=False)
    _total: float = field(init=False, repr=False)
    _compensation: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.unit) or self.unit <= 0.0:
            raise ValueError("unit must be a positive, finite float")
        if not math.isfinite(self.initial_total):
            raise ValueError("initial_total must be finite")
        if not math.isfinite(self.initial_compensation):
            raise ValueError("initial_compensation must be finite")
        self._rng = self.rng or random.Random()
        self._total = float(self.initial_total)
        self._compensation = float(self.initial_compensation)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add(self, value: float) -> None:
        """Add a single value into the accumulator."""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError("value must be finite")

        if abs(self._total) > abs(value):
            self._total, self._compensation = _kahan_step(
                self._total, self._compensation, value
            )
        else:
            self._total, self._compensation = _neumaier_step(
                self._total, self._compensation, value
            )

    def extend(self, values: Iterable[float]) -> None:
        """Add multiple values into the accumulator."""

        for value in values:
            self.add(value)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def total(self) -> float:
        """Exact running total including compensation."""

        return self._total + self._compensation

    @property
    def compensation(self) -> float:
        """Expose the current compensation term (mostly for diagnostics)."""

        return self._compensation

    @property
    def residual(self) -> float:
        """Return the sub-unit residual that remains after a rounding pass."""

        return self.total

    def drain(self) -> float:
        """Emit a stochastically-rounded amount and retain the residual.

        The accumulator keeps the leftover fractional part (always strictly
        smaller than ``unit``) so that subsequent additions benefit from the
        preserved precision.
        """

        total = self.total
        rounded = _stochastic_round(total, self.unit, self._rng)
        residual = total - rounded
        self._total = residual
        self._compensation = 0.0
        return rounded

    def reset(self) -> None:
        """Reset the accumulator to zero."""

        self._total = 0.0
        self._compensation = 0.0

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> tuple[float, float]:
        """Return the raw total/compensation pair for persistence."""

        return self._total, self._compensation

    def restore(self, total: float, compensation: float) -> None:
        """Restore a previously captured state."""

        if not math.isfinite(total) or not math.isfinite(compensation):
            raise ValueError("snapshot values must be finite")
        self._total = float(total)
        self._compensation = float(compensation)
