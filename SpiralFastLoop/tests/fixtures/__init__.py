"""Pytest fixtures housing regression reproductions for SpiralFastLoop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RoundingRegressionCase:
    """Fixture describing historical rounding residue regressions."""

    value: float
    expected: float


def rounding_regression_cases() -> List[RoundingRegressionCase]:
    """Return the curated rounding regression scenarios."""

    # These values capture a bug where microscopic negative residues leaked back
    # into subsequent calls, yielding spurious budget credit.  Both polarities
    # and values just above/below the epsilon are covered so we can assert that
    # the guard rail stays in place.
    return [
        RoundingRegressionCase(value=-1e-16, expected=0.0),
        RoundingRegressionCase(value=1e-16, expected=0.0),
        RoundingRegressionCase(value=-9.999e-13, expected=0.0),
        RoundingRegressionCase(value=9.999e-13, expected=0.0),
        RoundingRegressionCase(value=-1.001e-12, expected=-1.001e-12),
        RoundingRegressionCase(value=1.001e-12, expected=1.001e-12),
    ]


__all__ = [
    "RoundingRegressionCase",
    "rounding_regression_cases",
]
