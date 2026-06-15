from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pytest
import torch

_FIXTURES_DIR = Path(__file__).resolve().parent
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.append(str(_FIXTURES_DIR))

from fixtures import RoundingRegressionCase, rounding_regression_cases


def pytest_configure(config: pytest.Config) -> None:
    # Keep tests deterministic if a local sitecustomize changes PyTorch's default device.
    set_default_device = getattr(torch, "set_default_device", None)
    if callable(set_default_device):
        set_default_device("cpu")


@pytest.fixture(scope="session")
def rounding_cases() -> Iterable[RoundingRegressionCase]:
    """Regression scenarios replicating historical rounding bugs."""

    return tuple(rounding_regression_cases())
