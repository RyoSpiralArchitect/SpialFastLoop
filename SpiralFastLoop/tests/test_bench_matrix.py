from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_matrix import _compile_requested, _parse_csv_choices, _parse_worker_counts


def test_parse_csv_choices_trims_and_validates_values() -> None:
    assert _parse_csv_choices(" generated,materialized ", {"generated", "materialized"}, name="modes") == [
        "generated",
        "materialized",
    ]

    with pytest.raises(ValueError):
        _parse_csv_choices("", {"generated"}, name="modes")
    with pytest.raises(ValueError):
        _parse_csv_choices("generated,other", {"generated"}, name="modes")


def test_parse_worker_counts_rejects_empty_or_negative_values() -> None:
    assert _parse_worker_counts("0, 2") == [0, 2]

    with pytest.raises(ValueError):
        _parse_worker_counts("")
    with pytest.raises(ValueError):
        _parse_worker_counts("-1")


def test_compile_requested_maps_modes() -> None:
    assert _compile_requested("compile") is True
    assert _compile_requested("no-compile") is False

    with pytest.raises(ValueError):
        _compile_requested("sometimes")
