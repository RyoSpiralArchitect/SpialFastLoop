from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_parallel_transactions import SyntheticTransactionDataset


def test_materialized_transaction_dataset_matches_shape_and_is_stable() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=True)

    features_a, target_a = dataset[2]
    features_b, target_b = dataset[2]

    assert features_a.shape == (4,)
    assert target_a.shape == ()
    assert torch.equal(features_a, features_b)
    assert torch.equal(target_a, target_b)
    assert dataset.materialized_bytes == (8 * 4 * 4) + (8 * 8)


def test_generated_transaction_dataset_is_index_deterministic() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=False)

    features_a, target_a = dataset[2]
    features_b, target_b = dataset[2]

    assert torch.equal(features_a, features_b)
    assert torch.equal(target_a, target_b)
    assert dataset.materialized_bytes == 0


def test_transaction_dataset_rejects_invalid_shapes() -> None:
    for kwargs in (
        {"size": -1, "features": 4, "classes": 3},
        {"size": 8, "features": 0, "classes": 3},
        {"size": 8, "features": 4, "classes": 0},
    ):
        try:
            SyntheticTransactionDataset(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_transaction_dataset_bounds_indices_consistently() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=True)

    last_features, last_target = dataset[-1]
    direct_features, direct_target = dataset[7]

    assert torch.equal(last_features, direct_features)
    assert torch.equal(last_target, direct_target)

    for index in (-9, 8):
        try:
            dataset[index]
        except IndexError:
            pass
        else:
            raise AssertionError(f"expected IndexError for {index}")
