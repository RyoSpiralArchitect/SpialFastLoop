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


def test_generated_transaction_dataset_is_index_deterministic() -> None:
    dataset = SyntheticTransactionDataset(8, 4, 3, seed=123, materialized=False)

    features_a, target_a = dataset[2]
    features_b, target_b = dataset[2]

    assert torch.equal(features_a, features_b)
    assert torch.equal(target_a, target_b)
