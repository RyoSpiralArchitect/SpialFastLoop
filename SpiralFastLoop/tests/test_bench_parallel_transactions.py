from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_parallel_transactions import SyntheticTransactionDataset, run_once


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


def test_transaction_benchmark_records_run_seed() -> None:
    class Args:
        transactions = 64
        feature_dim = 8
        num_classes = 3
        seed = 100
        dataset_mode = "materialized"
        batch_size = 16
        device = "cpu"
        workers = 0
        prefetch_factor = 2
        learning_rate = 3e-4
        compile = False
        grad_accum = 2
        log_interval = 0
        steps = 2
        collect_profile = False
        profile_sync = False
        profile_distribution = True
        profile_window = 16
        profile_model = False
        profile_model_depth = 1
        profile_model_max_modules = 8
        profile_model_include = None
        warmup_steps = 0

    result = run_once(Args, 3).as_dict()

    assert result["seed"] == 103
    assert result["dataset_mode"] == "materialized"
    assert result["dataset_materialized_bytes"] == (64 * 8 * 4) + (64 * 8)
