import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiralfastloop.auto_epsilon import AutoEpsilonOptimizer, simulate_replay


def _generate_batches(seed: int, n_batches: int = 6, batch_size: int = 256):
    rng = random.Random(seed)
    batches = []
    for _ in range(n_batches):
        dominant = rng.uniform(0.02, 0.08)
        noise = rng.uniform(0.001, 0.008)
        batch = []
        for _ in range(batch_size):
            if rng.random() < 0.35:
                batch.append(rng.gauss(0.0, noise))
            else:
                batch.append(rng.gauss(0.0, dominant))
        batches.append(batch)
    return batches


def test_optimizer_updates_towards_reasonable_threshold():
    batches = _generate_batches(42)

    optimizer = AutoEpsilonOptimizer(
        bounds=(1e-5, 0.2),
        initial_epsilon=0.05,
        normalization_weight=0.75,
        stability_weight=0.05,
        random_state=1,
    )

    learned_epsilons = [optimizer.update(batch)["epsilon"] for batch in batches]

    assert learned_epsilons[0] < 0.02
    assert learned_epsilons[-1] < 0.01
    assert learned_epsilons[-1] > 1e-4


def test_simulation_reports_superior_metrics_for_adaptive_epsilon():
    batches = _generate_batches(7, n_batches=5)

    optimizer = AutoEpsilonOptimizer(
        bounds=(1e-5, 0.15),
        initial_epsilon=0.04,
        normalization_weight=0.5,
        stability_weight=0.1,
        random_state=3,
    )

    results = simulate_replay(batches, optimizer=optimizer, baseline_epsilon=0.04)

    baseline = results["baseline"]
    learned = results["learned"]

    assert learned["unnecessary_normalisations"] <= baseline["unnecessary_normalisations"]
    assert learned["mean_error"] <= baseline["mean_error"] * 1.6
    assert learned["epsilon_mean"] < baseline["epsilon_mean"]

