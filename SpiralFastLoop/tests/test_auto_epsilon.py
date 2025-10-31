import random
import sys
from pathlib import Path

try:
    from spiralfastloop.auto_epsilon import AutoEpsilonOptimizer
except ModuleNotFoundError:  # pragma: no cover - local editable checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spiralfastloop.auto_epsilon import AutoEpsilonOptimizer


def _generate_residuals(seed: int = 7, total: int = 400) -> list[float]:
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(total):
        if rng.random() < 0.65:
            samples.append(rng.gauss(0.0, 0.015))
        else:
            samples.append(rng.gauss(0.0, 0.08))
    return samples


def test_auto_epsilon_reduces_unnecessary_zeroing():
    residuals = _generate_residuals()
    optimiser = AutoEpsilonOptimizer(
        initial_epsilon=0.08,
        bounds=(0.01, 0.12),
        optimisation_interval=25,
        optimisation_steps=3,
        min_history=60,
        smoothing=0.4,
        weight_zero=0.55,
        weight_error=0.45,
        random_state=123,
    )

    for value in residuals:
        optimiser.observe(value)

    report = optimiser.report()
    baseline = optimiser.evaluate(residuals=residuals, epsilon=0.08)

    assert report.total == baseline.total
    assert 0.0025 <= report.epsilon <= 0.12
    assert report.zero_ratio < baseline.zero_ratio
    assert report.avg_abs_error <= baseline.avg_abs_error + 0.015


def test_auto_epsilon_history_is_stable():
    residuals = _generate_residuals(seed=99)
    optimiser = AutoEpsilonOptimizer(
        initial_epsilon=0.05,
        bounds=(0.002, 0.1),
        optimisation_interval=20,
        optimisation_steps=4,
        min_history=50,
        smoothing=0.5,
        random_state=11,
    )

    for value in residuals:
        optimiser.observe(value)

    history_std = optimiser.report().epsilon_std
    assert history_std < 0.02
