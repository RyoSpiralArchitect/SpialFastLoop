import random
import sys
from pathlib import Path

import pytest

try:
    from spiralfastloop.auto_epsilon import AutoEpsilonOptimizer
except ModuleNotFoundError:  # pragma: no cover - local editable checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spiralfastloop.auto_epsilon import AutoEpsilonOptimizer


class _FailingIterable:
    def __iter__(self):
        yield 0.1
        raise RuntimeError("iteration failed")


class _BrokenIterable:
    def __iter__(self):
        raise RuntimeError("iterator creation failed")


def _generate_residuals(seed: int = 7, total: int = 400) -> list[float]:
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(total):
        if rng.random() < 0.65:
            samples.append(rng.gauss(0.0, 0.015))
        else:
            samples.append(rng.gauss(0.0, 0.08))
    return samples


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"initial_epsilon": -0.1}, "initial_epsilon"),
        ({"initial_epsilon": float("nan")}, "initial_epsilon"),
        ({"initial_epsilon": True}, "initial_epsilon"),
        ({"initial_epsilon": "0.1"}, "initial_epsilon"),
        ({"initial_epsilon": b"0.1"}, "initial_epsilon"),
        ({"bounds": object()}, "bounds"),
        ({"bounds": "ab"}, "bounds"),
        ({"bounds": b"ab"}, "bounds"),
        ({"bounds": {1e-6: 0.1, 0.2: 0.3}}, "bounds"),
        ({"bounds": (0.1,)}, "bounds"),
        ({"bounds": _FailingIterable()}, "bounds"),
        ({"bounds": _BrokenIterable()}, "bounds"),
        ({"bounds": (True, 0.2)}, "bounds"),
        ({"bounds": ("0.1", 0.2)}, "bounds"),
        ({"bounds": (float("nan"), 0.2)}, "bounds"),
        ({"bounds": (0.2, 0.1)}, "bounds"),
        ({"history_size": 0}, "history_size"),
        ({"history_size": 1.5}, "history_size"),
        ({"history_size": "2"}, "history_size"),
        ({"history_size": True}, "history_size"),
        ({"epsilon_history": 0}, "epsilon_history"),
        ({"optimisation_interval": 0}, "optimisation_interval"),
        ({"optimisation_steps": 0}, "optimisation_steps"),
        ({"min_history": 0}, "min_history"),
        ({"candidate_points": 2}, "candidate_points"),
        ({"candidate_points": 3.5}, "candidate_points"),
        ({"candidate_points": True}, "candidate_points"),
        ({"weight_zero": -0.1}, "weight_zero"),
        ({"weight_zero": "0.1"}, "weight_zero"),
        ({"weight_error": float("inf")}, "weight_error"),
        ({"weight_error": b"0.1"}, "weight_error"),
        ({"length_scale": 0.0}, "length_scale"),
        ({"length_scale": "0.2"}, "length_scale"),
        ({"variance": -0.1}, "variance"),
        ({"variance": "0.1"}, "variance"),
        ({"noise": -0.1}, "noise"),
        ({"noise": "0.1"}, "noise"),
        ({"exploration": float("nan")}, "exploration"),
        ({"exploration": "0.1"}, "exploration"),
        ({"smoothing": -0.1}, "smoothing"),
        ({"smoothing": 1.1}, "smoothing"),
        ({"smoothing": True}, "smoothing"),
        ({"smoothing": "0.5"}, "smoothing"),
        ({"random_state": 1.5}, "random_state"),
        ({"random_state": "2"}, "random_state"),
        ({"random_state": True}, "random_state"),
    ],
)
def test_auto_epsilon_rejects_invalid_settings(kwargs: dict[str, object], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        AutoEpsilonOptimizer(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "residual",
    [float("nan"), float("inf"), True, "1.0", b"1.0", object()],
)
def test_auto_epsilon_rejects_invalid_observed_residuals(residual: object) -> None:
    optimiser = AutoEpsilonOptimizer()

    with pytest.raises(ValueError, match="residual"):
        optimiser.observe(residual)  # type: ignore[arg-type]


def test_auto_epsilon_rejects_invalid_evaluation_inputs() -> None:
    optimiser = AutoEpsilonOptimizer()

    with pytest.raises(ValueError, match="epsilon"):
        optimiser.evaluate(epsilon=float("nan"))
    with pytest.raises(ValueError, match="epsilon"):
        optimiser.evaluate(epsilon="0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residual"):
        optimiser.evaluate(residuals=[0.1, float("inf")])
    with pytest.raises(ValueError, match="residuals"):
        optimiser.evaluate(residuals=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residuals"):
        optimiser.evaluate(residuals="0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residuals"):
        optimiser.evaluate(residuals={0.1: "ignored"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residuals"):
        optimiser.evaluate(residuals=_FailingIterable())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residuals"):
        optimiser.evaluate(residuals=_BrokenIterable())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="objective"):
        optimiser.evaluate(residuals=[sys.float_info.max, sys.float_info.max])


def test_auto_epsilon_observe_rolls_back_when_optimisation_fails() -> None:
    optimiser = AutoEpsilonOptimizer(
        min_history=2,
        optimisation_interval=1,
        optimisation_steps=1,
        weight_zero=0.0,
        weight_error=1.0,
    )
    first = optimiser.observe(sys.float_info.max)
    before_report = optimiser.report()
    before_evaluations = list(optimiser._evaluations)
    before_pending_steps = optimiser._pending_steps

    with pytest.raises(ValueError, match="objective"):
        optimiser.observe(sys.float_info.max)

    after_report = optimiser.report()
    assert optimiser.epsilon == first
    assert after_report.total == before_report.total == 1
    assert after_report.epsilon_mean == pytest.approx(before_report.epsilon_mean)
    assert after_report.epsilon_std == pytest.approx(before_report.epsilon_std)
    assert optimiser._evaluations == before_evaluations
    assert optimiser._pending_steps == before_pending_steps


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
