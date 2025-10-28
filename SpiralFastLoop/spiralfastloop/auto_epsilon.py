"""Adaptive epsilon tuning via lightweight Bayesian optimisation."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


ArrayLike = Iterable[float]


def _ensure_1d(residuals: ArrayLike) -> List[float]:
    if isinstance(residuals, list):
        return [float(value) for value in residuals]
    if isinstance(residuals, tuple):
        return [float(value) for value in residuals]
    if hasattr(residuals, "__iter__") and not isinstance(residuals, (str, bytes)):
        return [float(value) for value in residuals]
    return [float(residuals)]


def _rbf_kernel(x: Sequence[float], y: Sequence[float], length_scale: float, scale: float) -> List[List[float]]:
    matrix: List[List[float]] = []
    multiplier = scale * scale
    inv_length = 1.0 / (length_scale * length_scale)
    for xi in x:
        row: List[float] = []
        for yj in y:
            diff = xi - yj
            row.append(multiplier * math.exp(-0.5 * diff * diff * inv_length))
        matrix.append(row)
    return matrix


def _cholesky(matrix: List[List[float]]) -> List[List[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - total
                if value <= 0.0:
                    value = 1e-12
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def _forward_substitution(lower: List[List[float]], vector: Sequence[float]) -> List[float]:
    n = len(lower)
    result = [0.0] * n
    for i in range(n):
        total = sum(lower[i][j] * result[j] for j in range(i))
        result[i] = (vector[i] - total) / lower[i][i]
    return result


def _backward_substitution(lower: List[List[float]], vector: Sequence[float]) -> List[float]:
    n = len(lower)
    result = [0.0] * n
    for i in reversed(range(n)):
        total = sum(lower[j][i] * result[j] for j in range(i + 1, n))
        result[i] = (vector[i] - total) / lower[i][i]
    return result


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class EpsilonObservation:
    epsilon: float
    objective: float
    mean_error: float
    unnecessary_normalisations: float
    stability: float


class AutoEpsilonOptimizer:
    """Online optimiser that tunes epsilon with a Gaussian Process surrogate."""

    def __init__(
        self,
        *,
        bounds: tuple[float, float] = (1e-8, 1e-1),
        initial_epsilon: float = 1e-4,
        normalization_weight: float = 0.6,
        stability_weight: float = 0.2,
        noise_level: float = 1e-6,
        length_scale: float = 0.7,
        kernel_scale: float = 0.2,
        random_state: Optional[int] = None,
    ) -> None:
        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError("Bounds must be strictly positive.")
        if bounds[0] >= bounds[1]:
            raise ValueError("Lower bound must be smaller than upper bound.")
        self.bounds = bounds
        self.current_epsilon = float(initial_epsilon)
        self._last_epsilon: Optional[float] = None
        self.normalization_weight = float(normalization_weight)
        self.stability_weight = float(stability_weight)
        self.noise_level = float(noise_level)
        self.length_scale = float(length_scale)
        self.kernel_scale = float(kernel_scale)
        self._rng = random.Random(random_state)

        self._observations: List[EpsilonObservation] = []
        self._log_epsilons: List[float] = []
        self._objectives: List[float] = []
        self._epsilon_history: List[float] = [float(initial_epsilon)]

    def update(self, residuals: ArrayLike, *, n_candidates: int = 24, exploration_bias: float = 1e-3) -> Dict[str, float]:
        data = _ensure_1d(residuals)

        current_metrics = self._compute_metrics(data, self.current_epsilon, self._last_epsilon)
        self._record(current_metrics)

        candidate = self._suggest_candidate(max(4, int(n_candidates)), exploration_bias)
        metrics = self._compute_metrics(data, candidate, self.current_epsilon)
        self._record(metrics)

        best_index = min(range(len(self._objectives)), key=self._objectives.__getitem__)
        best = self._observations[best_index]

        self._last_epsilon = self.current_epsilon
        self.current_epsilon = best.epsilon
        self._epsilon_history.append(best.epsilon)

        return {
            "epsilon": best.epsilon,
            "mean_error": best.mean_error,
            "unnecessary_normalisations": best.unnecessary_normalisations,
            "stability": best.stability,
        }

    def metrics_for(
        self,
        residuals: ArrayLike,
        *,
        epsilon: Optional[float] = None,
        last_epsilon: Optional[float] = None,
    ) -> Dict[str, float]:
        data = _ensure_1d(residuals)
        value = self.current_epsilon if epsilon is None else float(epsilon)
        metrics = self._compute_metrics(data, value, last_epsilon)
        return {
            "epsilon": metrics.epsilon,
            "mean_error": metrics.mean_error,
            "unnecessary_normalisations": metrics.unnecessary_normalisations,
            "stability": metrics.stability,
        }

    @property
    def epsilon_history(self) -> Sequence[float]:
        return tuple(self._epsilon_history)

    def _suggest_candidate(self, n_candidates: int, exploration_bias: float) -> float:
        lower, upper = (math.log(self.bounds[0]), math.log(self.bounds[1]))
        samples = [lower + (upper - lower) * self._rng.random() for _ in range(n_candidates)]
        epsilons = [math.exp(sample) for sample in samples]

        if len(self._log_epsilons) < 2:
            return sorted(epsilons)[n_candidates // 2]

        mu, sigma = self._posterior(samples, self._log_epsilons, self._objectives)
        best = min(self._objectives)
        expected_improvements: List[float] = []
        for mean, deviation in zip(mu, sigma):
            if deviation <= 0:
                expected_improvements.append(0.0)
                continue
            improvement = best - mean - exploration_bias
            z = improvement / deviation
            expected_improvement = improvement * _normal_cdf(z) + deviation * _normal_pdf(z)
            expected_improvements.append(expected_improvement)

        best_index = max(range(len(expected_improvements)), key=expected_improvements.__getitem__)
        return epsilons[best_index]

    def _posterior(
        self,
        x_star: Sequence[float],
        x_train: Sequence[float],
        y_train: Sequence[float],
    ) -> tuple[List[float], List[float]]:
        n = len(x_train)
        K = _rbf_kernel(x_train, x_train, self.length_scale, self.kernel_scale)
        for i in range(n):
            K[i][i] += self.noise_level * self.noise_level + 1e-10

        L = _cholesky(K)
        y_vector = list(y_train)
        tmp = _forward_substitution(L, y_vector)
        alpha = _backward_substitution(L, tmp)

        K_s = _rbf_kernel(x_train, x_star, self.length_scale, self.kernel_scale)

        mu: List[float] = []
        for column in zip(*K_s):
            mu.append(sum(value * weight for value, weight in zip(column, alpha)))

        sigma: List[float] = []
        kernel_diag = self.kernel_scale * self.kernel_scale
        for column in zip(*K_s):
            sol = _forward_substitution(L, list(column))
            variance = max(1e-12, kernel_diag - sum(value * value for value in sol))
            sigma.append(math.sqrt(variance))

        return mu, sigma

    def _record(self, metrics: EpsilonObservation) -> None:
        self._observations.append(metrics)
        self._log_epsilons.append(math.log(metrics.epsilon))
        self._objectives.append(metrics.objective)

    def _compute_metrics(
        self,
        residuals: Sequence[float],
        epsilon: float,
        last_epsilon: Optional[float],
    ) -> EpsilonObservation:
        clipped = max(self.bounds[0], min(self.bounds[1], float(epsilon)))
        if last_epsilon is None:
            stability = 0.0
        else:
            prev = max(self.bounds[0], float(last_epsilon))
            stability = abs(math.log(clipped) - math.log(prev))

        thresholded = [value if abs(value) >= clipped else 0.0 for value in residuals]
        mean_error = statistics.fmean(abs(value) for value in thresholded) if thresholded else 0.0
        unnecessary = statistics.fmean(1.0 if abs(value) < clipped else 0.0 for value in residuals) if residuals else 0.0
        penalty = unnecessary * unnecessary
        objective = mean_error + self.normalization_weight * penalty + self.stability_weight * stability

        return EpsilonObservation(
            epsilon=clipped,
            objective=objective,
            mean_error=mean_error,
            unnecessary_normalisations=unnecessary,
            stability=stability,
        )


def _aggregate_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(metrics)
    if not items:
        raise ValueError("No metrics supplied for aggregation.")

    mean_error = statistics.fmean(item["mean_error"] for item in items)
    unnecessary = statistics.fmean(item["unnecessary_normalisations"] for item in items)
    stability = statistics.fmean(item["stability"] for item in items)
    epsilons = [item["epsilon"] for item in items]

    return {
        "mean_error": mean_error,
        "unnecessary_normalisations": unnecessary,
        "stability": stability,
        "epsilon_mean": statistics.fmean(epsilons),
        "epsilon_std": statistics.pstdev(epsilons) if len(epsilons) > 1 else 0.0,
    }


def simulate_replay(
    residual_batches: Iterable[ArrayLike],
    *,
    optimizer: AutoEpsilonOptimizer,
    baseline_epsilon: float,
) -> Dict[str, Dict[str, float]]:
    learned_metrics: List[Dict[str, float]] = []
    baseline_metrics: List[Dict[str, float]] = []

    last_baseline = float(baseline_epsilon)
    for batch in residual_batches:
        baseline_stats = optimizer.metrics_for(batch, epsilon=baseline_epsilon, last_epsilon=last_baseline)
        baseline_metrics.append(baseline_stats)
        last_baseline = baseline_stats["epsilon"]

        learned_stats = optimizer.update(batch)
        learned_metrics.append(learned_stats)

    return {
        "baseline": _aggregate_metrics(baseline_metrics),
        "learned": _aggregate_metrics(learned_metrics),
    }


__all__ = ["AutoEpsilonOptimizer", "simulate_replay"]

