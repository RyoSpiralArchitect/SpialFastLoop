# SPDX-License-Identifier: Apache-2.0

"""Adaptive epsilon scheduling based on online residual statistics."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of applying a specific epsilon to a residual sequence."""

    epsilon: float
    zero_count: int
    total: int
    zero_ratio: float
    avg_abs_error: float
    objective: float


@dataclass(frozen=True)
class AutoEpsilonReport(SimulationResult):
    """Extended statistics reported by :class:`AutoEpsilonOptimizer`."""

    epsilon_mean: float
    epsilon_std: float


def _rbf_value(x: float, y: float, length_scale: float, variance: float) -> float:
    scale = max(length_scale ** 2, 1e-12)
    sqdist = (x - y) ** 2 / scale
    return variance * math.exp(-0.5 * sqdist)


def _rbf_kernel_matrix(
    x_values: Sequence[float],
    y_values: Sequence[float],
    length_scale: float,
    variance: float,
) -> List[List[float]]:
    return [
        [_rbf_value(x, y, length_scale, variance) for y in y_values]
        for x in x_values
    ]


def _cholesky_decomposition(matrix: List[List[float]], jitter: float) -> List[List[float]]:
    size = len(matrix)
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = matrix[i][j] - math.fsum(
                lower[i][k] * lower[j][k] for k in range(j)
            )
            if i == j:
                diag = value if value > jitter else jitter
                lower[i][j] = math.sqrt(diag)
            else:
                denom = lower[j][j]
                if abs(denom) <= 1e-12:
                    raise ValueError("matrix is not positive definite")
                lower[i][j] = value / denom
    return lower


def _forward_substitution(lower: List[List[float]], vector: Sequence[float]) -> List[float]:
    result = [0.0 for _ in range(len(vector))]
    for i in range(len(vector)):
        acc = math.fsum(lower[i][k] * result[k] for k in range(i))
        denom = lower[i][i]
        if abs(denom) <= 1e-12:
            raise ValueError("matrix is singular")
        result[i] = (vector[i] - acc) / denom
    return result


def _backward_substitution(lower: List[List[float]], vector: Sequence[float]) -> List[float]:
    size = len(vector)
    result = [0.0 for _ in range(size)]
    for idx in range(size - 1, -1, -1):
        acc = math.fsum(lower[k][idx] * result[k] for k in range(idx + 1, size))
        denom = lower[idx][idx]
        if abs(denom) <= 1e-12:
            raise ValueError("matrix is singular")
        result[idx] = (vector[idx] - acc) / denom
    return result


def _solve_cholesky(lower: List[List[float]], vector: Sequence[float]) -> List[float]:
    return _backward_substitution(lower, _forward_substitution(lower, vector))


def _matrix_vector_column(matrix: Sequence[Sequence[float]], column: int) -> List[float]:
    return [row[column] for row in matrix]


def _gp_posterior(
    x_train: Sequence[float],
    y_train: Sequence[float],
    x_test: Sequence[float],
    *,
    length_scale: float,
    variance: float,
    noise: float,
) -> Tuple[List[float], List[float]]:
    if not x_train:
        prior_mean = [0.0 for _ in x_test]
        prior_var = [variance for _ in x_test]
        return prior_mean, prior_var

    jitter = 1e-9
    k_xx = _rbf_kernel_matrix(x_train, x_train, length_scale, variance)
    for idx in range(len(k_xx)):
        k_xx[idx][idx] += noise + jitter

    try:
        chol = _cholesky_decomposition(k_xx, jitter)
        alpha = _solve_cholesky(chol, list(y_train))
    except ValueError:
        return [0.0 for _ in x_test], [variance for _ in x_test]

    k_xs = _rbf_kernel_matrix(x_train, x_test, length_scale, variance)

    means: List[float] = []
    variances: List[float] = []
    for column in range(len(x_test)):
        cross = _matrix_vector_column(k_xs, column)
        mean_val = math.fsum(c * a for c, a in zip(cross, alpha))
        try:
            v = _solve_cholesky(chol, cross)
            var_val = variance - math.fsum(c * vv for c, vv in zip(cross, v))
        except ValueError:
            var_val = variance
        variances.append(max(var_val, 1e-12))
        means.append(mean_val)

    return means, variances


def _expected_improvement(
    mean: Sequence[float],
    var: Sequence[float],
    best: float,
    exploration: float,
) -> List[float]:
    results: List[float] = []
    sqrt_two = math.sqrt(2.0)
    normaliser = math.sqrt(2.0 * math.pi)
    for mu, variance in zip(mean, var):
        sigma = math.sqrt(max(variance, 1e-12))
        improvement = best - mu - exploration
        if sigma <= 1e-12:
            results.append(max(improvement, 0.0))
            continue
        z = improvement / sigma
        pdf = math.exp(-0.5 * z * z) / normaliser
        cdf = 0.5 * (1.0 + math.erf(z / sqrt_two))
        results.append(improvement * cdf + sigma * pdf)
    return results


def _simulate_objective(
    residuals: Sequence[float],
    epsilon: float,
    weight_zero: float,
    weight_error: float,
) -> SimulationResult:
    values = [abs(float(r)) for r in residuals]
    total = len(values)
    if total == 0:
        return SimulationResult(epsilon, 0, 0, 0.0, 0.0, 0.0)

    zero_mask = [value <= epsilon for value in values]
    zero_count = sum(1 for flag in zero_mask if flag)
    zero_ratio = zero_count / total

    adjusted_sum = math.fsum(
        0.0 if mask else value for mask, value in zip(zero_mask, values)
    )
    avg_abs_error = adjusted_sum / total

    objective = weight_zero * zero_ratio + weight_error * avg_abs_error
    return SimulationResult(
        epsilon=epsilon,
        zero_count=zero_count,
        total=total,
        zero_ratio=zero_ratio,
        avg_abs_error=avg_abs_error,
        objective=objective,
    )


class AutoEpsilonOptimizer:
    """Online optimiser that adapts epsilon via Bayesian optimisation."""

    def __init__(
        self,
        *,
        initial_epsilon: float = 1e-3,
        bounds: Tuple[float, float] = (1e-6, 1e-1),
        history_size: int = 512,
        epsilon_history: int = 128,
        optimisation_interval: int = 32,
        optimisation_steps: int = 4,
        min_history: int = 64,
        weight_zero: float = 0.6,
        weight_error: float = 0.4,
        length_scale: float = 0.02,
        variance: float = 0.05,
        noise: float = 1e-4,
        exploration: float = 1e-3,
        smoothing: float = 0.4,
        candidate_points: int = 64,
        random_state: Optional[int] = None,
    ) -> None:
        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError("epsilon bounds must be positive")
        if bounds[0] >= bounds[1]:
            raise ValueError("invalid epsilon bounds")
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("smoothing must lie in [0, 1]")

        self.bounds = bounds
        self.weight_zero = weight_zero
        self.weight_error = weight_error
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.exploration = exploration
        self.optimisation_interval = optimisation_interval
        self.optimisation_steps = optimisation_steps
        self.min_history = min_history
        self.smoothing = smoothing
        self.candidate_points = max(3, candidate_points)

        self._epsilon = self._clip(initial_epsilon)
        self._residuals: Deque[float] = deque(maxlen=history_size)
        self._epsilon_history: Deque[float] = deque(maxlen=epsilon_history)
        self._epsilon_history.append(self._epsilon)

        self._evaluations: List[SimulationResult] = []
        self._pending_steps = 0
        self._rng = random.Random(random_state)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def observe(self, residual: float) -> float:
        self._residuals.append(float(residual))
        if len(self._residuals) >= self.min_history:
            self._pending_steps += 1
            if self._pending_steps >= self.optimisation_interval:
                self._pending_steps = 0
                self._run_optimisation()

        self._epsilon_history.append(self._epsilon)
        return self._epsilon

    def evaluate(
        self,
        residuals: Optional[Sequence[float]] = None,
        epsilon: Optional[float] = None,
    ) -> SimulationResult:
        seq = residuals if residuals is not None else list(self._residuals)
        value = self._clip(self._epsilon if epsilon is None else epsilon)
        return _simulate_objective(seq, value, self.weight_zero, self.weight_error)

    def report(self) -> AutoEpsilonReport:
        base = self.evaluate()
        epsilon_stats = list(self._epsilon_history)
        if epsilon_stats:
            eps_mean = mean(epsilon_stats)
            eps_std = pstdev(epsilon_stats) if len(epsilon_stats) > 1 else 0.0
        else:
            eps_mean = base.epsilon
            eps_std = 0.0
        return AutoEpsilonReport(
            epsilon=base.epsilon,
            zero_count=base.zero_count,
            total=base.total,
            zero_ratio=base.zero_ratio,
            avg_abs_error=base.avg_abs_error,
            objective=base.objective,
            epsilon_mean=eps_mean,
            epsilon_std=eps_std,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clip(self, value: float) -> float:
        return float(min(max(value, self.bounds[0]), self.bounds[1]))

    def _register(self, epsilon: float) -> None:
        candidate = self._clip(epsilon)
        if any(abs(candidate - ev.epsilon) <= 1e-9 for ev in self._evaluations):
            return
        result = self.evaluate(epsilon=candidate)
        self._evaluations.append(result)

    def _suggest_candidate(self) -> float:
        if len(self._evaluations) < 2:
            return self._rng.uniform(self.bounds[0], self.bounds[1])

        x_train = [ev.epsilon for ev in self._evaluations]
        y_train = [ev.objective for ev in self._evaluations]

        step = (self.bounds[1] - self.bounds[0]) / (self.candidate_points - 1)
        x_test = [self.bounds[0] + i * step for i in range(self.candidate_points)]

        mean, var = _gp_posterior(
            x_train,
            y_train,
            x_test,
            length_scale=self.length_scale,
            variance=self.variance,
            noise=self.noise,
        )
        best = min(y_train)
        acquisition = _expected_improvement(mean, var, best, self.exploration)

        evaluated = [ev.epsilon for ev in self._evaluations]
        order = sorted(range(len(x_test)), key=lambda i: acquisition[i], reverse=True)
        for idx in order:
            candidate = x_test[idx]
            if all(abs(candidate - seen) > 1e-6 for seen in evaluated):
                return candidate
        return self._rng.uniform(self.bounds[0], self.bounds[1])

    def _best_epsilon(self) -> float:
        if not self._evaluations:
            return self._epsilon
        best = min(self._evaluations, key=lambda ev: ev.objective)
        return best.epsilon

    def _run_optimisation(self) -> None:
        if not self._evaluations:
            self._register(self._epsilon)

        for _ in range(self.optimisation_steps):
            candidate = self._suggest_candidate()
            self._register(candidate)

        target = self._best_epsilon()
        if self.smoothing == 0.0:
            self._epsilon = self._clip(target)
        else:
            blended = (1.0 - self.smoothing) * self._epsilon + self.smoothing * target
            self._epsilon = self._clip(blended)


__all__ = [
    "AutoEpsilonOptimizer",
    "AutoEpsilonReport",
    "SimulationResult",
]

