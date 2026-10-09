"""bandit/environment.py — deterministic synthetic contextual bandit DGP.

Two regimes:
  * ``linear``    : r = x·θ_a + ε   (the canonical LinUCB/LinTS setting)
  * ``quadratic`` : r = x·θ_a + 0.3((x·θ_a)²−1) + ε  (mild misspecification,
                   reported honestly as a stress test where linear methods degrade)

The environment is the *oracle*: it knows the true reward surfaces, so it can
supply the optimal arm and the noise-free optimal reward used to score regret.
"""

from __future__ import annotations

import numpy as np

from core.errors import DataError
from core.interfaces import Environment


class LinearBandit(Environment):
    def __init__(self, d: int, k: int, sigma: float, regime: str = "linear", seed: int = 0):
        if d < 1 or k < 2 or sigma <= 0:
            raise DataError(f"invalid env params d={d} k={k} sigma={sigma}")
        if regime not in ("linear", "quadratic"):
            raise DataError(f"unknown regime {regime!r}")
        self.d, self.k, self.sigma, self.regime = d, k, sigma, regime
        self.rng = np.random.default_rng(int(seed))
        self.Theta = self.rng.standard_normal((k, d))

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.Theta = self.rng.standard_normal((self.k, self.d))

    def sample_context(self) -> np.ndarray:
        return self.rng.standard_normal(self.d)

    def _mean(self, x: np.ndarray, a: int) -> float:
        lin = float(np.dot(x, self.Theta[a]))
        if self.regime == "linear":
            return lin
        return lin + 0.3 * (lin * lin - 1.0)

    def reward(self, x: np.ndarray, a: int) -> float:
        return self._mean(x, a) + float(self.rng.standard_normal()) * self.sigma

    def optimal_arm(self, x: np.ndarray) -> int:
        return int(np.argmax([self._mean(x, a) for a in range(self.k)]))

    def optimal_reward(self, x: np.ndarray) -> float:
        return float(max(self._mean(x, a) for a in range(self.k)))

    @property
    def true_sigma(self) -> float:
        return self.sigma
