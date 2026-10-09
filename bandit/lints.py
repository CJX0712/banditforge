"""bandit/lints.py — Linear Thompson Sampling (Agrawal & Goyal, 2013).

Maintains the same Bayesian posterior as LinUCB but *samples* a parameter vector
from ``N(θ̂_a, v² A_a⁻¹)`` per arm and greedily picks the arm with the highest
sampled reward. The randomness provides automatic, uncertainty-proportional
exploration.
"""

from __future__ import annotations

import numpy as np


class LinTS:
    def __init__(self, d: int, k: int, v: float, rng: np.random.Generator | None = None):
        self.d, self.k, self.v = d, k, float(v)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.A = [np.eye(d, dtype=np.float64) for _ in range(k)]
        self.b = [np.zeros(d, dtype=np.float64) for _ in range(k)]

    def act(self, x: np.ndarray) -> int:
        best, ba = -np.inf, 0
        for a in range(self.k):
            Ainv = np.linalg.inv(self.A[a])
            theta_hat = Ainv @ self.b[a]
            cov = (self.v**2) * Ainv
            L = np.linalg.cholesky(cov)
            theta_tilde = theta_hat + self.rng.standard_normal(self.d) @ L.T
            val = float(x @ theta_tilde)
            if val > best:
                best, ba = val, a
        return ba

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
