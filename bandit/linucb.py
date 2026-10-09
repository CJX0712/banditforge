"""bandit/linucb.py — Disjoint Linear UCB (Li et al., 2010).

The canonical optimistic contextual bandit. Per arm it maintains the ridge
least-squares posterior (A_a, b_a) and selects the arm maximising
``x·θ̂_a + α·√(xᵀ A_a⁻¹ x)`` — a confidence-radius around the reward estimate.
"""

from __future__ import annotations

import numpy as np


class LinUCB:
    def __init__(self, d: int, k: int, alpha: float):
        self.d, self.k, self.alpha = d, k, float(alpha)
        self.A = [np.eye(d, dtype=np.float64) for _ in range(k)]
        self.b = [np.zeros(d, dtype=np.float64) for _ in range(k)]

    def act(self, x: np.ndarray) -> int:
        best, ba = -np.inf, 0
        for a in range(self.k):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            bonus = self.alpha * np.sqrt(max(float(x @ Ainv @ x), 1e-12))
            ucb = float(x @ theta) + bonus
            if ucb > best:
                best, ba = ucb, a
        return ba

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
