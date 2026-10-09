"""bandit/ucb1.py — UCB1, the classic context-free stochastic bandit (Auer 2002).

This is the *context-free* SOTA baseline. It learns only the empirical mean of
each arm and ignores the context entirely, using a count-based optimism bonus
``√(2 ln t / n_a)``. On a contextual problem it is provably sub-optimal, which
is exactly the contrast BanditForge demonstrates.
"""

from __future__ import annotations

import numpy as np


class UCB1:
    def __init__(self, k: int):
        self.k = k
        self.n = np.zeros(k, dtype=np.float64)
        self.q = np.zeros(k, dtype=np.float64)

    def act(self, x: np.ndarray) -> int:
        t = self.n.sum() + 1.0
        for a in range(self.k):
            if self.n[a] == 0:
                return a
        ucb = self.q + np.sqrt(2.0 * np.log(t) / self.n)
        return int(np.argmax(ucb))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.n[a] += 1.0
        self.q[a] += (r - self.q[a]) / self.n[a]
