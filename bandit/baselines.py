"""bandit/baselines.py — context-free baselines (no exploration / ε-greedy / random).

These establish the floor: policies that either ignore context entirely or
explore with a fixed, context-blind rate. They are the honest lower bounds that
contextual algorithms must beat.
"""

from __future__ import annotations

import numpy as np


class EpsilonGreedy:
    """ε-greedy on empirical arm means (context-free)."""

    def __init__(self, k: int, eps: float, rng: np.random.Generator | None = None):
        self.k, self.eps = k, float(eps)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n = np.zeros(k, dtype=np.float64)
        self.q = np.zeros(k, dtype=np.float64)

    def act(self, x: np.ndarray) -> int:
        if self.n.sum() < self.k or self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.q))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.n[a] += 1.0
        self.q[a] += (r - self.q[a]) / self.n[a]


class GreedyCF:
    """Pure greedy on empirical means — zero exploration, context-free."""

    def __init__(self, k: int):
        self.k = k
        self.n = np.zeros(k, dtype=np.float64)
        self.q = np.zeros(k, dtype=np.float64)

    def act(self, x: np.ndarray) -> int:
        if self.n.sum() < self.k:
            return int(self.n.sum())
        return int(np.argmax(self.q))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        self.n[a] += 1.0
        self.q[a] += (r - self.q[a]) / self.n[a]


class RandomCF:
    """Uniform-random arm selection (context-free, no learning)."""

    def __init__(self, k: int, rng: np.random.Generator | None = None):
        self.k = k
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def act(self, x: np.ndarray) -> int:
        return int(self.rng.integers(self.k))

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        pass
