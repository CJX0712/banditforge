"""core/interfaces.py — Protocol contracts for pluggable components.

Keeping the environment and algorithm behind Protocols guarantees that the
pipeline, evaluators and tests can swap implementations (SOTA backends, offline
fallbacks, oracles) without touching orchestration code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BanditAlgorithm(Protocol):
    """Any online bandit policy."""

    def act(self, x: np.ndarray) -> int:
        """Observe context ``x`` and return a chosen arm index."""
        ...

    def update(self, x: np.ndarray, a: int, r: float) -> None:
        """Incorporate the observed reward ``r`` for chosen arm ``a``."""
        ...


@runtime_checkable
class Environment(Protocol):
    """A contextual bandit environment."""

    def sample_context(self) -> np.ndarray:
        """Draw an i.i.d. context vector."""
        ...

    def reward(self, x: np.ndarray, a: int) -> float:
        """Return a (possibly noisy) reward for arm ``a`` under context ``x``."""
        ...

    def optimal_arm(self, x: np.ndarray) -> int:
        """Return the arm with the highest true mean reward for ``x``."""
        ...

    def optimal_reward(self, x: np.ndarray) -> float:
        """Return the highest true mean reward for ``x``."""
        ...
