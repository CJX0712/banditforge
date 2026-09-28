"""Base policy class + registry.

Numerics: per-arm ridge precision matrices A_k start at identity (Tikhonov
prior), guaranteeing invertibility; inverse is recomputed on update (d is
small, so O(K d^3) per update is negligible).

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.errors import PolicyError


class BasePolicy:
    """Stateful contextual bandit policy base."""

    name = "base"

    def __init__(self, num_actions: int, context_dim: int,
                 rng: Optional[np.random.Generator] = None) -> None:
        if num_actions < 2:
            raise PolicyError("num_actions must be >= 2")
        if context_dim < 1:
            raise PolicyError("context_dim must be >= 1")
        self.K = num_actions
        self.d = context_dim
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def select_arm(self, context: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        raise NotImplementedError

    @property
    def params(self) -> Dict[str, float]:
        return {}


def _as_context(context: np.ndarray, d: int) -> np.ndarray:
    x = np.asarray(context, dtype=np.float64)
    if x.shape != (d,):
        raise PolicyError(f"context shape {x.shape} != ({d},)")
    return x


class LinearGreedy(BasePolicy):
    """epsilon-greedy over per-arm ridge estimates (strong classical baseline)."""

    name = "eps_greedy"

    def __init__(self, num_actions: int, context_dim: int, eps: float = 0.1,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(num_actions, context_dim, rng)
        if not 0.0 <= eps <= 1.0:
            raise PolicyError("eps must be in [0, 1]")
        self.eps = eps
        self._A = np.array([np.eye(self.d) for _ in range(self.K)])
        self._b = np.zeros((self.K, self.d))

    @property
    def params(self) -> Dict[str, float]:
        return {"eps": self.eps}

    def _theta(self) -> np.ndarray:
        return np.stack([np.linalg.solve(self._A[k], self._b[k]) for k in range(self.K)])

    def select_arm(self, context: np.ndarray) -> int:
        x = _as_context(context, self.d)
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.K))
        scores = self._theta() @ x
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = _as_context(context, self.d)
        self._A[action] += np.outer(x, x)
        self._b[action] += reward * x


class LinUCB(BasePolicy):
    """Disjoint LinUCB (Li et al. 2010): greedy + confidence bonus."""

    name = "linucb"

    def __init__(self, num_actions: int, context_dim: int, alpha: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(num_actions, context_dim, rng)
        if alpha < 0:
            raise PolicyError("alpha must be >= 0")
        self.alpha = alpha
        self._A = np.array([np.eye(self.d) for _ in range(self.K)])
        self._b = np.zeros((self.K, self.d))
        self._Ainv = np.array([np.linalg.inv(a) for a in self._A])

    @property
    def params(self) -> Dict[str, float]:
        return {"alpha": self.alpha}

    def select_arm(self, context: np.ndarray) -> int:
        x = _as_context(context, self.d)
        best, best_score = 0, -np.inf
        for k in range(self.K):
            theta = self._Ainv[k] @ self._b[k]
            bonus = self.alpha * float(np.sqrt(max(x @ self._Ainv[k] @ x, 0.0)))
            score = float(theta @ x) + bonus
            if score > best_score:
                best, best_score = k, score
        return best

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = _as_context(context, self.d)
        self._A[action] += np.outer(x, x)
        self._b[action] += reward * x
        self._Ainv[action] = np.linalg.inv(self._A[action])


class LinTS(BasePolicy):
    """Linear Thompson Sampling (Agrawal & Goyal 2013) with Gaussian prior."""

    name = "lints"

    def __init__(self, num_actions: int, context_dim: int, scale: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(num_actions, context_dim, rng)
        if scale <= 0:
            raise PolicyError("scale must be > 0")
        self.scale = scale
        self._A = np.array([np.eye(self.d) for _ in range(self.K)])
        self._b = np.zeros((self.K, self.d))

    @property
    def params(self) -> Dict[str, float]:
        return {"scale": self.scale}

    def select_arm(self, context: np.ndarray) -> int:
        x = _as_context(context, self.d)
        scores = np.empty(self.K)
        for k in range(self.K):
            Ainv = np.linalg.inv(self._A[k])
            mu = Ainv @ self._b[k]
            cov = (self.scale ** 2) * Ainv
            theta = self.rng.multivariate_normal(mu, cov)
            scores[k] = theta @ x
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = _as_context(context, self.d)
        self._A[action] += np.outer(x, x)
        self._b[action] += reward * x


class UCB1(BasePolicy):
    """Classic UCB1 (context-agnostic MAB baseline; contexts still logged)."""

    name = "ucb1"

    def __init__(self, num_actions: int, context_dim: int, c: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(num_actions, context_dim, rng)
        if c <= 0:
            raise PolicyError("c must be > 0")
        self.c = c
        self.counts = np.zeros(num_actions)
        self.sums = np.zeros(num_actions)
        self.t = 0

    @property
    def params(self) -> Dict[str, float]:
        return {"c": self.c}

    def select_arm(self, context: np.ndarray) -> int:
        _ = _as_context(context, self.d)
        self.t += 1
        for k in range(self.K):
            if self.counts[k] == 0:
                return k
        means = self.sums / self.counts
        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        _ = _as_context(context, self.d)
        self.counts[action] += 1
        self.sums[action] += reward


POLICY_REGISTRY = {
    "eps_greedy": LinearGreedy,
    "linucb": LinUCB,
    "lints": LinTS,
    "ucb1": UCB1,
}


def build_policy(name: str, num_actions: int, context_dim: int,
                 rng: np.random.Generator = None, **params) -> BasePolicy:
    if name not in POLICY_REGISTRY:
        raise PolicyError(f"unknown policy {name!r}; known: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name](num_actions, context_dim, rng=rng, **params)
