"""Numpy closed-form ridge reward model + K-fold cross-fitting.

Per-arm ridge with closed-form solve (d is small): theta_k = (X_k'X_k + lam I)^-1 X_k'r_k.
Cross-fitting produces out-of-fold (OOF) action-score predictions, removing
the own-observation overfitting that biases plug-in DR corrections.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.errors import OPEError
from ..core.types import BanditLog


class RidgeRewardModel:
    """Per-arm ridge regression mu_hat_k(x)."""

    def __init__(self, num_actions: int, context_dim: int, lam: float = 1.0) -> None:
        if lam <= 0:
            raise OPEError("ridge lambda must be > 0")
        self.K = num_actions
        self.d = context_dim
        self.lam = lam
        self._theta = np.zeros((self.K, self.d))

    def fit(self, log: BanditLog) -> "RidgeRewardModel":
        for k in range(self.K):
            mask = log.actions == k
            if mask.sum() < 2:
                self._theta[k] = 0.0
                continue
            Xk = log.contexts[mask]
            Ak = Xk.T @ Xk + self.lam * np.eye(self.d)
            bk = Xk.T @ log.rewards[mask]
            self._theta[k] = np.linalg.solve(Ak, bk)
        return self

    def predict_all(self, contexts: np.ndarray) -> np.ndarray:
        """(n, K) matrix of mu_hat_k(x_i)."""
        return contexts @ self._theta.T

    def predict_action(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """(n,) mu_hat at each logged action."""
        allp = self.predict_all(contexts)
        return allp[np.arange(len(actions)), actions]

    def oof_r2(self, log: BanditLog) -> float:
        """Pseudo-R^2 of the model on the log (in-sample diagnostic)."""
        pred = self.predict_action(log.contexts, log.actions)
        ss_res = float(np.sum((log.rewards - pred) ** 2))
        ss_tot = float(np.sum((log.rewards - log.rewards.mean()) ** 2))
        if ss_tot <= 0:
            return 0.0
        return 1.0 - ss_res / ss_tot


def cross_fit_predictions(log: BanditLog, num_actions: int, context_dim: int,
                          n_folds: int = 5, lam: float = 1.0,
                          rng: np.random.Generator = None) -> Tuple[np.ndarray, RidgeRewardModel]:
    """Return (oof_action_scores, full_model).

    oof_action_scores[i] = mu_hat_{(-fold(i))}(x_i, a_i): prediction at the
    logged action from the model trained WITHOUT fold i.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = log.n
    if n_folds < 2:
        raise OPEError("n_folds must be >= 2")
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    oof = np.empty(n)
    for f in range(n_folds):
        train_mask = folds != f
        test_mask = folds == f
        if test_mask.sum() == 0:
            continue
        sub = BanditLog(
            contexts=log.contexts[train_mask],
            actions=log.actions[train_mask],
            rewards=log.rewards[train_mask],
            propensities=log.propensities[train_mask],
        )
        m = RidgeRewardModel(num_actions, context_dim, lam=lam).fit(sub)
        oof[test_mask] = m.predict_action(log.contexts[test_mask], log.actions[test_mask])
    full = RidgeRewardModel(num_actions, context_dim, lam=lam).fit(log)
    return oof, full


def clip_weights(w: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        raise OPEError("clip tau must be > 0")
    return np.minimum(w, tau)


def ess_ratio(w: np.ndarray) -> float:
    """Effective sample size ratio: (sum w)^2 / (sum w^2) / n."""
    s = float(np.sum(w))
    q = float(np.sum(w ** 2))
    if q <= 0:
        return 0.0
    return (s * s / q) / max(len(w), 1)


# Grid fixed by a tau sweep on the benchmark DGPs (see docs/architecture.md):
# tau<=10 over-clips and inflates bias at low overlap; tau=20 is the sweet
# spot, tau=inf recovers unclipped DR as the conservative endpoint.
TAU_GRID: Tuple[float, ...] = (20.0, 50.0, float("inf"))
