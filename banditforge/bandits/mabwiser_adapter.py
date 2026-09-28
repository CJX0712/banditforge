"""Optional mabwiser SOTA-backend adapter (auto-skipped when unavailable).

mabwiser (Fidelity International) provides battle-tested MAB learners; the
contextual usage wraps a neighborhood policy over historical contexts.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.types import BanditLog

_MABWISER_ERR: Optional[str] = None


def available_mabwiser() -> bool:
    global _MABWISER_ERR
    if _MABWISER_ERR is not None:
        return False
    try:
        from mabwiser.mab import MAB  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover - depends on env
        _MABWISER_ERR = str(exc)
        return False


class MabwiserLinTS:
    """mabwiser LinTS backend for contextual decisions (SOTA cross-check)."""

    def __init__(self, num_actions: int, context_dim: int,
                 alpha: float = 1.0, rng: Optional[np.random.Generator] = None) -> None:
        del context_dim, rng
        if not available_mabwiser():
            raise ImportError(f"mabwiser unavailable: {_MABWISER_ERR}")
        from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        self.arms: List[int] = list(range(num_actions))
        self._hist_X = np.empty((0, 0))
        self._hist_a = np.empty(0, dtype=int)
        self._hist_r = np.empty(0)
        self._mab = MAB(
            arms=self.arms,
            learning_policy=LearningPolicy.LinTS(alpha=alpha),
            neighborhood=NeighborhoodPolicy.KNearest(k=10),
        )

    name = "mabwiser-lints"

    @property
    def params(self):
        return {}

    def select_arm(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=np.float64).reshape(1, -1)
        if self._hist_X.shape[0] == 0:
            return int(np.random.randint(len(self.arms)))
        return int(self._mab.predict(x))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64).reshape(1, -1)
        if self._hist_X.shape[0] == 0:
            self._hist_X = x
            self._mab.fit(x, np.array([action]), np.array([reward]))
        else:
            self._hist_X = np.vstack([self._hist_X, x])
            self._mab.partial_fit(x, np.array([action]), np.array([reward]))


def mabwiser_crosscheck(log: BanditLog) -> dict:
    """Cross-check: train mabwiser LinTS on a log, return its greedy decisions.

    Used as an external sanity reference for the hand-written LinTS: on a
    static log both should rank contexts similarly. Returns empty dict when
    backend is unavailable (benchmark marks the row 'skipped').
    """
    if not available_mabwiser():
        return {"status": "skipped", "reason": _MABWISER_ERR or "mabwiser not importable"}
    try:
        from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        arms = list(range(int(log.actions.max()) + 1))
        mab = MAB(arms=arms, learning_policy=LearningPolicy.LinTS(alpha=1.0),
                  neighborhood=NeighborhoodPolicy.KNearest(k=10))
        mab.fit(log.contexts, log.actions, log.rewards)
        preds = mab.predict(log.contexts[:100])
        return {"status": "ok", "first100_predictions": [int(p) for p in preds]}
    except Exception as exc:  # pragma: no cover - backend drift
        return {"status": "skipped", "reason": str(exc)}
