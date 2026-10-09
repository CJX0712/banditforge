"""Interface contracts (Protocols) binding the layers together.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import BanditLog, OPEEstimate


@runtime_checkable
class BanditPolicy(Protocol):
    """Online contextual bandit policy.

    Contract: ``select_arm`` must be callable before any update; policies are
    stateful and updated strictly after each observation.
    """

    name: str

    def select_arm(self, context: np.ndarray) -> int: ...

    def update(self, context: np.ndarray, action: int, reward: float) -> None: ...


@runtime_checkable
class OPEEstimator(Protocol):
    """Off-policy value estimator.

    Contract: returns an estimated policy value (higher = better policy);
    ``estimate`` must not mutate the input log.
    """

    name: str

    def estimate(self, log: BanditLog) -> OPEEstimate: ...


@runtime_checkable
class EvalPolicyWithPropensity(Protocol):
    """Evaluation policy exposing per-action action-selection probabilities.

    Contract: ``propensity`` must return probabilities summing to 1 over arms
    for any context; used by IPS/SNIPS/DR.
    """

    name: str

    def propensity(self, context: np.ndarray, num_actions: int) -> np.ndarray: ...

    def value_of_action(self, context: np.ndarray, action: int, beta: np.ndarray) -> float:
        """Expected reward of ``action`` under the known ground-truth model."""
        ...
