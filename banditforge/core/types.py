"""Typed data structures shared across BanditForge modules.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BanditLog:
    """Offline logged bandit data.

    Attributes:
        contexts: (n, d) context matrix.
        actions: (n,) chosen arm indices in [0, K).
        rewards: (n,) observed rewards (higher is better).
        propensities: (n,) logging policy probability of the chosen action.
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray

    def __post_init__(self) -> None:
        n = self.contexts.shape[0]
        if self.actions.shape[0] != n or self.rewards.shape[0] != n:
            raise ValueError("contexts/actions/rewards must share the same length")
        if self.propensities.shape[0] != n:
            raise ValueError("propensities must match contexts length")
        if np.any(self.propensities <= 0.0):
            raise ValueError("propensities must be strictly positive")

    @property
    def n(self) -> int:
        return int(self.contexts.shape[0])

    def num_actions(self) -> int:
        return int(self.actions.max()) + 1

    def context_dim(self) -> int:
        return int(self.contexts.shape[1])

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "contexts": self.contexts,
            "actions": self.actions,
            "rewards": self.rewards,
            "propensities": self.propensities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> "BanditLog":
        return cls(
            contexts=np.asarray(data["contexts"], dtype=np.float64),
            actions=np.asarray(data["actions"], dtype=np.int64),
            rewards=np.asarray(data["rewards"], dtype=np.float64),
            propensities=np.asarray(data["propensities"], dtype=np.float64),
        )


@dataclass
class PolicyRunResult:
    """Result of one on-policy simulation run.

    Attributes:
        cumulative_regret: regret accumulated at each step (length T).
        realized_rewards: observed reward per step (length T).
        chosen_actions: arm index per step (length T).
        best_expected_reward: E[r*|x_t] per step (length T), ground truth.
    """

    cumulative_regret: np.ndarray
    realized_rewards: np.ndarray
    chosen_actions: np.ndarray
    best_expected_reward: np.ndarray

    @property
    def final_regret(self) -> float:
        return float(self.cumulative_regret[-1]) if len(self.cumulative_regret) else 0.0


@dataclass
class OPEEstimate:
    """One off-policy value estimate.

    Attributes:
        estimator: estimator name (ips / snips / dm / dr / cfdrac / gated).
        value: estimated policy value.
        ess_ratio: effective sample size ratio (ESS/n) used for diagnostics.
        clip_tau: ratio clipping threshold used (inf = no clipping).
        reason: human-readable rationale for estimator choice (gating).
    """

    estimator: str
    value: float
    ess_ratio: float = float("nan")
    clip_tau: float = float("inf")
    reason: str = ""

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "estimator": self.estimator,
            "value": round(self.value, 6),
            "ess_ratio": None if np.isnan(self.ess_ratio) else round(float(self.ess_ratio), 6),
            "clip_tau": None if np.isinf(self.clip_tau) else float(self.clip_tau),
            "reason": self.reason,
        }


@dataclass
class PolicyBenchmarkRow:
    """Aggregated policy-learning benchmark result for one policy."""

    policy: str
    regret_mean: float
    regret_std: float
    seed_values: List[float] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "regret_mean": round(self.regret_mean, 4),
            "regret_std": round(self.regret_std, 4),
            "seed_values": [round(v, 4) for v in self.seed_values],
        }


@dataclass
class OPEBenchmarkRow:
    """Aggregated OPE benchmark result for one estimator at one dataset spec.

    rmse computed across (seed x repeat) log datasets against the Monte-Carlo
    ground-truth value of the evaluation policy.
    """

    estimator: str
    dataset: str
    rmse_mean: float
    rmse_std: float
    bias_mean: float
    seed_values: List[float] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "estimator": self.estimator,
            "dataset": self.dataset,
            "rmse_mean": round(self.rmse_mean, 6),
            "rmse_std": round(self.rmse_std, 6),
            "bias_mean": round(self.bias_mean, 6),
            "seed_values": [round(v, 6) for v in self.seed_values],
        }


@dataclass
class DatasetSpec:
    """Specification of one synthetic benchmark dataset config."""

    name: str
    num_actions: int
    context_dim: int
    overlap: str  # "high" | "medium" | "low" (logging softmax temperature)
    n_log: int
    logging_temp: float
    reward_noise: float
    true_value: float  # Monte-Carlo ground truth of the evaluation policy

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "num_actions": self.num_actions,
            "context_dim": self.context_dim,
            "overlap": self.overlap,
            "n_log": self.n_log,
            "logging_temp": self.logging_temp,
            "reward_noise": self.reward_noise,
            "true_value": round(self.true_value, 6),
        }


@dataclass
class GateCheck:
    """One acceptance-gate check item."""

    name: str
    passed: bool
    detail: str


@dataclass
class PipelineReport:
    """Top-level pipeline report (serialized to benchmark.json)."""

    version: str
    seed: int
    specs: List[DatasetSpec] = field(default_factory=list)
    policy_benchmark: List[PolicyBenchmarkRow] = field(default_factory=list)
    ope_benchmark: List[OPEBenchmarkRow] = field(default_factory=list)
    ablation: List[OPEBenchmarkRow] = field(default_factory=list)
    failure_cases: List[Dict[str, object]] = field(default_factory=list)
    gates: List[GateCheck] = field(default_factory=list)
    hpo: Optional[Dict[str, object]] = None
    elapsed_sec: float = 0.0

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "datasets": [s.to_json_dict() for s in self.specs],
            "policy_benchmark": [r.to_json_dict() for r in self.policy_benchmark],
            "ope_benchmark": [r.to_json_dict() for r in self.ope_benchmark],
            "ablation": [r.to_json_dict() for r in self.ablation],
            "failure_cases": self.failure_cases,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail} for g in self.gates
            ],
            "hpo": self.hpo,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }
