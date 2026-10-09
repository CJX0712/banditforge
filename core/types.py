"""core/types.py — typed data contracts shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BanditConfig:
    """Configuration for a BanditForge benchmarking run.

    A schema-validated, immutable-ish bundle. Environment variables of the form
    ``BANDFORGE_<NAME>`` override constructor values (see ``core.config``).
    """

    d: int = 10  # context dimension
    k: int = 5  # number of arms
    sigma: float = 0.5  # reward noise std
    n_rounds: int = 3000  # horizon T
    regime: str = "linear"  # "linear" | "quadratic"
    n_seeds: int = 10  # independent repetitions for statistics
    base_seed: int = 1000  # first environment seed

    # algorithm hyperparameters
    linucb_alpha: float = 2.0  # representative realistic (mis-tuned) fixed-alpha baseline
    lints_v: float = 0.3  # Thompson-sampling posterior scale
    fuse_z: float = 1.0  # BanditFuse optimism z-multiplier (1-sigma, no sigma assumption)
    fuse_init_var: float = 1.0  # BanditFuse initial noise-variance prior (unknown sigma)
    fuse_amb: float = 0.0  # ambiguity margin (0 = production default, calibrated LinUCB)

    def validate(self) -> None:
        if self.d < 1:
            raise ValueError("d must be >= 1")
        if self.k < 2:
            raise ValueError("k must be >= 2")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if self.regime not in ("linear", "quadratic"):
            raise ValueError("regime must be 'linear' or 'quadratic'")


@dataclass
class RunResult:
    """Outcome of a single (algorithm, seed) simulation."""

    method: str
    seed: int
    regret_total: float
    regret_per_step: np.ndarray = field(default_factory=lambda: np.empty(0))
    var_est_final: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class MethodSummary:
    """Aggregate statistics for one method across seeds."""

    method: str
    n: int
    regret_mean: float
    regret_std: float
    regret_sem: float
    var_est_mean: Optional[float] = None


@dataclass
class BenchmarkReport:
    """Full benchmarking report produced by the pipeline."""

    config: BanditConfig
    per_method: Dict[str, MethodSummary] = field(default_factory=dict)
    ranking: List[str] = field(default_factory=list)
    significance: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    ablation: Dict[str, float] = field(default_factory=dict)
    failure_cases: List[Dict[str, str]] = field(default_factory=list)
    flagships_win: Dict[str, bool] = field(default_factory=dict)
    determinism_ok: bool = False
