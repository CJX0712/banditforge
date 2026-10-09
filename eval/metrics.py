"""eval/metrics.py — regret statistics and significance testing.

World-class statistical rigor: every performance claim is averaged over
``>=3`` seeds and a win is declared only when the mean gap exceeds the
*combined* half-standard-deviation threshold ``½(σ_a + σ_b)``.
"""

from __future__ import annotations

import numpy as np

from core.types import MethodSummary, RunResult


def significance_mean_diff(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """True iff |mean_a − mean_b| exceeds ½(σ_a + σ_b)."""
    return bool(abs(mean_a - mean_b) > 0.5 * (std_a + std_b))


def relative_reduction(baseline_mean: float, method_mean: float) -> float:
    """Fractional regret reduction of ``method`` relative to ``baseline``."""
    if abs(baseline_mean) < 1e-12:
        return 0.0
    return (baseline_mean - method_mean) / abs(baseline_mean)


def summarize(method: str, results: list[RunResult]) -> MethodSummary:
    regrets = np.array([r.regret_total for r in results], dtype=np.float64)
    n = len(regrets)
    mean = float(regrets.mean())
    std = float(regrets.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    var_est = [r.var_est_final for r in results if r.var_est_final is not None]
    vm = float(np.mean(var_est)) if var_est else None
    return MethodSummary(method, n, mean, std, sem, vm)


def regret_reduction_vs(results: list[RunResult], baseline_mean: float) -> float:
    return relative_reduction(baseline_mean, float(np.mean([r.regret_total for r in results])))
