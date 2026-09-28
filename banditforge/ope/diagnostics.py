"""Overlap diagnostics for offline logs (oracle-free)."""
from __future__ import annotations

import numpy as np

from ..core.types import BanditLog
from .reward_model import ess_ratio


def weight_ratios(log: BanditLog, pe_chosen: np.ndarray) -> np.ndarray:
    """Importance weights w_i = p_e(a_i|x_i) / p_log(a_i)."""
    pe = np.asarray(pe_chosen, dtype=np.float64)
    return pe / log.propensities


def overlap_report(log: BanditLog, pe_chosen: np.ndarray) -> dict:
    """Oracle-free overlap diagnostics: ESS ratio, weight quantiles, max w."""
    w = weight_ratios(log, pe_chosen)
    qs = np.quantile(w, [0.5, 0.9, 0.99])
    return {
        "ess_ratio": round(ess_ratio(w), 6),
        "w_median": round(float(qs[0]), 6),
        "w_p90": round(float(qs[1]), 6),
        "w_p99": round(float(qs[2]), 6),
        "w_max": round(float(w.max()), 6),
    }
