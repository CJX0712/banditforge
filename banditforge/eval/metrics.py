"""Metrics: regret, RMSE, and the simple significance gate.

Significance gate (V4 statistical rigor): the mean difference must exceed
half the sum of the two std's:  |m1 - m2| > 0.5 * (s1 + s2).

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def rmse(values: Sequence[float], truth: float) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((v - truth) ** 2)))


def bias(values: Sequence[float], truth: float) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.mean(v) - truth)


def significant(win_mean: float, win_std: float, base_mean: float, base_std: float) -> bool:
    """True if the mean gap exceeds 0.5*(std1+std2)."""
    return abs(win_mean - base_mean) > 0.5 * (win_std + base_std)


def relative_reduction(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (base - new) / base
