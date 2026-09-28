"""Global deterministic seeding.

Single entry point ``set_all`` seeds stdlib random and numpy. Library-level
seeds (sklearn uses numpy global RNG) follow automatically.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

import random

import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """Seed every RNG BanditForge uses; return a dedicated Generator.

    The returned Generator is used for all data generation and simulation so
    that two runs with the same seed produce bit-identical results.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return np.random.default_rng(seed)
