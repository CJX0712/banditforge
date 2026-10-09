"""core/seed.py — global deterministic RNG management.

World-class reproducibility contract: a single integer seed fans out to every
random source used by BanditForge (NumPy, Python ``random``). All algorithms and
the synthetic environment consume an explicit ``np.random.Generator`` so that two
runs with the same seed produce bit-identical regret trajectories.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class RngBundle:
    """A bundle of deterministic random sources derived from one master seed."""

    seed: int
    np: np.random.Generator
    py: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> "RngBundle":
        seed = int(seed)
        return cls(
            seed=seed,
            np=np.random.default_rng(seed),
            py=random.Random(seed),
        )

    def spawn(self, salt: int) -> np.random.Generator:
        """Derive a child generator for a sub-component without mutating state."""
        mixed = (int(self.seed) * 0x9E3779B1 + int(salt) * 0x85EBCA77) & 0xFFFFFFFF
        return np.random.default_rng(mixed)


def set_all(seed: int) -> RngBundle:
    """Convenience entry point: build the full deterministic RNG bundle."""
    return RngBundle.from_seed(seed)


def new_generator(seed: int) -> np.random.Generator:
    """Create a standalone NumPy generator (used by algorithms needing their own)."""
    return np.random.default_rng(int(seed))
