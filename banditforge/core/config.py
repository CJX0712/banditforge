"""Runtime configuration with ENV_BF_* overrides and schema validation.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from .errors import ConfigError

# (env_suffix, field, default, type)
_SCHEMA: List[tuple] = [
    ("SEED", "seed", 20260928, int),
    ("N_LOG", "n_log", 2000, int),
    ("N_MC", "n_mc", 100_000, int),
    ("T_STEPS", "t_steps", 3000, int),
    ("N_SEEDS", "n_seeds", 3, int),
    ("N_REPEAT", "n_repeat", 20, int),
    ("HPO_TRIALS", "hpo_trials", 20, int),
]


@dataclass
class EnvConfig:
    """Pipeline configuration; every field overridable via ENV_BF_<SUFFIX>."""

    seed: int = 20260928
    n_log: int = 2000
    n_mc: int = 100_000
    t_steps: int = 3000
    n_seeds: int = 3
    n_repeat: int = 16
    hpo_trials: int = 20
    extras: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        if self.n_log < 200:
            raise ConfigError("n_log must be >= 200 for stable OPE")
        if self.n_mc < 10_000:
            raise ConfigError("n_mc must be >= 10000 for reliable ground truth")
        if self.n_seeds < 3:
            raise ConfigError("n_seeds must be >= 3 (statistical rigor gate)")
        if self.n_repeat < 4:
            raise ConfigError("n_repeat must be >= 4 for RMSE aggregation")
        if self.t_steps < 500:
            raise ConfigError("t_steps must be >= 500")

    @classmethod
    def from_env(cls, env: Dict[str, str] = None) -> "EnvConfig":
        env = dict(os.environ) if env is None else dict(env)
        kwargs = {}
        for suffix, field_name, default, cast in _SCHEMA:
            raw = env.get(f"ENV_BF_{suffix}")
            if raw is None or raw == "":
                kwargs[field_name] = default
                continue
            try:
                kwargs[field_name] = cast(raw)
            except ValueError as exc:
                raise ConfigError(
                    f"ENV_BF_{suffix}={raw!r} is not a valid {cast.__name__}"
                ) from exc
        known = {f"ENV_BF_{s}" for s, *_ in _SCHEMA}
        extras = {k: v for k, v in env.items() if k.startswith("ENV_BF_") and k not in known}
        return cls(**kwargs, extras=extras)
