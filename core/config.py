"""core/config.py — environment-overridable configuration with schema check."""

import os
from typing import Any, Dict

from .types import BanditConfig

_ENV_PREFIX = "BANDFORGE_"


_TYPE_COERCION = {
    "d": int,
    "k": int,
    "sigma": float,
    "n_rounds": int,
    "regime": str,
    "n_seeds": int,
    "base_seed": int,
    "linucb_alpha": float,
    "lints_v": float,
    "fuse_z": float,
    "fuse_init_var": float,
    "fuse_amb": float,
}


def load_config(**overrides: Any) -> BanditConfig:
    """Build a BanditConfig, applying ``BANDFORGE_*`` env overrides then kwargs."""
    values: Dict[str, Any] = {}
    for key in _TYPE_COERCION:
        env_val = os.environ.get(f"{_ENV_PREFIX}{key.upper()}")
        if env_val is not None:
            try:
                values[key] = _TYPE_COERCION[key](env_val)
            except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Invalid env override BANDFORGE_{key.upper()}={env_val!r}: {exc}"
                ) from exc
    values.update(overrides)
    cfg = BanditConfig(**values)
    cfg.validate()
    return cfg
