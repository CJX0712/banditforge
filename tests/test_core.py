"""Core layer tests: config, seed, errors, types."""
import numpy as np
import pytest

from banditforge.core.config import ConfigError, EnvConfig
from banditforge.core.errors import BanditForgeError, OPEError
from banditforge.core.seed import set_all
from banditforge.core.types import BanditLog


def test_config_defaults_valid():
    cfg = EnvConfig.from_env({})
    assert cfg.seed == 20260928 and cfg.n_seeds >= 3


def test_config_env_override_and_reject():
    cfg = EnvConfig.from_env({"ENV_BF_N_LOG": "500", "ENV_BF_SEED": "1"})
    assert cfg.n_log == 500 and cfg.seed == 1
    with pytest.raises(ConfigError):
        EnvConfig.from_env({"ENV_BF_N_LOG": "abc"})
    with pytest.raises(ConfigError):
        EnvConfig.from_env({"ENV_BF_N_SEEDS": "2"})


def test_seed_determinism():
    g1 = set_all(123)
    a = g1.standard_normal(5)
    g2 = set_all(123)
    b = g2.standard_normal(5)
    assert np.array_equal(a, b)


def test_banditlog_validation():
    ctx = np.zeros((3, 2))
    act = np.array([0, 1, 0])
    rew = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        BanditLog(ctx, act[:-1], rew, np.full(3, 0.5))
    with pytest.raises(ValueError):
        BanditLog(ctx, act, rew, np.array([0.5, 0.5, 0.0]))  # zero propensity


def test_error_codes():
    assert OPEError("x").code == "E400"
    assert BanditForgeError("x").code == "E000"
