"""Tests for the synthetic contextual bandit environment."""

import numpy as np

from bandit.environment import LinearBandit
from core.errors import DataError


def test_deterministic_theta_and_contexts():
    e1 = LinearBandit(10, 5, 0.5, seed=42)
    e2 = LinearBandit(10, 5, 0.5, seed=42)
    assert np.array_equal(e1.Theta, e2.Theta)
    c1 = [e1.sample_context() for _ in range(20)]
    c2 = [e2.sample_context() for _ in range(20)]
    assert all(np.array_equal(a, b) for a, b in zip(c1, c2, strict=True))


def test_optimal_arm_is_bruteforce():
    e = LinearBandit(10, 5, 0.5, seed=7)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.standard_normal(10)
        best = max(range(5), key=lambda a: e._mean(x, a))
        assert e.optimal_arm(x) == best


def test_quadratic_regime_runs():
    e = LinearBandit(10, 5, 0.5, regime="quadratic", seed=1)
    x = e.sample_context()
    a = e.optimal_arm(x)
    r = e.reward(x, a)
    assert np.isfinite(r)
    assert e.optimal_reward(x) >= e._mean(x, a) - 1e-9


def test_invalid_params_raise():
    import pytest

    with pytest.raises(DataError):
        LinearBandit(10, 1, 0.5)
    with pytest.raises(DataError):
        LinearBandit(10, 5, 0.0)
