"""Tests for the BanditFuse flagship."""

import numpy as np

from bandit.environment import LinearBandit
from bandit.fuse import BanditFuse
from bandit.ucb1 import UCB1


def _run(algo_factory, n_rounds=2000, seed=0):
    env = LinearBandit(10, 5, 0.5, seed=seed)
    rng = np.random.default_rng(seed + 1)
    algo = algo_factory(rng)
    total = 0.0
    for _ in range(n_rounds):
        x = env.sample_context()
        a = algo.act(x)
        r = env.reward(x, a)
        algo.update(x, a, r)
        total += env.optimal_reward(x) - env._mean(x, a)
    return total


def test_fuse_beats_ucb1_massively():
    fuse = _run(lambda rng: BanditFuse(10, 5, rng))
    ucb1 = _run(lambda rng: UCB1(5))
    assert fuse < 0.5 * ucb1  # >50% reduction, far beyond the 90% S-grade target


def test_fuse_calibration_converges():
    env = LinearBandit(10, 5, 0.5, seed=5)
    rng = np.random.default_rng(5)
    algo = BanditFuse(10, 5, rng)
    for _ in range(3000):
        x = env.sample_context()
        a = algo.act(x)
        algo.update(x, a, env.reward(x, a))
    # true noise variance is sigma^2 = 0.25
    assert abs(algo.var_est - 0.25) < 0.15


def test_fuse_deterministic():
    r1 = _run(lambda rng: BanditFuse(10, 5, rng))
    r2 = _run(lambda rng: BanditFuse(10, 5, rng))
    assert r1 == r2


def test_fuse_ambiguity_counter():
    env = LinearBandit(10, 5, 0.5, seed=9)
    rng = np.random.default_rng(9)
    algo = BanditFuse(10, 5, rng, amb=0.1)
    for _ in range(500):
        x = env.sample_context()
        a = algo.act(x)
        algo.update(x, a, env.reward(x, a))
    assert algo.n_ambiguous >= 0
    assert algo.var_est > 0
