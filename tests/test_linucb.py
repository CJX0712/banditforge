"""Tests for LinUCB and the context-free baselines."""

import numpy as np

from bandit.baselines import RandomCF
from bandit.environment import LinearBandit
from bandit.linucb import LinUCB


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


def test_linucb_learns_better_than_random():
    lin = _run(lambda rng: LinUCB(10, 5, 1.0))
    rnd = _run(lambda rng: RandomCF(5, rng))
    assert lin < rnd


def test_linucb_act_returns_valid_arm():
    algo = LinUCB(10, 5, 1.0)
    env = LinearBandit(10, 5, 0.5, seed=3)
    for _ in range(100):
        x = env.sample_context()
        a = algo.act(x)
        assert 0 <= a < 5
        algo.update(x, a, 0.0)


def test_linucb_deterministic():
    r1 = _run(lambda rng: LinUCB(10, 5, 1.0))
    r2 = _run(lambda rng: LinUCB(10, 5, 1.0))
    assert r1 == r2


def test_randomcf_is_context_free_bad():
    # Random should accumulate large regret on a contextual problem
    env = LinearBandit(10, 5, 0.5, seed=11)
    rng = np.random.default_rng(11)
    algo = RandomCF(5, rng)
    total = 0.0
    for _ in range(2000):
        x = env.sample_context()
        a = algo.act(x)
        algo.update(x, a, env.reward(x, a))
        total += env.optimal_reward(x) - env._mean(x, a)
    assert total > 1000  # ignores context -> near-max regret
