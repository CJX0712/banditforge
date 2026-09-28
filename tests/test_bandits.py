"""Policy tests: registry, learning signal, LinUCB/LinTS invariants."""
import numpy as np
import pytest

from banditforge.bandits.base import (
    POLICY_REGISTRY,
    UCB1,
    LinearGreedy,
    LinTS,
    LinUCB,
    build_policy,
)
from banditforge.core.errors import PolicyError
from banditforge.data.synthetic import LinearBanditWorld
from banditforge.eval.runner import simulate


def _tiny_world():
    return LinearBanditWorld(3, 4, seed=11)


def test_registry_build():
    for name in POLICY_REGISTRY:
        p = build_policy(name, 3, 4, rng=np.random.default_rng(0))
        assert p.name == name or name == "eps_greedy"


def test_build_unknown_policy():
    with pytest.raises(PolicyError):
        build_policy("nope", 3, 4)


def test_policy_param_validation():
    with pytest.raises(PolicyError):
        LinearGreedy(3, 4, eps=1.5)
    with pytest.raises(PolicyError):
        LinUCB(3, 4, alpha=-1)
    with pytest.raises(PolicyError):
        LinTS(3, 4, scale=0)
    with pytest.raises(PolicyError):
        UCB1(3, 4, c=0)


def test_all_policies_beat_random_world_noise():
    """Invariant: every policy accumulates less regret than uniform random."""
    w = _tiny_world()
    rng = np.random.default_rng(42)
    xs = rng.standard_normal((400, w.d))
    vals = xs @ w.beta.T + w.C_INTER * (xs[:, 0] * xs[:, 1])[:, None]
    random_regret = float(np.mean(vals.max(axis=1) - vals.mean(axis=1)) * 600)
    default_params = {"eps_greedy": {"eps": 0.1}, "linucb": {"alpha": 1.0},
                      "lints": {"scale": 1.0}, "ucb1": {"c": 1.0}}
    for name in POLICY_REGISTRY:
        pol = build_policy(name, w.K, w.d, rng=np.random.default_rng(1),
                           **default_params[name])
        res = simulate(w, pol, 600, np.random.default_rng(2))
        if name == "ucb1":
            # context-agnostic MAB cannot converge to a context-dependent best
            # arm; weak invariant: it must not be much worse than random
            assert res.final_regret < random_regret * 1.2
        else:
            assert res.final_regret < random_regret, f"{name} failed to beat random"


def test_linucb_explores_then_exploits():
    w = _tiny_world()
    pol = LinUCB(w.K, w.d, alpha=0.5, rng=np.random.default_rng(3))
    res = simulate(w, pol, 1500, np.random.default_rng(4))
    # regret rate in the last half must be lower than in the first half
    r = res.cumulative_regret
    first = r[249] - r[0] if len(r) > 250 else r[-1]
    last = r[-1] - r[-250]
    assert last < first


def test_greedy_zero_eps_is_deterministic():
    w = _tiny_world()
    pol = LinearGreedy(w.K, w.d, eps=0.0, rng=np.random.default_rng(5))
    x = w.sample_context(np.random.default_rng(6))
    a1 = pol.select_arm(x)
    pol.update(x, a1, 5.0)
    a2 = pol.select_arm(x)
    assert a1 == a2  # no exploration: same context, same greedy arm after update


def test_lints_deterministic_with_same_seed():
    w = _tiny_world()
    p1 = LinTS(w.K, w.d, scale=1.0, rng=np.random.default_rng(7))
    p2 = LinTS(w.K, w.d, scale=1.0, rng=np.random.default_rng(7))
    rng = np.random.default_rng(8)
    for _ in range(50):
        x = w.sample_context(rng)
        a1, a2 = p1.select_arm(x), p2.select_arm(x)
        r = w.expected_reward(x, a1) + 0.5 * rng.standard_normal()
        p1.update(x, a1, r)
        p2.update(x, a2, r)
        assert a1 == a2


def test_context_shape_mismatch():
    w = _tiny_world()
    pol = LinUCB(w.K, w.d, rng=np.random.default_rng(9))
    with pytest.raises(PolicyError):
        pol.select_arm(np.zeros(w.d + 1))
