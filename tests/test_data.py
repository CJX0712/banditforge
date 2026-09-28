"""Data layer tests: synthetic world ground truth + loading roundtrip."""
import numpy as np
import pytest

from banditforge.core.errors import DataError
from banditforge.data.loading import load_log, save_log
from banditforge.data.synthetic import LinearBanditWorld, build_specs


def test_world_shapes_and_determinism():
    w1 = LinearBanditWorld(5, 10, seed=7)
    w2 = LinearBanditWorld(5, 10, seed=7)
    assert np.array_equal(w1.beta, w2.beta)
    assert w1.beta.shape == (5, 10)


def test_logging_propensities_valid():
    w = LinearBanditWorld(5, 10, seed=7)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = w.sample_context(rng)
        p = w.logging_propensities(x, temp=2.0)
        assert abs(p.sum() - 1.0) < 1e-9
        assert np.all(p > 0)


def test_gen_log_deterministic():
    w = LinearBanditWorld(5, 10, seed=7)
    l1 = w.gen_log(200, seed=1, temp=2.0)
    l2 = w.gen_log(200, seed=1, temp=2.0)
    assert np.array_equal(l1.contexts, l2.contexts)
    assert np.array_equal(l1.rewards, l2.rewards)
    assert np.array_equal(l1.propensities, l2.propensities)


def test_true_eval_value_monotone_eps():
    w = LinearBanditWorld(5, 10, seed=7)
    v_small = w.true_eval_value(20000, eps=0.0)
    v_large = w.true_eval_value(20000, eps=0.4)
    assert v_small >= v_large - 0.05  # more uniform mixing lowers the value


def test_mc_truth_matches_bruteforce_small():
    """Invariant: MC truth matches an independent brute-force integral."""
    w = LinearBanditWorld(3, 2, seed=3)
    rng = np.random.default_rng(555)  # independent seed, NOT the MC seed
    xs = rng.standard_normal((40000, 2))
    vals = xs @ w.beta.T + w.C_INTER * (xs[:, 0] * xs[:, 1])[:, None]
    per_ctx = 0.95 * vals.max(axis=1) + 0.05 * vals.mean(axis=1)
    brute = float(per_ctx.mean())
    mc = w.true_eval_value(40000, eps=0.05)
    assert abs(brute - mc) < 0.05  # max-statistics MC converges slowly; 1-sigma ~0.02


def test_build_specs():
    specs = build_specs(n_mc=20000)
    assert len(specs) == 4
    names = [s["name"] for s in specs]
    assert any(n.endswith("low") for n in names)
    assert all(s["true_value"] > 0 for s in specs)


def test_load_save_roundtrip(tmp_path):
    w = LinearBanditWorld(4, 6, seed=5)
    log = w.gen_log(100, seed=2, temp=1.5)
    p = tmp_path / "log.npz"
    save_log(log, str(p))
    log2 = load_log(str(p))
    assert np.array_equal(log.contexts, log2.contexts)
    assert np.array_equal(log.rewards, log2.rewards)


def test_load_missing_file(tmp_path):
    with pytest.raises(DataError):
        load_log(str(tmp_path / "nope.npz"))
