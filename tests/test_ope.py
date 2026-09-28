"""OPE tests: estimator invariants, cross-fitting, adaptive clipping, gating."""
import numpy as np
import pytest

from banditforge.core.errors import OPEError
from banditforge.data.synthetic import EvalPolicyVectorized, LinearBanditWorld
from banditforge.ope.diagnostics import overlap_report
from banditforge.ope.estimators import (
    CFDRACEstimator,
    DMEstimator,
    DREstimator,
    GatedOPEEstimator,
    IPSEstimator,
    SNIPSEstimator,
)
from banditforge.ope.reward_model import (
    RidgeRewardModel,
    clip_weights,
    cross_fit_predictions,
    ess_ratio,
)


@pytest.fixture(scope="module")
def medium_setup():
    world = LinearBanditWorld(4, 8, seed=7)
    log = world.gen_log(600, seed=101, temp=2.0)
    evalp = EvalPolicyVectorized(world)
    pe_full = evalp.propensities_batch(log.contexts)
    pe_chosen = pe_full[np.arange(log.n), log.actions]
    return world, log, pe_full, pe_chosen


def test_ips_snips_consistency_high_overlap(medium_setup):
    """Invariant: with near-uniform logging, IPS ~= SNIPS ~= DM within tolerance."""
    world, log, pe_full, pe_chosen = medium_setup
    ips = IPSEstimator(pe_chosen).estimate(log).value
    snips = SNIPSEstimator(pe_chosen).estimate(log).value
    dm = DMEstimator(pe_full, world.K, world.d).estimate(log).value
    truth = world.true_eval_value(20000)
    for v in (ips, snips, dm):
        assert abs(v - truth) < 0.5


def test_ips_equals_weighted_mean(medium_setup):
    _, log, _, pe_chosen = medium_setup
    w = pe_chosen / log.propensities
    expected = float(np.mean(log.rewards * w))
    got = IPSEstimator(pe_chosen).estimate(log).value
    assert abs(got - expected) < 1e-12


def test_dr_unbiased_hard_invariant(medium_setup):
    """Hard invariant: DR/CFDRAC must be near-unbiased when the reward model
    is well-specified up to the small interaction term (bias < 0.1)."""
    world, log, pe_full, _ = medium_setup
    truth = world.true_eval_value(20000)
    dr = DREstimator(pe_full, world.K, world.d).estimate(log).value
    cf = CFDRACEstimator(pe_full, world.K, world.d).estimate(log).value
    assert abs(dr - truth) < 0.10, f"DR bias too large: {dr - truth}"
    assert abs(cf - truth) < 0.10, f"CFDRAC bias too large: {cf - truth}"


def test_clip_weights():
    w = np.array([0.5, 3.0, 10.0])
    assert np.array_equal(clip_weights(w, 3.0), np.array([0.5, 3.0, 3.0]))
    with pytest.raises(OPEError):
        clip_weights(w, 0)


def test_ess_ratio():
    assert abs(ess_ratio(np.ones(100)) - 1.0) < 1e-12
    w = np.zeros(100)
    w[0] = 1.0
    assert abs(ess_ratio(w) - 0.01) < 1e-9


def test_cross_fit_shapes_and_finite(medium_setup):
    _, log, _, _ = medium_setup
    oof, full = cross_fit_predictions(log, 4, 8, n_folds=5)
    assert oof.shape == (log.n,)
    assert np.all(np.isfinite(oof))
    assert isinstance(full, RidgeRewardModel)


def test_cfdrac_selects_tau_and_finite(medium_setup):
    world, log, pe_full, pe_chosen = medium_setup
    est = CFDRACEstimator(pe_full, 4, 8)
    out = est.estimate(log)
    assert np.isfinite(out.value)
    assert out.clip_tau is not None
    assert out.reason.startswith("IRM-selected")


def test_cfdrac_beats_or_matches_dr_low_overlap():
    """Invariant: adaptive clipping must not be worse than fixed tau=inf DR
    on a low-overlap log (variance-dominated regime)."""
    world = LinearBanditWorld(4, 8, seed=7)
    log = world.gen_log(600, seed=202, temp=0.6)
    evalp = EvalPolicyVectorized(world)
    pe_full = evalp.propensities_batch(log.contexts)
    dr_vals = [DREstimator(pe_full, 4, 8).estimate(log).value for _ in range(1)]
    cf = CFDRACEstimator(pe_full, 4, 8).estimate(log).value
    truth = world.true_eval_value(20000)
    err_dr = abs(dr_vals[0] - truth)
    err_cf = abs(cf - truth)
    assert err_cf <= err_dr + 0.02


def test_no_cross_fit_ablation_differs(medium_setup):
    _, log, pe_full, _ = medium_setup
    v_cf = CFDRACEstimator(pe_full, 4, 8, use_cross_fit=True).estimate(log).value
    v_no = CFDRACEstimator(pe_full, 4, 8, use_cross_fit=False).estimate(log).value
    assert np.isfinite(v_cf) and np.isfinite(v_no)
    assert v_cf != v_no  # ablation switch must have an effect


def test_gated_router_reasons(medium_setup):
    _, log, pe_full, pe_chosen = medium_setup
    est = GatedOPEEstimator(pe_chosen, pe_full, 4, 8)
    out = est.estimate(log)
    assert "SNIPS" in out.reason or "CF-DR-AC" in out.reason


def test_gated_low_overlap_routes_to_cfdrac():
    world = LinearBanditWorld(4, 8, seed=7)
    log = world.gen_log(600, seed=303, temp=0.5)
    evalp = EvalPolicyVectorized(world)
    pe_full = evalp.propensities_batch(log.contexts)
    pe_chosen = pe_full[np.arange(log.n), log.actions]
    est = GatedOPEEstimator(pe_chosen, pe_full, 4, 8)
    out = est.estimate(log)
    assert "CF-DR-AC" in out.reason  # low overlap must route to the corrected estimator


def test_estimators_reject_bad_pe(medium_setup):
    _, log, _, _ = medium_setup
    with pytest.raises(OPEError):
        IPSEstimator(np.ones(log.n + 1)).estimate(log)
    with pytest.raises(OPEError):
        IPSEstimator(np.zeros(log.n)).estimate(log)


def test_overlap_report(medium_setup):
    _, log, _, pe_chosen = medium_setup
    rep = overlap_report(log, pe_chosen)
    assert 0 < rep["ess_ratio"] <= 1.0
    assert rep["w_max"] >= rep["w_p99"] >= rep["w_median"]
