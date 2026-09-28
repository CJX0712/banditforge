"""Pipeline + HPO + determinism tests (small-scale, fast)."""
import json

import numpy as np
import pytest

from banditforge.core.config import EnvConfig
from banditforge.core.types import OPEBenchmarkRow, PolicyBenchmarkRow
from banditforge.data.synthetic import LinearBanditWorld
from banditforge.hpo.tune import tune_by_dr, verify_selected_alpha
from banditforge.pipeline.pipeline import BanditForgePipeline

SMALL_CFG = {"seed": 42, "n_log": 400, "n_mc": 20000, "t_steps": 600,
             "n_seeds": 3, "n_repeat": 6, "hpo_trials": 6}


@pytest.fixture(scope="module")
def small_report():
    pipe = BanditForgePipeline(EnvConfig(**SMALL_CFG))
    return pipe.run()


def test_report_structure(small_report):
    data = small_report.to_json_dict()
    assert data["version"] == "0.1.0"
    assert len(data["datasets"]) == 4
    assert {r["policy"] for r in data["policy_benchmark"]} == \
        {"eps_greedy", "linucb", "lints", "ucb1"}
    est_names = {r["estimator"] for r in data["ope_benchmark"]}
    assert {"ips", "snips", "dm", "dr", "cfdrac", "gated"} <= est_names


def test_regret_positive_and_finite(small_report):
    for r in small_report.policy_benchmark:
        assert np.isfinite(r.regret_mean) and r.regret_mean > 0
        assert len(r.seed_values) == 3


def test_p1_gate_holds_small(small_report):
    g = {x.name: x for x in small_report.gates}["P1_policy_regret_reduction"]
    assert g.passed, g.detail


def test_p2_gate_holds_small(small_report):
    g = {x.name: x for x in small_report.gates}["P2_cfdrac_beats_fixed_dr"]
    assert g.passed, g.detail


def test_p3_gate_holds_small(small_report):
    g = {x.name: x for x in small_report.gates}["P3_cfdrac_vs_baselines"]
    assert g.passed, g.detail


def test_ablation_has_all_switches(small_report):
    names = {r.estimator for r in small_report.ablation}
    for switch in ("no_cross_fit", "no_adaptive_clip", "no_gating"):
        assert any(switch in n for n in names), f"missing ablation {switch}"


def test_failure_cases_derived(small_report):
    assert len(small_report.failure_cases) >= 3
    for fc in small_report.failure_cases:
        assert {"case", "dataset", "evidence", "diagnosis"} <= set(fc.keys())


def test_hpo_not_degenerate(small_report):
    hpo = small_report.hpo
    assert hpo["distinct_objectives"] >= 2, "HPO objective is flat - selection degenerate"
    assert hpo["best_alpha"] in (0.1, 0.3, 0.5, 1.0, 2.0, 4.0)
    assert hpo["verification"]["value_best_alpha"] >= \
        hpo["verification"]["value_base_alpha"]


def test_determinism_bitwise():
    """Invariant: same seed -> identical report except elapsed_sec."""
    cfg = EnvConfig(seed=7, n_log=300, n_mc=20000, t_steps=500,
                    n_seeds=3, n_repeat=4, hpo_trials=6)
    d1 = BanditForgePipeline(cfg).run().to_json_dict()
    d2 = BanditForgePipeline(cfg).run().to_json_dict()
    d1.pop("elapsed_sec")
    d2.pop("elapsed_sec")
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_json_serializable(tmp_path, small_report):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(small_report.to_json_dict(), ensure_ascii=False),
                 encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["seed"] == 42


def test_row_types():
    row = PolicyBenchmarkRow(policy="x", regret_mean=1, regret_std=0)
    assert row.to_json_dict()["policy"] == "x"
    ope_row = OPEBenchmarkRow(estimator="e", dataset="d", rmse_mean=1,
                              rmse_std=0, bias_mean=0)
    assert ope_row.to_json_dict()["rmse_mean"] == 1


def test_tune_and_verify():
    world = LinearBanditWorld(4, 8, seed=7)
    log = world.gen_log(500, seed=11, temp=2.0)
    sel = tune_by_dr(log, world)
    assert sel["distinct_objectives"] >= 2
    ver = verify_selected_alpha(world, sel["best_alpha"])
    assert ver["value_best_alpha"] >= ver["value_base_alpha"]
