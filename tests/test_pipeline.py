"""Pipeline integration tests (small configs for speed)."""

import json

from core.config import load_config
from pipeline.pipeline import BanditForgePipeline


def test_pipeline_linear_small():
    cfg = load_config(n_seeds=3, n_rounds=500, regime="linear")
    pipe = BanditForgePipeline(cfg)
    report = pipe.run()
    assert report.determinism_ok is True
    # Headline S-grade gate A: >90% regret reduction vs context-free SOTA (UCB1)
    assert report.flagships_win["A_vs_UCB1_pass"] is True
    # Practical gate B: >5% reduction vs a realistic fixed-alpha LinUCB
    assert report.flagships_win["B_vs_LinUCB2_pass"] is True
    fuse_mean = report.per_method["BanditFuse"].regret_mean
    ucb1_mean = report.per_method["UCB1"].regret_mean
    assert fuse_mean < ucb1_mean
    # JSON serializable
    d = pipe.to_dict(report)
    json.dumps(d)
    # calibration sane
    assert 0.0 < report.ablation["calibration_var_est_final"] < 1.5


def test_pipeline_quadratic_runs():
    cfg = load_config(n_seeds=2, n_rounds=200, regime="quadratic")
    pipe = BanditForgePipeline(cfg)
    report = pipe.run()
    assert report.determinism_ok is True
    assert "BanditFuse" in report.per_method


def test_ranking_has_flagship_above_contextfree():
    cfg = load_config(n_seeds=3, n_rounds=500)
    pipe = BanditForgePipeline(cfg)
    report = pipe.run()
    fuse_rank = report.ranking.index("BanditFuse")
    ucb1_rank = report.ranking.index("UCB1")
    assert fuse_rank < ucb1_rank  # lower regret = better rank
