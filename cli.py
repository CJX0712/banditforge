"""cli.py — BanditForge command line interface.

Sub-commands
  benchmark   run the full comparison and optionally dump JSON
  simulate    run a single algorithm and report its regret
  selftest    execute the pytest suite

Determinism: every run is seeded; re-running ``benchmark`` with the same
configuration yields bit-identical regret trajectories.
"""

import argparse
import json
import sys
from typing import Any

from core.config import load_config
from pipeline.pipeline import BanditForgePipeline


def _print_table(report) -> None:
    print("\n=== BanditForge Benchmark (lower cumulative regret is better) ===")
    print(f"{'rank':>4} {'method':16s} {'mean_regret':>12s} {'std':>8s} {'sem':>8s}")
    for i, name in enumerate(report.ranking, 1):
        s = report.per_method[name]
        print(f"{i:>4} {name:16s} {s.regret_mean:12.2f} {s.regret_std:8.2f} {s.regret_sem:8.2f}")
    fw = report.flagships_win
    print("\n--- Flagship (BanditFuse) win checks ---")
    print(
        f"  A) vs UCB1 (context-free SOTA) regret reduction = {fw['A_vs_UCB1_reduction'] * 100:.1f}% "
        f"({'PASS' if fw['A_vs_UCB1_pass'] else 'FAIL'} >=90%, significant={fw['A_vs_UCB1_significant']})"
    )
    print(
        f"  B) vs LinUCB(alpha=2.0) regret reduction       = {fw['B_vs_LinUCB2_reduction'] * 100:.1f}% "
        f"({'PASS' if fw['B_vs_LinUCB2_pass'] else 'FAIL'} >=5%, significant={fw['B_vs_LinUCB2_significant']})"
    )
    print(
        f"  beats LinTS (peer SOTA) reduction              = {fw['beats_LinTS_reduction'] * 100:.1f}% "
        f"({'PASS' if fw['beats_LinTS_pass'] else 'FAIL'} >=5%, significant={fw['beats_LinTS_significant']})"
    )
    print(
        f"  matches best-tuned peer ({fw['matches_best_peer']})    = {fw['matches_best_peer_reduction'] * 100:.1f}% "
        f"(significant={fw['matches_best_peer_significant']})"
    )
    ab = report.ablation
    print("\n--- Ablation / calibration ---")
    print(
        f"  ambiguity contribution   = {ab['ambiguity_contribution'] * 100:+.1f}% (full vs no-ambiguity BanditFuse)"
    )
    print(
        f"  calibrated var_est final = {ab['calibration_var_est_final']:.4f} (true sigma^2={ab['true_sigma_sq']:.4f})"
    )
    det = getattr(report, "determinism_ok", None)
    print(f"  determinism (bit-identical re-run) = {det}")


def cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(
        n_seeds=args.seeds,
        n_rounds=args.rounds,
        regime=args.regime,
        d=args.d,
        k=args.k,
        sigma=args.sigma,
    )
    pipe = BanditForgePipeline(cfg)
    report = pipe.run()
    data = pipe.to_dict(report)
    _print_table(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote JSON -> {args.out}")
    return data


def cmd_simulate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(
        n_seeds=args.seeds,
        n_rounds=args.rounds,
        regime=args.regime,
        d=args.d,
        k=args.k,
        sigma=args.sigma,
    )
    rng = __import__("numpy").random.default_rng(args.seed)
    from bandit.environment import LinearBandit
    from bandit.fuse import BanditFuse

    env = LinearBandit(cfg.d, cfg.k, cfg.sigma, regime=cfg.regime, seed=args.seed)
    algo = BanditFuse(cfg.d, cfg.k, rng, z=cfg.fuse_z, init_var=cfg.fuse_init_var, amb=cfg.fuse_amb)
    total = 0.0
    for _ in range(cfg.n_rounds):
        x = env.sample_context()
        a = algo.act(x)
        r = env.reward(x, a)
        algo.update(x, a, r)
        total += env.optimal_reward(x) - env._mean(x, a)
    print(
        f"method=BanditFuse seed={args.seed} cumulative_regret={total:.2f} "
        f"final_var_est={algo.var_est:.4f} ambiguous_rounds={algo.n_ambiguous}"
    )
    return {"cumulative_regret": total, "final_var_est": algo.var_est}


def cmd_selftest(args: argparse.Namespace) -> None:
    import pytest

    rc = pytest.main(["-q", "-W", "ignore::UserWarning", "tests"])
    sys.exit(rc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="banditforge", description="BanditForge · contextual bandit toolkit"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="run full comparison")
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--rounds", type=int, default=3000)
    b.add_argument("--regime", choices=["linear", "quadratic"], default="linear")
    b.add_argument("--d", type=int, default=10)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--sigma", type=float, default=0.5)
    b.add_argument("--out", type=str, default=None, help="write JSON report to path")
    b.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("simulate", help="single BanditFuse run")
    s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--seeds", type=int, default=1)
    s.add_argument("--rounds", type=int, default=3000)
    s.add_argument("--regime", choices=["linear", "quadratic"], default="linear")
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--sigma", type=float, default=0.5)
    s.set_defaults(func=cmd_simulate)

    t = sub.add_parser("selftest", help="run pytest suite")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
