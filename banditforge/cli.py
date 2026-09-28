"""BanditForge CLI: benchmark / demo / diagnose subcommands.

Windows console safety: force utf-8 stdout to avoid GBK encode crashes.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _force_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def _print_table(headers: list, rows: list) -> None:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(f"{c!s:<{w}}" for c, w in zip(r, widths, strict=True)))


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .core.config import EnvConfig
    from .pipeline.pipeline import BanditForgePipeline

    cfg = EnvConfig.from_env()
    pipe = BanditForgePipeline(cfg)
    report = pipe.run()
    data = report.to_json_dict()
    out = Path(args.output)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== Policy learning (cumulative regret, lower is better) ==")
    _print_table(["policy", "regret_mean", "regret_std"],
                 [[r["policy"], r["regret_mean"], r["regret_std"]]
                  for r in data["policy_benchmark"]])
    print("\n== OPE benchmark (RMSE vs Monte-Carlo truth, lower is better) ==")
    for ds in sorted({r["dataset"] for r in data["ope_benchmark"]}):
        rows = sorted([r for r in data["ope_benchmark"] if r["dataset"] == ds],
                      key=lambda r: r["rmse_mean"])
        _print_table(["estimator", "rmse_mean", "rmse_std", "bias_mean"],
                     [[r["estimator"], r["rmse_mean"], r["rmse_std"], r["bias_mean"]]
                      for r in rows])
        print()
    print("== Gates ==")
    for g in data["gates"]:
        mark = "[OK]" if g["passed"] else "[FAIL]"
        print(f"{mark} {g['name']}: {g['detail']}")
    print(f"\nbenchmark.json written to {out} (elapsed {data['elapsed_sec']}s)")
    return 0 if all(g["passed"] for g in data["gates"]) else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    import numpy as np  # local import keeps CLI startup light

    from .data.loading import load_log
    from .data.synthetic import LinearBanditWorld
    from .ope.diagnostics import overlap_report

    log = load_log(args.log)
    _ = LinearBanditWorld(log.num_actions(), log.context_dim(), seed=args.seed)
    # diagnose with a uniform-eval reference policy (works for arbitrary logs)
    pe_chosen = np.full(log.n, 1.0 / log.num_actions())
    rep = overlap_report(log, pe_chosen)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Quick smoke: tiny run end-to-end, prints gates only."""
    from .core.config import EnvConfig
    from .pipeline.pipeline import BanditForgePipeline

    cfg = EnvConfig(seed=42, n_log=400, n_mc=20000, t_steps=600,
                    n_seeds=3, n_repeat=6, hpo_trials=6)
    report = BanditForgePipeline(cfg).run()
    data = report.to_json_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    for g in data["gates"]:
        print(("[OK] " if g["passed"] else "[FAIL] ") + g["name"] + ": " + g["detail"])
    return 0 if all(g["passed"] for g in data["gates"]) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="banditforge",
                                description="BanditForge: contextual bandit + OPE")
    p.add_argument("--version", action="version", version=f"banditforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="full benchmark -> benchmark.json")
    b.add_argument("--output", default="benchmark.json")
    b.set_defaults(func=cmd_benchmark)

    d = sub.add_parser("diagnose", help="overlap diagnostics for a log file")
    d.add_argument("--log", required=True)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--uniform-eval", action="store_true")
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("demo", help="fast end-to-end smoke run")
    m.add_argument("--output", default="")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv: list = None) -> int:
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
