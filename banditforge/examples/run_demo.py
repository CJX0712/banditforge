"""End-to-end demo: full benchmark -> benchmark.json (deterministic).

Run twice: core metrics must be bit-identical (elapsed_sec differs only).

Author: 晨星 (CJX0712)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from banditforge.core.config import EnvConfig
from banditforge.pipeline.pipeline import BanditForgePipeline


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cfg = EnvConfig.from_env()
    report = BanditForgePipeline(cfg).run()
    data = report.to_json_dict()
    out = Path(__file__).resolve().parents[2] / "benchmark.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== Datasets ==")
    for s in data["datasets"]:
        print(f"  {s['name']}: overlap={s['overlap']}, n_log={s['n_log']}, "
              f"true_value={s['true_value']}")
    print("\n== Policy learning (cumulative regret, lower is better) ==")
    for r in sorted(data["policy_benchmark"], key=lambda r: r["regret_mean"]):
        print(f"  {r['policy']:<12} regret={r['regret_mean']:>12.1f} ± {r['regret_std']:.1f}")
    print("\n== OPE benchmark (RMSE, lower is better) ==")
    for ds in sorted({r["dataset"] for r in data["ope_benchmark"]}):
        print(f"  [{ds}]")
        for r in sorted((r for r in data["ope_benchmark"] if r["dataset"] == ds),
                        key=lambda r: r["rmse_mean"]):
            print(f"    {r['estimator']:<22} rmse={r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}"
                  f"  bias={r['bias_mean']:+.4f}")
    print("\n== Gates ==")
    all_ok = True
    for g in data["gates"]:
        mark = "[OK]" if g["passed"] else "[FAIL]"
        all_ok &= g["passed"]
        print(f"  {mark} {g['name']}: {g['detail']}")
    print(f"\nbenchmark.json -> {out} (elapsed {data['elapsed_sec']}s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
