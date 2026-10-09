"""examples/run_demo.py — end-to-end BanditForge demonstration.

Runs the full benchmark, verifies determinism (bit-identical re-runs) and noise
calibration, prints a summary table, and persists ``benchmark.json`` (excluded
from git via ``.gitignore``). Re-running produces byte-identical metrics.
"""

from __future__ import annotations

import json
import os
import sys

# allow running directly: banditforge/ is the import root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.config import load_config  # noqa: E402
from pipeline.pipeline import BanditForgePipeline  # noqa: E402


def main() -> dict:
    cfg = load_config()
    pipe = BanditForgePipeline(cfg)
    report = pipe.run()
    data = pipe.to_dict(report)

    out_path = os.path.join(os.path.dirname(__file__), "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print("=== BanditForge demo ===")
    print(
        f"config: d={cfg.d} k={cfg.k} sigma={cfg.sigma} T={cfg.n_rounds} "
        f"regime={cfg.regime} seeds={cfg.n_seeds}"
    )
    print(f"\n{'rank':>4} {'method':16s} {'mean_regret':>12s} {'std':>8s}")
    for i, name in enumerate(report.ranking, 1):
        s = report.per_method[name]
        print(f"{i:>4} {name:16s} {s.regret_mean:12.2f} {s.regret_std:8.2f}")

    fw = report.flagships_win
    print("\nFlagship BanditFuse win checks:")
    print(
        f"  A) vs UCB1 (context-free SOTA): reduction={fw['A_vs_UCB1_reduction'] * 100:.1f}% "
        f"pass={fw['A_vs_UCB1_pass']} significant={fw['A_vs_UCB1_significant']}"
    )
    print(
        f"  B) vs LinUCB(alpha=2.0):       reduction={fw['B_vs_LinUCB2_reduction'] * 100:.1f}% "
        f"pass={fw['B_vs_LinUCB2_pass']} significant={fw['B_vs_LinUCB2_significant']}"
    )
    print(
        f"  beats LinTS (peer SOTA):       reduction={fw['beats_LinTS_reduction'] * 100:.1f}% "
        f"pass={fw['beats_LinTS_pass']}"
    )
    print(
        f"  matches best-tuned peer ({fw['matches_best_peer']}): reduction="
        f"{fw['matches_best_peer_reduction'] * 100:.1f}%"
    )
    ab = report.ablation
    print(
        f"\nAblation: ambiguity contribution={ab['ambiguity_contribution'] * 100:+.1f}% | "
        f"calibrated var_est={ab['calibration_var_est_final']:.4f} vs true sigma^2="
        f"{ab['true_sigma_sq']:.4f}"
    )
    print(f"Determinism (bit-identical re-run): {report.determinism_ok}")
    print(f"\nwrote {out_path}")
    return data


if __name__ == "__main__":
    main()
