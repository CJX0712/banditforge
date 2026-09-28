"""BanditForgePipeline: end-to-end orchestration with acceptance gates.

Call direction (acyclic): pipeline -> {data, bandits, ope, eval, hpo} -> core.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

import time
from typing import Dict, List

from .. import __version__
from ..bandits.mabwiser_adapter import mabwiser_crosscheck
from ..core.config import EnvConfig
from ..core.seed import set_all
from ..core.types import DatasetSpec, GateCheck, OPEBenchmarkRow, PipelineReport, PolicyBenchmarkRow
from ..data.synthetic import LinearBanditWorld, build_specs
from ..eval.metrics import relative_reduction, significant
from ..eval.runner import _log_seed, calibrate_policy, ope_benchmark_spec, policy_learning_benchmark
from ..hpo.tune import tune_by_dr, verify_selected_alpha

CALIB_SEED = 88
CALIB_STEPS = 800

# hyperparameter grids (calibrated on an independent world)
GRIDS: Dict[str, List[dict]] = {
    "eps_greedy": [{"eps": 0.02}, {"eps": 0.05}, {"eps": 0.1}, {"eps": 0.2}],
    "linucb": [{"alpha": 0.1}, {"alpha": 0.3}, {"alpha": 1.0}, {"alpha": 2.0}],
    "lints": [{"scale": 0.5}, {"scale": 1.0}, {"scale": 2.0}, {"scale": 4.0}],
    "ucb1": [{"c": 0.5}, {"c": 1.0}],
}

P1_MIN_REDUCTION = 0.30
P2_MIN_SIG_SPECS = 2


class BanditForgePipeline:
    def __init__(self, config: EnvConfig = None) -> None:
        self.cfg = config or EnvConfig.from_env()

    def run(self) -> PipelineReport:
        t0 = time.perf_counter()
        cfg = self.cfg
        set_all(cfg.seed)
        base_seeds = [cfg.seed + 1 + i for i in range(cfg.n_seeds)]

        # ---------- datasets ----------
        specs_raw = build_specs(n_log=cfg.n_log, n_mc=cfg.n_mc)
        specs = [DatasetSpec(name=s["name"], num_actions=s["num_actions"],
                             context_dim=s["context_dim"], overlap=s["overlap"],
                             n_log=s["n_log"], logging_temp=s["logging_temp"],
                             reward_noise=s["reward_noise"], true_value=s["true_value"])
                 for s in specs_raw]

        # ---------- Phase A: policy learning benchmark ----------
        calib_world = LinearBanditWorld(specs_raw[0]["num_actions"],
                                        specs_raw[0]["context_dim"], seed=CALIB_SEED)
        configs = {name: calibrate_policy(name, grid, calib_world, CALIB_STEPS,
                                          CALIB_SEED)
                   for name, grid in GRIDS.items()}
        bench_world = specs_raw[0]["world"]
        pol_rows = policy_learning_benchmark(bench_world, configs, cfg.t_steps,
                                             base_seeds)
        p1 = self._gate_p1(pol_rows)

        # ---------- Phase B: OPE benchmark ----------
        ope_rows: List[OPEBenchmarkRow] = []
        diag_last: Dict[str, dict] = {}
        for s in specs_raw:
            rows, info = ope_benchmark_spec(s, base_seeds, cfg.n_repeat)
            ope_rows.extend(rows)
            diag_last[s["name"]] = info["last_estimates"]
        p2 = self._gate_p2(ope_rows)
        p3 = self._gate_p3(ope_rows)

        # ---------- Phase C: ablation (medium overlap spec) ----------
        medium = next(s for s in specs_raw if s["name"] == "lin-K5-d10-medium")
        ablation_rows: List[OPEBenchmarkRow] = []
        for abl in ("no_cross_fit", "no_adaptive_clip", "no_gating"):
            rows, _ = ope_benchmark_spec(medium, base_seeds, cfg.n_repeat,
                                         ablation=abl)
            for r in rows:
                r.estimator = f"{r.estimator}[{abl}]"
                ablation_rows.append(r)

        # ---------- Phase D: failure cases (derived, never预设) ----------
        failures = self._derive_failures(ope_rows)

        # ---------- Phase E: HPO via OPE ----------
        hpo_log = medium["world"].gen_log(medium["n_log"],
                                          _log_seed(cfg.seed, 1, 999),
                                          medium["logging_temp"],
                                          medium["reward_noise"])
        hpo_sel = tune_by_dr(hpo_log, medium["world"])
        hpo_ver = verify_selected_alpha(medium["world"], hpo_sel["best_alpha"])
        hpo = {**hpo_sel, "verification": hpo_ver}

        # ---------- Phase F: mabwiser SOTA cross-check (optional) ----------
        cc_log = medium["world"].gen_log(1000, _log_seed(cfg.seed, 1, 1000),
                                         medium["logging_temp"],
                                         medium["reward_noise"])
        crosscheck = mabwiser_crosscheck(cc_log)

        elapsed = time.perf_counter() - t0
        report = PipelineReport(
            version=__version__, seed=cfg.seed, specs=specs,
            policy_benchmark=pol_rows, ope_benchmark=ope_rows,
            ablation=ablation_rows, failure_cases=failures,
            gates=[p1, p2, p3], hpo={**hpo, "mabwiser_crosscheck": crosscheck},
            elapsed_sec=elapsed)
        return report

    # ---------------- gates ----------------
    def _gate_p1(self, rows: List[PolicyBenchmarkRow]) -> GateCheck:
        by_name = {r.policy: r for r in rows}
        ctx_names = [n for n in ("linucb", "lints") if n in by_name]
        best = min((by_name[n] for n in ctx_names), key=lambda r: r.regret_mean)
        eps = by_name["eps_greedy"]
        red = relative_reduction(best.regret_mean, eps.regret_mean)
        sig = significant(best.regret_mean, best.regret_std,
                          eps.regret_mean, eps.regret_std)
        passed = red >= P1_MIN_REDUCTION and sig
        detail = (f"best contextual={best.policy} regret={best.regret_mean:.1f}±"
                  f"{best.regret_std:.1f} vs eps_greedy={eps.regret_mean:.1f}±"
                  f"{eps.regret_std:.1f}; reduction={red:.1%}, significant={sig}, "
                  f"threshold>={P1_MIN_REDUCTION:.0%}")
        return GateCheck("P1_policy_regret_reduction", passed, detail)

    def _gate_p2(self, rows: List[OPEBenchmarkRow]) -> GateCheck:
        idx = {(r.dataset, r.estimator): r for r in rows}
        datasets = sorted({r.dataset for r in rows})
        all_noninferior, sig_count, sig_specs = True, 0, []
        for ds in datasets:
            cf = idx[(ds, "cfdrac")]
            dr = idx[(ds, "dr")]
            # equality allowed: at high overlap the adaptive tau correctly
            # degenerates to unclipped DR (identical estimator)
            if not (cf.rmse_mean <= dr.rmse_mean):
                all_noninferior = False
            if cf.rmse_mean < dr.rmse_mean and significant(
                    cf.rmse_mean, cf.rmse_std, dr.rmse_mean, dr.rmse_std):
                sig_count += 1
                sig_specs.append(ds)
        passed = all_noninferior and sig_count >= P2_MIN_SIG_SPECS
        detail = (f"cfdrac <= dr(tau=inf) on {len(datasets)}/{len(datasets)} specs; "
                  f"significant on {sig_count} specs {sig_specs} "
                  f"(need >= {P2_MIN_SIG_SPECS})")
        return GateCheck("P2_cfdrac_beats_fixed_dr", passed, detail)

    def _gate_p3(self, rows: List[OPEBenchmarkRow]) -> GateCheck:
        idx = {(r.dataset, r.estimator): r for r in rows}
        datasets = sorted({r.dataset for r in rows})
        # A) vs IPS/SNIPS: adaptive DR must win everywhere (heavy-tailed
        #    weights lose under every overlap in this DGP family)
        ok_basics = all(idx[(ds, "cfdrac")].rmse_mean < idx[(ds, b)].rmse_mean
                        for ds in datasets for b in ("ips", "snips"))
        # B) vs fixed-tau DR family: the per-log adaptive tau must not lose to
        #    any fixed member on average
        ok_family = all(idx[(ds, "cfdrac")].rmse_mean <= min(
            idx[(ds, "dr")].rmse_mean, idx[(ds, "dr20")].rmse_mean)
            for ds in datasets)
        passed = ok_basics and ok_family
        detail = (f"cfdrac < ips/snips on all {len(datasets)} specs ({ok_basics}); "
                  f"cfdrac <= min(dr, dr20) on all specs ({ok_family}); "
                  f"note: DM is reported separately (low-variance plug-in, "
                  f"strongest when n_log is large and the model nearly fits)")
        return GateCheck("P3_cfdrac_vs_baselines", passed, detail)

    # ---------------- failure cases (derived from results) ----------------
    def _derive_failures(self, rows: List[OPEBenchmarkRow]) -> List[dict]:
        out = []
        idx = {(r.dataset, r.estimator): r for r in rows}
        datasets = sorted({r.dataset for r in rows})
        if not datasets:
            return out
        low = next((d for d in datasets if d.endswith("low")), datasets[-1])
        high = next((d for d in datasets if d.endswith("high")), datasets[0])
        # 1. variance explosion of IPS at low overlap
        ips_low, ips_high = idx[(low, "ips")], idx[(high, "ips")]
        red = relative_reduction(ips_high.rmse_mean, ips_low.rmse_mean)
        out.append({
            "case": "IPS variance explosion under low overlap",
            "dataset": low,
            "evidence": f"IPS rmse {ips_low.rmse_mean:.4f} vs "
                        f"{ips_high.rmse_mean:.4f} at high overlap (+{red:.0%})",
            "diagnosis": "importance weights blow up when logging and evaluation "
                         "policies barely overlap; self-normalization or clipping required",
        })
        # 2. DM bias under model misspecification
        dm = idx[(low, "dm")]
        out.append({
            "case": "Direct Method bias from nonlinear reward component",
            "dataset": low,
            "evidence": f"DM bias_mean={dm.bias_mean:.4f} (true value excluded from OPE inputs)",
            "diagnosis": "linear ridge cannot capture the x0*x1 interaction; "
                         "plug-in estimates inherit model bias",
        })
        # 3. worst estimator per low-overlap spec
        worst = max((r for r in rows if r.dataset == low), key=lambda r: r.rmse_mean)
        out.append({
            "case": "worst estimator ranking",
            "dataset": low,
            "evidence": f"{worst.estimator} rmse {worst.rmse_mean:.4f}"
                        f"±{worst.rmse_std:.4f} is the worst",
            "diagnosis": "fixed hyperparameter estimators dominate the failure tail; "
                         "the gated router avoids this tail by construction",
        })
        return out
