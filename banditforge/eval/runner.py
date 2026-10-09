"""Benchmark runners: on-policy simulation + offline OPE evaluation.

Determinism: every random stream derives from explicit integer seeds;
ground-truth MC uses an independent fixed seed inside the data layer.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..bandits.base import POLICY_REGISTRY, BasePolicy
from ..core.types import (
    BanditLog,
    OPEBenchmarkRow,
    OPEEstimate,
    PolicyBenchmarkRow,
    PolicyRunResult,
)
from ..data.synthetic import EvalPolicyVectorized, LinearBanditWorld
from ..ope.estimators import build_ope_estimators
from .metrics import bias, rmse


def simulate(world: LinearBanditWorld, policy: BasePolicy, t_steps: int,
             rng: np.random.Generator, noise: float = 0.5) -> PolicyRunResult:
    """Run a policy online; regret uses expected rewards (noise-free), the
    standard pseudo-regret convention."""
    cum_regret = np.empty(t_steps)
    rewards = np.empty(t_steps)
    actions = np.empty(t_steps, dtype=np.int64)
    best_exp = np.empty(t_steps)
    acc = 0.0
    for t in range(t_steps):
        x = world.sample_context(rng)
        a = policy.select_arm(x)
        exp_r = world.expected_reward(x, a)
        r = exp_r + noise * rng.standard_normal()
        best = world.best_expected_reward(x)
        acc += best - exp_r
        cum_regret[t] = acc
        rewards[t] = r
        actions[t] = a
        best_exp[t] = best
        policy.update(x, a, r)
    return PolicyRunResult(cumulative_regret=cum_regret, realized_rewards=rewards,
                           chosen_actions=actions, best_expected_reward=best_exp)


def calibrate_policy(policy_name: str, grid: List[dict], world: LinearBanditWorld,
                     t_steps: int, seed: int, noise: float = 0.5) -> dict:
    """Pick hyperparameters on an independent calibration world/run.

    The calibration world has a DIFFERENT seed from the benchmark world, so
    benchmark results are not hyperparameter-overfitted.
    """
    best_params, best_regret = None, np.inf
    for params in grid:
        policy = POLICY_REGISTRY[policy_name](world.K, world.d,
                                              rng=np.random.default_rng(seed), **params)
        res = simulate(world, policy, t_steps, np.random.default_rng(seed + 1), noise)
        if res.final_regret < best_regret:
            best_regret = res.final_regret
            best_params = params
    return {"params": best_params, "calib_regret": float(best_regret)}


def policy_learning_benchmark(world: LinearBanditWorld, configs: Dict[str, dict],
                              t_steps: int, seeds: List[int],
                              noise: float = 0.5) -> List[PolicyBenchmarkRow]:
    """Run each policy at its calibrated params across seeds; report mean±std."""
    rows = []
    for name, cfg in configs.items():
        finals = []
        for sd in seeds:
            policy = POLICY_REGISTRY[name](world.K, world.d,
                                           rng=np.random.default_rng(sd * 7 + 13),
                                           **cfg["params"])
            res = simulate(world, policy, t_steps, np.random.default_rng(sd), noise)
            finals.append(res.final_regret)
        arr = np.asarray(finals)
        rows.append(PolicyBenchmarkRow(policy=name, regret_mean=float(arr.mean()),
                                       regret_std=float(arr.std(ddof=1)),
                                       seed_values=[float(v) for v in finals]))
    return rows


def _log_seed(seed: int, spec_idx: int, rep: int) -> int:
    """Deterministic, collision-free log seed derivation."""
    return (seed * 1_000_003 + spec_idx * 10_007 + rep * 977) % (2**31 - 1)


def ope_benchmark_spec(spec: dict, seeds: List[int], n_repeat: int,
                       estimator_names: List[str] = None,
                       ablation: str = None) -> Tuple[List[OPEBenchmarkRow], dict]:
    """Evaluate all OPE estimators on one dataset spec.

    For each seed we draw ``n_repeat`` independent logs, estimate the fixed
    evaluation policy value with each estimator, and score RMSE against the
    spec's Monte-Carlo ground truth. Aggregation: per-seed RMSE over repeats,
    then mean±std across seeds.
    """
    world: LinearBanditWorld = spec["world"]
    truth: float = spec["true_value"]
    evalp = EvalPolicyVectorized(world)
    rows_out: Dict[str, List[float]] = {}
    biases: Dict[str, List[float]] = {}
    last_estimates: Dict[str, OPEEstimate] = {}
    for si, sd in enumerate(seeds):
        per_est_seed: Dict[str, List[float]] = {}
        for rep in range(n_repeat):
            log: BanditLog = world.gen_log(spec["n_log"], _log_seed(sd, si, rep),
                                           spec["logging_temp"], spec["reward_noise"])
            pe_full = evalp.propensities_batch(log.contexts)
            pe_chosen = pe_full[np.arange(log.n), log.actions]
            ests = build_ope_estimators(pe_chosen, pe_full, world.K, world.d)
            for est in ests:
                if ablation is not None and hasattr(est, "name"):
                    _apply_ablation(est, ablation)
                out = est.estimate(log)
                per_est_seed.setdefault(est.name, []).append(out.value)
                if rep == n_repeat - 1:
                    last_estimates[est.name] = out
        for name, vals in per_est_seed.items():
            rows_out.setdefault(name, []).append(rmse(vals, truth))
            biases.setdefault(name, []).append(bias(vals, truth))
    rows = []
    for name, seed_rmses in rows_out.items():
        arr = np.asarray(seed_rmses)
        rows.append(OPEBenchmarkRow(
            estimator=name, dataset=spec["name"],
            rmse_mean=float(arr.mean()), rmse_std=float(arr.std(ddof=1)),
            bias_mean=float(np.mean(biases[name])),
            seed_values=[float(v) for v in seed_rmses]))
    return rows, {"last_estimates": {k: v.to_json_dict() for k, v in last_estimates.items()}}


def _apply_ablation(est, ablation: str) -> None:
    """Apply one ablation switch to an estimator instance (no-op otherwise)."""
    if ablation == "no_cross_fit" and hasattr(est, "use_cross_fit"):
        est.use_cross_fit = False
    elif ablation == "no_adaptive_clip" and hasattr(est, "tau_grid"):
        est.tau_grid = (float("inf"),)
    elif ablation == "no_gating" and hasattr(est, "use_gating"):
        est.use_gating = False
        est.use_cfdrac = False  # gating off -> falls back to fixed SNIPS
