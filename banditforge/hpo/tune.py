"""OPE-based hyperparameter selection (Optuna).

Closed loop without any ground-truth values:
1. Fit a per-arm ridge reward model on the offline log.
2. For candidate alpha, define the soft-greedy policy family
   pi_alpha(k|x) = (1-eps) softmax(alpha * mu_hat(k|x)) + eps/K.
3. Score pi_alpha with a DR estimator on the log (oracle-free).
4. Verify the selected alpha against the analytic MC value on an
   independent context sample.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..data.synthetic import MC_SEED, LinearBanditWorld
from ..ope.estimators import DREstimator
from ..ope.reward_model import RidgeRewardModel

ALPHA_GRID = (0.1, 0.3, 0.5, 1.0, 2.0, 4.0)
EPS_EVAL = 0.05


def soft_greedy_propensity(mu_all: np.ndarray, K: int, alpha: float,
                           eps: float = EPS_EVAL) -> np.ndarray:
    """(n, K) propensity matrix of the soft-greedy family at temperature 1/alpha."""
    scores = alpha * mu_all
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=1, keepdims=True)
    return (1.0 - eps) * p + eps / K


def tune_by_dr(log, world: LinearBanditWorld) -> Dict[str, object]:
    """Select alpha for the soft-greedy family by DR on the log (oracle-free)."""
    import optuna

    model = RidgeRewardModel(world.K, world.d).fit(log)
    mu_all = model.predict_all(log.contexts)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.trial.Trial") -> float:
        alpha = trial.suggest_categorical("alpha", list(ALPHA_GRID))
        p_full = soft_greedy_propensity(mu_all, world.K, alpha)
        dr = DREstimator(p_full, world.K, world.d)
        return dr.estimate(log).value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler({"alpha": list(ALPHA_GRID)}),
    )
    study.optimize(objective, n_trials=len(ALPHA_GRID))
    best_alpha = float(study.best_params["alpha"])
    values = {t.params["alpha"]: t.value for t in study.trials if t.value is not None}
    distinct = len({round(v, 9) for v in values.values()})
    return {
        "best_alpha": best_alpha,
        "n_trials": len(ALPHA_GRID),
        "best_dr_value": round(float(study.best_value), 6),
        "all_dr_values": {str(k): round(float(v), 6) for k, v in sorted(values.items())},
        "distinct_objectives": distinct,  # guard against degenerate flat objectives
    }


def verify_selected_alpha(world: LinearBanditWorld, best_alpha: float,
                          base_alpha: float = 0.1, n_mc: int = 50_000) -> Dict[str, float]:
    """Analytic MC value of soft-greedy(best_alpha) vs the weakest candidate.

    Uses an INDEPENDENT MC seed and a fresh context sample; this is the
    ground-truth verification, never used during selection.
    """
    rng = np.random.default_rng(MC_SEED + 1)
    xs = rng.standard_normal((n_mc, world.d))
    true_all = xs @ world.beta.T + world.C_INTER * (xs[:, 0] * xs[:, 1])[:, None]
    mu_hat_all = true_all  # verification evaluates the TRUE value of pi_alpha

    def policy_value(alpha: float) -> float:
        p = soft_greedy_propensity(mu_hat_all, world.K, alpha)
        return float(np.mean(np.sum(p * true_all, axis=1)))

    return {
        "value_best_alpha": round(policy_value(best_alpha), 6),
        "value_base_alpha": round(policy_value(base_alpha), 6),
    }
