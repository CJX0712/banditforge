"""OPE estimators: IPS / SNIPS / DM / DR (fixed tau) + CF-DR-AC (novel) + gating.

Estimator contracts
-------------------
* All estimators return an estimated policy value; higher = better policy.
* All estimators are read-only over the log.
* ``pe_chosen`` (n,) is the evaluation policy's propensity at the logged
  actions; ``pe_full`` (n, K) is the full propensity matrix (DM only).

CF-DR-AC (cross-fitted DR with adaptive clipping)
-------------------------------------------------
1. K-fold cross-fitting yields out-of-fold mu_hat at logged actions, removing
   own-observation overfitting bias in the DR correction term.
2. The clipping threshold tau is selected on an IRM/CRM-style criterion
   (Swaminathan & Joachims 2015) computed from cross-fitted DR pseudo-outcomes:
       tau* = argmax_tau  mean(psi_tau) - std(psi_tau) / sqrt(n)
   which is oracle-free (no ground-truth value needed) and balances the
   variance reduction from clipping against the induced bias.
3. Falls back gracefully: if tau*=inf is selected the estimator reduces to
   standard cross-fitted DR.

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

import numpy as np

from ..core.errors import OPEError
from ..core.types import BanditLog, OPEEstimate
from .reward_model import TAU_GRID, RidgeRewardModel, clip_weights, cross_fit_predictions, ess_ratio


def _validate_log_pe(log: BanditLog, pe_chosen: np.ndarray) -> np.ndarray:
    pe = np.asarray(pe_chosen, dtype=np.float64)
    if pe.shape != (log.n,):
        raise OPEError(f"pe_chosen shape {pe.shape} != ({log.n},)")
    if np.any(pe <= 0):
        raise OPEError("evaluation propensities must be > 0")
    return pe


class IPSEstimator:
    """Clipped Inverse Propensity Scoring (Horvitz-Thompson)."""

    name = "ips"

    def __init__(self, pe_chosen: np.ndarray, clip_tau: float = float("inf")) -> None:
        self.pe_chosen = np.asarray(pe_chosen, dtype=np.float64)
        self.clip_tau = clip_tau

    def estimate(self, log: BanditLog) -> OPEEstimate:
        pe = _validate_log_pe(log, self.pe_chosen)
        w = clip_weights(pe / log.propensities, self.clip_tau)
        value = float(np.mean(log.rewards * w))
        return OPEEstimate(estimator=self.name, value=value,
                           ess_ratio=ess_ratio(w), clip_tau=self.clip_tau)


class SNIPSEstimator:
    """Self-Normalized IPS (weighted average; no clipping parameter needed)."""

    name = "snips"

    def __init__(self, pe_chosen: np.ndarray) -> None:
        self.pe_chosen = np.asarray(pe_chosen, dtype=np.float64)

    def estimate(self, log: BanditLog) -> OPEEstimate:
        pe = _validate_log_pe(log, self.pe_chosen)
        w = pe / log.propensities
        denom = float(np.sum(w))
        if denom <= 0:
            raise OPEError("SNIPS weights sum to zero")
        value = float(np.sum(log.rewards * w) / denom)
        return OPEEstimate(estimator=self.name, value=value, ess_ratio=ess_ratio(w))


class DMEstimator:
    """Direct Method: plug-in value via per-arm ridge reward model."""

    name = "dm"

    def __init__(self, pe_full: np.ndarray, num_actions: int, context_dim: int,
                 lam: float = 1.0) -> None:
        self.pe_full = np.asarray(pe_full, dtype=np.float64)
        self.K = num_actions
        self.d = context_dim
        self.lam = lam

    def estimate(self, log: BanditLog) -> OPEEstimate:
        if self.pe_full.shape != (log.n, self.K):
            raise OPEError(f"pe_full shape {self.pe_full.shape} != ({log.n}, {self.K})")
        model = RidgeRewardModel(self.K, self.d, lam=self.lam).fit(log)
        mu_all = model.predict_all(log.contexts)  # (n, K)
        value = float(np.mean(np.sum(self.pe_full * mu_all, axis=1)))
        return OPEEstimate(estimator=self.name, value=value)


class DREstimator:
    """Doubly Robust (Dudik et al. 2011) with cross-fitted mu.

    Correct DR form per sample:
        psi_i = mu_pie(x_i) + w_i * (r_i - mu_hat(x_i, a_i))
    where mu_pie(x) = sum_k pi_e(k|x) * mu_hat(k|x) is the eval-policy plug-in
    value (FULL-fit model) and the correction term uses the LOGGED-action
    prediction. Cross-fitting removes own-observation overfitting from the
    correction term.
    """

    name = "dr"

    def __init__(self, pe_full: np.ndarray, num_actions: int, context_dim: int,
                 clip_tau: float = float("inf"), n_folds: int = 5,
                 lam: float = 1.0, use_cross_fit: bool = True,
                 label: str = None) -> None:
        self.pe_full = np.asarray(pe_full, dtype=np.float64)
        self.K = num_actions
        self.d = context_dim
        self.clip_tau = clip_tau
        self.n_folds = n_folds
        self.lam = lam
        self.use_cross_fit = use_cross_fit  # ablation switch
        if label is not None:
            self.name = label  # disambiguate fixed-tau variants (dr / dr20)

    def _mu_parts(self, log: BanditLog):
        """Return (baseline mu_pie(x_i), correction mu_hat(x_i, a_i))."""
        if self.use_cross_fit:
            mu_oof_action, full = cross_fit_predictions(
                log, self.K, self.d, n_folds=self.n_folds, lam=self.lam)
        else:  # ablation: plug-in full-fit (own-observation bias in correction)
            full = RidgeRewardModel(self.K, self.d, lam=self.lam).fit(log)
            mu_oof_action = full.predict_action(log.contexts, log.actions)
        mu_all = full.predict_all(log.contexts)  # (n, K), full-fit
        baseline = np.sum(self.pe_full * mu_all, axis=1)
        return baseline, mu_oof_action

    def _pseudo_outcomes(self, log: BanditLog, baseline: np.ndarray,
                         mu_action: np.ndarray, w: np.ndarray) -> np.ndarray:
        return baseline + w * (log.rewards - mu_action)

    def estimate(self, log: BanditLog) -> OPEEstimate:
        if self.pe_full.shape != (log.n, self.K):
            raise OPEError(f"pe_full shape {self.pe_full.shape} != ({log.n}, {self.K})")
        pe_chosen = self.pe_full[np.arange(log.n), log.actions]
        baseline, mu_action = self._mu_parts(log)
        w = clip_weights(pe_chosen / log.propensities, self.clip_tau)
        psi = self._pseudo_outcomes(log, baseline, mu_action, w)
        return OPEEstimate(estimator=self.name, value=float(np.mean(psi)),
                           ess_ratio=ess_ratio(w), clip_tau=self.clip_tau)


class CFDRACEstimator(DREstimator):
    """Novel: Cross-Fitted DR with Adaptive Clipping (IRM-style tau selection).

    tau is chosen oracle-free from TAU_GRID by maximizing
        mean(psi_tau) - std(psi_tau)/sqrt(n)
    on pseudo-outcomes. No ground-truth value is used.
    """

    name = "cfdrac"

    def __init__(self, pe_full: np.ndarray, num_actions: int, context_dim: int,
                 n_folds: int = 5, lam: float = 1.0,
                 tau_grid: tuple = TAU_GRID, use_cross_fit: bool = True) -> None:
        super().__init__(pe_full, num_actions, context_dim,
                         clip_tau=float("inf"), n_folds=n_folds, lam=lam,
                         use_cross_fit=use_cross_fit)
        self.tau_grid = tau_grid

    def estimate(self, log: BanditLog) -> OPEEstimate:
        if self.pe_full.shape != (log.n, self.K):
            raise OPEError(f"pe_full shape {self.pe_full.shape} != ({log.n}, {self.K})")
        pe_chosen = self.pe_full[np.arange(log.n), log.actions]
        baseline, mu_action = self._mu_parts(log)
        raw_w = pe_chosen / log.propensities
        best_tau, best_score, best_value, best_ess = None, -np.inf, None, None
        for tau in self.tau_grid:
            w = clip_weights(raw_w, tau)
            psi = self._pseudo_outcomes(log, baseline, mu_action, w)
            e = ess_ratio(w)
            # CLT-correct penalty: scale by the EFFECTIVE sample size so that
            # low overlap (small ESS) automatically discourages aggressive
            # clipping (bias) as much as it encourages variance reduction.
            n_eff = max(log.n * e, 1.0)
            score = float(np.mean(psi) - 2.0 * np.std(psi) / np.sqrt(n_eff))
            if score > best_score:
                best_score = score
                best_tau = tau
                best_value = float(np.mean(psi))
                best_ess = e
        return OPEEstimate(estimator=self.name, value=best_value,
                           ess_ratio=best_ess, clip_tau=best_tau,
                           reason=f"IRM-selected tau={best_tau}")


class GatedOPEEstimator:
    """Overlap-gated estimator router (diagnostics-driven, oracle-free).

    Routing rules (in order):
      1. ESS/n < ESS_LOW  -> CF-DR-AC (importance weights too heavy for SNIPS).
      2. reward-model pseudo-R^2 < R2_LOW -> CF-DR-AC (DM/DR baseline untrustworthy).
      3. otherwise -> SNIPS (healthy overlap: lowest variance, asymptotically tight).

    The chosen route and its rationale are attached to the returned estimate.
    """

    name = "gated"
    ESS_LOW = 0.05
    R2_LOW = 0.01

    def __init__(self, pe_chosen: np.ndarray, pe_full: np.ndarray,
                 num_actions: int, context_dim: int, n_folds: int = 5,
                 lam: float = 1.0, use_cfdrac: bool = True,
                 use_gating: bool = True) -> None:
        self.pe_chosen = np.asarray(pe_chosen, dtype=np.float64)
        self.pe_full = np.asarray(pe_full, dtype=np.float64)
        self.K = num_actions
        self.d = context_dim
        self.n_folds = n_folds
        self.lam = lam
        self.use_cfdrac = use_cfdrac  # ablation switch
        self.use_gating = use_gating  # ablation switch

    def _route(self, log: BanditLog) -> tuple:
        pe_chosen = self.pe_full[np.arange(log.n), log.actions]
        w = pe_chosen / log.propensities
        e = ess_ratio(w)
        if not self.use_gating:
            est = (CFDRACEstimator(self.pe_full, self.K, self.d,
                                   n_folds=self.n_folds, lam=self.lam)
                   if self.use_cfdrac else
                   SNIPSEstimator(pe_chosen))
            return est, f"gating=off -> fixed {est.name}"
        if e < self.ESS_LOW:
            return (CFDRACEstimator(self.pe_full, self.K, self.d,
                                    n_folds=self.n_folds, lam=self.lam),
                    f"low overlap: ESS/n={e:.3f} < {self.ESS_LOW} -> CF-DR-AC")
        model = RidgeRewardModel(self.K, self.d, lam=self.lam).fit(log)
        r2 = model.oof_r2(log)
        if r2 < self.R2_LOW:
            return (CFDRACEstimator(self.pe_full, self.K, self.d,
                                    n_folds=self.n_folds, lam=self.lam),
                    f"weak reward model: R2={r2:.4f} < {self.R2_LOW} -> CF-DR-AC")
        return SNIPSEstimator(pe_chosen), (
            f"healthy overlap: ESS/n={e:.3f}, R2={r2:.4f} -> SNIPS")

    def estimate(self, log: BanditLog) -> OPEEstimate:
        est, reason = self._route(log)
        out = est.estimate(log)
        return OPEEstimate(estimator="gated", value=out.value,
                           ess_ratio=out.ess_ratio, clip_tau=out.clip_tau,
                           reason=reason)


def build_ope_estimators(pe_chosen: np.ndarray, pe_full: np.ndarray,
                         num_actions: int, context_dim: int) -> list:
    """Default estimator suite used by the benchmark."""
    return [
        IPSEstimator(pe_chosen),
        SNIPSEstimator(pe_chosen),
        DMEstimator(pe_full, num_actions, context_dim),
        DREstimator(pe_full, num_actions, context_dim, label="dr"),      # tau=inf control
        DREstimator(pe_full, num_actions, context_dim, clip_tau=20.0, label="dr20"),
        CFDRACEstimator(pe_full, num_actions, context_dim),
        GatedOPEEstimator(pe_chosen, pe_full, num_actions, context_dim),
    ]
