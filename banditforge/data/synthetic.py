"""Synthetic contextual linear bandit world with known ground truth.

Design notes
------------
* Ground truth is analytic up to a Monte-Carlo integral over the context
  distribution: E[r(x,a)] = x @ beta_a. The evaluation policy is an
  epsilon-greedy rule on the *true* beta, so its per-context expected value
  is (1-eps) * max_k x@beta_k + eps * mean_k x@beta_k.
* Logging policy is a softmax over the true scores with temperature T:
  T high -> near-uniform logging (high overlap), T low -> near-deterministic
  (low overlap). Propensities are floored at PROP_FLOOR to avoid zeros.
* Monte-Carlo ground truth uses an independent, fixed seed (MC_SEED) so that
  evaluation never shares randomness with benchmark seeds (no leakage).

Author: 晨星 (CJX0712)
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.errors import DataError
from ..core.types import BanditLog

PROP_FLOOR = 1e-4
MC_SEED = 97


class LinearBanditWorld:
    """Contextual bandit with K arms, context dim d, known ground truth.

    Expected reward of arm a at context x is x @ beta_a + C_INTER * x0 * x1.
    The multiplicative interaction term makes linear reward models mildly
    misspecified, which is the realistic regime where DR correction and
    adaptive clipping actually matter.
    """

    C_INTER = 0.6

    def __init__(self, num_actions: int, context_dim: int, seed: int = 7,
                 rng: np.random.Generator = None, beta_scale: float = 1.5) -> None:
        if num_actions < 2:
            raise DataError("num_actions must be >= 2")
        if context_dim < 2:
            raise DataError("context_dim must be >= 2")
        self.K = num_actions
        self.d = context_dim
        if rng is None:
            rng = np.random.default_rng(seed)
        # beta_scale=1.5 fixed by a parameter sweep (docs/architecture.md):
        # wide arm gaps make the adaptive-exploration advantage of
        # LinUCB/LinTS over eps-greedy measurable and significant.
        self.beta = beta_scale * rng.standard_normal((self.K, self.d)) / np.sqrt(self.d / 2.0)
        # Independent logging preferences: the logging policy is NOT aligned
        # with the ground-truth greedy rule, so softmax temperature genuinely
        # controls logging/evaluation overlap (high T -> uniform -> high overlap;
        # low T -> logging concentrates on its own favourite arms -> low overlap).
        self.beta_log = beta_scale * rng.standard_normal((self.K, self.d)) / np.sqrt(self.d / 2.0)

    def expected_reward(self, context: np.ndarray, action: int) -> float:
        inter = self.C_INTER * context[0] * context[1]
        return float(context @ self.beta[action]) + inter

    def best_expected_reward(self, context: np.ndarray) -> float:
        return float(np.max(self.all_expected_rewards(context)))

    def all_expected_rewards(self, context: np.ndarray) -> np.ndarray:
        inter = self.C_INTER * context[0] * context[1]
        return self.beta @ context + inter

    def logging_propensities(self, context: np.ndarray, temp: float) -> np.ndarray:
        """Softmax logging propensity over arms at temperature ``temp``.

        Scores come from the INDEPENDENT logging preferences beta_log, not
        from the ground-truth beta, so temperature controls true overlap.
        """
        if temp <= 0:
            raise DataError("logging temperature must be > 0")
        scores = (self.beta_log @ context) / temp
        scores = scores - scores.max()
        p = np.exp(scores)
        p = p / p.sum()
        p = np.maximum(p, PROP_FLOOR)
        return p / p.sum()

    def sample_context(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.d)

    def gen_log(self, n: int, seed: int, temp: float, noise: float = 0.5) -> BanditLog:
        """Generate an offline log drawn from the logging policy."""
        if n < 1:
            raise DataError("n must be >= 1")
        rng = np.random.default_rng(seed)
        contexts = np.empty((n, self.d))
        actions = np.empty(n, dtype=np.int64)
        rewards = np.empty(n)
        props = np.empty(n)
        for i in range(n):
            x = self.sample_context(rng)
            p = self.logging_propensities(x, temp)
            a = int(rng.choice(self.K, p=p))
            r = self.expected_reward(x, a) + noise * rng.standard_normal()
            contexts[i] = x
            actions[i] = a
            rewards[i] = r
            props[i] = p[a]
        return BanditLog(contexts=contexts, actions=actions,
                         rewards=rewards, propensities=props)

    def eval_policy_propensity(self, context: np.ndarray, eps: float = 0.05) -> np.ndarray:
        """Evaluation policy: eps-greedy on the TRUE beta (eps-mixed uniform)."""
        scores = self.all_expected_rewards(context)
        best = int(np.argmax(scores))
        p = np.full(self.K, eps / self.K)
        p[best] += 1.0 - eps
        return p

    def eval_policy_expected_value(self, context: np.ndarray, eps: float = 0.05) -> float:
        """Per-context expected reward of the eval policy under true model."""
        vals = self.all_expected_rewards(context)
        return float((1.0 - eps) * vals.max() + eps * vals.mean())

    def true_eval_value(self, n_mc: int, eps: float = 0.05) -> float:
        """Monte-Carlo ground-truth value of the eval policy (fixed MC seed)."""
        if n_mc < 10_000:
            raise DataError("n_mc must be >= 10000 for reliable ground truth")
        rng = np.random.default_rng(MC_SEED)
        xs = rng.standard_normal((n_mc, self.d))
        vals = xs @ self.beta.T  # (n_mc, K)
        vals = vals + self.C_INTER * (xs[:, 0] * xs[:, 1])[:, None]
        per_ctx = (1.0 - eps) * vals.max(axis=1) + eps * vals.mean(axis=1)
        return float(per_ctx.mean())


class EvalPolicyVectorized:
    """Vectorized eval-policy propensity/value for OPE over a whole log."""

    def __init__(self, world: LinearBanditWorld, eps: float = 0.05) -> None:
        self.world = world
        self.eps = eps
        self.name = "true-eps-greedy(0.05)"

    def propensities_batch(self, contexts: np.ndarray) -> np.ndarray:
        """(n, K) propensity matrix of the eval policy."""
        vals = contexts @ self.world.beta.T  # (n, K)
        best = np.argmax(vals, axis=1)
        p = np.full_like(vals, self.eps / self.world.K)
        p[np.arange(vals.shape[0]), best] += 1.0 - self.eps
        return p

    def expected_values_batch(self, contexts: np.ndarray) -> np.ndarray:
        vals = contexts @ self.world.beta.T
        return (1.0 - self.eps) * vals.max(axis=1) + self.eps * vals.mean(axis=1)


def build_specs(world_seed: int = 7, n_log: int = 2000, n_mc: int = 100_000,
                noise: float = 0.5) -> List[dict]:
    """Build the default benchmark dataset specs (world + ground truth).

    Overlap ladder: logging softmax temperature controls the overlap between
    logging and evaluation behaviour. Returns dicts with everything the
    pipeline needs; ground-truth value computed once per spec.
    """
    configs = [
        # name,                K,  d,   temp,  overlap label
        ("lin-K5-d10-high",    5, 10,  8.0, "high"),
        ("lin-K5-d10-medium",  5, 10,  2.0, "medium"),
        ("lin-K5-d10-low",     5, 10,  0.6, "low"),
        ("lin-K10-d10-medium", 10, 10,  2.0, "medium"),
    ]
    specs = []
    for name, k, d, temp, overlap in configs:
        world = LinearBanditWorld(k, d, seed=world_seed)
        specs.append({
            "name": name,
            "world": world,
            "num_actions": k,
            "context_dim": d,
            "overlap": overlap,
            "n_log": n_log,
            "logging_temp": temp,
            "reward_noise": noise,
            "true_value": world.true_eval_value(n_mc),
        })
    return specs
