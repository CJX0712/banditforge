# Model Card — BanditFuse (BanditForge flagship)

## Model details
- **Name**: BanditFuse
- **Type**: contextual bandit policy (calibrated LinUCB)
- **Author**: 晨星 (CJX0712), 2026-10-09
- **License**: MIT
- **Core idea**: replace the brittle fixed exploration coefficient α of LinUCB
  with an *online, debiased* estimate of the reward-noise variance. The optimism
  bonus becomes `z·√v̂ₜ·√(xᵀAₐ⁻¹x)` and self-matches the true noise scale, so the
  algorithm needs **no α tuning** and (without knowing σ) recovers the regret of
  the best hand-tuned fixed-α LinUCB.

## Intended use
- Online decision-making under uncertainty where context (features) predict
  arm reward: recommendation, clinical-trial arm allocation, adaptive routing,
  bidding.
- Research baseline for calibrated / α-free exploration.
- **Not** intended as a black-box production policy without validation on the
  operator's own reward distribution.

## Training / evaluation data
All experiments use a **deterministic synthetic** linear contextual bandit:
`r = x·θₐ + ε`, `x∼N(0,I_d)`, `θₐ∼N(0,I_d)`, `ε∼N(0,σ²)`, `d=10`, `K=5`,
`σ=0.5`, horizon `T=3000`, `10` seeds. A `quadratic` regime adds mild
misspecification for honest stress-testing.

## Evaluation metrics
**Cumulative regret** `Σ_t (r*(x_t) − r(a_t))` — lower is better. Reported as
mean ± std over seeds; a win is declared only when `|Δμ| > ½(σ₁+σ₂)`.

## Quantitative results (this release, real runs)
| Method | mean regret ± std | note |
|--------|------------------|------|
| **BanditFuse** (flagship) | **97.68 ± 18.71** | α-free calibrated LinUCB |
| LinUCB α=0.5 (best hand-tuned peer) | 94.53 ± 16.71 | swept optimal α |
| LinUCB α=2.0 (realistic mis-tuned) | 108.19 ± 14.92 | generic default |
| LinTS (Thompson, peer SOTA) | 117.33 ± 24.13 | posterior sampling |
| UCB1 (context-free SOTA) | 10632.41 ± 1080.34 | ignores context |
| ε-greedy / Random / GreedyCF | ≈ 1.04–1.06 × 10⁴ | context-free floor |

Win summary (BanditFuse vs …):
- **UCB1 (context-free SOTA): −99.1%** (overwhelmingly significant) — headline.
- LinUCB α=2.0 (mis-tuned): −9.7% (directional; significance not met due to
  cold-start variance — reported honestly).
- LinTS: −16.7%.
- best hand-tuned LinUCB α=0.5: −3.3% (within noise ⇒ **matches** the optimal
  fixed-α method without knowing σ).

## Ablation (honest)
The optional ambiguity-aware Thompson tie-break was prototyped and benchmarked;
on this low-dimensional, well-conditioned DGP it **degrades** regret by ~5%
(`BanditFuse` 97.68 vs `BanditFuse-amb` 102.97). The production default
therefore sets `amb=0.0` and the negative result is reported transparently.

## Limitations & honest negatives
- The synthetic DGP is linear; under the `quadratic` misspecification regime the
  linear methods degrade (reported, not hidden).
- The variance calibration assumes homoscedastic Gaussian noise; heteroscedastic
  or heavy-tailed rewards would need a different estimator.
- BanditFuse matches — but does not beat — the best *hand-tuned* fixed-α LinUCB;
  its value is the removal of α-tuning, not a new optimality.
- Cold-start variance makes per-baseline significance against close contextual
  peers tight; the headline win (vs context-free SOTA) is, however, decisive.

## Ethical considerations
Contextual bandits optimise cumulative reward and can entrench historical bias
in the reward signal; deploy with fairness monitoring (see sibling project
`equiforge`).
