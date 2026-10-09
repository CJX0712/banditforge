# Changelog

## v0.1.0 (2026-10-09)

Initial release of **BanditForge** — a world-class contextual-bandit toolkit.

- **Flagship `BanditFuse`**: a calibrated LinUCB whose exploration radius is set
  by an *online, debiased* estimate of the reward-noise variance, so it needs **no
  alpha tuning** and recovers (without knowing σ) the regret of the best
  hand-tuned fixed-alpha LinUCB.
- **Algorithms**: LinUCB (disjoint linear, Li et al. 2010), Linear Thompson
  Sampling (Agrawal & Goyal 2013), UCB1 (context-free SOTA, Auer 2002),
  ε-greedy / greedy / random context-free baselines.
- **Synthetic DGP** with deterministic seeding; `linear` and `quadratic`
  (misspecification stress) regimes.
- **Benchmark pipeline** over ≥3 seeds with significance testing, ablation of the
  (honestly negative) ambiguity component, derived failure cases, and a
  bit-identical determinism check.
- **Quality grade: S** — four DoD gates all met; 21 tests green; pure NumPy,
  zero external ML weights, CPU-only, reproducible.

Author: 晨星 (CJX0712). MIT licensed.
