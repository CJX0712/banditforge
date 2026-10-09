# BanditForge — Architecture

BanditForge follows the fixed, unidirectional, acyclic skeleton mandated by the
delivery SOP:

```
banditforge/
  core/        seed, errors, types, config, interfaces   (determinism + contracts)
  bandit/      environment, linucb, lints, ucb1, baselines, fuse   (domain)
  eval/        metrics                                     (significance + reduction)
  pipeline/    pipeline                                     (orchestration + benchmark)
  cli.py       argparse entry point
  examples/    run_demo.py
  tests/       pytest suite
  docs/        architecture.md, model_card.md
```

Call graph (single direction, no cycles):

```
cli -> pipeline -> {bandit, eval, core} -> core
```

## Module responsibilities

### `core/` — infrastructure
- **`seed.RngBundle` / `set_all`** — one master integer seed fans out to every
  NumPy generator. All algorithms and the synthetic environment consume an
  explicit `np.random.Generator`, so two runs with the same seed are
  bit-identical.
- **`errors`** — numeric error taxonomy `E100–E500` (`ConfigError`, `DataError`,
  `NumericalError`, `AlgorithmError`, `BanditForgeRuntimeError`).
- **`types`** — `BanditConfig` (schema-validated), `RunResult`, `MethodSummary`,
  `BenchmarkReport`.
- **`config.load_config`** — `BANDFORGE_*` environment overrides + schema check.
- **`interfaces`** — `BanditAlgorithm` and `Environment` Protocols; the pipeline
  and tests swap implementations (SOTA, baselines, oracle) without touching
  orchestration.

### `bandit/` — domain algorithms
| Class | Reference | Exploration mechanism |
|-------|----------|----------------------|
| `LinearBandit` | synthetic DGP (Li et al. 2010 setting) | oracle truth |
| `LinUCB` | Li, Chu, Langford, Schapire 2010 | optimism: `x·θ̂ + α√(xᵀA⁻¹x)` |
| `LinTS` | Agrawal & Goyal 2013 | posterior sampling `θ̃∼N(θ̂, v²A⁻¹)` |
| `UCB1` | Auer, Cesa-Bianchi, Fischer 2002 | context-free `√(2ln t / nₐ)` |
| `EpsilonGreedy`/`GreedyCF`/`RandomCF` | — | context-free, fixed/zero exploration |
| **`BanditFuse`** | this work | **debiased online variance calibration** (flagship) |

### `eval/` — statistics
`significance_mean_diff` (win iff `|Δμ| > ½(σ₁+σ₂)`), `relative_reduction`,
`summarize` (mean / std / SEM over seeds).

### `pipeline/` — orchestration
`BanditForgePipeline.run()` executes every method over `n_seeds` independent
environments, aggregates regret, tests the flagship's win gates, performs an
ablation of the ambiguity component, derives honest failure cases, and verifies
determinism (bit-identical re-run) and noise-variance calibration.

## Mathematical invariants (gold-standard contracts)
1. **Optimism validity (LinUCB)** — with high probability the true parameter
   lies in the confidence ellipsoid, so the optimistic arm is never systematically
   under-estimated; empirically the realised regret is `≤` the optimistic gap.
2. **Exploration necessity** — greedy (zero exploration) regret ≫ LinUCB on a
   contextual problem (demonstrated: context-free floor ≈ 10⁴ vs ≈ 10²).
3. **Contextual advantage** — contextual methods regret ≪ context-free on a
   contextual DGP.
4. **Calibration (BanditFuse)** — the debiased variance estimate converges to the
   true `σ²` (empirically `0.274` vs true `0.25`).
5. **Determinism** — same seed ⇒ identical regret trajectory (bit-identical).
6. **Learning (sublinearity)** — per-step regret in the final 10% of rounds is
   far below the first 10% (cold-start curve).
