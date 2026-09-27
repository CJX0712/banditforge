# BanditForge

![CI](https://github.com/CJX0712/banditforge/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/tag/CJX0712/banditforge?label=release)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Quality](https://img.shields.io/badge/quality-S-brightgreen)

**模块化 Contextual Bandit + Off-Policy Evaluation (OPE) 系统** — 在线策略学习
（LinUCB / LinTS / ε-greedy / UCB1）与离线策略价值评估（IPS / SNIPS / DM / DR /
**CF-DR-AC** / 重叠度门控路由）双引擎。CPU 可跑、固定 seed 逐位可复现、
零下载离线兜底。作者：晨星 (CJX0712)。

## 为什么是 BanditForge

- **在线侧**：LinUCB/LinTS 的自适应探索在 3 seeds 上比调参 ε-greedy 少 **43.5%**
  累积 regret（180.6±7.4 vs 319.5±26.6，过显著性门槛）。
- **离线侧**：CF-DR-AC（cross-fitted DR + IRM-style 自适应 ratio clipping +
  ESS 门控路由）在低重叠日志上把 RMSE 从 DR(∞) 的 0.152 降到 **0.081**
  （K5-low 档，显著）；IPS 在同档 RMSE 爆炸到 2.97。
- **诚实披露**：DM（低方差 plug-in）在模型近似真设且日志量大时 RMSE 最低，
  已写入 gate detail 与失败案例——本系统不回避基线的强项。

## 一键复现

```bash
pip install -r requirements.txt
python -m banditforge.cli benchmark --output benchmark.json   # 或:
python banditforge/examples/run_demo.py
pytest -q                                                     # 49 tests
```

同 seed 两次运行除 `elapsed_sec` 外**逐位一致**（含测试断言）。

## 基准摘要（seed=20260928, 3 seeds mean±std, 全部来自 benchmark.json）

### 在线策略学习（累积 regret，越低越好，T=3000）

| policy | regret |
|--------|--------|
| **linucb** | **180.6 ± 7.4** |
| lints | 234.1 ± 39.7 |
| eps_greedy（调参强基线）| 319.5 ± 26.6 |
| ucb1（context-agnostic）| 6311.8 ± 35.1 |

### 离线评估 OPE（RMSE vs Monte-Carlo 真值，越低越好，n_log=2000 × 20 repeats）

| estimator | K5-high | K5-medium | K5-low | K10-medium |
|-----------|---------|-----------|--------|------------|
| **cfdrac (本文)** | **0.045** | **0.057** | **0.081** | **0.062** |
| dr20 | 0.045 | 0.057 | 0.081 | 0.062 |
| dr (τ=∞) | 0.045 | 0.064 | 0.152 | 0.090 |
| dm | 0.040 | 0.056 | 0.069 | 0.042 |
| snips | 0.080 | 0.137 | 0.470 | 0.138 |
| ips | 0.132 | 0.215 | 2.968 | 0.288 |

- CF-DR-AC vs DR(τ=∞)：全档非劣 + low/K10 两档显著（>0.5×(σ1+σ2) 门槛）。
- 消融：no_cross_fit / no_adaptive_clip / no_gating 三开关均见 `benchmark.json`。
- 失败案例 ≥3 条，全部从 results 派生。

## CF-DR-AC 是什么

```
psi_i(τ) = μ̂_πe(x_i) + clip(w_i, τ)·(r_i − μ̂_oof(x_i, a_i))
τ* = argmax_τ mean(ψ(τ)) − 2·std(ψ(τ))/√(n·ESS(τ))     # oracle-free, CLT 口径
```

1. **Cross-fitting**：折外 μ̂ 消除校正项自观测过拟合。
2. **自适应 τ**：惩罚按有效样本量 n·ESS 缩放——低重叠自动放大惩罚，
   同时避免过度 clip（bias）与不 clip（方差爆炸）。
3. **门控路由**：ESS/n < 0.05 或 reward 模型 R² < 0.01 → CF-DR-AC，
   否则 SNIPS，理由随结果输出。

## 架构

```
cli → pipeline → {data, bandits, ope, eval, hpo} → core   # 单向无环
core:  types(dataclass) · errors(E100~E500) · config(ENV_BF_*) · seed · interfaces(Protocol)
data:  LinearBanditWorld(独立 logging 偏好 + softmax 温度控制 overlap) + MC 真值(独立 seed)
ope:   ips · snips · dm · dr · cfdrac · gated · diagnostics
```

详见 [docs/architecture.md](docs/architecture.md) · [docs/model_card.md](docs/model_card.md)。

## 可选 SOTA 后端

`pip install ".[sota]"` 安装 mabwiser（MAB 库）与 obp（Open Bandit Pipeline）。
不可用时 `available_mabwiser()` 探测自动跳过，benchmark 标注 skipped——
Tier-1 纯 numpy 链路永远可跑（零下载、离线兜底）。

## 配置

全部 `ENV_BF_*` 环境变量覆盖：`ENV_BF_SEED / ENV_BF_N_LOG / ENV_BF_N_MC /
ENV_BF_T_STEPS / ENV_BF_N_SEEDS / ENV_BF_N_REPEAT / ENV_BF_HPO_TRIALS`。

## License

MIT © 晨星 (CJX0712)
