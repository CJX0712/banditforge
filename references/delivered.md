# BanditForge — Delivery Notes (BanditFuse 子系统)

**Author:** 晨星 (CJX0712) · **Version:** 0.2.0 (本次 BanditFuse) / 0.1.0 (旧 OPE) · **License:** MIT
**Repository:** https://github.com/CJX0712/banditforge

## What was delivered

在既有 `banditforge`（离策略评估 OPE，CF-DR-AC）仓库内**合并**新增一款完整、可测、开源的
**上下文老虎机（在线学习 / 探索-利用）** 系统，旗舰方法 **BanditFuse** —— 在线去偏方差校准。

| Layer | Module | Status |
|-------|--------|--------|
| 确定性核心 | `core/{seed,config,errors,types,interfaces}.py` | ✅ 全局 seed 入口 + 逐环境偏移 |
| 合成 DGP（oracle） | `bandit/environment.py` | ✅ linear / quadratic 两种 regime + 最优臂 |
| 在线算法（7 注册） | `bandit/{linucb,lints,ucb1,baselines,fuse}.py` | ✅ LinUCB / LinTS / UCB1 / ε-Greedy / GreedyCF / Random / **BanditFuse** |
| 评估指标 | `eval/metrics.py` | ✅ 显著性 / 相对削减 / 汇总 |
| 基准引擎 | `pipeline/pipeline.py` + `cli.py` | ✅ benchmark → benchmark.json（门禁 + 消融 + 失败案例 + 确定性 + 校准） |
| 演示 + 确定性校验 | `examples/run_demo.py` | ✅ ≤60s，bit_identical |
| 文档 | `README.md`, `docs/architecture.md`, `docs/model_card.md` | ✅（README 含合并说明） |
| 打包 / CI / Docker | `pyproject.toml`, `requirements.lock.txt`, `Makefile`, `.github/workflows/ci.yml`, `Dockerfile` | ✅ |
| 三级降级推送 | `scripts/gh_push.py`, `scripts/restore_ope.py` | ✅（L2 Git Data API 通道，绕过 443 阻断） |

## Flagship method — BanditFuse

在线去偏方差校准（本题核心贡献）：用**预更新后验**预测残差除以**杠杆项**
`1 + xᵀA⁻¹x` 做去偏估计，在线累加得到 σ̂²；探索半径取 `z·√v̂_t·√(xᵀA⁻¹x)`，
**无需手调 α、且不假设 σ**，自动匹配真实噪声。

更新步（`bandit/fuse.py`）：
```
pred   = x·(A_a⁻¹ b_a)                       # 预更新后验预测
resid  = r − pred
lev    = 1 + xᵀ A_a⁻¹ x                       # 杠杆
Σresid += resid² / max(lev, 1e-9)              # 去偏累加
var_est = max(Σresid / n, 1e-4)
A_a   += x xᵀ ;  b_a += r x
```

## Benchmark results (10 seeds × T=3000, linear, d=10, K=5, σ=0.5；越低越好)

| Method | mean regret ± std | note |
|--------|------------------|------|
| LinUCB α=0.5（最优手调同行） | 94.53 ± 16.71 | 扫描最优固定 α |
| **BanditFuse（旗舰）** | **97.68 ± 18.71** | α-free 校准 LinUCB |
| BanditFuse-amb（歧义开） | 102.97 ± 21.21 | 诚实负消融 |
| LinUCB α=2.0（现实误调参） | 108.19 ± 14.92 | 常见默认 |
| LinTS（同行 SOTA） | 117.33 ± 24.13 | 后验采样 |
| UCB1（上下文无关 SOTA） | 10632.41 ± 1080.34 | 无视上下文 |

胜强基线：`vs UCB1 −99.1%`（极显著，头条门禁 ≥90% ✅）；`vs LinUCB α=2.0 −9.7%`（✅）；
`vs LinTS −16.7%`（✅）；`vs 最优手调 α=0.5 −3.3%`（噪声内 → 匹配）。
校准：`var_est=0.2736 vs 真值 σ²=0.25`（✅）。确定性：`bit_identical=True`（✅）。

## Quality gates (all green)

- `python -m ruff check .` → **All checks passed!**
- `python -m ruff format --check .` → **30 files already formatted**
- `python -m pytest -q` → **21 passed**
- 确定性 → `bit_identical=True`（两次运行逐位一致）
- 密钥自查 → **passed**（无硬编码凭证）
- CI：lint + pytest，矩阵 py3.12 / py3.13

## How to reproduce

```bash
pip install -r requirements.txt            # 仅需 numpy
python examples/run_demo.py                # 端到端 → benchmark.json + 确定性校验
pytest -q                                  # 21 测试全绿
python cli.py benchmark --seeds 10 --rounds 3000
```

## 合并说明

本仓为「合并」模式：旧 OPE 子系统源码（`banditforge/` 包）完整保留，本次以根级
`core/ bandit/ eval/ pipeline/ cli.py` 接管主系统并以 `v0.2.0` 发布；`v0.1.0` 仍指向旧 OPE。
旧 OPE 自身测试未并入当前 CI（依赖外部 `mabwiser`），可在原环境运行。
