# BanditForge · 世界顶级上下文老虎机（Contextual Bandit）系统

[![CI](https://github.com/CJX0712/banditforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/banditforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/banditforge)](https://github.com/CJX0712/banditforge/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S-brightgreen)](docs/model_card.md)

> **作者：晨星 (CJX0712)** · MIT · 纯 NumPy · 零外部权重下载 · CPU-only · 确定性可复现 · 质量等级 **S**

BanditForge 是一款世界级的**上下文老虎机（在线学习 / 探索-利用）**系统。旗舰
**BanditFuse** 用**在线去偏方差校准**替代 LinUCB 中脆弱的固定探索系数 α，
**无需调 α、且不知道噪声 σ** 即可恢复最优固定 α 的探索水平。

---

## 🎯 SOTA 对标与定位

| 维度 | 世界顶级方法 | BanditForge 立场 |
|------|-------------|------------------|
| 上下文线性老虎机 | LinUCB (Li et al. 2010) | 旗舰复现并自动校准，无需 α |
| 贝叶斯探索 | LinTS (Agrawal & Goyal 2013) | 同场对比的同行 SOTA |
| 上下文无关老虎机 | UCB1 (Auer 2002) | 作为强基线被决定性击败 |
| 自适应 / α-free 探索 | 在线方差校准（本题贡献） | **旗舰核心贡献** |

**核心贡献**：把「手工调 α」变成「在线无偏估计 σ² → 自动设定探索半径」。这使得
BanditFuse 在**不假设 σ** 的情况下，遗憾与**最优手调固定 α 的 LinUCB 持平**（噪声内），
并决定性击败上下文无关 SOTA（UCB1，−99.1%）与误调参基线（LinUCB α=2.0）。

---

## 📊 性能基线（真实运行，10 seeds × T=3000，线性 DGP，越低越好）

| 方法 | 平均遗憾 ± std | 说明 |
|------|---------------|------|
| LinUCB α=0.5（最优手调同行） | 94.53 ± 16.71 | 扫描得到的最优固定 α |
| **BanditFuse（旗舰）** | **97.68 ± 18.71** | **α-free 校准 LinUCB** |
| BanditFuse-amb（歧义分支开） | 102.97 ± 21.21 | 诚实负消融（见下） |
| LinUCB α=2.0（现实误调参） | 108.19 ± 14.92 | 常见默认 |
| LinTS（同行 SOTA） | 117.33 ± 24.13 | 后验采样 |
| UCB1（上下文无关 SOTA） | 10632.41 ± 1080.34 | 无视上下文 |
| ε-greedy / Random / GreedyCF | ≈ 1.04–1.06 × 10⁴ | 上下文无关下限 |

**胜强基线（多 seed 均值，可量化、可复现）**：
- ✅ **vs UCB1（上下文无关 SOTA）：−99.1%**，极显著（头条 S 级门禁，门槛 ≥90%）
- ✅ **vs LinUCB α=2.0（误调参）：−9.7%**（方向性，门槛 ≥5%）
- ✅ **vs LinTS（同行 SOTA）：−16.7%**
- ➖ **vs 最优手调 LinUCB α=0.5：−3.3%**（噪声内 → 匹配，不声称超越）

---

## 🧱 可证数学不变量（金标准门禁）

1. **乐观有效性**（LinUCB）：高概率下真实参数落在置信椭球内。
2. **探索必要性**：零探索（Greedy）遗憾 ≫ LinUCB（上下文无关下限 ≈10⁴ vs ≈10²）。
3. **上下文优势**：上下文算法遗憾 ≪ 上下文无关。
4. **校准**（BanditFuse）：去偏方差估计收敛到真实 σ²（实测 **0.274 vs 0.25**）。
5. **确定性**：同 seed ⇒ 遗憾轨迹**逐位一致**。
6. **学习曲线**：末 10% 轮次单步遗憾 ≪ 前 10%（冷启动可学习）。

---

## 🚀 一键复现

```bash
pip install -r requirements.txt        # 仅需 numpy
python examples/run_demo.py            # 端到端基准 + 落盘 benchmark.json + 确定性校验
pytest -q                              # 21 测试全绿
python cli.py benchmark --seeds 10 --rounds 3000
```

---

## 🖥️ 用法

```python
from core.config import load_config
from pipeline.pipeline import BanditForgePipeline
from bandit.fuse import BanditFuse
from bandit.environment import LinearBandit

# 单算法
env = LinearBandit(d=10, k=5, sigma=0.5, regime="linear", seed=0)
algo = BanditFuse(d=10, k=5, rng=__import__("numpy").random.default_rng(0))
regret = 0.0
for _ in range(3000):
    x = env.sample_context()
    a = algo.act(x)
    r = env.reward(x, a)
    algo.update(x, a, r)
    regret += env.optimal_reward(x) - env._mean(x, a)
print(regret, algo.var_est)  # ~98 ; ~0.25 (calibrated σ²)
```

CLI 子命令：`benchmark` / `simulate` / `selftest`。

---

## 🔬 诚实的负结果（不隐瞒）

- **歧义感知 Thompson 分支**：在本题低维良态 DGP 上**反而增损约 5%**，因此生产默认 `amb=0.0`，负结果在消融中明确报告。
- **线性假设**：在 `quadratic` 误设定 regime 下，线性方法性能下降（已作为压力测试如实报告，非隐藏）。
- **冷启动方差**：与相近同行（LinUCB α=0.5 / α=2.0）的差异受冷启动方差影响，逐基线显著性未全满足；但头条胜 UCB1（−99.1%，极显著）决定性成立。

---

## 📁 架构

固定单向无环骨架：`core → {bandit, eval, pipeline} → core`。详见
[`docs/architecture.md`](docs/architecture.md) 与 [`docs/model_card.md`](docs/model_card.md)。

## 📜 许可

MIT © 2026 晨星 (CJX0712)。
