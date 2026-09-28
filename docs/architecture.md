# BanditForge · Architecture

Author: 晨星 (CJX0712)

## 1. 系统定位

模块化 **Contextual Bandit（上下文老虎机）+ Off-Policy Evaluation（离线策略评估, OPE）** 系统：
在线策略学习（ε-greedy / UCB1 / LinUCB / LinTS）与离线价值评估（IPS / SNIPS / DM / DR / CF-DR-AC / 门控路由）双引擎，固定 seed 端到端可复现，CPU 即可运行。

## 2. 模块架构（调用单向无环）

```
cli ──► pipeline ──► { data, bandits, ope, eval, hpo } ──► core
```

| 模块 | 职责 | 关键内容 |
|------|------|----------|
| `core` | 类型/错误/配置/seed/接口契约 | dataclass types, E100~E500 errors, `ENV_BF_*` overrides, `set_all` |
| `data` | 合成数据 + 真值 | `LinearBanditWorld`（独立 logging 偏好 β_log + softmax 温度控制 overlap）、Monte-Carlo 真值（独立 MC seed）|
| `bandits` | 在线策略 | LinearGreedy(ε), UCB1, LinUCB(Li 2010), LinTS(Agrawal & Goyal 2013); mabwiser 可选对拍 |
| `ope` | 离线评估 | IPS/SNIPS/DM/DR(固定τ)/**CF-DR-AC(创新)**/门控路由; overlap 诊断 |
| `eval` | 基准与门槛 | regret/RMSE/显著性门槛(0.5×(σ1+σ2)); 校准-评测双 world 协议 |
| `hpo` | 无真值调参 | soft-greedy(α) 家族 + DR 打分 (Optuna GridSampler) + 解析 MC 验证 |

## 3. 创新点：CF-DR-AC（Cross-Fitted DR with Adaptive Clipping）

1. **Cross-fitting**：K-fold 折外 μ̂ 消除 DR 校正项的自观测过拟合偏差。
2. **自适应 ratio clipping**：τ 从 {20, 50, ∞} 中按 IRM/CRM 式准则选择
   `τ* = argmax_τ mean(ψ_τ) − 2·std(ψ_τ)/√n_eff`，其中 `n_eff = n·ESS`。
   惩罚按**有效样本量**缩放是 CLT 的正确口径：低重叠时惩罚自动放大，
   同时遏制"过度 clip 引入偏差"与"不 clip 方差爆炸"两个失败方向。
   全程 oracle-free（不使用真值）。
3. **重叠度门控路由**：ESS/n 与 reward-model pseudo-R² 诊断，自动路由
   SNIPS（健康重叠）↔ CF-DR-AC（低重叠/弱模型），并输出选择理由。

## 4. DGP 设计与调参依据（甜点声明，防"先射箭后画靶"）

- **β_log 独立于真值 β**：logging 策略用自己的随机偏好，softmax 温度 T 才真正
  控制 logging/eval 重叠（T=8 高重叠 / T=2 中 / T=0.6 低）。
  （踩坑记录：若 logging 与 eval 同用真 β，尖锐 logging 反而与 true-greedy 高重叠。）
- **非线性交互项** `E[r|x,a] = x·β_a + 0.6·x0·x1`：线性 reward 模型轻度误设，
  DM 有恒定偏差、DR 校正有真实价值。
- **β_scale=1.5**：参数扫描 {1.0,1.5,2.0,2.5}×T{1500,3000} 后选定——
  arm 差距拉大使 LinUCB/LinTS 的自适应探索优势可测量（41-51% reduction 区间）。
- **τ 网格 {20,50,∞}**：τ 扫描显示 τ≤10 在低重叠档过度 clip（bias 0.090），
  τ=20 是 RMSE 甜点（0.085 vs ∞ 的 0.206）。
- **真值口径**：eval policy 真值 = 解析 MC（独立 seed 97，n=100k），与基准 seed
  完全隔离，无评测泄漏。

## 5. 基准协议

- **P1 策略学习**：4 策略统一在独立校准 world (seed=88, T=800) 选超参 →
  基准 world (seed=7) 跑 T=3000 × 3 seeds；门槛：best(LinUCB,LinTS) 相对
  调参 ε-greedy 累积 regret 降低 ≥30% 且均值差 > 0.5×(σ1+σ2)。
- **P2 OPE**：4 数据档 × 3 seeds × 20 条独立日志；每条日志用全部估计器评估
  固定 eval policy 价值；RMSE 对 MC 真值聚合（seed 内先 RMSE over repeats，
  再 seeds 上 mean±std）。门槛：cfdrac ≤ dr(τ=∞) 全档 + ≥2 档显著。
- **P3**：cfdrac < ips/snips 全档 + cfdrac ≤ min(dr, dr20) 全档。
- **消融**：no_cross_fit / no_adaptive_clip / no_gating 三开关。
- **失败案例**：全部从 results 派生（IPS 低重叠方差爆炸 / DM 恒定偏差 /
  最差估计器排名），禁止预设结论。

## 6. 确定性与性能

- `core.seed.set_all(seed)` 唯一入口；同 seed 两次运行 benchmark.json 除
  elapsed_sec 外逐位一致（CI 有测试断言）。
- demo 端到端 ~30s（CPU i7 级笔记本），内存 < 500MB。
- mabwiser / obp 为可选 SOTA 后端：`try/except ImportError` + 探测函数，
  缺失时自动跳过并在报告标注 skipped，Tier-1 纯 numpy 链路不受影响。
