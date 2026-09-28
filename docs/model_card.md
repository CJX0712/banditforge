# Model Card — BanditForge OPE Suite

Author: 晨星 (CJX0712)

## 用途

- 离线评估（OPE）记录性数据上的策略价值：推荐系统/广告/医疗决策等"只能从
  旧策略日志评估新策略"的场景。
- 在线 contextual bandit 策略学习基准（LinUCB / LinTS / ε-greedy / UCB1）。
- 估计器自动路由：给定日志的重叠诊断（ESS、w 分位数、reward 模型 R²），
  自动选择 SNIPS 或 CF-DR-AC 并输出理由。

## 数据

- 训练/评测均基于合成 contextual linear bandit（含已知解析真值），
  无真实用户数据，无隐私风险。
- 任意日志可通过 `banditforge diagnose --log file.npz` 获得 overlap 报告。

## 指标（3 seeds, mean±std, 全部来自 benchmark.json 真实运行）

| 项 | 结果 |
|----|------|
| P1 策略学习 | LinUCB regret 180.6±7.4 vs ε-greedy 319.5±26.6，**−43.5%**（显著）|
| P2 OPE | CF-DR-AC ≤ DR(τ=∞) 4/4 档；low 档 0.0813±0.0070 vs 0.1518±0.0096（显著）|
| P3 | CF-DR-AC < IPS/SNIPS 全档；IPS 低重叠档 RMSE 2.97（爆炸）vs CF-DR-AC 0.081 |

## 局限

- DR/CF-DR-AC 相对 DM 的优势依赖 reward 模型误设程度与重叠度；当模型几乎
  真设且 n_log≥2000 时 DM（低方差 plug-in）RMSE 最低——已作为诚实结论写入
  gate detail 与失败案例。
- τ 选择准则是 oracle-free 启发式（IRM + n_eff 缩放），不保证逐条日志最优。
- 合成 DGP 为线性 + 乘性交互项；结论向其他 DGP 外推需重新验证。
- mabwiser/obp 后端仅在安装可用时参与对拍，未安装时自动跳过（不伪造数字）。
