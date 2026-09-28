# Changelog

## v0.1.0 (2026-09-28)

- 首个发布：Contextual Bandit（LinUCB/LinTS/ε-greedy/UCB1）+ OPE
  （IPS/SNIPS/DM/DR/CF-DR-AC/门控路由）双引擎。
- 创新：CF-DR-AC（cross-fitted DR + IRM-style 自适应 clipping + ESS 门控）。
- 49 单测全绿，行覆盖 90%，ruff clean，benchmark 逐位确定，3 seeds mean±std。
- 可选 mabwiser/obp SOTA 后端 + 纯 numpy 离线兜底。
- 作者：晨星 (CJX0712)
