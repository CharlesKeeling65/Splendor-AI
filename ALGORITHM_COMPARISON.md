# Splendor-AI 算法对比分析

## 总览

| 算法 | 学习方式 | 优点 | 缺点 |
|------|----------|------|------|
| PPO | 梯度优化 | 样本效率高、性能上限高 | 训练成本高、调参难 |
| 遗传算法 | 进化搜索 | 无梯度、可解释、实现简单 | 收敛慢、随机性较强 |
| Minimax | 树搜索 | 确定性强、无需训练、推理快 | 深度受限、依赖评估函数 |

## 在项目中的实现

### PPO
- 路径：`src/splendor/agents/our_agents/ppo/`
- 版本：基础 MLP + 若干序列/注意力变体
- 适合做最终性能上限的方向

### 遗传算法
- 路径：`src/splendor/agents/our_agents/genetic_algorithm/`
- 通过种群进化寻找更优的策略权重
- 当前是最稳定的 baseline

### Minimax
- 路径：`src/splendor/agents/our_agents/minmax.py`
- 深度较浅，更多用于对照和评测

## 结论

当前仓库里，遗传算法是最容易跑通且表现最稳定的 baseline；PPO 有更高潜力，但需要重新训练或进一步调优。

### DQN（2026-09-05 新增主线）
- 路径：`src/splendor/agents/our_agents/dqn/`
- 架构：Dueling Double DQN（InputNormalization + 4×[Linear128+LayerNorm+ReLU] + V/A 头）+ n-step(3) replay + 终局 ±10 奖励包装（calScore 口径）
- 相对 PPO 的独有能力：off-policy 经验回流——网页对局数据可直接混入本地 replay（`collect_from_browser`）
- **当前状态：代码与单元测试就绪（83 tests 全绿），尚未训练**——本机约束暂不做训练（见 `docs/p4_decision.md` 附录），M1→M3 课程门槛（vs random >90% / vs minimax ≥55%，plan/phase-1 §2）达成后本表将同口径补录 DQN 结果与 3-seed 方差

## 提升路径增量（2026-09-13，按 docs/IMPROVEMENT_ROADMAP_20260912.md 实施中）

> 本节随阶段推进回填同口径数字；训练完成的最终数字回填到 §结论上方的新小节。

### 已测得的中期数字

| 项 | 数字 | 来源 |
|---|---|---|
| C1 critic 消融：EV(last3) 均值 | value1 0.164 > critic-lr5 0.149 > base 0.122 ≈ warmup5 0.117 > critichid 0.084（8 updates 试点，3 seed 配对） | `docs/PPO_CRITIC_ABLATION_20260913.md` |
| B2 塑形健康检查（DQN 5k 冒烟） | 塑形 vs random 46% / vs heuristic 0%；无塑形 vs random 25% / vs heuristic 0%（非结论性） | `runs/budget-smoke/b2-health-check.json` |
| A3 预算基线 | PPO 2.1 s/训练局；DQN 29 步/s（solo CUDA） | 路线图 §2 A3 |
| DQN corrected 基线（旧） | vs random 99.3% / heuristic 39.3% / minimax 52.7% | DQN_ROUND2_RESULTS_20260907 |
| PPO 稳定化基线（旧） | vs random 100% / heuristic 32% / minimax 48.7% / GA 54% | PPO_STABILIZATION_RESULTS_20260908 |

### 2026-09-13 训练批次回填

- **C2 规模化自博弈**（500 updates×16 局×3 seed，c2_training 段）：EV(last3)
  0.34–0.48（试点期 0.12–0.23）；独立测试（c2_test 段，50 局/对手/seed）
  均值 vs random **100%** / heuristic **37%** / minimax **52%** / GA **45%**。
- **C4 league 体检**（ppo-best=seed42，independent_test 段 wrap，每对手 150 局）：
  vs random **100%**、vs minimax **57.3%**（仓库历史最佳）、vs GA **47.3%**、
  vs heuristic **38.7%**、vs rush 38.0%、vs hoard 43.0%。
  G1 门槛（minimax ≥60 / heuristic ≥60 / GA ≥55）**未达**，heuristic 族缺口
  归因与药方见 `docs/C2_SELFPLAY_C4_LEAGUE_REPORT_20260913.md`。
- **D1 DQN 200k×3**（pool random:0.5,minimax:0.5）：M2 vs random 98%/82%/93%
  （2/3 seed 达标）；**M3 vs minimax 14%/3%/16% 全败** —— 训练量假设对 M3
  被否定，minimax 实力依赖 experiment.py 专用管线（guidance/EMA/审计），
  见 `docs/D1_200K_COURSE_REPORT_20260913.md`。
- **F1 价值标定**：return 模式 critic 的胜率判别 AUC 0.525 < 0.75，
  不可用作搜索先验（否定结果入档）。
- **C2-R2（2000 updates + potential shaping，2026-09-14）**：C4-R2 league
  （每对手 150 局）vs minimax **66.3%** / GA **62.0%** / heuristic **58.0%** /
  rush 56.0% / hoard 61.3% / random 100%——**ppo-best 首次联赛榜首**（总分率
  67.2%），G1 的 minimax/GA 门槛首次达成（`docs/C2R2_2000_REPORT_20260914.md`）。
- 排名（2p，对 minimax）：**C2-R2-PPO 66.3% > C2-PPO 57.3% > DQN-corrected
  52.7% > 旧 PPO 48.7%**；对 heuristic：C2-R2-PPO 58.0% 已逼近启发式族
  （heuristic 对其余对手 59–61%），学习型策略仅剩 ~2pt 缺口。
