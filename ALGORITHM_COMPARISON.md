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

### 待回填（训练进行中，2026-09-13 启动）

- C2 规模化自博弈 500×16×3 seed（c2_training 种子段）：EV 曲线、验证选模、独立测试 vs random/minimax（M3 门槛）。
- D1 DQN 200k×3（pool random:0.5,minimax:0.5）：M2（vs random ≥90%）与 M3（vs minimax ≥55%/100 局）逐 seed 报告。
- C4 league 体检矩阵（G1 门槛：vs minimax ≥60%、vs heuristic ≥60%、vs GA ≥55%）。
