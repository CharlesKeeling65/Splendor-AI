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
