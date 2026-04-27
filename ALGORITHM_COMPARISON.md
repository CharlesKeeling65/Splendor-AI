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
