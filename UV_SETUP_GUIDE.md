# Splendor-AI 项目 - UV 环境配置指南

## 项目分析

Splendor-AI 是一个基于 Splendor 桌游的 AI 研究项目，主要包含三类智能体：PPO、遗传算法和 Minimax。

### 使用的算法

#### 1. PPO (Proximal Policy Optimization)
- 路径：`src/splendor/agents/our_agents/ppo/`
- 变体：基础 MLP、Self-Attention、GRU、LSTM
- 优点：样本效率高、训练稳定、适合复杂策略学习
- 缺点：训练成本高、超参数敏感、需要较多算力

#### 2. 遗传算法 (Genetic Algorithm)
- 路径：`src/splendor/agents/our_agents/genetic_algorithm/`
- 结构：ManagerGene + 3 个 StrategyGene
- 优点：无需梯度、可解释、适合黑盒优化
- 缺点：收敛慢、结果波动较大

#### 3. Minimax (Alpha-Beta 剪枝)
- 路径：`src/splendor/agents/our_agents/minmax.py`
- 优点：无需训练、推理快、行为确定
- 缺点：搜索深度浅、依赖手工评估函数

## UV 配置步骤

### 1. 安装 UV
```bash
brew install uv
# 或
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 同步依赖并创建虚拟环境
```bash
cd /home/wangyb/TestProject/Splendor-AI
uv sync
```

### 3. 激活虚拟环境
```bash
source .venv/bin/activate
which python
python --version
```

### 4. 安装项目
```bash
pip install -e .
```

### 5. 验证 CLI
```bash
splendor --help
```

## Baseline 运行

### 遗传算法 vs Minimax
```bash
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t
```

### 多局测试
```bash
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t -m 10
```

### PPO vs Minimax
```bash
splendor -a splendor.agents.our_agents.ppo.ppo_agent,splendor.agents.our_agents.minmax --agent_names=ppo,minimax -t
```

## 故障排除

- 如果找不到 `splendor`，重新执行 `pip install -e .`
- 如果提示 CUDA 驱动太旧，可以先忽略，CPU 会自动兜底
- 如果要重新开始，删除 `.venv` 后执行 `uv sync`
