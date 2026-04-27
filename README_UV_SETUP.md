# Splendor-AI UV 快速开始

## 5 分钟内跑通

### 1. 激活环境
```bash
cd /home/wangyb/TestProject/Splendor-AI
source .venv/bin/activate
```

### 2. 跑遗传算法 baseline
```bash
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t
```

### 3. 跑 10 局
```bash
splendor -a splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent,splendor.agents.our_agents.minmax --agent_names=genetic,minimax -t -m 10
```

## 预期结果

- 遗传算法通常能稳定赢过 Minimax
- PPO 目前能运行，但权重表现不佳
