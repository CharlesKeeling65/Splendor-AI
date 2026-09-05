# Splendor-AI · DQN 应用完整指导文件

> 本文档基于对仓库的完整并行调研（游戏引擎 / Gym 环境 / PPO 及现有 agent / 基础设施），所有关键接口均已在源码中逐一核对。
> 目标读者：要在此仓库上实现 DQN（Deep Q-Network）系列算法的开发者。
> 结论先行：**该仓库对 DQN 极其友好**——Gym 环境的固定离散动作空间 + 合法动作掩码 + 向量化观测三件套完全就绪（`splendor_env.py:46-59` 的 docstring 明确说明该设计为 DQN 准备），网络主干与权重保存约定可从 PPO 代码近乎逐行改写。真正的工作量集中在：**① 奖励改造（加终局胜负信号）、② Replay Buffer + 目标网络训练循环（新写约 200~300 行）、③ 训练对手的课程设计**。

---

## 目录

1. [仓库架构总览](#1-仓库架构总览)
2. [MDP 三要素在本仓库的落点](#2-mdp-三要素在本仓库的落点)
3. [Agent 接口与运行/评估路径](#3-agent-接口与运行评估路径)
4. [现有算法资产盘点：复用什么、避开什么](#4-现有算法资产盘点复用什么避开什么)
5. [DQN 总体设计方案](#5-dqn-总体设计方案)
6. [工程实现指南（代码骨架）](#6-工程实现指南代码骨架)
7. [训练对手策略与课程](#7-训练对手策略与课程)
8. [训练与评估流程](#8-训练与评估流程)
9. [陷阱清单（必读）](#9-陷阱清单必读)
10. [里程碑与验收标准](#10-里程碑与验收标准)
11. [进阶路线](#11-进阶路线)

---

## 1. 仓库架构总览

```
Splendor-AI/
├── src/splendor/
│   ├── game.py                      # 对局运行器（评测用游戏主循环）
│   ├── general_game_runner.py       # CLI 评测器（splendor 命令的入口）
│   ├── template.py                  # Agent/GameState/GameRule 基类
│   ├── splendor/                    # ── 游戏引擎层 ──
│   │   ├── splendor_model.py        # 规则核心（~580 行，完全信息博弈）
│   │   ├── features.py              # 状态特征提取（265 维）
│   │   ├── constants.py             # 游戏常数
│   │   ├── types.py                 # ActionType 等 TypedDict
│   │   └── gym/                     # ── Gym 环境层 ──
│   │       └── envs/
│   │           ├── splendor_env.py  # SplendorEnv（对手自动模拟）
│   │           ├── actions.py       # ALL_ACTIONS（3510 个固定动作）
│   │           └── utils.py         # 掩码 / 索引映射工具
│   └── agents/
│       ├── generic/                 # random / first_move / timeout
│       └── our_agents/
│           ├── ppo/                 # PPO 家族（MLP/GRU/LSTM/self-attn）
│           ├── minmax.py            # 深度 2 alpha-beta
│           └── genetic_algorithm/   # 遗传算法（当前最稳 baseline）
├── pyproject.toml                   # console scripts: splendor / ppo / evolve
└── ALGORITHM_COMPARISON.md          # 官方结论：GA 最稳，PPO 需重训
```

三层分工：**引擎层**实现规则（`SplendorGameRule`），**Gym 层**把多智能体回合制博弈折叠成单智能体 MDP（对手回合在 `step()`/`reset()` 内部自动模拟），**Agent 层**通过统一接口 `SelectAction` 接入两条路径——训练走 Gym、评测走 `general_game_runner.py`。

游戏基本参数（`constants.py`）：2~4 人局、胜利 15 分、手宝石上限 10、预留上限 3、贵族各 3 分、100 回合强制截断。**完全信息博弈**：`private_information = None`（`splendor_model.py:147`），牌库顺序在 `state.board.decks` 中可见，无需 determinization。

---

## 2. MDP 三要素在本仓库的落点

### 2.1 状态 / 观测：`Box(shape=(265,))`，float32，ego-centric

| 项 | 事实 | 出处 |
|---|---|---|
| 观测空间 | `Box(-inf, inf, (265,))`，dtype float32 | `splendor_env.py:78-82` |
| 构成 | 70 维数值指标 + 15 张牌 × 13 维 | `features.py:470-487` |
| 视角 | 以 `my_turn` 为 agent_index 提取；对手只见分数（按前后座次分组），对手宝石/手牌不可见 | `splendor_env.py:201-206`，`features.py:276-282` |
| ⚠️ 归一化 | **env 输出的 obs 未归一化**（`normalize_metrics` 定义了但 env 不调用）；PPO 用网络内 `InputNormalization` 解决 | `features.py:313-322` |

特征提取函数（DQN 直接复用）：

```python
from splendor.splendor.features import extract_metrics_with_cards, normalize_metrics

obs = extract_metrics_with_cards(game_state, agent_index)   # → NDArray, shape (265,)
```

70 维指标包括：常数 1、回合数、分数、是否≥15 分、持卡数、预留数、购买力方差、黄金数、总宝石数、每色卡数/购买力/对数购买力、对手分数、12 张明牌的"缺宝石距离/预计回合距离"、3 张预留牌距离、5 个贵族距离。卡牌编码 13 维 = 颜色 one-hot(6) + 费用(5) + tier(1) + 分值(1)，空槽全零。

### 2.2 动作：`Discrete(3510)`，固定枚举 + 合法掩码

| 项 | 事实 | 出处 |
|---|---|---|
| 动作空间 | `Discrete(3510)`，与状态无关的全局固定枚举 `ALL_ACTIONS` | `splendor_env.py:74`，`actions.py:222-237` |
| 构成 | COLLECT_DIFF 2280 / COLLECT_SAME 630 / RESERVE 504 / BUY_AVAILABLE 72 / BUY_RESERVE 18 / PASS 6 | 实测验证 |
| 关键设计 | 买卡动作**不编码支付费用**（`returned_gems=None`），执行时由引擎按 `card.cost` + 折扣 + 黄金通配结算——这是动作空间从数万压缩到 3510 的原因 | `actions.py:100-104` |
| 合法动作数 | 每步实测 median 15、mean ≈ 22、max 136（远小于 3510） | 458 个随机决策点实测 |

掩码与映射工具（签名已核对，`gym/envs/utils.py`）：

```python
from splendor.splendor.gym.envs.utils import (
    create_legal_actions_mask,   # (legal_actions, state, agent_index) → NDArray (3510,) 0/1
    create_action_mapping,       # (legal_actions, state, agent_index) → dict[int, ActionType]
)

# 训练时（env 路径）：
mask = env.unwrapped.get_legal_actions_mask()          # 内部即调用上面两者
# 对局时（game.py 路径，SelectAction 内）：
mask = create_legal_actions_mask(actions, game_state, self.id)
mapping = create_action_mapping(actions, game_state, self.id)
```

⚠️ **非法动作直接抛异常使训练崩溃**（`step()` 先查 `mapping[action]`，KeyError 无兜底，`splendor_env.py:146-153`）。**采样/argmax 前必须过掩码，这是硬性要求。**

### 2.3 回合推进与奖励

- `step(action_idx)` 执行我方动作后，**自动模拟所有对手回合**直到再次轮到我或终局（`_simulate_opponents`，`splendor_env.py:219-231`）。学习侧只见到"轮到自己"的帧——多智能体博弈被折叠成单智能体 MDP。
- 返回 5 元组 `(obs, reward, terminated, truncated=False, info={})`。
- **当前奖励 = 我方分数增量**（买卡 +points、贵族 +3、其余为 0），**没有终局胜负奖励、没有任何额外塑形**（`splendor_env.py:142-161`）。
- `terminated=True` 的三种来源：≥15 分且回合完整 / 全员 pass 死锁 / 每人打满 100 回合（`ROUNDS_LIMIT`）。
- `reset()` 返回 `(obs, {"my_id": my_turn})`——**必须记录 my_id**。
- 环境属性（已核对）：`env.unwrapped.state`、`env.unwrapped.game_rule`、`env.unwrapped.my_turn`。

### 2.4 随机性与可复现（三个独立 RNG）

| 随机源 | 使用的 RNG | 固定方法 |
|---|---|---|
| 发牌/贵族抽样 | Python 全局 `random`（`splendor_model.py:83,89-99`） | `random.seed(s)` |
| 座次/先手 | numpy 全局 RNG（`splendor_env.py:104-112`） | `np.random.seed(s)` |
| 网络初始化/采样 | torch | `torch.manual_seed(s)` |

⚠️ `env.reset(seed=...)` **只 seed gymnasium 的 np_random，不固定发牌**。可复现必须三件套同时设置（`game.py:43-44` 正是这样做的）。

---

## 3. Agent 接口与运行/评估路径

### 3.1 Agent 接口（`template.py:59-68`）

```python
class Agent(object):
    def __init__(self, _id):
        self.id = _id

    def SelectAction(self, actions, game_state, game_rule):
        return random.choice(actions)
```

- 输入：`actions: list[ActionType]`（引擎 dict 格式的合法动作）、`game_state`（deepcopy 副本）、`game_rule`。
- 输出：从 `actions` 中选出一个动作 dict。
- **模块底部必须导出 `myAgent = XxxAgent`**（`general_game_runner.py` 动态加载的唯一入口）。
- 时限：接口本身无限制；引擎侧 `game.py:22` 的 `FREEDOM=True` 时不限时，`False` 时 1 秒/步 + 首回合 15 秒 warmup，3 次超时/非法直接判负。MLP 前向 + 掩码构建远快于 1 秒，不是问题。

### 3.2 两条运行路径

**训练路径（Gym）**：直接实例化 `gym.make("splendor-v1", agents=[对手列表])`——注意 `agents` 参数是**全部对手**（学习智能体不在其中），玩家总数 = `len(agents) + 1`。

**评测路径（CLI）**（命令格式已从 README 核对）：

```bash
# DQN vs minimax 打 100 局（文本模式）
splendor -a splendor.agents.our_agents.dqn.dqn_agent,splendor.agents.our_agents.minmax \
         --agent_names=dqn,minimax -t -m 100

# DQN vs 遗传算法
splendor -a splendor.agents.our_agents.dqn.dqn_agent,splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent \
         --agent_names=dqn,genetic -t -m 100

# 人机对战调试
splendor -a splendor.agents.our_agents.dqn.dqn_agent --agent_names=dqn,human --interactive
```

`-a` 接逗号分隔的模块导入路径，每个模块须暴露 `myAgent`。运行器自动统计平均分/胜/平/负。

---

## 4. 现有算法资产盘点：复用什么、避开什么

### 4.1 PPO（`agents/our_agents/ppo/`）

**结构**：`InputNormalization`（running mean/var）→ 4 × [Linear(128) + LayerNorm + Dropout(0.2) + ReLU] → actor 头 `Linear(128→3510)` + critic 头 `Linear(128→1)`；正交初始化。掩码方式：`torch.where(action_mask == 0, -1e8, actor_output)` 再 softmax（`network.py:114-115`）。

**训练**：每 episode 采一整局轨迹 → 10 epoch 全批量更新；loss = policy + 0.5·value − 0.005·entropy；Adam lr=1e-6 / wd=1e-4；γ=0.99；50000 episodes；"GAE" 实为归一化 MC 回报（无 TD(λ)）。默认对手 random、默认测试对手 minimax；支持真自博弈（对手共享同一网络对象，`ppo.py:177-184`）。

**官方评价（ALGORITHM_COMPARISON.md）**：PPO 有更高潜力但**当前权重不被信任，需要重训**；GA 是最稳 baseline。

| 直接复用 | 明确避开/修改 |
|---|---|
| 265 维特征 + 3510 掩码的全部环境层代码 | lr=1e-6（对 DQN 过小，DQN 用 1e-4 量级） |
| MLP trunk 结构（[128×4] + LayerNorm + ReLU） | Dropout(0.2)（off-policy 下建议去掉或调小） |
| 掩码手法（`torch.where(mask==0, 大负数, q)`） | 单 episode 全批量更新（DQN 必须 replay + minibatch） |
| checkpoint 约定 `{"model_state_dict", "running_mean", "running_var"}`（`ppo.py:71-93`，加载时 running stats 要 `squeeze(0)`，见 `ppo/utils.py:38-47`） | MC 回报目标（DQN 天然走 TD 目标） |
| `PPOAgentBase` 的 device/eval/myAgent 导出模式 | PPO 的 argmax 评测方式（DQN 需加 ε-greedy 训练/贪心评测区分） |
| γ=0.99、grad clip 1.0、SmoothL1 型损失、stats.csv 字段设计 | |

### 4.2 Minimax（`agents/our_agents/minmax.py`）

深度 2 + alpha-beta，仅支持 2 人。评估函数：终局 ±99999；否则 `score×2 + 手牌×0.7 + 宝石×0.1 − 宝石方差×0.2 − 卡牌费用距离×0.1`。搜索用 `generateSuccessor`（**原地修改 state**）+ `generatePredecessor`（精确逆操作回滚）。它是当前默认的评测基准对手。

### 4.3 遗传算法（`agents/our_agents/genetic_algorithm/`）

3 个 24 维策略基因 + 1 个 24×3 manager 基因，1 层前瞻打分选动作。**当前实测最稳的 baseline**，是 DQN 应当超越的目标。

### 4.4 引擎层可复用机制（如做搜索/DQN 混合）

- `game_rule.getLegalActions(state, agent_id)`：合法动作生成（含归还组合 × 贵族组合的枚举）。
- `generateSuccessor` / `generatePredecessor`：原地修改 + 精确回滚，免 deepcopy 的搜索模式。
- `calScore(state, agent_id)`：计分；**平分时买卡数少者 +0.5**（即引擎的胜负判定口径，DQN 终局奖励应沿用同一口径）。
- `game_rule.gameEnds()`：终局判定。

---

## 5. DQN 总体设计方案

### 5.1 MDP 建模

- **状态**：`obs ∈ R^265`（env 直接给出，未归一化，需自行处理）。
- **动作**：`a ∈ {0..3509}`，配合每步的合法掩码 `m(s) ∈ {0,1}^3510`。
- **转移**：`step(a)` 内部折叠了对手回合，因此转移核已含对手策略（对手更换 = 环境更换，这正是需要对手课程的原因）。
- **奖励**：`r = Δscore + 终局胜负奖励`（见 5.4，需自行包装 env）。
- **episode**：一局完整对局，约 50 个己方决策步（2 人局）。

由于 argmax/采样只在掩码内进行（典型 15~40 个合法动作），3510 的输出头并不构成学习负担——本质是"掩码内选择"。Dueling + Double DQN 完全可行。

### 5.2 网络架构：Dueling DQN

从 PPO 双头改单 Q 头（或 Dueling 双头），trunk 完全照搬但**去掉 Dropout**：

```
obs(265) → [Linear(128)+LayerNorm+ReLU] ×4 → ┬ V(s)      : Linear(128→1)
                                             └ A(s,·)    : Linear(128→3510)
Q(s,a;θ) = V(s) + A(s,a) − mean_a' A(s,a')      # Dueling 聚合
输出时非法动作 masked_fill(−1e9)                  # 与 PPO 掩码手法一致
```

- 掩码在 forward 内应用：argmax 天然合法；TD 学习只 gather 已执行（必合法）的动作值，`-1e9` 不会污染损失；bootstrap 的 `max Q(s',·)` 掩码后必落合法动作。
- 归一化二选一：(a) 复用 `InputNormalization`（checkpoint 必须连 running_mean/var 一起存取）；(b) 在 obs 进网络前调 `normalize_metrics` 的等价逻辑（注意它只归一化前 70 维指标，卡牌 195 维本身量纲已小）。推荐 (a)，与 PPO 生态一致。

### 5.3 训练算法：Double DQN + 目标网络 + Replay（+ 可选 n-step / PER）

| 组件 | 推荐配置 | 理由 |
|---|---|---|
| 基础 | **Double DQN**：在线网络选动作、目标网络评估 | 3510 头的过估计在稀疏奖励下会被放大，Double 是必要保险 |
| 目标网络 | 软更新 `θ⁻ ← θ⁻ + τ(θ − θ⁻)`，τ=0.005（或硬更新每 2000 步） | 稳定 bootstrap 目标 |
| Replay Buffer | 容量 5×10⁵，uniform，batch 256~512 | 一局约 50 步，跨 episode 混采打破相关性 |
| n-step（M2 起） | n=3 | 奖励稀疏（多数 step r=0），n-step 加速终局信号回传 |
| PER（M5 起，可选） | proportional，α=0.6，重要性采样 β 从 0.4 退火到 1 | 聚焦买卡/贵族/终局等低频高信息样本 |
| 损失 | Huber（`smooth_l1_loss`） | 沿用 PPO 的 value loss 类型 |
| 梯度裁剪 | 全局范数 1.0 | 沿用 PPO |

### 5.4 奖励改造（DQN 成败的第一关键）

当前 `r = Δscore` 没有胜负信号。MC 回报（PPO）尚可容忍，**TD 学习的 bootstrap 必须有终局信号沿轨迹反向传播**，否则学到的 Q 只是"剩余得分估计"而非"赢面估计"。必须包装 env：

```python
r_t = Δscore_t                                # 保留：即时分数收益作塑形
    + B · 𝟙[terminal] · (+1 赢 / 0 平 / −1 输)   # B ∈ [5, 20]，与分数增量拉开量级
```

胜负判定**必须沿用引擎口径** `calScore`（含 +0.5 买卡数平局裁定）。实现见 [6.2](#62-奖励包装rewardwrapper)。可选进阶：把 B 设为终局分差 `my_score − best_rival`（连续信号，比 ±1 更平滑）；或改成纯势函数塑形 `φ(s) = −(15 − my_score)`（保证最优策略不变性）。

> **勘误/裁决（2026-09-05，UPGRADE_ROADMAP §7）**：奖励改造仍由 wrapper 承担，但 (a) `SplendorEnvBase.step`
> 签名已增加 `payment: int | None = None` 前瞻参数（P4 支付维度预留），wrapper 与训练循环应透传该参数；
> (b) 同一 wrapper 原样适用于浏览器环境（两个环境奖励语义已对齐），无需 if-else 分支。

### 5.5 探索策略

- ε-greedy，ε 从 1.0 线性衰减到 0.05，衰减期覆盖前 20% 总步数。
- **随机动作必须在掩码内均匀采样**：`np.random.choice(np.flatnonzero(mask))`——直接 `np.random.randint(3510)` 会以极高概率非法并崩溃。
- 不要用 entropy 正则（那是 on-policy 手段）。

### 5.6 超参数表（推荐起点）

| 超参数 | 值 | 备注 |
|---|---|---|
| γ | 0.99 | 与 PPO 一致 |
| lr（Adam） | 1e-4 | **不要沿用 PPO 的 1e-6**；不稳定再降到 5e-5 |
| batch size | 512 | 265 维小网络，大 batch 更稳 |
| buffer 容量 | 5×10⁵ | ≈ 1 万局；内存约 265×8B×2×5e5 ≈ 2 GB（float64），用 float32 减半 |
| warmup | 5000 步 | 纯随机填充后再开始学习 |
| 目标网络 | 软更新 τ=0.005 | |
| ε | 1.0 → 0.05，前 20% 步数线性 | 掩码内均匀采样 |
| n-step | 3 | M2 阶段引入 |
| 网络宽度 | 128 × 4 层 | 沿用 PPO trunk，去 Dropout |

> **新增裁决（2026-09-05，UPGRADE_ROADMAP §1/§7）**：
> 1. **观测不含公共宝石供给**——`extract_metrics` 从不读 `board.gems`，265 维特征里没有对手可见的宝石池信息；
>    供给维度是 P4 可选项（T4.2），不改变本方案的观测契约。
> 2. **训练循环的掩码/映射构建走 P0.1 的 ActionIndexCache**（O(1) 查表），不要在热路径手写 `ALL_ACTIONS.index()`。
| 总环境步数 | 2×10⁵ ~ 10⁶ | 视吞吐调整，先实测 steps/sec |
| 评估频率 | 每 5000 步：vs random 20 局 + vs minimax 20 局 | 胜率而非得分作为主指标 |

---

## 6. 工程实现指南（代码骨架）

### 6.1 目录结构

```
src/splendor/agents/our_agents/dqn/
├── __init__.py
├── constants.py        # 超参数（对照 ppo/constants.py 风格）
├── network.py          # QNetwork（Dueling）+ InputNormalization 复用
├── replay_buffer.py    # 均匀 buffer + n-step + （可选）PER
├── reward_wrapper.py   # 终局胜负奖励包装
├── dqn.py              # 训练入口 main()（对照 ppo.py 结构）
├── dqn_agent.py        # 对局 agent（对照 ppo_agent.py 结构，导出 myAgent）
└── dqn_model.pth       # 训练产物（随包分发，参照 ppo_model.pth）
```

`pyproject.toml` 增加一行（与 `scripts.ppo` 并列）：

```toml
scripts.dqn = "splendor.agents.our_agents.dqn.dqn:main"
```

### 6.2 奖励包装（reward_wrapper.py）

```python
import gymnasium as gym


class SplendorRewardWrapper(gym.Wrapper):
    """保留 Δscore 塑形，追加终局胜负奖励（胜负口径 = calScore，含 +0.5 平局裁定）。"""

    def __init__(self, env, win_bonus: float = 10.0):
        super().__init__(env)
        self.win_bonus = win_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            unwrapped = self.env.unwrapped
            state = unwrapped.state
            my_id = unwrapped.my_turn
            my = unwrapped.game_rule.calScore(state, my_id)
            best_rival = max(
                unwrapped.game_rule.calScore(state, i)
                for i in range(len(state.agents)) if i != my_id
            )
            if my > best_rival:
                reward += self.win_bonus
            elif my < best_rival:
                reward -= self.win_bonus
        return obs, reward, terminated, truncated, info
```

> 注：`env.unwrapped.state / my_turn / game_rule` 属性名已在源码核对（`splendor_env.py:70,85,122,142`）。

### 6.3 网络（network.py）

```python
import torch
import torch.nn as nn

from splendor.splendor.features import METRICS_WITH_CARDS_SIZE
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

OBS_DIM = METRICS_WITH_CARDS_SIZE      # 265
ACTION_DIM = len(ALL_ACTIONS)          # 3510


class QNetwork(nn.Module):
    """Dueling DQN：Linear(265→128) ×4 trunk（LayerNorm+ReLU，无 Dropout）→ V 头 + A 头。"""

    def __init__(self, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(OBS_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, ACTION_DIM)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        v = self.value_head(h)                    # (B, 1)
        a = self.advantage_head(h)                # (B, 3510)
        q = v + a - a.mean(dim=-1, keepdim=True)  # Dueling 聚合
        return q.masked_fill(mask == 0, -1e9)     # 非法动作屏蔽（手法同 PPO 的 -1e8）

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> int:
        return int(self.forward(obs.unsqueeze(0), mask.unsqueeze(0)).argmax())
```

如复用 `InputNormalization`，在 trunk 前插入并随 checkpoint 保存 running_mean/var（加载参照 `ppo/utils.py:38-47` 的 `squeeze(0)` 处理）。

### 6.4 Replay Buffer（replay_buffer.py）

```python
import numpy as np


class ReplayBuffer:
    """均匀采样 buffer；n-step 用前置小队列实现（M2 阶段启用）。"""

    def __init__(self, capacity: int, obs_dim: int = 265, action_dim: int = 3510, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._pending: list[tuple] = []   # n-step 暂存队列

    def _store(self, obs, action, reward, next_obs, next_mask, done):
        i = self.pos
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.next_mask[i], self.done[i] = next_obs, next_mask, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add(self, obs, action, reward, next_obs, next_mask, done):
        """n-step=1 时直存；n>1 时攒满 n 条（或遇到 done）合并成一条 n-step 转移。"""
        self._pending.append((obs, action, reward, next_obs, next_mask, done))
        if len(self._pending) < self.n_step and not done:
            return
        # 合并暂存队列：R = Σ γ^k r_k，s' 取队尾的 s'
        obs0, act0, R, discount = self._pending[0][0], self._pending[0][1], 0.0, 1.0
        for (_, _, r_k, _, _, d_k) in self._pending:
            R += discount * r_k
            discount *= self.gamma
            if d_k:
                break
        *_, last_next_obs, last_next_mask, last_done = self._pending[-1]
        # 注意：若队列中途 done，截断的 n-step 转移仍以 done=1 收尾，TD 目标正确
        self._store(obs0, act0, R, last_next_obs, last_next_mask, last_done)
        if done:
            self._pending.clear()
        else:
            self._pending.pop(0)   # 滑动窗口

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.next_mask[idx]),
            torch.from_numpy(self.done[idx]),
        )

    def __len__(self):
        return self.size
```

> PER（可选）：将均匀 `sample` 换成按 `|TD 误差|^α` 的比例采样（sum-tree 实现），损失乘重要性采样权重 `w = (N·p)^{-β}`，β 从 0.4 退火到 1。

### 6.5 训练入口核心循环（dqn.py 的 main() 摘要）

```python
import random
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

def train(cfg):
    # ── 可复现三件套（env.reset(seed=) 不固定发牌！）──
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # ── 环境：agents 参数 = 对手列表，玩家数 = len+1 ──
    raw_env = gym.make("splendor-v1", agents=cfg.opponents)
    env = SplendorRewardWrapper(raw_env, win_bonus=cfg.win_bonus)

    q_net = QNetwork().float()
    target_net = QNetwork().float()
    target_net.load_state_dict(q_net.state_dict())

    buffer = ReplayBuffer(cfg.buffer_size, n_step=cfg.n_step)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    obs, info = env.reset(seed=cfg.seed)
    my_id = info["my_id"]
    mask = env.unwrapped.get_legal_actions_mask()

    for step in range(cfg.total_steps):
        # ── ε-greedy（掩码内均匀随机）──
        eps = max(cfg.eps_end, cfg.eps_start - cfg.eps_start * step / cfg.eps_decay_steps)
        if random.random() < eps:
            action = int(np.random.choice(np.flatnonzero(mask)))
        else:
            action = q_net.act(torch.from_numpy(obs.astype(np.float32)),
                               torch.from_numpy(mask.astype(np.float32)))

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_mask = (env.unwrapped.get_legal_actions_mask()
                     if not terminated else np.zeros_like(mask))

        buffer.add(obs, action, reward, next_obs, next_mask, terminated)
        obs, mask = next_obs, next_mask

        if terminated:
            obs, info = env.reset(seed=step)      # 发牌随机性由全局 random 控制
            my_id = info["my_id"]
            mask = env.unwrapped.get_legal_actions_mask()

        # ── 学习 ──
        if len(buffer) >= cfg.warmup:
            b_obs, b_act, b_rew, b_next_obs, b_next_mask, b_done = buffer.sample(cfg.batch_size)

            with torch.no_grad():
                # Double DQN：在线网络选动作，目标网络评估
                best_actions = q_net(b_next_obs, b_next_mask).argmax(dim=1)
                q_next = target_net(b_next_obs, b_next_mask).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target_q = b_rew + cfg.gamma * (1.0 - b_done) * q_next

            q_pred = q_net(b_obs, torch.ones_like(b_mask_placeholder)).gather(1, b_act.unsqueeze(1)).squeeze(1)
            # 注意：b_obs 对应的动作 b_act 必然合法，掩码可用训练时记录的该步 mask；
            #   为省内存也可传全 1 掩码——gather 只取合法动作处，结果不受影响。
            loss = F.smooth_l1_loss(q_pred, target_q)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

            # 软更新目标网络
            with torch.no_grad():
                for p, p_t in zip(q_net.parameters(), target_net.parameters()):
                    p_t.mul_(1 - cfg.tau).add_(cfg.tau * p)

        if step % cfg.eval_every == 0:
            evaluate(q_net, cfg)     # 见 8.1
        if step % cfg.save_every == 0:
            save_checkpoint(q_net, cfg)   # 约定见 6.7
```

> 代码中 `b_mask_placeholder` 一处：实际实现时建议在 buffer 里连该步 mask 一起存（本骨架为省篇幅省略），gather 处传该 mask；因 `b_act` 必合法，传全 1 掩码数学上等价。

### 6.6 对局 Agent（dqn_agent.py，对照 `ppo_agent.py` 逐行改写）

```python
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.envs.utils import create_action_mapping, create_legal_actions_mask
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .network import QNetwork

DEFAULT_SAVED_DQN_PATH = ...  # Path(__file__).parent / "dqn_model.pth"


class DQNAgent(Agent):
    def __init__(self, _id: int) -> None:
        super().__init__(_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(DEFAULT_SAVED_DQN_PATH, map_location=self.device, weights_only=False)
        self.net = QNetwork().float()
        self.net.load_state_dict(ckpt["model_state_dict"])
        # 若用 InputNormalization，此处恢复 running_mean/var（参照 ppo/utils.py:38-47）
        self.net = self.net.to(self.device)
        self.net.eval()

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        with torch.no_grad():
            state: NDArray = extract_metrics_with_cards(game_state, self.id).astype(np.float32)
            state_tensor = torch.from_numpy(state).to(self.device)
            action_mask = torch.from_numpy(
                create_legal_actions_mask(actions, game_state, self.id)
            ).float().to(self.device)

            q = self.net(state_tensor, action_mask)
            chosen_index = int(q.argmax())
            mapping = create_action_mapping(actions, game_state, self.id)

        return mapping[chosen_index]


myAgent = DQNAgent  # pylint: disable=invalid-name
```

### 6.7 Checkpoint 约定（沿用 PPO）

```python
torch.save({
    "model_state_dict": q_net.state_dict(),
    "running_mean": ...,   # 若用 InputNormalization：(1, 265)
    "running_var": ...,
    "step": step, "config": cfg.__dict__,
}, path)
```

参照 `ppo.py:71-93` 的保存与 `ppo/utils.py` 的 `load_saved_model`（注意 running stats 存的是 `(1, D)` 形状、加载时 `squeeze(0)`；加载用 `weights_only=False`）。

---

## 7. 训练对手策略与课程

### 7.1 三阶段课程

| 阶段 | 对手配置 | 目标 | 退出标准 |
|---|---|---|---|
| **M1** | 纯 `random`（模板 Agent 默认即随机） | 学会基本闭环：攒宝石→买分卡→拿贵族 | 平均终局得分持续上升（>10） |
| **M2** | 混合池：random + minimax(深度2) + GA agent，每局随机抽取 | 防过拟合单一对手；接触对抗性局面 | vs random 胜率 > 90% |
| **M3** | 自博弈（见 7.2）+ 定期回放混合池 | 追逐自身上限 | vs minimax 100 局胜率 ≥ 55%，与 GA 五五开以上 |

对手接入方式：`gym.make("splendor-v1", agents=[MinimaxAgent(0), GAAgent(1)])`——每局想换对手就重新 `gym.make`（对手列表在构造时传入；reset 时会自动洗牌座次）。

### 7.2 自博弈：DQN 的 off-policy 红利

PPO 的自博弈是"对手共享同一网络对象"（`ppo.py:177-184`）。DQN 有一个 PPO 没有的优势——**对手的经验也能进 replay buffer**（off-policy 允许旧策略/他视角数据）：

- **简单版（推荐先做）**：对手用旧 checkpoint 快照（冻结权重 + 固定 ε=0.1），每个训练阶段换一次快照——"league" 式伪自博弈，稳定性最好。
- **进阶版（真自博弈 + 经验回收）**：对手是当前网络的包装 Agent。在对手的 `SelectAction` 里记录 `(s_op, a_op)`，**下一次它被调用时**补齐 `(r_op, s'_op)`（此刻的 game_state 正是折叠了双方行动后的对手视角下一状态；r_op = 对手分数增量），终局由训练循环在 `terminated` 时调用 wrapper 的 `finalize(state)` 补最后一条。`extract_metrics_with_cards` 本身就是按 agent_index 提取的视角化特征，因此这些经验与主经验**同分布可直接混采**，数据量翻倍。

⚠️ 自博弈的非平稳性对 DQN 的目标网络冲击比 PPO 大：务必保留混合池对手（30% 比例）、降低 lr、加密评估频率。若训练发散，退回简单版。

---

## 8. 训练与评估流程

### 8.1 训练中内置评估（轻量，每 eval_every 步）

用独立 eval env（对手固定、`myAgent` 贪心无 ε），打 N 局统计胜/平/负与平均分：

```python
def evaluate(q_net, cfg, n_games=20):
    wins = draws = losses = 0
    for g in range(n_games):
        env = gym.make("splendor-v1", agents=[cfg.eval_opponent])   # 每局重建
        obs, info = env.reset(seed=g)
        mask = env.unwrapped.get_legal_actions_mask()
        done = False
        while not done:
            a = q_net.act(torch.from_numpy(obs.astype(np.float32)),
                          torch.from_numpy(mask.astype(np.float32)))
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            mask = env.unwrapped.get_legal_actions_mask() if not term else mask
        # 终局用 calScore 口径判胜负（同 6.2 的实现）
        ...
    return wins / n_games
```

指标写入 `stats.csv`（沿用 PPO 的字段设计：step、loss、q_mean、eps、train_reward、win_rate_vs_random、win_rate_vs_minimax）。

### 8.2 正式评估（仓库标准路径）

```bash
# 安装后（pip install -e .）
dqn                                        # 训练（console script，同 ppo/evolve）
splendor -a splendor.agents.our_agents.dqn.dqn_agent,splendor.agents.our_agents.minmax \
         --agent_names=dqn,minimax -t -m 100
```

对照基准（与 ALGORITHM_COMPARISON.md 的结论对齐）：
- vs `generic.random`：≥ 95%
- vs `minmax`（深度 2）：≥ 55%（100 局以上）
- vs `genetic_algorithm`：≥ 50%（GA 是当前最强稳定 baseline）

### 8.3 环境

- Python **3.12+**（引擎代码用 `typing.override`，3.11 下 ImportError，尽管 `pyproject.toml` 声明 >=3.11）。
- 依赖：torch、gymnasium（版本见 `requirements/runtime.txt`）；uv/conda 安装方式见 README 与 `environment.yaml`。

---

## 9. 陷阱清单（必读）

1. **非法动作 = 崩溃**：`step()` 先查 `mapping[action]` 直接 KeyError（`splendor_env.py:146-153`），无惩罚/无兜底。所有采样（含 ε-greedy 随机、Double DQN 的 argmax）必须过掩码。
2. **obs 未归一化**：env 不调用 `normalize_metrics`，Box 上下界 ±inf。必须用 `InputNormalization` 或自归一化，并把 running stats 存进 checkpoint。
3. **`reset(seed=)` 不固定发牌**：发牌走全局 `random`、座次走全局 numpy RNG。可复现需 `random.seed + np.random.seed + torch.manual_seed` 三件套。
4. **奖励无胜负信号**：不改奖励，DQN 学到的是"剩余得分估计"而非策略价值。见 5.4 / 6.2。
5. **每次 step/reset 后必须重新取掩码**：掩码与状态一一对应，不能缓存跨步使用。
6. **100 回合截断也是 `terminated=True`**：属于合法终局，配合胜负奖励即可，无需特判。
7. **`build_action()` 不可用于构造买卡动作**：它不处理黄金通配与已购卡折扣，会产生负宝石状态（其 docstring 自认，`gym/envs/utils.py:54-57`）。env 内部实际走 `create_action_mapping`；DQN 也只用后者。
8. **性能热点**：掩码/映射构建用 `ALL_ACTIONS.index()` 做O(3510) 线性查找（每个合法动作一次），`getLegalActions` 内部还有 deepcopy。单环境训练可接受；若做大规模自博弈，建议预构建 `{可哈希动作键: index}` 字典（`Action` 是含 dict 字段的 dataclass，需自定义键）。
9. **Card 不可哈希**：相等性按 `code + points`；做缓存/去重时注意。
10. **PPO 的 Dropout(0.2) 与 lr=1e-6 不要照搬**：DQN 去 Dropout、lr 用 1e-4 量级。
11. **`generateSuccessor` 原地修改状态**：若做 DQN+搜索混合（如 Q 值引导的一步前瞻），务必配对 `generatePredecessor` 回滚或 deepcopy。
12. **训练用 env 与评测用 game.py 的细微差异**：env 内对手拿真实 state 且无超时；game.py 给 agent 的 state/actions 是 deepcopy 且有 1 秒时限。完全信息博弈下无正确性影响，但注意推理延迟（1 秒内完成特征提取 + 前向 + 掩码构建，余量很大）。

---

## 10. 里程碑与验收标准

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M0** | 跑通环境：random 策略与 env 交互 1000 局不崩溃；确认 mask/obs/奖励语义；实测 steps/sec | 无崩溃；记录吞吐基线 |
| **M1** | Vanilla DQN（1-step、uniform buffer、vs random） | 平均终局得分从 ~5 升到 >10（学会买分卡、触发贵族） |
| **M2** | 奖励改造 + Double DQN + n-step(3) + 软更新 | vs random 胜率 > 90% |
| **M3** | 混合对手池（random/minimax/GA） | vs minimax(2) 100 局胜率 ≥ 55% |
| **M4** | 自博弈（先快照 league，后真自博弈+经验回收） | 与 GA agent 五五开以上；训练曲线不发散 |
| **M5** | 消融（n-step / Double / Dueling / 归一化各开关）+ 可选 PER | 出一张消融表；确认各组件贡献 |

每个里程碑跑通后再进入下一个；M2 若不达标，优先排查奖励设计（9 成的 DQN 失败是奖励/归一化问题）。

---

## 11. 进阶路线

1. **Rainbow 化**：PER + n-step + Dueling + Double + 噪声网络探索（替代 ε-greedy）+ 分布式 Q（C51）。本方案的代码结构（buffer 带 n_step 参数、Dueling 头、Double 目标）已为这些开关留好位置。
2. **Q 值引导搜索**：用 `generateSuccessor/generatePredecessor` 做 1 层前瞻 + Q(s') 评估（类似 GA agent 的结构但用学到的 Q 当评估函数）——AlphaZero 式"网络提速搜索"的轻量版。
3. **特征升级**：265 维手工特征是当前天花板的一部分。可尝试把 15 张牌改为 set-transformer 编码（仓库 `ppo/self_attn/` 已有先例）、加入对手建模特征。
4. **多人局扩展**：特征已按 `MAX_RIVALS=3` 设计支持 4 人；DQN 的 Q 头无需改动，但对手课程与胜负奖励（多人平局语义）需重设计。
5. **复现性基准**：固定 seed 跑 3 个 seed 出均值±方差，写入 ALGORITHM_COMPARISON.md，与 GA/PPO/minimax 同表对比。

---

## 附：关键文件行号速查

| 内容 | 位置 |
|---|---|
| Agent 基类 / `SelectAction` 接口 | `src/splendor/template.py:59-68` |
| 游戏规则核心 / 合法动作生成 / 终局与计分 | `src/splendor/splendor/splendor_model.py:142-576`（`gameEnds` :299，`calScore` :308） |
| 特征提取（265 维） | `src/splendor/splendor/features.py:470-487` |
| 固定动作枚举 ALL_ACTIONS（3510） | `src/splendor/splendor/gym/envs/actions.py:222-237` |
| 掩码 / 索引映射 | `src/splendor/splendor/gym/envs/utils.py:148-181` |
| Gym 环境（对手模拟 / 奖励 / 掩码入口） | `src/splendor/splendor/gym/envs/splendor_env.py`（`step` :127，`get_legal_actions_mask` :178） |
| PPO 网络（掩码手法 / trunk） | `src/splendor/agents/our_agents/ppo/network.py:18-116` |
| PPO 权重保存/加载（checkpoint 约定） | `src/splendor/agents/our_agents/ppo/ppo.py:71-93`，`ppo/utils.py:19-51` |
| 对局 agent 模板 | `src/splendor/agents/our_agents/ppo/ppo_agent.py` |
| 评测 CLI 用法 | `README.md`（"Let them play by them selves" 一节） |
| 官方算法对比结论 | `ALGORITHM_COMPARISON.md` |
