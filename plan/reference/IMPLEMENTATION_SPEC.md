# Splendor-AI 实现规格书（Implementation Spec）

> 上游文档：[UPGRADE_ROADMAP.md](./UPGRADE_ROADMAP.md)（架构裁决与阶段规划）· [DQN_GUIDE.md](./DQN_GUIDE.md)（DQN 算法方案）· [BROWSER_RL_MAPPING.md](./BROWSER_RL_MAPPING.md)（网页映射）。
> 本文是**可直接开工**的实现级规格：任务清单（含依赖/估时/验收）+ 每个模块的类与函数框架（签名、职责、关键实现注记、TODO）。
> 所有签名与字段均已于 2026-09-03 对源码逐一核实；本文新增的两项设计裁决见 §0.3。

---

## 目录

- [0. 总览](#0-总览)
- [1. P0 地基模块规格](#1-p0-地基模块规格)
- [2. P1 DQN 模块规格](#2-p1-dqn-模块规格)
- [3. P2 浏览器模块规格](#3-p2-浏览器模块规格)
- [4. P3 部署 Harness 规格](#4-p3-部署-harness-规格)
- [5. P4 接口预留：支付维度](#5-p4-接口预留支付维度)
- [6. 测试矩阵与 CI](#6-测试矩阵与-ci)
- [7. 编码约定](#7-编码约定)

---

## 0. 总览

### 0.1 任务看板

| ID | 任务 | 产出文件 | 依赖 | 估时 | 验收标准 |
|---|---|---|---|---|---|
| **T0.1** | ActionIndexCache | `gym/envs/utils.py`（改） | — | 0.5d | 新旧掩码/映射输出全等（全量 3510 断言）；构建提速 ≥50× |
| **T0.2** | Card/Noble Registry | `browser/card_registry.py` | — | 0.5d | 90 卡四元组、10 贵族 cost 全部命中且唯一 |
| **T0.3** | 环境协议 | `gym/base.py` | — | 0.25d | mypy 通过；`SplendorEnv` 满足协议 |
| **T0.4** | 网页规则实测（6 项） | `docs/web_experiments.md` | 浏览器 | 1~2d | 每项有截图/记录；结论回填 BROWSER 文档 |
| **T0.5** | 源文档勘误 | 两份 md | — | 0.25d | 按 UPGRADE_ROADMAP §7 修正 |
| **T1.1** | DQN 网络 | `dqn/network.py` | T0.3 | 0.5d | 形状/掩码/act 单测通过 |
| **T1.2** | Replay Buffer | `dqn/replay_buffer.py` | — | 0.5d | n-step 折扣累计正确性单测 |
| **T1.3** | 奖励包装 | `dqn/reward_wrapper.py` | — | 0.25d | 终局 ±B、平局 0 的口径 = calScore |
| **T1.4** | 训练核心 | `dqn/training.py` | T1.1-1.3 | 1d | Double DQN 更新单测；冒烟 2000 步不崩 |
| **T1.5** | 入口与对局 Agent | `dqn/dqn.py` `dqn/dqn_agent.py` `dqn/utils.py` | T1.4 | 1d | `dqn`/`splendor -a ...dqn_agent` 两命令可用 |
| **T1.6** | 训练与课程（M1→M3） | checkpoint + stats | T1.5 | 3~5d 挂机 | DQN_GUIDE §10 验收（vs random >90%，vs minimax ≥55%） |
| **T2.1** | DOM 抽取器 | `browser/dom_extractor.py` | T0.4 | 2d | 离线夹具单测全绿；真实页面抽取成功 |
| **T2.2** | 伪状态构建器 | `browser/state_builder.py` | T0.2, T2.1 | 1d | 与 T2.3 联合验收 |
| **T2.3** | 特征奇偶校验测试 | `tests/test_feature_parity.py` | T2.2 | 0.5d | ≥1000 随机状态逐位相等（obs + mask） |
| **T2.4** | 动作执行器 | `browser/action_executor.py` | T0.4, T2.1 | 2d | 四类动作 + 支付药丸 + 贵族在网页执行成功 |
| **T2.5** | BrowserSplendorEnv | `browser/browser_env.py` | T2.1-2.4 | 1.5d | 协议全实现；自主完成 ≥20 局 |
| **T2.6** | 会话管理 | `browser/session.py` | — | 1d | 建房/双身份/开始/恢复全流程 |
| **T2.7** | 掩码奇偶监控 | `browser/monitor.py` | T2.5 | 0.5d | 10 局差异为零或全部归因 |
| **T3.1** | play-web 入口 | `play_web.py` | T1.6, T2.5 | 1d | 网页 50 局稳定 |
| **T3.2** | 鲁棒性 | 同上 | T3.1 | 1~2d | 计时房间不超时；断线可恢复 |
| **T3.3** | sim-to-real 对照报告 | `docs/s2r_report.md` | T3.1 | 1d | 本地/网页胜率对照表 |
| **T5.1** | 测试与 CI | `tests/` + workflow | 各阶段 | 2d | 全部单测进 CI |

依赖图：`T0.* → {T1.*, T2.*}（并行）→ T3.* → P4（可选）`。关键路径：T0.4（网页实测）同时阻塞 T2.1/T2.4，**最先启动**。

### 0.2 本轮源码核实的承重事实（写代码前必读）

| # | 事实 | 出处 |
|---|---|---|
| F1 | `Action` dataclass 字段：`type_enum / collected_gems(dict\|None) / returned_gems(dict\|None) / position(CardPosition\|None) / noble_index(int\|None)`；`CardPosition(tier, card_index, reserved_index)`；两者均为**不可哈希** dataclass | `gym/envs/actions.py:42-66` |
| F2 | `Action.to_action_element(action, state, agent_index)` 需要 state（查 noble 下标与卡位置）；buy 动作把两个 gems 字段置 None | `actions.py:68-106` |
| F3 | `InputNormalization` 缓冲区形状 `(1, num_features)`，动量 0.9/0.1，`model.training` 时更新 | `ppo/input_norm.py:30-56` |
| F4 | PPO 隐藏层构造范式：`Linear + LayerNorm + Dropout + ReLU`（`PPOBase.create_hidden_layers`） | `ppo/ppo_base.py:71-90` |
| F5 | `ppo.py` 的入口结构：`train(**options)` + `main()`（argparse）、stats.csv、`save_model` 含 running_mean/var | `ppo/ppo.py:53-93,139-308` |
| F6 | `AgentState` 字段：`id/score/gems/cards/nobles/passed/agent_trace/last_action`；`cards["yellow"]` 存预留卡 | `splendor_model.py:120-129` |
| F7 | `turns_made_by_agent = len(agent.agent_trace.action_reward)`；`agent_buying_power = gems[color] + len(cards[color])`；`resources_sufficient` 只读 `agent.gems`（含 yellow）与 `len(agent.cards[colour])` —— **占位卡可满足** | `features.py:100-105,65-73`，`splendor_model.py:371-394` |
| F8 | 贵族表示为 `(code, cost)` 元组，`to_action_element` 用 `board.nobles.index(action["noble"])`（值相等即可匹配）；10 个贵族 **cost 全部唯一**（实测） | `splendor_model.py:83`，`splendor_utils.py:110-121` |
| F9 | `getLegalActions(state, agent_id)` 只读 `board.dealt / board.gems / board.nobles` 与该 agent 的 `gems/cards`——**不读 `board.decks`**，可在伪状态上运行 | `splendor_model.py:404-576` |
| F10 | gym 注册：`register(id="splendor-v1", entry_point="splendor.splendor.gym.envs:SplendorEnv")`，需 `import splendor.splendor.gym` 触发 | `gym/__init__.py:8-11` |

### 0.3 本轮新增的两项设计裁决

1. **浏览器层的合法动作掩码 = 引擎规则复用，而非 DOM 重算**。既然伪状态（§3.2）能喂给 `extract_metrics_with_cards`，它同样能喂给 `SplendorGameRule.getLegalActions(pseudo_state, my_id)`（依据 F9，不读牌库内容）。于是浏览器层**零规则代码复用**：`mask = create_legal_actions_mask(rule.getLegalActions(pseudo, id), pseudo, id)`；DOM 可交互元素只作**交叉验证信号**（T2.7 监控两侧差异，正是规则奇偶风险 R2 的探测器）。这比 BROWSER_RL_MAPPING §4.2"由 DOM 推导掩码"更稳——DOM 只能证明"可点"，不能证明"合法全集"。
2. **贵族注册表以 cost 为键**（F8 唯一性已验证），与卡注册表（四元组键）配套，网页 DOM 只需读出需求向量即可还原引擎的 `(code, cost)` 元组。

---

## 1. P0 地基模块规格

### T0.1 ActionIndexCache（`src/splendor/splendor/gym/envs/utils.py` 改造）

问题：`ALL_ACTIONS.index(x)` 是 O(3510) 线性扫描且依赖 `__eq__` 逐字段比较（`utils.py:162,175`），每步 × 每合法动作各一次。

```python
# gym/envs/utils.py 新增 ------------------------------------------------

def _gems_key(gems: GemsCount | None) -> tuple[tuple[str, int], ...] | None:
    """dict|None -> 可哈希键。None 与空 dict 必须区分（buy 动作为 None，无归还为空 dict）。"""
    return None if gems is None else tuple(sorted(gems.items()))


def action_key(a: Action) -> tuple:
    """Action -> 可哈希键，保持与 dataclass __eq__ 完全一致的判等语义。"""
    position_key = (
        (a.position.tier, a.position.card_index, a.position.reserved_index)
        if a.position is not None
        else None
    )
    return (a.type_enum, _gems_key(a.collected_gems), _gems_key(a.returned_gems),
            position_key, a.noble_index)


# 模块加载时构建一次：O(3510)，之后所有查表 O(1)
ACTION_INDEX: dict[tuple, int] = {action_key(a): i for i, a in enumerate(ALL_ACTIONS)}
assert len(ACTION_INDEX) == len(ALL_ACTIONS)  # 键碰撞 = ALL_ACTIONS 有重复，启动即暴露


def _index_of(action_element: Action) -> int:
    """替代 ALL_ACTIONS.index()；未命中时抛出带细节的 ValueError（旧实现是静默 mask=0，
    属于隐性 bug，新实现应尽早暴露引擎动作与 ALL_ACTIONS 的失配）。"""
    try:
        return ACTION_INDEX[action_key(action_element)]
    except KeyError as err:
        raise ValueError(f"action not in ALL_ACTIONS: {action_element}") from err
```

改造 `create_legal_actions_mask` / `create_action_mapping`：**签名与返回值不变**，内部 `ALL_ACTIONS.index(...)` 全部替换为 `_index_of(...)`。既有调用方（`splendor_env.py:146,184-186`、`ppo_agent.py:51,62`）零改动。

```python
# tests/test_action_index_cache.py ------------------------------------------------
def test_mask_and_mapping_equivalence():
    """随机生成 ≥50 个引擎状态，断言新旧实现输出完全一致（保留旧实现为 _slow_* 供测试）。"""

def test_cache_build_is_bijection():
    """len(ACTION_INDEX) == 3510 且键互异（构造期 assert 的运行时副本）。"""
```

### T0.2 Card/Noble Registry（`src/splendor/browser/card_registry.py` 新建）

```python
"""卡/贵族身份注册表：网页 DOM 四元组 -> 引擎 Card / (code, cost) 元组。

依据（已验证）：(tier, colour, points, cost) 在全部 90 张卡上零重复；
10 个贵族的 cost 全部唯一。tier 方向转换在此模块收口（网页行序上→下 = tier 2/1/0）。
"""
from splendor.splendor.splendor_model import Card
from splendor.splendor.splendor_utils import CARDS, NOBLES

CardKey = tuple[int, str, int, tuple[tuple[str, int], ...]]   # (deck_id, colour, points, sorted cost)
NobleKey = tuple[tuple[str, int], ...]                        # sorted cost

CARD_REGISTRY: dict[CardKey, Card] = {}       # 值为引擎同构 Card(colour, code, cost, deck_id, points)
NOBLE_REGISTRY: dict[NobleKey, tuple[str, dict]] = {}

def _build_registries() -> None:
    """模块加载时从 CARDS/NOBLES 构建；code 串取自 CARDS 键，保证 Card 与引擎逐字段一致。"""
    # TODO: for code, (colour, cost, deck_id, points) in CARDS.items(): ...

def card_key(deck_id: int, colour: str, points: int, cost: dict[str, int]) -> CardKey: ...
def noble_key(cost: dict[str, int]) -> NobleKey: ...
def lookup_card(deck_id: int, colour: str, points: int, cost: dict[str, int]) -> Card:
    """未命中抛 KeyError（含输入四元组，便于排查 DOM 抽取 bug）。"""
def lookup_noble(cost: dict[str, int]) -> tuple[str, dict]: ...
```

```python
# tests/test_card_registry.py ------------------------------------------------
def test_all_cards_hit():        # 90/90 命中且键唯一
def test_all_nobles_hit():       # 10/10 命中且 cost 唯一
def test_card_fields_match_engine():
    """registry 还原的 Card 与 initialGameState 中同名卡逐字段相等（含 code）。"""
```

### T0.3 环境协议（`src/splendor/splendor/gym/base.py` 新建）

```python
"""SplendorEnvBase：本地 SplendorEnv 与 BrowserSplendorEnv 的统一契约。"""
from typing import Protocol, runtime_checkable
import gymnasium as gym
import numpy as np

@runtime_checkable
class SplendorEnvBase(Protocol):
    observation_space: gym.spaces.Box      # shape (265,), dtype float32
    action_space: gym.spaces.Discrete      # 3510

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]: ...
        # 返回 (obs(265,), {"my_id": int})

    def step(self, action: int, payment: int | None = None
             ) -> tuple[np.ndarray, float, bool, bool, dict]: ...
        # obs, reward, terminated, truncated, info
        # payment：P4 支付维度前瞻参数，tier-1 实现恒忽略（见 §5）

    def get_legal_actions_mask(self) -> np.ndarray: ...
        # shape (3510,) 的 0/1 数组；必须在每次 reset/step 后重新调用

    def get_payment_options(self, action: int) -> list[dict] | None: ...
        # tier-1 恒返回 None；tier-2 与浏览器层返回该动作的支付方案列表（§5）
```

注意：`SplendorEnv.step` 现签名是 `step(self, action: int)`——为其加上 `payment: int | None = None` 的兼容参数（默认忽略），即满足协议。

### T0.4 网页规则实测（实验协议，人工 + 浏览器自动化）

六项实验，每项记录：操作序列、DOM 前后快照、结论（与引擎一致/不一致）、截图。产出 `docs/web_experiments.md`。

| # | 实验 | 判定问题 | 影响的下游 |
|---|---|---|---|
| E1 | 多贵族选择 UI | 选择器结构、默认行为、能否不选 | `action_executor` 贵族分支 |
| E2 | >10 宝石返还 | 返还子流程点击序列、能否返还刚拿的颜色 | `action_executor` 返还分支 |
| E3 | 终局结算 DOM | 结束信号选择器、比分展示 | `browser_env` 终局检测 |
| E4 | 空牌库 reserve | 是否仍发金 | 执行器 + 引擎一致性（引擎：`collected_gems = {"yellow":1} if board.gems["yellow"]>0 else {}`，与牌库无关——若网页不同则记差异） |
| E5 | **自愿少拿宝石** | 手 ≤7 颗时能否只拿 1 个（标准规则）vs 引擎强制 ≥min(3,可用色) | **R2 规则奇偶**：若网页允许少拿 → 引擎 `getLegalActions` 加 `standard_take_rules` 开关 |
| E6 | 同色 7 张购卡上限 | 网页是否存在 | 同上 |

---

## 2. P1 DQN 模块规格

目录：`src/splendor/agents/our_agents/dqn/`（文件清单与 DQN_GUIDE §6.1 一致，此处给出精确函数框架）。

### T1.1 `network.py`

```python
"""Dueling DQN 网络。trunk 沿用 PPO 范式（F4）但去 Dropout；掩码手法与 PPO 一致。"""
import torch
from torch import nn

from splendor.splendor.features import METRICS_WITH_CARDS_SIZE
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

OBS_DIM = METRICS_WITH_CARDS_SIZE        # 265
ACTION_DIM = len(ALL_ACTIONS)            # 3510
HUGE_NEG = -1e9


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = OBS_DIM, output_dim: int = ACTION_DIM,
                 hidden_layers: tuple[int, ...] = (128, 128, 128, 128),
                 use_input_norm: bool = True, dueling: bool = True) -> None:
        """InputNormalization(F3 兼容，buffer 形状 (1,265)) → Linear+LayerNorm+ReLU ×N
        → V 头 Linear(h,1) + A 头 Linear(h,3510)；dueling=False 时退化为单头。"""
        ...

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """输入 obs (B,265) float、mask (B,3510) 0/1；返回掩码后 Q (B,3510)。
        非法动作置 HUGE_NEG——argmax 天然合法；训练只 gather 合法动作，
        HUGE_NEG 不进损失；bootstrap max 掩码后必落合法动作。"""
        ...

    @torch.no_grad()
    def act(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        """单状态贪心动作索引（评测/部署用）。"""
        ...

    def raw_q(self, obs: torch.Tensor) -> torch.Tensor:
        """未掩码 Q（调试/T4 支付头复用 trunk 特征）。"""
        ...
```

### T1.2 `replay_buffer.py`

```python
"""均匀采样 Replay Buffer，n-step 折叠由前置滑动队列实现。float32 存储。"""
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int = OBS_DIM,
                 action_dim: int = ACTION_DIM, n_step: int = 1, gamma: float = 0.99) -> None: ...

    def add(self, obs, action: int, reward: float, next_obs, next_mask, done: bool) -> None:
        """n_step=1 直存；n_step>1 滑动窗口折叠：R=Σγ^k r_k，s' 取队尾；
        窗口中途 done 则截断收尾（done=1）。注意：**当前步 mask 不存**
        ——gather 只取已执行动作（必合法），无需当前掩码。"""
        ...

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """返回 (obs, action, reward, next_obs, next_mask, done)，dtype 齐整可直接进网络。"""
        ...

    def __len__(self) -> int: ...
```

```python
# tests/test_replay_buffer.py
def test_nstep_discount_sum():      # 手工构造 reward 序列断言折扣累计
def test_nstep_done_truncation():   # 窗口中途 done 的截断语义
def test_wraparound():              # 环形覆盖后采样正确
```

### T1.3 `reward_wrapper.py`

```python
"""终局胜负奖励包装（DQN_GUIDE §5.4）。胜负口径 = calScore（含 +0.5 买卡数平局裁定）。"""
import gymnasium as gym

class TerminalRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, win_bonus: float = 10.0) -> None: ...

    def step(self, action, payment: int | None = None):
        """reward = Δscore + win_bonus·(+1/0/-1)|terminal。
        终局判定用 env.unwrapped.{state, my_turn, game_rule}（属性名已核实）。"""
        ...

    def reset(self, *, seed=None, options=None):
        """透传 reset，记录 my_id 供终局判定。"""
        ...
```

### T1.4 `training.py`

```python
"""DQN 训练核心：采集循环 + Double DQN 更新 + 评估。"""
from dataclasses import dataclass

@dataclass
class DQNParams:
    gamma: float; lr: float; batch_size: int; warmup: int
    eps_start: float; eps_end: float; eps_decay_steps: int
    tau: float; target_update_freq: int   # 软更新 τ；freq>0 时改硬更新
    max_grad_norm: float = 1.0
    seed: int = 42


def epsilon_at(step: int, p: DQNParams) -> float:
    """线性衰减；随机动作用 np.random.choice(np.flatnonzero(mask)) 掩码内均匀采样。"""
    ...

def dqn_update(q_net, target_net, buffer, optimizer, p: DQNParams) -> dict:
    """一次梯度步（Double DQN）：
    1) 采样 batch（含 next_mask）；
    2) with no_grad: a* = argmax q_net(s', m')；y = r + γ·(1-done)·q_target(s',m')[a*]；
    3) q_pred = q_net(s, 全1掩码).gather(a)   ← 合法性由数据保证；
    4) loss = smooth_l1(q_pred, y)；梯度裁剪；软/硬更新 target。
    返回 {"loss", "q_mean", "td_abs_mean"} 供 stats.csv。"""
    ...

def collect_one_step(env, q_net, buffer, p: DQNParams, step: int) -> dict:
    """ε-greedy 采集一步并入库；terminated 时 reset。返回 episode 统计增量。
    随机性三件套在 train() 入口统一 seed（DQN_GUIDE §2.4）。"""
    ...

@torch.no_grad()
def evaluate(env, q_net, n_games: int = 20) -> dict:
    """贪心策略打 n 局（每局独立 gym.make 保证随机座位/发牌），
    终局用 calScore 口径统计 {"win","draw","loss","avg_score"}。"""
    ...
```

### T1.5 `dqn.py` / `dqn_agent.py` / `utils.py`

```python
# dqn/dqn.py —— 结构完全镜像 ppo.py（F5） -----------------------------------
def train(working_dir: Path, learning_rate: float, seed: int, device_name: str,
          opponent: str, test_opponent: str,
          total_steps: int, buffer_size: int, win_bonus: float, ...) -> QNetwork:
    """对手工厂复用 ppo/arguments_parsing.OPPONENTS_AGENTS_FACTORY（random/minimax/...）；
    'itself' 自博弈：对手 = DQNAgent(load_net=False) + load_policy 共享网络（镜像 ppo.py:177-184），
    并给对手挂经验回收 wrapper（DQN_GUIDE §7.2 进阶版，M4 再启用）。
    stats.csv 字段：step, episode, epsilon, loss, q_mean, train_score, eval_wr, eval_avg_score。
    checkpoint 命名 dqn_model_{step}.pth，每 N 步一存（修掉 ppo.py:293 的优先级 bug：
    `f"dqn_model_{(step // save_every):d}.pth"`）。"""
    ...

def main() -> None:   # argparse（镜像 ppo 的 parse_args），console script `dqn`
    ...

# dqn/dqn_agent.py —— 镜像 ppo_agent.py --------------------------------------
class DQNAgent(Agent):
    def __init__(self, _id: int) -> None:
        """device 解析 + load_saved_dqn() + eval()（镜像 PPOAgentBase，ppo_agent_base.py:22-27）。"""
    def SelectAction(self, actions, game_state, game_rule) -> ActionType:
        """镜像 ppo_agent.py:32-64：extract_metrics_with_cards → create_legal_actions_mask
        → net.act → create_action_mapping[...]。"""
    ...

myAgent = DQNAgent

# dqn/utils.py —— 镜像 ppo/utils.py -------------------------------------------
def save_model(model: QNetwork, path: Path) -> None:
    """{"model_state_dict", "running_mean", "running_var", "step", "config"}；
    running stats 形状保持 (1, 265)（与 ppo checkpoint 约定一致，F3）。"""
def load_saved_dqn(path: Path | None = None) -> QNetwork:
    """默认加载包内 dqn_model.pth；running_mean/var 依 input_norm 存在与否分别处理
    （镜像 ppo/utils.py:36-47 的 squeeze(0) 逻辑）。"""
```

`pyproject.toml`：`[project.scripts] dqn = "splendor.agents.our_agents.dqn.dqn:main"`。

### T1.6 训练课程（挂机任务）

| 阶段 | 对手 | 步数预算 | 门槛（不过不进下一阶段） |
|---|---|---|---|
| M1 | random | 2×10⁵ | 平均终局得分 >10 |
| M2 | random(50%)+minimax(50%) | 3×10⁵ | vs random 胜率 >90%（20 局滑窗） |
| M3 | 三者混合 + 快照自博弈 | 5×10⁵ | vs minimax 100 局 ≥55% |

每次实验固定 3 个 seed，记录均值±方差（防单 seed 虚假结论）。

---

## 3. P2 浏览器模块规格

目录：`src/splendor/browser/`。**驱动抽象先行**——不绑定具体浏览器工具（ego-browser / playwright / CDP 均可替换）：

```python
# browser/driver.py -----------------------------------------------------------
class BrowserDriver(Protocol):
    def evaluate(self, js: str) -> Any: ...            # 执行 JS 并返回 JSON 可序列化结果
    def click(self, selector: str, index: int = 0) -> None: ...
    def wait_for(self, condition_js: str, timeout: float) -> None: ...
    def navigate(self, url: str) -> None: ...
    def get_cookies(self, domain: str) -> list[dict]: ...
    def set_cookie(self, cookie: dict) -> None: ...
    def delete_cookies(self, name: str, domain: str) -> None: ...  # 双域删除坑见 BROWSER §2
    def screenshot(self, path: str) -> None: ...
```

### T2.1 `dom_extractor.py`

```python
"""DOM 快照抽取（BROWSER_RL_MAPPING §3.1 schema 的代码化）。"""
from typing import TypedDict

class CardInfo(TypedDict):
    tier: int; colour: str; points: int; cost: dict[str, int]
class NobleInfo(TypedDict):
    requirements: dict[str, int]        # 只需需求向量，身份走 NobleRegistry（§0.3-2）
class PanelInfo(TypedDict):
    seat: int; score: int
    card_counts: dict[str, int]         # 每色永续卡数（rect）
    gems: dict[str, int]                # 6 circle（含金）
    reserved_tiers: list[int]           # 卡背泄露的 tier（信息可用可不用）
class Snapshot(TypedDict):
    dealt: list[list[CardInfo | None]]  # 3×4，tier 0..2（行序转换在此收口）
    deck_counts: list[int]
    nobles: list[NobleInfo]
    supply: dict[str, int]              # 公共宝石供给
    panels: list[PanelInfo]             # 按座位
    my_seat: int
    my_reserved: list[CardInfo]         # ≤3，明牌
    status: str                         # 等待你操作 / 等待玩家N操作 / 终局文案（E3 定）
    payment_options: list[str] | None   # 待定支付药丸文本（"3白"/"2白1金"）
    noble_options: list[NobleInfo] | None

EXTRACT_SNAPSHOT_JS: str = ...   # 单次 evaluate 注入的 IIFE，输出与 Snapshot 对齐的 JSON

def extract_snapshot(driver: BrowserDriver) -> Snapshot:
    """执行 EXTRACT_SNAPSHOT_JS 并做 schema 校验（缺字段/类型错即抛，快速定位页面改版）。"""
```

离线夹具：`fixtures/*.html` 保存真实局面（含待定支付、终局等特殊局面），测试用无头浏览器加载后走同一 `extract_snapshot` 断言输出。

### T2.2 `state_builder.py`

```python
"""DOM 快照 -> 引擎同构伪状态。特征与合法动作全部经由引擎代码计算（§0.3-1）。"""
from splendor.splendor.splendor_model import SplendorState, Card

def build_pseudo_state(snapshot: Snapshot, my_index: int) -> SplendorState:
    """构造要点（字段需求 = F6/F7/F9）：
    - 用 object.__new__(SplendorState) 绕过 BoardState 的随机初始化；
    - board.dealt: 3×4，每张 lookup_card(...) 还原真实 Card（registry）；
    - board.gems: snapshot["supply"]；board.nobles: [lookup_noble(n["requirements"]) ...]；
    - agents[my_index]: score/gems 直填；cards[色] 用占位 Card 填到面板数量
      （F7：规则与特征只取 len）；cards["yellow"] = [lookup_card(...) for 我的预留明牌]；
      agent_trace = 简单对象，action_reward = [None] * 自维护回合数；
    - agents[对手]: 只需 id/score（特征只读对手分数；getLegalActions 只算我的动作）。
    """
```

### T2.3 特征奇偶校验测试（`tests/test_feature_parity.py`，P2 质量门）

```python
def project_visible(state: SplendorState, my_index: int) -> Snapshot:
    """引擎状态 -> Snapshot（DOM 能看到的一切）。测试专用，也是夹具生成器。
    刻意只暴露：桌面 12 卡、贵族需求、供给、双方面板、我的预留明牌、状态。"""

# tests/test_feature_parity.py
def test_obs_and_mask_parity():
    """随机对局 ≥1000 决策点，逐点断言：
    1) extract_metrics_with_cards(state, i) == extract_metrics_with_cards(build_pseudo_state(project_visible(state, i), i), i)  # np.allclose 逐位
    2) create_legal_actions_mask(rule.getLegalActions(state,i), state, i)
       == create_legal_actions_mask(rule.getLegalActions(pseudo,i), pseudo, i)      # 引擎规则在伪状态上可复用（F9）的直接证据
    失败时打印首个不匹配维度与两侧值，定位到特征段落。"""
```

### T2.4 `action_executor.py`

```python
"""动作索引 -> 网页点击序列（BROWSER_RL_MAPPING §4.1）。"""
HUMAN_CLICK_DELAY: tuple[float, float] = (0.2, 0.5)   # 礼仪：人类节奏（含抖动）

class ActionExecutor:
    def __init__(self, driver: BrowserDriver) -> None: ...

    def execute(self, action_index: int, snapshot: Snapshot,
                pseudo_state: SplendorState) -> None:
        """ALL_ACTIONS[action_index] -> 分支：
        PASS:           放弃 → 确认放弃
        COLLECT_*:      取宝石 → 点 collected_gems 各色筹码 → 确认拿这些
                        （若引擎动作含 returned_gems → E2 实测的返还子流程）
        RESERVE:        预定 → 点 position 对应卡的"预定"按钮
        BUY_*:          购买 → 点 position 对应卡（BUY_RESERVE 走我的预留区）的"购买"
                        → 若弹出支付药丸：select_payment_greedy()
        动作含 noble_index 且页面出现贵族选择（E1）：点对应贵族。
        每步之间 driver.wait_for 确认 UI 状态迁移，失败重试 1 次后抛 ActionExecutionError。"""
        ...

    def select_payment_greedy(self, pills: list[str], action: Action,
                              pseudo_state: SplendorState) -> int:
        """tier-1 支付策略：模拟引擎贪心（resources_sufficient 语义：彩色优先、金补差，
        F7/splendor_model.py:371-394）——选"金用量最少"的药丸，保持与本地训练分布一致。
        药丸文本解析："2白1金" -> {"white":2,"yellow":1}。"""
        ...

def parse_pill(pill_text: str) -> dict[str, int]: ...
```

### T2.5 `browser_env.py`

```python
"""BrowserSplendorEnv：网页座位包装为 SplendorEnvBase（§0.3-1：掩码来自引擎规则复用）。"""
import gymnasium as gym

class BrowserSplendorEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (265,), np.float32)
    action_space = gym.spaces.Discrete(3510)

    def __init__(self, driver: BrowserDriver, session: SessionManager,
                 rule: SplendorGameRule | None = None, poll_interval: float = 0.4,
                 step_timeout: float = 120.0) -> None: ...

    def reset(self, *, seed=None, options=None):
        """session.new_game()（建房/重开）→ 等待开局 → 快照 → 返回 (obs, {"my_id": seat})。"""
        ...

    def step(self, action: int, payment: int | None = None):
        """1) executor.execute(action)；
        2) 轮询 status 直至"等待你操作"或终局（poll_interval，step_timeout 超时先降级点放弃保回合）；
        3) reward = 面板"N分"差分（= 引擎 action_reward 语义）；
        4) terminated 由 E3 定的终局特征判定；truncated 恒 False。
        返回 (obs, reward, terminated, truncated, info)。"""
        ...

    def get_legal_actions_mask(self) -> np.ndarray:
        """§0.3-1：build_pseudo_state → rule.getLegalActions(pseudo, my_id)
        → create_legal_actions_mask(...)（走 T0.1 缓存）。
        DOM 可交互元素同时收集为 affordance 集合，交 monitor 交叉验证。"""
        ...

    def get_payment_options(self, action: int) -> list[dict] | None:
        """tier-1 返回 None；若页面存在待定药丸则返回解析结果（供 §5 tier-2 / 调试）。"""
        ...
```

### T2.6 `session.py` 与 T2.7 `monitor.py`

```python
# browser/session.py -----------------------------------------------------------
class SessionManager:
    def create_room(self, seats: int = 2) -> str: ...          # 返回房间 URL
    def join_seat(self, seat: int) -> None: ...
    def start_game(self) -> None: ...                           # 需 ≥2 入座
    def new_game(self) -> None: ...                             # reset 用：重开或 rematch
    # 双身份自博弈（BROWSER §2 配方）：本类只管"我方"身份；
    # 对面座位由另一进程/另一 driver 实例的 SessionManager+Env 驱动。
    def switch_identity(self) -> None: ...
        """deleteCookies(gid) 双域清理（game.hullqin.cn 与 .game.hullqin.cn）→ 服务器分配新 gid。
        注意切换时序：已在页面的 ws 连接身份在握手时固定，必须先载入再换 cookie。"""
    def recover(self) -> None: ...                              # 断线/异常房间重建

# browser/monitor.py -------------------------------------------------------------
class MaskParityMonitor:
    def check(self, engine_mask: np.ndarray, dom_affordances: set[int]) -> list[str]:
        """两侧差异归因：规则差异（E5/E6）/ DOM 抽取 bug / 页面改版。
        输出人类可读报告；T2.7 验收 = 10 局零未解释差异。"""
```

---

## 4. P3 部署 Harness 规格

```python
# src/splendor/play_web.py ------------------------------------------------------
"""`play-web` console script：DQN checkpoint + BrowserSplendorEnv 网页对局。"""

def load_agent(checkpoint: Path) -> QNetwork: ...          # dqn.utils.load_saved_dqn

def run_game(env: BrowserSplendorEnv, q_net: QNetwork, max_steps: int = 200) -> GameReport: ...
    """贪心策略打一局；GameReport: {"result": win/draw/loss, "my_score", "rival_score",
    "steps", "duration", "mask_anomalies": int}。"""

def main() -> None:
    """argparse: --checkpoint --games N --seats 2 --room-url --poll ...
    循环 run_game → 汇总胜率/得分 → 写 JSON 报告（docs/s2r_report.md 的数据源）。
    每局之间随机休息 5~15s（礼仪）；任何一局异常先 session.recover() 重试一次。"""
```

`pyproject.toml`：`scripts.play-web = "splendor.play_web:main"`。

**T3.3 对照实验设计**：同一 checkpoint，四个战场各 ≥50 局——本地 vs random / 本地 vs minimax / 网页 vs 脚本座席 / 网页 vs 人类。产出落差分解表（支付方式 / 对手分布 / 观测噪声各自贡献多少，用 P4 的开关逐项关闭来归因）。

---

## 5. P4 接口预留：支付维度

tier-2 才实施，但 **T0.3 协议已把接口留好**（`step(payment=...)`、`get_payment_options()`）。规格如下，供实施时直接照抄：

```python
# splendor_model.py 侧（新子类，不改动默认行为） --------------------------------
class PaymentEnumeratingGameRule(SplendorGameRule):
    def enumerate_payments(self, agent, card_cost) -> list[dict]:
        """对单卡枚举全部合法支付：逐色 k_c ∈ [max(0, cost_c - wild), min(cost_c, 可用_c)]
        的笛卡尔积，约束 Σ 黄金用量 ≤ agent.gems["yellow"]。返回 returned_gems 列表。"""

# dqn/network.py 侧：支付辅助头 -------------------------------------------------
class PaymentHead(nn.Module):
    """输入 trunk 隐层特征 + 所选 buy 动作的卡编码，输出 K(=32) 维支付 Q 值，
    掩码来自 env.get_payment_options(action)。仅 len(options)>1 时参与决策。"""

# replay/训练侧：transition 增 payment 字段；对手支付策略随机化 ------------------
```

验收：消融实验 tier-2 ≥ tier-1（本地 + 网页双战场）。

---

## 6. 测试矩阵与 CI

| 测试 | 层 | 依赖网络 | 阶段 |
|---|---|---|---|
| `test_action_index_cache` | 单元 | 否 | T0.1 |
| `test_card_registry` | 单元 | 否 | T0.2 |
| `test_replay_buffer` | 单元 | 否 | T1.2 |
| `test_dqn_update`（手工构造 TD 目标断言） | 单元 | 否 | T1.4 |
| `test_dqn_smoke`（2000 步训练 + loss 有限） | 集成 | 否 | T1.4 |
| `test_feature_parity`（**核心质量门**） | 集成 | 否 | T2.3 |
| `test_browser_adapter`（离线夹具 → snapshot → pseudo → obs） | 集成 | 否（夹具） | T2.1-2.2 |
| `test_mask_parity_monitor`（注入差异能否归因） | 单元 | 否 | T2.7 |
| 网页实测脚本（E1-E6） | 手动/半自动 | 是 | T0.4 |

CI（GitHub Actions）：`ruff + mypy + pytest`（上表"依赖网络=否"全量）；Python 3.12（F10 + `typing.override` 约束）。Makefile：`make lint / test / parity / train-dqn / play-web`。

---

## 7. 编码约定

1. **镜像优先**：`dqn/` 各文件与 `ppo/` 对应文件保持结构与命名风格一致（train/main/stats.csv/save_model），降低维护者的上下文切换成本。
2. **不破坏存量**：引擎与 PPO 路径零行为变化；`utils.py` 掩码函数签名不变；唯一可见差异是非法索引从"静默 mask=0"变为显式 ValueError（这是修 bug，在 CHANGELOG 注明）。
3. **浏览器层零规则代码**：一切合法性判断走 `getLegalActions(pseudo_state, ...)`；DOM 只做观测与执行（§0.3-1）。发现引擎与网页规则不一致时，改引擎（加开关）而非在浏览器层打补丁。
4. **类型注解全量**：新代码 mypy strict 通过；dataclass/TypedDict 优先。
5. **随机性纪律**：任何新入口（训练/评测/部署）开头的 seed 三件套（`random/np.random/torch`）不可省略（DQN_GUIDE §2.4：`env.reset(seed=)` 不固定发牌）。
