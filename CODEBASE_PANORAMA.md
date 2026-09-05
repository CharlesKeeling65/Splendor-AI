# Splendor-AI 初始代码库全景图

> **范围**：本仓库**升级改造开始前**的原始代码库（main 分支 `0172d31`）。全部函数与模块的完整盘点，行号以当前源码为准。
> **数据来源**：AST 程序化提取（59 个 Python 文件、7,415 行、223 个方法/函数）+ 逐文件人工核对语义。
> 配套文档：[AGENTS.md](./AGENTS.md)（工作指南）· [plan/README.md](./plan/README.md)（升级计划）。

---

## 0. 总览数字

| 维度 | 数值 |
|---|---|
| Python 文件 | 59（48 个含代码 + 11 个空 `__init__.py`） |
| Python 总行数 | 7,415 |
| 类 | 41 个（含嵌套类与 TypedDict） |
| 方法/函数 | 223 个（含 `__init__`） |
| 游戏资产 | 90 张卡 + 10 个贵族 + 6 色宝石的 GUI 图片（大小两套，~300 个 png） |
| 已训权重 | PPO×4（MLP 4.1M / GRU 4.4M / LSTM 4.6M / SelfAttn 6.3M）+ GA 基因×4（.npy） |
| Console 命令 | `splendor`（GUI 评测）、`splendor`（文本评测 `-t`）、`ppo`（训练）、`evolve`（GA 训练） |

## 1. 分层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│ F 对局运行层   general_game_runner.py(694) game.py(258)             │
│               CLI 评测器 / 对局主循环 / 回放器                        │
├─────────────────────────────────────────────────────────────────────┤
│ E 可视化层     splendor_displayer.py(754)                            │
│               tkinter GUI 显示器 + 文本显示器（含人机交互输入）        │
├─────────────────────────────────────────────────────────────────────┤
│ G Agent 层    agents/                                                │
│  generic/      random / first_move / timeout（3 个基线，~30 行/个）    │
│  minmax.py(139)          深度 2 alpha-beta 搜索                       │
│  our_agents/ppo/(22 文件) PPO 家族：MLP/GRU/LSTM/SelfAttn 四架构       │
│  our_agents/genetic_algorithm/(5 文件) 遗传算法                       │
├─────────────────────────────────────────────────────────────────────┤
│ D Gym 环境层   splendor/gym/envs/                                    │
│  actions.py(237)         ALL_ACTIONS 固定枚举（3510 个动作）           │
│  splendor_env.py(245)    SplendorEnv（对手自动模拟、掩码、Δscore 奖励）│
│  utils.py(181)           掩码/索引映射/动作构造                        │
├─────────────────────────────────────────────────────────────────────┤
│ C 特征层       splendor/features.py(487)                             │
│               265 维观测 = 70 维指标 + 15 张卡 × 13 维                 │
├─────────────────────────────────────────────────────────────────────┤
│ B 游戏引擎层   splendor/splendor_model.py(579)  规则核心（完全信息博弈）│
│  splendor_utils.py(245)  90 卡定义/10 贵族/AgentTrace/字符串渲染       │
│  constants.py(31) types.py(66) utils.py(25, 100 回合上限规则)          │
├─────────────────────────────────────────────────────────────────────┤
│ A 框架基座     template.py(89)  GameState/GameRule/Agent/Displayer 基类│
│               utils.py(13) version.py(14)                            │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 模块依赖图

```
general_game_runner.py ──┬─→ game.py ──┬─→ template.py（GameRule 轮转 / Agent / Displayer 接口）
   （CLI 入口，动态加载）  │             ├─→ splendor_model.py ─┬─→ splendor_utils.py（CARDS/NOBLES/COLOURS）
                         │             │                      ├─→ constants.py / types.py
                         │             └─→ splendor_displayer.py（GUI/Text，读 resources/ 图片）
                         └─→ agents/**（import myAgent）

ppo.py（训练入口）──→ gym.make("splendor-v1")
                        └─→ gym/envs/splendor_env.py ──┬─→ features.py ──→ splendor_model.py（取状态字段）
                                                      ├─→ actions.py（ALL_ACTIONS）
                                                      └─→ utils.py（掩码/映射）

ppo 家族：ppo.py → training.py → {rollout.py, common.py, network.py→ppo_base.py→input_norm.py}；utils.py（存取权重）
          ppo_agent*.py → ppo_agent_base.py → ppo/utils.py（加载权重）
          gru/lstm → recurrent_ppo.py（RecurrentPPO 基类）

evolve.py（GA 入口）──→ genes.py + genetic_algorithm_agent.py ──→ features.py（normalize_metrics）+ splendor_model.py（generateSuccessor 1 层前瞻）

minmax.py ──→ splendor_model.py（generateSuccessor / generatePredecessor 原地搜索）
```

---

## 3. 逐模块文档

### A. 框架基座层

#### `src/splendor/template.py`（89 行）— 所有抽象的根

| 成员 | 行 | 说明 |
|---|---|---|
| `GameState` | 8 | 占位基类（`__init__` 空实现） |
| `Action` | 13 | 占位类 |
| `GameRule` | 17 | **框架的回合制引擎骨架**：持有 `current_game_state` / `current_agent_index` / `action_counter` |
| `GameRule.update(action)` | 47 | 主循环推进：`generateSuccessor` + `getNextAgentIndex` 轮转（`(i+1) % n`）+ 计数 |
| `GameRule.getNextAgentIndex()` | 32 | 座位轮转取模 |
| `GameRule` 其余方法 | 24-55 | `initialGameState / generateSuccessor / getLegalActions / calScore / gameEnds / getCurrentAgentIndex`，全部 `raiseNotDefined` 留给子类 |
| `Agent` | 59 | **Agent 接口基类**：`__init__(self, _id)` 持 `self.id`；`SelectAction(actions, game_state, game_rule)` 默认 `random.choice` |
| `Displayer` | 71 | 显示器接口：`InitDisplayer / ExcuteAction / TimeOutWarning / EndGame` |

> 意义：课程框架的"合同层"。引擎、Agent、显示器全部以此为基类；`update()` 是整个对局循环的心脏。

#### `src/splendor/utils.py`（13 行）、`version.py`（14 行）

`raiseNotDefined()`（utils.py:7，占位异常）与 `get_version()`（version.py:10，版本号读取）。

### B. 游戏引擎层

#### `src/splendor/splendor/splendor_model.py`（579 行）— 规则核心，全仓库最重要的文件

| 成员 | 行 | 说明 |
|---|---|---|
| `Card` | 24 | 卡牌值对象：`colour / code（费用码，全局唯一）/ cost(dict) / deck_id(0-2) / points`；`__eq__` 按 code+points，**不可哈希** |
| `SplendorState` | 52 | 游戏状态：`board: BoardState` + `agents: list[AgentState]` + `agent_to_move`；`private_information = None`（完全信息） |
| `└ BoardState` | 68 | `decks`（3 层牌库，初始共 90 张 40/30/20，开局洗牌后各发 4 张）、`dealt`（3×4 桌面明牌矩阵）、`gems`（公共宝石，2/3/4 人局各色 4/5/7，yellow 恒 5）、`nobles`（从 10 个中抽人数+1 个，`(code, cost)` 元组）；`deal(deck_id)` 从牌库顶补牌 |
| `└ AgentState` | 120 | `id / score / gems / cards(dict 颜色→list，**"yellow" 槽存预留卡**) / nobles / passed / agent_trace / last_action` |
| `SplendorGameRule` | 142 | 规则引擎主体 |
| `· initialGameState()` | 153 | 新建 `SplendorState` |
| `· generatePredecessor(state, action, agent_id)` | 156 | `generateSuccessor` 的**精确逆操作**（回滚宝石/卡/贵族/分数/trace）——搜索类算法免 deepcopy 的关键 |
| `· generateSuccessor(state, action, agent_id)` | 222 | **原地修改**状态并返回同一对象：执行拿宝石/预留/买卡 + 超限归还 + 贵族来访 + 补牌 |
| `· gameEnds()` | 299 | 有人 ≥15 分**且轮次转回 0 号位**，或全员 passed（死锁） |
| `· calScore(state, agent_id)` | 308 | 计分；**平分时买卡数（非预留）少者 +0.5** ——引擎的官方胜负裁定口径 |
| `· generate_return_combos(current_gems, collected_gems)` | 329 | 枚举 >10 颗时的归还组合（不允许归还刚拿的颜色） |
| `· resources_sufficient(agent, costs)` | 371 | 可负担性判定 + **唯一贪心支付方案**（永久卡折扣→彩色宝石→yellow 补差）；不可负担返回 False |
| `· noble_visit(agent, noble)` | 397 | 贵族来访判定（只看永久卡数量） |
| `· getLegalActions(game_state, agent_id)` | 404 | **合法动作生成的唯一入口**：collect_diff（含强制最少拿取：≤7 颗 min(3,可用色)、=8 颗 min 2、≥9 颗 min 1）→ collect_same（该色供给 ≥4）→ reserve（预留 <3 张）→ buy（12 明牌 + 3 预留，同色 7 张上限）→ 无动作时 pass；每类 × 归还组合 × 贵族组合。典型每步 15~40 个，实测 max 136 |

#### `src/splendor/splendor/splendor_utils.py`（245 行）— 游戏数据与渲染

| 成员 | 行 | 说明 |
|---|---|---|
| `CARDS` | 15 | **90 张卡的定义表**（dict：费用码 → `(colour, cost, deck_id, points)`；tier 1/2/3 = 40/30/20） |
| `NOBLES` | 110 | 10 个贵族 `(code, cost)`，各 3 分，cost 全部唯一 |
| `COLOURS` | ~15 | 颜色名映射（黑红黄绿蓝白） |
| `convert_filename(filename)` | 138 | GUI 图片文件名 ↔ 卡牌码转换 |
| `AgentTrace` | 158 | `pid` + `action_reward` 列表（每步 `(action, reward)` 历史；`len()` 即回合数） |
| `GemsToString / ActionToString / AgentToString / BoardToString` | 166-244 | 文本渲染（TextDisplayer 使用） |

#### `src/splendor/splendor/constants.py`（31 行）— 游戏常数

`WINNING_SCORE_TRESHOLD=15`、`MAX_SCORE=22`、`MAX_GEMS=10`、`MAX_RESERVED=3`、`MAX_WILDCARDS=5`、`ROUNDS_LIMIT=100`、`MAX_RIVALS=3`、`MAX_NOBLES=5`、`MAX_TIER_CARDS=4`、`NUMBER_OF_TIERS=3`、`NORMAL_COLORS`（5 色）、`RESERVED="yellow"`。

#### `src/splendor/splendor/types.py`（66 行）— 引擎动作的 TypedDict

`YellowGemCount / CollectAction / ReserveAction / BuyAction`（引擎 dict 格式的类型定义）；`ActionType = CollectAction | ReserveAction | BuyAction`。

#### `src/splendor/splendor/utils.py`（25 行）

| 成员 | 行 | 说明 |
|---|---|---|
| `LimitRoundsGameRule(SplendorGameRule)` | 9 | `gameEnds()` 叠加**100 回合强制截断**（Gym 环境用它防止随机对局无限拖延） |

### C. 特征层

#### `src/splendor/splendor/features.py`（487 行）— 状态 → 265 维向量

| 函数 | 行 | 说明 |
|---|---|---|
| `get_agent(state, i)` | 55 | 取玩家状态 |
| `agent_buying_power(agent)` | 65 | 各色购买力 = `gems[color] + len(cards[color])` |
| `diminish_return(value)` | 77 | 对数收益递减变换 |
| `agent_won(agent)` | 90 | `score >= 15` |
| `turns_made_by_agent(agent)` | 100 | `len(agent.agent_trace.action_reward)` |
| `missing_gems_to_card(card, buying_power)` | 107 | 到某卡各色缺口 |
| `turns_to_buy_card(missing_gems)` | 124 | 预计回合数距离 |
| `build_array(base_array, instruction)` | 138 | 按 shape 指令重复填充（GA 基因展开用） |
| `extract_metrics(state, i)` | 214 | **70 维指标**：常数 1 / 回合数 / 分数 / 是否≥15 / 持卡 / 预留 / 买力方差 / 黄金 / 总宝石 + 每色卡数·买力·对数买力 + 前后对手分数 + 12 明牌与 **自己** 3 预留卡的宝石/回合距离 + 5 贵族距离。**不读 `board.gems`（无公共供给）** |
| `normalize_metrics(metrics)` | 313 | 按 `METRIC_NORMALIZATION` 逐维除并 clip [-1,1]（**env 不调用**，GA agent 调用） |
| `get_color_encoder()` | 326 | 颜色 sklearn OneHotEncoder（缓存） |
| `get_yellow_gem_index()` / `get_indices_access_by_color()` | 346/363 | 颜色索引工具 |
| `vectorize_card(card)` | 377 | 单卡 **13 维**：颜色 one-hot(6) + 费用(5) + tier(1) + 分值(1)；None → 全零 |
| `extract_reserved_cards(state, i)` | 416 | **指定玩家自己的** ≤3 张预留卡向量（空位补零） |
| `extract_cards(state, i)` | 443 | 12 明牌 + 自己 3 预留 = 15×13 维 |
| `extract_metrics_with_cards(state, i)` | 470 | **总入口：70 + 195 = 265 维**（Gym 观测与全部 RL agent 的输入） |
| 模块级常量 | 156/183/483 | `METRICS_SHAPE`（70 维布局）、`METRIC_NORMALIZATION`、`METRICS_WITH_CARDS_SIZE=265` |

### D. Gym 环境层

#### `src/splendor/splendor/gym/envs/actions.py`（237 行）— 固定动作空间

| 成员 | 行 | 说明 |
|---|---|---|
| `ActionEnum` | 29 | 6 种动作类型枚举 |
| `CardPosition` | 43 | `(tier, card_index, reserved_index)` 定位（dataclass，不可哈希） |
| `Action` | 57 | 动作 dataclass：`type_enum / collected_gems / returned_gems / position / noble_index`（不可哈希） |
| `Action.to_action_element(action, state, i)` | 69 | 引擎 dict → Action 转换；**buy 动作的两个 gems 字段置 None**（把动作空间压缩一个数量级的关键设计） |
| 6 个生成器函数 | 109-219 | 组合枚举 reserve/collect_same/collect_diff/buy_reserve/buy_available 的全部变体 |
| `ALL_ACTIONS` | 222 | **3510 个动作的静态列表**（PASS 6 / COLLECT_DIFF 2280 / COLLECT_SAME 630 / RESERVE 504 / BUY_AVAILABLE 72 / BUY_RESERVE 18；每类含 noble_index 变体） |

#### `src/splendor/splendor/gym/envs/splendor_env.py`（245 行）— RL 环境本体

| 成员 | 行 | 说明 |
|---|---|---|
| `SplendorEnv(gym.Env)` | 20 | 构造参数 `agents` 是**对手列表**（玩家数 = len+1）；`shuffle_turns` / `fixed_turn` 控制座位 |
| `reset(seed, options)` | 88 | 新建 `LimitRoundsGameRule` → 洗对手座次 → 随机/固定我方座位 → 模拟到我的回合 → 返回 `(obs, {"my_id"})`。**发牌随机走全局 `random`，reset(seed=) 不固定它** |
| `step(action)` | 127 | 执行我方动作（先查 `mapping[action]`，**非法直接 KeyError**）→ `reward = Δscore`（我方分数增量，无胜负信号）→ 模拟全部对手回合 → 返回 5 元组 `(obs, r, terminated, truncated=False, {})` |
| `render()` | 174 | 占位 |
| `get_legal_actions_mask()` | 178 | **(3510,) 0/1 掩码**——每次状态变化后必须重新调用 |
| `_get_opponent_by_turn / _vectorize / _set_opponents_ids / _simulate_opponents` | 191-231 | 对手回合模拟（对手拿真实 state，无超时）与 ego-centric 观测 |
| `turn / state` (property) | 234/241 | 当前回合与状态（`game_rule.current_agent_index / current_game_state` 代理） |

#### `src/splendor/splendor/gym/envs/utils.py`（181 行）— 掩码与映射

| 函数 | 行 | 说明 |
|---|---|---|
| `_valid_position / _valid_reserved_position` | 15/30 | 卡位有效性检查 |
| `build_action(action_index, state, i)` | 43 | 索引 → 引擎动作 dict 的**独立构造路径**；⚠️ 买卡分支不处理黄金通配（docstring 自认可产生负宝石状态）——env 不用它，**新代码也不应使用** |
| `create_legal_actions_mask(legal_actions, state, i)` | 148 | 合法动作 → **(3510,) 0/1 掩码**；内部 `ALL_ACTIONS.index()` 为 O(3510) 线性扫描（性能热点） |
| `create_action_mapping(legal_actions, state, i)` | 167 | 索引 → 可执行引擎动作 dict 的映射（env.step 与全部 RL agent 使用） |

#### `src/splendor/splendor/gym/__init__.py` / `envs/__init__.py`

`register(id="splendor-v1", entry_point="splendor.splendor.gym.envs:SplendorEnv")`——**必须 `import splendor.splendor.gym` 才触发注册**。

### E. 可视化层

#### `src/splendor/splendor/splendor_displayer.py`（754 行）— GUI 与文本显示

| 成员 | 行 | 说明 |
|---|---|---|
| `make_label` | 28 | tkinter 标签工厂 |
| `AgentArea` / `BoardArea` | 37/134 | 玩家面板 / 棋盘面板组件（读 `resources/` 卡牌宝石图片） |
| `can_buy(agent, card)` | 125 | GUI 高亮"可买卡"的判定 |
| `GUIDisplayer(Displayer)` | 266 | tkinter 主显示器：`InitDisplayer` 建窗、`user_input` **人类交互选动作**（`--interactive` 模式）、`ExcuteAction/TimeOutWarning/EndGame` 钩子、全屏切换 |
| `TextDisplayer(Displayer)` | 690 | 文本模式显示器（`-t`），同样含 `user_input` 交互输入 |

### F. 对局运行层

#### `src/splendor/game.py`（258 行）— 对局主循环

| 成员 | 行 | 说明 |
|---|---|---|
| `Game(GameRule, agent_list, ...)` | 29 | 构造校验 agent id；`FREEDOM=True`（直接信任 agent 不限时）或 `False`（`func_timeout` 1 秒/步 + 首回合 `WARMUP=15` 秒）；超时/非法记 warning，3 次判负 |
| `Game.Run()` | 104 | **评测主循环**：`while not gameEnds()` → `getLegalActions` + **deepcopy** 状态与动作给 agent → `SelectAction` → `game_rule.update`；每步用预生成 `seed_list` 重播全局 random 保证可复现；终局 `calScore` + replay 序列化 |
| `GameReplayer` | 216 | 回放器（重放保存的对局） |

#### `src/splendor/general_game_runner.py`（694 行）— CLI 评测器（`splendor` 命令）

| 函数/类 | 行 | 说明 |
|---|---|---|
| `is_git_repo / get_commit_time / gitCloneTeam` | 51-160 | 课程竞赛基础设施：从 git 仓库克隆并加载各队 agent |
| `loadAgent(matches)` | 165 | **动态 import 各 agent 模块并实例化 `myAgent(i)`**——`myAgent` 导出约定的执行点 |
| `HidePrint` | 204 | 上下文管理器，静默第三方打印 |
| `add_cwd_to_sys_path` | 230 | 支持从工作目录加载未安装的 agent（`--absolute-imports` 可关） |
| `run(options, msg)` | 239 | 多局对局编排：`-m` 局数循环调 `Game.Run`，累计平均分/胜/平/负 |
| `loadParameter()` | 495 | 全量 CLI 参数解析（`-a` 模块路径、`--agent_names`、`-t` 文本模式、`-n` 人数、`--setRandomSeed`、replay 存取、`--interactive`） |
| `main()` | 670 | 入口（pyproject 的 gui-script `splendor`） |

### G. Agent 层

#### `src/splendor/agents/generic/`（3 个基线，各 ~30 行）

| 文件 | 类 | 行为 |
|---|---|---|
| `random.py` | `RandomAgent` | `random.choice(actions)` |
| `first_move.py` | `FirstActionAgent` | 恒取第一个合法动作 |
| `timeout.py` | `TimeoutAgent` | `sleep(2)` 后选择（测试引擎超时惩罚用） |

#### `src/splendor/agents/our_agents/minmax.py`（139 行）

| 成员 | 行 | 说明 |
|---|---|---|
| `MiniMaxAgent` | 22 | **仅支持 2 人**；深度 2 |
| `SelectAction` | 32 | 入口：对每个合法动作调 `_select_action_recursion` 取最优 |
| `_select_action_recursion(state, rule, depth, is_maximizing, alpha, beta)` | 43 | alpha-beta 剪枝；动作先 `random.shuffle` 再按类型排序；**`generateSuccessor` 原地搜索 + `generatePredecessor` 回滚** |
| `_evaluation_function(state)` | 90 | 终局 ±99999；否则 `score×2 + 手卡×0.7 + 宝石×0.1(≥8 颗改 −0.7) − 宝石方差×0.2 − 费用距离×0.1` |

#### `src/splendor/agents/our_agents/ppo/`（22 个文件）— PPO 家族主干

| 文件 | 关键成员 | 说明 |
|---|---|---|
| `ppo_base.py`(100) | `PPOBase(ABC)` + `create_hidden_layers` + `PPOBaseFactory` | 网络抽象基类；隐藏层范式 `Linear+LayerNorm+Dropout+ReLU`；forward 契约 `(x, action_mask) → (probs, value, ·)` |
| `input_norm.py`(56) | `InputNormalization` | running mean/var 输入归一化（buffer 形状 `(1, D)`，动量 0.9/0.1，仅 training 模式更新） |
| `network.py`(116) | `PPO(PPOBase)` | MLP 版：InputNorm → [128×4] → actor `Linear(128→3510)` + critic `Linear(128→1)`；正交初始化；**掩码 = `torch.where(mask==0, -1e8, logits)` 后 softmax** |
| `ppo.py`(308) | `save_model` / `extract_game_stats` / `train` / `main` | **训练入口**（`ppo` 命令）：对手工厂或自博弈（对手共享网络对象）；seed 三件套；stats.csv；⚠️ checkpoint 命名 `episode + 1 // N_TRIALS` 有优先级 bug（互相覆盖） |
| `training.py`(264) | `LearningParams` / `train_single_episode` / `update_policy` / `evaluate` | 每 episode 一整局轨迹 → 10 epoch 全批量更新；`evaluate` 贪心打 1 局 |
| `rollout.py`(198) | `RolloutBuffer`：`remember / clear / calculate_gae / unpack` | 轨迹缓冲（上限 1000 步）；`calculate_gae` **实为归一化 MC 回报**（无 TD(λ)） |
| `common.py`(118) | `calculate_returns / calculate_advantages / calculate_policy_loss / calculate_loss` | 损失计算：PPO-clip(0.2)；总损失 `policy + 0.5·value − 0.005·entropy` |
| `arguments_parsing.py`(222) | `NeuralNetArch` / `NN_ARCHITECTURES` / `OPPONENTS_AGENTS_FACTORY` / `parse_args` | 四架构注册表（mlp/gru/lstm/self_attn）；对手工厂（random/minimax/…） |
| `utils.py`(63) | `load_saved_model` / `load_saved_ppo` | 权重加载（含 running stats 的 squeeze(0) 处理；`weights_only=False`） |
| `ppo_agent_base.py`(53) | `PPOAgentBase(Agent)` | agent 基类：device 解析 + `load_policy`（`net.eval()`） |
| `ppo_agent.py`(71) | `PPOAgent` + `myAgent` | **对局 agent 模板**：特征→掩码→forward argmax→`create_action_mapping` 反查 |
| `constants.py`(34) | 超参 | lr=1e-6、γ=0.99、MAX_EPISODES=50000、PPO_CLIP=0.2 等 |

**RNN 变体**（`ppo_rnn/`，7 文件）：

| 文件 | 说明 |
|---|---|
| `recurrent_ppo.py`(87) | `RecurrentPPO(PPOBase)`：循环网络基类（GRU/LSTM 权重正交初始化） |
| `gru/network.py`(181) + `gru/ppo_agent.py`(94) | `PpoGru`：GRU(265→64, 1 层) → 取末时间步 → 同款 trunk → 双头；agent 跨回合维护 hidden state |
| `lstm/network.py`(199) + `lstm/ppo_agent.py`(94) | `PpoLstm`：同上，带 cell state |
| `gru/lstm constants.py`(各 10) | 隐藏维 64、层数等 |

**注意力变体**（`self_attn/`，3 文件）：`PPOSelfAttention`（`network.py`:19，MLP 前插 `nn.MultiheadAttention(265, 1头)`）+ agent。

#### `src/splendor/agents/our_agents/genetic_algorithm/`（5 个文件）

| 文件 | 关键成员 | 说明 |
|---|---|---|
| `genes.py`(125) | `Gene` / `StrategyGene` / `ManagerGene` | 基因抽象（dna 属性、random/load/save/mutate）；策略基因 `(24,)` 与状态指标点积打分；manager 基因 `(24,3)` 选策略 |
| `genetic_algorithm_agent.py`(130) | `GeneAlgoAgent` + `myAgent` | `SelectAction`：manager 选 1/3 策略 → 对每个合法动作 `generateSuccessor` **1 层前瞻** → 提取 `normalize_metrics` 后的指标与策略基因点积 → 取最高分 |
| `evolve.py`(405) | `mutate / crossover / mate / evaluate / sort_by_fitness / generate_initial_population / evolve / main` | **GA 训练入口**（`evolve` 命令）：非均匀变异（强度随代数衰减）、单点混合交叉、适应度 = 2/3/4 人局分数总和（WINNER_BONUS=0）、每代保留前 1/3、支持多进程评估 |
| `argument_parsing.py`(92) | `parse_args` | `--population-size / --generations / --mutation-rate / --seed` |
| `constants.py`(32) | 种群/代数/变异率默认值 | 附带已训基因：`manager.npy`(24×3) + `strategy1-3.npy`(24,) |

---

## 4. 三条运行路径的调用链

### 4.1 训练路径（PPO，`ppo` 命令）

```
ppo:main → parse_args → train
  ├─ seed 三件套（random / np.random / torch）
  ├─ gym.make("splendor-v1", agents=对手)          [import splendor.splendor.gym 触发注册]
  ├─ 每 episode:
  │   train_single_episode
  │   ├─ env.reset() → obs, {"my_id"}
  │   ├─ 循环: get_legal_actions_mask() → policy.forward(obs, mask)
  │   │        → 采样动作 → env.step(a)[内部模拟对手] → RolloutBuffer.remember
  │   └─ update_policy: calculate_gae(MC) → 10 epoch 全批量 → common.calculate_loss
  ├─ evaluate(test_env, policy)  [贪心 1 局]
  └─ extract_game_stats → stats.csv；每 N 局 save_model(含 running stats)
```

### 4.2 评测路径（`splendor` 命令）

```
general_game_runner:main → loadParameter → loadAgent(动态 import 各 myAgent)
  → run: 循环 -m 局
      → Game.Run:
          while not gameEnds():
            getLegalActions(state, i) → deepcopy(state, actions)
            → agent.SelectAction(actions, state, rule)     [FREEDOM 直调 / func_timeout 1s]
            → game_rule.update(action)
                = generateSuccessor(原地) + getNextAgentIndex 轮转
          → calScore 定胜负（平局买卡少者 +0.5）→ 汇总胜/平/负
      → Displayer(GUI/Text).ExcuteAction/EndGame 钩子全程跟随
```

### 4.3 GA 训练路径（`evolve` 命令）

```
evolve:main → parse_args → evolve
  ├─ generate_initial_population(N 个 GeneAlgoAgent)
  └─ 每代: evaluate[多进程 single_game: 2/3/4 人局各一轮，分数总和为适应度]
           → sort_by_fitness → mate(前 1/3 繁殖) → mutate_population → 循环
  → 代末 save 为 manager.npy / strategy*.npy
```

---

## 5. 非代码资产清单

| 资产 | 位置 | 说明 |
|---|---|---|
| 已训 PPO 权重 ×4 | `agents/our_agents/ppo/{ppo_model.pth, ppo_rnn/gru|lstm/*, self_attn/*}` | 四架构各一份；官方评价"需重训"（ALGORITHM_COMPARISON.md） |
| GA 基因 ×4 | `agents/our_agents/genetic_algorithm/*.npy` | 当前最稳 baseline 的权重 |
| GUI 资源 | `splendor/resources/`（cards/gems/nobles 大小两套 + 背景 + 图标） | ~300 个 png，Displayer 渲染用；`convert_filename` 完成卡码↔文件名映射 |
| 课程模板 | `wiki-template/`、`README.md`(src 内)、`img/` | COMP90054 课程 wiki 模板（占位文本） |
| docker 脚本 | `docker/` | 竞赛评测容器化脚本 |
| Sphinx 文档 | `docs/source/*.rst` + `conf.py` | API 文档（`make docs` 生成） |
| 打包配置 | `pyproject.toml`（console scripts）、`Makefile`、`environment.yaml`、`requirements/`、`uv.lock` | 安装与命令入口 |

## 6. 完整文件索引（59 个 .py）

| # | 文件 | 行数 | 层 |
|---|---|---|---|
| 1 | `src/splendor/template.py` | 89 | A 框架基座 |
| 2 | `src/splendor/utils.py` | 13 | A |
| 3 | `src/splendor/version.py` | 14 | A |
| 4 | `src/splendor/game.py` | 258 | F 对局运行 |
| 5 | `src/splendor/general_game_runner.py` | 694 | F |
| 6 | `src/splendor/splendor/splendor_model.py` | 579 | B 引擎 |
| 7 | `src/splendor/splendor/splendor_utils.py` | 245 | B |
| 8 | `src/splendor/splendor/constants.py` | 31 | B |
| 9 | `src/splendor/splendor/types.py` | 66 | B |
| 10 | `src/splendor/splendor/utils.py` | 25 | B |
| 11 | `src/splendor/splendor/features.py` | 487 | C 特征 |
| 12 | `src/splendor/splendor/splendor_displayer.py` | 754 | E 可视化 |
| 13 | `src/splendor/splendor/gym/__init__.py` | 11 | D Gym |
| 14 | `src/splendor/splendor/gym/envs/__init__.py` | 9 | D |
| 15 | `src/splendor/splendor/gym/envs/actions.py` | 237 | D |
| 16 | `src/splendor/splendor/gym/envs/splendor_env.py` | 245 | D |
| 17 | `src/splendor/splendor/gym/envs/utils.py` | 181 | D |
| 18-20 | `agents/generic/{random,first_move,timeout}.py` | 30/29/31 | G 基线 |
| 21 | `agents/our_agents/minmax.py` | 139 | G |
| 22-32 | `agents/our_agents/ppo/{ppo,training,rollout,common,network,ppo_base,input_norm,arguments_parsing,utils,constants,ppo_agent,ppo_agent_base}.py` | 308/264/198/118/116/100/56/222/63/34/71/53 | G PPO 主干 |
| 33 | `ppo/ppo_rnn/recurrent_ppo.py` | 87 | G |
| 34-36 | `ppo/ppo_rnn/gru/{network,ppo_agent,constants}.py` | 181/94/10 | G |
| 37-39 | `ppo/ppo_rnn/lstm/{network,ppo_agent,constants}.py` | 199/94/10 | G |
| 40-42 | `ppo/self_attn/{network,ppo_agent,constants}.py` | 103/76/7 | G |
| 43-47 | `genetic_algorithm/{evolve,genes,genetic_algorithm_agent,argument_parsing,constants}.py` | 405/125/130/92/32 | G GA |
| 48 | `docs/source/conf.py` | — | Sphinx 配置 |
| 49-59 | 其余 `__init__.py` ×11 | 0 | 空占位 |

---

*本全景图对应升级改造前的代码基线；后续按 [plan/](./plan/README.md) 新增的模块（`dqn/`、`browser/`、`tests/` 等）不在本文范围内。*
