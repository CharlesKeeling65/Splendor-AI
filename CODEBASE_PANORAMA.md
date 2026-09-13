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

*§1-§6 对应升级改造前的代码基线快照；升级新增的模块见 §7 增量记录。*

## 7. 升级增量记录（dev 分支，随阶段实时更新）

### 7.1 Phase-0 地基与对齐（2026-09-05 落地）

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/splendor/splendor/gym/envs/utils.py` | 修改 | **ActionIndexCache**：`action_key()`（Action→可哈希键，区分 `None` 与空 dict）、模块级 `ACTION_INDEX`（加载期构建 + 双射断言）、`_index_of()`（未命中抛带细节 ValueError）；`create_legal_actions_mask`/`create_action_mapping` 内部全部改走缓存，**签名与返回值零变化**；旧实现保留为 `_slow_create_legal_actions_mask`/`_slow_create_action_mapping` 供等价性测试对照。实测掩码+映射构建 ×75 提速（见 docs/web_experiments.md M0.2） |
| `src/splendor/splendor/gym/base.py` | 新增 | **`SplendorEnvBase`** 协议（`@runtime_checkable Protocol`）：`observation_space / action_space / reset / step(action, payment=None) / get_legal_actions_mask / get_payment_options` 五件套——"单接口、双环境"的接口契约，`payment` 为 P4 支付维度前瞻参数 |
| `src/splendor/splendor/gym/envs/splendor_env.py` | 修改 | `step()` 增加 `payment: int \| None = None` 兼容参数（显式忽略）；新增 `get_payment_options()` 恒返 None（tier-1 语义）——使 SplendorEnv 满足协议 |
| `src/splendor/browser/`（新包） | 新增 | `card_registry.py`：卡/贵族身份注册表。90 张卡以 `(deck_id, colour, points, sorted_cost)` 四元组为键、10 贵族以 sorted cost 为键，值为引擎同构对象；`web_row_to_deck_id()` 收口网页行序（上→下 = tier 2/1/0）与引擎 deck_id 的方向转换。未命中抛含输入内容的 KeyError |
| `tests/`（新目录） | 新增 | `test_action_index_cache.py`（A0.1 等价性 + A0.2 双射）、`test_card_registry.py`（A0.3 90 卡/10 贵族命中与字段一致）、`test_env_protocol.py`（A0.4 协议符合 + payment 参数冒烟），共 14 例全绿 |
| `docs/web_experiments.md` | 新增 | E1-E6 网页实测协议与结论登记表 + M0.2 吞吐留档（E1-E6 实测随 P2 网页联调进行） |
| `plan/reference/BROWSER_RL_MAPPING.md`、`DQN_GUIDE.md` | 修改 | T0.5 勘误：牌库 90 张（78 是发牌后剩余）、265 维特征不需要记忆重建、支付差距根因在引擎 `getLegalActions`、step 返回 5 元组、`step` 增加 payment 前瞻参数、掩码走缓存路径（就地标注"勘误 2026-09-05"，完整修正表见 UPGRADE_ROADMAP §7） |

存量回归（A0.5）：`splendor -a ...ppo,...minimax -t -m 5` 5 局全部 valid（minimax 5:0，与 ALGORITHM_COMPARISON 既有结论一致）；PPO 训练 3-episode 冒烟正常。
环境备注：本机改用 Homebrew Python 3.13 + `python-tk@3.13` 重建 venv（uv Python 无 tkinter，`splendor` 命令无法运行）。

### 7.2 Phase-1 DQN 本地训练（2026-09-05 落地，worktree phase-1-dqn 并行产出后合并）

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/splendor/agents/our_agents/dqn/`（9 文件，~1480 行） | 新增 | `constants.py`（超参集中）/ `network.py`（Dueling QNetwork：InputNormalization + 4×[Linear128+LayerNorm+ReLU] 无 Dropout → V/A 头，forward 内掩码 -1e9）/ `replay_buffer.py`（float32 环形缓冲 + n-step 滑窗折叠，不存当前掩码）/ `reward_wrapper.py`（终局 ±10，calScore 口径）/ `training.py`（Double DQN 更新 + ε-greedy 采集 + 独立建局评估 + `collect_from_browser` 回流）/ `dqn.py`（train+main，stats.csv，checkpoint 命名修掉 ppo.py:293 优先级 bug）/ `dqn_agent.py`（`myAgent` 导出）/ `utils.py`（save/load 含 running stats） |
| 关键设计偏离 | — | QNetwork 终生 eval + `observe()` 手动 EMA：InputNormalization 单样本 train 前向方差为 0 会退化（PPO 整局批量无此问题）——采集/更新/部署三处归一化语义一致的必要处理 |
| pyproject.toml | 修改 | `scripts.dqn` |
| tests | 新增 | test_replay_buffer / test_dqn_network / test_dqn_update（手工构造 TD 目标）/ test_reward_wrapper / test_dqn_smoke |
| 未完成（待训练条件） | — | T1.6 训练课程 M1-M3、checkpoint、3-seed 方差（用户约束：本机不做训练） |

### 7.3 Phase-2 浏览器适配层（2026-09-05 落地，worktree phase-2-browser 并行产出后合并 + 真实 DOM 回填）

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/splendor/browser/`（8 文件 + fixtures×5） | 新增 | `driver.py`（BrowserDriver 协议 + MockBrowserDriver + `read_raw_snapshot`）与 `dom_extractor.py`（EXTRACT_SNAPSHOT_JS 单次往返 + schema 校验）互为镜像；`state_builder.py`（object.__new__ 伪状态，占位卡只答 len()，身份全走注册表）；`action_executor.py`（四类动作点击序列 + 贪心药丸 + E2 丢弃流程）；`browser_env.py`（协议实现 + 引擎规则掩码 + 面板差分奖励 + 轮询/降级放弃）；`session.py`（建房/入座/开始/双域 cookie/恢复）；`monitor.py`（掩码奇偶三类归因） |
| 真实 DOM 修正（T0.4 回填） | 修改 | 六个 [ASSUMED] 类名全部不存在（实测 ccbs 类清单已核）→ 供给=space-x-6 容器、面板=my-2 容器+我标记、状态=叶子文本扫描、药丸=灰条+正则；覆盖按钮无 ccbs 类 → 协议新增 `click_labelled`/`click_card_button` 文本点击原语（mock 同语义实现）；终局判定=棋盘消失（E3）；夹具按真实结构重生成 |
| tests | 新增 | `test_feature_parity`（**核心质量门**：≥1000 随机状态 obs 逐位 + 掩码全等，实测 ~3s）/ test_browser_adapter / test_mask_parity_monitor |
| 真机验证 | — | 修正后 EXTRACT_SNAPSHOT_JS 在真实对局页全字段正确；ego-browser 冒烟通过 |

### 7.4 Phase-3 网页部署 harness（2026-09-05 落地）

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/splendor/play_web.py` | 新增 | GameReport / run_game（贪心 + 计分：面板快照权威、奖励 telescoping 兜底）/ main（局间随机休息 5-15s、recover 重试一次、JSON 报告） |
| `src/splendor/browser/ego_driver.py` | 新增 | EgoBrowserDriver：ego-browser CLI 参考适配器（async IIFE 包装、JS 经临时文件传输、stderr 结果行、双流解析）；真机冒烟通过（navigate/evaluate/cookies/screenshot/click_labelled） |
| `dqn/training.py` | 修改 | `collect_from_browser`：网页对局转移直接入库（off-policy 回流通路） |
| pyproject.toml | 修改 | `scripts.play-web` |
| tests | 新增 | test_play_web（run_game 终局报告 / checkpoint 往返 / 回流形状与种子确定性） |
| 未完成（待训练+部署条件） | — | 50 局真实部署、s2r 报告数字、A3.3 回流 1000 步验证 |

### 7.5 Phase-5 工程化固化（2026-09-05 落地）

| 文件 | 类型 | 内容 |
|---|---|---|
| `.github/workflows/ci.yml` | 新增 | Python 3.12/3.13 矩阵：ruff + mypy（新代码路径）+ pytest 全量离线 |
| `Makefile` | 修改 | `test` / `parity`（部署前强制质量门）/ `train-dqn` / `play-web`，PYTHON 变量优先 venv |
| 文档 | 修改 | 本文件 §7、AGENTS.md（进度/命令/测试矩阵/浏览器实测事实）、README（命令闭环）、ALGORITHM_COMPARISON（DQN 行） |
| 全量质量门 | — | **83 tests 全绿**（离线，~6s）；ruff/mypy 对全部新代码路径通过 |

### 7.6 DQN 训练正确性与搜索消融（2026-09-07）

用户授权重新开启 P4 公共特征实验；不变更 PPO/GA/minimax、不执行真实网页对局。
实现及复现协议见 [DQN_SEARCH_EXPERIMENTS](docs/DQN_SEARCH_EXPERIMENTS.md)。

- 修复 n-step bootstrap 折扣、终局尾部样本丢失；手算目标回归。
- DQN 专用 `public-v2`（312维），旧 v1（265维）仍为默认并兼容历史权重；
  checkpoint 保存输入维度、特征版本、归一化与辅助头配置。
- 浏览器伪状态保留对手公开资源面板，`play-web` 按模型版本选择观测。
  奇偶测试改为保存真实决策点快照，避免重复比较被原地修改的终局状态。
- `population.py`：规则启发式与最多四个冻结历史快照，整局选择对手。
- `search.py`：有预算上限的隐藏牌重采样 PUCT，独立胜负价值与策略头；
  搜索蒸馏为本地实验功能，并非完整 AlphaZero，也不用于浏览器伪状态搜索。
- `experiment.py` / `benchmark.py`：独立进程、固定预算、多训练种子、
  成对牌局双座次；全部训练完成后才开启独立测试集；不自动部署。
- `assessment.py`：从逐局记录重算指标，拒绝重复/缺失牌局、错误分母、
  非有限训练统计；额外新牌局复核保留所有修正版训练种子，避免挑赢家。
- 已完成五版本×三种子×一万步短程消融；此结果不等于 M1–M3 课程验收。
  最终结论及限制见 [本轮结果报告](docs/DQN_EXPERIMENT_RESULTS_20260907.md)。
- 全量离线质量门：110 passed / 1 skipped；DQN 与涉及浏览器路径的
  Ruff、mypy 通过；`make parity PYTHON=.venv-p5000/bin/python` 32项通过。

### 7.8 DQN 第二轮 EMA / 训练引导（2026-09-07）

- 新增 public-ema、public-sync、public-demo 可选分支；辅助 margin loss 与 TD
  共用一次更新，引导探索与损失权重退火，评测仅使用贪心 Q。
- 12 次 20k 步训练、2400 局测试及 150 局选模纠错重测完成。
- 发现浮点宏平均破坏同分保留较早权重的约定，改用精确分数并补回归测试；
  原始审计显式保留偏差，更正结果单独保存，未覆盖旧模型。
- 纠错后基线 vs minimax 52.7%，引导版 53.3%，但两强对手均值后者更低，
  不宣称整体提升。120 tests passed / 1 skipped，DQN Ruff/mypy 通过。
- 详情与参数见 [第二轮报告](docs/DQN_ROUND2_RESULTS_20260907.md)。

### 7.9 提升路径阶段 A/B/C1/C3/D2/D3/E2/E4/F2 代码（2026-09-13，按 docs/IMPROVEMENT_ROADMAP_20260912.md 实施）

| 任务 | 交付 | 要点 |
|---|---|---|
| A1 league 评测器 | `src/splendor/league.py` + console script `splendor-league` | 全配对 ordered-seat round-robin、Wilson 区间、行为度量（买卡/预留/拿宝石/达 15 分轮数/卡数 tie-break）、JSON manifest + Markdown 矩阵、可选进程池；`tests/test_league_runner.py`（冒烟 + 配对种子可复现） |
| A2 种子段注册 | `src/splendor/seed_registry.py` + `docs/seed_registry.md` | 声明段/封存段/历史禁区，`allocate_seeds` 扩容需显式改注册表；C1/C2 消费记录入文档 |
| A3 预算冒烟 | `runs/budget-smoke/`（不入库），数字入路线图 §2 | PPO 2.1 s/训练局；DQN 29 步/s（solo CUDA） |
| B1 特征 v2 多席化 | `src/splendor/splendor/features_v2.py`，DQN 改 re-export | legacy 312 维逐位保留；新增 337 维 `public-v2-multi`（rival 面板 MAX_RIVALS 槽位、显式席位特征、2 席前缀 = legacy）；`tests/test_features_v2.py` |
| B2 奖励塑形 | `policy_imitation/shaping.py` + dqn `--shaping` | potential-based（Ng 1999，势能复用 calScore + noble coverage）；event 对照组显式标注非策略不变；γ 空间与 γ=1 telescoping 测试；健康检查：塑形 5k vs random 46% > 无塑形 25%，vs heuristic 持平 |
| C1 critic 消融 | `PPOConfig.critic_learning_rate/critic_hidden_dim` + stabilization 4 旋钮 | 5 分支 × 3 seed 配对消融：value1（+35% EV）> critic-lr5（+23%）> base ≈ warmup5 > critichid（−31%）；报告 `docs/PPO_CRITIC_ABLATION_20260913.md`；C2 采纳 value1+critic-lr5 |
| C3 对手池风格化 | `WeightedHeuristicAgent`（rush/hoard）+ `--pool-names` | 默认权重精确复刻 frozen heuristic；池组合入 manifest |
| D2 网页回流混采 | `dqn/web_replay.py` + `collect_from_browser` 奇偶过滤 | 独立 web buffer 按比例混采（默认 20%）、特征 schema fail-closed 校验、奇偶异常局默认整局丢弃；夹具离线测试 |
| D3 价值先验蒸馏 | `policy_imitation/distillation.py` + `policy-imitation distill-dqn` | masked-KL 蒸馏（q-softmax 任意 checkpoint / policy-head 双模式）；NaN-safe 手工掩码 KL；确定性 seed 切分入元数据 |
| E2 排名效用 | shaping.py `RANK_UTILITIES` + `RankUtilityWrapper` | {1:+1, 2:0, 3:−0.5, 4:−1}，同分平均；wrapper 与 TerminalRewardWrapper 同构组合 |
| E4 胜率估计器多席化 | `remote/rollout.py` 守卫放宽 | 仅 legacy public-v2 锁 2 席；multi schema 支持 2..4 席，测试覆盖 |
| F2 多确定化树搜索 | `dqn/search.py` `multi_tree_search_policy` | 每树一次隐藏采样树内复用 + 预算分配（uniform/root-value spread priority）+ 树间根访问分布平均；原单样本 PUCT 零改动 |

C2 规模化自博弈（500×16×3 seed，c2_training 种子段）已完成：EV(last3) 0.34–0.48
（试点期 0.12–0.23），独立测试均值 vs random 100% / heuristic 37% / minimax 52%
/ GA 45%（`docs/C2_SELFPLAY_C4_LEAGUE_REPORT_20260913.md`）。C4 league 体检
（每对手 150 局，封存段）：vs minimax **57.3%**（仓库最佳，超 DQN 52.7%）、
vs heuristic 38.7%、vs GA 47.3% —— G1 门槛未达，heuristic 族缺口归因与下一步
入报告。F1 标定：return 模式价值头 AUC 0.525 < 0.75，**记录为不可用作搜索
先验**（`runs/f1-calibration/`）。E1 席位参数化 + 3p/4p 冒烟零非法
（`docs/E_PHASE_IMPLEMENTATION_20260913.md`）。D1（DQN 200k×3，pool
random:0.5,minimax:0.5）当晚训练完成后按 M2/M3 门槛出报告。

### 7.10 浏览器对局辅助面板 P7 v0/v1（2026-09-14，按 plan/phase-7-browser-advisor.md 实施）

- **定位**：只读 Advisor——人在 ego-browser 窗口手动打牌，进程每 0.4s（对手回合 0.25s）
  轮询 DOM 并输出走法排名/牌堆直方图/可负担性/预留记忆；零点击（advisor 包禁 import
  `action_executor`，AST 级测试强制），礼仪约束由构造保证。
- **交付**：`browser/advisor/`（observer 去抖观察回路 + tracker 预留记忆重建 + engine
  确定性重建与建议 + server/dashboard.html stdlib 仪表盘）+ `play_advisor.py` 入口 +
  console script `play-advisor`；`describe_action` 三件套平移至 `splendor/action_text.py`
  （原位 re-export，advisor 不背 torch）。
- **关键技术裁决**：打分必在**确定性重建状态**上进行——伪状态牌堆为空（F9）且引擎不枚举
  牌堆预留动作；重建 = 伪状态 + tracker 已识别预留落位 + 未知按 tier 从未见牌池采样
  （比 rollout.py 盲采样更紧）；牌堆直方图带 `deck_counts` 守恒自检。
- **质量门**：advisor 专属测试 48 项（observer/tracker/engine/cli/server，全部夹具离线）；
  全量 pytest 377 passed + parity 通过；advisor 文件 ruff/mypy 全绿。
- **待办**：E7 实验（对手预留瞬间 DOM 证据，tracker 证据钩子已留位）→ 油猴叠加（可选）→
  真实房间人工验收（§5.2 清单）。commit 序列 11b949f→8731e90→43ff2db→171ed86→ac3f7f0。
