# Splendor-AI 升级路线图：本地 DQN 训练 → 网页版部署（Sim-to-Real）

> 本文整合 [DQN_GUIDE.md](./DQN_GUIDE.md)（本地 DQN 训练方案）与 [BROWSER_RL_MAPPING.md](./BROWSER_RL_MAPPING.md)（网页版 game.hullqin.cn/ccbs 的 RL 环境映射），
> 给出把本仓库升级为"**本地引擎高速训练、统一接口、浏览器层部署真实对局**"完整管线的分阶段计划。
> 所有承重事实均于 2026-09-03 在源码中逐一验证；对两份源文档的勘误见 §1 与 §7。

---

## 目录

- [0. 总体判断](#0-总体判断)
- [1. 关键裁决与勘误（源码验证，改变架构假设）](#1-关键裁决与勘误源码验证改变架构假设)
- [2. 目标架构：单接口、双环境](#2-目标架构单接口双环境)
- [3. 分阶段升级计划](#3-分阶段升级计划)
- [4. 文件级改动清单](#4-文件级改动清单)
- [5. 关键设计展开](#5-关键设计展开)
- [6. 风险登记册](#6-风险登记册)
- [7. 对两份源文档的修正表](#7-对两份源文档的修正表)

---

## 0. 总体判断

1. **两份文档合起来是一条 sim-to-real 管线**：DQN_GUIDE 负责"模拟器内高速训练"（每秒可跑几十局），BROWSER_RL_MAPPING 负责"真实环境部署"（每步受限于网页渲染与人类节奏，每局分钟级）。浏览器环境**不可能**承担 DQN 的样本量需求（10⁵~10⁶ 步），因此架构必须是：**本地训练为主、浏览器部署为用、浏览器数据可回流**（DQN 是 off-policy，网页经验可直接进 replay buffer——这是选择 DQN 而非重训 PPO 的额外理由）。
2. **对齐程度好于 BROWSER_RL_MAPPING.md 的预期**：经源码验证，仓库 265 维观测**天然与网页信息集对齐**（只含自己的预留牌、对手只暴露分数），文档中设想的"记忆重建模块"不在关键路径上（§1.2），浏览器环境的最小可用版因此大幅简化。
3. **真正的语义差距只有一处硬伤**：买卡支付方式。引擎对每张可买卡只生成**一种贪心支付**（`resources_sufficient` 返回单一 `returned_gems`），网页版让玩家**自选支付方式**且选择影响后续状态（留金 vs 留彩色宝石不等价）。需要两层方案（§5.1）。
4. **总工作量**：关键路径约 4~6 周（P0→P3），P4/P5 为增强项。P1（DQN 本地）与 P2（浏览器层）在 P0 定完接口后**可并行**。

---

## 1. 关键裁决与勘误（源码验证，改变架构假设）

以下 5 项均已在源码中直接验证，其中 #1、#2 修正了 BROWSER_RL_MAPPING.md 的假设：

### 1.1 牌库是 90 张（40/30/20），"78 张"是误读 —— 无冲突

`splendor_utils.CARDS` 共 **90 张**（tier 1/2/3 = 40/30/20，实测统计），`initialGameState` 把全部 90 张装入三层牌库（`splendor_model.py:89-99`）。网页实测"初始牌库剩余 36/26/16" = 40/30/20 **每层发 4 张后的剩余量**（36=40−4）。两份数据完全自洽，BROWSER_RL_MAPPING.md §0-4 把剩余量误当成了总牌数（78 + 12 桌面 = 90）。

**且验证了 `(tier, colour, points, cost)` 四元组在全部 90 张卡上零重复**——比原文档声称的 78 张更强，卡面身份查表方案成立且更稳。

### 1.2 265 维观测天然与网页信息集对齐 —— 记忆重建移出关键路径

BROWSER_RL_MAPPING.md §3.2/§3.3 假设特征向量包含对手预留卡槽位、需要记忆重建填写。**源码验证结论相反**：

- `extract_reserved_cards(game_state, agent_index)` 提取的是**指定玩家自己的** 3 张预留牌（`features.py:416-440`，读 `agent.cards[RESERVED]`）；
- 70 维 metrics 中对手信息**只有分数**（按前后座次分组），预留牌距离也是自己的（`features.py` `extract_metrics` 内 `agent.cards[RESERVED]`）；
- 网页上"我的预留区"是**明牌**（§3.1 已实测）。

因此 265 维观测的每一个分量都能从网页 DOM **直接**读出，**不需要**事件流记忆重建。记忆模块（§3.2）只在两个可选场景才有价值：① 升级特征加入对手预留信念（P4）；② 想在浏览器侧自行校验动作合法性（我们不用——掩码直接来自 DOM 可交互元素）。

### 1.3 支付方式是唯一的硬语义差距

`splendor_model.py:536-538`：买卡合法动作的 `returned_gems` 来自 `resources_sufficient` 的**单一确定性返回**（贪心规则：优先彩色宝石与永久卡，黄金补差），即引擎**不枚举**支付方式。网页版把支付方式暴露为显式子选择（实测出现 `3白` / `2白1金` 药丸），且不同支付留下不同零钱、影响后续状态。

后果：在本地训练的策略**学不到支付偏好**；部署到网页时由 adapter 代选（两层方案见 §5.1）。

### 1.4 引擎有非标准规则，与网页的一致性未验证 —— 掩码奇偶风险

- **强制拿宝石**：手宝石 ≤7 时最少拿 `min(3, 可用色数)` 个、=8 时最少 2、≥9 时最少 1（`splendor_model.py:443-452`）——即**不允许自愿少拿**。标准璀璨宝石规则允许自愿拿 1~2 个。网页版遵循哪套未知。
- **同色 7 张购卡上限**（`splendor_model.py:534`）。
- 若网页遵循标准规则而引擎强制，两侧合法动作集不一致：网页会出现掩码外的动作（我们不会选，损失机会）、或我们掩码内的动作网页不提供（不会发生，掩码取自 DOM）。
- **处置**：P0 期在网页上实测这两条规则；如不一致，以网页行为为准给引擎加 `strict_standard_rules` 兼容开关（或直接修 `getLegalActions` 的 collect_diff 最少数量逻辑）。

### 1.5 意外发现：265 维观测不含公共宝石供给

`extract_metrics` 从不读取 `board.gems`（只读 `board.dealt` 与 `board.nobles`），METRICS_SHAPE（`features.py:156-181`）中没有任何供给维度。即**策略完全看不见公共宝石池余量**（它影响"现在囤宝石 vs 以后再拿"的规划；合法性本身已由掩码编码）。

对浏览器层这是简化（特征不需要供给数字）；对策略质量这是一个**特征盲区**。列为可选升级 `features-v2`（加 5 维供给，需新建 observation 空间版本，与 PPO 的 265 维 checkpoint 断开兼容）——P4 决策项，不阻塞主线。

---

## 2. 目标架构：单接口、双环境

```
┌────────────────────────────────────────────────────────────────┐
│ L4 部署与评测                                                    │
│   本地: general_game_runner（dqn vs minimax/ga/random）          │
│   网页: play-web harness（DQN + BrowserSplendorEnv，胜率日志）    │
│   回流: 网页经验 → 本地 replay buffer（off-policy 微调）          │
├────────────────────────────────────────────────────────────────┤
│ L3 浏览器层（新建, BROWSER_RL_MAPPING）                          │
│   CardRegistry(90) │ DOMExtractor │ ActionExecutor              │
│   BrowserStateBuilder(伪状态) │ BrowserSplendorEnv │ SessionMgr │
├────────────────────────────────────────────────────────────────┤
│ L2 DQN 算法层（新建, DQN_GUIDE）—— 环境无关，只依赖 L1 接口        │
│   QNetwork(+可选支付头) │ ReplayBuffer(n-step/PER) │ 训练循环     │
│   课程: random → mixed(minimax/ga) → self-play                  │
├────────────────────────────────────────────────────────────────┤
│ L1 引擎与接口层（升级）                                           │
│   SplendorEnvBase 协议 │ ActionIndexCache(性能)                  │
│   PaymentEnumeratingGameRule(可选) │ 规则奇偶开关(可选)            │
│   SplendorEnv(现有) + RewardWrapper(DQN_GUIDE §6.2)             │
└────────────────────────────────────────────────────────────────┘
```

**核心原则：DQN 代码（L2）对环境无感知。** 本地 `SplendorEnv` 与 `BrowserSplendorEnv` 实现同一协议（reset / step / `get_legal_actions_mask` / 观测 265 维 float32 / 掩码 3510 维），训练好的 checkpoint 不改一行代码即可在两个环境运行。网页结构化动作（BROWSER §4.2 的 z1/z2/z3）只在 L3 内部存在，对外仍翻译成 `ALL_ACTIONS` 索引（BROWSER §4.3 的对应表已给全）。

协议草案（P0 落地为 `gym/envs/base.py`）：

```python
class SplendorEnvBase(Protocol):
    observation_space: Box        # (265,) float32
    action_space: Discrete        # 3510
    def reset(self, *, seed=None) -> tuple[NDArray, dict]: ...       # info 含 "my_id"
    def step(self, action: int, payment: int | None = None) -> tuple[NDArray, float, bool, bool, dict]: ...
    def get_legal_actions_mask(self) -> NDArray: ...                  # (3510,) 0/1
    def get_payment_options(self, action: int) -> list | None: ...    # §5.1 支付维度，默认 None
```

`step` 的 `payment` 参数是**前瞻兼容设计**：tier-1（当前引擎）恒为 None；tier-2（支付枚举引擎）与浏览器层（DOM 药丸）都实现它。现在就把参数留好，P4 才不用改接口。

---

## 3. 分阶段升级计划

```
P0 地基与对齐 ──┬── P1 DQN 本地训练（DQN_GUIDE M0-M3）
（~3 天）       └── P2 浏览器层最小闭环
                        │
P3 网页部署（sim-to-real 首秀）
                        │
P4 高保真与增强（支付枚举重训 / features-v2 / 记忆重建 / 网页回流微调）
                        │
P5 工程化固化（测试 / CI / 文档 / 对比表更新）
```

### P0 · 地基与对齐（~3 人日，阻塞后续所有阶段）

| # | 任务 | 产出 / 验收 |
|---|---|---|
| 0.1 | **ActionIndexCache**：`ALL_ACTIONS.index()` 是 O(3510) 线性扫描（`gym/envs/utils.py:162,175`），每步 × 每合法动作一次。预构建 `{可哈希键: 索引}` 字典（`Action` 是含 dict 字段的 dataclass，需自定义键序列化），改造 `create_legal_actions_mask` / `create_action_mapping` | 单测：新旧函数输出全等；掩码构建提速 ≥50× |
| 0.2 | **CardRegistry**：从 `splendor_utils.CARDS` 构建 `{(tier,colour,points,cost): Card}` 查表（90 张零重复已验证），供浏览器层卡面识别直接复用 | 单测覆盖全部 90 张 |
| 0.3 | **网页规则实测**（补 BROWSER §8.2 待办 + §1.4 风险）：① 多贵族选择 UI；② >10 宝石返还 UI；③ 终局结算 DOM 特征；④ 牌库为空时 reserve 是否发金；⑤ **强制拿宝石规则**（能否自愿拿 1 个）；⑥ 同色 7 张上限是否存在 | 每项写成 ADR 式记录，回填 BROWSER 文档 |
| 0.4 | **落地 `SplendorEnvBase` 协议**（§2 草案），`SplendorEnv` 声明实现它（现属鸭子类型兼容，零改动） | mypy 通过 |
| 0.5 | **勘误两份源文档**（见 §7 修正表） | 文档更新 |

### P1 · DQN 本地训练（1~2 周，= DQN_GUIDE M0–M3）

按 [DQN_GUIDE.md](./DQN_GUIDE.md) §6 骨架实现 `agents/our_agents/dqn/` 全套（网络 / buffer / 奖励包装 / 训练循环 / 对局 agent），跑通其 §10 的 M0→M3：

- **验收**：vs random 胜率 >90%；vs minimax(深度2) 100 局胜率 ≥55%；训练曲线与 stats.csv 可复现（seed 三件套）。
- 与 DQN_GUIDE 的差异点：`step` 签名带 `payment=None` 前瞻参数；训练循环读 mask 用 P0.1 的缓存路径。
- 并行推进 P2（互不依赖）。

### P2 · 浏览器层最小闭环（1~2 周，可与 P1 并行）

目标：`BrowserSplendorEnv` 不接 DQN 也能自主打完整局（对面坐一个脚本/人），且**特征与引擎逐位一致**。

| # | 任务 | 要点 |
|---|---|---|
| 2.1 | `DOMExtractor`（BROWSER §3.1 schema） | 一次注入式 JS 抽出 board/decks/nobles/supply/panels/my-reserved/status；输出纯 Python dict 快照。**离线夹具**：保存若干真实局面 HTML，单测不依赖网络 |
| 2.2 | `BrowserStateBuilder`（伪状态适配器，§5.2） | DOM 快照 → 真实 `Card` 对象（走 CardRegistry）+ 轻量 state 对象，直接调用 `features.extract_metrics_with_cards`。`turns_made_by_agent` 读 `agent.agent_trace.action_reward` 长度（`features.py:100-105`），伪状态需自维护计数器填充 |
| 2.3 | **特征奇偶校验测试**（§5.2，本阶段的核心质量门） | 引擎状态 → 可见投影 → 伪状态 → 断言 265 维向量与直接对引擎提取**逐位相等**，跑 ≥1000 个随机状态 |
| 2.4 | `ActionExecutor`（BROWSER §4.1 点击序列） | 3510 索引 → 结构化 → 1~3 步点击；支付药丸 tier-1 策略 = 模拟引擎贪心规则（§5.1）；贵族选择 = 取 `noble_index` 对应项或首项；点击节奏人类量级（礼仪） |
| 2.5 | `BrowserSplendorEnv` | 实现完整协议；`step` 阻塞至再轮到己方（轮询状态文本 0.3~0.5s + 超时保护）；奖励 = 面板"N分"差分；终局检测用 P0.3-③ 采到的 DOM 特征 |
| 2.6 | `SessionManager`（BROWSER §2 配方） | 建房 / 双身份 cookie（含双域删除坑）/ 入座 / 开始；会话恢复 |
| 2.7 | 掩码奇偶监控 | 记录"DOM 推导的合法动作集"与"引擎对同局面计算的合法动作集"的差异日志，10 局内差异为零或全部可解释（规则差异归 P0.3-⑤ 处置） |

**验收**：环境自主完成 ≥20 局完整对局无崩溃；单步平均延迟 <2s；奇偶测试全绿。

### P3 · 网页部署（~1 周，sim-to-real 首秀）

| # | 任务 | 要点 |
|---|---|---|
| 3.1 | `play-web` console script | 加载 DQN checkpoint + BrowserSplendorEnv，对局 + 胜率/得分日志；`pyproject.toml` 增 `scripts.play-web` |
| 3.2 | 鲁棒性 | 有操作计时的房间限时保护；瞬态弹窗（购买揭示）错过不致命（不依赖它）；断线/房间异常重建会话 |
| 3.3 | 首轮对照实验 | 同一 checkpoint：本地 vs 各对手胜率 ↔ 网页 vs 同类对手胜率，量化 sim-to-real 落差（预期主要来自支付方式与对手池分布） |
| 3.4 | 网页经验回流（首个 off-policy 红利） | 网页对局的 `(s,a,r,s',mask,done)` 直接写入本地 replay buffer 混合训练——PPO 做不到这一点 |

**验收**：网页 50 局稳定运行；胜率报告产出；无超时判负。

### P4 · 高保真与增强（可选，2+ 周，按 P3 观察到的落差决定取舍）

| # | 任务 | 触发条件 |
|---|---|---|
| 4.1 | **tier-2 支付方案**：`PaymentEnumeratingGameRule` 子类枚举全部合法 `returned_gems`；DQN 加支付辅助 Q 头（§5.1）；重训 | P3 发现贪心支付造成明显策略损失（如留金局面对局劣势） |
| 4.2 | **features-v2**：加 5 维公共宝石供给（§1.5）；重建 observation 空间 | 策略出现"看不见宝石池"导致的失误模式 |
| 4.3 | 记忆重建模块（BROWSER §3.2）+ 对手预留信念特征 | 想做对手建模/更高水平 |
| 4.4 | 网页双开自博弈 + 回流微调常态化 | 需要逼近人类水平 |
| 4.5 | ws protobuf 快通道 | 仅作为调试/加速选项（BROWSER §6 已定位为非主路径） |

### P5 · 工程化固化（持续）

- 测试目录与 CI（当前仓库无 tests/）：奇偶测试、缓存测试、registry 测试、DOM 夹具测试、DQN 冒烟测试（100 步训练不崩）。
- `ALGORITHM_COMPARISON.md` 增补 DQN（本地）与 sim-to-real（网页）两列结果。
- Makefile 增 `make train-dqn / make play-web / make parity-test`。

---

## 4. 文件级改动清单

```
src/splendor/
├── splendor/
│   ├── gym/
│   │   ├── base.py                        [新增 P0] SplendorEnvBase 协议
│   │   └── envs/
│   │       ├── utils.py                   [改 P0] ActionIndexCache；create_* 走缓存
│   │       └── splendor_model.py 的引擎侧 [改 P4 可选]
│   │           └── PaymentEnumeratingGameRule / 标准规则开关
├── agents/our_agents/dqn/                 [新增 P1, DQN_GUIDE §6.1 全套]
│   ├── network.py / replay_buffer.py / reward_wrapper.py
│   ├── dqn.py / dqn_agent.py / constants.py / dqn_model.pth
├── browser/                               [新增 P2，独立子包]
│   ├── card_registry.py                   [P0.2] 90 张卡四元组查表
│   ├── dom_extractor.py                   [P2.1] §3.1 schema + 离线夹具
│   ├── state_builder.py                   [P2.2] DOM → 伪状态 → 特征
│   ├── action_executor.py                 [P2.4] 索引 → 点击序列 + 支付/贵族子选择
│   ├── browser_env.py                     [P2.5] BrowserSplendorEnv（实现协议）
│   ├── session.py                         [P2.6] 房间/双身份/恢复
│   └── fixtures/*.html                    [P2.1] 真实局面快照
├── play_web.py                            [新增 P3] 部署 harness（console script）
tests/
├── test_action_index_cache.py             [P0]
├── test_card_registry.py                  [P0]
├── test_feature_parity.py                 [P2] 核心质量门
├── test_browser_adapter.py                [P2] 离线 DOM 夹具
└── test_dqn_smoke.py                      [P5]
pyproject.toml                             [改] scripts.dqn / scripts.play-web
```

现有文件改动刻意最小化：引擎与 PPO 路径零行为变化（缓存是纯性能优化，协议是纯声明），保证既有 `splendor`/`ppo`/`evolve` 命令与已训权重完全不受影响。

---

## 5. 关键设计展开

### 5.1 支付方式差距的两层方案

**Tier-1（P2/P3 采用，零训练成本）**：adapter 在网页端模拟引擎贪心规则选药丸——引擎贪心 = 优先彩色宝石与永久卡、黄金补差，映射到药丸即"彩色宝石用得最多、金用得最少"的那个。这样**网页上的状态转移与本地训练分布保持一致**（这是关键：一致性比对优劣更重要），代价是策略永远学不到"留金更灵活"这类偏好。贵族选择同理（各贵族均 3 分，取 `noble_index` 对应项即可）。

**Tier-2（P4 可选，完整方案）**：

1. 引擎侧：`PaymentEnumeratingGameRule(SplendorGameRule)` 覆写买卡动作生成——对每张可买卡枚举全部合法 `(彩色用量, 黄金用量)` 组合（逐色 `k_c ∈ [max(0, cost_c−黄金), min(cost_c, 彩色可用)]` 的笛卡尔积，受黄金总数约束），每个组合一个合法动作。
2. 动作空间：**不**扩平铺 3510（组合爆炸且破坏 checkpoint 兼容），改为**因子化**：主头仍 3510，新增支付辅助头 `Q_pay(obs, action) → K 维`（K≈32 封顶，掩码来自 `get_payment_options(action)`），仅当 `len(options)>1` 时参与决策。这正是 BROWSER §4.2"结构化动作"在训练侧的镜像。
3. 训练侧：支付选择进入 replay（transition 增 `payment` 字段）；对手侧支付策略随机化（避免过拟合贪心对手）。
4. 验收：消融实验证明 tier-2 ≥ tier-1（本地对局 + 网页对照）。

### 5.2 伪状态适配器与特征奇偶校验（P2 的核心质量门）

思路：**不重写特征代码，复用它**。`extract_metrics_with_cards` 只访问 `game_state.agents[i]`（score/gems/cards/agent_trace）与 `board.dealt` / `board.nobles`（已逐一验证）。因此浏览器层只需把 DOM 快照组装成"真 Card + 轻量容器"：

```python
# browser/state_builder.py 思路
def build_state(snapshot: dict, my_index: int) -> SplendorStateLike:
    # board.dealt: 3×4 的 Card|None —— 每张经 CardRegistry[(tier,colour,points,cost)] 还原
    #   （注意网页 tier 行序上→下 = 2/1/0，与 deck_id 方向转换，BROWSER §7-4）
    # agents[i]: score ← "N分"; gems ← 6 circle; cards[色] ← 5 rect 数量拼合成同色 Card 列表
    #   （永久卡只需数量——features 只用 len(cards[color])，可用占位 Card 填充）
    # agent_trace: 自维护步数计数器填充（turns_made_by_agent 只取长度）
    # nobles: (code, cost) ← .ccbs-noble 需求（2 人局 3 张，特征段 5 位补零，天然兼容）
```

**奇偶校验测试**（`test_feature_parity.py`）：

```
引擎随机对局 → 取每个决策点 state
  ├─ 直接 extract_metrics_with_cards(state, i)          （基准）
  └─ state → 可见投影 dict（模拟 DOM 能看到的一切）
            → build_state(投影, i) → extract_metrics_with_cards   （被测）
断言两向量逐位相等（≥1000 状态）
```

该测试同时守住了：卡牌 registry 正确性、tier 方向转换、特征代码对隐藏信息的零依赖、伪状态字段完备性。它是整个 sim-to-real 管线最重要的单测，**网页端任何 DOM 抽取 bug 都会先在这里被本地复现和修复**。

---

## 6. 风险登记册

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R1 | 网页待测行为（贵族 UI、>10 返还、终局 DOM、空牌库 reserve）与预期不符 | 高（阻塞 P2/P3） | P0.3 逐项实测后再写 executor 分支；全部列为 P2 前置 |
| R2 | 强制拿宝石 / 7 张同色上限等规则与网页不一致 → 掩码奇偶破坏 | 高 | P0.3-⑤ 实测；如不一致给引擎加标准规则开关，训练用与网页一致的规则集 |
| R3 | 支付方式贪心 vs 自选的 sim-to-real 落差 | 中 | tier-1 保证分布一致（无落差但次优）；tier-2 彻底解决（P4） |
| R4 | 网页有操作计时，决策或等待超时判负 | 中 | DQN 前向毫秒级；等待循环带硬超时与降级（超时前先点"放弃"保回合）；默认进无计时房间 |
| R5 | 瞬态弹窗（购买揭示 1~2s）被错过 | 低 | 基线特征不依赖它（§1.2 裁决）；仅 P4 记忆模块需要，用 MutationObserver 而非轮询 |
| R6 | 网页对手分布与训练对手池差异大（人类玩法） | 中 | P3.3 对照实验量化；P4.4 网页经验回流微调 |
| R7 | 服务器反自动化 / 礼仪 | 中 | 只在自建房间双开（BROWSER §7-6）；人类量级点击频率；控制在线时长 |
| R8 | `ALL_ACTIONS.index()` 与 `getLegalActions` 的 deepcopy 使本地训练吞吐不足 | 中 | P0.1 缓存解决索引；deepcopy 热点若仍显著，P1 期用 cProfile 定位后局部优化 |
| R9 | DQN 训练不收敛（DQN_GUIDE §9 已列 12 陷阱） | 中 | 严格按 DQN_GUIDE M0→M3 里程碑推进，失败先查奖励/归一化 |
| R10 | 特征盲区（无公共宝石供给，§1.5）限制策略上限 | 低 | P4.2 features-v2 |

---

## 7. 对两份源文档的修正表

| 源文档位置 | 原表述 | 修正（本文档裁决） |
|---|---|---|
| BROWSER §0-4 / §1 | 引擎与网页均为 78 张（36/26/16） | 引擎 90 张（40/30/20）；36/26/16 是发牌后剩余；四元组唯一性已在 90 张上验证成立 |
| BROWSER §3.2 / §3.3 | 特征需含对手预留槽位，需记忆重建填写 | 265 维特征只含**自己的**预留牌与对手分数（`features.py:416-467`），与网页信息集天然对齐；记忆重建仅 P4 可选 |
| BROWSER §4.1-C | "仓库把支付方式折叠掉了"（归因于 build_action） | 根因在**引擎** `getLegalActions`：`resources_sufficient` 只返回单一贪心支付（`splendor_model.py:536-538`）；`build_action` 只是沿用了它 |
| BROWSER §6 | `step()` 返回 4 元组 | 对齐 gymnasium 5 元组（`obs, r, terminated, truncated, info`），与仓库 `SplendorEnv` 一致 |
| DQN_GUIDE §5.4 | 奖励改造由 wrapper 承担 | 不变，但 `step` 签名增加 `payment=None` 前瞻参数（本文档 §2）；wrapper 同时适用于浏览器环境 |
| DQN_GUIDE（新增裁决） | — | 观测不含公共宝石供给（§1.5）；训练循环的掩码构建走 P0.1 缓存 |

两份文档的其余结论（DOM schema、点击序列、双开配方、DQN 超参、课程设计等）经本次验证**全部有效**，本路线图直接引用不重复。
