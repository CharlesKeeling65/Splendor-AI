# Phase 2 · 浏览器层最小闭环

> **定位**：把 game.hullqin.cn/ccbs 的一个座位包装成与本地 `SplendorEnv` **同协议**的 gym 环境——不接 DQN 也能自主打完整局，且特征与规则判断与引擎**逐位一致**。与 P1 完全并行。
> **依赖**：T0.2（注册表）、T0.4（E1–E6 实测，阻塞执行器与终局检测分支）、T0.1（缓存）。
> **估时**：8~9 人日。
> **DOM schema 与点击序列依据**：[reference/BROWSER_RL_MAPPING.md](./reference/BROWSER_RL_MAPPING.md) §3–§4；函数签名：[reference/IMPLEMENTATION_SPEC.md](./reference/IMPLEMENTATION_SPEC.md) §3。

## 1. 阶段目标

1. 驱动抽象 + DOM 抽取 schema 代码化（含离线夹具，测试不依赖网络）
2. 伪状态构建器——特征与合法动作**全部经由引擎代码计算**，浏览器层零规则代码
3. 特征/掩码双奇偶校验测试（整条管线的质量门）
4. 四类动作 + 支付药丸 + 贵族选择的点击执行器
5. `BrowserSplendorEnv` 实现协议并自主完成 ≥20 局；会话管理与掩码奇偶监控

## 2. 任务清单

| ID | 任务 | 产出 | 估时 | 依赖 |
|---|---|---|---|---|
| T2.1 | DOM 抽取器 | `browser/dom_extractor.py` + `fixtures/` | 2d | T0.4 |
| T2.2 | 伪状态构建器 | `browser/state_builder.py` | 1d | T0.2, T2.1 |
| T2.3 | 奇偶校验测试 | `tests/test_feature_parity.py` | 0.5d | T2.2 |
| T2.4 | 动作执行器 | `browser/action_executor.py` | 2d | T0.4, T2.1 |
| T2.5 | 浏览器环境 | `browser/browser_env.py` | 1.5d | T2.1–2.4 |
| T2.6 | 会话管理 | `browser/session.py` | 1d | — |
| T2.7 | 掩码奇偶监控 | `browser/monitor.py` | 0.5d | T2.5 |
| （前置） | 驱动抽象 | `browser/driver.py` | 0.5d | — |

## 3. 代码改动详解（说明 / 意义 / 原因）

### 3.1 `browser/driver.py`【新增】BrowserDriver 协议

**内容**：八方法协议——`evaluate(js) / click(selector) / wait_for(condition_js, timeout) / navigate(url) / get_cookies / set_cookie / delete_cookies / screenshot`。

**意义**：浏览器工具的防波堤：具体驱动（ego-browser / playwright / CDP）在此层之下可替换。

**原因**：(a) BROWSER_RL_MAPPING 的实测基于 ego-browser，但项目不应绑死单一工具——浏览器自动化工具的 API 稳定性远低于本仓库代码，隔离层让工具迁移成本收缩到一个 adapter 文件；(b) **协议层让单测可以注入 mock driver**——全部浏览器逻辑（抽取、构建、执行）可离线测试，这是"网页端 bug 本地复现"策略的工程前提，也是 CI 不碰网络的保证。

### 3.2 `browser/dom_extractor.py`【新增】DOM 快照抽取

**内容**：TypedDict schema（`Snapshot / CardInfo / NobleInfo / PanelInfo`，字段与 BROWSER §3.1 一一对应）+ `EXTRACT_SNAPSHOT_JS`（单次注入的 IIFE，输出对齐 schema 的 JSON）+ `extract_snapshot(driver)`（执行 + **schema 校验**，缺字段/类型错即抛）。

**意义**：网页世界 → Python 世界的唯一入口。所有下游模块只消费 `Snapshot`，不触碰 DOM。

**原因**：(a) **单次 evaluate 而非多次查询**：每次 CDP 往返几十毫秒，一局几十次决策 × 十几项数据，多次查询会显著拖慢单步延迟；IIFE 一次往返拿全量；(b) **schema 校验即时报错**：页面改版（`ccbs-*` 类名变化）会在抽取层炸出清晰错误，而不是在下游变成"agent 变菜了"这种无从下手的症状；(c) TypedDict 而非裸 dict：mypy 拦截字段名手误，schema 本身成为文档；(d) **schema 不含对手预留卡槽位**——源码裁决（roadmap §1.2）：265 维特征只用自己的预留牌与对手分数，BROWSER 文档设想的记忆重建（其 §3.2/§3.3）**不在最小闭环内**，这是本阶段比原文档预想简化一整个模块的原因。

### 3.3 `browser/state_builder.py`【新增】伪状态构建器

**内容**：`build_pseudo_state(snapshot, my_index) -> SplendorState`。实现要点：`object.__new__(SplendorState)` 绕过 `BoardState` 构造器的随机初始化；`board.dealt` 用注册表还原真 `Card`；`board.gems` = 供给；`board.nobles` = 注册表还原的 `(code, cost)` 元组；我的 agent：score/gems 直填，`cards[色]` 用占位卡填到面板数量，`cards["yellow"]` = 我的预留明牌（真卡），`agent_trace` 填自维护回合计数；对手只填 id/score。

**意义**：**整个浏览器层的枢纽**。特征提取（`extract_metrics_with_cards`）与合法动作生成（`getLegalActions`）都经由伪状态**复用引擎代码**——浏览器层一行规则/特征代码都不写。

**原因**：(a) 已核实（IMPLEMENTATION_SPEC F7/F9）这两条引擎路径只读**面板级字段**（`len(cards[color])`、`gems`、`score`、`dealt`、`nobles`），不读牌库内容与卡牌身份细节——所以占位卡方案成立：`resources_sufficient` 与 `agent_buying_power` 只用 `len()`；(b) **单一代码源原则**：规则若在浏览器层重写一遍，引擎任何规则改动都要双改，且两版语义漂移是最隐蔽的 bug 源（恰是本计划要消灭的那类）；(c) `object.__new__`：`BoardState.__init__` 会 `random.sample` + `shuffle`，每步白做随机初始化还污染全局 RNG；(d) 贵族用值相等元组：`to_action_element` 用 `board.nobles.index(action["noble"])` 查下标，元组值相等即可匹配，注册表还原的元组天然满足。

### 3.4 `tests/test_feature_parity.py`【新增】奇偶校验测试（质量门）

**内容**：`project_visible(state, i) -> Snapshot`（引擎状态 → 只含 DOM 可见信息的投影，测试专用 + 夹具生成器）+ `test_obs_and_mask_parity`：随机对局 ≥1000 个决策点，逐点断言 (i) `extract_metrics_with_cards(state, i)` 与经投影→伪状态再提取的结果**逐位相等**；(ii) 引擎直算的合法动作掩码与伪状态上算的**完全一致**。

**意义**：整条 sim-to-real 管线**最重要的单测**。它验证的不是"代码能跑"，而是"两个世界同构"。

**原因**：(a) obs 奇偶守住特征管线（颜色映射、tier 方向、费用读取任何一处错都会被抓）；mask 奇偶守住规则复用（F9 论断的直接证据，也是 §3.3 占位卡方案的正确性证明）；(b) **任何网页抽取 bug 都会先在这里以本地可调试的方式暴露**——比起在真实网页上表现为"agent 下臭棋"然后人肉二分排查，本地逐位断言的定位成本是分钟级；(c) `project_visible` 一份代码两用：既是测试的投影器，又是离线夹具的生成器（把引擎状态渲染成 Snapshot 喂给下游测试）。

### 3.5 `browser/action_executor.py`【新增】动作执行器

**内容**：`ActionExecutor.execute(action_index, snapshot, pseudo_state)`——按 `ALL_ACTIONS[i]` 的 dataclass 分派四类点击序列（PASS 二步 / COLLECT 三步 / RESERVE 二步 / BUY 二至三步，BROWSER §4.1 实测序列）；`select_payment_greedy(pills, ...)` tier-1 支付策略；`parse_pill("2白1金") -> {"white":2,"yellow":1}`；`HUMAN_CLICK_DELAY = (0.2, 0.5)` 节奏控制；每步 `wait_for` 确认 UI 迁移，失败重试一次后抛 `ActionExecutionError`。

**意义**：策略意志 → 网页操作的出口。

**原因**：(a) 点击序列按实测整理，**每步之间确认 UI 状态迁移**：网页有渲染延迟与动画，盲目连点是竞态事故源；重试一次覆盖瞬时抖动，超过一次则明确失败让人接管；(b) **tier-1 支付策略 = 模拟引擎贪心**（选"金用量最少"的药丸，对应 `resources_sufficient` 的彩色优先语义）：**一致性优于局部最优**——策略在本地训练中从未见过"留金"的状态分支，让 adapter 选贪心药丸就是把网页状态拉回训练分布；贸然选"更优"的留金药丸反而制造训练时未见过的状态，行为不可预测（彻底解决要靠 P4 tier-2）；(c) **人类节奏 + 随机抖动**：对真实服务器的礼仪自律（roadmap R7），实现成本只是 sleep 区间，却把"机器人特征"降到最低。

### 3.6 `browser/browser_env.py`【新增】BrowserSplendorEnv

**内容**：实现 `SplendorEnvBase` 协议五件套。`step` = 执行 → 轮询状态文本直至再轮到我或终局（0.4s 间隔，硬超时先降级点"放弃"保回合）→ 奖励 = 面板"N分"差分 → 终局检测（E3 实测特征）。`get_legal_actions_mask` = 伪状态 → 引擎 `getLegalActions` → 缓存版掩码构造；同时收集 DOM 可交互元素为 affordance 集合交监控。`get_payment_options` tier-1 返回 None。

**意义**：**sim-to-real 的接缝完成**——P1 的 checkpoint 零改动可跑（含 `TerminalRewardWrapper` 原样套用，因为奖励语义与本地对齐）。

**原因**：(a) **掩码走引擎规则复用而非 DOM 重算**（roadmap §0.3-1 裁决）：DOM 可交互元素只能证明"可点"，不能证明"合法全集"——"自愿少拿一颗宝石是否合法"是规则问题（E5 裁决的对象）不是 UI 问题；引擎规则是唯一权威源，DOM affordance 降级为交叉验证信号，**反而免费得到了规则奇偶探测器**（R2 风险的常驻监控）；(b) 轮询 + 硬超时 + 降级放弃：防对手挂机/网络异常导致死等；损失一步（放弃）永远好过超时判负；(c) 奖励 = 面板差分：与本地 env 的 Δscore 语义对齐（`splendor_env.py:142-161` 同源），P1 的奖励 wrapper（终局 ±10）直接复用，**两个环境的训练/部署代码不需要任何 if-else**。

### 3.7 `browser/session.py`【新增】SessionManager

**内容**：`create_room / join_seat / start_game / new_game / switch_identity / recover`。内置双域 cookie 清理（`game.hullqin.cn` 与 `.game.hullqin.cn` 都要 `deleteCookies`）与身份切换时序约束（先载入页面再换 cookie——ws 握手身份在连接时固定）。

**原因**：(a) BROWSER §2 的实测配方代码化：**双域删除坑**与**时序坑**是双开自博弈的两个隐形雷，必须收进库函数而不是留在文档里靠记忆遵守；(b) `recover` 是一等公民而非异常处理附属品：网页部署的现实就是会掉线（网络抖动、房间异常），恢复路径的质量决定 P3 能否挂机 50 局。

### 3.8 `browser/monitor.py`【新增】MaskParityMonitor

**内容**：`check(engine_mask, dom_affordances) -> list[str]` 差异归因报告——三类归因：已知规则差异（E5/E6 裁决登记过的）/ DOM 抽取 bug / 页面改版。

**意义**：R2（规则奇偶）风险的常驻探测器，也是 §3.6 掩码设计的验收配套。

**原因**：引擎掩码与 DOM affordance 是**同一合法动作集的两个独立测量**，差异 = 某处假设错了。没有归因工具时，差异只能人肉翻日志——P3 部署期它会变成主要时间黑洞；有了它，每次差异自动分类，只有"新类别"才需要人看。

### 3.9 `browser/fixtures/*.html`【新增】离线夹具

**内容**：真实局面的 HTML 快照，至少覆盖：普通局面、**待定支付药丸**、**终局结算**、宝石将满（>10 返还）、贵族多选、空牌库。

**原因**：让 §3.2/§3.3 的测试离线可跑（CI 不碰网络）；**特殊局面必须有专门夹具**——普通局面采样再多也覆盖不到支付/终局/返还这些分支路径，而它们恰是执行器最容易写错的地方。

## 4. 自动验收目标

| ID | 验收项 | 判定标准 |
|---|---|---|
| A2.1 | 特征/掩码双奇偶 | ≥1000 随机状态：obs 逐位相等（`np.allclose` 零容差）+ 掩码 `np.array_equal` |
| A2.2 | 离线夹具测试 | 全部夹具（含特殊局面）抽取→伪状态→obs 流水线测试通过 |
| A2.3 | 监控归因单测 | 构造三类差异各一例，归因输出正确 |
| A2.4 | 协议符合 | `isinstance(BrowserSplendorEnv(...), SplendorEnvBase)` 为 True |
| A2.5 | 环境自主对局 | 脚本驱动 20 局完整结束，无未捕获异常（对面为脚本座席或人） |

## 5. 人工验收目标

| ID | 验收项 | 要点 |
|---|---|---|
| M2.1 | 网页人工观战 ≥5 完整局 | 四类动作点击正确、支付药丸选择合理、无卡死；瞬态弹窗（购买揭示 1~2s）不影响后续操作 |
| M2.2 | 掩码奇偶 10 局复核 | monitor 报告零未解释差异；有差异的逐条归因到 E5/E6 已知规则差或登记为新 bug（不许"看起来没事"就放行） |
| M2.3 | 夹具覆盖度审查 | 对照 §3.9 的关键局面清单逐项确认有夹具且测试真的走到了那个分支 |
| M2.4 | 延迟记录 | 单步平均延迟（执行+等待+抽取+特征）<2s，记录分布（p50/p95）留档 |
| M2.5 | 礼仪自查 | 脚本日志时间戳间隔审查：点击频率人类量级；只操作自建房间 |

## 6. 风险与回退

- **E1/E2/E3 实测结果与预期不符** → 执行器相应分支返工。这正是 T0.4 前置的原因：把返工压缩在执行器代码量最小的时点。
- **页面改版** → schema 校验立刻失败（fail-fast 设计），修复收口在 `EXTRACT_SNAPSHOT_JS` 一处 + 重采夹具，下游模块零感知。
- **引擎规则与网页不一致（E5/E6 裁决不一致）** → 按 P0 的 ADR 决定：给引擎加 `standard_take_rules` 开关并在训练中启用网页一致规则——改引擎而非在浏览器层打补丁（单一代码源原则）。
