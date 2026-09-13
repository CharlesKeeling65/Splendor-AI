# Phase-7：浏览器对局辅助面板（只读 Advisor）

> 状态：**v0/v1 已实现（2026-09-14，离线质量门全绿）**；T7.2（E7 实验）、T7.7（油猴叠加，可选）
> 与 §5.2 真实房间人工验收待执行。需求来源：用户对 v0 设计草案的六项逐条确认（§1）。
> 实施记录见 §9。
> 定位：复用 P2 浏览器层与 P3 部署链路，为人类玩家提供"每步走法建议 + 牌堆颜色规划"的只读辅助面板；
> 与 P4 记忆重建的关系见 §1.2。操作者视角的使用手册待实施完成后写入
> [docs/WEB_DEPLOYMENT_GUIDE.md](../docs/WEB_DEPLOYMENT_GUIDE.md) 新章。

## 1. 需求与裁决

### 1.1 需求 → 裁决

| # | 需求（用户原话归纳） | 裁决 |
|---|---|---|
| 1 | 读取浏览器网页**全部**牌局信息，指导每步怎么走：有哪些选择、哪个最优、哪个次优 | 复用 `extract_snapshot → build_pseudo_state` 管线；建议 = GA 快评 top-k（常开）+ minimax 深评（按钮触发），输出中文动作描述 + 分数条 + 一句话归因 |
| 2 | 显示**未翻开牌堆**中还有哪些颜色的宝石卡，方便后续规划 | 注册表差集（确定性计算，零模型成本）：`CARDS` 90 张 − 桌面明牌 − 各家已购 − 已知预留；按 层级×颜色 直方图展示，附"可负担性标尺" |
| 3 | 计算基座用**无推理需求**的模型（GA / minimax），不起 checkpoint 服务 | GA 权重 checkpoint 已在仓库（`genetic_algorithm/manager.npy` 等），纯 numpy 前向；minimax 为 α-β 搜索，均为本地毫秒~秒级 |
| 4 | 呈现形式 | v0 CLI 文本 → v1 本地仪表盘（推荐主形态）→ v2 可选油猴叠加；均为新代码，零新依赖 |

### 1.2 用户已确认的六项决策（2026-09-14）

| # | 决策 | 对设计的影响 |
|---|---|---|
| 1 | **在用户游玩的窗口获取信息**（不另开观战身份） | daemon 启动/复用一个 ego-browser 任务空间，用户在该 Chromium 窗口手动游玩；daemon 同进程只读轮询。由此 `my_seat ≥ 1`，`build_pseudo_state` 的 `my_index` 报错路径（`state_builder.py:67`）天然规避，且**我的预留明牌**可见 |
| 2 | 呈现形式按 §1.1#4 执行 | 任务 T7.5（CLI）→ T7.6（仪表盘）→ T7.7（叠加，可选） |
| 3 | **对手预留的瞬间可以看到是什么牌**，同步记录；对方购买预留牌时及时更新 | 新增有状态模块 `ReservationTracker`（§3.2）：对手预留事件发生时抓取牌面，购买时核销；先跑 E7 实验锚定 DOM 证据（T7.2，阻塞后续）。这是对 BROWSER_RL_MAPPING"记忆重建不在最小回路"裁决的**advisory 侧**实现 |
| 4 | 引擎合法集 ⊆ 网页合法集 的处理按原建议 | 方向安全（建议必合法）+ `MaskParityMonitor` 告警上屏 + 牌堆预留动作缺位的显式局限声明（§3.6） |
| 5 | 深评用确定性重建（牌堆按注册表差集采样） | §3.3，且被一个新核实的事实**升级为必选项**（伪状态牌堆为空，见 §3.3 开头） |
| 6 | 分期按 v0 → v1 → v2 | §4 任务清单 |

### 1.3 非目标（Non-goals）

- **不做任何自动点击**：advisor 包禁止 import `browser.action_executor`（§6 不变式，测试强制）。
- 不改引擎、不改 `features.py`、不改 265 维 obs、不改训练链路；P4 记忆重建（RL 特征升级）仍按
  [docs/p4_decision.md](../docs/p4_decision.md) 暂缓——本阶段的 tracker 只服务面板展示与建议引擎，不进观测向量。
- 3–4 人局的**深评模式**延后（minimax 硬编码 2 人断言，`minmax.py:38`）；GA 快评与全部展示天然支持 2–4 人局。
- 不做胜负预测（phase-6 胜率估算器已覆盖，需要时可在仪表盘并列展示，非本阶段任务）。

## 2. 架构

### 2.1 进程模型

```
┌───────────────────────── 单进程 daemon：play-advisor ─────────────────────────┐
│                                                                              │
│  EgoBrowserDriver(task_space="splendor-advisor-<ts>", profile_id=…)          │
│      │ 启动/attach 一个 Chromium 窗口（用户在此窗口手动创建/进入房间并游玩）        │
│      ▼                                                                       │
│  只读轮询循环（0.4s；对手回合加密到 0.25s 抓预留瞬间，--poll 可调）                 │
│      extract_snapshot()  ──►  Snapshot（桌面牌/供给/双方面板/我的预留/状态文本）    │
│      │                       │                                              │
│      │                 ReservationTracker.update(snapshot)                  │
│      │                       │（对手预留事件记忆，§3.2）                        │
│      ▼                       ▼                                              │
│  state_fingerprint 变化？ ──否──► 复用缓存建议                                │
│      │是                                                                     │
│      ▼                                                                       │
│  AdvisorEngine（§3.3/§3.4）：确定性重建 → GA top-k ┐                           │
│                              → 牌堆直方图/可负担性 ┘→ AdviceSet（缓存）           │
│                                       │ 深算按钮 → 后台线程 minimax d=2/3      │
│      ▼                                                                       │
│  出口① CLI 文本（--no-dashboard）      出口② stdlib http.server → localhost 仪表盘 │
└──────────────────────────────────────────────────────────────────────────────┘
```

要点：

- **一个 daemon、一个窗口、零点击**。用户与页面的全部交互（建房、入座、选支付、选贵族、丢宝石）都由人手完成；
  daemon 的 `evaluate` 是 CDP 被动只读，不碰页面状态。
- task-space 命名用 `splendor-advisor-` 前缀，与 `play-web` 的空间互不干扰；同 profile 空间共享 cookie
  （`ego_driver.py:38-44`），advisor 默认独立 profile，避免占用用户的登录身份。
- 深评在后台线程跑，轮询循环永不阻塞；建议缓存按 `state_fingerprint`（`alphazero/state_utils.py:54`）命中。

### 2.2 复用清单（全部已核实）

| 能力 | 现成零件 | 位置 |
|---|---|---|
| DOM 抽取 | `extract_snapshot()` | `browser/dom_extractor.py:367` |
| 快照校验/离线回放 | `snapshot_from_raw(raw)`（接受任意 Mapping） | `browser/dom_extractor.py:446` |
| 伪状态 | `build_pseudo_state(snapshot, my_index, turns)` | `browser/state_builder.py:47` |
| 回合状态解析 | 三分支语法 + 过渡态折叠 | `browser/dom_extractor.py:55-77` |
| 合法动作枚举 | `rule.getLegalActions` + `create_action_mapping`（O(1) 索引缓存） | `gym/envs/utils.py:262` |
| 动作中文化 | `describe_action(action) -> str` | `play_vs_humans.py:106` |
| GA 评估器 | `GeneAlgoAgent`（构造时自动加载 4 个 `.npy`）+ `extract_metrics`/`normalize_metrics` | `genetic_algorithm/genetic_algorithm_agent.py:27-31`、`splendor/features.py:214/313` |
| 精确回滚 | `Transactor.apply/undo`（补齐 nobles 顺序与 last_action） | `alphazero/state_utils.py:115-146` |
| 确定性重建先例 | `_reconstruct_state` / `_unseen_cards`（对手预留按 tier 采样 + 牌堆重灌） | `remote/rollout.py:191/284` |
| 差异归因 | `MaskParityMonitor` + `dom_affordances` | `browser/monitor.py:65/162` |
| 仪表盘技术先例 | `play-dashboard`：stdlib `ThreadingHTTPServer` + 1.5s 轮询 `/api/state` | `remote/dashboard.py:24/158`（默认 8899，advisor 用 8900） |
| 只读观察先例 | `snapshot_listener` 钩子 + `LiveReporter.on_snapshot` | `browser/browser_env.py:72/272/298`、`play_vs_humans.py:230` |
| 离线夹具 | 8 个 HTML 场景（opening/payment_pills/empty_deck/…） | `browser/fixtures/*.html` |
| 卡牌注册表 | `CARD_REGISTRY`（deck_id 0 起）/ `NOBLE_REGISTRY`；引擎 `CARDS` 90 张（deck_id 1-3） | `browser/card_registry.py:32`、`splendor_utils.py:15` |

### 2.3 新增文件与入口

```
src/splendor/browser/advisor/
├── __init__.py        # 模块地图与"禁止 executor"说明
├── tracker.py         # ReservationTracker：对手预留事件记忆（§3.2）
├── engine.py          # AdvisorEngine：确定性重建 + GA top-k + minimax 包装 + 牌堆差集（§3.3/§3.4）
├── server.py          # ThreadingHTTPServer + /api/state + 深算触发（§3.5）
└── dashboard.html     # 静态面板，零 CDN 依赖（与 remote/dashboard.html 同目录风格）
src/splendor/play_advisor.py   # 入口 main()，镜像 play_web.py 的形态
pyproject.toml                 # [project.scripts] 增 play-advisor = "splendor.play_advisor:main"
tests/
├── test_advisor_tracker.py
├── test_advisor_engine.py
└── test_advisor_server.py
```

CLI 形态（镜像 `play-web` 参数风格）：

```
play-advisor [--room-url URL] [--task-space NAME] [--profile-id ID] [--port 8900]
             [--poll 0.4] [--topk 5] [--depth 2] [--seed S] [--no-dashboard]
```

- `--room-url` 缺省时打开主页由用户手动导航；advisor 检测到棋盘 DOM（`dealt` 非空且 `status` 命中回合语法）即开始工作，检测到 `游戏结束`（`DEFAULT_GAME_OVER_MARKERS`）即待机。
- `--no-dashboard` 为 v0 纯 CLI 模式；默认两者都开（CLI 打印精简版，仪表盘全量）。

## 3. 关键设计

### 3.1 只读观察回路

不实例化 `BrowserSplendorEnv`（它的 `step()` 绑定 executor），直接持有 `EgoBrowserDriver` +
`extract_snapshot` 自建轻量循环；`snapshot_listener`/`LiveReporter`（`play_vs_humans.py:230`）是同型先例，
但 advisor 不需要 gym 协议，直接循环更薄。每帧：

1. `extract_snapshot(driver)` → `snapshot_from_raw` 校验；
2. `ReservationTracker.update(snapshot)`（§3.2）；
3. `state_fingerprint` 与上帧比对，变化才重算建议；连续两帧一致才采纳新状态（**防动画中间态去抖**，
   与 `browser_env` 等待循环的轮询语义一致）；
4. `MaskParityMonitor.check(engine_mask, dom_affordances(snapshot))` 照常运行，`anomaly_count` 上屏。

### 3.2 预留记忆重建（`tracker.py::ReservationTracker`）

用户新增的核心需求（决策 #3）。**事实前提待 E7 实验锚定（T7.2）**：对手点击"预留"的瞬间，牌面在某处可见
（动画/日志/面板明牌一闪），存在一个可轮询捕获的时间窗。tracker 是纯事件记忆，不做规则推断（单一代码源纪律）。

**diff 语义表**（对相邻两帧 Snapshot 的面板差分；`PanelInfo` 见 `dom_extractor.py:152-159`）：

| 观测差分 | 推断事件 | 记忆动作 |
|---|---|---|
| 对手 `reserved_tiers` 数 +1，桌面 `dealt` 某格消失 | 预留**桌面明牌** | 牌面直接确定（消失那张 = 预留那张），无需 E7 |
| 对手 `reserved_tiers` 数 +1，`deck_counts` 对应层 −1 | 预留**牌堆顶** | 按 E7 锚定的 DOM 证据抓牌面；**抓不到 → 记"未知，tier 已知"** |
| 对手 `reserved_tiers` 数 −1，其 `card_counts` 某色 +1（或分数 +1） | 购买预留牌 | 核销记忆条目；若原为"未知"，购买后牌面自然公开（进入已购，牌堆差集自动正确） |
| 我的 `my_reserved` 变化 | 我的预留 | Snapshot 直接给牌面（`dom_extractor.py:171`），不依赖 tracker |

**接口**：`update(snapshot) -> TrackerDelta`；内部维护 `dict[seat, list[TrackedReserved]]`，
`TrackedReserved = {card: Card | None, tier: int, turn_no: int}`。

**边界情形**：

- **中段接入**（advisor 半途启动）：初始化为"每座位 N 张未知（tier 分布按 `reserved_tiers`）"，后续事件逐步补全；
- **新局重置**：自动（全部座位分数为 0 且 tracker 非空 → 判定新局，清空并记日志）+ 仪表盘手动"重置记忆"按钮；
- **漏检自愈**：购买预留牌事件使未知条目自然核销，因此漏检不会累积污染，只会短暂降低直方图精度；
- tracker 的记忆同时注入 §3.3 的确定性重建（替换 rollout.py 的"按 tier 盲采样"）与 §3.4 的直方图（未知条目单列灰桶）。

### 3.3 评估底座：确定性重建（本阶段最重要的技术裁决）

**新核实的事实（写文档前逐行确认）**：

1. 伪状态的 `board.decks = [[], [], []]`（`state_builder.py:83-89`，注释 F9：`getLegalActions` 不读牌堆，
   `deal()` 永不在伪状态上调用）；
2. 引擎的预留动作**只从桌面明牌枚举**（`splendor_model.py:503-520`，循环只遍历 `board.dealt`），
   不存在"预留牌堆顶"动作；
3. 引擎补牌走 `decks[deck_id].pop()`（`splendor_model.py:96-98`），空牌堆时静默不补。

**推论**：GA 逐动作打分与 minimax 若直接跑在伪状态上——(a) 买牌后继状态牌位不补，GA 特征里的桌面牌块
失真；(b) 永远无法评估"预留牌堆顶"这类网页真实存在的选项。因此**所有打分都在确定性重建状态上进行**，
这正是 phase-6 胜率估算器已验证的路线（`remote/rollout.py:191 _reconstruct_state`：对手暗预留按 tier
采样、牌堆用 `_unseen_cards` 重灌洗匀）。advisor 的差异与增强：

- **重建输入** = 伪状态 + tracker 记忆：对手预留槽位**已知牌面直接落位，未知条目才按 tier 采样**
  （比 rollout.py 的全盲采样信息更多；采样走 `random.Random(seed)`，seed 三件套纪律）；
- 未见过牌集合 = 注册表 90 − 桌面明牌 − 各家已购（`card_counts` 只提供各色数量，具体牌面由
  "未见过 = 全库 − 已见牌面" 收窄，与 rollout.py 同口径）− 我的预留 − tracker 已知预留；
- 重建状态仅用于**打分**，是每帧一次性的一次性工件（`Transactor.apply/undo` 保证逐动作精确回滚），
  显示与 tracker 仍以伪状态/Snapshot 为准，二者不混用。

### 3.4 建议引擎（`engine.py::AdvisorEngine`）

- **GA 快评（常开）**：在重建状态上枚举 `rule.getLegalActions` → 逐动作
  `apply → extract_metrics → normalize_metrics → gene.evaluate_state → undo`（照抄
  `genetic_algorithm_agent.py:82-96` 的回滚模式，但保留 `(action, value)` 列表取 top-k）。
  策略基因选择照 `ManagerGene.select_strategy`。成本 ≈ 动作数 ×（毫秒级），top-5 全程 < 0.5s。
- **minimax 深评（按钮触发，后台线程）**：不改 `minmax.py`（baseline 纪律），在 advisor 内实现根层
  α-β 包装：根层收集全部 `(action, value)` 排序，叶子用 `MiniMaxAgent(0)._evaluation_function`，
  递归模式镜像 `minmax.py:43-71`；**播种 `random`**（`minmax.py:57` 的 shuffle 导致原版平手排序不定）。
  默认 d=2（"我走 X，对手最狠回应 Y"），`--depth 3` 可调；座位数 ≠2 时按钮禁用并提示。
- **解释行**：GA 路径取 `weight × Δmetric` 绝对值 top-3 分量渲染（"分数 +2、蓝产能 +1"）；
  minimax 路径附"对手最狠回应：`describe_action(...)`"。
- **牌堆颜色直方图**：`CARDS` 90 按 tier 分桶 − 上述已见集合，输出 `{tier: {色: n}}` + 未知灰桶
  （tracker 漏检的预留）+ 每层剩余张数对照 Snapshot 的 `deck_counts` 做**守恒自检**（不一致 = tracker
  或抽取有漏，标黄上屏，这正是 phase-6 `diff_label` 的守恒差分思想）。
- **可负担性标尺**：对每张桌面牌与我的预留牌，`cost − (card_counts 产能 + gems)` 逐色算缺口，
  黄金可补的缺口标金；输出"还差 白1 蓝2（金可补）"。
- **贵族区**：Snapshot 的 nobles 列表 + 各座位对每张贵族的进度（cost vs `card_counts`）。
- **网页特有动作的显式局限（v1）**："预留牌堆顶"不在引擎动作枚举里，v1 直方图/标尺照常展示，
  但建议列表不会出现它；面板标注此局限。**v2 扩展（T7.4 可选开关，默认关）**：在重建状态上为每个非空
  tier 合成 `{"type": "reserve", "card": <重建牌堆顶 sampled Card>, ...}` 动作参与打分
  （动作 dict 形状与引擎一致，`describe_action` 需补一个无 `card_position` 的回退分支；
  因走的是自建打分循环而非 `create_action_mapping` 反查，不需要 3510 索引存在）。

### 3.5 仪表盘（`server.py` + `dashboard.html`，v1 主形态）

- 技术栈照 `play-dashboard` 先例：stdlib `ThreadingHTTPServer` + 轮询 `/api/state`，**零 CDN、零新依赖**
  （phase-6 用了 Chart.js CDN；advisor 的条形/直方图用纯 CSS 足够，离线可用）；
- `GET /api/state` → `{fingerprint, status, my_seat, players[], dealt, supply, nobles, my_reserved,
  advice[], deck_hist, tracker{known, unknown, coverage}, parity_warnings, elapsed_ms}`；
- `POST /api/deep` 触发深算（后台线程，返回时带 `depth` 与耗时）；`POST /api/reset_tracker`；
- 布局：顶部状态条（`等待你操作` 徽章 / 等待玩家N / 丢弃与贵族子流程提示 / 奇偶告警数 / tracker 覆盖率）；
  左列局面镜像（3×4 桌面 + 供给 + 双方面板，供人工校对同步）；右列建议 top-5（最优/次优标注、分数条、
  归因行、深算按钮）；下条牌堆直方图 + 贵族进度 + 可负担性标尺；
- 深算结果独立缓存（同指纹不重算），面板显示"深算中…"与耗时。

### 3.6 引擎合法集与网页合法集的差距处理

- 引擎掩码 ⊆ 网页合法集（T0.4 ADR）：**建议必合法（方向安全）**；网页特有的"少拿宝石/丢弃刚拿色"
  不在枚举中，奇偶监控的差异归因原样上屏，让用户知道差在哪；
- "预留牌堆顶"缺位是**引擎建模局限**而非奇偶问题，按 §3.4 的 v1 标注 / v2 合成处理；
- 支付方式：引擎只生成一种贪心支付，网页让人自选——advisor 建议"买 X"时按贪心支付打分，
  归因行注明"支付方式按默认贪心"；`payment_options` 非空（支付子流程）时面板提示"等待玩家自选支付，
  建议暂停"（ advisor 在子流程中不下建议）。

## 4. 任务清单

依赖链：`T7.1 → T7.2(E7) → T7.3 → T7.4 → T7.5 → T7.6 → {T7.7, T7.8}`；T7.7 可选。

| 任务 | 内容 | 关键验收 | 估时 |
|---|---|---|---|
| **T7.1** 骨架与只读回路 | `browser/advisor/` 包 + `play_advisor.py` + pyproject script；轮询循环、状态解析、去抖、game-over 待机；`--no-dashboard` 的最小打印 | 真实房间只读跑通一局不崩溃；夹具回放单测过；全程零点击 | 0.5d |
| **T7.2** E7 网页实验（阻塞 T7.3） | 实测对手"预留桌面牌 / 预留牌堆顶 / 购买预留牌"三个瞬间的 DOM 证据（要素：何处可见牌面、持续多久、选择器）；记录 `docs/web_experiments.md` E7 + ADR 增补；若需新抽取字段 → `dom_extractor` 增量（须过 `make parity`） | E7 记录含可复核的 selector 与截图；双开自对弈可复现 | 0.5~1d |
| **T7.3** ReservationTracker | §3.2 全部语义；中段接入；自动/手动新局重置；漏检降级 | 单测覆盖 diff 表四行 + 边界三分支（合成 Snapshot 序列，不依赖网络） | 1~1.5d |
| **T7.4** 确定性重建 + 建议引擎 | §3.3/§3.4；GA top-k；minimax 包装；直方图守恒自检；可负担性；解释行；（可选开关）牌堆预留合成动作 | 同状态同 seed 下 GA top-1 与 `GeneAlgoAgent.SelectAction` 一致；重建后 `deck_counts` 与直方图守恒；延迟基准留档（GA <0.5s、d=2 <5s）；seed 固定输出可复现 | 1.5~2d |
| **T7.5** CLI 文本模式（**v0 交付**） | top-5 中文建议 + 直方图 + tracker 状态 + 告警的终端排版 | 一局双人对局中人工使用，信息可读不刷屏 | 0.5d |
| **T7.6** 仪表盘（**v1 交付**） | §3.5 全部 | 夹具回放下 `/api/state` 200 且 schema 稳定（单测）；深算按钮线程安全；真实对局人工验收清单（§5）通过 | 2~3d |
| **T7.7** 油猴叠加（可选，v2） | ~20 行 userscript 注入 `http://localhost:8900` iframe；实测 HTTPS 页嵌 localhost iframe 的混合内容策略；失败即弃（回退双窗口），结论记入部署手册 | 游戏页内正常显示、无控制台报错；或明确记录"不可行 + 证据" | 0.5d |
| **T7.8** 文档收尾 | AGENTS.md（文档地图 + 常用命令）、WEB_DEPLOYMENT_GUIDE 新章、CODEBASE_PANORAMA §7、本文件状态改"已实现" | 文档地图可达、命令可复制执行 | 0.5d |

**合计 ≈ 7~9 人日**（T7.7 不计）。

## 5. 测试与验收（双轨制，同 plan/README 体系）

### 5.1 自动验收（进 CI）

- `make test` 全绿，新增三个测试文件：
  - `test_advisor_tracker.py`：diff 语义表逐行 + 中段接入 + 新局重置 + 漏检降级；
  - `test_advisor_engine.py`：GA top-1 与 `GeneAlgoAgent` 一致性（同一伪状态）、seed 确定性、
    守恒自检（人为抽走一张已购牌 → 直方图标黄）、`Transactor.apply/undo` 后状态逐位还原
    （复用 `state_fingerprint`）；
  - `test_advisor_server.py`：MockBrowserDriver + 夹具回放，`/api/state` 200 与 JSON 键集稳定；
    **源码断言：advisor 包不含 `action_executor` 引用**（grep 级测试，§6 不变式的机器化）。
- `make lint` / `mypy`（新代码全量注解）；若 E7 触碰 `dom_extractor` → `make parity` 必跑。
- 延迟门槛：GA top-5 < 0.5s、minimax d=2 < 5s（典型中盘状态，基准脚本与数字留档，参照
  `alphazero/benchmark.py` 的做法）。

### 5.2 人工验收（关键项）

真实房间 ≥3 局双人对局 checklist：

1. **礼仪终审**：全程 advisor 零点击（结合 5.1 的源码断言 + 房间内无异常动作记录）；
2. **tracker 准确率**：对手预留事件 识别数/发生数 ≥90%，漏检全部降级为"未知灰桶"且守恒自检不误报；
3. **建议质量抽检**：top-5 与人类直觉对照，归因行可解释；深算的"对手最狠回应"合理性抽查；
4. **同步校对**：局面镜像与网页逐项一致（含丢弃/支付/贵族子流程中的提示）；
5. **延迟体感**：轮询与建议刷新不卡顿、不干扰手动操作。

## 6. 不变式与纪律

- **零点击**：advisor 包禁止 import `browser.action_executor`；对真实服务器只有 CDP 被动只读 evaluate。
- **零行为变化**：不改 `play_web`/`BrowserSplendorEnv`/引擎/features；全部新增文件 + pyproject 一行。
- **单一代码源**：规则判断全部经引擎（伪状态 + `getLegalActions` + 注册表）；tracker 只记事件、不推规则。
- **确定性**：任何采样（牌堆重灌、未知预留采样、minimax 内部 shuffle）走 `random.Random(seed)`，
  CLI `--seed` 贯穿；同指纹同 seed 输出逐字节可复现。
- **诚实降级**：tracker 漏检 → 灰桶；深算失败 → 提示且不影响快评；奇偶异常 → 上屏不阻断。
- 新代码全量类型注解（mypy）、镜像既有命名风格；随机性遵守 AGENTS.md 事实 6。

## 7. 风险与开放问题

| # | 风险/开放问题 | 缓解 |
|---|---|---|
| 1 | E7 证据可能只闪现一瞬（< 轮询间隔），"预留牌堆顶"牌面漏检 | 对手回合加密轮询至 0.25s；漏检降级灰桶（不会污染）；开放问题：是否允许向页面注入 MutationObserver（写 `window.__advisor` 变量）——**默认否**（触碰"只读"纪律边界），如需启用走 ADR |
| 2 | 用户手点与 daemon evaluate 并发 | CDP evaluate 被动无副作用，理论无干扰；5.2-4 实测确认 |
| 3 | 多人局 GA 评分是分布外（本地训练全 2 人局） | 面板标注"评分在 2 人局内校准"；3-4 人局先只供直方图/标尺，不建议路线图 |
| 4 | minimax 深评 d≥3 超时 | 后台线程 + 前端"深算中"态；d=2 为默认；必要时加时间预算中断 |
| 5 | `card_counts` 只给各色数量，"未见过牌集合"收窄依赖抽取的已见牌面（同 rollout.py 口径） | 守恒自检兜底；如需精确到"对手手里不可能有的牌"属 P4 特征升级，不做 |
| 6 | 90 端口/任务空间冲突 | 端口默认 8900（play-dashboard 8899）；task-space 独立前缀 |
| 7 | 网页发版导致选择器失效（bundle 过期） | 第一动作重下 chunk（AGENTS.md 事实 15）；advisor 复用抽取层，修复自然下沉 |

## 8. 与既有计划的关系

- **P2/P3**：完全复用其产出，是其"部署 harness"的读侧衍生，不进入 RL 回路；
- **P4 记忆重建**：本 tracker 是 advisory 侧实现，若未来 P4 解冻，可将 tracker 的记忆接入特征 v2，
  但那是新的 ADR；
- **phase-6**：复用其确定性重建先例与仪表盘技术栈；advisor 的直方图守恒自检与其 `diff_label` 同源；
- **训练课程（T1.6）/ 50 局部署**：无依赖，可并行推进，互不阻塞。

## 9. 实施记录（2026-09-14）

| 任务 | 状态 | 提交 |
|---|---|---|
| T7.1 骨架与只读回路 | ✅ | `11b949f` |
| T7.2 E7 实验 | ⏳ 待执行（需真实浏览器，本轮明确不做）；tracker 的 `ReserveEvidence` 钩子已留位，默认回退灰色未知桶 | — |
| T7.3 ReservationTracker | ✅（含 E7 证据钩子设计） | `8731e90` |
| T7.4 确定性重建 + 建议引擎 | ✅（GA top-1 与 `GeneAlgoAgent.SelectAction` 一致性、minimax 与裸 deepcopy 链等值均有测试） | `43ff2db` |
| T7.5 CLI 文本模式（v0） | ✅ | `171ed86` |
| T7.6 本地仪表盘（v1） | ✅（stdlib http.server + 深算工作线程 + 奇偶告警上屏） | `ac3f7f0` |
| T7.7 油猴叠加 | ⏳ 可选，未实施 | — |
| T7.8 文档收尾 | ✅（本文、AGENTS.md、部署手册 §9、全景图 §7.10） | 本提交 |

偏离原计划的两点（均已记录在对应模块 docstring）：

1. **确定性重建从 v2 优化升格为必选底座**（§3.3 已预判，实施时确认）：伪状态牌堆为空
   （`state_builder.py` F9）且引擎不枚举牌堆预留动作（`splendor_model.py:503-520`），
   打分直接跑在伪状态上会令买牌后继的桌面补牌静默失效。
2. **`describe_action` 三件套平移**至 `splendor/splendor/action_text.py`（原位 re-export）：
   `play_vs_humans` 模块级 import torch，advisor 从它导入会违背"无推理需求"定位。

质量门：advisor 专属 48 项测试（observer/tracker/engine/cli/server，夹具离线）；
全量 pytest 377 passed + `test_feature_parity` 通过；advisor 文件 ruff/mypy 全绿。
