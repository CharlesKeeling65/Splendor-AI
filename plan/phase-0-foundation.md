# Phase 0 · 地基与对齐

> **定位**：所有后续阶段的地基——性能优化、身份桥梁、接口契约、规则事实核查。本阶段不产出任何"智能"，但决定 P1–P3 的代码形态。
> **依赖**：无。**阻塞关系**：T0.4 阻塞 P2 全部；T0.1/T0.3 阻塞 P1/P2 部分任务。
> **估时**：2~3 人日开发 + 1~2 天网页实测（可与开发并行安排）。
> **函数签名**：见 [reference/IMPLEMENTATION_SPEC.md](./reference/IMPLEMENTATION_SPEC.md) §1。

## 1. 阶段目标

1. 消除动作索引查找的性能瓶颈（训练吞吐的前置条件）
2. 建立网页卡面/贵族 → 引擎对象的身份注册表（浏览器层的唯一桥梁）
3. 落地统一环境协议 `SplendorEnvBase`（sim-to-real 的接口契约，payment 维度前瞻预留）
4. 用 6 项网页实测把"规则奇偶"从猜测变成已裁决事实（P2 的前置门）
5. 勘误两份源文档，防止错误假设传导给后续实现者

## 2. 任务清单

| ID | 任务 | 产出 | 估时 | 依赖 |
|---|---|---|---|---|
| T0.1 | ActionIndexCache | `gym/envs/utils.py`（修改） | 0.5d | — |
| T0.2 | Card/Noble Registry | `browser/card_registry.py`（新增） | 0.5d | — |
| T0.3 | 环境协议 | `gym/base.py`（新增）+ `SplendorEnv.step` 加参数 | 0.25d | — |
| T0.4 | 网页规则实测 E1–E6 | `docs/web_experiments.md` | 1~2d | 浏览器 |
| T0.5 | 源文档勘误 | BROWSER/DQN 两份文档 | 0.25d | — |

## 3. 代码改动详解（说明 / 意义 / 原因）

### 3.1 `src/splendor/splendor/gym/envs/utils.py`【修改】ActionIndexCache

**改动内容**：新增三个成员——`action_key(a: Action) -> tuple`（可哈希键函数：五个字段转元组，gems dict → 排序元组且区分 `None` 与空 dict）、模块级 `ACTION_INDEX: dict[tuple, int]`（加载时一次性构建）、`_index_of(action_element) -> int`（查表，未命中抛带细节的 ValueError）。`create_legal_actions_mask` 与 `create_action_mapping` 内部的 `ALL_ACTIONS.index(...)` 全部替换为 `_index_of(...)`，**签名与返回值零变化**。

**说明**：`Action` 是含 dict 字段的 dataclass（不可哈希），Python 的 `list.index()` 退化为 `__eq__` 逐字段比较，3510 项线性扫描。键函数把动作身份压缩为可哈希元组后，查表变 O(1)。

**意义**：每步的掩码 + 映射构建从约 O(3510 × 22) 次字段级比较（~7 万次）降到 22 次哈希查表。这是本地训练吞吐的主要热点（DQN 需要 10⁵~10⁶ 步 × 每步两次调用）；浏览器环境每步真实成本更高（DOM + 网络等待），省下的 CPU 直接缩短单步延迟。

**原因**：(a) 性能是 P1 训练周期的硬约束；(b) **顺带修一个隐性 bug**——旧实现中若引擎动作不在 `ALL_ACTIONS` 里，`index()` 返回的是抛 ValueError（`list.index` 会抛）……准确地说旧代码同样会抛，但两处调用的异常信息不含动作细节且发生在最深的循环里；新实现把异常前置为带完整动作内容的 ValueError，且模块加载期的 `assert len(ACTION_INDEX) == len(ALL_ACTIONS)` 保证键无碰撞（若 ALL_ACTIONS 存在重复项，启动即暴露）；(c) 签名不变 → `splendor_env.py:146,184`、`ppo_agent.py:51,62` 等既有调用方零改动，PPO 路径零风险。

**回退**：纯内部优化，保留 `_slow_*` 旧实现供等价性测试对照，出问题可直接还原。

### 3.2 `src/splendor/browser/card_registry.py`【新增】卡/贵族身份注册表

**改动内容**：模块加载时从 `splendor_utils.CARDS`（90 张）与 `NOBLES`（10 个）构建两张字典——卡以 `(deck_id, colour, points, sorted_cost)` 四元组为键，值为与引擎 `initialGameState` 构造的卡**逐字段一致**的 `Card` 对象（code 串取自 CARDS 的键）；贵族以排序 cost 为键，值为 `(code, cost)` 元组。暴露 `lookup_card(deck_id, colour, points, cost)` / `lookup_noble(cost)`，未命中抛 KeyError 含输入四元组。网页 tier 行序（上→下 = tier 2/1/0）与引擎 `deck_id` 的方向转换在本模块收口。

**意义**：这是网页 DOM 世界与引擎对象世界的**唯一身份桥梁**。浏览器层后续的一切（特征向量、规则判断）都始于"网页上看到的这张卡是引擎里的哪张卡"。

**原因**：(a) 已实测四元组在全部 90 张卡上零重复、10 个贵族 cost 全部唯一——查表方案可行，且不依赖网页美术编号（`ccbs-img`）这类不稳定标识；(b) 键选**内容四元组**而非**位置**：桌面位置会因补牌而变，四元组是卡的不变身份；(c) 值直接构造引擎同构 `Card`（而非自造轻量结构）：保证 `vectorize_card` 与 `getLegalActions` 拿到的对象行为与本地完全一致，奇偶测试才有意义；(d) tier 转换收口一处：这是最容易埋 off-by-one 错误的地方，散落多处必然出错。

### 3.3 `src/splendor/splendor/gym/base.py`【新增】环境协议 `SplendorEnvBase`

**改动内容**：`@runtime_checkable` 的 `Protocol`，声明 `observation_space / action_space / reset / step(action, payment=None) / get_legal_actions_mask / get_payment_options` 五件套（精确签名见 IMPLEMENTATION_SPEC §1 T0.3）。

**意义**：这是"单接口、双环境"架构的合同。DQN 代码（L2）只依赖此协议类型，本地 `SplendorEnv` 与 `BrowserSplendorEnv` 都实现它——**训练好的 checkpoint 不改一行代码即可换环境运行**，这是整个 sim-to-real 计划的接缝。

**原因**：(a) 没有协议，两个 env 是类型系统里的无关类型，训练/部署代码必然分叉出 `if isinstance(env, ...)` 式分支，长期维护成本最高；(b) `payment` 参数**现在就留好**——P4 的支付维度升级若没有这个参数就是破坏性接口变更（要改所有调用方与所有已写测试），预留成本为零；(c) 选 `Protocol`（结构化子类型）而非抽象基类：`SplendorEnv` 已存在且不宜动继承结构，Protocol 让存量类**零改动**即满足契约——这是 Python 鸭子类型的静态化，专为"先有实现后有接口"的场景设计。

### 3.4 `SplendorEnv.step` 增加 `payment` 兼容参数【修改，~3 行】

**改动内容**：签名加 `payment: int | None = None`，实现体忽略该参数。

**意义/原因**：让 `SplendorEnv` 满足 3.3 协议。显式占位优于隐式缺省——"tier-1 不支持支付选择"成为代码里可见、可 grep 的事实，而非散落在注释里的口头约定。

### 3.5 `docs/web_experiments.md`【新增】六项网页实测记录

**改动内容**：E1 多贵族选择 UI / E2 >10 宝石返还子流程 / E3 终局结算 DOM / E4 空牌库 reserve 是否发金 / E5 **自愿少拿宝石** / E6 同色 7 张购卡上限。每项记录：操作序列、前后 DOM 快照、结论（与引擎一致/不一致）、截图。

**意义**：把 BROWSER_RL_MAPPING §8.2 的待办清单与 UPGRADE_ROADMAP §1.4 的规则奇偶风险，变成**已裁决的事实**，直接决定 P2 执行器与环境终局检测的实现分支。

**原因**：E5 是全计划最高风险项——引擎强制"手 ≤7 颗最少拿 min(3, 可用色数) 个"（`splendor_model.py:443-452`）是**非标准规则**（标准规则允许自愿拿 1~2 个）；若网页按标准规则实现，两侧合法动作集不一致：网页会出现掩码外的动作（我方永远不会选，白丢机会），策略价值被系统性低估。先实测再写代码，避免 P2 返工。E1/E2/E3 分别决定执行器的贵族分支、返还分支和环境的终局检测——这三个分支写错的表现都是"网页上卡死"，排查代价远高于一次实测。

### 3.6 两份源文档勘误【修改】

**改动内容**：按 `reference/UPGRADE_ROADMAP.md` §7 修正表更新：BROWSER_RL_MAPPING（牌库 90 张而非 78；265 维特征不含对手预留牌、记忆重建非必需；支付差距根因在引擎 `getLegalActions` 而非 `build_action`）与 DQN_GUIDE（step 签名前瞻参数；掩码走缓存路径）。

**原因**：错误假设会误导后续实现者。最危险的一条是"记忆重建是必需品"——它会让 P2 平白多出整个事件流模块（管线里最难的部件之一），而源码裁决已证明基线特征根本不需要它。

## 4. 自动验收目标

| ID | 验收项 | 判定标准 |
|---|---|---|
| A0.1 | 掩码/映射等价性 | ≥50 个随机引擎状态上，缓存版与 `_slow_*` 旧版输出 `np.array_equal` 全等 |
| A0.2 | 索引双射 | `len(ACTION_INDEX) == 3510` 且键互异（模块加载 assert + 单测副本） |
| A0.3 | 注册表完备 | 90/90 卡、10/10 贵族命中且键唯一；还原 Card 与引擎卡逐字段相等（含 code） |
| A0.4 | 协议符合 | mypy 通过；`isinstance(SplendorEnv(...), SplendorEnvBase)` 为 True |
| A0.5 | 存量回归 | `splendor -a splendor.agents.our_agents.ppo.ppo_agent,splendor.agents.our_agents.minmax --agent_names=ppo,minimax -t -m 5` 正常完成；`ppo` 训练 3 episode 冒烟正常 |

## 5. 人工验收目标

| ID | 验收项 | 要点 |
|---|---|---|
| M0.1 | E1–E6 实测完成 | 每项有截图、DOM 快照与明确结论；**E5 有裁决**（一致 / 需引擎加 `standard_take_rules` 开关） |
| M0.2 | 吞吐对比记录 | 改造前后 steps/sec 实测数字（验证 ≥50× 掩码构建提速的说法，留档） |
| M0.3 | 勘误复查 | 修正表逐项核对两份文档，确认无残留错误表述 |
| M0.4 | 规则差异 ADR | 若 E5/E6 判定不一致，写一页决策记录：引擎加开关 vs 接受差异，含取舍理由 |

## 6. 风险与回退

- 唯一行为变更是"未命中显式报错"（更严格的失败模式）；已排查无静默依赖方，异常情况可直接还原 `_slow_*` 实现。
- 网页实测依赖真实浏览器会话：只在自建房间操作，不进入真人房间（礼仪约束，见 P2/P3 的合规项）。
