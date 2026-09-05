# Phase 5 · 工程化固化

> **定位**：把前四阶段的成果固化为仓库的一等公民——测试进 CI、命令进 Makefile、结论进正式文档。
> **节奏**：测试随各阶段同步编写（P0–P3 的自动验收项即是测试需求清单），本阶段是集中收尾与全量接入 CI。
> **估时**：2 人日（不含各阶段已随写随测的部分）。

## 1. 阶段目标

1. 全部离线测试进 CI，"不回归"从纪律变成机器保证
2. 常用操作命令化（Makefile 一键入口）
3. DQN 与 sim-to-real 结论进入仓库正式文档体系

## 2. 任务清单

| ID | 任务 | 产出 | 估时 |
|---|---|---|---|
| T5.1 | 测试目录与 CI | `tests/` + GitHub Actions workflow | 1d |
| T5.2 | Makefile 目标 | `lint / test / parity / train-dqn / play-web` | 0.25d |
| T5.3 | 文档更新 | README 导航 + ALGORITHM_COMPARISON 增列 | 0.75d |

## 3. 代码改动详解（说明 / 意义 / 原因）

### 3.1 `tests/` 目录与 CI workflow【新增】

**内容**：测试矩阵落地（IMPLEMENTATION_SPEC §6 的 9 类）：

| 测试 | 验证什么 | 阶段来源 |
|---|---|---|
| `test_action_index_cache` | 缓存与旧实现全等、索引双射 | P0 |
| `test_card_registry` | 90 卡 / 10 贵族命中且唯一、Card 字段一致 | P0 |
| `test_replay_buffer` | n-step 折扣、done 截断、环形覆盖 | P1 |
| `test_dqn_update` | Double DQN 目标的手工构造断言 | P1 |
| `test_dqn_smoke` | 2000 步训练不崩、loss 有限 | P1 |
| `test_feature_parity` | **obs + 掩码双奇偶逐位相等（≥1000 状态）** | P2 |
| `test_browser_adapter` | 离线夹具 → Snapshot → 伪状态 → obs 流水线 | P2 |
| `test_mask_parity_monitor` | 三类差异注入的归因正确性 | P2 |
| （网页实测 E1–E6） | 人工/半自动，不进 CI | P0 |

GitHub Actions：`ruff + mypy + pytest`（上表全部离线测试）；Python 3.12（`typing.override` 的硬约束）。

**意义**：当前仓库**零测试**——所有质量保障靠人工与运气。CI 把"不回归"变成机器职责。

**原因**：(a) 本计划改动了公共路径（`utils.py` 掩码构造），没有等价性测试的性能优化等于盲改——P0 的等价性测试就是为此而生，进 CI 后永久防回归；(b) **奇偶测试是 sim-to-real 的长期安全网**：页面改版、引擎演进、特征升级（v2）都会再次打破奇偶，它是唯一能在事故发生前报警的机制；(c) 测试与各阶段的"自动验收"是同一批东西的两面——验收清单就是测试需求清单，本阶段只是把它们从各阶段收拢进统一的 `tests/` 目录与 CI，无新增工作量。

### 3.2 `Makefile`【修改】

**内容**：新增目标 `make lint / make test / make parity / make train-dqn / make play-web`（后两者是薄封装，转发到对应 console script 与文档化参数）。

**原因**：仓库已有 Makefile 自动化文档生成等流程（仓库历史提交"automate formal procedures with makefile"），训练/评测/部署命令应遵循同一惯例；`make parity` 单列是因为它是部署前的强制检查项（"改了引擎或抽取逻辑后必跑"），一条命令降低执行摩擦。

### 3.3 `README.md` 与 `ALGORITHM_COMPARISON.md`【修改】

**内容**：README 增补：`plan/` 目录导航、四条命令闭环（`dqn` 训练 / `splendor` 本地评测 / `play-web` 网页部署 / `evolve` GA 对照）、环境要求（Python 3.12+）。ALGORITHM_COMPARISON 增补 DQN（本地）与 sim-to-real（网页）两列结果：与 GA / PPO / minimax 同表同口径对比。

**原因**：(a) ALGORITHM_COMPARISON 是仓库的正式结论表——DQN 结果必须以**同表同口径**进入才能与既有结论公平对比（此前 PPO 的结论"需重训"正是在此表确立，DQN 的成败要能在同一张表上被检验）；(b) 数字必须引用门槛口径（M3 的 100 局 ≥55% 等）与 3-seed 方差，杜绝 cherry-pick 单次好结果——这是 P1 人工验收 M1.3 的延续；(c) README 是新贡献者的第一入口，plan/ 目录与命令闭环不写进 README 等于不存在。

## 4. 自动验收目标

| ID | 验收项 | 判定标准 |
|---|---|---|
| A5.1 | CI 全绿 | ruff + mypy（新代码 strict）+ pytest 全部通过 |
| A5.2 | 测试覆盖关键路径 | 测试矩阵 8 项离线测试全部存在于 `tests/` 并在 CI 执行 |
| A5.3 | 命令可用 | `make lint/test/parity` 本地一键执行成功 |

## 5. 人工验收目标

| ID | 验收项 | 要点 |
|---|---|---|
| M5.1 | 文档审查 | README 导航准确、命令示例可复制执行；plan/ 各阶段文档与最终实现无漂移（文档里的签名/路径抽查核对） |
| M5.2 | 结论表审查 | ALGORITHM_COMPARISON 的 DQN 数字与门槛口径一致、含 seed 方差说明、无选择性引用 |
| M5.3 | 交接自查 | 一个新贡献者仅凭 README + plan/ 能在 30 分钟内跑通：加载 checkpoint → 本地评测 → 网页部署（以真人模拟或同事实测验证） |

## 6. 完成定义（整个计划的 Definition of Done）

当以下全部成立时，本升级计划视为完成：

1. P0–P3 双轨验收全部通过（P4 可选项按触发条件另行决策）；
2. `dqn` 训练的 checkpoint 在本地 vs minimax ≥55%、网页 50 局稳定；
3. sim-to-real 落差 <5 个百分点且归因清晰（或已立项对应的 P4 项）；
4. CI 全绿，`tests/` 覆盖测试矩阵；
5. ALGORITHM_COMPARISON 增列 DQN 结果，README 完成命令闭环文档。
