# AGENTS.md — Splendor-AI 仓库工作指南

> 本文件面向在本仓库工作的 AI 编码 agent。内容基于 2026-09-03 的全量源码调研，所有事实均已逐一验证。
> 项目升级计划见 [plan/](./plan/README.md)——**动代码前先读对应阶段文档**。
> 升级进度（dev 分支，2026-09-05）：**P0-P3 已落地**（索引缓存/注册表/协议/勘误 + DQN 六件套 +
> 浏览器层七件套含真实 DOM 实测回填 + play-web 部署 harness），**P4 经 ADR 裁决暂缓**，
> **P5 已完成**（CI/Makefile/文档）。训练课程（T1.6）与 50 局网页部署待训练条件解除后执行。
> 增量明细见 [CODEBASE_PANORAMA.md §7](./CODEBASE_PANORAMA.md)。

## 项目概述

《璀璨宝石》(Splendor) 桌游 AI 仓库，COMP90054 课程框架风格。三层架构：

| 层 | 位置 | 说明 |
|---|---|---|
| 游戏引擎 | `src/splendor/splendor/` | `splendor_model.py` 规则核心；完全信息博弈（`private_information = None`） |
| Gym 环境 | `src/splendor/splendor/gym/` | 把多智能体回合制折叠为单智能体 MDP；对手回合在 `step()`/`reset()` 内自动模拟；统一协议 `gym/base.py::SplendorEnvBase`（P0） |
| Agent 层 | `src/splendor/agents/` | `generic/`（random 等基线）、`our_agents/`（PPO 家族 / minimax / 遗传算法 / **DQN**） |
| 浏览器层 | `src/splendor/browser/` | 网页版（game.hullqin.cn/ccbs）适配：driver 协议 + ego-browser 适配器、DOM 抽取（真实页面实测校准）、伪状态、执行器、会话、奇偶监控、夹具 |

当前主线：按 `plan/` 实施「本地 DQN 训练 → 浏览器层部署网页版（game.hullqin.cn/ccbs）」的 sim-to-real 管线。

## 常用命令

```bash
# 本地对局评测（-a 是逗号分隔的模块导入路径，每个模块须暴露 myAgent）
splendor -a splendor.agents.our_agents.ppo.ppo_agent,splendor.agents.our_agents.minmax \
         --agent_names=ppo,minimax -t -m 10

ppo          # PPO 训练（console script）
dqn          # DQN 训练（Dueling + Double DQN + n-step replay）
evolve       # 遗传算法训练
play-web     # DQN checkpoint 部署到网页版对局（依赖 ego-browser CLI）
# 快捷入口：make test / make parity / make train-dqn / make play-web
```

环境：**Python 3.12+**（引擎用 `typing.override`，3.11 会 ImportError，尽管 pyproject 声明 >=3.11）；uv 管理（见 README_UV_SETUP.md）。
注意：`splendor` 命令的 displayer 依赖 **tkinter**——uv 托管的 Python 不带 Tk，需用 Homebrew Python 建 venv 并 `brew install python-tk@3.13`。

## Agent 接口约定（新增 agent 必须遵守）

- 继承 `splendor.template.Agent`，实现 `SelectAction(self, actions, game_state, game_rule) -> ActionType`（`template.py:59-68`）。
- **模块底部必须导出 `myAgent = XxxAgent`**——`general_game_runner.py` 动态加载的唯一入口。
- 参考 `agents/our_agents/ppo/ppo_agent.py` 的实现模板（特征 → 掩码 → 前向 → `create_action_mapping` 反查）。
- 新 RL agent 的目录结构、入口形态、checkpoint 约定**镜像 `ppo/` 对应文件**（详见 plan/phase-1 §3）。

## 必读事实（踩坑即崩的承重事实）

1. **非法动作直接崩溃**：`SplendorEnv.step()` 先查 `mapping[action]`，非法索引 KeyError 无兜底（`splendor_env.py:146-153`）。任何采样/argmax 前必须过合法掩码；ε-greedy 随机动作用 `np.random.choice(np.flatnonzero(mask))`。
2. **掩码获取**：训练路径 `env.unwrapped.get_legal_actions_mask()` → shape `(3510,)` 的 0/1 数组，每次 reset/step 后必须重新获取；对局路径 `create_legal_actions_mask(actions, game_state, agent_id)`。
3. **观测 265 维**（`Box(float32)`）= 70 维指标 + 15 张牌 × 13 维（12 桌面 + **自己的** 3 张预留）。**未归一化**（env 不调用 `normalize_metrics`）；**不含公共宝石供给**（`extract_metrics` 从不读 `board.gems`）；对手只暴露分数。
4. **动作空间 `Discrete(3510)`**：固定枚举 `ALL_ACTIONS`（`gym/envs/actions.py:222-237`）；`Action`/`CardPosition` 是含 dict 字段的 dataclass，**不可哈希**，`Card` 也不可哈希。
5. **奖励只有分数增量**（`Δscore`，`splendor_env.py:142-161`），无终局胜负信号——DQN 类 TD 算法必须加终局奖励包装（plan/phase-1 §3.4）。
6. **`env.reset(seed=)` 不固定发牌**：发牌走全局 `random`、座次走全局 numpy RNG。可复现必须三件套：`random.seed(s)` + `np.random.seed(s)` + `torch.manual_seed(s)`。
7. **引擎特殊规则**：手宝石 ≤7 时最少拿 `min(3, 可用色数)` 个（非标准规则！`splendor_model.py:443-452`）；同色已购 7 张禁买；买卡只生成**一种贪心支付**（彩色优先、黄金补差，`resources_sufficient`）。
8. **`generateSuccessor` 原地修改状态**，须配对 `generatePredecessor` 回滚（minmax.py 的搜索模式）或 deepcopy。
9. **性能**：动作索引查找已缓存化（P0-T0.1：`ACTION_INDEX` 查表，×75 提速）。热路径**禁止**再写 `ALL_ACTIONS.index()`——一律走 `create_legal_actions_mask` / `create_action_mapping`（内部 O(1)）；引擎动作若不在 `ALL_ACTIONS` 中会抛带细节的 ValueError（旧版为静默行为）。`getLegalActions` 内仍有 deepcopy。
10. `build_action()`（`gym/envs/utils.py:43`）不处理黄金通配，**不可**用于构造买卡动作；一律走 `create_action_mapping`。
11. **浏览器层实测事实**（T0.4，全部经真实页面验证，详见 `docs/web_experiments.md`）：覆盖按钮无任何 ccbs 类（纯 Tailwind+文本），必须走 `click_labelled`/`click_card_button`；供给筹码仅在取宝石模式是 `<button>`，全局查 `.ccbs-circle` 会先命中卡面费用；同步连点会丢点击（执行器逐颗慢速点击）；网页允许少拿宝石与丢弃刚拿色（引擎更严，方向安全）；终局=棋盘 DOM 消失回房间页。

## 代码修改纪律

- **不破坏存量路径**：PPO / GA / minimax 是 baseline 与课程对手，引擎与既有评测命令零行为变化（历史变更中唯一例外见 plan/phase-0 的索引缓存"显式报错"修正）。
- **单一代码源**：浏览器层的特征与规则判断一律复用引擎代码（经伪状态适配器），禁止在浏览器层重写规则（plan/phase-2 §3.3）。
- **新代码全量类型注解**（mypy），镜像既有文件的风格与命名。
- 随机性入口必须带 seed 三件套（见上文第 6 条）。
- 对真实服务器（game.hullqin.cn）只操作自建房间、点击频率保持人类量级——礼仪是硬约束（plan/phase-2 M2.5）。

## 测试

`tests/` 已随阶段同步建立，全量离线（CI 不碰网络）：`.venv/bin/python -m pytest tests/`（或 `make test`）。

测试矩阵（plan/phase-5 §3.1）：`test_action_index_cache`（P0 缓存等价/双射）、`test_card_registry`（P0 90 卡/10 贵族）、`test_env_protocol`（P0 协议）、`test_replay_buffer`/`test_dqn_network`/`test_dqn_update`/`test_reward_wrapper`/`test_dqn_smoke`（P1）、`test_feature_parity`（**P2 核心质量门：obs+掩码双奇偶 ≥1000 状态逐位相等**）、`test_browser_adapter`（P2 夹具流水线）、`test_mask_parity_monitor`（P2 归因）、`test_play_web`（P3 harness/回流）。
CI（GitHub Actions）：ruff + mypy（新代码路径）+ pytest，Python 3.12/3.13 矩阵。
**改动 `utils.py` 掩码逻辑、`features.py`、或浏览器抽取层后必须跑 `make parity`。**

## 文档地图

| 文档 | 用途 |
|---|---|
| `CODEBASE_PANORAMA.md` | **代码库全景图**：初始盘点（§1-§6）+ 升级增量记录（§7，随阶段更新） |
| `plan/README.md` | 升级计划总览 + 阶段依赖图 + 双轨验收体系 |
| `plan/phase-0..5-*.md` | 各阶段任务清单、代码改动说明（含意义与原因）、验收标准 |
| `plan/reference/UPGRADE_ROADMAP.md` | 架构裁决与对两份源文档的勘误（§1 的五项源码裁决必读） |
| `plan/reference/DQN_GUIDE.md` | DQN 算法完整方案（超参表、十二陷阱清单） |
| `plan/reference/BROWSER_RL_MAPPING.md` | 网页版 DOM/动作/奖励映射（实测依据；T0.5 勘误已就地标注） |
| `plan/reference/IMPLEMENTATION_SPEC.md` | 函数级签名规格与任务看板 |
| `docs/TRAINING_GUIDE.md` | **DQN 训练实操手册**（课程/参数/监控/验收，操作者视角） |
| `docs/WEB_DEPLOYMENT_GUIDE.md` | **浏览器部署与可视化手册**（人机对战/挂机/双开/旁观/回流） |
| `docs/REMOTE_DEPLOYMENT_GUIDE.md` | **本地控制 + 远程推理部署手册**（phase-6：inference-server / play-web-remote / play-dashboard，Z8 部署与隧道） |
| `docs/web_experiments.md` | **E1-E6 网页规则实测记录 + M0.2 吞吐留档 + 规则差异 ADR** |
| `docs/p4_decision.md` | P4 go/no-go 决策（全部暂缓，复审条件） |
| `docs/s2r_report.md` | sim-to-real 对照报告模板（数字待部署后回填） |
| `ALGORITHM_COMPARISON.md` | 算法对比结论（GA 最稳、PPO 需重训——DQN 要超越的目标） |

关键裁决速记：牌库 **90 张**（40/30/20，"78"是发牌后剩余的误读）；四元组 `(tier, colour, points, cost)` 全库零重复；265 维观测天然与网页信息集对齐；支付方式是两侧唯一硬语义差距。
**规则差异裁决（T0.4 实测）**：网页为标准规则（允许少拿宝石/丢弃刚拿色），引擎更严格——方向安全（引擎掩码 ⊆ 网页合法集），接受差异不改引擎；细节与监控白名单见 `docs/web_experiments.md` ADR。
