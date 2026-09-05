# AGENTS.md — Splendor-AI 仓库工作指南

> 本文件面向在本仓库工作的 AI 编码 agent。内容基于 2026-09-03 的全量源码调研，所有事实均已逐一验证。
> 项目升级计划见 [plan/](./plan/README.md)——**动代码前先读对应阶段文档**。
> 升级进度（dev 分支）：**P0 已落地**（索引缓存 / 身份注册表 / 环境协议 / 源文档勘误）；
> P1（DQN）与 P2（浏览器层）进行中，P3-P5 未开始。增量明细见 [CODEBASE_PANORAMA.md §7](./CODEBASE_PANORAMA.md)。

## 项目概述

《璀璨宝石》(Splendor) 桌游 AI 仓库，COMP90054 课程框架风格。三层架构：

| 层 | 位置 | 说明 |
|---|---|---|
| 游戏引擎 | `src/splendor/splendor/` | `splendor_model.py` 规则核心；完全信息博弈（`private_information = None`） |
| Gym 环境 | `src/splendor/splendor/gym/` | 把多智能体回合制折叠为单智能体 MDP；对手回合在 `step()`/`reset()` 内自动模拟；**P0 起有统一协议 `gym/base.py::SplendorEnvBase`** |
| Agent 层 | `src/splendor/agents/` | `generic/`（random 等基线）、`our_agents/`（PPO 家族 / minimax / 遗传算法） |
| 浏览器层 | `src/splendor/browser/` | 网页版（game.hullqin.cn/ccbs）适配：身份注册表已就位（P0-T0.2），DOM 抽取/伪状态/执行器按 plan/phase-2 落地 |

当前主线：按 `plan/` 实施「本地 DQN 训练 → 浏览器层部署网页版（game.hullqin.cn/ccbs）」的 sim-to-real 管线。

## 常用命令

```bash
# 本地对局评测（-a 是逗号分隔的模块导入路径，每个模块须暴露 myAgent）
splendor -a splendor.agents.our_agents.ppo.ppo_agent,splendor.agents.our_agents.minmax \
         --agent_names=ppo,minimax -t -m 10

ppo          # PPO 训练（console script）
evolve       # 遗传算法训练
# 计划新增：dqn（DQN 训练）、play-web（网页部署）——见 plan/phase-1、phase-3
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

## 代码修改纪律

- **不破坏存量路径**：PPO / GA / minimax 是 baseline 与课程对手，引擎与既有评测命令零行为变化（历史变更中唯一例外见 plan/phase-0 的索引缓存"显式报错"修正）。
- **单一代码源**：浏览器层的特征与规则判断一律复用引擎代码（经伪状态适配器），禁止在浏览器层重写规则（plan/phase-2 §3.3）。
- **新代码全量类型注解**（mypy），镜像既有文件的风格与命名。
- 随机性入口必须带 seed 三件套（见上文第 6 条）。
- 对真实服务器（game.hullqin.cn）只操作自建房间、点击频率保持人类量级——礼仪是硬约束（plan/phase-2 M2.5）。

## 测试

`tests/` 已建立（P0 起随阶段同步补充，最终矩阵见 plan/phase-5 §3.1）。运行：`.venv/bin/python -m pytest tests/`。

现有测试：`test_action_index_cache.py`（缓存等价性/双射）、`test_card_registry.py`（90 卡/10 贵族命中）、`test_env_protocol.py`（协议符合）。
核心质量门是**特征/掩码奇偶校验测试**（`test_feature_parity`，plan/phase-2 §3.4）——改动 `utils.py` 掩码逻辑、`features.py`、或浏览器抽取层后必须跑奇偶测试。

## 文档地图

| 文档 | 用途 |
|---|---|
| `CODEBASE_PANORAMA.md` | **初始代码库全景图**：59 文件 / 223 函数的完整盘点、分层架构、三条运行路径调用链 |
| `plan/README.md` | 升级计划总览 + 阶段依赖图 + 双轨验收体系 |
| `plan/phase-0..5-*.md` | 各阶段任务清单、代码改动说明（含意义与原因）、验收标准 |
| `plan/reference/UPGRADE_ROADMAP.md` | 架构裁决与对两份源文档的勘误（§1 的五项源码裁决必读） |
| `plan/reference/DQN_GUIDE.md` | DQN 算法完整方案（超参表、十二陷阱清单） |
| `plan/reference/BROWSER_RL_MAPPING.md` | 网页版 DOM/动作/奖励映射（实测依据） |
| `plan/reference/IMPLEMENTATION_SPEC.md` | 函数级签名规格与任务看板 |
| `ALGORITHM_COMPARISON.md` | 算法对比结论（GA 最稳、PPO 需重训——DQN 要超越的目标） |

关键裁决速记：牌库 **90 张**（40/30/20，"78"是发牌后剩余的误读）；四元组 `(tier, colour, points, cost)` 全库零重复；265 维观测天然与网页信息集对齐；支付方式是两侧唯一硬语义差距。
