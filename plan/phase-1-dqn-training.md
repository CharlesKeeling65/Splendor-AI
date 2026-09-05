# Phase 1 · DQN 本地训练

> **定位**：在本地引擎上把 DQN 从零训到超越现有 baseline，产出可部署的 checkpoint。与 P2 **完全并行**。
> **依赖**：T0.1（缓存，训练吞吐前提）、T0.3（协议，代码形态前提）。
> **估时**：4 人日开发 + 3~5 天挂机训练（M1→M3 课程）。
> **函数签名与训练循环骨架**：见 [reference/DQN_GUIDE.md](./reference/DQN_GUIDE.md) §6 与 [reference/IMPLEMENTATION_SPEC.md](./reference/IMPLEMENTATION_SPEC.md) §2。

## 1. 阶段目标

1. 建成 DQN 全栈（网络 / 缓冲 / 奖励 / 训练 / 对局五件套），工程形态**镜像 PPO**（降低维护者切换成本）
2. 通过三阶段课程门槛：M1 学会买分 → M2 碾压 random → M3 超越 minimax
3. 产出可复现的 stats.csv 与 3-seed 方差数据（防单 seed 虚假结论）

## 2. 任务清单

| ID | 任务 | 产出 | 估时 | 依赖 |
|---|---|---|---|---|
| T1.1 | DQN 网络 | `dqn/network.py` | 0.5d | T0.3 |
| T1.2 | Replay Buffer | `dqn/replay_buffer.py` | 0.5d | — |
| T1.3 | 奖励包装 | `dqn/reward_wrapper.py` | 0.25d | — |
| T1.4 | 训练核心 | `dqn/training.py` | 1d | T1.1–1.3 |
| T1.5 | 入口与对局 Agent | `dqn/dqn.py` `dqn/dqn_agent.py` `dqn/utils.py` + pyproject | 1d | T1.4 |
| T1.6 | 训练与课程 | checkpoint + stats.csv | 3~5d 挂机 | T1.5 |

课程安排（T1.6）：

| 里程碑 | 对手 | 步数预算 | 门槛（不过不进下一阶段） |
|---|---|---|---|
| M1 | random | 2×10⁵ | vs random 平均终局得分 >10 |
| M2 | random(50%) + minimax(50%) | 3×10⁵ | vs random 胜率 >90%（20 局滑窗） |
| M3 | 三者混合 + 快照自博弈 | 5×10⁵ | vs minimax 100 局 ≥55% |

## 3. 代码改动详解（说明 / 意义 / 原因）

### 3.1 `agents/our_agents/dqn/constants.py`【新增】超参数集中

**内容**：全部超参（γ=0.99 / lr=1e-4 / batch=512 / buffer=5×10⁵ / warmup=5000 / τ=0.005 / ε 调度 1.0→0.05 前 20% / n_step=3 / win_bonus=10 / 课程对手配比 / seed）。

**意义/原因**：镜像 `ppo/constants.py` 的仓库惯例。所有实验调参只动一个文件——超参与代码版本可对齐是复现实验的前提；散落的魔法数字会让"哪个 checkpoint 用哪组超参"变成考古。

### 3.2 `agents/our_agents/dqn/network.py`【新增】QNetwork（Dueling）

**内容**：`InputNormalization` → 4 × [Linear(128) + LayerNorm + ReLU]（**无 Dropout**）→ V 头 `Linear(128→1)` + A 头 `Linear(128→3510)`，Dueling 聚合 `q = V + A − mean(A)`；`forward(obs, mask)` 内 `masked_fill(mask==0, -1e9)`；`act()` 贪心助手；`raw_q()` 调试接口。

**说明**：掩码在 forward 内应用（与 PPO `network.py:114-115` 的 `-1e8` 手法一致）。掩码后 `-1e9` 不进损失：训练只 gather 已执行动作（数据保证合法），bootstrap 的 max 掩码后必落合法动作。

**意义**：策略的载体。Dueling 结构把"局面价值 V(s)"与"动作优势 A(s,a)"分离——3510 个动作中多数在多数状态非法，单头 Q 要对每个动作独立学价值，Dueling 让 V(s) 只学一次、A 只学相对差异。

**原因**：(a) trunk 照搬 PPO 已调通的 [128×4]+LayerNorm——这是仓库里唯一经过训练验证的架构，把架构风险降到最低；(b) **去掉 Dropout(0.2)**：Dropout 在 PPO 里正则化的是 on-policy 整局批量，在 DQN 的 off-policy minibatch 上则是纯噪声源（同一批样本来自不同 episode 不同策略，本就无过拟合单批的问题）；(c) **lr=1e-4 而非 PPO 的 1e-6**：PPO 的极小 lr 配合整局批量与 clip 是合理配置，但 DQN 用 minibatch + replay，1e-6 会让训练慢到不可用——这是"镜像结构但不可镜像超参"的典型例子；(d) 掩码进 forward 而非调用点：保证任何路径（act / 训练 / bootstrap）都拿不到非法动作值，杜绝"某个调用点忘了 mask"这类灾难事故。

### 3.3 `agents/our_agents/dqn/replay_buffer.py`【新增】ReplayBuffer + n-step

**内容**：numpy 环形数组均匀缓冲，存 `(obs, action, reward, next_obs, next_mask, done)` 六元组（float32）；n-step 由前置滑动队列实现（窗口满 n 或遇 done 时折叠为一条 n-step 转移，`R = Σγ^k r_k`，s' 取队尾，中途 done 截断）。**不存当前步 mask**。

**意义**：off-policy 的核心部件——打破轨迹相关性、跨 episode 混采，这是 DQN 相对 PPO 的本质优势（也是后文"网页经验回流"的基础）。

**原因**：(a) 一局仅 ~50 个决策步且高度相关，无缓冲的梯度会被单局主导（PPO 用整局批量 + clip 硬扛这个相关性，DQN 用缓冲优雅解决且样本效率高一个量级）；(b) **n-step=3**：奖励稀疏（多数步 r=0，只有买卡/贵族/终局非零），1-step TD 要几十步才能把终局 ±10 传回开局，n-step 直接缩短信用分配链条；(c) **不存当前 mask**：训练只 gather 已执行动作（必合法，无需当前掩码），存了是纯冗余——3510 × 4B × 5×10⁵ ≈ 7GB 内存，省掉一半以上存储；(d) 环形数组 + float32：预分配零拷贝，5×10⁵ 容量 ≈ 2.6GB，普通开发机可承受。

### 3.4 `agents/our_agents/dqn/reward_wrapper.py`【新增】TerminalRewardWrapper

**内容**：`gym.Wrapper`；`step` 透传后，若 `terminated` 则用 `calScore` 口径计算 ±win_bonus（赢 +10 / 平 0 / 输 −10），叠加在原有 Δscore 之上。

**意义**：**DQN 成败的第一关键**（DQN_GUIDE §5.4 的原话，也是本模块存在的原因）。

**原因**：(a) 现有奖励只有分数增量（`splendor_env.py:142-161`），**没有胜负信号**——MC 回报（PPO）尚可容忍，TD bootstrap 没有终局信号时 Q 只能学到"剩余得分估计"，学不到"赢面"；一个 14 分领先但必输的局面和 14 分稳赢的局面在 Δscore 奖励下无差别；(b) **胜负口径必须沿用 calScore**（含平局 +0.5 买卡数裁定）：评测路径 `general_game_runner` 用同一函数判胜负，训练与评测口径不一致会出现"训练赢、评测输"的假象且极难排查；(c) 用 **Wrapper 而非改 env**：不侵入引擎路径，PPO 等既有用户零影响，且同一 wrapper 之后可以原样套在浏览器环境上（协议一致的红利）；(d) **win_bonus=10**：与单卡 0~5 分、贵族 3 分的量级拉开——终局信号主导回报但不淹没中间塑形（太小则策略学成"刷分"，太大则中间信号丢失、探索困难）。

### 3.5 `agents/our_agents/dqn/training.py`【新增】训练核心

**内容**：`DQNParams` 数据类、`epsilon_at`（线性衰减）、`collect_one_step`（ε-greedy 采集 + 入库 + 终局 reset）、`dqn_update`（Double DQN：在线网选动作、目标网评估；Huber 损失；梯度裁剪 1.0；软更新 τ）、`evaluate`（贪心策略 N 局，calScore 口径统计胜率，每局独立建 env）。

**意义**：算法心脏。

**原因**：(a) **Double DQN 是必要保险而非可选优化**：3510 头 + 稀疏奖励下，max 算子的过估计会被 bootstrap 反复放大（每个动作头的乐观噪声都进入目标值）；Double 解耦"选动作"与"评估"切断这一正反馈；(b) **软更新（τ=0.005）而非硬更新**：目标网络平滑漂移，对 M3 自博弈阶段的非平稳性更鲁棒；(c) **ε 随机动作必须 `np.random.choice(np.flatnonzero(mask))`**：直接 `randint(3510)` 以 99%+ 概率非法，`step()` 会在 `mapping[action]` 处 KeyError 直接崩溃（`splendor_env.py:146-148` 无兜底）——这是本仓库对 DQN 最不友好的一点，也是 DQN_GUIDE 陷阱清单第一条；(d) **evaluate 每局重建 env**：座次与发牌随机（且 `reset(seed=)` 不固定发牌，需全局 random seed），独立建 env 保证评估在座位维度上平均，防止"只会后手"的假象；(e) 采集与更新同循环交替（每步一次梯度更新）：实现最简且样本效率最高，不引入不必要的异步复杂度。

### 3.6 `agents/our_agents/dqn/dqn.py`【新增】训练入口

**内容**：`train(**options)` + `main()`（argparse）；对手工厂**复用** `ppo/arguments_parsing.OPPONENTS_AGENTS_FACTORY`（random/minimax/ppo/...）；stats.csv（step/episode/ε/loss/q_mean/train_score/eval_wr）；checkpoint 命名 `dqn_model_{step // save_every}.pth`；seed 三件套在入口统一设置。

**意义**：训练命令成为一等公民（`dqn` console script），实验管理有标准形态。

**原因**：(a) **镜像 ppo.py 的入口布局**（train kwargs + main + stats.csv + 时间戳目录）：仓库维护者已熟悉这套形态，认知成本最低；且 stats.csv 字段设计与 PPO 同源，未来画对比曲线可以直接拼表；(b) **对手工厂 import 复用而非复制**：对手注册表只维护一份，PPO 侧新增对手 DQN 自动获得；(c) checkpoint 命名用显式整除——修掉 `ppo.py:293` 的 `episode + 1 // N_TRIALS` 优先级 bug（该 bug 会让 checkpoint 互相覆盖，PPO 侧遗留但 DQN 侧不再复制）；(d) 自博弈选项（M3 用）：对手 = `DQNAgent(load_net=False)` + 共享网络对象，镜像 `ppo.py:177-184` 的先例。

### 3.7 `agents/our_agents/dqn/dqn_agent.py`【新增】对局 Agent

**内容**：`DQNAgent(Agent)`，`SelectAction` 镜像 `ppo_agent.py:32-64`（`extract_metrics_with_cards` → `create_legal_actions_mask` → `net.act` → `create_action_mapping[...]`）；底部 `myAgent = DQNAgent`。

**意义**：接入 `game.py` 评测路径的唯一入口（`general_game_runner` 动态加载 `myAgent`）。

**原因**：**两条路径用同一网络与同一特征管线**（gym 训练路径 / runner 评测路径）是评测有效性的前提——若评测路径另写一套特征组装，训练与评测的观测分布悄然不同，胜率数字失去意义。镜像 PPO agent 是保证这一点的最省力方式。

### 3.8 `agents/our_agents/dqn/utils.py`【新增】save / load

**内容**：`save_model`（`model_state_dict` + `running_mean/var`（形状 `(1,265)`，沿用 PPO 约定）+ `step` + `config`）；`load_saved_dqn`（默认加载包内 `dqn_model.pth`，镜像 `ppo/utils.py:36-47` 的 squeeze(0) 处理）。

**原因**：**InputNormalization 的 running stats 是模型的一部分**——不随 checkpoint 存取，部署时的归一化与训练时不同，Q 值整体失真且症状只是"变菜了"，极难排查。形状沿用 PPO 的 `(1, D)` 约定便于未来共享工具函数；`weights_only=False` 与 PPO 加载方式一致。

### 3.9 `pyproject.toml`【修改】console script

**内容**：`[project.scripts]` 增 `dqn = "splendor.agents.our_agents.dqn.dqn:main"`。

**原因**：与 `ppo`/`evolve` 并列的仓库惯例；训练命令可发现、可文档化（README 的训练章节可直接引用）。

## 4. 自动验收目标

| ID | 验收项 | 判定标准 |
|---|---|---|
| A1.1 | 单元测试 | n-step 折扣累计手工断言、done 截断语义、环形覆盖；Double DQN 目标的手工构造断言 |
| A1.2 | 冒烟测试 | 2000 步训练：loss 有限、buffer 正常填充、无异常抛出 |
| A1.3 | M1 门槛 | vs random 平均终局得分 >10（≥100 局脚本统计） |
| A1.4 | M2 门槛 | vs random 胜率 >90%（20 局滑窗稳定达标） |
| A1.5 | M3 门槛 | vs minimax 100 局胜率 ≥55% |
| A1.6 | 可复现性 | 同 seed 重跑，stats.csv 前 1000 行逐行一致 |
| A1.7 | 时限达标 | `dqn_agent` 在 game.py 路径（`FREEDOM=False`，1s/步 + 15s 首回合）无 timeout warning 跑完 10 局 |

## 5. 人工验收目标

| ID | 验收项 | 要点 |
|---|---|---|
| M1.1 | 训练曲线审查 | loss 与 q_mean 无发散/爆炸模式；win_rate 上升是渐进的——**突然跳到 100% 通常是 bug**（如奖励泄漏、评估用了训练状态）而非突破 |
| M1.2 | 对局观战（5 局） | `splendor -a ...dqn_agent,...random -t` 或 GUI 观战：行为定性合理——按目标卡颜色囤宝石、优先买分卡、临 15 分收敛、会为高价值卡预留。赢率达标但行为荒谬（靠 exploit 对手 bug）不算通过 |
| M1.3 | 3-seed 方差审查 | M3 门槛须在 ≥2/3 个 seed 上达成；单 seed 达标而方差大时**不得宣布通过** |
| M1.4 | checkpoint 抽查 | `load_saved_dqn` 加载后 vs random 胜率与训练末期 evaluate 一致（防保存/加载管线 bug——特别是 running stats） |
| M1.5 | 反直觉行为登记 | 记录观战发现的奇怪决策（如永不预留、忽视贵族），作为 P4 特征升级（如供给维度）的输入证据 |

## 6. 风险与回退

- **不收敛**：按 DQN_GUIDE §9 的十二陷阱清单排查——九成失败源于奖励设计与输入归一化，而非算法本身。
- **胜率虚高**：检查 evaluate 是否意外共享了训练 env 的状态（座位/发牌污染）。
- **与 PPO 的关系**：DQN 是新主线而非替换——PPO/GA/minimax 全部保留为 baseline 与课程对手，本阶段零改动它们。
