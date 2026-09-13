# Splendor 直接研究文献调研（arXiv）：训练方法、策略技巧与对本仓库的取长补短

> 日期：2026-09-12。检索方法：arXiv API 全字段 `all:"splendor"`（按提交日期倒序，30 条上限），
> 命中 12 条，其中**仅 4 条与桌游 Splendor 相关**（其余为暗物质探测器 SPLENDOR、菲舍尔市场
> equilibrium 问题名等，已逐一排除）。四篇均为公开摘要级核实（2026-09-12 抓取）；摘要未给出的
> 公式细节在本文中标注为"一般形式（依摘要与公开方法重构）"，未声称原文精确形式。
> 仓库内部算法分析见 [ALGORITHM_SURVEY_20260912.md](./ALGORITHM_SURVEY_20260912.md)；
> 据此落地的计划见 [IMPROVEMENT_ROADMAP_20260912.md](./IMPROVEMENT_ROADMAP_20260912.md)。

---

## 0. 文献全景

| # | 论文 | 作者 | 年份 | arXiv | 与本仓库的关系 |
|---|---|---|---|---|---|
| 1 | Rinascimento: Optimising Statistical Forward Planning Agents for Playing Splendor | Bravi, Lucas, Perez-Liebana, Liu | 2019 | [1904.01883](https://arxiv.org/abs/1904.01883) | 同域框架 + SFP/MCTS 基线与超参敏感性 |
| 2 | Rinascimento: using event-value functions for playing Splendor | Bravi, Lucas | 2020 | [2006.05894](https://arxiv.org/abs/2006.05894) | **奖励塑形**：直接对应本仓库"Δscore 稀疏"痛点 |
| 3 | Rinascimento: searching the behaviour space of Splendor | Bravi, Lucas | 2021 | [2106.08371](https://arxiv.org/abs/2106.08371) | **行为空间覆盖**：对应本仓库"打不过 rush 风格"异常 |
| 4 | Dynamic Resource Allocation for Ensemble Determinization MCTS | Kowalski, Ciężkowski, Krzyżyński, Winands | 2026 | [2607.13007](https://arxiv.org/abs/2607.13007) | **隐藏信息搜索**：直接对应 `dqn/search.py` 的理论缺陷 |

结论先行：arXiv 上 Splendor 的**直接研究就这四篇**（外加大量课程项目/学位论文未入 arXiv）。
四篇的共同主题恰好覆盖本仓库的三大痛点——稀疏奖励（#2）、策略分布覆盖（#3）、隐藏信息搜索（#4），
加上评测方法学（#1）。**这是一个非常幸运的对位：文献的每一块短板理论都能落在本仓库的一个已诊断
缺口上。**

---

## 1. Rinascimento 框架与统计前向规划基线（arXiv:1904.01883, CoG 2019）

### 1.1 问题设定

论文提出 **Rinascimento**：基于 Splendor 的"参数化部分可观察多人卡牌棋盘游戏框架"，规则/目标/
物品可参数化修改，作为通用博弈 AI 基准。与本项目的关系：**同一游戏、同一信息结构刻画**
（partially-observable + multiplayer）——佐证本仓库"引擎完全信息近似 + 牌库序隐藏"的信息模型
是学界共识。

### 1.2 方法：统计前向规划（SFP）基线

以 SFP 家族（MCTS 及其变体、带树复用/state 求值的滚动规划 agent）为基线，重点放在**超参数调优
对性能的巨大影响**——原文：超参调优"can heavily influence the performance"。

UCT/MCTS 的一般形式（标准文献公式，非该文原创）：

$$a^{\star} = \arg\max_{a\in\mathcal{A}(s)}\left[\bar Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}\right],$$

其中 $N(s)$ 为状态访问数、$N(s,a)$ 为边访问数、$\bar Q(s,a)$ 为平均回报名。树内回传用各自行动者
视角的效用（multiplayer 下每层视角不同，max\(^n\) 式回传）。

### 1.3 对本仓库的启示

1. **任何算法结论必须绑定超参与种子协议**：本仓库 9 次 PPO 训练 × 3 seed、DQN 5 版本 × 3 seed 的
   协议方向正确，但 10k–20k 步/128 局的训练量使"超参敏感"被系统性低估——先拉满预算再谈算法差异。
2. SFP 在同域是公认强基线：本仓库 minimax(d=2) + 确定性近似其实是"劣化版 SFP"；文献提示正确的
   强化方向是**确定性化采样 + 多次模拟**（见 §4），而非更深的全宽搜索。

---

## 2. Event-Value Functions（arXiv:2006.05894, CoG 2019/2020）

### 2.1 核心思想

针对 Splendor 的根本痛点——**点数奖励稀疏且后置**（原文：score-based reward 有 "severe limitations
when the point rewards are very rare or absent until the end of the game"）：

- 游戏状态**每次特征变化都触发一个事件**（原文："the game state triggers an event every time one
  of its features changes"）；
- 事件被 **Event-Value Function (EF)** 处理，"assigns a value to a single action or a sequence"；
- 调优后的 EF 能"synchronise the relevance of the events in the game"，缓解稀疏点数奖励并提升性能；
- 附带发现：用 EF 的 agent 在**多对手**对局中"show more robust"——与本仓库 3/4p 计划直接相关。

### 2.2 一般形式（重构）

设状态转移 $s\xrightarrow{a}s'$ 触发事件向量 $\mathbf{e}(s,a,s')\in\{0,1\}^K$（如"获得第 3 张同色
卡""拿走某色最后一颗公共宝石""预留对手可达的卡""达到贵族阈值"），EF 把塑形奖励定义为

$$\tilde r(s,a,s') \;=\; \Delta\text{score}(s,s') \;+\; \mathbf{w}^\top \mathbf{e}(s,a,s'),$$

其中 $\mathbf{w}$ 为（手工调优或搜索得到的）事件权重。学习目标随之从"总分"变为"事件价值 + 分数"的
累计和。与 Ng et al. (1999) 的 potential-based 塑形不同，EF 是**事件驱动的非势能塑形**——不保证
最优策略不变，其正确性由"事件权重编码领域知识"来保证，因此**事件权重的调优协议是该方法的核心
成本**（论文强调 EF 能被"tuned"来综合事件相关性）。

### 2.3 与本仓库现状的对位

| 本仓库现状 | 文献对位 |
|---|---|
| 基础环境 $r=\Delta\text{score}$，无终局信号（`splendor_env.py:169`） | EF 明确诊断的问题："分数奖励稀疏或直到终局才出现" |
| DQN/PPO 终局 $\pm 10$（calScore 口径） | EF 的最粗粒度特例：仅"终局事件"有权重 |
| 无中间事件奖励 | EF 的主战场：拿宝石/买卡/达贵族/拒止等中间事件 |

**取**：事件塑形思路 + "对多对手更鲁棒"的经验证据。**补（本仓库需自行解决的）**：事件集与权重的
调优协议——建议用 potential-based 形式（$\phi(s)$ 取 calScore − 对手最大 calScore + 小系数贵族
进度/买力）先保证策略不变性，再叠加少量事件项做实验对照（见 ROADMAP 阶段 B2）。

---

## 3. MAP-Elites 行为空间搜索（arXiv:2106.08371, 2021）

### 3.1 方法

用 **MAP-Elites**（quality-diversity 算法）在 Rinascimento AI agent 的**超参空间**中搜索，把每个
配置映射到由若干**行为度量**（behavioural metrics）定义的 **行为空间（BSpace）** 网格 archive，
每格保留精英。数学骨架（MAP-Elites 标准形式）：

$$\mathcal{A}[b(x)] \leftarrow \arg\max_{x:\, b(x)=\text{cell}} f(x),$$

其中 $x$ 为 agent 配置、$b(\cdot)$ 为行为描述子（把配置投到行为网格）、$f(\cdot)$ 为适应度；
**coverage** = 被占用格数 / 总格数。

### 3.2 发现

1. **覆盖度**："the use of event-value functions has generally shown a remarkable improvement in the
   coverage of the BSpace compared to agents based on classic score-based reward signals"——事件价值
   函数让 agent 的行为覆盖显著优于纯分数奖励（注意：这里强调的是覆盖/多样性优势，不是直接的对战
   胜率优势）。
2. **设计诊断**：方法"able to highlight both exemplary and degenerated behaviours in the original
   game design of Splendor and two variations"——即 Splendor 策略空间**存在多个行为域**（急速抢分
   型 / 引擎构建型 / 拒止型…），且能自动暴露退化行为。

### 3.3 与本仓库现状的对位

本仓库的"heuristic 异常"（§ALGORITHM_SURVEY §10.5）正是该发现的镜像：所有学习型/GA/minimax 都
收敛在"引擎构建型"行为域，而 heuristic 是"急速抢分型"域的样本，前者没学过后者的对策。
**取**：①对手池必须显式做**风格覆盖**（把 heuristic 变体加入池，而非当作弱基线陪练）；②评测端
引入**行为度量**（每局记录：买卡数、预留数、拿宝石次数、达 15 分的回合数、对手达 15 分回合数），
使"行为域覆盖"可度量、可回归——这直接修正"只看胜率的单维评测"。**不照搬**：完整 MAP-Elites 搜索
流程（成本高，且本仓库当前缺的是"覆盖一个池"而非"搜索整个空间"）。

---

## 4. Ensemble Determinization MCTS 与动态资源分配（arXiv:2607.13007, 2026）

### 4.1 方法

针对"随机性 + 隐藏信息"博弈（Jaipur / Lost Cities / **Splendor** 三域基准）：并行维护**多个确定性化
树**（每个树对隐藏信息——Splendor 中即牌库序与对手预留——采样一个确定化），并做两个维度的动态
资源分配：

- **动态确定性化数量**："increases or decreases the number of currently used determinization trees
  depending on the behavior of so-far search"——按根节点动作排序的稳定性增减树数；
- **动态模拟分配**："splits the simulation budget nonuniformly across the determinization trees"，
  逐模拟决策把预算给"potentially the best knowledge gain"的树。

一般形式（重构）：对信息集 $\mathcal{I}(h)$ 采样 $m$ 个确定化 $z_1,\dots,z_m$，在每棵树上跑 UCT，
根动作值聚合

$$\bar Q(a) = \frac{1}{m}\sum_{j=1}^{m} Q_j(a)\quad\text{（或按访问数加权）},$$

树间预算分配本身可建模为一个 bandit 问题（哪个树的知识增益最大）。

### 4.2 与本仓库 `dqn/search.py` 的对位

本仓库已有实现是**单样本信息集 PUCT**：每次模拟对"未见牌库 ∪ 对手预留"联合重采样
（`sample_hidden`，`search.py:40-59`），PUCT 带策略先验、有界深度（`search.py:64+`；协议见
`docs/DQN_SEARCH_EXPERIMENTS.md:27-33`：每 8 个决策搜索一次、16 次模拟、辅助头权重 0.2）。实验
结论：298ms/步 vs 贪心 6.1ms，**无胜率收益**。

文献对位的诊断：
1. 本仓库实现把 $m=16$ 次模拟**分散在单棵树内、每次重新采样**——等价于"每次模拟一个新确定化但
   不复用树"，比 ensemble-of-determinizations（每确定化一棵树、树内复用、跨树聚合）信息利用效率
   更低；
2. **价值函数未标定**是更根本的原因（结果文档 §4.5 已自我诊断："本轮无界 return critic 不是胜率
   预测器"）——搜索放大 $V/Q$ 的误差；
3. 文献证明在**价值/先验质量足够**时，确定性化系 MCTS 在 Splendor 上可以显著变强
   （"particular configurations yield a statistically significant increase in the algorithm's strength"）。

**取**：确定性化系搜索作为**未来离线教师/分析工具**的正确形式（多确定化树 + 树内复用 + 预算
bandit）。**不照搬**：不作为网页部署决策器（单次前向约束 + 延迟实测 298ms/步不可接受）。

---

## 5. 相邻方法学文献（非 Splendor 专属，供方案设计引用）

| 文献 | 与本仓库的接口 |
|---|---|
| Ng, Harada & Russell 1999（potential-based shaping） | 奖励塑形**策略不变性**定理：$F(s,s')=\gamma\phi(s')-\phi(s)$ 不改变最优策略集——本仓库加中间奖励时的安全形式 |
| Fictitious Self-Play（Heinrich & Silver 2016） | 自博弈收敛到 Nash 的经典框架：最优响应内环 + 历史混合外环；PPO 池中 current/history 桶即其简化版 |
| Policy Space Response Oracles（Lanctot et al. 2017） | 把"对手池 + 最优响应"形式化为元博弈求解；league 评测与 exploiter 的理论基础 |
| AlphaStar（Vinyals et al., Nature 2019） | 多 agent 联赛训练的工程形态：主代理 + exploiter + 历史 league——对手池风格化的成熟参照 |
| Suphx（Li et al. 2020, arXiv:2003.13590） | 4 人麻将：**全局奖励预测**（把终局排名预测作为辅助信号）+ 运行时策略微调——本仓库 3/4p 排名奖励与 critic 辅助头的直接参照 |
| DouZero（Liu et al. 2021, arXiv:2106.06135） | 3 人斗地主：深蒙特卡洛 + 分布式 actor——"大动作空间 + 排名效用"规模化训练的参照 |
| Double/Dueling DQN、PPO/GAE 原始文献 | 本仓库已实现；不再展开（见 ALGORITHM_SURVEY §5–§6） |

---

## 6. 取长补短总矩阵

| 文献机制 | 本仓库对应缺口（证据） | 落点 | 优先级 | 取/不取 |
|---|---|---|---|---|
| Event-value 中间事件奖励（#2） | Δscore 稀疏 + 绝对分差无防御动机（heuristic 异常） | ROADMAP 阶段 B2：potential-based φ(s) 先行 + 少量事件项对照 | **高** | 取思想；权重用势能形式保证策略不变性，事件项做对照实验 |
| MAP-Elites 行为覆盖（#3） | 对手池风格单一；评测只看胜率 | 阶段 C3 对手池风格化 + 阶段 A1 评测记录行为度量 | **高** | 取"覆盖"思想与行为度量；不取完整 QD 搜索流程 |
| Ensemble Determinization MCTS（#4） | `search.py` 单样本 PUCT 理论缺陷 + critic 未标定 | 阶段 F（最后）：先标定价值，再做多确定化树对照，仅作离线教师 | 低（后置） | 取形式修正；不取部署用途 |
| SFP 超参敏感性（#1） | 训练量不足导致超参敏感被低估；无 league 评测 | 阶段 A 评测协议 + 全阶段"先拉满预算再下结论"原则 | **高（协议性）** | 取方法学纪律 |
| Suphx 全局奖励/排名预测 | 3/4p 无排名效用 | 阶段 E2 rank utility + 可选排名预测辅助头 | 中 | 取 |
| FSP/PSRO/AlphaStar league | current/history 池已有雏形，缺 exploiter 检验 | 阶段 C4 定期 exploiter 评测（固定对手矩阵即轻量 exploiter） | 中 | 取轻量版 |

**不照搬清单**（含理由）：
1. Rinascimento 参数化框架本身——本仓库引擎已固定且经 parity 门禁锁定，重写引擎零收益。
2. EF 作为"行为控制器"（论文另一用途）——本仓库不缺行为控制，缺泛化与防御。
3. 纯 MCTS/搜索型 agent 作为部署决策器——网页部署单次前向约束 + 298ms/步实测。
4. 完整 AlphaZero 式自博弈搜索训练——critic 未标定前必然放大误差（结果文档 §4.5 自我诊断一致）。

---

## 7. 参考文献清单

1. Bravi, Lucas, Perez-Liebana, Liu. *Rinascimento: Optimising Statistical Forward Planning Agents for
   Playing Splendor*. CoG 2019. arXiv:1904.01883.
2. Bravi, Lucas. *Rinascimento: using event-value functions for playing Splendor*. CoG 2019/arXiv 2020.
   arXiv:2006.05894.
3. Bravi, Lucas. *Rinascimento: searching the behaviour space of Splendor*. 2021. arXiv:2106.08371.
4. Kowalski, Ciężkowski, Krzyżyński, Winands. *Dynamic Resource Allocation for Ensemble Determinization
   MCTS*. 2026. arXiv:2607.13007.
5. Ng, Harada, Russell. *Policy invariance under reward transformations: Theory and application to
   reward shaping*. ICML 1999.
6. Lanctot et al. *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning* (PSRO). NeurIPS 2017.
7. Heinrich, Silver. *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (FSP). 2016.
8. Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. Nature 575, 2019.
9. Li et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning*. arXiv:2003.13590, 2020.
10. Liu et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*. ICML 2021. arXiv:2106.06135.
11. Schulman et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. / *High-Dimensional
    Continuous Control Using GAE*. arXiv:1506.02438.
12. van Hasselt et al. *Deep Reinforcement Learning with Double Q-learning*. AAAI 2016. / Wang et al.
    *Dueling Network Architectures*. ICML 2016. / Mnih et al. *Human-level control through deep
    reinforcement learning*. Nature 518, 2015.
13. Sturtevant, Korf. *On Pruning Techniques for Multi-Player Games*. AAAI 2000（max\(^n\) 与 paranoid 搜索）.
