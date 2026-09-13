# Splendor-AI 策略算法调研：启发式 / Minimax / DQN / PPO 的数学本质、代码实现与效果归因

> 日期：2026-09-12。基于 dev 分支全量源码核查（引用均带 `文件:行号`）、仓库已有实验数字
> （`docs/PPO_STABILIZATION_RESULTS_20260908.md`、`docs/DQN_EXPERIMENT_RESULTS_20260907.md`、
> `docs/DQN_ROUND2_RESULTS_20260907.md`、`docs/DQN_SEARCH_EXPERIMENTS.md`）以及博弈论/强化学习标准结论。
> 文献对照与取长补短见 [SPLENDOR_LITERATURE_SURVEY_20260912.md](./SPLENDOR_LITERATURE_SURVEY_20260912.md)；
> 据此制定的具体提升路径见 [IMPROVEMENT_ROADMAP_20260912.md](./IMPROVEMENT_ROADMAP_20260912.md)。

---

## 0. 摘要

1. **Splendor 的数学结构**决定了四类算法的成败：2 人局是二人零和完全信息有限博弈（极小极大定
   理适用），3/4 人局是排名博弈（定理失效）；真实游戏对牌库序部分可观察；奖励稀疏且"分数≠胜利"；
   动作空间 3510 维而每步合法动作只有约 10–40 个。
2. 五种实现（启发式/GA/minimax/DQN/PPO）本质上是**同一个问题——如何获得 Splendor 状态评估函数
   $V(s)$——在三根轴上的不同选择**：评估函数来源（手写 vs 学习）、决策前瞻深度（0/1/2-ply 搜索
   vs 价值隐含全程）、训练对手分布（固定脚本 vs 自博弈池）。
3. 当前数字（2 人局）：GA vs minimax **58%** 最强，corrected DQN **52.7%**、稳定化 PPO **48.7%**
   紧随，启发式**对全体强 agent 胜率 60–68%**（最反直觉的诊断信号）。所有结论都在**欠训练**下得出
   （DQN 只跑了默认预算的 5–10%，PPO 只跑了 128 局）。
4. 瓶颈排序：**训练量 > 目标函数对齐（终局/相对分差） > 训练对手分布覆盖 > 观测信息（缺公共宝石
   供给） > 算法本身**。3/4 人局训练量为零，只有结构兼容与明确标注的分布外代理值。

---

## 1. Splendor 博弈的数学形式化

### 1.1 博弈模型

Splendor 是一个**随机、回合制、多人扩展式博弈**，可形式化为元组

$$
\Gamma = \bigl(\mathcal{N},\ \mathcal{H},\ P(h),\ \mathcal{A}(h),\ \{u_i\}_{i\in\mathcal{N}},\ \mathcal{F}\bigr),
$$

其中 $\mathcal{N}=\{0,\dots,n-1\}$（$n\in\{2,3,4\}$）为席位集合，$\mathcal{H}$ 为历史（状态）集合，
$P(h)$ 决定当前行动者，$\mathcal{A}(h)$ 为合法动作集，$u_i$ 为 $i$ 的终局效用，$\mathcal{F}$ 为机会
（chance）机制——**发牌**：每层牌库初始洗牌后逐张抽取。

代码事实：动作空间为固定枚举 `ALL_ACTIONS`，$|\mathcal{A}|=3510$（`gym/envs/actions.py:222`）；
每步合法动作约 10–40 个（掩码 `create_legal_actions_mask`，`gym/envs/utils.py:226-242`）；终局条件
为任一 agent ≥15 分**且回合轮转到 0 号位**（`splendor_model.py:299-305`），另有全员 pass 死锁与
100 轮上限（`utils.py:9-25`）；支付方式固定为单一贪心（彩色优先、黄金补差，
`splendor_model.py:371-394`），这是与真实网页规则的唯一硬语义差距（方向安全，见
`docs/web_experiments.md` ADR）。

### 1.2 人数与支付结构：为什么 2 人局和 3/4 人局是不同的数学问题

- **$n=2$**：设胜/平/负效用为 $u\in\{+1,0,-1\}$（引擎 `calScore` 含 +0.5 少卡 tie-break，
  `splendor_model.py:307-323`），则 $u_1 + u_2 = 0$，博弈**严格零和**。由 Zermelo 定理与逆推归纳，
  有限二人零和完全信息博弈存在**纯策略最优解**，且

  $$\max_{\sigma_1}\min_{\sigma_2} u_1(\sigma_1,\sigma_2) \;=\; \min_{\sigma_2}\max_{\sigma_1} u_1(\sigma_1,\sigma_2).$$

  这条**极小极大定理**是 minimax 方法在 2 人局合法性的全部来源，也是 DQN/PPO 训练出的"最优响应"
  有良定义的原因。
- **$n\in\{3,4\}$**：终局效用是**名次置换**（第一名 1 分、其余 0 分，或 Borda 型），支付向量和不再
  为常数，博弈既非零和也非合作。此时：
  - 不存在极小极大定理；"所有对手联合针对我"（paranoid 假设）只是众多对手模型之一，且在 4 人局
    中通常过强（Sturtevant & Korf 的 multiplayer max\(^n\) 研究的标准结论）；
  - **kingmaking**：败者在其任一行动都无法改变自己名次时，其行动决定谁赢——"最优策略"依赖于对
    对手效用偏好的假设，均衡选择不良定义；
  - 隐性联盟（两人默契压制领先者）是策略的一部分，单一策略函数难以表达。

### 1.3 信息结构

引擎内 `private_information = None`（完全信息近似）。真实游戏隐藏的是**牌库顺序**（每层 40/30/20
张中仅顶层 4 张可见），对手预留牌在真实规则中其实是公开的。因此正确的搜索形式是对信息集的
**确定性化（determinization）**：对未见的"牌库余牌 ∪ 对手预留"联合分布采样。而学习器的观测比真实
信息集**更窄**：265 维观测（`features.py:470-485`，= 70 维指标 + 15 张卡 × 13 维）**不含公共宝石
供给**（`extract_metrics` 从不读 `board.gems`）、不含对手预留身份与对手宝石，且未归一化。观测公式
上，$\mathbf{o} = [\phi_{\text{metrics}}(s); \mathbf{c}_1;\dots;\mathbf{c}_{15}]\in\mathbb{R}^{265}$，
而博弈经济规划需要的 $\text{board.gems}$ 这一坐标在 $\phi_{\text{metrics}}$ 中缺失。

### 1.4 策略构念的定量表述

Splendor 的核心策略构念在数学上都是**多步反事实量**，一步奖励 $\Delta\text{score}$ 几乎测不到：

- **节奏（tempo）**：单位己方回合的期望得分产速 $\rho = \mathbb{E}[\Delta\text{score}]/\mathbb{E}[\Delta t]$。
  启发式是纯 tempo 选手——这解释了它对全体"引擎构建型"对手的胜率异常（§6）。
- **拒止（deny）**：动作 $a$ 对对手 $j$ 的拒止价值 $= V_j(s) - V_j(s \mid a)$，需要**对手价值函数**；
  当前所有观测中对手只有分数槽（`MAX_RIVALS=3` 个，`constants.py:21`、`features.py:169-170`），
  deny 的价值对学习器不可见。
- **宝石饥饿**：桌面 4/4/4/4/5（5 色+黄金）的公共池被吸取后收缩对手可达集——依赖公共宝石供给信息
  （观测缺失项）。
- **贵族竞速**：$\min_j \min_k \{t : \text{noble}_k \text{ 由 } j \text{ 获得}\}$ 的竞速，价值在
  $+3$ 分且不可交易。

---

## 2. 事实底座：已有对局矩阵与训练规模

2 人局头对头胜率（`docs/PPO_STABILIZATION_RESULTS_20260908.md:21-29`，每格 50–150 局，平局不计胜）：

| 模型 ↓ 对手 → | GA | heuristic | minimax | random |
|---|---:|---:|---:|---:|
| GA（作为选手） | 50% | 34% | **58%** | 100% |
| corrected DQN | 54% | 39.3% | 50% | 100% |
| 稳定化 PPO（fixed） | 54% | 32% | 48.7% | 100% |
| anchor PPO（KL 锚定） | 38% | 26.7% | 47.3% | 100% |
| 旧 PPO | 36% | 20% | 44% | 100% |
| scratch PPO（随机初始化） | 0% | 0% | 0% | 26.7% |

补充：DQN 第二轮实验最好配置 vs minimax 52.7%、vs random 99.3%
（`docs/DQN_ROUND2_RESULTS_20260907.md`）。

三个**训练规模**事实：

1. DQN 默认 200k 步（`dqn/constants.py:61`）从未跑满，正式数字全部来自 10k–20k 步短程实验
   （占默认预算 5–10%）；第二轮实验协议为 lr 5e-5、batch 128、buffer 30000、warmup 1000、
   ε: 1→0.05 over 8000 步（`docs/DQN_SEARCH_EXPERIMENTS.md:50-52`）。
2. 稳定化 PPO 每次训练 8 更新 × 16 局 = **128 局**（约 3900 transition），9 次训练总墙钟 27 分 31 秒
   （Quadro P5000）——算力不是瓶颈。
3. 结果文档自述：非等算力等数据的算法因果比较，且部分格子"训练对手=测试对手"
   （DQN 训 random/minimax；PPO 池含 GA/heuristic）。**跨对手泛化才是真指标**。

部署形态约束：网页对战每步只允许**一次前向 + 掩码 argmax**，无搜索；DQN PUCT 搜索实测 298ms/步
（贪心 6.1ms）且无胜率收益（`docs/DQN_EXPERIMENT_RESULTS_20260907.md:60-63`），且搜索不用于浏览器
伪状态（`search.py:5-6`）。

---

## 3. 启发式（`dqn/population.py:18-52` HeuristicAgent）

### 3.1 数学形式

无学习、无模拟的单步动作打分：

$$
a^* = \arg\max_{a\in\mathcal{A}(s)} f(a,s),\qquad f(a,s)=
\begin{cases}
3.0\cdot\mathbb{1}[\text{a 含贵族}] + 2p(c) + 1 + \dfrac{1}{1+N_{\text{col}(c)}} & a=\text{buy}(c)\\[2ex]
3.0\cdot\mathbb{1}[\text{a 含贵族}] + 0.2\sum_{g}\dfrac{n_g}{1+G_g + N_g} & a=\text{collect}\\[2ex]
-0.1 & a=\text{reserve}
\end{cases}
$$

其中 $p(c)$ 为卡分，$N_{\text{col}}$ 为已购同色卡数，$G_g,N_g$ 为持有宝石/已购卡数。两个设计点有
真实的领域含义：$1/(1+N_{\text{col}})$ 是**卡位边际收益递减**；$1/(1+G_g+N_g)$ 是**宝石边际递减**
（越接近可买，再拿同色宝石价值越低）。

### 3.2 为什么意外地强 / 为什么上限低

- **强**（对全体强 agent 胜率 60–68%）：它是唯一"每回合都在产出分数"的纯节奏型策略，客观上执行
  了被动宝石饥饿（桌面宝石被持续吸取）与恒定 tempo。而它的对手——minimax（评估函数奖励慢速引擎
  构建）、GA、DQN、PPO——全部未学过"对抗 rush 的防守"（预留拒止、抢关键卡、跟速）。这是**训练元
  策略分布覆盖缺失**的直接证据（文献佐证：MAP-Elites 行为空间研究，见文献调研 §3）。
- **上限低**：无长程价值（tempo 之外的一切）、无对手建模、行为完全确定可预测——任何带学习的策略
  理论上都能剥削它。定位：课程对手池中的"节奏风格样本"与压力测试，与仓库当前用法一致。
- **$n$ 玩家**：公式不含任何人数假设，天然 2–4 人可用——是全场唯一开箱即用 3/4p 的实现。

---

## 4. Minimax（`minmax.py`）

### 4.1 数学本质

对二人零和完全信息有限博弈，定义状态价值

$$V^*(s) = \begin{cases}
\displaystyle\max_{a\in\mathcal{A}(s)} V^*(T(s,a)) & s\text{ 轮到我}\\[1ex]
\displaystyle\min_{a\in\mathcal{A}(s)} V^*(T(s,a)) & s\text{ 轮到对手}\\[1ex]
u(s) & s\text{ 终局}
\end{cases}$$

逆推归纳保证 $V^*$ 存在且最优纯策略可取得极小极大值。α-β 剪枝在最优排序下把复杂度从
$O(b^d)$ 降到 $O(b^{d/2})$；实现为（`minmax.py:43-88`）固定深度 $d=2$ 的带剪枝递归 +
`generateSuccessor`/`generatePredecessor` 原地配对回滚。

评估函数是 $V^*$ 的**线性代理**（`minmax.py:90-136`）：

$$\hat V(s) = 2\,\text{score}_i + 0.7\,N_{\text{cards}} + c_G\,\textstyle\sum G
\;-\;0.2\,\mathrm{Var}(G)\;-\;\sum_{c}\big|cost(c) - (G_{\text{col}(c)}+N_{\text{col}(c)})\big|\cdot 0.1\,(p(c)+1+0.5\,r(c))$$

其中 $c_G=0.1$，但当 $\sum G\ge 8$ 时 $c_G=-0.7$（逼近持有上限时的囤积惩罚）；$r(c)$ 为卡 $c$ 对
贵族的相关度；终局项 $\pm 99999$。

### 4.2 为什么 2-ply 就能打 / 为什么天花板低

- **能打**：$d=2$ 恰好覆盖"我买→对手回应"的一阶交换与拒止效应；评估函数中的贵族相关加权与囤宝石
  惩罚正中要害；不学习、不漂移——面对学习型对手时**确定性本身是防御**。
- **天花板低**：
  1. **视野 2 ply**：中期引擎构建的长线价值不可见；线性 $\hat V$ 无法表达组合协同（某色卡+黄金+
     贵族的支付组合）。加深受 `getLegalActions` 内含 deepcopy 与原地 successor 拖累，深度翻倍成本
     指数级。
  2. **机会节点处理不正确**：正确的形式是期望极小极大（expectiminimax）
     $V(s)=\max_a \sum_{s'\sim P(\cdot\mid s,a)} V(s')$，或对信息集做多次确定性化。当前实现里
     `generateSuccessor` 内部从牌库随机抽一张——树中每个机会点**只采样了一个样本**，且采样偏差
     不可控：$\mathbb{E}[\hat V_{\text{1-sample}}] \ne V$，方差 $=\mathrm{Var}(V(s'))$。
  3. **$n\ge3$ 数学失效**：`minmax.py:38` 直接 `assert len(agents)==2`。极小极大定理不覆盖 3 人局；
     max\(^n\)（每个对手各自最大化自身效用）与 paranoid（所有对手最小化我）在 $n\ge3$ 给出不同的
     "最优"，且都不有定理保证；kingmaking 使均衡选择不良定义。
  4. 终局引导只在终局前 2 步出现，中期全靠线性代理。

**结论**：作为 2p 压力测试对手与领域知识载体已物尽其用；继续加深是理论天花板 + 算力 + $n$ 玩家失效
三重浪费。

---

## 5. DQN（`dqn/` 目录）

### 5.1 折叠式单智能体 MDP

环境把对手回合折叠进 `step()`/`reset()`（`splendor_env.py:238-250` `_simulate_opponents`）：对手为
构造时传入的 agent 列表（**完全可配置**）。学习器面对的等效 MDP 为

$$s_{t+1} = F(s_t, a_t) = \big(\text{chance draws}\big)\circ\big(\pi_{-1}\text{ 的 } n-1 \text{ 次响应}\big)\circ(s_t,a_t),$$

即一次"环境步"内含 $n-1$ 个对手动作与若干次发牌。奖励为

$$r_t = \text{score}_i\big(F(s_t,a_t)\big) - \text{score}_i(s_t)\quad(\text{仅自己分数增量，无终局信号，\texttt{splendor\_env.py:169}})，$$

DQN 的 `TerminalRewardWrapper`（`reward_wrapper.py:86-104`）在终局补

$$r_T \mathrel{+}= B\cdot z,\qquad z=\mathrm{sign}\big(\mathrm{calScore}(s,i)-\mathrm{calScore}(s,j)\big)\in\{-1,0,+1\},\quad B=10.$$

**数学必要性**：不加 $B$ 时，$\sum_t r_t$ 与"获胜"弱相关（15 分封顶后多拿分无意义），价值函数会收敛
到"慢速堆分"的伪最优。加上后优化目标对齐真效用（这正是 scratch/旧 PPO 与 corrected DQN 的分水岭）。

### 5.2 学习目标与收敛性

Bellman 最优算子 $\;(TQ)(s,a) = \mathbb{E}\big[r + \gamma \max_{a'} Q(s',a')\big]$ 在 $\gamma<1$ 时是
$\ell_\infty$ 上的 **γ-压缩映射**（$\|TQ_1-TQ_2\|_\infty\le\gamma\|Q_1-Q_2\|_\infty$），由 Banach
不动点定理，$Q \to Q^*$ 唯一收敛。实践中以 TD 误差
$\delta = r + \gamma\,\hat Q(s',a^*) - Q(s,a)$ 做自举。

- **Double DQN**：$a^*=\arg\max_{a'} Q_{\text{online}}(s',a')$，目标值用 $Q_{\text{target}}(s',a^*)$
  评估，解耦 argmax 与评估，缓解过估计。过估计的数学根源是 Jensen 不等式的直接推论：

  $$\mathbb{E}\Big[\max_{a\in\mathcal{A}} X_a\Big] \;\ge\; \max_{a\in\mathcal{A}} \mathbb{E}[X_a],$$

  噪声越大、$|\mathcal{A}|$ 越大偏差越大——3510 维动作空间使该偏差不可忽略。
- **n-step（n=3）**：目标改为
  $G_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n \max_{a'}\hat Q(s_{t+n},a')$，终局窗口
  **不自举**（修复 `78caed0`：非终局窗口恒 n 项、`gamma**n_step` 折扣；见
  `docs/DQN_SEARCH_EXPERIMENTS.md:8-10`）。稀疏奖励下把 TD 误差向前传播距离 ×3。
- **Dueling 分解**（`network.py:25-152`）：

  $$Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a'),$$

  去均值使 $V,A$ 可辨识。动机：$|\mathcal{A}|=3510$ 中绝大多数非法，状态价值共享、只学相对优势，
  样本效率远高于单头独立学习每动作（代码注释明示）。
- **掩码在 forward 内部**：$Q(s,a)= -\infty\ \forall a\notin\mathcal{A}(s)$（`network.py:177`），保证
  贪心 argmax、训练 gather、bootstrap max 三条路径都不可能落在非法动作上。
- **输入归一化**：观测未归一化，`InputNormalization`（运行均值/方差，随 checkpoint 保存）处理；
  训练模式单样本前向会使 1 样本方差退化为 0，故网络终身 eval 模式 + `observe()` 在线更新统计
  （`network.py:44-52,191-218`）。

### 5.3 为什么最强学习型 / 为什么不足

- **强**：off-policy + replay 样本可复用，10–20k 步即从 random 学出完整赢局（vs random 99.3%）；
  固定对手下"最优响应"收敛定义良好；Dueling+掩码对大动作空间是正确偏置；终局奖励对齐真目标。
- **不足**：
  1. **训练量只跑了默认的 5–10%**——一切"DQN 上限"结论都是欠训练下的结论。
  2. **best response ≠ Nash**：对手是固定脚本时收敛到 $\mathrm{BR}(\pi_{-i})$；replay 里的转移来自
     旧对手分布，对手一变目标整体过期（off-policy 的非平稳代价）。对分布外策略（GA、heuristic）
     掉到 50% 上下。
  3. **观测信息瓶颈**：$Q$ 看不到公共宝石供给，无法做"该色还剩几颗"的经济规划与拒止——特征工程
     问题，任何值函数都补不回（`public-v2` 312 维已实现：265 + 供给 6 + 对局面板 24 + 贵族费用 15
     + 席位/阈值 2，但硬编码 2 席，`docs/DQN_SEARCH_EXPERIMENTS.md:11-15`）。
  4. **折叠式信用分配**：奖励取"自己动作+对手全响应后"的分数差，对手干扰被折叠进自己的信用；
     $n$ 越大窗口内随机性越多，$\mathrm{Var}[r_t]$ 放大。
  5. **搜索路线已被证伪**（当前价值函数质量下）：`search.py` 为采样隐藏状态的 PUCT——每次模拟对
     "牌库余牌 ∪ 对手预留"重采样（信息集近似，非精确 AlphaZero），实测 298ms/步、无胜率收益。
     **搜索放大的是评估函数的误差**：先有价值，后有搜索。

---

## 6. PPO（旧版 `ppo/`；稳定化 `policy_imitation/ppo_selfplay.py`）

### 6.1 策略梯度与 clip 代理目标

策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, A^{\pi_\theta}(s,a)\big].$$

PPO 用重要性采样比率 $\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$
构造裁剪代理目标：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big],\qquad \epsilon=0.2,$$

总损失 $L = -L^{\text{CLIP}} + c_v\,\mathbb{E}_t[(\hat R_t - V_\theta(s_t))^2] - c_e\,\mathbb{E}_t[\mathcal{H}(\pi_\theta(\cdot\mid s_t))]$。
掩码处理为 $\pi_\theta = \mathrm{softmax}(\ell)\ \text{s.t.}\ \ell_a=-\infty\ \forall a\notin\mathcal{A}(s)$
（`ppo/network.py:114-116`，DQN 同理但作用于 Q）。

### 6.2 GAE 优势估计

定义 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，则

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l}.$$

推导要点：$\lambda\to1$ 退化为蒙特卡洛优势（低偏差、高方差）；$\lambda\to0$ 退化为 TD(0)
（高偏差、低方差）；$(\gamma\lambda)$ 指数加权在二者间做偏差-方差折中。稳定化实现终局感知：
完整对局批内跨终止边界重置递推（`ppo_selfplay.py:580+` "terminal-aware GAE"）。

### 6.3 旧版为什么失败、稳定化为什么有效

旧版（`ppo/training.py`）的结构性缺陷，每一条都有数学解释：

| 缺陷 | 机制 |
|---|---|
| 每局一条轨迹（≈60 transition）更新大网络 | 梯度方差 $\propto 1/|\mathcal{B}|$，信噪比灾难 |
| lr = 1e-6 | 单步参数位移过小，128 局内无有效学习 |
| 无终局奖励 | 优化 $\Delta\text{score}$ 而非胜负（§5.1 的必要性） |
| 只训 random | 收敛为"random 剥削者"，对其它分布零泛化 |
| policy network 含 dropout | 给策略分布注入无意义噪声 |

稳定化（`PPOConfig`，`ppo_selfplay.py:236-305`；正式运行参数见结果文档 §2）：lr 1e-4、minibatch 256、
4 epochs、$\gamma=0.99$、GAE $\lambda=0.95$、clip 0.2、entropy 0.005、value coef 0.5、grad norm 1、
target KL 0.02、**终局 ±10（calScore 口径，与 DQN 同口径）**、对手池 = GA/heuristic/current/history
各 25%（current=自博弈，history=最多 4 个冻结快照）、BC（DAgger-2）初始化点火、anchor 变体加
reference-KL 0.02。结果：+18/+12/+4.7 pp（对 GA/heuristic/minimax）但未全面超过 corrected DQN。

**关键诊断（结果文档自带）**：**价值解释方差仅 0.03–0.21**。策略梯度方法的方差近似
$\propto (1-\mathrm{EV})$——critic 这么弱等于 PPO 在和被废掉 critic 的自己比赛；advantage 全靠
GAE 硬扛。此外训练胜率与实力倒挂（scratch 训练胜 41–61/128、fixed 只 8–10/128）：对手池强弱混杂
且快照贪心/学习者采样不对称，**训练胜率不是实力指标**。

**scratch 0% 的数学解释**：128 局 × ~40 步 ≈ 5000 决策，3510 维掩码分布下随机探索几乎摸不到"赢局
+10"信号——稀疏终局 + 大动作空间下，**on-policy 纯探索冷启动不可行**。BC/蒸馏点火（fixed/anchor）
起步、自博弈接力是结构必需，不是工程偏好。

### 6.4 PPO 相对 DQN 的结构性优劣（为什么值得做主线）

- **自博弈无过期回放**：DQN 的 buffer 混着旧对手分布的转移（§5.3.2）；PPO 的样本永远来自当前
  混合策略，自博弈下的对手漂移是良性的。文献里 FSP（Heinrich & Silver 2016）与 PSRO
  （Lanctot 2017）的"策略迭代式外循环 + 最优响应内环"结构，PPO 是最自然的内环。
- **探索可控**：熵系数随训练调度（冷启动调大、后期退火）优于 ε-greedy 在掩码空间上的粗糙探索。
- **部署同构**：分布 → argmax = 单次前向，与网页部署形态完全匹配。
- **代价**：on-policy 样本效率低一个数量级；每步推理要算完整 3510 维 softmax（与 DQN argmax 同
  量级）。

---

## 7. GA（`genetic_algorithm/`，对照项）

- **基因型**：70 维指标上的线性权重 $w\in[-20,20]^{70}$ ×3 组（`StrategyGene`，`genes.py:89-100`）+
  ManagerGene $M\in\mathbb{R}^{70\times3}$ 按状态在 3 组策略间门控（`genes.py:103-125`）。
- **表型（决策）**：对每个合法动作做 1 步 successor 评估（`genetic_algorithm_agent.py:82-98`）：
  $$a^* = \arg\max_a\; w_{\sigma(s)}^\top \phi_{\text{norm}}(T(s,a)),\qquad \sigma(s)=\arg\max_k\; (M^\top\phi_{\text{norm}}(s))_k.$$
  即**进化出的 1-ply 线性评估器**——结构上是 minimax($d$=1) × 学习版 $\hat V$。
- **fitness**：种群 round-robin 对局的**累计分数**（`evolve.py:227-247`；`WINNER_BONUS=0`），且
  **天生打 4 人局**（`constants.py:11` `FOUR_PLAYERS`）。
- **为什么最稳**：搜索空间仅数百个连续参数；fitness 直接是"对局得分"——**优化目标与评测目标零
  mismatch**；种群天然是对手池，抗非平稳；无梯度需求，对稀疏奖励免疫。**对 minimax 58% 是全场
  唯一"训练时没见过 minimax 也赢"的数字**——泛化是真的。上限低在线性 $\hat V$ 的表达力。

---

## 8. 统一视角：三根轴

| | 评估函数来源 | 决策前瞻 | 训练对手分布 |
|---|---|---|---|
| 启发式 | 人工手写 | 0 步（直接动作打分） | 无 |
| GA | 进化学出（线性） | 1 步 successor | 种群自博弈（天生 4p） |
| minimax | 人工手写 | 2 步对抗搜索 | 无 |
| DQN | TD 自举学出（$Q=V+A$） | 0 步搜索（价值隐含全程） | 固定脚本 / 快照池 |
| PPO | 策略梯度学出（$V+\pi$） | 0 步搜索 | 自博弈 + 风格池 |

由此可解释全部主要现象：minimax 2 层仍能与学习型五五开（手写评估函数的知识密度 × 2-ply）；DQN
搜索实验无收益（$Q$ 不准时搜索放大误差）；GA 泛化最好（fitness 与评测目标零 mismatch + 种群自博弈）；
学习型对 heuristic 集体失灵（对手风格分布未覆盖 + 奖励无相对项）。

---

## 9. 2 / 3 / 4 人局的系统性差异

| | 2 人局 | 3/4 人局 |
|---|---|---|
| 博弈结构 | 零和，$\max\min=\min\max$ 良定义 | 排名博弈，kingmaking、隐性联盟、均衡选择问题 |
| 拒止/宝石饥饿 | 一对一明牌 | 多向竞争，reserve 价值上升，桌面宝石竞争者 ×2–3 |
| 折叠式信用分配 | 每步窗口含 1 个对手响应 | 含 2–3 个对手响应，$\mathrm{Var}[r_t]$ 放大 |
| 观测充分性 | 对手只有分数尚可容忍 | 对手买力/预留不可见的代价被人数放大 |
| 现状 | 全部训练与正式数字 | **训练为零**；仅结构兼容（env 接受 2–4 席、estimator 2..4 席）+ OOD 代理值；public-v2 硬编码 2 席 |

关键推论：3/4 人局不能直接复用 2p 模型（策略分布 OOD + 奖励语义错位），应 per-seat-count 训练；
奖励须从"胜负 ±10"改为**排名效用**（calScore 名次 → $\{+1,0,-1,-2\}$ 或排名优势函数），因为在 4p
里"第三 vs 第四"与"第一 vs 第二"是同一量级的问题。折扣也要复核：折叠 MDP 的 $\gamma$ 折扣的是己方
决策步，4p 折叠窗口变长后等效时间尺度 ×3，$\gamma=0.99$ 的有效视野相应缩短，应上调（如 0.997）。

---

## 10. 格局归因（五条）

1. **训练量**是第一解释变量：DQN 5–10% 预算、PPO <1%。scratch 0% 与 DQN 99.3% 之差主要是"有没有
   摸到终局信号"。
2. **目标函数对齐**是第二变量：有终局 ±10 的全部 ≥44%；没有的（旧 PPO）垫底；GA 的 fitness 直接
   对齐得分故最稳。
3. **对手分布决定泛化形状**：DQN vs minimax 52.7%（训练见过）> vs heuristic 39.3%（没见过 rush）；
   跨分布一致掉到 ~50% 是"学到对手特定响应而非游戏理解"的指纹。
4. **共享特征瓶颈**压住学习型上限：265 维缺公共宝石供给 = 蒙眼做经济规划。
5. **heuristic 异常**是 3+4 的联合后果：奖励是绝对分差无防御动机；对手池缺 rush 风格；决策无搜索
   无法局内切换元策略。

---

## 11. 代码索引（本报告引用）

| 事实 | 位置 |
|---|---|
| 奖励 = 己方 Δscore、无终局信号 | `gym/envs/splendor_env.py:163-169` |
| 对手折叠模拟（可配置 agent 列表） | `gym/envs/splendor_env.py:238-250` |
| 终局条件（15 分+轮转 / pass 死锁 / 100 轮） | `splendor_model.py:299-305`；`utils.py:9-25` |
| calScore tie-break | `splendor_model.py:307-323` |
| 动作枚举 3510 / 索引缓存 | `gym/envs/actions.py:222`；`gym/envs/utils.py:49-51,226-242` |
| 观测 265 维组成 / 缺公共宝石 | `features.py:156-181,470-485` |
| `MAX_RIVALS=3` 对手分数槽 | `splendor/constants.py:21`；`features.py:169-170,276` |
| HeuristicAgent 打分公式 | `dqn/population.py:18-52` |
| minimax 深度/断言/评估函数 | `minmax.py:18,38,90-136` |
| Dueling 网络、掩码、归一化 | `dqn/network.py` |
| DQN 终局包装 | `dqn/reward_wrapper.py:86-104`；`dqn/constants.py:54` |
| DQN 第二轮协议参数 | `docs/DQN_SEARCH_EXPERIMENTS.md:50-52` |
| PUCT 信息集搜索 | `dqn/search.py` |
| PPO 网络与损失循环 | `ppo/network.py`；`ppo/training.py`；`ppo/rollout.py` |
| 稳定化 PPO 配置与对手池 | `policy_imitation/ppo_selfplay.py:236-580` |
| 稳定化实验协议 | `policy_imitation/stabilization.py:48-118` |
| GA 基因/fitness/推理 | `genetic_algorithm/genes.py`；`evolve.py:227-247`；`genetic_algorithm_agent.py:101-121` |
