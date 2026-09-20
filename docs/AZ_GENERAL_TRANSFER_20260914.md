# 从 alpha-zero-general 借鉴：PPO/DQN 现状对照与可迁移思路

> 日期：2026-09-14。参考仓库：
> **[cestpasphoto/alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general)**
> （CPU 优化的通用 AlphaZero，含 Splendor 2/3/4 人局预训练模型；
> 技术细节见该仓库 [README_features.md](https://github.com/cestpasphoto/alpha-zero-general/blob/master/README_features.md)）。
>
> 本仓库自身的 AZ 失败归因见 [Z_PHASE_IMPLEMENTATION_20260913.md](./Z_PHASE_IMPLEMENTATION_20260913.md)；
> 算法数学本质见 [ALGORITHM_SURVEY_20260912.md](./ALGORITHM_SURVEY_20260912.md)；
> 提升路径主计划见 [IMPROVEMENT_ROADMAP_20260912.md](./IMPROVEMENT_ROADMAP_20260912.md)。
>
> **定位**：本文是“外部优秀工程/算法思路 → 本仓库 PPO/DQN 主线”的迁移备忘，
> **不是**重启完整 AlphaZero 的实施计划。主线不变：PPO 自博弈为主，DQN 为副线与价值/回流通道，
> 搜索仅在价值可用后作离线放大器。

---

## 0. 一句话结论

`alpha-zero-general` 成功的关键不是“更会写 PUCT”，而是：

1. **为搜索重写规则层**（Numba 原地棋盘、81 动作、~3000 rollouts/s/core）；
2. **用课程把 value 从弱抬到可用**（100→200→400→800 sims）；
3. **大量工程增益叠加**（对称增广、FPU、强制 playout、KL loss、universes）。

对本仓库的含义：

- **不能**把对方 AZ 管线原样搬来——本仓库动作空间 3510、引擎 deepcopy、AZ 价值头零净贡献（Z3 gate 42.7%）。
- **应该**把其中与 PPO/DQN 瓶颈正交的思路抽出来，按“吞吐 / 信号 / 采样覆盖 / 训练目标”四类挂到现有阶段（B/C/D/F）。

---

## 1. 本仓库 PPO / DQN 优劣势（证据化）

### 1.1 DQN

| 维度 | 优势 | 劣势 / 实验证据 |
|---|---|---|
| 样本复用 | off-policy replay，10–20k 步即可从 random 学出赢局 | buffer 混旧对手分布，对手一变目标过期 |
| 动作空间 | Dueling + 掩码对 3510 稀疏动作是正确偏置 | 掩码下探索仍粗糙；非法动作零兜底 |
| 目标对齐 | `TerminalRewardWrapper` 终局 ±10 对齐胜负 | 原生奖励只有 Δscore；折叠式信用分配放大噪声 |
| 部署形态 | 贪心一次前向，与 `play-web` DQN loader 同构 | 浏览器伪状态不能开搜索 |
| 规模结论 | 修正管线（experiment.py）曾达 vs minimax **52.7%** | plain CLI 200k 步 M3 **失败**（均值 11%）——**量不能替代管线** |
| 价值质量 | `policy_value` 双头机制已落地 | F1 标定 **AUC 0.525**，不可作搜索先验；D3 蒸馏 2/60 vs 22/60 |
| 搜索协同 | sampled-hidden PUCT / 多树确定化机制已入库 | 实测 **298ms/步、无胜率收益**；搜索放大的是评估误差 |

**定位**：副线。价值先验、网页回流、与 PPO 对照；不是当前棋力主升通道。

### 1.2 PPO（稳定化 / C2-R2 主线）

| 维度 | 优势 | 劣势 / 实验证据 |
|---|---|---|
| 自博弈 | 当前策略采样，无过期回放；对手漂移良性 | on-policy 样本效率低一个数量级 |
| 对手覆盖 | 风格化池（GA/heuristic×3/minimax）可扩展 | **heuristic 族仍是唯一 G1 缺口**（58.0% vs 60%） |
| 目标对齐 | 终局胜负 + potential shaping（κ=0.05） | 策略梯度方差受 critic 质量制约 |
| 规模效应 | 2000 updates × shaping 后 minimax/GA 首次过 G1 | C2-R2 未拆分规模 vs 塑形单因素 |
| Critic | C1 后 EV@u1000 可达 0.40–0.43 | 自博弈下 EV 非实力单调代理（u2000 回落） |
| 部署 | 分布 → argmax 与网页单次前向同构 | **play-web 当前只接受 DQN checkpoint**，PPO 需适配器 |
| 冷启动 | DAgger-2 初始化有效 | scratch 冷启动 0%，结构必需 BC 点火 |

**定位**：主线。C2-R2 已证明「规模 × 塑形 × 风格化池」是当前最强升力。

### 1.3 共同瓶颈（无论 PPO 还是 DQN）

1. **动作空间 3510**：学习与搜索都在稀疏离散空间上硬扛。  
2. **观测信息**：v1 265 维无公共宝石；public-v2 已落地但主线接入仍在推进。  
3. **启发式 rush 克制**：训练分布对“即时买分”覆盖不足。  
4. **引擎热路径**：`getLegalActions` deepcopy 限制一切搜索/仿真类收益。  
5. **价值未标定**：F1 AUC 0.525 → 任何“先搜索后蒸馏”闭环都缺教师。

---

## 2. alpha-zero-general 做对了什么（与本仓库对照）

| 对方能力 | 对方做法 | 本仓库现状 | 迁移价值 |
|---|---|---|---|
| 搜索吞吐 | Numba board + Numba MCTS + ONNX；~3000 rollouts/s/core | deepcopy 规则；sims=100 时 2–5 moves/s | **基础设施级**，惠及一切仿真 |
| 动作空间 | 显式 81 动作（买/预留/拿还宝石组合 + pass） | 固定枚举 3510 | **高**：网络与搜索双重受益 |
| 价值 bootstrap | 课程：sims 100→800，lr 0.003→3e-4，epoch 递增 | Z2 仅 10 迭代 × 1000 局 × 100 sims | 重启 AZ 的必要条件；对 PPO 无直接对应 |
| 数据效率 | 卡槽/预留槽对称置换增广 | 无对称增广 | **高且便宜** |
| MCTS 质量 | FPU、Playout Cap Randomization、Forced Playout、自动 Dirichlet α、Q/Z 目标 | 基础 PUCT + Dirichlet ε=0.25/α=1.0 | 中：有合格 value 后再加 |
| 不完整信息 | PC-PIMC “universes” + 机会节点正确处理 | 4 树确定化（审计通过） | 中：形式已正确，吞吐优先 |
| 训练目标 | KL-div、invalid-action masking 的策略梯度修正、OneCycleLR | PPO：masked CE/GAE；DQN：TD + 掩码 | 中：可局部替换 loss / LR 调度 |
| 多人数 | 同一引擎 2/3/4p，预训练分模型 | E1 入口落地，E3 训练未启 | 中：验证“per-seat 分模型”路线 |
| 规则简化 | 禁止同时拿还宝石，换取逻辑与动作空间可控 | 引擎更严/网页更宽，方向安全 | 低：不改引擎语义 |

对方“试过但无效”的项目（高级 cpuct、surprise weight、Dropout/EfficientNet 等）
说明：**架构花活收益低于吞吐、课程与数据增广**。这与本仓库路线图“训练量 > 目标函数 > 对手分布 > 观测 > 算法本体”一致。

---

## 3. 迁移设计：按 PPO / DQN 瓶颈挂载

```text
alpha-zero-general 思路
        │
        ├─ 吞吐 ──────────► 规则热路径 / 动作空间（惠及 PPO rollout、DQN 仿真、search、league）
        ├─ 数据 ──────────► 对称增广（PPO/DQN 自博弈与回放通用）
        ├─ 目标 ──────────► 训练目标与调度（PPO loss/LR；DQN auxiliary）
        ├─ 覆盖 ──────────► 课程与“多宇宙”采样思想（对手池 / 确定化）
        └─ 搜索 ──────────► 仅在 F1 价值可用后：离线教师 / 分析工具
```

### 3.1 高优先级：直接服务 PPO/DQN 主线

#### T-1 规则热路径加速（不改语义）

- **现状**：`getLegalActions` 内 deepcopy；`generateSuccessor` 原地但需配对回滚。  
- **借鉴**：对方 `valid_moves` / `make_move` 全程 int8 原地。  
- **落点**：
  1. 去掉 `getLegalActions` 对动作对象的无谓 deepcopy（或提供 cached/table API）；
  2. 热路径一律 `create_legal_actions_mask` / `create_action_mapping`（AGENTS.md 已要求）；
  3. 可选：为仿真提供“轻量状态视图”（仅搜索/rollout 用，不改浏览器 parity 门）。
- **受益**：PPO 对局生成、DQN 仿真、league 评测、未来 search、play-web 回流回放。  
- **验收**：`make test` / `make parity` 全绿；league 同 seed 胜率矩阵可复现；微基准记录 Δms。

#### T-2 动作空间压缩 / 参数化（学习目标友好）

- **现状**：`Discrete(3510)`，绝大多数非法；PPO 每步全维 softmax，DQN gather 同维。  
- **借鉴**：对方 81 动作：位置买卡 + 预留槽 + 有限宝石组合 + pass。  
- **落点（分阶段，不破坏 v1 部署）**：
  1. **训练侧参数化头（实验变体）**：policy/value 输出分解为  
     `(action_type, card_slot/gem_combo)`，推理时再映射回 `ALL_ACTIONS` 索引；  
  2. 保持 `ALL_ACTIONS` 与浏览器/评测协议不变；  
  3. 对照实验：同预算 parameterized-head vs 3510 全维 head。
- **受益**：降低策略/价值学习难度；为未来 search 提供更稳的 prior。  
- **风险**：映射一对多时需训练目标定义清晰（合法集合上的聚合概率）。

#### T-3 对称数据增广（最便宜的“免费样本”）

- **现状**：自博弈/回放/BC 均无状态-动作对称。  
- **借鉴**：对方同层桌面卡置换 + 自己预留槽置换，并同步置换 policy。  
- **落点**：
  1. 在 `public-v2` / 特征空间定义可验证置换（同 tier 桌面 12 卡 → 4 卡槽；自己的 3 预留槽）；  
  2. PPO 轨迹缓冲与 DQN replay 采样后可选增广；  
  3. 单测：置换后合法掩码与特征切片一致，策略目标同步置换。
- **受益**：等价于样本 ×（卡槽置换数），对稀疏终局与 heuristic 覆盖都有帮助。  
- **优先级**：与 C 类训练可并行，成本约 2–4 天。

#### T-4 对手分布课程化 / “多宇宙”思想 → 对手池

- **现状**：PPO 池已有 ga/heuristic×3/minimax，但 heuristic 族仍是缺口。  
- **借鉴**：对方用 universes / 课程扩大信息与局面覆盖，而不是单一固定分布。  
- **落点（对齐路线图 C3/C4）**：
  1. 池权重向 heuristic 族倾斜（`heuristic:2,rush:2,ga:1,minimax:1` 等）；  
  2. 引入“风格宇宙”对照：rush / hoard / deny（预留拒止）/ tempo 混合；
  3. 评测矩阵保留 exploiter 式保持型测试（训练池内外对照）。
- **受益**：直接攻打 G1 唯一剩余项（heuristic 58.0% → 60%）。

### 3.2 中优先级：训练目标与采样质量

#### T-5 PPO 训练目标局部升级

| 对方项 | 可迁移到 PPO 的形态 | 注意 |
|---|---|---|
| KL-div 替代 CE（策略匹配） | 蒸馏/BC 段：`KL(π_teacher ‖ π_student)` 替代 CE 对照 | 主线 PPO 仍是 clipped PG，不替换 |
| OneCycleLR | stabilization 训练后期 LR 调度消融 | 与 critic lr×5 配置联合消融 |
| invalid-action masking 的 PG 修正 | 确认 masked softmax 后的梯度无非法泄漏（已有测试则补审计） | 不改变掩码语义 |
| Q/Z 分离学习目标 | PPO value_mode / rank utility 已部分覆盖；可加“短程 return + 终局 z”双头对照 | E2 排名效用优先 |

#### T-6 DQN 副线：修正管线 + 课程，而非 plain 长跑

- **证据**：D1 200k plain CLI M3 失败；corrected experiment.py 管线曾达 52.7%。  
- **借鉴**：对方课程化训练（低 sims 起步、逐级抬难度）。  
- **落点**：
  1. 用 `experiment.py` 的 search/guidance/EMA/冻结对手池 **修正课程** 重训 auxiliary heads；
  2. 对手课程：random → minimax 混池 → 风格化池，而非单一 50/50 长跑；
  3. 产出合格教师后，再做 F1 复标定 → D3 蒸馏重试。
- **借鉴边界**：不要把 AZ 的 MCTS sims 课程原样套到 DQN；DQN 的“课程”是**对手与引导课程**。

#### T-7 确定化搜索：保留审计成果，只在价值可用后启用

- **现状**：Z0 审计通过（隐藏信息零泄漏）；F1 否定当前价值头；Z3 证明搜索零增益。  
- **对方可借鉴**：FPU（parent value 作未访问子节点初值）、Playout Cap Randomization、树间预算分配。  
- **落点**：
  1. 先 T-6/D3 产出可用 value（或 PPO critic 在冻结对手批上 AUC≥0.75）；
  2. 再在 `alphazero/mcts.py` / `dqn/search.py` 增加 FPU 与 cap randomization 消融；
  3. 用途限定：**离线教师标签 / 分析**，不进浏览器部署路径（维持既有裁决）。

### 3.3 低优先级 / 明确不取

| 项目 | 原因 |
|---|---|
| 完整重写为 Numba 通用 AZ 引擎 | 与课程引擎 parity 门、浏览器伪状态、3510 动作协议冲突；成本数月 |
| 把 alpha-zero-general 的 81 动作硬编码进引擎 | 支付/宝石规则语义不同（对方禁止同时拿还）；方向安全但训练分布漂移 |
| 用对方预训练 .pt 直接打本网页版 | 规则、观测、动作协议三重不一致，不可迁移权重 |
| 在价值 AUC 0.525 时扩大 MCTS 部署 | Z3/mcts-gate 已三次否定；搜索只放大误差 |
| Dropout / 大 ResNet / 玄学 cpuct 公式 | 对方仓库自己标注“试过无效”；本仓库瓶颈在数据与覆盖 |

---

## 4. 与现有提升路径的挂载表

| 迁移项 | 对应路线图 | 服务对象 | 预期收益 | 粗成本 |
|---|---|---|---|---|
| T-1 热路径 | 跨阶段基建 | PPO/DQN/search/league | 吞吐、可扩预算 | 3–5 天 |
| T-2 参数化动作 | 新实验支线（可并 C2 后） | PPO/DQN 学习难度 | 中长期样本效率 | 1–2 周 |
| T-3 对称增广 | C2/D 训练可选组件 | PPO+DQN | 免费样本，抑过拟合 | 2–4 天 |
| T-4 池课程/风格宇宙 | **C3/C4 攻 G1** | PPO | heuristic 缺口专项 | 1–3 天配置+过夜训 |
| T-5 loss/LR 消融 | C1 延伸 | PPO | 边际 | 每分支数小时 |
| T-6 DQN 修正课程 | **D1/D3 纠偏** | DQN 教师 → PPO | 修复蒸馏闭环 | 数晚 |
| T-7 FPU/cap random | F2（后置） | 离线 search | 仅在教师可用后 | 按需 |

**建议执行顺序（对齐“下一轮优先级”）**：

1. **立刻做**：T-4（heuristic 池权重/风格）——唯一剩余 G1 项，几乎零新基建。  
2. **并行做**：T-3 对称增广进 PPO 训练；T-1 热路径摸底（先测 deepcopy 占比）。  
3. **副线**：T-6 用 corrected DQN 课程重训教师 → F1 复标定 → D3。  
4. **暂缓**：T-2 大改、T-7 搜索增强——等教师/价值过门后再动。

---

## 5. 可借鉴思路 → 具体实验清单（可复现口径）

> 沿用仓库纪律：manifest、`PYTHONHASHSEED=0`、独立测试种子段、league 150 局口径、失败局不静默剔除。

| ID | 实验 | 对照 | 成功判据 |
|---|---|---|---|
| X1 | PPO 池权重：`heuristic:2,rush:2,hoard:2,ga:1,minimax:1` × 500–1000 updates | C2-R2 同预算池 | vs heuristic ≥ 55%（中期）/ ≥60%（G1） |
| X2 | 自博弈轨迹对称增广（卡槽/预留槽） | 无增广同 seed | 验证胜局或 vs heuristic 提升，且非法动作=0 |
| X3 | BC/KL 目标替换 CE | 现有 CE BC | masked loss 不劣，DAgger 更新更稳 |
| X4 | corrected DQN（experiment.py）重训 auxiliary heads | plain D1 200k | M3 ≥55% 或明确记录仍失败 |
| X5 | F1 复标定（冻结对手校准批，非自博弈流） | AUC 0.525 | AUC ≥0.75 才允许进入搜索教师协议 |
| X6 | （门后）FPU + playout cap 的等墙钟 search 消融 | 现 PUCT | 仅当 X5 通过；报告等墙钟胜率差 |

---

## 6. 对 alpha-zero-general 的诚实评价

**值得学的**：

- 以“每秒可模拟多少”为第一性原理的产品/工程取舍；
- 状态与动作的紧凑表示；
- 课程训练与大量小幅有效技巧的叠加；
- 文档化“什么有效/什么无效”的实验态度。

**不能照搬的**：

- 对方已在**自己的**规则/状态/动作空间上完成 AZ 闭环；本仓库信息协议、浏览器 parity、课程对手与部署路径都不同；
- 对方的预训练权重在本仓库**不可加载、不可评测**；
- Splendor 在本仓库是**课程项目 + 网页 sim-to-real**，不是“CPU AlphaZero 基准套件”。

**正确关系**：

> 把 alpha-zero-general 当作 **“搜索友好的板游 RL 工程参照系”**，  
> 而不是 **“把本仓库改造成第二个 alpha-zero-general”**。

---

## 7. 参考链接与本仓库文档

### 外部

- [alpha-zero-general（GitHub）](https://github.com/cestpasphoto/alpha-zero-general)
- [alpha-zero-general README_features（工程改进清单）](https://github.com/cestpasphoto/alpha-zero-general/blob/master/README_features.md)
- 基线上游：[suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
- 搜索技巧参考：KataGo / Accelerating Self-Play Learning in Go（对方 README 已引）

### 内部

| 文档 | 用途 |
|---|---|
| [Z_PHASE_IMPLEMENTATION_20260913.md](./Z_PHASE_IMPLEMENTATION_20260913.md) | 本仓库 AZ Z0–Z3 失败与 gate 数字 |
| [ALGORITHM_SURVEY_20260912.md](./ALGORITHM_SURVEY_20260912.md) | PPO/DQN/GA/minimax 数学归因 |
| [IMPROVEMENT_ROADMAP_20260912.md](./IMPROVEMENT_ROADMAP_20260912.md) | 主计划与进度审计 |
| [C2R2_2000_REPORT_20260914.md](./C2R2_2000_REPORT_20260914.md) | PPO 主线最新 league 结果 |
| [D1_200K_COURSE_REPORT_20260913.md](./D1_200K_COURSE_REPORT_20260913.md) | DQN 长跑 M3 否定结论 |
| [SPLENDOR_LITERATURE_SURVEY_20260912.md](./SPLENDOR_LITERATURE_SURVEY_20260912.md) | 文献侧“搜索最后”裁决 |
| [POLICY_IMITATION_IMPLEMENTATION.md](./POLICY_IMITATION_IMPLEMENTATION.md) | MCTS gate blocked 记录 |

---

## 8. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-09-14 | 初版：对照 alpha-zero-general，梳理 PPO/DQN 优劣势与迁移优先级；不改变“搜索后置”的主线裁决 |
