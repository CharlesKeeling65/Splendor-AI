# Splendor-AI 提升路径与分阶段实施计划

> 日期：2026-09-12。本计划是**新立项文档**，不替代 `plan/phase-0..5`（它们记录已落地的 P0–P5）；
> 验收沿用仓库双轨验收体系与 M1–M3 课程门槛。依据文档：
> [ALGORITHM_SURVEY_20260912.md](./ALGORITHM_SURVEY_20260912.md)（仓库四策略数学归因）、
> [SPLENDOR_LITERATURE_SURVEY_20260912.md](./SPLENDOR_LITERATURE_SURVEY_20260912.md)（arXiv 文献取长补短）。
> 如需按仓库惯例迁入 `plan/`，可整体移动，任务编号保持不变。

---

## 0. 总原则与角色定位

**主线判断**（依据两份调研文档的归因）：当前差距主要来自**训练量、目标函数、对手分布**，而非算法
本体。因此路径设计为"先修协议/特征/奖励，再扩算法规模，搜索最后"：

| 角色 | 定位 |
|---|---|
| **PPO（稳定化训练器）** | **主线**：自博弈无过期回放、熵可控、输出形态与网页部署单次前向同构 |
| **DQN** | **副线两角色**：① 200k 完整课程跑满；② off-policy 网页数据回流 + 价值先验蒸馏给 PPO |
| **GA** | 4p 基线保留（fitness 天生 4 人局，是唯一"训练-评测同源"合法的多席选手） |
| **minimax / 启发式** | 课程组件与压力测试对手；启发式另升格为"rush 风格样本"；不再加深 |
| **搜索（确定性化 MCTS）** | 最后：仅在价值函数标定后做离线教师/分析工具，不进部署 |

**硬纪律**（沿用 AGENTS.md）：不破坏存量路径（v1 观测、PPO/GA/minimax/引擎零行为变化）；
新代码全量类型注解过 mypy；随机性入口 seed 三件套；改动特征/浏览器抽取必须 `make parity`；
lint 只从 `.[dev]`；所有 runner 固定 `PYTHONHASHSEED=0`（2026-09-08 中止批次的教训）；
对真实服务器只操作自建房间、人类节奏。

**算力假设**：Quadro P5000 单卡可用（2026-09-08 稳定化运行 9 训练 + 5700 评测局 = 27.5 分钟；
DQN 20k 步 ≈ 21 分钟/任务）。按此估算各阶段成本；若算力不可用，**砍 seed 数不减步数**（配对
复现优先于独立重复）。

---

## 1. 现状基线与本计划总目标

### 1.1 基线（2 人局，详见 ALGORITHM_SURVEY §2）

- corrected DQN：vs random 99.3% / heuristic 39.3% / minimax 52.7%
- 稳定化 PPO（fixed）：vs random 100% / heuristic 32% / minimax 48.7% / GA 54%
- GA：vs heuristic 34% / minimax 58%
- PPO critic 价值解释方差 0.03–0.21；scratch 冷启动 0%；
- 3/4 人局：训练量为零，仅结构兼容 + OOD 代理估计。

### 1.2 总目标（DoD）

| 编号 | 目标 | 量化验收 |
|---|---|---|
| G1 | 2p 对固定基线全面超过 GA | vs minimax ≥ **60%**、vs heuristic ≥ **60%**、vs GA ≥ **55%**（≥150 局，独立测试种子段） |
| G2 | 学习信号质量 | PPO critic 解释方差 ≥ **0.5**；DQN TD 损失稳定（progress.csv 无发散段） |
| G3 | 训练量 | DQN 200k 步完整课程 ×3 seed；PPO ≥ **8,000 训练局**/seed，配对消融可复现 |
| G4 | 3/4 人局 | 3p、4p 各建立 per-seat 模型与 league 基线表；移除"分布外"标注的前提条件达成 |
| G5 | 协议 | league 评测器 + 行为度量入库；测试种子段封存规则生效 |

---

## 2. 阶段 A：评测与协议地基（无训练，先行）

> 依据：SFP 文献的"超参敏感性"纪律 + MAP-Elites 文献的"行为覆盖"度量（文献调研 §1.3、§3.3）。
> 现状缺口：**仓库没有 round-robin/league 评测器**（所有跨对手数字来自专用实验运行器）。

### A1 league 评测器
- **改动**：新增 `src/splendor/league.py` + console script `splendor-league`（pyproject 注册）。
  功能：给定 agent 模块列表与席位数 `-n`、局数 `-m`、种子段，跑全配对 round-robin（双座次平衡），
  输出 JSON + Markdown 胜率矩阵、Wilson 区间、每局行为度量。复用 `general_game_runner` 的加载
  逻辑（`myAgent` 约定）与 `LimitRoundsGameRule`。
- **行为度量**（每局每 agent）：买卡数、预留数、拿宝石次数、达 15 分所用轮数、终局分、卡数
  tie-break 值；从 `agent_trace` 统计（trace 已存在，见 `dqn/population.py:101` 引用）。
- **测试**：`tests/test_league_runner.py`（random × random 小局数冒烟 + 配对种子可复现断言）。
- **验收**：`splendor-league -a random,random -n 2 -m 10` 出矩阵；CI 离线通过。
- **成本**：1–2 天。

### A2 种子段管理与测试集封存
- **改动**：沿用 stabilization 协议的分段习惯（训练 820000–821999 / 验证 822xxx / 测试 823xxx，
  历史禁区 800–820k、900–940k 已声明）。新分配：**训练 826000–827999、验证 828000–828999、
  独立测试 829000–829049**；写入 `docs/seed_registry.md` 并在 A1 的 manifest 中强制校验。
- **验收**：任何新实验的种子段在 manifest 中可审计，不复用 823000 封存测试集调参。
- **成本**：0.5 天。

### A3 算力与预算基线确认
- 一次冒烟：PPO 8 更新 × 16 局 + DQN 5k 步，确认 P5000 环境与墙钟估算（产出 `runs/budget-smoke/`）。
- **验收**：两份 budget 数字入本文档 §2 的估算校准。
- **成本**：0.5 天。
- **实测（2026-09-13，Quadro P5000 CUDA，`.venv-p5000`，PYTHONHASHSEED=0）**：
  - PPO 稳定化冒烟（8 updates × 16 局，fixed 变体，seed42）：训练 273.5 s（≈ **2.1 s/训练局**），
    含验证评测的总墙钟 404 s。外推 C2（500 updates ≈ 8,000 训练局/seed）：**≈ 4.7 h/seed**
    （+验证评测开销），3 seed 串行 ≈ 15 h，可一夜完成。
  - DQN 冒烟（5,000 步 vs random）：172.4 s（**29 步/s**）。外推 D1（200k 步）：**≈ 1.9 h/seed**，
    3 seed 串行 ≈ 6 h。
  - 佐证文件：`runs/budget-smoke/ppo-stab-smoke/training/fixed-seed42/runresult.json`、
    `runs/budget-smoke/dqn-5k/26-09-13_16-13-44__dqn/progress.csv`（runs/ 不入库）。

---

## 3. 阶段 B：观测与奖励修正（特征 v2 化 + 塑形）

> 依据：ALGORITHM_SURVEY §5.3.3/§10.4（观测缺公共宝石是学习型共享瓶颈）与文献调研 §2
> （event-value functions）。
> **纪律红线**：`test_feature_parity` 锁定 v1（265 维）逐位奇偶——**v1 一字不动**；所有新特征走
> 版本化提取器。此项实质上部分重开 `docs/p4_decision.md` 暂缓的特征升级项，重开理由
> （训练条件已解除 + 文献证据）记录进该文档的复审附录。

### B1 统一特征提取器与 public-v2 多席化
- **改动**：将 `dqn/features.py` 的 `extract_observation`（312 维 public-v2：265 + 供给 6 + 对手
  面板 24 + 贵族费用 15 + 席位/阈值 2，见 `docs/DQN_SEARCH_EXPERIMENTS.md:11-15`）提升为共享模块
  （如 `splendor/splendor/features_v2.py`，DQN 改为 re-export，行为零变化）；解除 312 维对 2 席的
  硬编码：对手面板按 `MAX_RIVALS=3` 槽位通用化，`seat` 特征显式化；checkpoint 元数据携带
  `feature_version`（play-web 已支持从 checkpoint 推导）。
- **PPO 接入**：`PPO` 网络 `input_dim` 已参数化（`ppo/network.py:23-29`），稳定化训练器改从
  feature_version 读取维度；**旧 265 维 checkpoint 与 DAgger-2 初始化路径保持可用**（初始化时若
  版本不符则截断/扩展 trunk 第一层的显式迁移脚本，或直接以 v2 重新做一轮 BC——BC 管道现成）。
- **测试**：v1 parity 不动仍绿；新增 `tests/test_features_v2.py`（3/4 席维度断言、掩码无关性、
  已知状态手算样例）；浏览器侧仅当 v2 模型上网页时才需伪状态回填，`make parity` 扩展 v2 用例。
- **验收**：`make parity`、`make test` 全绿；v2 提取器对同一引擎状态在 2/3/4 席下维度一致且
  2 席结果与 DQN public-v2 逐位相等。
- **成本**：3–5 天（含测试）。

### B2 奖励塑形：potential-based 先行，事件项对照
- **改动**：新增 `policy_imitation/shaping.py` 训练侧包装（镜像 `dqn/reward_wrapper.py` 的口径，
  不改基础 env）：
  - 主变体（策略不变性安全）：$r' = r + \gamma\phi(s') - \phi(s)$，势能
    $\phi(s) = \mathrm{calScore}(s,i) - \max_{j\ne i}\mathrm{calScore}(s,j) + \kappa\cdot\text{noble\_progress}(s,i)$，
    $\kappa$ 默认 0.05；
  - 对照变体（文献 EF 的轻量版）：少量事件项（买卡/达贵族/拒止预留），权重手调，标注为
    非势能实验组。
- **数学约束**：主变体依 Ng et al. 1999 保证最优策略集不变——这是把它设为默认、事件项设为实验组
  的理由。
- **测试**：`tests/test_reward_shaping.py`（势能 telescoping 性质：完整轨迹 $\sum(\gamma\phi_{t+1}-\phi_t)=\gamma^T\phi(s_T)-\phi(s_0)$ 断言；非法动作无关性）。
- **验收**：单元测试绿；在 5k 步冒烟中 vs heuristic 胜率不劣于无塑形（快速健康检查，不作结论）。
- **成本**：2–3 天。

---

## 4. 阶段 C：PPO 自博弈主线（扩规模 + 修 critic + 风格化池）

> 依据：ALGORITHM_SURVEY §6.3（EV 0.03–0.21 是当前最大 handicap）、结果文档 §4.2"先提高 critic
> 与采样轨迹质量，再扩大规模"、文献调研 §3（风格覆盖）。

### C1 critic 修复（先于扩规模，单因素消融）
- **改动**（全部在 `ppo_selfplay.py` 配置层，不改训练器骨架）：actor/critic 分离学习率（critic
  lr ×5 起试）；`critic_warmup_epochs` 2→5；value loss 系数 0.5→1.0 对照；可选 critic 独立 trunk
  顶层（共享底层）对照。每项**单独**开分支做配对消融（同 seed、同对手抽样计划）。
- **验收**：最优配置在验证批上 EV ≥ **0.5**（G2）；结果文档化（新
  `docs/PPO_CRITIC_ABLATION_<date>.md`，沿 stabilization 报告格式）。
- **成本**：每分支 ≈ 3 分钟训练 + 40 分钟验证评测 × 3 seed；总计 1–2 天。

### C2 规模化自博弈训练（执行训练课程）
- **改动**：扩 `stabilization.py` 默认预算：updates 8 → **≥500**（每 seed ≥8,000 训练局，G3），
  验证评测频次与规模同步放大；对手池构成下 C3。训练对手为池采样 + 学习方采样 / 快照贪心的
  不对称保留（协议已记录），但新增**对称采样对照分支**（结果文档 §4.2 的待验证项）。
- **成本估算**：每 seed ≈ 3.5 h（按 27.5 min/6,852 局线性外推），3 seed ≈ 10 h，可过夜串行。
- **验收**：训练曲线（EV、clip fraction、熵）无发散；选模按验证整数胜局（协议已有）。
- **验收门槛**：独立测试 vs random ≥99%、vs minimax ≥ **55%**（达成 M3，超过 DQN 52.7%）。

### C3 对手池风格化（治"打不过 heuristic"）
- **改动**：`policy_imitation/policies.py` 新增启发式变体工厂（`heuristic-rush`：贵族/买卡权重
  上调、收藏宝石权重衰减；`heuristic-hoard`：囤宝石路径），静态 candidate 机制现成
  （`build_builtin_candidate`）；训练池 = ga + heuristic ×2 变体 + minimax + current/history，
  权重显式记录（沿用"记录实际对局数而非配置权重"的协议纪律）。
- **验收**：稳定化训练器对池的采样统计入 manifest；独立测试 vs heuristic ≥ **50%**（中期检查，
  最终 60% 在 C4）。
- **成本**：1 天实现 + 并入 C2 训练。

### C4 独立测试与"exploiter"式体检
- **改动**：用 A1 league 评测器对最优 PPO checkpoint 跑固定矩阵（random/heuristic/heuristic-rush/
  minimax/GA/DQN-best，各 ≥150 局，A2 测试种子段）；额外做一组**保持型测试**（fix 一半席位给
  训练池内对手，检验是否只对训练分布过拟合）。
- **验收**：G1 达成（vs minimax ≥60%、vs heuristic ≥60%、vs GA ≥55%）；league 矩阵 + 行为度量
  入 `docs/`。
- **成本**：评测 2–3 h；人工分析 1 天。

---

## 5. 阶段 D：DQN 副线（可与 B/C 并行）

### D1 200k 步完整课程
- **改动**：无代码（入口已支持）；按 `docs/TRAINING_GUIDE.md` M1→M2→M3 手动课程，对手
  random → random → minimax，3 seed，`--opponent-pool` 与 population 机制视 M3 结果启用。
- **成本**：≈ 3.5–4 h/seed（20k 步 ≈ 21 分钟线性外推），3 seed 一晚。
- **验收**：M2（vs random ≥90%）与 M3（vs minimax 100 局 ≥55%）逐 seed 报告；与 20k 短程结果
  对比，验证"训练量假设"。

### D2 网页真实数据回流
- **改动**：`collect_from_browser` 数据（特征版本校验 + 支付语义差异白名单过滤）混入 replay 的
  采样比例配置（默认 ≤20%）；离线测试用已存夹具，CI 不碰网络。
- **验收**：单测绿（夹具回流 → buffer 内容断言）；真实回流留到 G1 后的部署阶段。
- **成本**：2 天。

### D3 价值先验蒸馏给 PPO
- **改动**：用 `QNetwork.policy_value` 双头（`dqn/network.py:232-245`，已实现）在自博弈状态上生成
  掩码策略软目标，替代/混合 DAgger-2 初始化（教师信息审计按结果文档 §4.4 要求先做
  public-info 一致性检查）。
- **验收**：蒸馏初始化的 update-0 验证胜局 ≥ DAgger-2 基线（22/60）。
- **成本**：2–3 天。

---

## 6. 阶段 E：3/4 人局（依赖 B1）

### E1 per-seat 训练入口
- **改动**：`stabilization.py`/`ppo_selfplay.py` 的对局构造按 `n` 参数化（agents 列表长度即席位数，
  env 已支持）；3p/4p 各自独立的对手池（自博弈 + GA + heuristic，均 n 席）；**每 seat count 一个
  模型**（2p/3p/4p 分开），不追求单模型通吃。
- **验收**：3p、4p 训练冒烟各 16 局零非法动作；观测维度与掩码在 3/4 席下一致。

### E2 排名效用与折扣
- **改动**：终局奖励从胜负 ±10 改为排名效用 $u_{\text{rank}} \in \{+1,\,0,\,-0.5,\,-1\}$
  （rank 1→4，calScore 名次），γ 0.99 → **0.997**（4p 折叠窗口 ×3 的等效时间尺度补偿）；
  可选 Suphx 式排名预测辅助头（`policy_value` 模式扩展，依赖 D3 管道）。
- **数学要点**：4p 中"第三 vs 第四"与"第一 vs 第二"同量级——胜负二元效用无法表达；排名效用是
  Suphx 在 4 人麻将上验证过的方向。
- **验收**：reward wrapper 单测（各名次→效用映射 + telescoping）。

### E3 GA 4p 基线与 league 建档
- **改动**：A1 league 评测器跑 4p：GA / heuristic / minimax(不可用则 random 顶替) / PPO-4p /
  DQN-4p；GA 无需改动（fitness 天生 4 人局）。
- **验收**：`docs/` 出 3p/4p 首张 league 矩阵表；OOD 标注的移除条件明确（E2 训练完成后重新校准）。

### E4 胜率估计器多席化
- **改动**：`remote/rollout.py:133-137` 解除 public-v2 的 2 席硬编码（依赖 B1 的多席 v2）；
  保留 OOD 警告直至 E3 完成。
- **验收**：单测覆盖 2..4 席构造；CI 离线。

---

## 7. 阶段 F：搜索（最后，前置：C1 + D1 + D3）

### F1 价值标定
- **改动**：在独立测试局上标定 critic/outcome head 的胜率预测（可靠性曲线/ECE、AUC）。
- **验收**：胜率预测 AUC ≥ 0.75 或明确记录"不可用作搜索价值"（结果文档 §4.5 的前提检查）。

### F2 确定性化系 MCTS 对照实验（仅离线）
- **改动**：扩展 `dqn/search.py`：**多确定化树**（每树对"未见牌库∪对手预留"采样一次后树内复用）
  + 树间预算分配（文献 2607.13007 的两个动态分配轴做简化版：固定 m ∈ {4,8,16} × 均匀/优先分配），
  与现有单样本 PUCT 同墙钟对照；不用于浏览器伪状态（维持 `search.py:5-6` 约束）。
- **验收**：等墙钟配对报告（胜率差 + 延迟分布）；若显著为正 → 仅作为 DAgger 教师标签源立项，
  不进部署路径。

---

## 8. 依赖图与排期

```
A1 league ──┬──────────────► C4 体检 / E3 建档
A2 种子段 ──┤
A3 冒烟 ────┤
            │
B1 特征v2 ──┬─► B2 塑形 ─► C1 critic ─► C2 规模化(含C3池) ─► C4 体检 ──► (G1)
            └─► E1/E4 多席化 ─► E2 排名奖励 ─► E3 3/4p 训练与建档
D1 200k课程（并行） ─► D3 蒸馏 ─┐
D2 回流（并行，G1后上线）        ├─► F1 标定 ─► F2 搜索对照
C1 + D1 + D3 ──────────────────┘
```

| 周 | 内容 | 里程碑 |
|---|---|---|
| W1 | A1–A3、B1 | league 评测器上线、v2 特征多席化 |
| W2 | B2、C1、D1 启动 | critic EV ≥0.5 的配置确定 |
| W3 | C2（+C3）过夜训练、D1 继续 | vs minimax ≥55%（M3） |
| W4 | C4 体检、D3、E1/E4 | G1 初判；蒸馏基线 |
| W5 | E2/E3 3/4p 训练 | 3p/4p league 首表 |
| W6 | F1/F2、G1 终审、文档回填 | 总 DoD 审计 |

（按"P5000 单卡 + 串行过夜"保守估计；W2 起多数夜晚为机器时间。）

---

## 9. 风险登记册

| 风险 | 缓解 |
|---|---|
| Python 哈希影响动作枚举（2026-09-08 中止批次教训） | 所有 runner 固定 `PYTHONHASHSEED=0`；复现性诊断保留 |
| 训练胜率 ≠ 实力（强弱混杂对手池 + 采样/贪心不对称） | 只用独立测试种子选模/宣称结果；协议已在 stabilization 落地 |
| 测试集反复选参过拟合 | A2 种子段封存；新调参必须新测试段 |
| 特征版本漂移破坏部署 | checkpoint 携带 `feature_version`；play-web 已从 checkpoint 推导；v1 零改动 |
| P4 复审边界 | B 阶段重开项在 `docs/p4_decision.md` 附录补记理由与范围 |
| 3/4p 观测不足导致训练无信号 | B1 的对手面板 24 维先落地再开 E；冒烟先验证可学习性 |
| 算力不可用 | 砍 seed 不减步数；配对消融优先 |
| 服务器礼仪 | 回流/部署沿用人类节奏与局间休息硬约束；CI 永不联网 |

---

## 10. 与既有文档/验收体系的关系

- **课程门槛**：C2/E2 的验收直接引用 M1–M3（`docs/TRAINING_GUIDE.md:102-116`）；G1 是 M3 之上的
  扩展目标。
- **双轨验收**：每阶段的代码验收走 `make lint / make test / make parity` + CI（3.12/3.13 矩阵）；
  实验验收走 stabilization 式 manifest + 独立复算审计。
- **文档回填**：每个阶段完成后在 `ALGORITHM_COMPARISON.md` 补录同口径数字，并在
  `CODEBASE_PANORAMA.md` §7 增量记账。
- **本计划完成判据**：§1.2 的 G1–G5 全部达成或明确记录未达成原因与下一步。
