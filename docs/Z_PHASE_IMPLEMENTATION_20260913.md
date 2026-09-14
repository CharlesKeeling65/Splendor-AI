# Z 阶段实施记录（Z0/Z1/Z2，2026-09-13）

> 依据提升路径评审结论（2026-09-13 会话）：F1 否定 TD critic 可标定性（AUC 0.525）后，
> "先标定价值再上搜索"的前置条件无法满足；AlphaZero 机制把前提倒置——**搜索 + 终局 z
> 自产训练信号**。Z 阶段按 Z0（地基）→ Z1（无网基线）→ Z2（AZ-lite 训练）→ Z3（league
> 体检，视 Z2 gate 排期）推进。产物：`runs/z0-benchmark/`、`runs/z0-audit/`、
> `runs/z1-uniform-search/`、`runs/z2-az-lite/`（runs 不入库，数字回填本档）。

## Z0 地基 — 达成

### 微基准（`runs/z0-benchmark/z0_report.json`，P5000 机，PYTHONHASHSEED=0）

| 项 | move_0 | move_12 | move_30 | 说明 |
|---|---|---|---|---|
| deepcopy(rule) | 788 µs | 907 µs | 1031 µs | F2 harness 每模拟的成本 |
| Transactor apply+undo | 4.4 µs | 4.2 µs | 4.0 µs | **原地回滚 ~250×于 deepcopy** |
| state_fingerprint | ~22 µs | ~22 µs | ~22 µs | undo 精确性 oracle |
| az_search sims=100/trees=4 | 0.039 s | 0.187 s | 0.441 s | 每步搜索墙钟 |

- apply+undo 精确性：全部 cycle 指纹零失配；贵族分支（引擎唯一不恢复的顺序细节，
  successor 中间删除/predecessor 末尾追加）由确定性构造测试覆盖
  （`tests/test_az_state_utils.py::test_raw_engine_undo_permutes_nobles_without_snapshot`
  证明无快照时指纹必然漂移，Transactor 快照恢复后逐位相等）。
- `last_action` 无任何引擎决策路径读取（getLegalActions/gameEnds/calScore/双特征提取器
  均验证），换位键排除它以提升命中率；undo oracle 含它。
- 搜索成本被 `getLegalActions` 内部 deepcopy 支配（每次展开 ~0.3–4 ms，随局面丰富度变化），
  已是每新节点的硬下限；Z1 实测 ~2–5 moves/s @sims=100。
- move_12 仅 4 个合法动作是引擎非标准规则（手宝石 ≤7 时整拿 min(3, 可用色数)）的合法结果。

### 引擎热路径优化（超出 Z0 预期的产出）

`extract_observation` 实测 5.6 ms/次，剖析定位到 `vectorize_card` 每卡调用 sklearn
OneHotEncoder.transform（含 check_array 全校验，0.76 ms/卡 × 15 卡）。按唯一 card code
加等值缓存（`splendor/features.py::_CARD_VECTOR_CACHE`，返回 copy 防突变）后：
**5.6 ms → 0.43 ms（13×）**。`make parity` 门通过（1000 状态 obs+掩码逐位相等），
全量离线测试绿。此优化惠及全部训练/评测/浏览器路径。

### 信息审计 — pass（`runs/z0-audit/audit_report.json`）

- 干净检查：UniformEvaluator 在 8 个随机中局位置上，原局 vs 重发隐藏序（公共信息不变）
  的搜索输出**逐位相等**（确定性 rng + 排序后洗牌的确定化契约）。
- 阳性对照：DeckPeekingEvaluator（按真实牌库顶与桌面卡 code 比较偏置 reserve 先验）
  在 ≥1 位置正确打破相等——审计本身不是空转。
- 结论：搜索路径（sample_hidden 确定化 + 原地回滚 + 换位键）对隐藏信息零泄漏。

### 测试与门禁

`tests/test_az_*` 共 19 项（状态工具 6 / 搜索 7 / 审计 3 / 基准冒烟 1 / 自博弈与训练 6 中的
Z0 部分），ruff + mypy 全绿；全量 331 passed。

## Z1 纯搜索基线 — 完成（否定性锚点，如实记录）

配置：uniform 先验 + 零叶价值、sims=100、trees=4、root_noise 关闭；
`league_entries/az_search.py` 入口，independent_test 段 75 deals × 双座次，每对阵 150 局。

| agent | 局数 | 总胜率 | vs az_search |
|---|---|---|---|
| heuristic(population) | 450 | 86.7% | 100% |
| minimax | 450 | 79.1% | 98.3% |
| random | 450 | 33.3% | **100%** |
| **az_search（纯搜索）** | 450 | **0.2%** | — |

**解读**：深度 24 内 playouts 几乎到不了终局（Z0 实测 terminal_hits=0），Q 信号全零，
PUCT 退化为对先验/访问探索分数的 argmax → 访问集中到低索引动作的退化确定性策略，
连 random 都全输。**这从定量上确证 AZ 前提：价值网络不是增强而是必需**——与文献 #4
"价值/先验质量足够时确定性化 MCTS 才显著变强"一致。注意本基线是"AZ 式深度截断 + 零价值"，
不是 Rinascimento SFP 的"到终局 rollout/快速状态评估器"家族——后者的每步成本在本引擎
getLegalActions 开销下高 10–30×，未纳入 Z1（记录为已知偏差）。

## Z2 AZ-lite — 机制落地，训练运行中

机制（全部入库，单测绿）：

- **网络**：复用 `QNetwork(auxiliary_heads=True, dueling=False, public-v2 312 维)`，
  masked policy 头 + tanh outcome 头（[-1,1]，与 z=±1/0 对齐）。
- **warm start**：DAgger-2 BC（`bc-dagger-2/best.pth`）trunk→net、normalizer→input_norm
  （冻结）、policy_head→policy_head；outcome 头随机初始化（z 信号自对局产生）。
- **自博弈**：`selfplay.py::play_game`——每步 az_search（root Dirichlet ε=0.25/α=1.0），
  根访问分布为策略目标，终局 calScore 符号为 z（平局 0）；τ=1 前 12 ply、其后贪心。
  发牌种子走 `z_training` 段（注册表已扩段 1040000–1139999，docs/seed_registry.md 同步修订），
  搜索流为独立 default_rng。
- **训练**：masked CE(π_MCTS, p) + MSE(z, v)，Adam 1e-3/wd 1e-4，grad clip 1，
  batch 512 × 2 epochs/迭代，回放窗口 300k 局面。
- **迭代内评测**：greedy 先验（无搜索）vs random/minimax 各 10 deals 双座次——
  这是训练曲线信号；**Z2 gate（vs minimax ~57%）必须由 league CLI 带搜索测**
  （`AZ_SEARCH_CHECKPOINT`），二者语义分开记录。
- 进程池 spawn + 每任务载入当前权重 + `torch.set_num_threads(1)`。

运行-1 配置：`runs/z2-az-lite`，10 迭代 × 1000 局，sims=100/trees=4，24 workers，
CUDA 训练。结果表随迭代回填：

| 迭代 | moves | loss(policy/value) | greedy vs random | greedy vs minimax | 用时 |
|---|---|---|---|---|---|
| 0 | 64,362 | 2.533 / 0.546 | 100%（20 局） | 65%（20 局） | ~37 min |
| 1 | 63,128 | 2.378 / 0.358 | 100% | 50% | ~34 min |
| 2 | 61,786 | 2.270 / 0.290 | 100% | **75%** | ~34 min |
| 3 | 60,820 | 2.190 / 0.259 | 100% | 55% | ~35 min |
| 4 | 60,744 | 2.112 / 0.237 | 100% | 65% | ~41 min |
| 5 | 60,586 | 1.946 / 0.223 | 100% | 70% | ~41 min |
| 6 | 60,406 | 1.856 / 0.212 | 100% | 70% | ~40 min |
| 7 | 60,134 | 1.798 / 0.209 | 100% | 50% | ~40 min |
| 8 | 60,360 | 1.751 / 0.194 | 100% | 55% | ~39 min |
| 9 | 60,500 | 1.729 / 0.192 | 100% | 50% | ~40 min |

run-1 读数：value MSE **单调下降 0.546→0.192**（z 信号被持续吸收——F1 中 TD critic
做不到的事）；policy CE 单调下降 2.53→1.73；贪心 vs minimax 在 50–75% 震荡
（20 局样本 ±20pt 噪声），均值 ~60%，全程 ≥50%，与 loss 下降不同步——策略向自博弈
分布特化，改进更多体现在带搜索模式（价值头×搜索）。

**Z3 gate 协议（预声明）**：两个候选 checkpoint——`iter_002`（验证贪心最优）与
`iter_009`（最终迭代、价值头最优）——各测一次 league（vs minimax，150 局，
sims=100/trees=4，independent_test 段 wrap），**两结果都报告**；取较优者判定
gate（≥ 57.3% = C2-PPO 纪录）。gate 通过后以该 checkpoint 跑 G1 全矩阵体检。

### Z3 gate 结果（2026-09-14）— 未达成，如实记录

| 测定 | vs minimax（150 局，independent_test wrap） | Wilson 95% |
|---|---|---|
| iter_002，sims=100/trees=4 | 40.7% | [33.1, 48.7] |
| iter_009，sims=100/trees=4 | **42.7%** | [35.3, 51.0] |
| iter_009，**纯贪心**（sims=1，鉴别实验） | **42.7%** | [35.0, 50.7] |
| iter_009，sims=400/trees=8（预算×4 判别实验） | 40.0% | [32.8, 48.3] |

**判读**：

1. gate 未达（42.7% < 57.3%）。迭代内 20 局贪心读数（50–75%）确认为小样本噪声，
   真实水平 ~43%。
2. **贪心 == sims=100 == sims=400（40–43%，噪声内重合）**：不是"价值头生、搜索放大
   误差"，而是当前价值头对胜负**零净贡献**——加搜索预算买不来强度。先验停在 BC
   水平（DAgger-2 当年 vs minimax ≈ 50%，不同测试段量级一致）。
3. 归因：10 迭代 × 1000 局 × 100 sims ≈ 1M 次模拟，在 AZ 标准下极小；价值头仍在
   单调改善（MSE 0.546→0.192 未平台）。策略 CE 下降主要是拟合"以先验为主导的
   访问分布"——V 不够好时搜索无法改进 π 的经典循环瓶颈。
4. ExIt 预案**不成立**（已检验）：MCTS 教师 ≈ 先验（贪心==搜索实锤），蒸馏目标
   不含增量信号——它生效的前提同样是搜索显著强于策略。
5. **决定（一次一个因子）**：run-2 从 iter_009 续训 30 迭代（`--init-from`，
   种子续用 z_training 段 1050000 起，其余配置不变），~20 h。判停条件：价值 MSE
   平台且贪心 150 局读数仍 ~43% → 该配置饱和，如实记录并重新评估（自博弈 sims
   上调 / 更大 buffer / 更长课程）。中途在 iter_019 / iter_029 做一次 150 局
   贪心读数（sims=1）监控，不作为 gate。

### run-2 预声明成败判据（2026-09-14，写于开跑后、看数前）

- **有效进展**：中途或终局 150 局贪心（sims=1）读数 ≥ **55%**（明确越过 BC 水平
  ~50% 与 run-1 的 42.7%，且在 gate 射程内）→ 继续扩规模并重闯 gate。
- **完全成功**：150 局带搜索 league ≥ **57.3%**（gate 达成）→ G1 全矩阵。
- **饱和/失败**：价值 MSE 相邻 10 迭代降幅 < 0.01 **且** 150 局贪心 < **50%**
  → 该配置饱和：如实记录，重估方向（自博弈 sims 上调、buffer/epoch 加大、
  更长 BC 课程初始化）。
- 已知瞬态：run-2 开局 optimizer/buffer 重置，value MSE 回弹（0.192→0.289），
  预期随 buffer 回填恢复——此瞬态不计入判据。

## Z3 — 待 Z2 gate 排期

判据：Z2 最优迭代 checkpoint 经 `splendor-league -a ...az_search,...`（AZ_SEARCH_CHECKPOINT
注入，同 C4 协议 150 局/对阵）对 minimax ≥ 57.3%（C2-PPO 纪录）→ 再投入 G1 全矩阵体检。
若 10 迭代追不平 C2-PPO，按预案转 ExIt 混合（MCTS 教师 + 现有 PPO 稳定化循环消费）。

## 附带修复（本阶段提交一并入库）

1. `splendor/league.py` matchups 聚合的**双解析崩溃**（`record.names` 本为按席位名，
   却用 seat_assignment 花名册索引再索引 → 花名册 > 席位数即 IndexError；Z1 首跑即触发）。
   C4 当年靠事后重聚合绕过（`names_bug_fixed`），league.py 本体未修。回归测试：
   3-agent 花名册端到端不崩且席位名解析正确。
2. `splendor/features.py` vectorize_card 按 code 缓存（见上）。

## 复现

```bash
# Z0
PYTHONHASHSEED=0 python -m splendor.agents.our_agents.alphazero.benchmark --output runs/z0-benchmark
PYTHONHASHSEED=0 python -m splendor.agents.our_agents.alphazero.audit --output runs/z0-audit
# Z1
PYTHONHASHSEED=0 AZ_SEARCH_SIMS=100 AZ_SEARCH_TREES=4 splendor-league \
  -a splendor.agents.our_agents.league_entries.az_search,splendor.agents.generic.random,\
splendor.agents.our_agents.minmax,splendor.agents.our_agents.dqn.population \
  -n 2 -m 75 --segment independent_test --wrap-seeds --workers 16 --output runs/z1-uniform-search
# Z2
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python -m splendor.agents.our_agents.alphazero.train \
  --output runs/z2-az-lite --iterations 10 --games-per-iter 1000 --workers 24 \
  --device cuda --seed 20260913
```
