# D1 报告：DQN 200k 完整课程 ×3 seed（2026-09-13）

> 依据 `docs/IMPROVEMENT_ROADMAP_20260912.md` §5 D1。产物 `runs/d1-200k/`，
> 门槛评测 `runs/d1-200k/gate-*.json`（评测种子 826100/826101，training 段内）。

## 配置

- `dqn --opponent-pool random:0.5,minimax:0.5 --test-opponent random
  --total-steps 200000 --seed {826000,826001,826002}`（默认超参，CUDA）。
- 训练用时 ≈ 3.4 h/seed（三 seed 并行 ~4.6 h，与 29 步/s 冒烟外推一致，
  GPU 由 C2 共享时降至 ~20 步/s）。

## 门槛结果（每 gate 100 局，双座次）

| seed | M2：vs random（≥90%） | M3：vs minimax（≥55%） |
|---|---|---|
| 826000 | **98%** ✓ | 14% ✗ |
| 826001 | 82% ✗ | 3% ✗ |
| 826002 | **93%** ✓ | 16% ✗ |
| 均值 | 91% | 11% |

训练内贪心评估（20 局/万步）在 50–60k 步即达 vs random ~100%，
但 200k 步仍远未学会对 minimax 拿胜势。

## 结论（训练量假设的检验）

1. **M2 基本达成但 seed 方差大**（82–98%）；50% random 混池下 200k 步对
   "碾压 random"是充分的（2/3 seed），第三 seed 提示课程前段噪声大。
2. **M3 未达成且差距悬殊**（11% vs 55% 门槛）。与 corrected DQN 20k 步
   52.7% 的对照说明：**对 minimax 的实力不来自"步数堆量"，而来自
   experiment.py 管线的专用组件**（搜索引导 guidance、EMA 稳定化、
   成对牌局审计、冻结历史对手池）。plain `dqn` CLI + 50/50 混池在 200k
   步上是对该管线的明显降级。
3. **训练量假设被否定（对 M3 而言）**：200k × 混池 ≠ M3。若要复现/超越
   52.7%，应走 `dqn/experiment.py` 的 search/guidance 变体及其多阶段
   课程（M1→M2→M3 逐级换对手），而非单一混合池长跑。
4. 与 C2（PPO 自博弈）对照：PPO 侧同预算量级拿到 minimax 57.3%——
   on-policy 自博弈 + 风格化池在 minimax 维度的样本效率显著更高。

## 对路线图的回写建议

- D1 验收（M2 与 M3 逐 seed 报告）已完成 reporting 义务；
- G3 的"DQN 200k 完整课程"达成，但 M3 门槛未过——**如实记录**；
- D3（价值蒸馏）的教师应选 corrected DQN（52.7% 那一代）而非本轮
  200k plain 产物；`auxiliary_heads` 教师需经 `experiment.py` 训练。
