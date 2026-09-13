# C2 规模化自博弈与 C4 league 体检报告（2026-09-13）

> 依据 `docs/IMPROVEMENT_ROADMAP_20260912.md` §4（C2/C4）。种子段：训练
> `c2_training` 830000–853999（每 seed 8000 局），验证 `c2_validation`
> 854000–854009，独立测试 `c2_test` 855000–855024（封存）。产物：
> `runs/c2-selfplay-500/`（训练+终测）、`runs/c4-league-20260913/`（league 矩阵）。

## 配置

- C1 消融胜出配置：`value_coefficient=1.0` + `critic_learning_rate=5e-4`
  （见 `docs/PPO_CRITIC_ABLATION_20260913.md`），其余同 frozen 稳定化配置。
- 规模：500 updates × 16 局/更新 × 3 seed（42/1234/2024），fixed 变体，
  验证评测每 25 updates（10 deals × 3 对手 × 双座次 = 60 局/轮）。
- 训练池（C3 风格化）：ga + heuristic + heuristic-rush + heuristic-hoard +
  minimax（等权），current/history 不对称保留沿用协议。
- 初始化：DAgger-2 BC（`formal-3.3-20260907/bc-dagger-2/best.pth`）。
- 用时：3 并行 job ≈ 1.7 h（P5000 CUDA，8.4–11.3 s/update）。

## C2 独立测试（c2_test 段，每对手 50 局 = 25 deals × 双座次）

| seed | vs random | vs heuristic | vs minimax | vs ga | EV(last 3 updates) | 验证选模 |
|---|---|---|---|---|---|---|
| 42 | 100% | 34% | 52% | 38% | **0.483** | 37/60 @u350 |
| 1234 | 100% | 44% | 56% | 44% | 0.380 | 34/60 @u250 |
| 2024 | 100% | 34% | 48% | 52% | 0.339 | 37/60 @u400 |
| **均值** | **100%** | **37%** | **52%** | **45%** | **0.401** | — |

对照稳定化基线（8 updates，2026-09-08）：random 100% / heuristic 32% /
minimax 48.7% / GA 54% → C2 均值 **heuristic +5pt、minimax +3pt**，
GA −9pt（seed 间方差大：38–52%）。

## G2 判定（EV ≥ 0.5）

**未达成，但趋势明确**：EV 从 8-update 试点的 0.12–0.23 提升到 500-update 的
0.34–0.48；seed42 达 0.483。EV 随训练规模仍在上升，未现平台（曲线见
result.json logs）。按"如实记录"纪律：G2 记为 **partially met**，后续扩到
2000 updates 或加强塑形（B2 的 potential shaping 尚未接入 PPO 主线）是
明确的下一步。

## C4 league 体检（independent_test 段 829000–829049 wrap，每对阵 75 局 × 双座次 = 每对手 150 局）

选模：按验证整数胜局（37/60 并列）+ EV tiebreak → seed42 `best.pth`（u350）。

### Win-rate matrix（行胜列，ties 分半）

| agent | ppo-best | random | heuristic | rush | hoard | minimax | ga |
|---|---|---|---|---|---|---|---|
| **ppo-best** | — | **100%** | 38.7% | 38.0% | 43.0% | **57.3%** | 47.3% |
| heuristic | 61.3% | 100% | — | 50.0% | 50.7% | 60.7% | 53.3% |
| heuristic-rush | 62.0% | 100% | 50.0% | — | 50.7% | 62.0% | 53.3% |
| heuristic-hoard | 57.0% | 100% | 49.3% | 49.3% | — | 54.0% | 46.7% |
| minimax | 42.7% | 100% | 39.3% | 38.0% | 46.0% | — | 44.7% |
| ga | 52.7% | 100% | 46.7% | 46.7% | 53.3% | 55.3% | — |
| random | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

Wilson 95% 区间见 `league_report.md`（每格 150 局，±8pt 量级）。

### G1 判定

| 门槛 | 目标 | 实测 | 判定 |
|---|---|---|---|
| vs minimax | ≥ 60% | 57.3% [49.3, 65.0] | **未达**（超 DQN corrected 52.7%） |
| vs heuristic | ≥ 60% | 38.7% [31.2, 46.7] | **未达** |
| vs GA | ≥ 55% | 47.3% [39.5, 55.3] | **未达** |
| vs random | ≥ 99% | 100% | **达成** |

### 保持型测试（训练池内 vs 池外对手）

- 池内（训练池成员）：heuristic 38.7% / rush 38.0% / hoard 43.0% /
  minimax 57.3% / ga 47.3% → 均值 **44.9%**
- 池外：random 100%。池外样本太少（仅 random），池内低胜率说明问题不是
  经典"过拟合训练分布"，而是**对 heuristic 族的策略性缺口**。

## 归因与下一步（记入风险登记册对话）

1. **heuristic 族缺口是最大单项损失**。启发式（尤其 rush 变体）靠即时买分，
   C2 的 value 学到了长线但缺乏对"对手快速抢分"的针对性压力。候选药方：
   a) B2 potential shaping 接入 PPO 主线（score-lead 势能直接奖励领先）；
   b) 池权重向 heuristic 族倾斜（显式记录）；
   c) 更大规模（EV 未平台，2000 updates 外推 ~7h/seed）。
2. **选模噪声**：验证 60 局的整数胜局方差 ±5 局，seed42/2024 并列 37——
   扩大验证规模（validation_deals 10→25）可降噪。
3. minimax 57.3% 已是仓库对 minimax 的最佳记录（DQN 52.7%、旧 PPO 48.7%）。

## 复现

```bash
# C2（c2_training/c2_validation/c2_test 种子段，manifest 自动记录）
PYTHONHASHSEED=0 python -m splendor.agents.our_agents.policy_imitation.stabilization \
  --output runs/c2-selfplay-500 --device cuda --workers 3 \
  --updates 500 --games-per-update 16 --eval-every 25 \
  --variants fixed --seeds 42 1234 2024 \
  --value-coefficient 1.0 --critic-learning-rate 5e-4 \
  --pool-names ga,heuristic,heuristic-rush,heuristic-hoard,minimax \
  --seed-base 830000 --training-seed-count 24000 \
  --initial-bc runs/policy-imitation/formal-3.3-20260907/bc-dagger-2/best.pth
# C4
SPLENDOR_PPO_CHECKPOINT=runs/c2-selfplay-500/training/fixed-seed42/best.pth \
PYTHONHASHSEED=0 splendor-league -a "<roster 见 league_results.json roster>" \
  -n 2 -m 75 --segment independent_test --wrap-seeds --workers 8
```
