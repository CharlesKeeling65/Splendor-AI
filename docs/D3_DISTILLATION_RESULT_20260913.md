# D3 结果记录：价值先验蒸馏的首次验收（2026-09-13）

> 依据 `docs/IMPROVEMENT_ROADMAP_20260912.md` §5 D3。产物：
> `runs/d3-distilled-bc/`（蒸馏 BC）、`runs/d3-distilled-ppo/`（2 updates × 3 seed 验收跑）。

## 机制交付（全部落地）

- `policy_imitation/distillation.py`：masked-KL 蒸馏，q-softmax 模式可用任意
  Q checkpoint；`policy-imitation distill-dqn` CLI（确定性 seed 切分入元数据）。
- 蒸馏教师：D1 的 seed826000 200k checkpoint（q-softmax，T=1.0，10 epochs），
  数据集 `ga-random-v1.npz`（教师信息审计依 D1 报告结论：该教师为 plain 管线
  产物，public-info 一致性审计在 corrected 代际完成）。
- 蒸馏出的 BC checkpoint 直接通过稳定化驱动的 `--initial-bc` 路径冷启动
  PPO（update-0 全程零非法动作）。

## 验收判定：未达标（如实记录）

| seed | update-0 验证胜局（ga/heuristic/minimax 各 20 局） |
|---|---|
| 42 | 2/60 |
| 1234 | 2/60 |
| 2024 | 2/60 |
| **均值** | **2/60（基线 DAgger-2 = 22/60）** |

## 归因

1. **教师太弱**：D1 报告已证 200k plain 课程 vs minimax 11%、且从未对局
   ga/heuristic——蒸馏目标本身不含对这三个验证对手的制胜信号。D3 的前提
   （结果文档 §4.4 的信息审计 + 有效教师）在 corrected DQN 代际才成立。
2. q-softmax 的温度/尺度（return 量级 Q 值）未调，分布过平或过尖都可能；
   属超参问题而非机制问题。
3. F1 已独立证明 return 模式价值头胜率判别力不足（AUC 0.525）——与本次
   否定结果互相印证。

## 结论与下一步

- 机制保留（管道、CLI、测试全绿）；验收门槛未达。
- 重试前提（按优先级）：① 用 `experiment.py` 训练 corrected 代际 +
  `auxiliary_heads=True` 的教师（policy-head 模式蒸馏 + 价值头可用性受 F1
  门约束）；② 教师须在 ga/heuristic/minimax 混合课程上有效；
  ③ 蒸馏超参（T、epochs）单独消融。
- 在上述前提满足前，PPO 初始化继续使用 DAgger-2 BC（22/60 基线）。
