# E 阶段实施记录（E1/E2/E4，2026-09-13）

> 依据 `docs/IMPROVEMENT_ROADMAP_20260912.md` §6。代码入库提交：
> E1 主体随 `43f37b8`（ppo_selfplay.py +102 行、stabilization.py +43 行）、
> E2 随 `e2bb3dc`、E4 随 `aca98a9`。冒烟产物 `runs/e1-smoke-3p/`、
> `runs/e1-smoke-4p/`（runs/ 不入库，证据字段见本文引用）。

## E1 per-seat 训练入口 — 达成

- `PPOConfig.n_seats`（2–4；>2 强制 `public-v2-multi` schema）。
- `collect_ppo_game`：对局构造 `LimitRoundsGameRule(n_seats)`，rival 按
  座位惰性构建（同一池条目服务全部对手席位），`_outcome`/`rival_score`
  改为对最强对手判定（2p 语义逐字节不变）。
- 跨 schema 冷启动：`_build_initial_model` 允许 `scratch` 初始化在
  BC schema ≠ 目标 schema 时以 identity normalizer 直接构建（3/4p 尚无
  multi BC 数据集前的唯一路径；其余组合仍硬报错）。
- 驱动 `--seats {2,3,4}` + `--feature-version` 覆写，manifest 记录。
- **冒烟**：3p/4p 各 1 update × 16 局 × seed42（scratch，池 ga+heuristic）：
  `training_failed_games=0`、`training_records=16`、0 非法动作；
  用时 132.7s（3p）/ ~135s（4p）/ job。
- 观测维度与掩码一致性由 `tests/test_features_v2.py` 锁定（2/3/4 席同维
  337，2 席前缀 = legacy 逐位相等）。

## E2 排名效用 — 代码就绪

- `RANK_UTILITIES = {1:+1, 2:0, 3:-0.5, 4:-1}`，同分平均；
  `RankUtilityWrapper` 与 `TerminalRewardWrapper` 同构（terminal_scale=10
  保持 ±10 量级），测试含名次映射/同分/4p 端到端局。
- γ 0.99→0.997 是 E3 训练配置项（E3 未启动，见下）。

## E4 胜率估计器多席化 — 达成

- 仅 legacy `public-v2` 仍锁 2 席；`public-v2-multi`（与 v1）支持 2..4 席。
- 测试：2/3/4 席估计构造 + legacy 拒绝 3 席（`tests/test_winrate_estimator.py`）。

## E3（3/4p league 建档与 GA 4p 基线）— 未启动，理由与入口

依赖"3/4p 完整训练"（每 seat count 一个模型 × 数百 update）。按 C2 实测
吞吐（3p ≈ 132.7s/update×16 局 → 500 updates ≈ 4.6h/seed），E3 立项建议：
- 先跑 3p/4p 各 1 seed × 500 updates 冒烟规模训练（~4.6h × 2），
- 再用 `splendor-league -n 3/4`（A1 已支持多席）建首表；
- GA 4p 无需改动（fitness 天生 4 人局）。
本轮执行到此为止：E1 验收（冒烟 + 维度一致）已达成，E3 属训练排期。
