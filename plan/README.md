# Splendor-AI 升级计划总览（plan/）

> **一句话定位**：把本仓库升级为「本地引擎高速训练 DQN → 统一环境接口 → 浏览器层部署真实网页对局」的完整 sim-to-real 管线。
> 目标网页版：https://game.hullqin.cn/ccbs（HullQin《璀璨宝石》）。

## 文档体系

### 训练方向增量计划（2026-09-07）

- [策略模仿与自博弈实施计划](policy-imitation-selfplay.md)：教师评测 → BC → DAgger → PPO 对手池训练；保留 corrected DQN 对照，MCTS 按收益门槛进入。**待实施**，不代表已完成训练或替换部署模型。
- 依据：[DQN 第二轮结果与选模纠错](../docs/DQN_ROUND2_RESULTS_20260907.md)；已有实验协议见 [dqn-round2.md](dqn-round2.md)。

### 对局辅助面板计划（2026-09-14）

- [Phase-7：浏览器对局辅助面板](phase-7-browser-advisor.md)：只读 Advisor（GA 快评 + minimax 深评 +
  牌堆差集直方图 + 对手预留记忆重建），复用 P2 浏览器层与 P3 部署链路，零执行器依赖。
  **v0/v1 已实现（2026-09-14，离线质量门全绿）**；E7 实验、油猴叠加与真实房间验收待执行。

以下 P0–P6 保留原有升级计划；上述文档补充后续训练路线，不覆盖引擎、浏览器及存量算法的兼容约束。

```
plan/
├── README.md                        ← 本文：阶段总览 + 验收体系说明 + 导航
├── phase-0-foundation.md            P0 地基与对齐（性能/注册表/协议/网页实测）
├── phase-1-dqn-training.md          P1 DQN 本地训练（与 P2 并行）
├── phase-2-browser-layer.md         P2 浏览器层最小闭环（与 P1 并行）
├── phase-3-web-deployment.md        P3 网页部署（sim-to-real 首秀）
├── phase-4-enhancements.md          P4 高保真与增强（可选，按 P3 数据决策）
├── phase-5-engineering.md           P5 工程化固化（测试/CI/文档）
├── phase-6-remote-inference.md      P6 本地浏览器控制 + 远程推理 + 实时胜率仪表盘
├── phase-7-browser-advisor.md       P7 浏览器对局辅助面板（只读 Advisor，v0/v1 已实现）
└── reference/                       四份原始文档（依据归档，内容未改动）
    ├── UPGRADE_ROADMAP.md           架构裁决与勘误（含 5 项源码验证裁决）
    ├── DQN_GUIDE.md                 DQN 算法完整方案（超参/骨架/陷阱）
    ├── BROWSER_RL_MAPPING.md        网页版 ↔ 仓库 RL 环境映射（实测依据）
    └── IMPLEMENTATION_SPEC.md       实现规格书（任务看板 + 函数级框架）
```

**阅读路径**：实施者按 `phase-N` 顺序读（每份阶段文档自带任务、代码说明、验收标准）；
想了解"为什么这样设计"先读 `reference/UPGRADE_ROADMAP.md`；写代码时对照 `reference/IMPLEMENTATION_SPEC.md` 的函数签名。

## 阶段总览

| 阶段 | 目标 | 关键产出 | 自动验收（门槛） | 人工验收（关键项） | 依赖 | 估时 |
|---|---|---|---|---|---|---|
| **P0** 地基与对齐 | 性能瓶颈消除、身份桥梁、环境契约、规则事实核查 | ActionIndexCache、Card/NobleRegistry、SplendorEnvBase 协议、6 项网页实测记录 | 等价性/完备性/回归测试全绿 | E1–E6 实测完成并裁决、吞吐对比记录 | — | 2~3 人日 + 1~2 天实测 |
| **P1** DQN 本地训练 | DQN 从零训到超越 baseline | `dqn/` 六件套 + checkpoint + stats.csv | M1 得分>10 / M2 vs random>90% / M3 vs minimax≥55% | 训练曲线审查、对局观战定性、3-seed 方差审查 | T0.1, T0.3 | 4 人日 + 3~5 天挂机 |
| **P2** 浏览器层闭环 | 网页座位 → 与本地同协议的 gym 环境 | `browser/` 七件套 + 奇偶测试 + 夹具 | 特征/掩码双奇偶逐位相等（≥1000 状态） | 网页观战 5 局、奇偶差异归因、夹具覆盖审查 | T0.2, T0.4 | 8~9 人日 |
| **P3** 网页部署 | checkpoint 上网页实战，量化落差 | `play-web` 命令 + s2r 对照报告 + 经验回流 | 50 局稳定、报告自动产出 | 落差归因评审、礼仪终审、人机对局体验 | P1 + P2 | 3~4 人日 |
| **P4** 高保真增强 | 按 P3 落差数据选择性升级 | 支付枚举/特征 v2/记忆重建/自博弈回流 | 消融 tier-2 ≥ tier-1 等 | go/no-go 评审、行为定性审查 | P3 | 2+ 周（可选） |
| **P5** 工程化固化 | 成固化为仓库一等公民 | tests + CI + Makefile + 文档更新 | CI 全绿 | 文档与结论表审查 | 各阶段 | 2 人日 |
| **P6** 远程推理与可视化 | 本地浏览器控制、远程 checkpoint 推理、胜率事件流 | TCP JSONL `inference-server`、`play-web-remote`、`play-dashboard`；DQN/前馈 imitation-PPO policy adapters；schema/seat guards | remote policy/protocol/options/dashboard 离线测试 + CI typecheck | 单/多 bot 实际 DOM panel guard、事件流与曲线人工验收（需账号） | P2/P3/P5 | 2~3 人日 |

依赖图：`P0 → {P1 ∥ P2} → P3 → P4`；P5 贯穿，P6 依赖 P2/P3 的浏览器边界与 P5 的质量门。**关键路径先启动项：T0.4（网页实测）同时阻塞 P2 的两个任务，应最先安排。**

## 验收体系说明（双轨制）

每个阶段同时给出两类验收标准，两者**都通过**才算阶段完成：

- **自动验收**：可由测试脚本 / CI / 统计命令机器判定的门槛。特点：客观、可重复、进 CI 后防回归。局限：只能验证"符合规格"，不能验证"规格本身对不对"。
- **人工验收**：需要人观察、判断或裁决的检查项。四类典型场景：
  1. **事实核查类**（如 P0 的网页实测）：只有人能操作真实浏览器并记录现象；
  2. **定性判断类**（如 P1 的对局观战）：策略"行为是否合理"没有客观指标——赢率达标但行为荒谬（靠 exploit bug）是真实风险；
  3. **统计纪律类**（如 3-seed 方差审查）：防止单次幸运实验被当成结论；
  4. **伦理合规类**（如 P2/P3 的礼仪审查）：对真实服务器的行为约束必须有人签字确认。

## 架构一图流

```
┌────────────────────────────────────────────────────────────┐
│ L4 部署评测：general_game_runner（本地）│ play-web（网页）│ 回流 │
├────────────────────────────────────────────────────────────┤
│ L3 浏览器层（P2/P3）：Driver │ DOM抽取 │ 伪状态 │ 执行器 │ 会话 │
├────────────────────────────────────────────────────────────┤
│ L2 策略层（P1/P6）：QNetwork │ imitation-PPO │ 训练/远程评分 │
│     —— 环境无关，只依赖 L1 协议；DQN replay 仍仅用于 DQN ——  │
├────────────────────────────────────────────────────────────┤
│ L1 引擎接口层（P0）：SplendorEnvBase 协议 │ 索引缓存 │ 注册表 │
└────────────────────────────────────────────────────────────┘
```

核心原则：**单接口、双环境**。本地 `SplendorEnv` 与 `BrowserSplendorEnv` 实现同一协议；本地 `play-web` 保持 DQN-only，P6 通过远程 scored-policy seam 支持 DQN 或前馈 imitation-PPO。DQN checkpoint 可在远程服务复用，DQN replay 回流仍不适用于 on-policy PPO。

## 五项源码裁决（必须先知道的事实）

详见 `reference/UPGRADE_ROADMAP.md` §1，摘要：

1. 牌库 **90 张**（40/30/20）——"78 张"是把发牌后剩余（36/26/16）误当总数；四元组唯一性在 90 张上成立。
2. 265 维观测**天然与网页信息集对齐**（只含自己的预留牌、对手只有分数）→ 记忆重建不在关键路径。
3. **支付方式是唯一硬语义差距**：引擎只生成一种贪心支付，网页让玩家自选且影响后续状态。
4. 引擎有**非标准规则**（强制拿宝石、同色 7 张上限），与网页一致性待实测（P0 的 E5/E6）。
5. 265 维观测**不含公共宝石供给**（`extract_metrics` 从不读 `board.gems`）——特征盲区，P4 可选项。
