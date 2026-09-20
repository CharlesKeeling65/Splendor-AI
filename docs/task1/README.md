# Task-1：当前路线与历史证据

## 当前：2p PPO 轻量提升（2026-09-19）

唯一执行入口是 [轻量主计划](../../plan/task-1-2p-improvement-and-seed-protocol.md)：
已有 PPO checkpoint → 最小 warm-start → 单 seed 定向练 heuristic/rush → 少量同局比较。
当前只完成计划重构，续训入口与新训练尚未实施。

首轮 100 updates × 16 新对局；开发评测最多 480 局；有改善才做一次 800 局复核，
仍受每轮 2 小时上限约束。最多两轮，不自动扩为网格、多 seed 或完整科研流程。
旧 ppo-best 不覆盖，validation-B/sealed/reserve 不消费。

## 保留的历史工作

旧 T1.0–T1.7 计划已[归档](../../plan/reference/TASK1_SCIENTIFIC_PROTOCOL_20260919.md)。
下列材料是实现与结果记录，不是当前训练前置任务，也不构成重启旧实验的授权。

| 记录 | 内容 |
|---|---|
| [T1.0 audit](T1.0_BASELINE_AUDIT.json) / [manifest](T1.0_BASELINE_MANIFEST_V2.json) | 原模型与历史联赛的冻结身份、样本边界 |
| [T1.1 RNG](T1.1_RNG_PROTOCOL.md) | 事件键随机流、配对 schedule 与显式 pool |
| [T1.2 banks](T1.2_SCENARIO_BANK.md) | ScenarioV1、split、跨 runner 初态与 sealed gate |
| [T1.3 statistics](T1.3_PAIRED_STATISTICS.md) | 正式配对评测、聚类统计与 power 设计 |
| [T1.4 seed roll](T1.4_SEED_ROLL.md) / [root record](T1.4_PRODUCTION_SEED_ROLL.json) | 已批准的一次性生产 roll；不重新抽 root |
| [基线结果](T1.4_BASELINE_RESULTS.md) | 1,800 局新协议基线；非 sealed A200/stress100 |
| [训练编排](T1.4_PILOT_ORCHESTRATION.md) | 已完成的六 job 生命周期、资源与证据契约 |
| [奖励契约](T1.4_REWARD_CONTRACT.md) | O 的 safe-PBRS 与历史 O_bridge 的区别 |
| [候选排序](T1.4_RANKING_UPDATE_20260917.md) | r1-O 优先作续训起点，但不是独立泛化排名 |
| [分析 notebook](T1.4_PILOT_ANALYSIS_20260919.ipynb) / [结果 JSON](T1.4_PILOT_ANALYSIS_20260919.json) | completion receipt 复核、selected/final 对比与证据限制 |
| [隔离观察](T1.4_HEAD_TO_HEAD_R1O_VS_PPO_20260917.md) | 缺逐局持久化证据的旧 head-to-head，不用于结论 |

## 已完成与未证明的边界

retry-4 的 O/O_bridge × r0/r1/r2 六 job 已各完成 2,000 updates；
receipt 绑定 534 文件、3,741,553,697 bytes，已复核。
但旧 selector 只有 10 scenarios、每 job best-of-41，且缺 heuristic-rush；
正式 T1.4 统计退出门**未完成**。轻量化不是把这个门改成通过，而是暂停正式研究路线。

2,800 局 joint pilot、power/N 冻结、多 replicate 确认与 sealed 批次不再是下一步。
尚未提交的 joint-proposal 扩展已退出活动代码，本地可恢复副本见主计划 §5。
现有协议实现、root、banks、checkpoints 和所有已完成结果不删除、不改写。
