# 任务计划（2）：3p/4p 多人模型训练

> 状态（2026-09-19）：后续草案，未启动。当前只执行 Task-1 轻量 PPO 路线。
> 前置计划：[任务计划（1）：2p PPO 轻量提升](task-1-2p-improvement-and-seed-protocol.md)。
> 本文保留多人设计参考；预算、manifest、多 seed 与大规模评测条目须在真正启动时重新轻量化，
> 不得自动继承为当前任务，也不再要求通过旧 Task-1 的 T1.7 科学退出门。

## 1. 启动条件与范围

### 1.1 强制启动闸门

任务（2）不与当前 2p 小实验并行启动；须用户另行要求。启动前至少完成：

- 2p 轻量实验已得到一个保留候选，或两轮内有明确的停止理由；不要求完成旧 power/N、sealed 或 T1.7；
- 固定 2p 模型身份并保留原文件，不让多人训练覆盖它；
- 为 3p 重新确定小预算、独立数据划分与 ranking utility，而非直接启动本文完整矩阵；
- `public-v2-multi` 的 337 维特征与 2p 前缀逐位关系已验证；
- 任务（1）的最终测试段不会被多人训练或多人调参复用。

### 1.2 范围

本计划建立两个独立的模型族：

```text
2p：保留当前 ppo-best，不参与多人训练覆盖
3p：public-v2-multi + n_seats=3 的独立 PPO checkpoint
4p：public-v2-multi + n_seats=4 的独立 PPO checkpoint
```

训练顺序为 **3p 先行，3p 通过多人验收后再进行 4p**。两者都使用本地引擎和离线 league，
暂不涉及浏览器部署、远程服务 schema 扩展或网页规则改写。

## 2. 设计事实与决策

### 2.1 特征与动作契约

`public-v2-multi` 为 337 维，前 312 维与 legacy `public-v2` 兼容，新增席位数和循环顺序的额外 rival panel；
动作空间仍为固定的 3510 个动作。

- [ ] 3p、4p 均使用 `feature_version=public-v2-multi`；
- [ ] 保持同一动作枚举、合法 mask、`create_action_mapping` 和终局处理契约；
- [ ] 对 3p/4p 分别检查观测维度、mask 维度、座位范围和初始宝石/贵族数量；
- [ ] 2p `public-v2` checkpoint 不直接加载为多人模型；
- [ ] 可选做“前 312 维权重迁移”和“完全重新初始化”的对照，但不能把迁移结果当作默认正确方案；
- [ ] 所有多人 checkpoint 必须携带 `n_seats`、feature schema、input/output dim 和 reward version。

重要边界：337 维 schema 只是信息布局支持 2–4 席，不表示 2p 模型已经学会 3p/4p 的排名策略。
人数、宝石供给、贵族数量、对手响应次数和终局效用都发生了变化，因此第一阶段采用 per-seat 模型。

### 2.2 多人目标不是二人零和

3p/4p 是排名博弈，不应直接把 2p 的“对手最小化我”或当前 2p minimax 当作唯一目标。
当前 minimax 实现也明确假设两名 agent，不能直接放入多人训练池。

初始 ranking utility 使用一个透明、可替换的基线：

```text
第 1 名：+1
第 2 名： 0
第 3 名：-0.5
第 4 名：-1
```

- [ ] 预先定义平局的平均名次或效用分摊；
- [ ] 以 centered-Borda 和 top-1 utility 做小规模对照，但不能在最终测试后挑选；
- [ ] 终局 critic 预测自己的排名效用，而不是只拟合绝对分数；
- [ ] 可增加辅助 rank/top-1 head，但辅助头的权重、标签和选择规则必须独立记录；
- [ ] 对折叠了多个对手回合的环境步重新校准 `gamma`，`.995`、`.997` 作为预注册候选，不直接假设 `.997` 最优；
- [ ] potential shaping 继续遵守 `r' = r + gamma * Phi(s') - Phi(s)`，非势能事件项单独作为消融。

### 2.3 对手池采用多人风格覆盖

每个 n-seat 模型的对手池应是 n-seat 可运行的策略，而不是把 2p 对手机械复制：

- random；
- n-player heuristic；
- heuristic-rush；
- heuristic-hoard；
- GA；
- 已冻结的 n-player PPO snapshot；
- 后续由当前策略生成的 best-response/exploiter snapshot。

- [ ] 每局记录所有 rival seat 的实际策略和 snapshot；
- [ ] 目标模型轮换到每个座位，敌方风格和座位位置使用平衡排列；
- [ ] 3p/4p 分别统计各对手风格的 rank 分布；
- [ ] minimax 不进入默认 3p/4p pool；若未来实现 max-n 或其他多人搜索，另建计划和独立对照；
- [ ] 保留历史 snapshot，避免新模型只适应当前一代对手而忘记旧策略。

多人训练采用 population / PSRO-like 的逐步扩充思路：当前策略产生新快照，新快照经过独立 league 评估后
才进入 pool；不在同一局中切换多个对手，也不让训练胜率直接决定快照入池。

## 3. 阶段任务

### T2.0 多人训练契约和冒烟

- [ ] 在任务（1）完成后新建 3p、4p 独立 manifest 和 registry seed 段；
- [ ] 验证 `public-v2-multi` 在 2/3/4 席下的维度、循环 rival panel、seat feature 和零填充；
- [ ] 验证初始宝石供应、贵族抽样、终局轮次和排名 utility；
- [ ] 3p、4p 各跑至少 16 局冒烟，覆盖所有目标座位，要求零非法动作、零 schema mismatch、零未处理终局；
- [ ] 冒烟只验收管线，不把胜率或排名当作模型效果结论；
- [ ] 检查 PPO rollout 是否在一局内保持同一策略快照，且对手回合折叠后 return/GAE 没有越界。

**退出门槛**：3p/4p 都能完整生成轨迹并复现相同 scenario；所有排名、平局和折扣样例通过手算测试。

### T2.1 n-player BC/DAgger 数据基础

优先使用同一 `public-v2-multi` schema 生成教师数据，避免先用 2p 观测训练再强行补多人字段。

- [ ] 分别收集 3p、4p 的 heuristic、rush、hoard、GA 轨迹；
- [ ] 轨迹保存观测、合法 mask、教师动作、当前 seat、n_seats、scenario、整局边界和 ranking label；
- [ ] 训练/验证按完整 scenario 切分，不按中间状态随机拆分；
- [ ] 用学生自身访问的 3p/4p 状态做 DAgger，记录每轮数据来源和教师查询数；
- [ ] 检查教师只使用学生可获得的公开信息，不能因为 GA 或搜索窥见隐藏牌而造成信息泄漏；
- [ ] 训练 from-scratch BC 与“从 2p ppo-best 前缀迁移”的小对照，先比较 update-0 的合法率、动作覆盖和排名表现。

**退出门槛**：3p/4p 数据集的 schema、标签、mask、seed split 和信息集均可重放；没有通过审计的教师不进入正式 PPO。

### T2.2 排名奖励与 critic 验证

- [ ] 实现并单测 rank utility 的第一至第四名映射和所有 tie case；
- [ ] 验证终局不 bootstrap，截断局和失败局不静默当作负胜负；
- [ ] 对 `gamma` 候选做小规模配对消融，记录折叠环境步与实际回合的时间尺度；
- [ ] 评估 critic 的 rank calibration、top-1 AUC、平均绝对误差和分层误差；
- [ ] critic 未校准前，不将其作为搜索先验，也不以搜索结果替代训练结果；
- [ ] 记录 absolute score、relative score、rank utility 三种 target 的差异，不用单一 EV 代替多人强度。

### T2.3 3p 训练课程

3p 是正式多人训练的第一站：

1. 3p BC/DAgger-2 初始化；
2. 3p heuristic/rush/hoard/GA 混合 pool；
3. 3p PPO 自博弈和冻结历史 snapshot；
4. 每个目标座位均衡采样；
5. 先进行小预算 pilot，再使用至少 3 个 model seed 做正式训练。

- [ ] pilot 使用固定预算和固定验证 bank，不能根据 pilot 胜率临时改奖励；
- [ ] 正式训练至少包含 3 个独立 model seed；
- [ ] 训练场景从 3p train bank 重新抽样，不复用任务（1）的 2p test；
- [ ] 记录各 opponent style、seat、rank 和行为指标；
- [ ] 3p 训练中保留 2p checkpoint 仅作迁移/参考，不把它算作 3p baseline。

### T2.4 3p league 与启动 4p 的闸门

3p 正式评测使用独立 3p scenario bank：

- [ ] 每个核心 matchup 至少 100–200 个 unique deals × 3 个目标座位的平衡排列；
- [ ] 最终 3p test 至少 200、最好 500 个 unique deals；
- [ ] 使用 CRN 比较 3p PPO、GA、heuristic、rush、hoard、random 和历史 snapshot；
- [ ] 报告第一名率、前二名率、平均排名、各名次分布、平均分、回合数和座位效应；
- [ ] 以 rank utility 的 paired difference 做主要比较，不把 pairwise win rate 当作多人唯一指标；
- [ ] 3p 模型通过预注册的平均排名/第一名率/不回退门槛，且无严重 seat 或 scenario stratum 退化后，才启动 4p。

3p 的门槛具体数值必须在看到正式 test 前写入 manifest。若只有“击败某个 baseline”的点估计而没有区间、
座位平衡或分层结果，不算通过。

### T2.5 4p 训练课程

4p 训练沿用 3p 的数据、奖励、RNG 和报告原则，但使用独立 checkpoint、独立 seed 段和 4p opponent pool：

- [ ] 重新收集 4p BC/DAgger-2 轨迹；不能只复制 3p 轨迹；
- [ ] 使用 4p 初始宝石供应、5 个贵族和 4 席循环 rival panel；
- [ ] 目标策略轮换到 4 个座位，敌方三席的风格排列使用 Latin-square 或等价平衡方案；
- [ ] 重点监控 kingmaking、领先者拒止、第三/第四名效用和多方抢同一贵族；
- [ ] 可把通过 3p 的 checkpoint 做迁移对照，但 from-scratch 4p 必须保留；
- [ ] 4p 正式训练至少 3 个 model seed，训练预算和早停规则在开始前固定。

### T2.6 4p league 与总验收

- [ ] 每个核心 matchup 使用至少 200、最好 500 个 unique deals，并平衡 4 个目标座位；
- [ ] 报告 1/2/3/4 名概率、top-1/top-2/top-3、平均排名、平均分、回合数、tie 和 seat effect；
- [ ] 报告不同 opponent mixture、不同 scenario stratum 和训练 seed 的结果；
- [ ] 与 random、heuristic、rush、hoard、GA 和历史 4p snapshot 做 cross-play；
- [ ] 所有最终 test 只读封存，完成后不再回调 ranking utility、gamma、pool 或 checkpoint；
- [ ] 将 3p/4p 模型和报告明确标记为 n-seat-specific，不能写成统一“通用模型”结论。

## 4. 训练、评测和模型选择协议

### 4.1 多人数据隔离参考（非当前执行要求）

以下保留原多人设计思路，不把旧 Task-1 的生产 roll 或统计门作为启动前提：

```text
master_seed
├── model_seed[n_seats]
├── train_deal_seed[n_seats, game]
├── validation/test scenario seed[n_seats, scenario]
├── seat/permutation seed
├── pool/snapshot seed
└── action/worker seed
```

- [ ] 3p、4p seed 段彼此不重叠；
- [ ] train、validation、test 彼此不重叠；
- [ ] 2p sealed test 不被多人训练消费；
- [ ] 每个 scenario 保存初始棋盘 hash、n_seats、规则版本和代码版本；
- [ ] 3p/4p 使用相同 scenario 的 A/B 比较时按 scenario 聚类 bootstrap，不把不同席位排列误当成完全独立样本。

### 4.2 选模规则

在正式训练前固定一个多人综合指标，例如：

```text
J_n = 平均排名的反向得分
      + lambda1 * top-1 rate
      + lambda2 * top-2 rate
      - lambda3 * seat effect
      - lambda4 * 最弱场景惩罚
```

实际权重必须写入 manifest。选模只看 validation-A/B；最终 test 只用于一次性报告。

不允许只因为某个 4p seed 的第一名率最高，就忽略平均排名、第三/第四名分布、seat effect 或 kingmaking 场景。

## 5. 产物与验收门槛

每个 n-seat 训练目录至少包含：

- `manifest.json`：n_seats、feature schema、reward version、gamma、pool、seed groups、模型选择规则；
- n-seat BC/DAgger 数据摘要和信息审计记录；
- 逐局 scenario、座位排列、对手 snapshot、WDL/rank、分数和行为 JSONL；
- PPO 更新日志、critic calibration、rank utility 统计、失败局和成本；
- 3p/4p 独立 checkpoint 及 SHA-256；
- 独立 league Markdown/JSON 报告；
- 迁移初始化与 from-scratch 的对照结果。

自动质量门：

- 3p/4p 所有 schema、mask、action mapping 和 terminal wrapper 测试通过；
- 冒烟和正式训练零非法动作、零未处理终局、零 seed split 冲突；
- rank utility、tie、return/GAE 和折叠回合的手算测试通过；
- 3p、4p 分别至少完成多 seed 训练和独立 league；
- 结果可由 manifest 和 scenario bank 重算。

人工审查门：

- 对局行为不存在明显的无意义囤宝石、永不抢贵族或固定座位偏好；
- 多人排名结果没有被异常 kingmaking 或规则 bug 主导；
- 3p/4p 的结论没有使用 2p 评测数字冒充多人泛化；
- 2p、3p、4p checkpoint 的 schema 和适用席位数在报告中明确标注。

## 6. 风险登记

| 风险 | 缓解措施 |
|---|---|
| 把 2p 胜负目标直接迁移到多人 | 使用排名 utility、平均排名和 top-k 指标；独立 n-seat critic |
| 将 public-v2-multi 的结构支持误当成已训练能力 | 3p/4p 独立训练、独立 league、保留 OOD 标注直到通过验收 |
| minimax 对多人失效 | 默认使用 n-player heuristic/GA/snapshot pool；max-n 另立计划 |
| 4p kingmaking 与隐性联盟 | rank 分布、领先者拒止和压力场景单独统计；多样 snapshot pool |
| 3p/4p 座位偏差 | 目标座位轮换、敌方排列平衡、报告 seat effect |
| 3p 轨迹直接复用到 4p | 4p 重新收集 BC/DAgger 数据和 train bank |
| 迁移权重掩盖真实学习收益 | from-scratch 与 prefix-transfer 并列对照 |
| 只优化第一名率而牺牲其他名次 | 预注册综合指标，同时报告完整 rank distribution |
| 把 folded step 当作普通回合 | 单独校准 gamma、return、GAE 和终局 bootstrap |

## 7. 里程碑与依赖图

```text
任务（1）完成
      ↓
T2.0 schema/RNG/reward 冒烟
      ↓
T2.1 3p BC/DAgger ──► T2.2 ranking critic
      ↓                         ↓
T2.3 3p PPO ───────────────► T2.4 3p league gate
                                      ↓
                              T2.5 4p PPO
                                      ↓
                              T2.6 4p league/总验收
```

建议的最小执行节奏：

1. 3p、4p 各做一次 16 局冒烟，只验证管线；
2. 完成 3p 数据、奖励和 pilot；
3. 3p 至少 3 个 model seed 正式训练并完成独立 league；
4. 3p 通过闸门后，建立 4p 数据和正式训练；
5. 4p 通过独立 league 后，才讨论统一 2p/3p/4p policy 或浏览器部署。

## 8. 依据与关联文档

- [任务计划（1）：2p PPO 轻量提升](task-1-2p-improvement-and-seed-protocol.md)
- [shared feature schemas](../src/splendor/splendor/features_v2.py)
- [整体提升路线](../docs/IMPROVEMENT_ROADMAP_20260912.md)
- [C4-R2 league report](../runs/c4r2-league-20260914/league_report.md)
- [remote inference feature/seat contract](phase-6-remote-inference.md)
- [seed registry](../docs/seed_registry.md)
