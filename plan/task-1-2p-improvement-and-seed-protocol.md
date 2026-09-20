# 任务计划（1）：2p PPO 轻量提升

> 2026-09-19 完整重构；这是当前唯一的 Task-1 执行计划。
> 目标：用较少训练与评测，找到比父模型更会打 heuristic/rush 的 2p 候选。
> 本轮只改计划，未启动续训。后续执行按下述小预算进行，不自动扩大实验。
> [旧科学协议](reference/TASK1_SCIENTIFIC_PROTOCOL_20260919.md)仅供历史追溯，不再是训练前置条件。

## 1. PPO 本质：提升来自哪里

PPO 根据当前策略新打出的对局，把“比预期更好的动作”概率调高，把更差的调低，并限制每次改动：

```text
ratio = π_new(action | state) / π_old(action | state)
L_actor = -mean(min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage))
L = L_actor + c_v × L_value - c_entropy × entropy
```

因此先看三个问题，而不是先加协议：

1. **有没有练到短板？** 对手分布决定访问哪些局面。不会应对 rush，就增加对阵 rush 的新轨迹。
2. **动作好坏是否估得准？** 奖励、终局处理、critic 和 GAE 共同决定 advantage；critic 拟合变好不等于棋力变强。
3. **每次更新是否适量？** 学习率、epoch、clip、KL 和熵控制改动幅度与探索；loss 下降不等于赢得更多。

当前 [ppo_selfplay.py](../src/splendor/agents/our_agents/policy_imitation/ppo_selfplay.py)
已有 clipped loss、GAE、优势归一化、合法 mask、熵、梯度裁剪和 KL 停更。
不重写 PPO；每次收集新的 on-policy 轨迹，用完即弃，不把旧对局日志当 replay buffer。

## 2. 起点与不动的部分

- 续训父模型：[r1-O/best.pth](../runs/task1-t14-crossed-pilot-20260915/pilot-retry-4/training/r1-O/best.pth)，
  对应 update-1850，SHA-256 `d6c32d039febd0c397838b485779bff7cd8df69c35adb34fd3ee495c410785e6`。
  选它是复用已经训练好的 safe-PBRS 模型，不是认定它已在独立测试胜过旧模型。
- 原部署候选 [ppo-best](../runs/c2r2-selfplay-2000/training/fixed-seed1234/best.pth)
  保持原样；SHA-256 `e225464c17a783bd91b51251336f917e9f867af4f475d8414372885fbb758102`。
- 保持 `public-v2`、312 输入、3510 动作、4×128 网络、2p、原 normalizer；
  不动引擎、掩码、浏览器或其他算法。
- 保持父模型奖励：Δscore + 终局 ±10/0 + safe potential（κ=0.05，终局 Φ=0）。
  这不是“纯胜负”目标；先保持不变，避免同时改采样和奖励。
- 复用已实现的 RNG、ScenarioV1 和评测组件；不改旧 formal API，不弱化 sealed gate。

历史六 job 已各完成 2,000 updates；但 10-scenario、best-of-41 排名只是开发线索。
[已有分析](../docs/task1/T1.4_PILOT_ANALYSIS_20260919.ipynb)保留，
不为继续开发补跑 2,800 局 joint pilot，也不把旧统计退出门写成“已通过”。

## 3. 只走三步

### L1：补上小型续训入口

现有 `train_ppo_selfplay(initial_bc=...)` 只加载 BC，**不能把 PPO 文件直接塞给它**。
`load_ppo_checkpoint()` 已存在，但不等于已有训练 resume 支持。

- [ ] 加一个显式 PPO warm-start 入口，复用现有 collector、GAE、`ppo_update` 与 evaluator。
- [ ] 原样加载 actor、critic、normalizer 和模型契约；拒绝维度、席位或 reward 不匹配。
- [ ] 使用新 Adam，记录 `warm_start` 而非“精确断点续跑”（旧 checkpoint 没有 optimizer state）。
  已有 critic 不重新初始化，显式设 critic_warmup_epochs=0，不重复 BC 阶段的 warmup。
- [ ] 只增加本轮需要的选项：父 checkpoint、pool 权重、updates、seed、输出目录。
  简单配置文件即可；不搭新 orchestrator、通用 manifest 或统计框架。
- [ ] 做定向测试：加载前后 logits/value/normalizer 相等；一个小 rollout/update 可完成；
  非法 mask/NaN 拒绝；旧 BC 入口仍可用。只跑相关测试，不跑全量仓库测试。

### L2：一次定向小续训

假设：增加 heuristic/rush 暴露，能改善当前策略在这些局面上的决策。
这次是候选开发，不做“pool 具有因果收益”的统计证明。

| 项目 | 首轮设置 |
|---|---|
| 训练量 | 1 个 seed；100 updates × 16 新对局 = 1,600 局 |
| 对手权重 | heuristic : heuristic-rush : heuristic-hoard : GA : minimax : current : history = 2 : 2 : 1 : 1 : 1 : 1 : 1 |
| 采样解释 | 保留原池成员，两个目标对手各 2/9，其余各 1/9；history 权重是整个桶的权重 |
| 初始 history | 放入冻结父模型；随后沿用现有 snapshot 逻辑，最多 4 个；记录实际采样次数 |
| 优化器 | actor/shared lr=1e-4；critic head lr=5e-4；value coefficient=1 |
| 其余超参 | γ=.99、λ=.95、clip=.2、entropy=.005、target KL=.02、4 epochs、batch=256 |
| 评测时点 | update 0 / 50 / 100；greedy + legal mask，不临时增加选模频率 |
| 时间上限 | 每轮训练与评测合计最多 2 小时；前 10 updates 实测速率，不预设一定跑满 |

上述超参沿用父配置；**首轮唯一的策略实验因素是 pool**。
warm-start 重建 Adam 是工程边界，所以即使变强，也不单独归因于 pool。
训练 seed 固定并写入配置；Python/NumPy/Torch 都设 seed，已有局部 RNG 可直接复用。
训练初态从现有非 sealed 96k train bank 的 r1 区块固定取 1,600 个，席位均衡；
这是重复训练初态上的新 on-policy 轨迹，不声称这些初态从未见过。不新 roll root，不读 reserve。

### L3：少量同局对比，决定保留还是停止

**快速开发集**：validation-A200 按 source_seed 排序后取零基切片 `[10:30]`，避开旧 selector 的前 10 个。
固定 20 scenarios × 4 对手（heuristic/rush/GA/minimax）× 2 座位 = **160 局/模型**。
update 0/50/100 合计最多 **480 局**。配置中保存确切 scenario IDs，不按文件偶然顺序抽取。

- 每个对手报告 W/D/L、score rate=(W+0.5D)/N，以及相对 update-0 的变化；失败局单列，不静默删分母。
- 先排除 GA 或 minimax 比父模型下降超过 5 个百分点的候选，再比较 heuristic/rush 平均 score rate；
  两个目标对手各自都不能低于父模型。平分选更早 checkpoint；父模型也参与选择。
- 5pp 是小样本开发阶段的容忍线，不是统计非劣化证明。座次分开列，不将双座次当独立场景。
- 只有开发集选出优于父模型的候选，才做**一次小复核**：
  同一 A bank 切片 `[50:100]`，50 scenarios × 4 对手 × 2 座位 × 2 模型（候选/父模型）= **800 局**。
  超出本轮时间预算则暂停，不悄悄扩容。
- 复核看 heuristic/rush 均不退步、平均至少 +3pp，且 GA/minimax 各不下降超过 5pp；
  达到则保留为“值得继续试用的候选”，否则保留父模型。
- A bank 已做过历史基线评测，部分局面也可能用于开发观察；
  此处只是与本轮开发切片不重叠的复核，**不是全新盲测或 sealed test**。

上述点估计只辅助低成本决策。一个 seed、少量局面不能证明训练方法总体更强。
不因该结果自动替换官方模型；确需替换时，再把唯一候选与原 `ppo-best` 做同局比较。
validation-B、sealed、reserve 继续不读、不生成、不消费。

## 4. 没提升时，只改一个最有依据的旋钮

看最近 10–20 updates 的趋势，不追逐单次抖动；现有日志能回答的，不增加新指标系统。

| 观察到的现象 | PPO 含义 | 下一次只试一项 |
|---|---|---|
| KL 常触顶、early_stopped 多、实际 optimizer_steps 少 | 策略每次走得太远，名义 epochs 没跑完 | actor/shared lr 从 1e-4 降到 5e-5 |
| KL/clip 很低，动作分布和目标对手表现长期不变 | 可能更新不足，也可能 advantage 没信号 | 先查 return/advantage 是否非退化；信号正常才小幅提高 lr |
| 回报有变化，但 value 范围/误差长期异常 | critic 或奖励尺度影响优势估计 | 先查终局与 GAE 接线；无 bug 再仅调 critic lr |
| 熵明显塌缩且弱项继续恶化 | 可能过早集中到旧动作 | 仅提高 entropy coefficient；不盲目追求高熵 |
| 训练 reward 上升而对局更差 | 奖励代理与获胜可能不一致 | 再考虑去掉 Δscore、只留终局效用的单因素试验 |
| 曲线正常且开发集持续上升，但 100 updates 不够 | 可能是预算不足 | 将同一路线延长至 200 updates；算第二轮，不无限续训 |

`explained_variance` 当前在同一训练 batch 拟合后计算，不是泛化指标或胜率；
删除“必须 EV≥0.5”的门槛。熵也受合法动作数影响，不机械套统一阈值。
最多 **两轮**小实验（包括延长预算）；都无清晰收益就停，给出结论与一个后续建议。
不自动扩成多 seed、网格搜索、DAgger、MCTS、更大网络或重新训练 BC。

## 5. 只保留必要产物与护栏

每轮新建输出目录，保留：
`config.json`（代码版本、父模型 hash、seed、scenario IDs、超参）、
`metrics.csv`（reward/KL/clip/entropy/value/实际更新数/耗时）、
`eval.jsonl`（逐局 opponent、scenario、seat、WDL/失败）、
`best.pth`、`last.pth` 和一页结论。不要每个 update 一份大 checkpoint 或全轨迹日志。

加载错误、非法动作、非有限数、训练/评测 scenario 重叠时停止；
硬时间预算到达就收尾，进程异常不自动重试。只监控本轮进程与输出，无变化无需反复汇报。
文档改动只检查差异和链接；代码改动跑相关 smoke/回归，只有改特征或掩码才要求 `make parity`。

旧 root、banks、训练结果和协议代码保留，不为“轻量化”删除历史证据。
未提交的 joint-proposal 扩展已撤出活动代码，恢复副本在
`runs/task1-plan-refactor-20260919/abandoned-heavy-proposal.patch`（本地忽略目录）。
[Task-2](task-2-3p4p-training.md)保持后续任务，不以旧 T1.7 科学门为前提，也不自动启动。

## 6. 下一步与完成定义

下一步只有 **L1：实现最小 PPO warm-start + 定向 smoke**，不是补统计平台。
这份计划完成不代表模型已提升；后续得到可复核的新候选，或两轮内明确“未发现收益”即可结束本轮。
仍需正式科研结论时，由用户另行启用历史协议，不能将开发结果追认成确认性实验。

## Material Passport

来源：academic-research-suite / experiment-agent，plan 模式，2026-09-19；
版本 `lightweight-ppo-v1`；效果状态 **UNVERIFIED（未训练）**。
按用户轻量化要求仅保留目标、预算、对照和停止条件，不启用完整研究流程。
