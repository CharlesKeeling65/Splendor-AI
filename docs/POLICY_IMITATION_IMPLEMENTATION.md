# 策略模仿路线实施记录

对应计划：[policy-imitation-selfplay.md](../plan/policy-imitation-selfplay.md)。本记录只
记录已经落地的代码和可复核的运行结果；`runs/` 下的 manifest、轨迹和权重不入 Git。

## 当前状态（2026-09-07）

3.0–3.4 已完成并通过离线验收；3.5 已完成门槛实验，但按预先声明的规则判定 blocked。
当前主线 checkpoint 是 DAgger-2 初始化、PPO update-2 选出的权重；未替换旧 DQN/PPO/GA/
minimax 权重，也未启动网页部署。

运行环境为 Python 3.13.2、PyTorch 2.11.0+cu126；本机 CUDA 不可用，所有正式运行均如实
记录为 requested `cuda`、resolved `cpu`。

## 阶段结果

### 3.0 协议冻结

已完成：manifest 校验、双座次与整局 seed 分组、三件套 RNG 隔离、历史测试 seed 禁用、
代码/硬件快照、预算/阈值和显式 approval gate。每个正式阶段均使用独立 manifest；用户提供的
`plan/README.md` 与 `plan/policy-imitation-selfplay.md` 改动保持未暂存。

### 3.1 教师统一评测与信息审计

正式产物：
`runs/policy-imitation/formal-3.1-3.2-20260907/teacher-eval.json`、
`information-audit.json`、`teacher-selection.json`。

以下均为 10 个牌局 seed、双座次，即每格 20 局；数字为胜局数：

| 候选 | random | heuristic | minimax | 宏平均 |
|---|---:|---:|---:|---:|
| corrected-DQN | 20/20 | 20/20 | 9/20 | 58.33% |
| GA | 20/20 | 9/20 | 13/20 | **70.00%** |
| heuristic | 20/20 | 10/20 | 11/20 | 68.33% |
| minimax | 20/20 | 8/20 | 10/20 | 63.33% |
| old PPO | 2/20 | 0/20 | 0/20 | 3.33% |

所有候选均为 0 非法/失败局。GA、corrected-DQN、heuristic、PPO 的信息审计稳定率为
1.0；minimax 有 1/22 个投影保持探针改变动作（0.9545），仅记录为潜在风险。因此首个
单教师按预注册的宏观 W/D/L 规则选择 GA，没有混合教师标签。

### 3.2 行为克隆

先完成 265 维 v1 与 public-v2 的单因素数据/训练对照；早期小数据 BC 对 heuristic 和
minimax 的最终池表现不足，未直接进入 DAgger。随后在独立 remediation manifest 中扩大到
30 个训练 seed、30 epoch，并按验证 masked loss 选择 public-v2 student：

- v1：最佳验证 masked loss 1.7753，最终测试 random 40/40、heuristic 3/40、minimax 1/40。
- public-v2：最佳验证 masked loss 1.7680，最终测试 random 38/40、heuristic 0/40、minimax 1/40。

两者训练/验证/最终测试按整局 seed 隔离，掩码后预测合法率均为 100%，但对局质量门槛只
达到 random 基线，因此继续 DAgger。正式产物位于
`runs/policy-imitation/remediation-3.2-20260907/`。

### 3.3 DAgger

正式产物位于 `runs/policy-imitation/formal-3.3-20260907/`。学生执行自身动作，教师只在
学生访问的状态查询标签；后续 `bc-eval` 完全不查询教师。

| student | 验证最佳 masked loss | random | heuristic | minimax | 教师查询 | 失败/非法 |
|---|---:|---:|---:|---:|---:|---:|
| BC remediation | 1.7680 | 38/40 | 0/40 | 1/40 | 0 | 0/0 |
| DAgger-1 | 1.5297 | 40/40 | 10/40 | 15/40 | 0（验收） | 0/0 |
| DAgger-2 | **1.4874** | 40/40 | 10/40 | 13/40 | 0（验收） | 0/0 |

两轮收集共 120 个 DAgger 游戏、4,268 次教师查询、0 失败；第二轮验证损失继续下降，
但最终池没有超过 DAgger-1，因此停止扩大聚合。DAgger 的无教师验收和成本统计均可独立
重放，满足进入 PPO 的退出条件。

### 3.4 PPO 对手池自博弈

正式 manifest：`runs/policy-imitation/formal-3.4-20260907/manifest.json`；以 DAgger-2
最佳验证 checkpoint 初始化，GA/heuristic/current 各 1/3 权重，每局固定一个对手，4 次
更新、每次 4 局，终局值为 focal win/loss/draw 的 `+10/-10/0`。

- 训练 rollout：16 局，0 失败，0 非法，0 教师查询。
- 验证按整数胜局选择 update-2：random 20/20、heuristic 9/20、minimax 9/20，共
  38/60；update-3/4 出现遗忘，未选用。
- update-2 最终测试：random 40/40、heuristic 14/40、minimax 21/40，0 失败/非法/教师
  查询。相同最终池下 DAgger-2 为 40/40、10/40、13/40。

这是小预算的有效性运行，不代表 PPO 已收敛；若要扩大更新数，必须重新建立并审核 manifest。

### 3.5 MCTS 门槛实验

正式产物：`runs/policy-imitation/formal-3.5-20260907/mcts-gate.json`。同一
`runs/staged-10k-parallel-20260907/search-42/best.pth` 比较 0/1/4 次 sampled-hidden-state
PUCT，10 个 seed、三类固定对手、每个预算 60 局：

| 搜索预算 | 胜局 | 胜率 | 平均决策耗时 | 相对无搜索 |
|---:|---:|---:|---:|---:|
| 0 | 20/60 | 33.33% | 0.00944 s | 1.0× |
| 1 | 16/60 | 26.67% | 0.04372 s | 4.63× |
| 4 | 18/60 | 30.00% | 0.12218 s | 12.95× |

价值头在 797 个决策状态上对 signed calScore 的方向准确率为 0.610，MSE 1.153，Pearson
相关性 0.107；隐藏 deck/对手预留牌审计 22/22 稳定。由于搜索相对无搜索没有增加胜局，
门禁判定 blocked；不进行搜索访问蒸馏、不把 MCTS checkpoint 纳入主线、不部署网页。

## 可复核入口

所有正式命令都要求 manifest 已 approved：

```bash
policy-imitation dagger-round <manifest> --teacher ga \
  --student-checkpoint <bc-or-dagger-checkpoint> --round <n> --output <round.npz>
policy-imitation dagger-aggregate <manifest> --inputs <base.npz> <round.npz> \
  --round <n> --output <aggregate.npz>
policy-imitation train-bc <manifest> --dataset <aggregate.npz> \
  --output <bc-run> --feature-version public-v2 --epochs 30
policy-imitation bc-eval <manifest> --checkpoint <bc-checkpoint> \
  --seed-group final_test --output <result.json>
policy-imitation ppo-selfplay <manifest> --initial-bc <bc-checkpoint> \
  --opponent-pool ga:1,heuristic:1,current:1 --output <ppo-run>
policy-imitation ppo-eval <manifest> --checkpoint <ppo-checkpoint> \
  --seed-group final_test --output <result.json>
policy-imitation mcts-gate <manifest> --checkpoint <search-checkpoint> \
  --simulations 0 1 4 --output <gate.json>
```

## 阶段复盘与后续调整

1. 协议、掩码、数据切分和无教师评测路径均已通过；CPU 回退和用户 dirty plan 文件都在
   manifest 中如实记录。
2. GA 是本次统一候选中最稳的单教师，但它不是最优策略证明；minimax 的潜在信息风险仍
   需在未来单独修复/审计，不能作为教师默认替代。
3. BC 首轮质量不足，扩大数据和训练预算后仍需 DAgger；DAgger-1 带来主要对局提升，
   DAgger-2 只带来验证损失改善而没有最终池继续提升，故停止聚合。
4. PPO update-2 在 minimax/heuristic 最终池优于 DAgger-2，但训练预算小且后续更新有遗忘；
   当前 checkpoint 可作为本地离线研究快照，不应解释为已收敛或已完成网页部署。
5. MCTS 的信息语义审计通过，但收益门失败且成本显著增加；后续只能在新 manifest 中重做
   价值校准/搜索方案，不能从本次结果直接进入搜索蒸馏。

以上结论均保留逐局记录、seed、座次、查询数、搜索节点、延迟和失败局分母；旧权重与浏览器
权限保持不变。
