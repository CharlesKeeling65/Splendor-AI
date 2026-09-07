# 策略模仿路线实施记录

对应计划：[policy-imitation-selfplay.md](../plan/policy-imitation-selfplay.md)。本记录只
记录已经落地的代码和可复核的冒烟结果；`runs/` 下的 manifest、轨迹和权重不入 Git。

## 当前状态（2026-09-07）

首轮范围按计划收敛到 3.0–3.2：

- 3.0 已完成：运行协议、三件套 seed 隔离、双座次和 seed 分组校验、历史测试 seed 禁用、
  代码/硬件快照、预算与阈值、显式人工 approval gate。
- 3.1 已完成代码实现：GA、minimax、旧 PPO、corrected DQN、heuristic 等候选可通过同一
  逐局评测器运行；记录 W/D/L、calScore、失败局、非法动作、教师查询、搜索 successor
  次数和延迟；提供隐藏 deck 顺序与对手预留牌身份的投影审计。
- 3.2 已完成代码实现：完整 focal-agent 轨迹、265/312 版本化数据集、整局 seed 切分、
  掩码 BC、训练集专属归一化、可加载 checkpoint 和固定对手评测适配器。

尚未完成正式实验：候选教师尚未按批准的统一分母正式复测，BC 尚未用正式 manifest 运行，
也没有选择首个教师。DAgger、PPO 对手池自博弈和 MCTS 按计划暂不启动。

## 可复核入口

```bash
# 创建提案；默认不会批准或启动实验
policy-imitation create-manifest \
  --output runs/policy-imitation/<id>/manifest.json \
  --experiment-id <id> --purpose formal --device cuda

# 人工检查 budget、seed 分组、阈值和快照后再批准
policy-imitation approve runs/policy-imitation/<id>/manifest.json \
  --reviewer '<name>' --note '<review decision>'

# 批准后才可运行
policy-imitation teacher-eval <manifest> --output <result.json> \
  --corrected-dqn-checkpoint <corrected-checkpoint>
policy-imitation information-audit <manifest> --output <audit.json> \
  --corrected-dqn-checkpoint <corrected-checkpoint>
policy-imitation collect-bc <manifest> --teacher <name> \
  --feature-version v1 --output <dataset.npz> \
  --corrected-dqn-checkpoint <corrected-checkpoint>
policy-imitation train-bc <manifest> --dataset <dataset.npz> \
  --output <bc-run> --feature-version v1
```

## 阶段复盘与后续调整

1. 3.0 的协议目标已满足，但当前工作区仍有用户提供的计划文件改动；manifest 会如实记录
   dirty paths，不把它们冒充为代码提交。正式运行前应复核 manifest 中的资源和成功阈值。
2. 3.1 的小样本适配检查通过；minimax 在 2 个状态的 deck 扰动冒烟中出现潜在信息依赖，
   只能作为待验证风险，不能据此直接淘汰或选它为教师。正式审计必须扩大状态数并保存证据。
3. 3.2 的 BC 规格和训练/验证/最终测试隔离已通过自动测试，但没有性能门槛的批准值和正式
   教师快照前，不能选模、宣布提升或进入 DAgger。

下一步保持为：人工批准首轮 manifest → 统一教师评测与信息审计 → 按门槛选择单教师 →
265/312 BC 小规模单因素运行。只有 BC 数据与评测门槛通过，才新建 manifest 开启 DAgger；
无教师查询的 DAgger 验收后才考虑 PPO 对手池。
