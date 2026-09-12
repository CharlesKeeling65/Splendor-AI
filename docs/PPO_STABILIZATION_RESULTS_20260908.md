# PPO 稳定化训练与独立测试结果（2026-09-08）

## 结论

本轮修正后的 **fixed PPO 相比原 PPO 有描述性改进，但没有全面超过 corrected DQN**。
保留 DQN 作为现有性能基线，继续研究「模仿初始化 → 稳定化 PPO」路线；
不将新增的初始策略 KL 约束设为默认，不继续按本轮小预算从随机初始化训练，
也不直接进入 MCTS 或网页部署。

正式运行完成于北京时间 2026-09-08 13:30:53：
9 次训练、1152 局训练、2700 局验证、3000 局独立测试，零失败局、零已记录非法动作。
正式运行总墙钟时间 27 分 31 秒（13:03:22–13:30:53），不包含此前冒烟、
复现性诊断及被中止批次。这里的“完成”不等于“达到部署门槛”。

## 1. 同场测试结果

胜率定义为胜局 / 全部安排局数；平局不计胜。每个模型每类对手使用相同的
25 个新牌局、双座次，共 50 局。三模型行是三次 PPO 阶段重复或三份历史 DQN
的汇总，每类对手 150 局；**不是把 150 局当作独立牌局或独立完整训练重复**。

| 模型组 | GA | heuristic | minimax | random |
|---|---:|---:|---:|---:|
| corrected DQN（3 模型） | 81/150（54.0%） | 59/150（39.3%） | 75/150（50.0%） | 150/150（100.0%） |
| 稳定化 PPO / fixed（3 模型） | 81/150（54.0%） | 48/150（32.0%） | 73/150（48.7%） | 150/150（100.0%） |
| 策略 KL 约束 / anchor（3 模型） | 57/150（38.0%） | 40/150（26.7%） | 71/150（47.3%） | 150/150（100.0%） |
| 随机初始化 / scratch（3 模型） | 0/150（0.0%） | 0/150（0.0%） | 0/150（0.0%） | 40/150（26.7%） |
| 原 DAgger-2（1 模型） | 20/50（40.0%） | 8/50（16.0%） | 25/50（50.0%） | 50/50（100.0%） |
| 原 PPO（1 模型） | 18/50（36.0%） | 10/50（20.0%） | 22/50（44.0%） | 50/50（100.0%） |
| GA（1 模型） | 25/50（50.0%） | 17/50（34.0%） | 29/50（58.0%） | 50/50（100.0%） |

三模型组每个模型权重相同。fixed 对 GA 另有 2 局平局，DQN 对 GA 和 heuristic
各有 1 局平局，其余表中模型/对手组合均无平局。逐模型 W/D/L 与完整分母保留在
[独立复算结果](../runs/policy-imitation/stabilization-formal-v2-20260908/independent-audit.json)。

主要差异：

- fixed 相比原 PPO：GA +18.0、heuristic +12.0、minimax +4.7 个百分点。
- fixed 相比 corrected DQN：GA 持平、heuristic -7.3、minimax -1.3 个百分点。
  因此不能宣称新策略网络已经胜过 Q 学习路线。
- fixed 相比 DAgger-2：GA +14.0、heuristic +16.0 个百分点，但 minimax -1.3。
  优化并不是对所有对手同时提升。
- anchor 的验证分数更高，独立测试却不如 fixed：GA -16.0、heuristic -5.3、
  minimax -1.3 个百分点。当前没有将该约束默认启用的证据。
- scratch 三次都选中 update 0；表中测试的是选模流程保留的初始网络。
  所有验证检查点对三类非随机对手均为零胜，因此本轮没有获得通过验证的
  从头训练改进。**不能把该表误写成“所有最终更新网络也被测试且都是零胜”。**

fixed 的逐模型测试胜率范围：GA 46%–58%，heuristic 28%–38%，minimax 44%–56%。
只有三个 PPO seed、共享同一 BC 来源、25 个测试牌局，不报告虚假的独立样本
置信区间，也不将上述百分点差异表述为统计显著或普遍的算法优劣。

## 2. 实际训练方案与对手

预声明见 [训练计划](../plan/ppo-stabilization-20260908.md)，最终配置及来源见
[manifest](../runs/policy-imitation/stabilization-formal-v2-20260908/manifest.json)。

- fixed：DAgger-2 trunk/policy 初始化，修正后的 PPO，无初始策略 KL 惩罚。
- anchor：相同初始化，增加对冻结 BC 策略的 KL 惩罚，系数 0.02。
- scratch：随机 trunk/policy，但仍共享 BC 拟合的观测归一化，不是零示范信息。
- 每组 seeds=42/1234/2024，每次 8 更新 × 16 完整对局；lr=1e-4、
  epochs=4、minibatch=256、gamma=.99、GAE lambda=.95、clip=.2、
  entropy=.005、value coefficient=.5、grad norm=1、target KL=.02。
- 首批仅 critic head 预热 2 epochs。回报保持 Δscore + 终局 ±10。
- 训练对手池：GA、heuristic、current、history 各占总权重 1；
  正常各为 25%。history 最多四个冻结快照，桶内均分。
  空历史桶质量转给 current，第一轮为 GA 25%、heuristic 25%、current 50%。
- current 是更新前的冻结模型，history 是已有更新快照。
  当前这些快照对手采用合法掩码后的 **argmax**；学习方采用策略分布采样。
  这是对固定基线和贪心快照的混合训练，不是双方完全对称的随机策略自博弈。
- 验证只对 GA/heuristic/minimax，在 update 0/2/4/6/8 各评测 60 局，
  按整数总胜局选模，同分保留较早检查点，random 不参与选模。
- 训练 seed 820000–821999 中按任务分配并实际消费 1152 个不同 seed；
  验证 822000–822009；最终测试 823000–823024。
  所有训练完成并固定检查点后，才开始最终测试。
- 各分支训练牌局不同；这是有界随机对照，不是严格逐训练牌局配对的
  KL 消融试验，不能把分支差异全部归因于 KL 系数。

| 训练任务 | 选中 update | 最佳验证胜局 | 最后更新价值解释方差 | PPO transition 数 |
|---|---:|---:|---:|---:|
| anchor-seed1234 | 2 | 30/60 | 0.076 | 3887 |
| anchor-seed2024 | 2 | 31/60 | 0.030 | 3838 |
| anchor-seed42 | 2 | 29/60 | 0.085 | 3895 |
| fixed-seed1234 | 4 | 28/60 | 0.170 | 3889 |
| fixed-seed2024 | 6 | 28/60 | 0.212 | 3886 |
| fixed-seed42 | 4 | 29/60 | 0.086 | 3862 |
| scratch-seed1234 | 0 | 0/60 | 0.084 | 4812 |
| scratch-seed2024 | 0 | 0/60 | 0.028 | 5197 |
| scratch-seed42 | 0 | 0/60 | 0.051 | 5238 |

六个 BC 初始化任务的 update 0 验证逐局 seed/seat、分数与胜负完全相同，
均为 GA 7/20、heuristic 6/20、minimax 9/20，总计 22/60。

训练胜率不能替代实力指标：fixed 每次训练仅胜 8–10/128，anchor 11–14/128，
scratch 却胜 41–61/128。不同强度的快照对手以及采样/贪心不对称，
使“训练胜局更多”不等于“对固定基线更强”。

## 3. 本轮完成的实现与检查

1. critic 默认改为无界回报输出；旧 checkpoint 缺少 value_mode 时保留旧 tanh
   语义，不修改历史权重。
2. 对手类别总权重固定，历史池有界；允许明确的零权重配置，保存实际分配。
3. 增加 critic-head 预热、PPO KL 早停、clip fraction、价值解释方差、
   回报/预测范围、参考策略 KL、更新步数及有限数检查。
4. update 0 参与选模；修复 current 快照记录为更新前文件；
   逐更新保存 status/result，失败轨迹不用于更新。
5. 信息审计同步 state 与 rule.current_game_state，加“故意读取隐藏牌”的阳性对照；
   评测输入按同一对象图防御性复制，避免基线修改原对局。
6. 新增三分支、多 seed、严格 CUDA、训练后测试屏障、原始记录及选模审计的运行器。
7. 固定所有 spawn 工作者的 PYTHONHASHSEED=0，不改引擎和存量基线逻辑。

验收证据：

- 全量离线测试：168 passed、1 skipped。跳过项是沙箱内不可用的 CUDA 监控；
  随后在实际 GPU 环境补跑该测试文件，3 passed、无跳过。
- policy_imitation 相关 Ruff 通过，mypy 17 个源码文件通过；git diff --check 通过。
- 两轮小规模 CUDA 冒烟分别完成训练和评测；正式重跑另有哈希种子回归测试。
- 72 次正式 PPO 更新指标均有限；对手池总质量始终 4，历史长度不超过 4。
  实际训练未触发 target-KL 早停；不能声称该保护机制已经带来实测收益。
- 从原始记录独立复算 1152/2700/3000 局分母、测试 W/D/L、全部测试 seed/seat，
  并核对验证胜局选模。运行器另核对实际 checkpoint 的 update。
- 初始/历史 checkpoint 哈希及 57 个训练相关源码哈希未变；
  引擎、generic agent、GA .npy 权重等另 28 个文件的补充快照也未变。
  补充快照在训练中、最终测试前采集，不冒充启动前快照。
- 严格 CUDA 运行：Quadro P5000 16GB、Python 3.13.2、PyTorch 2.11.0+cu126、
  CUDA 12.6，3 workers，Torch/OMP 单线程；无 CPU 静默回退。
  监控抽样见显存约 879 MiB、温度 52°C，这些不是峰值统计。

### 被中止的诊断批次

[原批次](../runs/policy-imitation/stabilization-formal-20260908/INTERRUPTED.md)
在最终测试前被主动中止，所有文件保留。相同 BC actor 的初始验证出现
20/60 与 23/60 差异；引擎 set 动作枚举受 Python 进程哈希影响。
固定启动哈希后，六次初始化验证逐局一致。

该批次已落盘至少 240 局训练、540 局验证，另可能存在中止时未落盘对局；
这些成本及冒烟成本**不包含**在正式 1152/2700/3000 统计和 27 分 31 秒中。
它不参与效果比较，不作为仍运行的任务；旧 status 文件只是最后进度快照。

### 信息口径限制

修正探针后，GA 在 824000/824001 诊断牌局出现 6/77 次可比隐藏信息扰动导致
动作改变，heuristic 为 0/77。这是有限经验性风险证据，不是完整信息安全证明。
GA 的一步 successor 评分可能受隐藏牌堆补牌影响；本轮未继续生成 GA 教师标签。
旧 DAgger 共享初始化仍有原教师信息审计不足的来源限制。

GA/minimax 保持原样，作为压力测试对手；对它们的胜率不代表公平公共信息博弈
最优性。DQN 与 PPO 的历史数据、初始化和训练成本不同，此实验不是等算力、
等数据的算法因果比较。历史报告不同 seed/哈希环境的百分比不能直接拿来计算提升。

## 4. 下一轮调整顺序

以下是本轮证据支持的待验证方向，**尚未执行新的预算，也未启用网页部署**。

1. **保留 corrected DQN 基线；PPO 沿 fixed 继续研究，reference KL 暂保持 0。**
   本轮不以最幸运的单个 PPO seed 替换默认权重。保持当前测试集封存。
2. **先提高 critic 与采样轨迹质量，再扩大 PPO 规模。**
   最后更新的解释方差约 .03–.21，仍偏弱。单独验证更多 critic 预热、
   actor/critic 不同学习率或减少共享 trunk 的梯度干扰；同时比较训练时
   学习方与快照对手的采样温度，避免直接把贪心评测能力等同于采样能力。
   每次只改一个因素；这些都是假设，不保证提升。
3. **下次做真正配对的消融与更稳的选模。**
   同一 PPO seed 在 fixed/anchor 使用相同训练牌局和外生对手抽样计划；
   扩大验证牌局、保留独立新测试 seed，报告逐 seed 及逐对手差异。
   anchor 的验证/测试排名反转提示当前验证规模有限，不能继续围绕 823000
   测试集反复选参数。
4. **再考虑公共信息一致的 DAgger 补数。**
   针对 heuristic 短板采集 learner 状态，但先修复/验证教师信息口径与标签质量；
   不直接扩大目前风险未解除的 GA 教师数据。
5. **MCTS 最后。**
   先建立公共信息一致的搜索与可信的价值评估。本轮无界 return critic 不是
   胜率预测器，不能未经标定直接当作 AlphaZero 的终局价值。
   后续搜索实验应使用同一新策略的有/无搜索对照，独立报告搜索成本。

## 5. 产物与复核入口

- [完整结果及逐局记录](../runs/policy-imitation/stabilization-formal-v2-20260908/results.json)
- [独立复算审计](../runs/policy-imitation/stabilization-formal-v2-20260908/independent-audit.json)
- [补充源码/GA 权重审计](../runs/policy-imitation/stabilization-formal-v2-20260908/supplemental-source-audit.json)
- [训练配置与运行来源](../runs/policy-imitation/stabilization-formal-v2-20260908/manifest.json)
- [最终进度](../runs/policy-imitation/stabilization-formal-v2-20260908/suite-status.json)
- [运行器](../src/splendor/agents/our_agents/policy_imitation/stabilization.py)
- 模型位于该运行目录 training/<分支>-seed<种子>/best.pth，
  同目录保留 initial.pth、各 update、final.pth、status.json 和 result.json。

运行器入口为 `python -m splendor.agents.our_agents.policy_imitation.stabilization --help`；
正式参数与全部 baseline checkpoint 来源已固定记录于 manifest。
输出目录禁止覆盖。复现原实验需另建目录；新调参则需另分配未用于选模的新测试种子。

Luna Max 子智能体参与实现与测试草稿，达到使用额度后由主代理完成剩余修正、
真实 CUDA 训练、监控与复核。validate-data 检查促使报告区分验证/测试、
胜率分母、配对牌局、失败记录和诊断额外成本。本轮未提交 Git，也未覆盖默认模型。

