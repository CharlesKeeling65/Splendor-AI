# 任务计划（1）：2p 模型提升与科学种子协议

> 复核版本：2026-09-15。
> 状态：待实施计划。本文定义任务、顺序、验收和产物，不表示训练或代码改动已经完成。
> 适用模型：当前 C4-R2 `ppo-best` 及其后续 2p PPO 候选。
> 强制顺序：本计划全部完成并通过 T1.7 退出门槛后，才允许启动
> [任务计划（2）：3p/4p 多人模型训练](task-2-3p4p-training.md)。

本文中的“必须”是正式实验的硬门；“建议”可因预算调整，但调整必须发生在观察 sealed-test 结果之前，
并写入新 manifest。历史结果只用于提出假设，不再承担选模职责。

## 0. 本次复核结论：需要补齐的承重环节

原计划方向正确，但若直接实施，仍有七处会削弱因果归因或复现性。本版将它们提升为显式任务和测试门：

| 缺口 | 当前仓库证据 | 本版修订 |
|---|---|---|
| 同名 seed 在不同 runner 中并非同一副牌 | `Game(seed)` 在创建规则前先消耗 1000 个 Python random 数；`evaluation.py` / `ppo_selfplay.py` 直接创建规则 | 引入可序列化 `ScenarioV1`，正式评测从快照装载；增加 `Game`、raw-rule、PPO 三路径初态一致性测试 |
| 发牌、策略采样和对手随机性仍耦合 | `isolated_seed(seed)` 同时重置 Python/NumPy/Torch；PPO `Categorical.sample()` 使用全局 Torch RNG | 引入带命名空间的局部 RNG bundle；发牌、pool、policy action、opponent action、minibatch、worker 分流 |
| 现有训练分支没有真正配对 | `make_training_jobs()` 给每个 variant × model seed 分配不同 training deals | 以 `replicate_id` 为区组；对照分支共享训练 scenario schedule 和事件键随机数，replicate 之间独立 |
| 方法效应与部署模型效应混用 | 单个历史 `ppo-best` 不能代表重训控制的 model-seed 分布 | 分开 `d_method(T_r-O_r)` 与 `d_deploy(official-ppo-best)`，sealed test 同时纳入配对 `O` replicates |
| C4-R2 名义样本量被放大 | 每个 matchup 150 行只来自 50 个 seed，且存在重复 `(seed, seat)`；旧 Wilson 区间按 150 个独立伯努利局计算 | 历史区间仅作描述；新评测以 scenario 为聚类单位，双座次先在 scenario 内聚合 |
| 当前 potential shaping 的“不改变策略”结论缺少 episodic 终局条件 | `PotentialRewardShaper.advance()` 对终局使用非零 `Phi(s_T)` | 增加终局势能归零审计；未修正的版本改标为非策略不变实验项，不能以 PBRS 定理背书 |
| sealed test 与 DAgger 边界不够严 | 稳定化 driver 会对所有训练候选自动跑 final test；“rush 教师”没有专家质量门 | final test 改为选模后一次性原子批次；DAgger 只允许合格教师在学生访问状态上标注 |

此外，`manifest.py`、`seed_registry.py` 和 `stabilization.py` 各自维护了部分 seed 规则，口径已有漂移；
本任务必须把注册表、派生协议、split 校验和消费记录统一到一个版本化事实源。

## 1. 任务边界、基线与假设

### 1.1 范围

本计划同时解决四个问题：

1. 在当前 `ppo-best` 基础上针对 heuristic/rush 短板提升 2p 策略；
2. 将训练副本、初始牌局、座位、对手池、动作采样、minibatch 和 worker 随机性分开；
3. 建立可跨 runner 重放的 scenario bank 和分层、配对评测；
4. 将“单个幸运 checkpoint”与“训练方法在不同 model seed 上的稳定收益”分开报告。

本计划只涉及本地引擎、PPO/DAgger 训练和离线评测，不启动真实网页对局，不把 2p checkpoint 改造成
3p/4p 模型，也不改变 legacy v1、PPO/GA/minimax/DQN 默认入口的行为。新增 scenario/RNG 路径必须是
显式 opt-in；已有入口的兼容性由回归测试锁定。

### 1.2 冻结基线

依据 [C4-R2 联赛报告](../runs/c4r2-league-20260914/league_report.md) 和
[C2-R2 报告](../docs/C2R2_2000_REPORT_20260914.md)：

| 项目 | 当前事实 | 解释边界 |
|---|---|---|
| 官方候选 | `runs/c2r2-selfplay-2000/training/fixed-seed1234/best.pth` | 只是在 C4-R2 2p 协议下排名第一 |
| SHA-256 | `e225464c17a783bd91b51251336f917e9f867af4f475d8414372885fbb758102` | T1.0 必须现场复核，不只抄文档 |
| 模型契约 | `public-v2`、312 输入、3510 动作、4×128 MLP、greedy masked argmax | 训练时策略仍为随机采样 |
| vs minimax / GA | score rate 66.3% / 62.0% | 达到既有 G1 点估计门槛 |
| vs heuristic / rush | score rate 58.0% / 56.0% | heuristic 距 60% 门槛 2pt；rush 是最弱方向 |
| vs random | 100.0% | 新模型仍需非劣化保护 |
| C4-R2 样本结构 | PPO 共 900 个 scheduled games；每个对手 150 行、仅 50 个 seed | 旧 Wilson 区间不能当作 150 个独立局的确认性证据 |

本文统一使用 **score rate**：胜 = 1、引擎平局 = 0.5、负 = 0。旧报告中把它简称“胜率”的位置，
在本任务产物中必须同时给出 W/D/L 和 score rate，避免与 `wins / scheduled_games` 混淆。

### 1.3 待验证假设，而非既定因果

- **H-pool**：训练对手分布对 heuristic/rush 行为域覆盖不足；
- **H-label**：学生对阵 heuristic/rush 时访问的状态缺少高质量反制标签；
- **H-objective**：终局效用、势能终止语义或中间事件信号与真实胜负目标仍有偏差；
- **H-budget**：目标风格已被覆盖，但完整预算仍不足。

行为指标只能帮助定位机制，不能单凭“买卡更多/贵族更多”证明上述因果。各假设必须通过预注册的
单因素或 2×2 实验验证。

## 2. 术语、实验单位与随机性契约

### 2.1 术语

| 名称 | 定义 |
|---|---|
| `treatment` | 一组完整训练方法配置，如 pool 权重或新 DAgger 数据；不是单个 checkpoint |
| `replicate_id` | 一个独立训练副本区组；决定 model/minibatch/action 等训练随机流 |
| `model_seed` | replicate 内模型初始化与优化随机流的可读别名；不能兼任 deal seed |
| `scenario` | 完整、不可变的 2p 初始局面；含贵族、三层牌堆顺序、12 张明牌和初始资源 |
| `source_seed` | 仅用于生成 scenario 的注册表整数；scenario 冻结后以内容 hash 为身份 |
| `game` | 一个固定 scenario、目标座位、候选、对手和动作 RNG 配置的实际对局 |
| `cluster` | 正式 2p 评测的最小独立环境单位：scenario；双座次与同 scenario 下的 A/B 都嵌套其中 |

“共享 scenario”是 Common Random Numbers（CRN）配对设计，不是 train/test 泄漏；泄漏指同一
scenario identity 同时进入训练/DAgger 与 validation/test，或依据 test 结果修改方法。

### 2.2 RNG Protocol v1

禁止按调用顺序 `spawn()` 或用 `seed + 常数` 暗含语义。每个随机事件使用稳定键派生：

```text
digest = SHA256("splendor-rng-v1\0" || canonical_json(rng_key))
seed63 = int.from_bytes(digest[0:8], "big") & ((1 << 63) - 1)
u53    = ((int.from_bytes(digest[8:16], "big") >> 11) + 0.5) / 2^53
```

`rng_key` 至少包含 `experiment_id / protocol_version / phase / coupling_group / replicate_id /
scenario_id / seat / opponent_id / update / focal_step / stream_name` 中适用的字段；`treatment_id` 必须写入
lineage，但 paired stream 是否共享随机数只由显式 `coupling_group` 决定，不能靠“碰巧省略 treatment”实现。
canonical JSON 的 UTF-8 编码、key 排序、数字/空值表示必须测试锁定，并保存完整 digest，而不只保存
截断整数。`seed63` 供只能接收 seed 的库使用；需要 treatment 间 CRN 的单次选择优先直接消费 `u53`，
避免不同 Python/NumPy/Torch 抽样器把同一 seed 解释成不同随机序列。

| stream | 控制内容 | 共享/独立规则 |
|---|---|---|
| `scenario_source` | 贵族采样、牌库洗牌 | split 之间独立；正式评测直接装载 snapshot |
| `model_init` | 权重初始化 | 架构/父 checkpoint 相同时按 replicate 配对；父 checkpoint 不同时记录该不可配对边界 |
| `policy_action` | PPO 训练动作采样 | 以 game + focal-step 的 `u53` 在固定 action-index 顺序作 inverse-CDF；不能依赖全局 Torch RNG |
| `minibatch` | epoch 内样本排列 | 以 replicate + update + epoch 派生 |
| `pool_draw` | 对手类别/历史 snapshot 选择 | treatment 间复用同一 `[0,1)` 变元，再按各自 CDF 映射 |
| `opponent_action` | random 或含随机 tie-break 的对手决策 | 以 scenario + seat + opponent-turn 事件键派生 |
| `worker` | 进程启动和非语义性本地缓冲 | worker 数或完成顺序不得改变其它流 |

2p 正式评测座位直接枚举 `[0, 1]`，不再用随机 `seat_seed` 伪装确定性平衡。训练座位 schedule 同样保存
为显式数组。正式候选用 greedy masked argmax，因此没有 candidate action RNG；随机对手仍必须有
actor-local RNG。pool 与随机动作的 CDF 都使用稳定候选顺序，并把顺序/version 写入 manifest。事件键 CRN
只能降低方差；策略分歧后合法集合和状态都会改变，不保证后续动作语义完全相同，报告不得夸大这一点。

## 3. 目标、估计量与完成定义

### 3.1 量化目标

| 编号 | 目标 | 完成定义 |
|---|---|---|
| T1-G1 | 可重放 | Scenario 与 CPU greedy 评测逐动作 hash 一致；GPU 训练只承诺声明配置下的统计复现，不虚构跨硬件 bitwise 一致 |
| T1-G2 | 随机流可审计 | train/validation/test scenario 不重叠；各 RNG stream 可单独改变，且不会改变其它流的 key/hash |
| T1-G3 | 牌局泛化 | final IID bank 至少 200 个 unique scenarios，按 T1.3 power analysis 可增至 500；每个 scenario 双座次 |
| T1-G4 | 策略提升 | 部署候选达到预注册 point-estimate 门槛；方法级收益与非劣化另用配对区间判定，不混为同一结论 |
| T1-G5 | 训练稳定 | pilot 可用 3 replicates；确认性 treatment 默认 5 replicates。若只做 3 个，只能作有限证据，不宣称训练分布稳定 |
| T1-G6 | 审计完整 | manifest、bank、lineage、逐局记录、checkpoint/opponent/code hash、依赖、硬件、成本均可复核 |
| T1-G7 | 测试封存 | sealed test 在 treatment 和官方 checkpoint 选定后只执行一个预注册批次；消费记录 append-only |

若要对“训练算法总体优于基线”做显著性断言，model-seed 数量必须由 pilot 的 effect size/power 决定；
资源只够 3–5 个时，报告每个 replicate、区间和结论限制。不能把数百个 deal 当成数百个独立训练副本。

### 3.2 主估计量与决策层级

必须拆开“训练方法有效”和“选出的部署模型更强”两个问题。对 opponent `o`、replicate `r`、scenario `i`
先在双座次内求平均：

```text
y_T(r,o,i,seat) ∈ {0, 0.5, 1}
ybar_T(r,o,i)    = mean_seat y_T(r,o,i,seat)

d_method(r,o,i)  = ybar_T(r,o,i) - ybar_O(r,o,i)
d_deploy(o,i)    = ybar_official(o,i) - ybar_ppo-best(o,i)
```

这里 `O` 是与 treatment 使用相同 `replicate_id`、训练 schedule 和评测 scenario 的重训控制；冻结
`ppo-best` 是历史部署基线。`d_method` 才能估计 pool/DAgger/shaping/budget 的方法效应，`d_deploy` 只回答
最终唯一 checkpoint 是否优于当前线上候选。不能用数百个 scenario 把单个历史 checkpoint 当成多个独立
训练控制，也不能把 `d_deploy` 外推成“该训练方法平均更优”。

- **方法级确认性主终点**：`d_method` 在 heuristic、heuristic-rush 上的配对分布；2×2 还报告
  `A-O`、`B-O` 与交互 `C-A-B+O`。model replicate 为外层、scenario 为内层，两个目标对手共同重采样；
- **部署级主终点**：`d_deploy` 在同一 IID scenarios 上的 paired difference，并逐 opponent 报告；
- **部署绝对门槛**：为延续现有 G1，官方 checkpoint 的 point estimate 对 minimax ≥60%、GA ≥55%、
  heuristic ≥60%、random ≥99%；只有相应 simultaneous one-sided lower bound 也越过阈值时，才称为
  “统计支持超过阈值”，否则只写“点估计达标”；
- **双重非劣化门**：入选 treatment 相对 `O`、官方 checkpoint 相对 `ppo-best` 的 minimax/GA 差值下界
  均高于 `-delta_NI`。默认建议 `delta_NI=3pt`，但必须在 T1.4 根据 power 和实际容忍度批准；
- **rush 目标**：官方 checkpoint 点估计至少高于冻结基线，60% 为期望目标；
- **探索性结果**：seat、scenario stratum、行为域、延迟和 critic 指标，不与确认性终点混报。

heuristic 与 heuristic-rush 的 superiority family、minimax 与 GA 的 non-inferiority family 分开预注册；
各 family 使用 simultaneous bootstrap CI、Holm 调整，或 Bonferroni 调整后的区间，不能事后挑一种。
报告给出两个正交结论：**部署资格**（绝对 point-estimate 门与安全门）和**方法证据**（`d_method` 的配对
区间）。前者通过而后者未决时，只能写“产生了合格候选，方法平均收益证据不足”；不得合并成笼统的
“算法已提升”，也不得把“CI 包含目标值”写成“统计上达到目标”。

## 4. 强制执行顺序

```text
T1.0 冻结基线、现状审计与 manifest v2 设计
  ↓
T1.1 RNG 分流、训练 schedule 配对与确定性测试
  ↓
T1.2 ScenarioV1、split、IID/压力 bank
  ↓
T1.3 配对评测器、统计实现与样本量设计
  ↓
T1.4 新协议基线复测、crossed pilot、预注册锁定
  ↓
T1.5 pool / DAgger / shaping / budget 实验
  ↓
T1.6 多 replicate 确认、validation 选模、一次性 sealed test
  ↓
T1.7 退出审查与任务（2）移交
```

发现 seed 泄漏、分母不可复核、schema 漂移、非法动作、终局错误或 scenario hash 不一致时立即停止。
结果目录只追加，不覆盖；修复后必须新建 manifest 和输出目录。

## 5. 阶段任务与退出门槛

### T1.0 冻结基线与把已知缺口变成失败测试

- [ ] 现场校验 `ppo-best` SHA-256、checkpoint config、normalizer、父 BC/DAgger-2 数据 hash 和推理模式。
- [ ] 冻结所有基线 opponent：代码 revision、heuristic 权重、GA 基因、minimax 配置、随机 tie-break 规则；
  无 checkpoint 的 built-in agent 也必须有 source/config hash。
- [ ] 记录 C4-R2 的 50 unique seeds、重复 `(seed, seat)` 和 scheduled/unique 分母；旧 Wilson 仅保留为历史描述。
- [ ] 为以下现状先写回归/预期失败测试：`Game(seed)` 与 raw-rule 初态不同；整局 seed 重置耦合 Torch；
  variant 的 training deals 不同；final test 自动覆盖全部候选；league error game 被排除分母。
- [ ] 设计 `manifest schema v2`，由 `seed_registry.py` 导出注册/禁区事实；删除 manifest 内平行的默认禁区。
- [ ] formal run 要求 clean worktree，或保存可重放 patch hash；记录 commit、`uv.lock`/依赖 hash、Python、
  PyTorch、NumPy、CUDA/driver、CPU/GPU、线程数、TF32/cuDNN/CUBLAS 设置。
- [ ] manifest 流程固定为 `proposed → approved → running → completed/blocked`；不得再用历史 authority 文案
  绕过本任务的人工批准门。

**退出门槛**：baseline manifest v2 可验证；上述差异均有测试覆盖；任意历史数字能追溯其真实分母和 runner。

### T1.1 RNG 分流与训练配对

- [ ] 在 `policy_imitation/protocol.py`（或等价单一模块）实现 `RngKey`、`SeedLineage`、`derive_seed()`
  和 actor-local Python/NumPy/Torch generator；禁止 formal path 依赖 Python `hash()`。
- [ ] `_policy_step()` 按固定 action-index 顺序用事件键 `u53` 对 masked categorical 作 inverse-CDF；
  pool 同样消费事件键 `u53`，minibatch 才使用独立 `torch.Generator`。不再用 deal seed 包住整局 Torch RNG。
- [ ] 对不接收 RNG 的 legacy opponent 增加只在新 runner 中启用的 per-decision RNG adapter；不修改其默认接口。
- [ ] 训练 schedule 由 `(replicate_id, update, game_index)` 生成。O/A/B/C treatment 在同一 replicate
  共享 scenario、seat、随机变元和 minibatch key；不同 replicate 的 lineage 全部独立。
- [ ] weighted pool 配置改为显式映射并保存归一化后概率、抽样 CDF、实际计数和 snapshot；禁止靠重复名字表达权重。
- [ ] worker 使用 `spawn`，`PYTHONHASHSEED=0` 在解释器启动前设置；改变 worker 数和 future 完成顺序后，
  schedule hash、scenario hash 和抽样 key 不变。
- [ ] CPU 评测要求逐动作完全一致。GPU formal training 开启 PyTorch 可用的 deterministic 设置；若某算子
  不支持，manifest 明确记录降级和允许差异，不能承诺 checkpoint hash 一致。

**退出门槛**：四类反事实测试通过——同牌换模型、同模型换牌、同牌换座位、同牌换对手；单独改变任一
stream 不改变其它 stream 的 digest，1 worker 与 N worker 生成同一语义 schedule。

### T1.2 ScenarioV1 与 bank

每个 scenario 至少包含：

```text
schema_version, scenario_id, n_seats=2, source_segment, source_seed,
rule_id, rounds_limit, card_registry_hash, action_registry_hash,
nobles_in_order, dealt[tier][slot], decks[tier], deck_top="list_end",
initial_gems, current_agent_index, canonical_state_sha256, strata_v1
```

- [ ] 用唯一 card/noble code 序列化，不保存 Python 对象 repr；明确 deck 顶端方向和 noble/明牌顺序。
- [ ] canonical state 采用 UTF-8、排序 key、固定数字编码后 SHA-256；装载后重新编码必须得到同一 hash。
- [ ] 新增可选 scenario factory/injection，不改变 `SplendorGameRule(n)` 默认构造；装载后校验 90 卡守恒、
  每层张数、12 张明牌、3 名贵族、初始宝石和 agent 空状态。
- [ ] 同一 snapshot 经 `Game`、`evaluation.py`、`ppo_selfplay.py` 进入时，初始 observation、legal mask、
  board hash 必须一致；这比“同一个整数 seed”更强。
- [ ] 建立逻辑 split：`train-schedule`、`dagger-rollout`、`teacher-validation`、`validation-A`、
  `validation-B`、`sealed-test-iid`、`stress`；实际整数范围须在运行前同步登记到 `seed_registry.py` 与
  `docs/seed_registry.md`，并通过不重叠测试。
- [ ] 除 source seed 外，再按 `scenario_id` 和 state hash 检查跨 split 重复；bank 为内容寻址、只读产物。
- [ ] sealed 的含义是“冻结且未消费”，不是“seed 对人保密”；建立 append-only consumption ledger，
  只有 `purpose=final_report` 且候选集合已冻结时才允许读取。

Scenario 分层只用初态，不用胜负或 rollout。删除信息量恒为零的“开局可买性”（玩家开局无宝石），改为
版本化的可计算描述子：明牌点数密度、最低/分位 token deficit、成本颜色熵/集中度、贵族需求重叠、
明牌与贵族需求对齐度、低成本高分卡数量。任何“rush index”“阻塞难度”“贵族竞速”名称都必须先给出
精确公式、阈值和单测。

- [ ] `sealed-test-iid` 是自然发牌分布上的主估计集；
- [ ] 平衡分层与 rush/贵族/阻塞压力集仅作稳健性诊断；
- [ ] 若从大候选池按 strata 过采样，保存 inclusion probability。不得把压力集不加权平均冒充 IID 胜率；
  需要总体估计时使用预注册权重并同时报告未加权分层结果。

建议规模：快速筛选 50 unique scenarios，validation-A 100–200，validation-B 100–200；final IID 为
`max(200, power_analysis_n)` 并向上取整到 50，初始上限 500。若 500 仍不足以识别预注册最小效应，
必须写“underpowered”，或在开封前扩容，不能观察结果后加局。

**退出门槛**：bank schema、hash、守恒、split 去重、三 runner 初态 parity 和 sealed access gate 全部通过。

### T1.3 配对评测、统计与 power analysis

- [ ] treatment/control 与 official/`ppo-best` 均对同一 scenario、同一 opponent、两个目标座位运行；
  随机对手使用相同事件键随机变元。
- [ ] 每局记录 checkpoint/代码 hash、scenario、seat、opponent snapshot、全部 RNG lineage、WDL、raw/calScore、
  回合数、动作类型、卡/贵族、延迟、失败原因和 action-trace hash。
- [ ] 唯一键至少为 `(candidate_hash, opponent_hash, scenario_id, seat, replicate_id)`；重复键拒绝写入。
- [ ] 失败局保留在 scheduled denominator，同时 formal batch 因任何失败而整体标记 invalid；修复后只能按
  原 schedule 续跑/补齐，不能换 seed。不得像 legacy league 那样静默排除错误局。
- [ ] checkpoint 级差值：先在 scenario 内合并双座次；bootstrap 时同一抽样索引联合重采样该 scenario
  的所有候选和 opponents，保持 CRN/多终点相关结构，不能为每个表格单元各自抽样。
- [ ] treatment 级方法差值只比较同一 `replicate_id` 下的 treatment 与 `O`。逐 replicate 报告，并给
  均值、标准差、最差 replicate；建议同时给 opponent-stratified IQM、performance profile 和
  probability of improvement。只有 ≥5 replicates 时才给标注清楚的 replicate→scenario 层级
  bootstrap；3 replicates 不做精确显著性声称，5 replicates 的区间也必须注明小样本限制。
- [ ] Wilson 仅可作为独立单局的附加描述；双座次、重复或 CRN 数据的正式决策不用 Wilson。
- [ ] 分别为 `d_method` 的 model-replicate 数与 `d_deploy` 的 scenario 数做 power analysis；不能只扩大
  scenario 数来补偿训练副本不足。以 pilot 的 paired difference 标准差 `s_d`、最小重要效应 `delta`、
  family-wise `alpha=0.05` 和 power ≥0.8 计算样本量；正态近似
  `n ≈ ((z_(1-alpha*) + z_(1-beta)) * s_d / delta)^2` 只作初值，再用模拟/重采样校验。
- [ ] final test 使用固定 N，禁止查看普通 95% CI 后 sequential stopping。筛选也使用预先固定的 50-deal block；
  若未来采用 group-sequential/always-valid 方法，必须另写统计规格和 alpha spending 单测。
- [ ] 确认性、非劣化、探索性 hypothesis family 在 manifest 分开；所有权重、方向、tie 规则、CI 算法、
  bootstrap seed/次数和缺失值处理在运行前冻结。

**退出门槛**：用合成数据验证聚类 bootstrap——机械复制同一 `(scenario, seat)` 不会缩窄区间；任意表格可从
`episodes.jsonl` 重算；power 报告给出选择 N 的输入和限制。

### T1.4 新协议基线复测与 crossed pilot

- [ ] 在新 IID/压力 bank 上复测冻结 `ppo-best`；明确这是新协议基线，不能直接与 C4-R2 的伪重复区间拼接。
- [ ] 固定 checkpoint × scenario × seat × opponent，估计 deal、seat、opponent 与交互造成的评测方差。
- [ ] 在新 RNG 协议下精确复训现行 C2-R2 recipe 至少 3 replicates，命名 `O_bridge`。它保留当前非零
  terminal potential，只用于量化 runner/protocol 迁移，不作为后续“safe PBRS”控制，也不进入正式 shortlist。
- [ ] 修正 terminal potential 后冻结主实验共同 reward contract：终局效用 `±10/0`、`gamma=0.99`、
  `safe-potential(kappa=0.05, terminal_phi=0)`；在同一 replicate/schedule 上训练 `O`。`O-O_bridge`
  单独估计 reward-contract 修正效应，不能并入 pool/DAgger 效应。
- [ ] `O_bridge` 与 `O` 都必须是新训练副本；不得只复用历史 seed42/1234/2024 checkpoint 来代表新
  runner 的 replicate 分布。若实现校验发现上述冻结值与 checkpoint manifest 不符，以现场 manifest 为准，
  记录勘误并在任何结果产生前重新批准。
- [ ] 做小型 crossed pilot：replicate × scenario × seat × opponent；分别报告固定模型的牌局方差和固定
  牌局的 model-seed 方差。方差分解模型只作设计工具，不把正态随机效应假设当成事实。
- [ ] 根据 pilot 冻结 `delta`、`delta_NI`、final N、确认性 replicate 数、训练预算、pool CDF、validation
  规则和总 GPU/墙钟上限，生成 approved manifest；之后不得因 validation-B/test 结果改动。

**退出门槛**：`O_bridge`/`O` 角色和 reward version 无歧义、control 可重放、方差与 power 报告完成、正式
manifest 获批、sealed test 尚未消费。

### T1.5 heuristic/rush 定向实验

所有主分支使用 T1.4 冻结的 `O` reward contract，并保持 `public-v2`、4×128 网络、3510 动作、optimizer、
训练局数和 checkpoint 频率一致。先做以下 2×2，避免把 pool 与新标签混为一个原因：

| 分支 | targeted pool | 新 target-state DAgger | 解释 |
|---|---|---|---|
| O | 否 | 否 | 修正 terminal PBRS 后的主实验重训控制 |
| A | 是 | 否 | 对手分布覆盖主效应 |
| B | 否 | 是 | 访问分布标签主效应 |
| C | 是 | 是 | 组合与交互；不能归因给单项 |

Targeted pool 的默认候选权重为
`heuristic:2, heuristic-rush:2, ga:1, minimax:1, current:1, history:1`，归一化为
25%/25%/12.5%/12.5%/12.5%/12.5%；history 是一个总 bucket，再在保留 snapshots 内均分。
该数值仍须在 T1.4 批准，runner 必须支持显式权重并记录实际抽样偏差。

DAgger 的硬边界：

- [ ] 轨迹由当前学生在 **对阵 heuristic/rush** 时产生，教师只在这些学生访问状态上给标签；
- [ ] 教师必须在独立 `teacher-validation` 上以预注册 margin 证明对目标对手优于学生、零非法动作、
  查询成本可接受，并通过 public-v2 信息审计；该 split 只做教师资格判定，不参与学生 checkpoint 选择。
  教师若读取隐藏 deck/对手隐藏预留，不得用于部署同信息集的标签；
- [ ] heuristic/rush 本身是对手风格，不自动等于“如何击败 heuristic/rush”的专家。若无合格教师，
  B/C 标记 blocked；若仍蒸馏其动作，只能命名为 style imitation，不能引用 DAgger 专家保证；
- [ ] 记录 DAgger 轮次、学生/教师 hash、`beta_i`。现实现“学生执行、教师标注”等价 `beta=0`；若改用
  teacher/student mixture，schedule 必须预注册；
- [ ] 聚合数据按完整 scenario 切分，记录查询数、动作一致率、每轮样本占比和重复状态策略。

奖励消融在 O/A/B/C 之后单独进行；正确性修复本身已经在 T1.4 完成：

- [ ] 先验证 episodic PBRS：`F_t = gamma*Phi(s_{t+1}) - Phi(s_t)`，在实际 discounted return 下检查
  `G' = G - Phi(s_0) + gamma^T Phi(s_T)`；所有真正终止（包括 rounds limit/deadlock）的 `Phi(s_T)=0`；
- [ ] 当前非零终局势能只允许作为 `O_bridge/terminal-biased-potential` 历史桥接，明确不保证策略不变；
- [ ] 比较 `none` 与修正后的 `safe-potential`，其余配置相同；终局 ±10 不在同一实验首次改变；
- [ ] Rinascimento 式 event-value 项是非策略不变的探索分支，事件表/权重/调参数据必须冻结，不能与
  safe-potential 合并后声称单因素收益。

预算分支只在最佳 O/A/B/C 配置上增加 **训练局数/updates**；不得同时改变 games-per-update、minibatch、
学习率或网络。若要测试 batch，另立 D2 分支。所有分支保留 frozen `ppo-best`、控制 O、BC/DAgger-2
初始化作为对照，并报告对 heuristic/rush 的收益是否以 GA/minimax 或行为退化为代价。

MAP-Elites 文献在这里仅用于“行为覆盖”诊断：报告买卡、预留、取宝、贵族、达 15 分回合、颜色/卡层
分布和行为 profile；本任务不引入完整 MAP-Elites 搜索。搜索教师仍后置，除非先通过独立 teacher gate。

**退出门槛**：2×2 主效应/交互可复算；DAgger 教师合格或分支明确 blocked；PBRS 终局测试通过；不存在
同时改变多个未标注变量的“最佳配置”。

### T1.6 多 replicate、选模与一次性 sealed test

1. 快速筛选：固定 50 scenarios，3 paired replicates，只淘汰明显回退/失败分支；
2. 正式训练：入选 treatment 跑完整预算，默认补足 5 independent replicates；
3. validation-A：用于每个 replicate 的 early stopping/checkpoint 选择，评测频率固定；
4. validation-B：只比较每个 treatment 已冻结的候选集合，选择 treatment 和官方 checkpoint；
5. sealed test：候选集合、代码和报告脚本 hash 冻结后执行一个不可扩展的批次。

建议使用可解释的字典序选模，而非事后调权：先排除失败/guardrail 不合格者，再最大化
`min(score_rate_heuristic, score_rate_rush)`，再比较两者均值、GA/minimax 均值、跨 replicate 离散度，
完全相同时选更早 checkpoint。确切规则必须在 T1.4 manifest 中冻结。

sealed-test 批次包含：冻结 `ppo-best`、与入选 treatment 配对的全部 `O` replicates、入选 treatment 的全部
预注册 replicates，以及已由 validation-B 指定的官方 checkpoint。官方 checkpoint 必须是 treatment 集合
成员；按 checkpoint hash 去重执行，不能因“官方”标签重复增加样本。这样 `d_method` 与 `d_deploy` 都能在
同一个 scenario/seat/opponent 矩阵上计算；test 结果绝不参与重新选 seed。runner 不得像当前
stabilization path 一样自动测试所有历史分支。

批次允许在进程故障后按原 schedule append/resume，但不能改候选、N、seed 或删掉已见结果。完成后写入
consumption ledger。若 test 未通过，只报告未通过；后续改进必须申请新的独立 sealed bank，旧 bank 只能
作为已公开回归集。

**退出门槛**：官方候选在 test 前已唯一确定；入选 treatment、配对 `O`、冻结 `ppo-best` 按同一
scenario/seat/opponent 矩阵完成；确认性与探索性结论分开；无 test-driven 返工。

### T1.7 退出审查与任务（2）移交

- [ ] RNG lineage、scenario replay、split audit、weighted pool 和 worker-order 测试通过；
- [ ] 新协议 `ppo-best` 基线、至少 5-replicate 确认结果或明确的资源限制结论已归档；
- [ ] heuristic/rush 实验完成可解释归因，或每个 blocked 分支有证据；
- [ ] 官方 2p checkpoint 按预注册规则产生，test 未用于调参；
- [ ] `make test`、ruff、新代码 mypy 和报告复算通过；若修改引擎初始化、特征、mask 或浏览器抽取，
  额外运行 `make parity`；
- [ ] 任务（2）的 3p/4p seed 段、scenario schema 扩展、ranking utility 和 seat/permutation 规则另行预注册。

任务（1）结束后，2p `ppo-best` 与新官方 2p 候选都以内容 hash 冻结；任何 3p/4p 训练不得覆盖、改名或
把 2p test bank 当作多人开发集。

## 6. 实现落点、产物与自动质量门

### 6.1 建议实现落点

| 能力 | 主要落点（可用等价模块） |
|---|---|
| RNG key/lineage/determinism | `policy_imitation/protocol.py` |
| Scenario schema、生成、装载、hash | 新建 `policy_imitation/scenario.py` |
| split/registry/consumption | `seed_registry.py`、`docs/seed_registry.md`、`manifest.py` |
| 显式 policy/pool/opponent RNG | `ppo_selfplay.py`、`policies.py`、`runner.py` |
| 配对评测与逐局记录 | `evaluation.py`；正式 league 不再依赖旧 permutation+wrap 语义 |
| cluster/nested bootstrap、power | 新建 `policy_imitation/statistics.py` |
| formal lifecycle / sealed gate | `stabilization.py` 或新建 task-1 driver |

### 6.2 最低产物

- `manifest.json`：schema v2、批准状态、hypothesis family、效应/非劣化阈值、固定 N、选模规则；
- `scenario_bank/*.jsonl.zst`：完整 ScenarioV1、strata、source segment、content hash 和 split hash；
- `training_schedule.jsonl.zst`：replicate/treatment/update/game 的 scenario、seat、pool/action/minibatch lineage；
- `episodes.jsonl.zst`：逐局 scheduled key、WDL、分数、行为、失败、延迟、action-trace hash；
- `training_metrics.csv`：loss、entropy、KL、clip、critic、梯度、采样量、实际 pool 概率；
- `model_hashes.json`：每个 checkpoint 的 SHA-256、契约、父 checkpoint/数据集；
- `statistics.json`：estimand、聚类层级、bootstrap 配置、power 输入和机器可读结果；
- `report.md`：官方 checkpoint 与 treatment-level 结果、每 replicate、IID/压力集、seat/opponent/strata、
  成本、失败和结论限制；
- `sealed_test_consumption.jsonl`：bank hash、候选集合 hash、开始/结束时间、状态和批准人。

Manifest 还必须保存 dirty/patch hash、依赖 lock hash、card/action registry hash、opponent source/config hash、
完整派生协议版本、schedule/bank hash、硬件和 deterministic flags。只记录“seed=1234”不再合格。

### 6.3 必增测试

- `test_rng_protocol.py`：canonical 派生、命名空间隔离、worker/order invariance、局部 RNG；
- `test_scenario_bank.py`：90 卡守恒、deck-top 方向、round-trip/hash、split/state 去重；
- `test_scenario_runner_parity.py`：Game/raw/PPO 初始 state-observation-mask 一致；
- `test_ppo_rng_schedule.py`：treatment 内 CRN、replicate 间独立、显式 policy/minibatch RNG；
- `test_weighted_pool.py`：CDF、history bucket、实际 snapshot lineage；
- `test_paired_statistics.py`：scenario 聚类、tie、失败分母、重复行不制造精度、固定 bootstrap seed；
- `test_manifest_v2.py`：注册表单一事实源、approval、sealed access/consumption、不可覆盖；
- `test_episodic_potential.py`：discounted telescoping、所有 terminal/truncation 的 `Phi=0`；
- 小型端到端：2 scenarios × 2 seats × 2 opponents、1/2 workers 输出相同 semantic hash，零非法动作。

## 7. 风险登记

| 风险 | 缓解措施 |
|---|---|
| 同一整数 seed 在 runner 间变义 | ScenarioV1 完整快照 + 三路径 parity；seed 仅作来源元数据 |
| CRN 因控制流分歧失效 | 事件键 RNG，不依赖可变 draw index；如实说明只能部分耦合 |
| treatment 与训练牌序混杂 | replicate 区组内共享 schedule；跨 replicate 独立 |
| 选中幸运 model seed | treatment 看全部 replicates；官方 checkpoint test 前冻结；报告最差 seed |
| 3–5 seeds 的 bootstrap 过度自信 | power 先行；逐 seed 报告；不足时降级结论，不用 deal 数代替 model seeds |
| balanced stress 结果冒充自然胜率 | IID 为主；保存 inclusion probability；压力集分层或加权报告 |
| rush 对手被误当成反制教师 | teacher quality/public-information gate；无合格教师则阻断 DAgger 分支 |
| 非零 terminal potential 改变策略 | terminal `Phi=0` 测试；legacy/event 统一标为非策略不变 |
| 对手池过拟合导致遗忘 | current/history + GA/minimax guardrail；精确记录 pool CDF/实际计数 |
| 多重比较与反复查看 | hypothesis family + simultaneous CI；固定 N final test；validation-A/B 分工 |
| 失败局被移出分母 | scheduled denominator + formal batch invalid；append-only 原 schedule 补齐 |
| manifest/registry 再次漂移 | schema v2 直接引用 registry 事实源；代码/文档同提交测试 |
| GPU 非确定性被误称 bitwise 复现 | CPU 评测逐动作复现；GPU 记录 flags/硬件并用独立 replicates 表征变异 |

## 8. 文献方法与本计划落点

| 方法 | 本计划采用 | 明确不采用/限制 |
|---|---|---|
| Deep RL 可靠评测 | 区间、IQM/performance profile、少 seed 时完整披露、power-based replicate 数 | 不以 3 个 seed 的均值或数百局伪装高置信算法结论 |
| DAgger | 学生诱导状态 + 专家标签 + 全轮次 provenance | 不把目标对手动作自动当成反制专家标签 |
| PBRS | `gamma Phi(s')-Phi(s)`，并处理 episodic terminal potential | 非零终局势能和 event bonus 不声称 policy-invariant |
| PSRO/多智能体评测 | 冻结多风格/历史 opponent、跨 seed cross-play 视角 | 本任务不实现完整 meta-solver 或 Nash 求解 |
| Rinascimento Event-Value | 事件/行为指标用于诊断，事件塑形单列消融 | 不把经验证据从其游戏/agent 直接外推为本仓库收益 |
| Splendor MAP-Elites | 用行为描述子检查覆盖与退化 | 不在本任务启动完整 quality-diversity 搜索 |

## 9. 依据与关联文档

仓库事实源：

- [C4-R2 league report](../runs/c4r2-league-20260914/league_report.md)
- [C2-R2 2000 updates report](../docs/C2R2_2000_REPORT_20260914.md)
- [seed registry](../docs/seed_registry.md)
- [shared feature schemas](../src/splendor/splendor/features_v2.py)
- [PPO self-play implementation](../src/splendor/agents/our_agents/policy_imitation/ppo_selfplay.py)
- [current reproducibility helpers](../src/splendor/agents/our_agents/policy_imitation/protocol.py)
- [current evaluation path](../src/splendor/agents/our_agents/policy_imitation/evaluation.py)
- [current shaping implementation](../src/splendor/agents/our_agents/policy_imitation/shaping.py)
- [整体提升路线](../docs/IMPROVEMENT_ROADMAP_20260912.md)
- [Splendor 文献综述](../docs/SPLENDOR_LITERATURE_SURVEY_20260912.md)

方法学原始来源：

1. Agarwal et al. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*.
   [arXiv:2108.13264](https://arxiv.org/abs/2108.13264).
2. Colas, Sigaud, Oudeyer. *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement
   Learning Experiments*. [arXiv:1806.08295](https://arxiv.org/abs/1806.08295).
3. Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret
   Online Learning* (DAgger). [PMLR 15](https://proceedings.mlr.press/v15/ross11a.html).
4. Ng, Harada, Russell. *Policy Invariance under Reward Transformations: Theory and Application to
   Reward Shaping*. [ICML 1999 PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
5. Grzes. *Reward Shaping in Episodic Reinforcement Learning*.
   [AAMAS 2017 PDF](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf).
6. Lanctot et al. *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning* (PSRO).
   [arXiv:1711.00832](https://arxiv.org/abs/1711.00832).
7. Li, Wellman. *A Meta-Game Evaluation Framework for Deep Multiagent Reinforcement Learning*.
   [arXiv:2405.00243](https://arxiv.org/abs/2405.00243).
8. Bravi, Lucas. *Rinascimento: using event-value functions for playing Splendor*.
   [arXiv:2006.05894](https://arxiv.org/abs/2006.05894).
9. Bravi, Lucas. *Rinascimento: searching the behaviour space of Splendor*.
   [arXiv:2106.08371](https://arxiv.org/abs/2106.08371).
