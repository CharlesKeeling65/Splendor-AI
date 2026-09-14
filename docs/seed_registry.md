# 种子段注册表（roadmap 2026-09-12 §A2）

> 代码侧唯一权威来源：[`src/splendor/seed_registry.py`](../src/splendor/seed_registry.py)。
> **修改任一段落必须同一次提交内同步改代码与本文件**；league/训练 runner 的 manifest
> 会在运行时强制校验（`resolve_segment` / `allocate_seeds` / `check_seeds_declared`）。

## 现行段位（2026-09-15 起生效）

| 段名 | 范围 | 容量 | 用途 | 封存 |
|---|---|---|---|---|
| `stabilization_training` | 820000–821999 | 2000 | 2026-09-08 PPO 稳定化训练（历史，仍可引用） | 否 |
| `stabilization_validation` | 822000–822999 | 1000 | 2026-09-08 PPO 稳定化验证 | 否 |
| `stabilization_test` | 823000–823999 | 1000 | 2026-09-08 PPO 稳定化独立测试 | **是** |
| `ci_smoke` | 825000–825999 | 1000 | CI / 冒烟测试专用，**永不用于结论** | 否 |
| `training` | 826000–827999 | 2000 | 新训练运行 | 否 |
| `validation` | 828000–828999 | 1000 | 模型选择 / 消融比较 | 否 |
| `independent_test` | 829000–829049 | 50 | **最终报告评测**（G1/G5 终审） | **是** |
| `c2_training` | 830000–853999 | 24000 | C2 规模化自博弈（500×16×3 seed，2026-09-13 扩容） | 否 |
| `c2_validation` | 854000–854099 | 100 | C2 模型选择 | 否 |
| `c2_test` | 855000–855099 | 100 | C2 最终报告评测 | **是** |
| `c2r2_training` | 940000–1035999 | 96000 | C2-R2 规模化（2000×16×3 seed，2026-09-13 扩容；上方为 900000–939999 历史禁区） | 否 |
| `c2r2_validation` | 1036000–1036099 | 100 | C2-R2 模型选择 | 否 |
| `c2r2_test` | 1037000–1037099 | 100 | C2-R2 最终报告评测 | **是** |
| `z_training` | 1040000–1139999 | 100000 | AZ（AlphaZero 搜索自博弈）训练发牌 | 否 |
| `z_validation` | 1140000–1140099 | 100 | AZ 迭代内模型选择 / 验证评测 | 否 |
| `z_test` | 1141000–1141099 | 100 | AZ 最终报告评测 | **是** |
| `task1_train_schedule` | 1150000–1349999 | 200000 | Task-1 配对 PPO 训练 ScenarioV1 来源（5×2000×16 后仍有余量） | 否 |
| `task1_dagger_rollout` | 1350000–1369999 | 20000 | Task-1 学生访问分布 DAgger rollout | 否 |
| `task1_teacher_validation` | 1370000–1371999 | 2000 | Task-1 教师质量/公开信息门 | 否 |
| `task1_validation_a` | 1372000–1372999 | 1000 | Task-1 筛选与诊断 validation-A | 否 |
| `task1_validation_b` | 1373000–1373999 | 1000 | Task-1 shortlist 锁定 validation-B | 否 |
| `task1_sealed_test_iid` | 1374000–1374499 | 500 | Task-1 自然发牌 final IID bank（计划上限） | **是** |
| `task1_stress` | 1374500–1394499 | 20000 | Task-1 压力集候选池，仅作诊断 | 否 |

Task-1 的逻辑 split 与上述代码段是一一映射：`train-schedule`、
`dagger-rollout`、`teacher-validation`、`validation-A`、`validation-B`、
`sealed-test-iid`、`stress`。runner 必须调用
`resolve_task1_scenario_split()`，不得自行复制整数边界。

## 历史禁区（永不复用）

| 范围 | 原因 |
|---|---|
| 800000–819999 | 2026-09 之前全部历史实验（预注册体系之前） |
| 900000–939999 | 早期 DQN 实验（`docs/DQN_*_20260907.md` 系列） |

## 使用规则

1. **段内分配**：runner 从声明段内顺序取种子（`allocate_seeds(segment, count)`）；
   请求量超过段容量时报错，除非显式 `wrap=True`（跨 *不同对阵* 重复发牌可接受，
   但 manifest 必须记录 `wrapped`）。
2. **封存段纪律**：`independent_test` / `stabilization_test` / `c2_test` /
   `c2r2_test` / `z_test` 与 `task1_sealed_test_iid` 只允许被"最终报告"类
   评测消费，**禁止用于调参、选模或消融**。Task-1 新路径还必须通过候选集合冻结、
   已批准 manifest 与 append-only consumption ledger 三重门；知道 seed 并不等于已消费。
3. **扩容须显式修注册表**：`independent_test` 仅 50 个种子；若 C4 需要每对阵
   ≥150 局，先在本文件与 `seed_registry.py` 同提交扩段（如 829000–829349）并在
   manifest 中记录扩容提交号，再开跑。
4. **manifest 审计**：`splendor-league` 产出的 `league_results.json` 记录
   `segment / seed_start / seed_count / sealed_segment`；手工实验沿用
   `policy_imitation/manifest.py` 的 seed-groups 校验（与禁区重叠即拒跑）。
5. **复现三件套不变**（AGENTS.md 事实 6）：`random.seed` + `np.random.seed` +
   `torch.manual_seed`；league 引擎侧只消耗全局 `random`（由 `Game(seed)` 重播）。
6. **PYTHONHASHSEED=0**：所有 runner 启动前固定（风险登记册第 1 条）；
   `splendor-league` 在未设置时打印警告并写入 manifest。
7. **manifest schema v2**：新协议只保存本文件所述 segment 的名字及
   `seed_registry.py::registry_snapshot()` 的内容哈希，不允许在 manifest 中另传一套
   forbidden ranges。历史 schema v1 的窄禁区常量仅为读取旧产物而保留，也由
   `seed_registry.py` 统一导出。
8. **ScenarioV1 去重**：跨 split 除整数 seed 外还要同时按 `scenario_id` 与
   `canonical_state_sha256` 去重。bank 使用未压缩 canonical JSONL 的 SHA-256
   内容寻址；`.jsonl.zst` 只是传输编码，不改变 bank identity。
9. **IID 与压力集**：`sealed-test-iid` 逐 seed 自然发牌，inclusion probability
   固定为 1；`stress` 的平衡过采样必须使用预注册 keyed-hash lottery，保存 candidate
   pool/design hash、bin counts 与逐 scenario inclusion probability；固定排序的抽样比例
   不是 inclusion probability。未加权压力集不得冒充 IID 估计。

## 已消费记录（append-only）

| 日期 | 实验 | 段 | 种子 | 产物 |
|---|---|---|---|---|
| 2026-09-13 | league 冒烟（random×random, m=10, CI 两次） | validation | 828000–828019 | `/tmp/league-smoke*`（不入库） |
| 2026-09-13 | C1 critic 消融（5 分支 × 3 seed × 128 局） | stabilization（沿 2026-09-08 协议） | 820000–821127 等 | `runs/c1-critic-ablation/` |
| 2026-09-13 | C2 启动（500 updates × 16 局 × 3 seed） | c2_training / c2_validation / c2_test | 830000–854009 / 855000–855024 | `runs/c2-selfplay-500/` |
| 2026-09-13 | C4 league 体检（7 agent × 42 对阵 × 75 局，wrap=50） | independent_test | 829000–829049 循环 | `runs/c4-league-20260913/` |
| 2026-09-13 | E1 多席冒烟（3p/4p 各 1×16 局） | training（训练种子） | 826000 段派生 | `runs/e1-smoke-{3,4}p/` |
| 2026-09-13 | D1 DQN 200k ×3（训练 RNG 种子）+ 门槛评测 | training | 826000–826002 / 826100–826101 | `runs/d1-200k/` |
| 2026-09-13 | F1 价值标定（150 局 × 双座次） | c2_validation | 854010–854059 | `runs/f1-calibration/` |
| 2026-09-13 | C2-R2 启动（2000×16×3 seed，potential shaping κ=0.05） | c2r2_training / c2r2_validation / c2r2_test | 940000–1036099 / 1037000–1037024 | `runs/c2r2-selfplay-2000/` |
| 2026-09-14 | C4-R2 league 确认（7 agent × 42 对阵 × 75 局，wrap=50） | independent_test | 829000–829049 循环 | `runs/c4r2-league-20260914/` |
| 2026-09-13 | Z1 纯搜索基线 league（az-uniform × random/minimax/heuristic，75 局/对阵 wrap=50） | independent_test | 829000–829049 循环 | `runs/z1-uniform-search/` |
| 2026-09-13 | Z2 AZ-lite 启动（自博弈 + 价值/先验网络训练） | z_training / z_validation | 1040000 起 / 1140000–1140099 | `runs/z2-az-lite/` |
