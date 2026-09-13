# 种子段注册表（roadmap 2026-09-12 §A2）

> 代码侧唯一权威来源：[`src/splendor/seed_registry.py`](../src/splendor/seed_registry.py)。
> **修改任一段落必须同一次提交内同步改代码与本文件**；league/训练 runner 的 manifest
> 会在运行时强制校验（`resolve_segment` / `allocate_seeds` / `check_seeds_declared`）。

## 现行段位（2026-09-13 起生效）

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

## 历史禁区（永不复用）

| 范围 | 原因 |
|---|---|
| 800000–819999 | 2026-09 之前全部历史实验（预注册体系之前） |
| 900000–939999 | 早期 DQN 实验（`docs/DQN_*_20260907.md` 系列） |

## 使用规则

1. **段内分配**：runner 从声明段内顺序取种子（`allocate_seeds(segment, count)`）；
   请求量超过段容量时报错，除非显式 `wrap=True`（跨 *不同对阵* 重复发牌可接受，
   但 manifest 必须记录 `wrapped`）。
2. **封存段纪律**：`independent_test` / `stabilization_test` 只允许被"最终报告"类
   评测消费（如 C4 体检），**禁止用于调参、选模或消融**。
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

## 已消费记录（append-only）

| 日期 | 实验 | 段 | 种子 | 产物 |
|---|---|---|---|---|
| 2026-09-13 | league 冒烟（random×random, m=10, CI 两次） | validation | 828000–828019 | `/tmp/league-smoke*`（不入库） |
| 2026-09-13 | C1 critic 消融（5 分支 × 3 seed × 128 局） | stabilization（沿 2026-09-08 协议） | 820000–821127 等 | `runs/c1-critic-ablation/` |
| 2026-09-13 | C2 启动（500 updates × 16 局 × 3 seed） | c2_training / c2_validation / c2_test | 830000–854009 / 855000–855024 | `runs/c2-selfplay-500/` |
