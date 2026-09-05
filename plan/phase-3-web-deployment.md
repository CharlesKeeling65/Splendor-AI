# Phase 3 · 网页部署（sim-to-real 首秀）

> **定位**：本地训练的 DQN checkpoint 接上 BrowserSplendorEnv，在真实网页环境对局；量化 sim-to-real 落差并打通经验回流。
> **依赖**：P1（checkpoint 达 M3 门槛）+ P2（环境达双轨验收）。
> **估时**：3~4 人日。
> **对照实验设计依据**：[reference/UPGRADE_ROADMAP.md](./reference/UPGRADE_ROADMAP.md) §3 P3。

## 1. 阶段目标

1. `play-web` 一键部署命令（加载 checkpoint → 网页对局 → 胜率报告）
2. 网页 50 局稳定运行（无崩溃、无超时判负、会话可恢复）
3. 本地 ↔ 网页四战场对照报告，**落差分解归因**（这份归因直接决定 P4 做不做、做哪项）
4. 网页经验回流本地 replay（off-policy 红利的通路验证）

## 2. 任务清单

| ID | 任务 | 产出 | 估时 |
|---|---|---|---|
| T3.1 | 部署入口 | `src/splendor/play_web.py` + pyproject `play-web` | 1d |
| T3.2 | 鲁棒性 | 计时保护、会话恢复、异常重试 | 1~2d |
| T3.3 | 对照与回流 | `docs/s2r_report.md` + 回流采集函数 | 1d |

## 3. 代码改动详解（说明 / 意义 / 原因）

### 3.1 `src/splendor/play_web.py`【新增】部署 harness

**内容**：`load_agent(checkpoint)`（`dqn.utils.load_saved_dqn`）；`run_game(env, q_net) -> GameReport`（贪心策略打一局，报告含 `result / my_score / rival_score / steps / duration / mask_anomalies`）；`main()`（argparse：`--checkpoint --games --seats --room-url --poll`；循环对局 + 汇总 + JSON 报告；**局间随机休息 5~15s**；任何一局异常先 `session.recover()` 重试一次）。

**意义**：部署的一等入口，也是 s2r 对照报告的数据源。

**原因**：(a) **独立入口而非塞进 `dqn/` 模块**：训练与部署是不同生命周期——部署要长时间挂机、要会话恢复、要输出报告，耦合进训练模块会让两者互相拖累（训练改了超参结构，部署脚本跟着崩）；(b) `GameReport` 携带 `mask_anomalies`：把 P2 的奇偶监控延续到部署期，**落差归因有数据可查**而非凭印象；(c) 局间随机休息：礼仪自律的可执行化——"人类量级"不能只靠单步点击延迟，还要有对局间的自然停顿；(d) recover-then-retry-once：一次网络抖动不应终止整个 50 局任务，但无限重试会掩盖系统性故障，一次是两者的平衡。

### 3.2 `pyproject.toml`【修改】

**内容**：`scripts.play-web = "splendor.play_web:main"`（与 `splendor`/`ppo`/`evolve`/`dqn` 并列）。

**原因**：命令可发现性；README 部署章节可直接引用，形成"训练用 `dqn`、本地评测用 `splendor`、网页部署用 `play-web`"的完整命令闭环。

### 3.3 经验回流采集【修改 `dqn/training.py`，新增函数】

**内容**：`collect_from_browser(browser_env, buffer, n_games)`——浏览器对局的转移 `(s, a, r, s', next_mask, done)` 直接 `buffer.add(...)` 入库；训练循环增加混合模式（本地经验 + 网页经验按比例采样）。

**意义**：**DQN 相对 PPO 的独有红利验证**——off-policy 允许把真实环境经验混入训练；PPO 的 on-policy 约束（数据必须来自当前策略）做不到这一点。

**原因**：网页对手（真人）的行为分布是本地对手池（random/minimax/自博弈）覆盖不到的——真人会贪宝石、会针对你锁资源、会犯非线性错误。即使网页采样慢（分钟级一局 vs 本地秒级），**少量真实数据混入 replay 就能修正分布偏移**，这是从"能赢 minimax"到"能赢人"的关键一步。这是整个架构选择 DQN 的核心理由之一，本阶段必须验证通路（哪怕只采 10 局验证可行性，常态化留给 P4.4）。

## 4. 自动验收目标

| ID | 验收项 | 判定标准 |
|---|---|---|
| A3.1 | 稳定性 | 50 局无崩溃、无超时判负、无未恢复的会话异常 |
| A3.2 | 报告产出 | GameReport JSON 自动生成：胜率/得分/步数/时长/异常计数齐全 |
| A3.3 | 回流通路 | 10 局网页经验入库后，本地继续训练 1000 步无异常 |
| A3.4 | 对照统计 | 四战场胜率自动统计：本地 vs random / 本地 vs minimax / 网页 vs 脚本座席 /（可选）网页 vs 人类 |

## 5. 人工验收目标

| ID | 验收项 | 要点 |
|---|---|---|
| M3.1 | **sim-to-real 落差归因评审** | 人工分析对照报告，把落差分解到（支付方式差异 / 对手分布差异 / 观测与延迟噪声）并写入 `docs/s2r_report.md`。**这份归因是 P4 的 go/no-go 依据**——落差 <5 个百分点且可解释则 P4 降级为可选；>15 个百分点且无法归因则升级 P4 优先级 |
| M3.2 | 礼仪合规终审 | 三方面日志审查：点击频率人类量级、在线时长合理、只操作自建房间无越界行为 |
| M3.3 | 人工对局体验 | 与 agent 打 3~5 局：定性判断水平（是否明显强于随手玩、有无明显可 exploit 的固定模式——如永远先拿某色宝石） |
| M3.4 | 异常日志复核 | 所有 `mask_anomalies` 与 `recover` 事件逐条过目，确认归因闭环 |

## 6. 风险

- **网页对手分布不可控**（真人水平差异大）→ 胜率方差大；结论必须基于足够局数（≥50）且报告注明对手构成，禁止用 5 局数字下结论。
- **落差无法归因** → 说明存在未建模的语义差异，回到 E 系列实测补充（可能涉及 BROWSER 文档未覆盖的规则细节），必要时人工复盘对局录像（GameReport 的 steps 序列 + 截图）。
