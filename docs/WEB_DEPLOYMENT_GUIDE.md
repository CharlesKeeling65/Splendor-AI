# 浏览器部署与可视化使用手册（训练后）

> 对象：训练完成后，要在 game.hullqin.cn/ccbs 上部署 DQN agent 并**亲眼观看/对战**的操作者。
> 前置阅读：[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)（先有 checkpoint）。
> 机制依据：[plan/phase-2-browser-layer.md](../plan/phase-2-browser-layer.md)、
> [docs/web_experiments.md](./web_experiments.md)（全部 DOM 行为均经真实页面实测）。

---

## 1. 前置条件

```bash
# ① checkpoint（训练产物）
ls runs/m3/*/dqn_model.pth            # 或任意满意的 checkpoint

# ② ego-browser CLI 可用（play-web 经它驱动真实浏览器）
ego-browser nodejs <<'EOF'
await useOrCreateTaskSpace('splendor-deploy-check')
cliLog(await (await pageInfo()).url)
EOF

# ③ 礼仪红线（硬约束，违反可能被服务器封禁）：
#    只在自建房间 / 只与自己的另一身份或自愿的人类对局 / 点击保持人类节奏 /
#    局间休息 5-15s（play-web 已内置）/ 单次在线 ≤2h
```

## 2. 场景 A：人机对局 + 全程旁观（推荐首次使用）

你自己建房、入座、开局，agent 加入另一个座位——你直接在浏览器里看它每一步怎么走，还可以与它对抗。

```bash
# 第 1 步：你在自己的浏览器打开 https://game.hullqin.cn/ccbs
#          点「创建房间」→ 记下房间号（如 /ccbs/bs42）→ 点「加入」入座座位 1
#          （不要点「开始游戏」——等 agent 入座后再点）

# 第 2 步：启动 agent（钉定你的房间）
play-web --checkpoint runs/m3/<时间戳>/dqn_model.pth \
         --room-url https://game.hullqin.cn/ccbs/bs42 \
         --games 3 --timeout 120
```

行为说明：
- agent 会导航到该房间并**自动点击「加入」占第一个空闲座位**（你坐 1 它坐 2；你坐 2 它坐 1 并自动开局）；
- 你是房主时由**你**点「开始游戏」；agent 每到你回合会在 0.2-0.5s 人类节奏内落子（取宝石/买卡/预定/放弃全流程自动，多支付方式自动选"金用量最少"方案）；
- 控制台会打印每局结果：`game 1/3: win (15.0 vs 8.0, 43 steps, 210s, 0 anomalies)`；
- **观战与对战都在你自己的浏览器里完成**——agent 的每次点击会实时呈现在页面上。

> 时序提示：若 agent 在你点「开始游戏」前就位，直接开始即可；`--timeout 120` 是它等待你行动的耐心上限，超时它会先"放弃"保命再报错。

## 3. 场景 B：全自动挂机 + 旁观

不钉定房间，agent 自建房间、自动开局、连打 N 局（需要对面是脚本座席——由另一台 ego-browser 任务空间驱动的 `SessionManager`+`BrowserSplendorEnv` 进程，或另一身份的 `play-web` 实例）：

```bash
play-web --checkpoint runs/m3/<时间戳>/dqn_model.pth --games 50 --task-space splendor-deploy
# 启动后控制台会打印：
#   room: https://game.hullqin.cn/ccbs/xxxx  <- open this URL to watch
```

**旁观方法**：把打印的房间 URL 打开在任意浏览器窗口，以观战身份进入（点「观战」或加入剩余座位旁观）。注意：
- 你的浏览器与 agent 的 ego-browser 共享登录态时，**不要**在 agent 运行中做会改 gid 的操作（见 §6）；
- 旁观只读页面不会干扰对局；但不要用第二个窗口操控 agent 的座位。

## 4. 场景 C：双开自博弈（两个身份自己打自己）

两个终端、两个 ego-browser 任务空间、两个身份（实测配方，含两个必踩的坑）：

```bash
# 终端 1（身份 A，建并钉定房间）
play-web --checkpoint <ckpt> --games 20 --task-space splendor-selfplay-a \
         --room-url https://game.hullqin.cn/ccbs/<房间号>

# 终端 2（身份 B，钉定同一房间）
play-web --checkpoint <ckpt> --games 20 --task-space splendor-selfplay-b \
         --room-url https://game.hullqin.cn/ccbs/<房间号>
```

房间号先由任一身份手动创建（ego-browser 打开大厅点「创建房间」即可）。两个坑（均实测过）：
1. **双域 cookie**：换身份必须同时删 `game.hullqin.cn` 与 `.game.hullqin.cn` 两个域的 `gid`（`SessionManager.switch_identity` 已封装）；
2. **共享 cookie jar + ws 身份固定**：ego-browser 多任务空间共享 cookie jar，且页面 ws 身份在握手时固定——**后换身份的一方才允许导航**，先入座的一方入座后不要再刷新/导航，否则两个窗口变成同一身份（症状：`你已在新的浏览器窗口进入该房间`）。

## 5. 对局数据回流（可选，off-policy 红利）

网页对局的转移可以直接混入本地 replay 继续训练（DQN 独有能力，PPO 做不到）：

```python
import splendor.splendor.gym  # noqa: F401
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer
from splendor.agents.our_agents.dqn.training import collect_from_browser
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.driver import BrowserDriver  # 你的驱动实现
from splendor.browser.session import SessionManager

driver: BrowserDriver = ...  # ego-browser / CDP 适配器
env = BrowserSplendorEnv(driver, SessionManager(driver))
buffer = ReplayBuffer(capacity=500_000, n_step=3)
stats = collect_from_browser(env, buffer, n_games=10,
                             q_net=load_saved_dqn())   # 不传 q_net 则随机策略采集
print(stats)  # {"games": 10.0, "steps": ..., "avg_score": ...}
# 之后照常用这些样本继续训练
```

## 6. 会话与身份管理须知

| 现象 | 原因 | 处置 |
|---|---|---|
| 页面显示"当前浏览器禁用了cookie" | 删除 gid 后服务器的判定竞态 | 点页面上的「重连」（`session.recover()` 已自动处理） |
| "你已在新的浏览器窗口进入该房间" | 同一身份双连接 | 关掉多余窗口/任务空间；重新走双身份配方（§4 坑 2 的顺序） |
| agent 卡在等待 | 对手挂机或网络异常 | `--timeout` 到点会先"放弃"保回合；连续失败请 `session.recover()` 或重启 play-web |
| `SnapshotSchemaError` | 页面改版（ccbs-* 类名变化） | fail-fast 设计；修 `src/splendor/browser/dom_extractor.py` 的 JS 一处 + 重采夹具 |

## 7. 输出与报告解读

- **控制台逐局行**：`result (my vs rival, steps, duration, anomalies)`——`win/draw/loss` 按面板分差；`aborted` 表示达到 `--max-steps` 未终局（如实报告，不伪造）；
- **`mask_anomalies`**：引擎掩码与 DOM 可点元素的差异计数。已裁决的规则差异（网页允许少拿宝石/丢刚拿色）会被 monitor 归因为"已知差异"，**零未解释差异**才算质量达标；持续非零请按 [web_experiments.md](./web_experiments.md) ADR 的白名单核对；
- **`s2r_games.json`**（`--working-dir` 下）：逐局明细 + 汇总（胜率/平局率/平均分/异常总数），是 [s2r_report.md](./s2r_report.md) 的数据源——四战场对照（本地 vs random / 本地 vs minimax / 网页 vs 脚本 / 网页 vs 人类）凑齐后做落差归因，决定 P4 是否重开。

## 8. 已知限制（部署前须知）

1. **自然终局 DOM 未实测**（E3 只实测了强退路径）：agent 以"棋盘消失"判定终局，天然 15 分终局首次遇到时请观察并回填 `web_experiments.md` E3；
2. **支付策略为贪心**（金用量最少，对齐本地训练分布）；"留金"等高级支付属 P4-T4.2；
3. **吞吐**：ego-browser 适配器每次操作起一个 Node 进程（~1s），单步延迟主要在对局节奏本身；追求低延迟可换 CDP 适配器（driver 协议已隔离，只改一处）；
4. **礼仪预算**：50 局 × 3-5 分钟 ≈ 3-4 小时在线——分多次部署，单次 ≤2h。
