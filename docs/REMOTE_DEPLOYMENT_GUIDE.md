# 远程推理部署手册（phase-6）

浏览器控制在本地（Mac），模型推理在远程服务器（Z8），两者通过 TCP JSONL 协议通讯。
设计裁决见 [plan/phase-6-remote-inference.md](../plan/phase-6-remote-inference.md)。

## 架构一览

```text
┌─ 本地 Mac ────────────────────────────────┐      ┌─ Z8 (server1) ──────────────────┐
│  ego-browser（Chrome 控制，每 bot 一个）   │      │  inference-server (port 8765)   │
│  DOM 抽取 / 执行器 / 掩码 / 会话           │ TCP  │   ├ checkpoint 注册表 (多个)     │
│  play-web-remote（N 个 bot 进程）          │◄────►│   ├ act: obs+mask → action      │
│  play-dashboard (port 8899) ←浏览器打开    │ JSONL│   └ winrate: MC 自对弈胜率       │
└───────────────────────────────────────────┘      └─────────────────────────────────┘
```

## 一、Z8 侧：部署推理服务

Z8 需要本仓库代码（引擎 + DQN 包）与 checkpoint 文件。

```bash
# 1. 同步仓库与 checkpoint（示例路径按实际情况调整）
rsync -a --exclude .venv --exclude .git \
    /Users/wyb/File/Programming/Git_Code/Splendor-AI/ z8:splendor-ai/
# 或 git 推送后在 Z8 拉取；checkpoint 在 runs/ 下（.pth）

# 2. 在 Z8 上启动推理服务（加载一个或多个 checkpoint）
uv run inference-server \
    --model round2=runs/round2-ema-guidance-20260907/dqn_model.pth \
    --port 8765 --n-rollouts 16
# 多模型：重复 --model 名字=路径；或 --models-dir runs/ 注册目录下全部 *.pth
```

启动成功会打印 `serving on 0.0.0.0:8765 models=['round2']`。

**网络注意**：校园网阻断直连 Z8 的非常用端口（与 3389/RDP 同理）。若 Mac 无法直连
Z8:8765，复用 server2 隧道模式做端口转发：

```bash
# 在 server2 上（或本地一条命令经 server2 跳板）：
ssh -L 8765:localhost:8765 wangyb@z8 -N   # 本地 8765 → Z8 8765
# 之后本地用 --server localhost:8765 即可
```

## 二、本地侧：跑 bot + 仪表盘

```bash
# 单机器人（自建房间，URL 会打进事件流/room_url.txt）
play-web-remote --server <Z8地址>:8765 --model round2 \
    --games 10 --bots 1 --events-dir web_events

# 双机器人同房间自对弈（bot0 建房，bot1 自动加入）
play-web-remote --server <Z8地址>:8765 --model round2 --bots 2 --games 20

# 加入已有房间与人类对局（礼仪约束不变）
play-web-remote --server <Z8地址>:8765 --model round2 --room-url "<房间URL>"

# 另开一个终端：实时仪表盘
play-dashboard --events-dir web_events --port 8899
# 浏览器打开 http://localhost:8899
```

常用参数：`--n-rollouts`（每次胜率估计的模拟局数，越大越平滑越慢，默认 16）、
`--timeout`（单步等待上限，也用作 socket 超时）、`--max-steps`（单局步数上限）。

## 三、仪表盘读法

- **折线图**：每个座位两条线——「操作前」（虚线圆点）与「操作后」（实线方块），
  一个操作即一对前后点；灰/缺点的位置表示该步胜率估计失败（降级，`stale`）。
- **胜率定义**：当前局面重建完整引擎状态 → 当前策略自对弈 N 局至终局
  （未知牌库/对手预留按注册表均匀抽样）→ 该座位最高分终局频率；同分按卡数
  tie-break，仍并列各计 0.5。
- **支持的座位数 = 2~4**：v1 的指标块为观察者两侧各留 `MAX_RIVALS`(=3) 个对手分数槽，
  结构上容得下 4 席，估算器因此接受 2~4 人局；超过 4 席或使用 `public-v2` 特征（只编码 2 席）
  会显式报 `supports 2..4 seats`。
  **但要注意 3~4 人局是分布外代理值**——本地 DQN 训练全部是 2 人局，所以 3~4 人局的数字
  可用于看趋势/相对变化，不等于策略在 3~4 人局的真实胜率。
  （2026-09-10 之前此处硬性要求恰好 2 席，于是 3 人局自对弈每次估计都失败、
  仪表盘永远打不出胜率点；现已修正，`--bots 3`/`--bots 4` 的房间可正常出点。）
- **累计战绩**：各座位已完成对局的 胜/平/负 与累计胜率。
- **实时日志**：🤖/🧑 动作、🗑️ 弃宝石（红底，含颜色×数量）、⚠️ 掩码奇偶、🏁/🏆 对局起止。

### 掩码奇偶那一行怎么读（⚠️ 掩码奇偶）

每次我方决策点，引擎掩码与"DOM 可点集"会各测一次并对比，结果推成一行日志。**只有 `engine-only` 需要担心**：

| 字段 | 含义 | 正常值 |
|---|---|---|
| `engine-legal` | 引擎 `getLegalActions` 认为当前合法 | 十几到几十 |
| `dom-affordable` | DOM 机械事实"可能支持"的动作数 | 数千（**故意过近似**） |
| `shared` | 两者交集 | = `engine-legal` |
| `engine-only` | 引擎说合法、但 DOM 支持不了 | **必须为 0** |
| `dom-only` | DOM 支持、引擎不合法 | 数千，正常 |

- `engine-only=0` 时报告会自带 `[direction safe: ...]`：策略只从**引擎掩码**里采样，所以 dom-only 的动作**根本选不到**，只是"看得见够不着"。
- `dom-only` 为什么这么大：`dom_affordances` **按设计不重复引擎规则**——它不检查买得起与否、不看 `returned_gems`（归还那些宝石到底可不可能）、不合并被预枚举的 5 个贵族槽。实测（`opening.html`，空手满桌）：3465 个 dom-only 里 **89% 带 `returned_gems`、84% 带 `noble_index`**，而真正的"自愿少拿"（E5）形态只占 **0.43%**。
- `KNOWN-DIFFERENCE CLASS E5/E6` 这两段的**数字不是"确认了多少处规则差异"**：它们的匹配条件是**动作类型**（所有 collect / 所有 buy），所以 E5 桶会把归还变体、贵族槽变体、以及恰好看不出差别的普通非法动作一起收进去，E6 桶同理会把"买不起的卡"和真正的 7 张上限混在一起。**要看的是"有没有哪个 dom-only 动作的类型不属于任何已登记类别"**，那才是需要人来判断的新情况。
- 需要报警的只有三种：`engine-only > 0`（引擎给了页面执行不了的动作）、`dom-affordable = 0`（页面重设计/抽取脚本全失配）、`engine-only` 里出现 `PASS`（顶部动作按钮消失）。


## 四、排障

| 现象 | 处置 |
|---|---|
| `connection refused` | Z8 服务未起 / 端口被校园网阻断 → 用 §一 的 SSH 隧道转发 |
| `unknown model_id` | 服务端未注册该名字；`--model 名字=路径` 或 `--models-dir` |
| bot 卡在等房间 | 双 bot 模式下 bot0 需先建房（`events_dir/room_url.txt` 出现即已就绪）；120s 超时说明 bot0 未起来 |
| bot 从不建房：`room URL did not appear after clicking 创建房间` | 已修（2026-09-11）：大厅链接是 `<a href="/ccbs/xxxx">👥 创建房间</a>`（emoji 前缀→contains 匹配），而 contains 在文档序里先命中 `HTML`/`BODY`/`#root` 等祖先，取第 0 个命中＝点 `<html>`，不会导航。已改为**只保留最内层命中**。若复现：先确认点击目标是那个 `<a>`，再查该身份是否还占着旧房间（页面只剩 `重连` 时点它即可，`recover()` 已自动处理） |
| 房主点了 `开始游戏` 但开局不了 | 需要每个座位都已入座；owner 现在会按人类节奏重试并在 `等待…` 状态出现后记录 `table live after N …click(s)`。若只有 1 个 bot（`--bots 1`），2 人房永远开不了——这是预期 |
| 胜率点缺失（stale） | 估计失败（超时/页面无面板），看 bot jsonl 里的 `log` 事件；可调小 `--n-rollouts` |
| 报 `supports 2..4 seats, got N` | N>4（房间开多了）或该模型是 `public-v2` 特征（只支持 2 席）；换 v1 模型或减座 |
| `确认丢弃按钮点不动` | 已修（2026-09-10）：scoping 到 `div.space-x-2.p-2.bg-gray-400` + 最内层匹配 + `已选 M/N` 前置校验。若复现，附 jsonl 与 `docs/web_experiments.md` 的 §勘误 1/2 一并回报 |
| `noble ... is not eligible to visit agent N` | 已修（2026-09-10）：贵族归属按**动作后**卡数判定。若复现，说明 bundle 版本已变，按 §勘误 3 重下 chunk 复核 |
| 推理很慢 | 降低 `--n-rollouts`；Z8 上确认走的是预期 device（`ping` 返回里有 `device` 字段） |
| 日志出现乱码数字墙 | phase-6 已修复（`_status_from_body`）；若复现请附 status 原文并回报 |
| `mask_anomalies` 是什么 | 每局累加的**需要人看**的奇偶行数（`engine-only>0` / 缺 PASS / 页面重设计），由 `monitor.anomaly_count` 统计。E5/E6 桶与 PASS/RESERVE 残差**不计入**——那是 `dom_affordances` 按设计的过近似，策略根本选不到。**恒为 0 是正常的，>0 才要查**（别用 `len(report)` 计数：报告恒有表头行，那样连完全一致也会报 2 个"异常"） |

## 五、多账号 Cookie 隔离（per-bot profile）

两个 bot 各登录一个账号，靠的是 **ego-browser 浏览器 profile 级隔离**：
cookie/localStorage 属于 profile，同一 profile 下的所有 task space 共享登录态，
不同 profile 互不相通（2026-09-10 实测：两空间 localStorage 35 vs 0）。

- **默认行为**：`play-web-remote` 启动时查询 `profiles()`，按顺序把
  bot0→第 1 个 profile、bot1→第 2 个 profile……bot 数超过 profile 数会直接报错。
- **显式指定**：`--profile-ids "Profile 1,Default"`（逗号分隔，数量必须等于 bot 数）。
- **空间命名**：启用 profile 后 task space 名带后缀（如 `splendor-play-web-bot0@Profile 1`），
  旧的无 profile 空间不会被复用，避免继承已污染的登录态。
- **首次登录**：每个 profile 需要手动登录一次 ccbs（在 ego-browser 界面里
  打开对应空间登录即可，cookie 持久保存，之后 bot 自动带着登录态跑）。
- **加 profile**：机器上只有 2 个 profile 时最多跑 2 账号；更多账号先导入：
  `ego-browser import --browser chrome --profile <目录名>`，再用
  `--profile-ids` 指定。
