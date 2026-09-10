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
- **累计战绩**：各座位已完成对局的 胜/平/负 与累计胜率。
- **实时日志**：🤖/🧑 动作、🗑️ 弃宝石（红底，含颜色×数量）、⚠️ 掩码奇偶、🏁/🏆 对局起止。

## 四、排障

| 现象 | 处置 |
|---|---|
| `connection refused` | Z8 服务未起 / 端口被校园网阻断 → 用 §一 的 SSH 隧道转发 |
| `unknown model_id` | 服务端未注册该名字；`--model 名字=路径` 或 `--models-dir` |
| bot 卡在等房间 | 双 bot 模式下 bot0 需先建房（`events_dir/room_url.txt` 出现即已就绪）；120s 超时说明 bot0 未起来 |
| 胜率点缺失（stale） | 估计失败（超时/页面无面板），看 bot jsonl 里的 `log` 事件；可调小 `--n-rollouts` |
| 推理很慢 | 降低 `--n-rollouts`；Z8 上确认走的是预期 device（`ping` 返回里有 `device` 字段） |
| 日志出现乱码数字墙 | phase-6 已修复（`_status_from_body`）；若复现请附 status 原文并回报 |

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
