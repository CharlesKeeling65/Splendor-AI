# Phase-6: 本地浏览器控制 + 远程推理（TCP JSONL）+ 实时胜率可视化

> 状态：已实现（dev 分支）。本文档是 phase-6 的设计裁决记录；
> 操作者视角的部署步骤见 [docs/REMOTE_DEPLOYMENT_GUIDE.md](../docs/REMOTE_DEPLOYMENT_GUIDE.md)。

## 1. 需求与裁决

| 需求 | 裁决 |
|---|---|
| 浏览器控制端在本地（Mac）运行，模型推理在远程（Z8）进行 | 沿 `BrowserSplendorEnv` 的现有边界拆分：**DOM 抽取/执行器/掩码/会话留在本地**（依赖本地浏览器），**checkpoint 加载与推理外发**。唯一改动是 harness 把 `_greedy_action` 的本地前向换成一次远程调用 |
| 支持单/多机器人同时操作 | 多机器人 = N 个 worker 进程（`--bots N`），各自独立 ego-browser task-space + 会话身份（会话层"一个进程一个座位"的既定契约），共享同一个远程推理服务（每 bot 一条 TCP 连接，asyncio 天然并发）。无 `--room-url` 时 bot0 建房并发布 URL（`events_dir/room_url.txt`），其余 bot 等待后加入，bot0 延迟后开局 |
| 实时胜率折线图，区分每个玩家每个操作前后 | 胜率定义见 §3；每个操作事件携带 `before`/`after` 两个逐座位胜率数组，仪表盘按「前→后」两点绘制 |
| 玩家弃宝石日志输出异常修复 | 根因与修复见 §4 |

协议选型：**TCP + JSONL**（用户裁决）。零新依赖（stdlib asyncio/socket），报文人类可读（`nc` 直接调试），一行一消息。265 维 obs + 3510 维 mask 序列化约 30 KB，LAN 上开销可忽略。`MAX_FRAME_BYTES=16MB` 是失控保护而非调参项。

## 2. 协议规范（`splendor/remote/protocol.py`）

帧：UTF-8 JSON 对象 + `\n`。信封：请求 `{"id": int, "op": str, ...}`；成功响应 `{"id", "ok": true, ...}`；失败响应 `{"id", "ok": false, "error"}`。

| op | 请求字段 | 响应字段 |
|---|---|---|
| `ping` | — | `models: [{id, feature_version, input_dim}]`, `device` |
| `act` | `model_id`, `obs[265]`, `mask[3510]` | `action: int` |
| `winrate` | `model_id`, `snapshot`, `actor_seat`, `n_rollouts?`, `max_steps?` | `win_rates[按座位序]`, `draw_rate`, `rollouts`, `aborted`, `elapsed_s` |

组件：
- `remote/client.py` — 同步阻塞客户端；每次调用透明重连一次（网络抖动不杀部署，二次失败上抛交给逐局恢复）。
- `remote/server.py` — asyncio 服务，每连接一个 task；`act` 内联处理，`winrate` 抛给工作线程并加锁（估算走引擎路径会消耗全局 RNG，必须串行——AGENTS.md 事实 6）。
- `remote/rollout.py` — Monte-Carlo 估算器（§3）。

## 3. 胜率定义（唯一权威表述）

**实时局面胜率（玩家 i，局面 s）**：在从 s 重建的完整引擎状态上，用当前 DQN 策略为**所有座位自对弈**模拟 N 局至终局，玩家 i 以最高分终局的频率。同分按终局卡数少者 tie-break（复用引擎 `calScore`），仍并列各计 **0.5** 胜。

- **未知信息补全**：牌库剩余与对手预留卡面按注册表（90 卡 − 已见牌面）均匀抽样；对手预留按泄漏的 tier 抽样。因此它是"该位置下策略强度"的估计，不是精确求解；seed 固定后可复现。
- **终局判定**：镜像引擎 `gameEnds`（任一 ≥15 分且回合轮转到 0 号座，或全员 pass）；超步数上限的 rollout 记为 aborted（所有座位 0 胜，单独计数）。
- **批量实现**：rollout 龙骨同步步进（lockstep），每拍只做一次批量 `forward` —— 同向量化 RL env 的技巧；引擎侧（合法动作/后继/特征）是逐 rollout 的不可压缩 Python 成本。
- **更新时机**：我方动作 = 精确 before/after（动作前估计 + 动作后估计）；对手动作在 env.step 折叠的对局回合内发生，无法在不阻塞轮询的前提下逐动作估计——对手事件继承相邻两次估计（`stale: true` 标记），诚实降级而非伪造精度。
- **约束**：v1/public-v2 特征仅支持 2 人局，估算器对 ≠2 座位显式报错。

**累计胜率**（仪表盘数字面板）：该座位已完成对局中获胜局数 / 总局数（`game_end` 事件聚合）。

## 4. 弃宝石日志异常：根因与修复

**症状**：日志出现"很多带换行的数字"。

**根因**（`dom_extractor._status_from_body`）：弃宝石过渡态（"请丢弃 N 个宝石"）与支付过渡态（"请选择支付方式"）的页面**没有** `等待(你|玩家N)操作` 状态叶子，JS 侧 status 扫描为空后，回退路径把**整个 body innerText** 原样塞进 `status` 字段——页面上宝石数、卡分等数字一行一个，任何携带 status 的日志/`TimeoutError` 消息（如 `_wait_for_my_turn` 超时）都被这面数字墙污染。

**修复**：已知过渡态短语 → 紧凑标签（`等待丢弃N个宝石` / `等待选择支付方式` / 终局标记）；未知文本折叠空白并截断至 120 字符。回归测试 `tests/test_status_fallback.py`。语义兼容性：`looks_like_game_over` 的标记匹配与房间页"无棋盘即终局"信号均不受影响。

## 5. 事件流与仪表盘

事件（`events_dir/bot<i>.jsonl`，追加式，每 bot 一个文件避免跨进程锁）：
`game_start` / `action`（含 `before`/`after` 胜率数组、`stale`）/ `discard`（我方动作携带精确 `returned_gems`；对手为面板差分推断）/ `parity`（掩码奇偶报告）/ `game_end` / `log`。

对手动作标注（`diff_label` 启发式）：守恒性使三类可分——取宝石 supply→panel、弃宝石 panel→supply（>10 张返还子流程）、购买分数上升。文档与 UI 均标注"推断"。

仪表盘（`remote/dashboard.py` + `dashboard.html`，stdlib `http.server` + Chart.js CDN，1.5s 轮询 `/api/state`）：
- 折线图：每座位「操作前」（虚线圆点）/「操作后」（实线方块）两条曲线，tooltip 显示操作者与动作描述；
- 累计战绩表（座位 × 局数/胜平负/累计胜率）；
- 实时日志流（弃宝石行红底高亮）。

## 6. 不变式与纪律

- **零行为变化**：`play_web`/`BrowserSplendorEnv` 主路径未动；唯一新增是 `BrowserSplendorEnv.session` 公开 property（与 `driver` 同级的 harness 内省层）。
- **礼仪硬约束**原样继承（执行器人类节奏点击 + 局间 5–15s 随机休息）。
- 胜率估计失败永不杀部署：降级为 `stale` 事件 + `log` 记录。
- 引擎语义复用纪律：状态重建走 `build_pseudo_state` + 注册表；动作回查走 `create_action_mapping`（P0 缓存，O(1)）；未重写任何规则。

## 7. 验收

- `make test` 全绿（新增：`test_status_fallback.py`、`test_winrate_estimator.py`、`test_remote_protocol.py`，全部离线——协议测试仅用 loopback）。
- 端到端验收（需真实页面，待部署条件解除）：单 bot 10 局零未解释奇偶差；双 bot 同房间自对弈完整跑通；仪表盘折线随操作推进更新。
