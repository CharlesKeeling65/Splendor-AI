# 远程推理部署手册（phase-6）

浏览器控制在本地（Mac），模型推理在远程服务器（例如 Z8），两者通过 TCP JSONL 通讯。
设计裁决见 [plan/phase-6-remote-inference.md](../plan/phase-6-remote-inference.md)。本手册
只描述当前已实现的远程 seam；真实网页操作仍须遵守自建房间和人类量级点击的礼仪约束。

## 架构

```text
┌─ 本地 Mac ────────────────────────────────┐      ┌─ Z8 ───────────────────────────────┐
│ ego-browser（每 bot 一个 profile/空间）    │      │ inference-server（TCP 8765）       │
│ DOM 抽取 / 执行器 / 掩码 / 会话            │ TCP  │ checkpoint 注册表（DQN/PPO）        │
│ play-web-remote（N 个 worker）             │◄────►│ act: obs+mask → action             │
│ play-dashboard（事件流可视化）             │ JSONL│ winrate: 同策略 MC 自对弈代理值     │
└───────────────────────────────────────────┘      └────────────────────────────────────┘
```

`play-web-remote` 不把 checkpoint 下载到本地。`inference-server` 在启动时加载模型，并
通过 `ping` 暴露每个模型的 schema、输入/输出维度、分数语义和设备。

## 一、Z8：启动推理服务

先将仓库和 checkpoint 放到 Z8（路径按实际机器调整）：

```bash
rsync -a --exclude .venv --exclude .git \
    /Users/wyb/File/Programming/Git_Code/Splendor-AI/ z8:splendor-ai/
```

当前支持的 checkpoint 类型如下：

| `model_type` | 模型 | `act` 的 `score_kind` | 备注 |
|---|---|---|---|
| `dqn`（及历史未标记 DQN） | `QNetwork` | `q` | 合法动作上取最大 Q |
| `imitation_ppo_policy_value` | 前馈 `PolicyValueNetwork` | `policy_logit` | 合法动作上按 logits 排序 |

旧课程 PPO、GRU/LSTM、self-attention PPO 和 BC-only checkpoint 会在加载时拒绝；这不是
“换个文件名”即可绕过的形状/状态契约问题。DQN 与 PPO 的 top-action 分数不可直接横向
比较，仪表盘会依据 `score_kind` 显示 Q 或 policy-logit。

### 单模型与混合模型

```bash
# DQN
uv run inference-server \
    --model dqn=runs/d1-200k/seed826001/26-09-13_17-31-58__dqn/models/dqn_model.pth \
    --port 8765 --n-rollouts 16 --device cuda

# 已验证的 feed-forward imitation-PPO（public-v2，312 维，只能 2 席）
uv run inference-server \
    --model ppo-best=runs/c2r2-selfplay-2000/training/fixed-seed1234/best.pth \
    --port 8765 --n-rollouts 16 --device cuda

# 混合注册表：重复 --model；每个名字在客户端作为 model_id 使用
uv run inference-server \
    --model dqn=/srv/checkpoints/dqn_model.pth \
    --model ppo=/srv/checkpoints/imitation_ppo.pth \
    --port 8765 --device cuda

# 也可注册目录中的全部 *.pth（文件 stem 作为 model_id）
uv run inference-server --models-dir /srv/checkpoints --port 8765 --device cpu
```

`--device` 支持 `cpu`、`cuda`、`mps`；不可用加速器会安全回退到 CPU。启动日志会列出
注册的 model_id。可用 `ping` 检查每个模型的 `feature_version`、`input_dim`、
`output_dim`、`score_kind` 和实际 device；混合注册表的顶层 device 可能显示 `mixed`，
不可替代每模型 metadata。

若网络不能直连 Z8 的 8765 端口，可使用跳板 SSH 隧道（只在已有授权的机器和端口上操作）：

```bash
ssh -L 8765:localhost:8765 wangyb@z8 -N
```

之后客户端使用 `--server localhost:8765`。

## 二、本地：运行 bot 和仪表盘

```bash
# 单 bot；本地只引用远程 model_id
play-web-remote --server <Z8地址>:8765 --model dqn \
    --games 10 --bots 1 --events-dir web_events

# 多 bot 同房间：一个 id 应用于所有 bot，或按顺序提供逗号分隔的 id
play-web-remote --server <Z8地址>:8765 --model dqn,ppo \
    --winrate-model dqn --games 20 --bots 2 --events-dir web_events

# 使用同一个 imitation-PPO 模型（ppo-best 为上面的注册名）
play-web-remote --server <Z8地址>:8765 --model ppo-best \
    --games 10 --bots 1 --events-dir web_events

# 加入已有房间与人类对局（仍只操作被授权的房间）
play-web-remote --server <Z8地址>:8765 --model dqn \
    --room-url "<房间URL>" --events-dir web_events

# 另开终端：实时仪表盘
play-dashboard --events-dir web_events --port 8899
# 浏览器打开 http://localhost:8899
```

`--winrate-model ID` 是可选的全局胜率评估模型；省略时，每个 bot 用自己的 acting
model 做评估。它不改变实际出牌模型。事件会记录 `acting_model_id`、`winrate_model_id`、
`score_kind` 和 `winrate_mode=homogeneous_selfplay_proxy`，便于发现混合模型运行中的
语义边界。

常用参数：`--n-rollouts`（每个估计的模拟局数，越大越平滑也越慢，默认 16）、`--timeout`
（单步和 socket 超时）、`--max-steps`（单局步数上限）、`--poll`（DOM 轮询间隔）、
`--profile-ids`（按 bot 指定隔离的 ego-browser profiles）。多 bot 时每个 bot 需要独立
登录态；profile 数不足会在启动前报错。

## 三、schema 与座位规则

| `feature_version` | 输入维度 | 支持实际座位 | 说明 |
|---|---:|---:|---|
| `v1` | 265 | 2–4 | 本地训练主要是 2 席；3/4 席是分布外代理值 |
| `public-v2` | 312 | 恰好 2 | 例如 `ppo-best`；不能用于 3/4 席房间 |
| `public-v2-multi` | 337 | 2–4 | 多席公共特征 |

座位 guard 在真实棋盘首次出现时读取 DOM 的 panel 数量，并校验 acting/win-rate 两个
模型。`--bots` 只是 worker 数，不能用来绕过 guard；房间实际 panel count 才是事实源。
因此一个 `--bots 1` 的 bot 加入三人房间时，仍按三席校验。未知 schema、超过四席、
legacy `public-v2` 非两席，或 schema 与 input_dim 不一致，都会在动作前失败并给出原因。

## 四、仪表盘与胜率如何解读

- 曲线为每座位每次我方操作的 `before`（虚线圆点）和 `after`（实线方块）；折叠在
  `env.step()` 中的对手动作沿用相邻估计并标 `stale`。
- top-action 显示 `Q` 或 `policy-logit`，由 `score_kind` 决定；旧事件没有该字段时是
  兼容回退，不应据此断言分数是 Q。
- 胜率是从当前快照重建引擎状态、以同一评估模型占据所有座位的 Monte-Carlo 自对弈代理值。
  `homogeneous_selfplay_proxy` 不是校准概率，也不是 PPO value head 的直接输出；未知牌
  补全和超步数 `aborted` 会在事件中保留。
- 累计战绩来自 `game_end` 的胜/平/负聚合，与局面代理值是两个指标。

## 五、掩码奇偶与常见日志

健康时 `engine-only=0` 的奇偶行不写入事件流；只有 `engine-only > 0`、缺少 PASS 或
页面无可用 DOM affordance 才发 `level=warn`。`dom-only` 的过近似动作不代表策略会选到，
策略始终从引擎合法掩码中选动作。终局时 `game_end.anomalies` 保留整局累计数。

弃宝石过渡态已经改用紧凑 status 标签，避免把页面 body 的数字墙写进日志。若出现 stale
点，先看对应 bot JSONL 的 `log` 事件，再调低 `--n-rollouts` 或检查服务端 `ping` 的设备。

## 六、排障

| 现象 | 处置 |
|---|---|
| `connection refused` | 确认 Z8 服务、端口和 SSH 隧道；再用 `--server host:port` 检查 |
| `unknown model_id` / `unknown --winrate-model` | 服务端必须用相同名字重复 `--model name=path` 注册；`--winrate-model` 也必须是 ping 返回的 id |
| `unsupported checkpoint model_type` | 只使用 `dqn` 或 `imitation_ppo_policy_value`；旧课程 PPO、RNN/self-attention、BC-only 不可直接部署 |
| `feature schema ... requires ... inputs` / `obs shape` | 检查 checkpoint metadata 与 schema；不要手动 reshape，换匹配模型 |
| `public-v2 ... exactly 2 seats` | 实际网页 panel 不是 2 席；换 `v1`/`public-v2-multi`，或使用两席房间；调整 `--bots` 不会绕过 guard |
| 混合模型 top-action 分数难比较 | 这是预期：DQN 是 Q，imitation-PPO 是 policy-logit；看 `score_kind` 和事件 metadata |
| 胜率点缺失/stale | 估计超时或页面无完整快照；降低 `--n-rollouts`，查看 `log`，不把 stale 当作 0% |
| bot 卡在等房间 | bot0 建房后会写 `events_dir/room_url.txt`；检查 profile 登录态和 bot 数 |
| `确认丢弃按钮点不动` | 检查丢弃条容器 scoping 与 bundle 版本，参见 [web_experiments.md](web_experiments.md) |

下面几项是已在真实页面核对过的房间/DOM 故障定位信息，不因改用远程模型而改变：

| 现象 | 处置 |
|---|---|
| bot 从不建房：`room URL did not appear after clicking 创建房间` | 大厅链接可能是带 emoji 前缀的 `<a href="/ccbs/xxxx">👥 创建房间</a>`；contains 匹配必须取最内层命中，不能点到 `HTML`/`BODY`/`#root` 祖先。若仍失败，先确认点击目标是该 `<a>`，再检查该 profile 是否还占着旧房间；页面只剩 `重连` 时可点击，`recover()` 会处理。 |
| 房主点了 `开始游戏` 但开局不了 | 每个座位都必须先入座；owner 会按人类节奏重试，并在 `等待…` 出现后记录 `table live after N …click(s)`。只有一个 bot（`--bots 1`）时，两人房无法开始是预期行为。 |
| `noble ... is not eligible to visit agent N` | 贵族归属按动作后的卡数判定。若复现，先按 [web_experiments.md](web_experiments.md) 的勘误 3 重下 ccbs bundle；线上发版可能改变选择器。 |
| 日志出现乱码数字墙 | 弃宝石/支付过渡态的 status 已改为紧凑标签；若复现，保存 status 原文和对应 JSONL，再检查 bundle/抽取版本。 |
| `mask_anomalies` 不清楚如何解释 | 它只累计需要人工查看的 `engine-only > 0`、缺 PASS 或页面重设计异常；E5/E6 与 PASS/RESERVE 残差不计入。恒为 0 是正常的，不能用 `len(report)` 代替 `anomaly_count`。 |

### 掩码奇偶（`⚠️ 掩码奇偶`）

每个我方决策点会把引擎掩码与 DOM affordance 集合比较。健康时 `engine-only=0` 的报告
不写入事件流；`dom-only` 很大并不等于策略会选到非法动作，因为策略只从引擎合法掩码
采样。需要关注的是 `engine-only > 0`、`dom-affordable = 0`，以及 `engine-only` 中出现
`PASS` 的情况。E5/E6 是按动作类型登记的已知差异桶，不是确认了多少条规则差异。

## 七、多账号 Cookie 隔离（per-bot profile）

两个 bot 各登录一个账号，依靠 **ego-browser 浏览器 profile 级隔离**：cookie/localStorage
属于 profile，同一 profile 的 task-space 共享登录态，不同 profile 互不相通。

- **默认行为**：`play-web-remote` 启动时查询 `profiles()`，按顺序分配 bot0→第一个 profile、
  bot1→第二个 profile……bot 数超过 profile 数会直接报错。
- **显式指定**：`--profile-ids "Profile 1,Default"`；逗号分隔且数量必须等于 bot 数。
- **空间命名**：启用 profile 后 task-space 名带后缀，例如
  `splendor-play-web-bot0@Profile 1`；旧的无 profile 空间不复用，避免继承污染状态。
- **首次登录**：每个 profile 先在 ego-browser 对应空间手动登录 ccbs；cookie 会持久保存，
  后续 bot 自动带着登录态运行。
- **增加 profile**：只有两个 profile 时最多跑两个账号；更多账号先导入：
  `ego-browser import --browser chrome --profile <目录名>`，再通过 `--profile-ids` 指定。

## 八、离线验证

协议和 loader 测试只使用夹具/loopback，不启动真实浏览器或访问外网：

```bash
make test-remote
# 等价于：
.venv/bin/python -m pytest \
  tests/test_remote_policies.py \
  tests/test_remote_protocol.py \
  tests/test_play_remote_options.py \
  tests/test_remote_dashboard.py \
  tests/test_winrate_estimator.py -q
```

完整离线回归仍使用 `make test`；真实页面的单 bot、多 bot、座位 panel guard 和曲线验收
是单独的人工步骤，不属于 CI。
