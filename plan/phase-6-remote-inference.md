# Phase-6：本地浏览器控制 + 远程推理（TCP JSONL）+ 实时胜率可视化

> 状态：已实现（dev 分支）。本文是 phase-6 的设计与验收裁决；操作者步骤见
> [docs/REMOTE_DEPLOYMENT_GUIDE.md](../docs/REMOTE_DEPLOYMENT_GUIDE.md)。

## 1. 需求与边界

浏览器控制端在本地（例如 Mac）运行，checkpoint 加载与推理在远程机器（例如 Z8）运行。
DOM 抽取、伪状态、执行器、掩码和会话留在本地；`inference-server` 提供统一的
`act`/`winrate` 服务。多机器人是 N 个 worker 进程，各自使用独立的 ego-browser
task-space、profile 和 TCP 连接，共享一个远程模型注册表。`--bots` 只表示 worker 数，
不代表房间实际座位数。

本阶段的远程动作接口支持两类当前的、可验证输入输出契约：

| checkpoint `model_type` | 适配器 | `act` 分数语义 |
|---|---|---|
| `dqn`（历史未标记的 DQN 也兼容） | Dueling/Double DQN `QNetwork` | `q`：Q 值 |
| `imitation_ppo_policy_value` | 前馈 imitation-PPO `PolicyValueNetwork` | `policy_logit`：掩码后的策略 logits |

旧课程 PPO、GRU/LSTM、self-attention PPO，以及仅有 BC 输出而没有本阶段
`PolicyValueNetwork`/feature metadata 的 checkpoint 明确拒绝；它们缺少可重建的输入
契约或需要远程端未提供的 recurrent state。远程 PPO 的 logits 只用于合法动作排序，
不是可与 DQN Q 值直接比较的价值估计。

## 2. 协议规范（`splendor/remote/protocol.py`）

报文是 UTF-8 JSON 对象加换行（JSONL）。请求信封为 `{"id": int, "op": str, ...}`；
成功响应为 `{"id", "ok": true, ...}`；失败响应为 `{"id", "ok": false, "error"}`。

| `op` | 请求字段 | 响应字段 |
|---|---|---|
| `ping` | — | `models: [{id, kind, feature_version, input_dim, output_dim, score_kind, device}]`、顶层 `device` |
| `act` | `model_id`, `obs[input_dim]`, `mask[output_dim]` | `action`, `top[{idx, score, q}]`, `kind`, `score_kind` |
| `winrate` | `model_id`, `snapshot`, `actor_seat`, `n_rollouts?`, `max_steps?` | `win_rates`（按座位）、`draw_rate`, `rollouts`, `aborted`, `elapsed_s` |

`score` 是动作排序的规范字段；`q` 是兼容旧客户端的同值别名，不能据此把 PPO
logit 称作 Q 值。每个模型 metadata 的 `feature_version`、`input_dim`、`output_dim`、
`score_kind` 和 `device` 是混合模型注册表中的权威来源，不能只看顶层 `device`。

- `remote/client.py`：同步阻塞客户端；每次调用透明重连一次，二次失败上抛给逐局恢复。
- `remote/server.py`：asyncio 服务；`act` 内联处理，`winrate` 在线程中处理并以锁串行，
  避免引擎路径消费全局 RNG 时互相干扰。
- `remote/policies.py`：checkpoint 类型、schema、形状和设备的 fail-closed 加载边界。
- `remote/rollout.py`：按选定策略自对弈的 Monte-Carlo 胜率估算器。
- `remote/dashboard.py` + `dashboard.html`：stdlib HTTP 服务和事件流仪表盘。

## 3. 特征与座位兼容性

远程服务支持以下已经注册的特征 schema：

| schema | 输入维度 | 支持座位 | 说明 |
|---|---:|---:|---|
| `v1` | 265 | 2–4 | 2 人训练是当前主要分布；3/4 人数值是分布外代理值 |
| `public-v2` | 312 | 恰好 2 | legacy 公共特征，不能用于 3/4 人房间 |
| `public-v2-multi` | 337 | 2–4 | 多席公共特征；2 席前缀与 legacy 保持兼容 |

启动 bot 后，在首次出现真实棋盘时读取 DOM 的玩家面板数量，并同时校验 acting model
和 win-rate model。校验依据是实际 panel count，不是 `--bots`：后者只控制本地 worker
数。房间超过 4 席、`public-v2` 遇到非 2 席、未知 schema，或 schema/input 维度不一致，
都应在动作前报错；房间页和终局页没有棋盘时不作座位判断。

## 4. 胜率定义与诚实降级

**局面胜率（玩家 i，局面 s）**：从 s 重建完整引擎状态，使用选定的 win-rate policy
为所有座位同策略自对弈 N 局至终局，玩家 i 以最高分终局的频率。同分按终局卡数少者
tie-break，仍并列则各计 0.5。未知牌库和对手预留牌按注册表均匀补全；超步数的 rollout
计入 `aborted`，不伪造胜负。

因此事件中的 `winrate_mode=homogeneous_selfplay_proxy` 表示“同一个评估模型占据所有
座位”的策略强度代理值，不是校准过的概率，也不是 PPO 的 value head 直接输出。默认每个
bot 使用自己的 acting model 做估计；`--winrate-model ID` 可令所有 bot 使用指定的服务
模型做估计。事件同时记录 `acting_model_id`、`winrate_model_id`、`score_kind` 和该模式。

我方动作记录精确 `before`/`after`；`env.step()` 折叠的对手动作继承相邻估计并标
`stale: true`。估计失败只产生 stale/log 事件，不杀死部署。

## 5. 事件流与仪表盘

每个 bot 向 `events_dir/bot<i>.jsonl` 追加 `game_start`、`action`、`discard`、`parity`、
`game_end` 和 `log`。`action` 除 `before`/`after` 外包含动作模型、分数语义和胜率模式；
旧事件没有这些字段时，仪表盘保留兼容回退，但展示为未知/兼容值。

仪表盘展示：

- 每座位的操作前（虚线圆点）/操作后（实线方块）胜率曲线；
- Q 值或 policy-logit 的 top-action 信息，按事件 `score_kind` 标注；
- 胜率代理模型和 `homogeneous self-play proxy` 标签；
- 座位 × 局数/胜平负/累计胜率；
- 弃宝石、奇偶异常和生命周期日志。

弃宝石过渡态没有普通“等待操作”叶子，状态抽取已改为紧凑标签并限制未知文本长度，
避免把 body 中的数字墙写入日志；回归测试覆盖该 fallback。

## 6. 不变式与纪律

- 本地 `play-web-remote` 不加载 checkpoint；服务端严格校验 model type、schema、形状、
  mask 和有限值，并拒绝不支持的旧模型。
- `play-web` 原有直接部署路径保持 DQN-only；远程 seam 才扩展到当前 feed-forward
  imitation-PPO。
- 引擎语义复用 `build_pseudo_state`、注册表和 `create_action_mapping`；不在浏览器层重写规则。
- 真实服务器只操作自建房间；执行器保持人类量级点击和局间随机休息。

## 7. 验收

离线质量门（不访问真实网页或外网）至少覆盖：

- `tests/test_remote_policies.py`：DQN/PPO 加载、metadata、schema/shape/device guard、
  不支持 checkpoint 拒绝；
- `tests/test_remote_protocol.py`：ping/act/winrate、score/q 兼容字段和 loopback 协议；
- `tests/test_winrate_estimator.py`：DQN/PPO rollout、seat/schema guard 与胜率聚合；
- `tests/test_play_remote_options.py`：model 映射、`--winrate-model` 和选项校验；
- `tests/test_remote_dashboard.py`：canonical `score`/legacy `q` 排名裁剪、动态 Q/policy-logit
  与 homogeneous-self-play-proxy 标签，以及实际 DOM seat metadata 文案；
- `tests/test_status_fallback.py`、既有 browser/parity 测试。

真实页面验收（需要单独授权和可用账号）包括单 bot、多 bot 同房间、实际 DOM panel
座位 guard、事件流和胜率曲线；它们不属于 CI，也不应在离线测试中访问网络。
