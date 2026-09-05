# DQN 训练实操指导手册

> 对象：要在本仓库训练《璀璨宝石》DQN agent 的操作者。
> 前置阅读：[plan/phase-1-dqn-training.md](../plan/phase-1-dqn-training.md)（设计与验收标准）、
> [plan/reference/DQN_GUIDE.md](../plan/reference/DQN_GUIDE.md)（算法细节与十二陷阱）。
> 训练完成后 → [WEB_DEPLOYMENT_GUIDE.md](./WEB_DEPLOYMENT_GUIDE.md)（浏览器部署与可视化）。

---

## 1. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.12+**（推荐 3.13） | 引擎用 `typing.override`，3.11 会 ImportError |
| 包管理 | uv（或 pip） | `uv venv --python 3.13 .venv && uv pip install -e ".[dev]" pytest` |
| 硬件 | CPU 可训，CUDA/MPS 更快 | `--device cpu` 回落安全；MPS 当前不启用（与 PPO 行为一致） |
| tkinter | 仅本地 GUI 评测需要 | macOS：`brew install python-tk@3.13` 后用 Homebrew Python 建 venv；纯训练不需要 |
| 时间预算 | M1 数小时 ~ 半天（CPU）→ M3 数天 | 挂机任务，见 §5 课程表 |

环境自检：

```bash
.venv/bin/python -m pytest tests/          # 85 tests 全绿 = 环境正确
dqn --help                                  # console script 可用
```

## 2. 30 秒快速开始

```bash
dqn -o random --test-opponent random --total-steps 200000 -w ./runs -s 1234
```

产物在 `./runs/<时间戳>/`：`stats.csv`（逐 episode 指标）+ `models/dqn_model_<N>.pth`（每 1 万步一个 checkpoint）+ 最终 `dqn_model.pth`。

## 3. 训练管线速览（理解在调参前）

```
SplendorEnv(265 维观测, 3510 动作, 对手回合自动折叠)
   │  ε-greedy（随机动作只在合法掩码内均匀采样）
   ▼
TerminalRewardWrapper ── r = Δscore + ±10|终局（calScore 口径）
   ▼
ReplayBuffer(n_step=3) ── 打破轨迹相关性；不存当前步掩码
   ▼
Double DQN 更新 ── 在线网选动作、目标网评估；Huber；梯度裁剪 1.0；软更新 τ=0.005
```

三条承重设计（改代码前必读，见 AGENTS.md 必读事实）：
1. 非法动作直接崩溃——任何采样/argmax 前必须过 `get_legal_actions_mask()`；
2. `env.reset(seed=)` 不固定发牌——复现靠入口的 seed 三件套（`train()` 已内置）；
3. 观测 265 维**不含公共宝石供给**——策略看不见宝石池余量（P4-T4.2 可选项）。

## 4. 命令参数全解（`dqn --help`）

| 参数 | 默认 | 说明 / 何时调整 |
|---|---|---|
| `-o --opponent` | `random` | 训练对手：`random` / `minimax` / `ppo`（及各 PPO 变体）/ `itself`（自博弈，共享网络）。课程见 §5 |
| `--test-opponent` | `minimax` | 评估对手；stats.csv 的 `eval_wr` 即对它的胜率 |
| `--total-steps` | 200000 | 总步数。M1 2×10⁵ / M2 3×10⁵ / M3 5×10⁵（每阶段独立训练会话） |
| `-l --learning-rate` | 1e-4 | **不要用 PPO 的 1e-6**；不稳定再降 5e-5 |
| `--buffer-size` | 500000 | ≈1 万局；内存 ~2.6GB |
| `--batch-size` | 512 | 265 维小网络，大 batch 更稳 |
| `--win-bonus` | 10.0 | 终局胜负奖励量级；太小学成"刷分"，太大丢中间塑形 |
| `-s --seed` | 1234 | **3-seed 纪律：正式结论至少跑 42/1234/2024 三个 seed** |
| `--save-every` | 10000 | checkpoint 间隔（步） |
| `--eval-every` | 5000 | 评估间隔；评估较慢（每 5000 步打 20 局），minimax 对手时可调大 |
| `--device` | cuda | 无 CUDA 自动回落 CPU |
| `-w --working-dir` | 当前目录 | 每次运行在其下建 `<时间戳>/` 子目录 |

吞吐参考：掩码/映射构建已缓存（×75），单步开销主要是 `getLegalActions` 的 deepcopy 与网络前向；先跑 5 分钟估算 steps/sec，再折算 M1-M3 挂机时长。

## 5. 三阶段课程（plan/phase-1 §2，不过门槛不进下一阶段）

| 里程碑 | 命令 | 门槛（自动判定） |
|---|---|---|
| **M1** 学会买分 | `dqn -o random --test-opponent random --total-steps 200000 -s <seed> -w runs/m1` | vs random 平均终局得分 **> 10**（≥100 局） |
| **M2** 碾压 random | `dqn -o random --test-opponent minimax --total-steps 300000 -s <seed> -w runs/m2` | vs random 胜率 **> 90%**（20 局滑窗稳定） |
| **M3** 超越 minimax | `dqn -o minimax --test-opponent minimax --total-steps 500000 -s <seed> -w runs/m3` | vs minimax 100 局 **≥ 55%** |

自博弈（M3 进阶混池）：`dqn -o itself --test-opponent minimax ...`——对手与己方共享网络。

门槛判定脚本（以 M3 为例，用与训练同源的 `evaluate`）：

```python
# eval_gate.py —— 放仓库根目录运行：.venv/bin/python eval_gate.py runs/m3/<时间戳>/dqn_model.pth minimax 100
import sys
from pathlib import Path
import splendor.splendor.gym  # noqa: F401  注册 gym 环境
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.training import evaluate
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.minmax import myAgent as MinMaxAgent

checkpoint, opponent, n_games = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
factory = {"random": RandomAgent, "minimax": MinMaxAgent}[opponent]
q_net = load_saved_dqn(checkpoint)
stats = evaluate(q_net, make_opponents=lambda: [factory(0)], n_games=n_games)
print(stats)
```

## 6. 监控与产物解读

`stats.csv` 逐 episode 字段：`step, episode, epsilon, loss, q_mean, train_score, eval_wr, eval_avg_score`。

**健康信号**：loss 在有限区间震荡后缓降；`q_mean` 缓升不爆；`eval_wr` 渐进上升。
**病态信号**（对照 DQN_GUIDE §9 十二陷阱）：
- `loss` 持续发散 / `q_mean` 指数上涨 → lr 降半或检查 win_bonus；
- `eval_wr` 突然跳 100% → 大概率 bug（评估泄漏训练状态、奖励重复计入），不是突破；
- 长期不涨 → 九成在奖励设计与输入归一化，而非算法。

checkpoint 加载与本地观战（需先把 checkpoint 放到包内默认路径，`DQNAgent` 从这里加载）：

```bash
cp runs/m3/<时间戳>/dqn_model.pth \
   src/splendor/agents/our_agents/dqn/dqn_model.pth
# 评测 5 局（-t 文本模式；去掉 -q 可观战 GUI）
splendor -a splendor.agents.our_agents.dqn.dqn_agent,splendor.agents.generic.random \
         --agent_names=dqn,random -t -m 5
```

> 注意：改 checkpoint 后重复 `splendor -a ...dqn_agent` 无需重装（editable 安装读包内文件）。

## 7. 验收清单（plan/phase-1 §4-§5 摘要）

- [ ] M1/M2/M3 门槛逐项达标（`eval_gate.py` 数字留档）
- [ ] **3-seed 方差**：门槛至少在 2/3 个 seed 上达成，单 seed 达标不得宣布通过
- [ ] 训练曲线人工审查（无发散/无突跳）
- [ ] 对局观战 5 局定性合理（按颜色囤宝石、优先买分卡、会预留）——赢但行为荒谬不算通过
- [ ] `load_saved_dqn` 加载后胜率与训练末期一致（防 running stats 保存/加载 bug）
- [ ] 反直觉行为登记（供给盲区等），作为 P4-T4.2 的触发证据

## 8. 常见问题

| 症状 | 处置 |
|---|---|
| `KeyError` / `illegal!` 崩溃 | 某处绕过掩码采样——检查自定义代码是否用了 `randint(3510)`；必须 `np.random.choice(np.flatnonzero(mask))` |
| 训练正常、评测很菜 | 先确认评测路径与训练路径用同一特征管线（`dqn_agent` 已保证）；再查 checkpoint 是否放到包内默认路径 |
| 同 seed 结果不同 | 检查是否所有入口都过了 `train()`（它统一设 seed 三件套）；第三方脚本自行 reset 时需自备三件套 |
| 想断点续训 | 当前 `train()` 不支持热续训（无 saved-weights 参数）；中断后从头重训，或以 M1 产物为新会话的初始化需自行扩展 `train()`（参照 PPO 的 transfer-learning 参数） |
| 网页经验回流 | 见 [WEB_DEPLOYMENT_GUIDE.md](./WEB_DEPLOYMENT_GUIDE.md) §5（`collect_from_browser`），off-policy 红利：网页对局可直接混入本地 replay |
