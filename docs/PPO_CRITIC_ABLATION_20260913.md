# PPO critic 修复消融报告（roadmap C1，2026-09-13）

> 配对消融协议：5 分支 × 3 seed（42/1234/2024），全部走 stabilization 驱动
> （frozen fixed 变体、8 updates × 16 局、同训练种子计划 820000–821127、
> 验证 822xxx 段）。决策指标 = 最后 3 次 update 的 explained variance 均值
> （EV），验证胜局（60 局）仅作噪声参考。产物：`runs/c1-critic-ablation/<branch>/`。

## 分支

| 分支 | 改动（其余与 frozen 配置逐字节相同） |
|---|---|
| `base` | 无（复刻 2026-09-08 稳定化配置：lr 1e-4、warmup 2、value coeff 0.5、共享 Adam） |
| `critic-lr5` | `--critic-learning-rate 5e-4`（value head 独立 Adam 参数组，×5） |
| `warmup5` | `--critic-warmup-epochs 5` |
| `value1` | `--value-coefficient 1.0` |
| `critichid` | `--critic-hidden-dim 128`（共享 trunk 上的 critic 私有隐层） |

## 结果

| 分支 | EV(last 3) seed42 | seed1234 | seed2024 | 3-seed 均值 | 验证胜局均值 /60 |
|---|---|---|---|---|---|
| base | 0.043 | 0.120 | 0.202 | 0.122 | 28.3 |
| critic-lr5 | 0.078 | 0.160 | 0.210 | **0.149** | 28.3 |
| warmup5 | 0.067 | 0.079 | 0.207 | 0.117 | 30.0 |
| value1 | 0.082 | 0.183 | 0.228 | **0.164** | 27.0 |
| critichid | 0.090 | 0.068 | 0.093 | 0.084 | 28.0 |

逐 update EV 曲线（举例 seed2024）：base 0.087→0.212，value1 0.094→0.042
（中段冲到 0.421），critichid 0.093 尾段（全程最低）。

## 结论与 C2 决策

1. **value 系数 0.5→1.0 是最强单因素**（EV +35%）；critic lr ×5 次之（+23%）；
   两者方向一致（提高 critic 学习压力），且不损害验证胜局（差异 ≤ 噪声）。
2. **critichid 有害**（EV −31%），从 C2 排除。
3. **warmup5 无 EV 收益**，维持 warmup 2。
4. **G2 的 EV ≥ 0.5 目标在 8-update 试点尺度上不可达**（天花板 ~0.23），
   与基线历史（EV 0.03–0.21）一致；该门槛顺延至 C2 规模化（500 updates）
   训练曲线上判定，若届时仍未达标将如实记录未达成原因。
5. **C2 采纳配置**：`--value-coefficient 1.0 --critic-learning-rate 5e-4`
   （两个 EV 正向单因素的组合；组合本身未单独消融，由 C2 自身的验证
   选模与独立测试兜底）。

## 成本

每分支 3 并行 job ≈ 7.2 分钟（P5000，CUDA，workers=3）；5 分支总计 ≈ 36 分钟。
