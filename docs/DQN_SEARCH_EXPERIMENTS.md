# DQN staged experiment protocol

Baseline implementation: `2e3451a`; n-step correctness fix: `78caed0`.
The user authorized implementation of all five analysis stages on 2026-09-07.

## Changes

- Fold every terminal suffix and use `gamma ** n_step` for nonterminal TD
  bootstrap. A nonterminal window always has exactly n entries; terminal
  windows do not bootstrap. The six-tensor replay interface stays compatible.
- `public-v2` = original 265 dimensions + supply (6), both resource/discount/
  reserved-count panels (24), three noble cost vectors (15), seat and
  score-threshold flag (2): **312**. No hidden rival faces or future deck order.
  Two-player only. Browser construction now preserves public rival panels;
  `play-web` derives feature version from checkpoint. PPO/GA/minimax unchanged.
- The frozen-normalization ablation estimates mean/variance from all stored
  warmup observations (variance floor 1), copies to target, then freezes.
  Original continuously updated EMA remains the `corrected` comparison.
- The population samples a complete-game policy: random/minimax/heuristic/
  history buckets equally once history is available. History holds at most
  four frozen snapshots, uniformly sampled within its bucket. Record actual
  game counts; do not infer actual proportions from configured weights.
- Optional auxiliary policy/outcome heads. All completed episode states get
  outcome targets; only searched states get visit-distribution policy targets.
  Search every 8th post-warmup decision by default; auxiliary update every 4th
  step with weight 0.2. No use of cumulative reward Q as a zero-sum value.
- PUCT uses bounded depth and fresh sampled hidden states each simulation.
  Rival reservation identities and unseen deck order are jointly randomized
  conditional on publicly visible reservation tiers and public purchase history.
  This is an approximate information-set search, **not exact AlphaZero**.
  It ignores memory of previously seen reserved faces; information-set strategy
  fusion remains a limitation. Search is local-only: browser pseudo cards do
  not provide purchase identities needed for reconstructing unseen cards.

## Reproduce

Use the established Pascal environment, not the absent `.venv`:

```bash
OMP_NUM_THREADS=1 .venv-p5000/bin/python -m pytest tests/ -q
make parity PYTHON=.venv-p5000/bin/python
OMP_NUM_THREADS=1 .venv-p5000/bin/python -m splendor.agents.our_agents.dqn.experiment \
  --output runs/staged-10k --steps 10000 --eval-every 5000 \
  --seeds 42 1234 2024 --test-deals 10
```

Each of `corrected`, `frozen`, `public`, `population`, `search` starts from
scratch. Defaults otherwise: Adam lr 5e-5, gamma .99, tau .001, batch 128,
warmup 1000, buffer 30000, epsilon 1→.05 over 8000 steps, terminal bonus 10,
snapshot interval 2000, 16 search simulations. All are saved in manifest.json.
No automatic deployment or overwrite of existing model directories.

Validation: deals 700000–700004, each in both seats, against isolated minimax.
Select by greedy validation win rate (ties keep earliest checkpoint). All jobs
finish before the test set is opened. Test: deals 900000 onward, each in both
seats, against random/heuristic/minimax; search variant also tested using MCTS.
Opponents receive cloned states/rules to prevent minimax rollback artifacts
from mutating real games. Archived models are re-evaluated using this protocol,
but their original training budgets differ and they are not equal-budget controls.

Per-run `training.csv`, `status.json`, validation JSON and checkpoints are
written live. Suite `results.json` records every seed/seat result, score, action
latency (mean/p95/max), elapsed time and descriptive Wilson intervals. The
`suite-status.json` completion marker is written only after all tests finish.

## Interpretation boundaries

10k-step runs are learning pilots, not the 200k–500k curriculum acceptance.
Independent training seeds are the replication unit. Deal/seat results are
paired and shared across models; never treat their concatenation as independent
Bernoulli samples. Wilson intervals are descriptive only. Compare paired score
differences per deal and variation across training seeds. Search consumes extra
compute: equal environment steps do not imply equal wall-time or decision-time
budgets; report both and do not call a slower search gain compute-matched.
Do not tune on these test seeds; reserve a new range for follow-up experiments.

Strict reproducibility on CUDA is not asserted: RNGs are seeded and evaluation
restores them, but deterministic-kernel enforcement is not enabled. Checkpoints
store model/config, not optimizer/replay; interrupted checkpoints support
inference, not exact training continuation.
