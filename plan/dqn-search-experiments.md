# DQN correctness → public features → population → search

Approved by the user on 2026-09-07. Baseline is commit `2e3451a`;
the worktree was clean before implementation (no empty snapshot commit).
This explicitly reopens the public-feature part of the P4 deferral. It does
not authorize live website games or change PPO/GA/minimax rules/policies.

1. Correct n-step bootstrap discount and flush all terminal origins; hand tests.
2. Independent validation/test deal seeds with paired seats; isolate RNGs;
   frozen-after-warmup normalization ablation against continuous EMA.
3. Versioned DQN-only public features (supply, both resource panels, noble
   requirements, seat/endgame flag), old checkpoint compatibility and ≥1000
   real distinct state projections. Do not use hidden reserved identities.
4. Add heuristic and bounded frozen historical opponent population; retain
   opponent for a complete episode and log actual selections.
5. Optional policy/value heads trained from episode outcomes and search visits;
   information-constrained, sampled-deck PUCT search using cloned engine rules.
   No negating score-return Q values, no mutation of live game/RNG state.
6. Run predeclared, equal-step ablations with independent training seeds.
   Report raw outcomes, uncertainty, steps and wall time. Pilot runs demonstrate
   execution, not convergence. Do not automatically replace deployed weights.

Each stage requires tests before training. A final held-out set must not be
used to select checkpoints or tune parameters. Retain old artifacts unchanged.
Search is an experimental local path, not a claim of full AlphaZero or a
browser-ready hidden-card belief model. Production search requires public
history reconstruction; the feature-based greedy path is browser-compatible.
