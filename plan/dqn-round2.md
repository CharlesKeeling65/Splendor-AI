# DQN round 2: isolate useful public information and improve early learning

User requested further improvement after the negative frozen-normalization
ablations. Baseline commit `14c4b73`. Old artifacts and deployed weights stay intact.

Predeclared design (2026-09-07):

| Variant | Features | Normalization | Training-only guidance |
|---|---|---|---|
| corrected | v1 | continuous EMA, lagged target stats | none |
| public-ema | public-v2 | same EMA | none |
| public-sync | public-v2 | EMA, exact online/target stats synchronization | none |
| public-demo | public-v2 | same EMA as public-ema | annealed heuristic exploration + legal margin |

These are BRANCHES, not a stack. public-sync and public-demo each compare to
public-ema. Learned weights still use the ordinary soft target update; the
sync option only copies preprocessing statistics. Guidance combines with TD
in one optimizer update, does not alter rewards, and disappears after step16000.
Margin .8, initial loss weight .25, exploration guidance probability .5,
both scaled linearly to zero; at evaluation only greedy learned Q is consulted.
The heuristic is a fallible public-state teacher, not an optimal action label.

12 independent runs, seeds42/1234/2024, each20000 steps. Shared lr5e-5,
tau.001, gamma.99, n-step3, batch128, warmup1000, buffer30000, epsilon
1→.05 over8000 steps. Validate every5000 steps on 710000–710019, both seats,
against minimax AND heuristic; choose highest equal-weight mean win rate,
ties keep earliest. 80 validation games/checkpoint, versus10 in round1.
Test only after all training completes: 930000–930024 in both seats, against
random/heuristic/minimax. Re-evaluate old mixed20k and all three round1 corrected
checkpoints as historical controls. No tuning on test seeds910000/920000/930000.
If validation detects a defect, preserve failed artifacts and rerun under a
new manifest; do not quietly exclude failed games. Three replicas and50 games
per opponent remain a limited pilot, not proof of full curriculum convergence.

Quality gates: hand-tested legal margin and gradients, guidance termination,
normalization synchronization without weight copying, all new variants smoke
train, old checkpoint compatibility and existing offline test suite. Report
actual selected steps, per-seed wins, opponent counts, resource use and caveats.
