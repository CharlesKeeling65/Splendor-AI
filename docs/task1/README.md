# Task 1 implementation ledger

This directory contains small, reviewable protocol artifacts for
[`plan/task-1-2p-improvement-and-seed-protocol.md`](../../plan/task-1-2p-improvement-and-seed-protocol.md).
It does not contain checkpoints or scenario banks.

## T1.0 baseline freeze

- [`T1.0_BASELINE_AUDIT.json`](T1.0_BASELINE_AUDIT.json) is rebuilt from the
  frozen C2-R2 checkpoint, its BC/DAgger parent data, checked-in opponent
  implementations, and all raw C4-R2 rows.
- [`T1.0_BASELINE_MANIFEST_V2.json`](T1.0_BASELINE_MANIFEST_V2.json) is a
  schema-v2, `proposed` evidence-only manifest. It consumed no validation or
  sealed-test scenarios and is not approval to train.
- The manifest was created from clean commit `c87ce25` using
  `.venv-p5000/bin/python`. CUDA was unavailable at capture time, so its
  runtime correctly records a CPU resolution and no GPU work was started.

Rebuild the audit (read-only source inputs):

```bash
.venv/bin/python -m \
  splendor.agents.our_agents.policy_imitation.baseline_audit \
  --repo . --output /tmp/task1-baseline-audit.json
diff -u docs/task1/T1.0_BASELINE_AUDIT.json /tmp/task1-baseline-audit.json
```

The C4-R2 report is historical description only. For each `ppo-best`
opponent matchup it contains 150 scheduled rows from 50 source seeds and 100
unique `(seed, focal-seat)` cells; 50 cells are repeated once. Its legacy
Wilson intervals must not be used for task-1 decisions.

## T1.1 RNG and paired training

[`T1.1_RNG_PROTOCOL.md`](T1.1_RNG_PROTOCOL.md) freezes the event-keyed RNG,
paired schedule, explicit weighted-pool, spawn-worker, deterministic runtime,
and manifest-binding contracts. All new behavior is opt-in through a
`FormalTrainingSpec`; legacy PPO and opponent APIs retain their previous path.

This is protocol evidence, not a training result. Scenario snapshots and
cross-runner deal parity begin in T1.2.

## T1.2 ScenarioV1 and banks

[`T1.2_SCENARIO_BANK.md`](T1.2_SCENARIO_BANK.md) records the immutable opening
schema, registry hashes, seven disjoint split ranges, opening-only strata,
content-addressed bank format, sealed consumption gate, and Game/raw/PPO
parity contract. Only CI fixtures were produced; the formal sealed bank remains
unmaterialized until the T1.4 crossed pilot and fixed-N approval gate.

## T1.3 paired evaluation and statistics

[`T1.3_PAIRED_STATISTICS.md`](T1.3_PAIRED_STATISTICS.md) records the strict
candidate × opponent × ScenarioV1 × two-seat evaluator, append-only episode
schema, failure-preserving denominator, joint scenario-cluster and nested
replicate→scenario bootstrap, separate scenario/model-replicate power designs,
checkpoint-to-live-policy attestation, and manifest-bound immutable
`statistics.json` contract. Power artifacts distinguish prospective pilot data
from a separately approved fixed-N run, whose manifest must bind the exact
pilot artifact before outcomes are read. Only CI/synthetic fixtures were
executed. The crossed pilot and its final fixed N remain behind the T1.4
approval gate; no formal or sealed bank was consumed.

## T1.4 seed-roll preflight

[`T1.4_SEED_ROLL.md`](T1.4_SEED_ROLL.md) specifies the one-shot
`splendor-seed-roll/1` authority, keyed finite-population SRSWOR allocation,
five pre-reserved replicate blocks, exact seat balance, root-derived model
initialization, streaming ScenarioV1 bank, and `paired-training-v2` runtime
bindings. The optimizer accepts only the explicit `task1-formal-5x32000`
profile; smaller rolls remain visibly tagged CI fixtures. Pilot and reserved
replicates, per-treatment inputs, and one-shot output paths are manifest-bound.
The formal gate additionally binds the exact root-derived scenario/seat order;
confirmatory reserve jobs require a separately frozen activation file tied to
the completed pilot manifest and its evidence, not a caller-supplied hash.
The production root was explicitly approved and drawn once for
`task1-t14-crossed-pilot-20260915`; its compact public record is
[`T1.4_PRODUCTION_SEED_ROLL.json`](T1.4_PRODUCTION_SEED_ROLL.json). No formal bank
was materialized as part of the root draw itself. After the later explicit
implementation/training authorization, the non-sealed 96,000-row pilot
`train-schedule` bank, the non-sealed 10-row checkpoint-selector bank, and the
non-sealed 200-row validation-A and 100-row stress baseline banks were
materialized and audited. The 1,800-game new-protocol frozen baseline is now
complete; its W/D/L, fixed-model variance decomposition, hashes, and recovery
record are in [`T1.4_BASELINE_RESULTS.md`](T1.4_BASELINE_RESULTS.md). No
validation-B, sealed, or reserved-replicate bank has been read or created, and
no GPU training job has yet run.

[`T1.4_PILOT_ORCHESTRATION.md`](T1.4_PILOT_ORCHESTRATION.md) documents the
fail-closed P5000/tmux production entry for the approved six-job pilot. It pins
the exact O/O_bridge recipes, 3×2 job matrix, five-member training pool,
ScenarioV1 validation selector, P5000 runtime, resource watchdog, one-shot
output reservations, checkpoint evidence, and lifecycle transitions. The
baseline replay is complete. The final production manifest and actual CUDA tmux
launch remain separate recorded steps.

## T1.4 episodic reward contract

[`T1.4_REWARD_CONTRACT.md`](T1.4_REWARD_CONTRACT.md) separates the historical
`O_bridge/terminal-biased-potential-v1` semantics from the explicit
`O/safe-potential-v1` contract. The safe contract fixes the absorbing terminal
potential at zero for normal, deadlock, round-limit, and truncation exits while
leaving the `+10 / 0 / -10` terminal utility unchanged. Legacy constructors and
CLI entries retain their old non-zero-terminal behavior.
