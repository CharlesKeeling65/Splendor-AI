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
