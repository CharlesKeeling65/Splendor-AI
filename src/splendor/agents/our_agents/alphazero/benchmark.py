"""Roadmap Z0 micro-benchmarks: the compute ground truth for Z1-Z3 budgets.

Measures, on seeded random-play positions at several game phases:

1. ``deepcopy(rule)`` cost - what the F2 harness pays per *simulation*;
2. in-place apply+undo cost (Transactor) plus its exactness oracle
   (``state_fingerprint`` equality), including how often noble-awarding
   actions (the engine-unrestored branch) are exercised;
3. end-to-end ``az_search`` wall clock with the uniform evaluator at the Z1
   operating point (sims/trees knobs) - the per-move cost of the search;
4. network forward latency for the Z2 evaluator (CPU vs CUDA, batch 1/64)
   and per-node feature extraction / legal-action table costs.

Usage::

    PYTHONHASHSEED=0 python -m splendor.agents.our_agents.alphazero.benchmark \
        --output runs/z0-benchmark [--sims 100 --trees 4 --device cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from splendor.agents.our_agents.alphazero.evaluator import UniformEvaluator
from splendor.agents.our_agents.alphazero.mcts import az_search
from splendor.agents.our_agents.alphazero.state_utils import (
    Transactor,
    legal_action_table,
    state_fingerprint,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.utils import LimitRoundsGameRule


def sample_positions(
    checkpoints: list[int], seed: int
) -> dict[int, SplendorGameRule]:
    """Random-play positions at the given move counts (seeded, 2 players)."""
    random.seed(seed)
    np.random.seed(seed)
    rule = LimitRoundsGameRule(2)
    wanted = sorted(set(checkpoints))
    positions: dict[int, SplendorGameRule] = {}
    move = 0
    while move <= max(wanted) and not rule.gameEnds():
        if move in set(wanted):
            positions[move] = deepcopy(rule)
        state = rule.current_game_state
        seat = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(state, seat)
        rule.update(legal[int(np.random.randint(len(legal)))])
        move += 1
    return positions


def bench_deepcopy(rule: SplendorGameRule, repeats: int) -> float:
    """Mean deepcopy wall clock in microseconds."""
    start = time.perf_counter()
    for _ in range(repeats):
        deepcopy(rule)
    return (time.perf_counter() - start) / repeats * 1e6


def bench_fingerprint(rule: SplendorGameRule, repeats: int) -> float:
    """Mean state_fingerprint wall clock in microseconds."""
    state = rule.current_game_state
    start = time.perf_counter()
    for _ in range(repeats):
        state_fingerprint(state, include_last_action=False)
    return (time.perf_counter() - start) / repeats * 1e6


def bench_apply_undo(
    rule: SplendorGameRule, repeats: int, actions_per_position: int
) -> dict[str, Any]:
    """In-place apply+undo cost, exactness and noble-branch coverage.

    Runs on a sacrificial deepcopy of ``rule``; every cycle is verified with
    the fingerprint oracle (both ``last_action`` variants - the engine-restored
    fields must match exactly, the unrestored ones prove the snapshot works).
    """
    work = deepcopy(rule)
    transactor = Transactor()
    state = work.current_game_state
    seat = work.getCurrentAgentIndex()
    indices, actions = legal_action_table(work, seat)
    order = np.random.default_rng(0).permutation(len(actions))[
        : min(actions_per_position, len(actions))
    ]
    noble_cycles = 0
    mismatches = 0
    before = state_fingerprint(state)
    elapsed = 0.0
    for _repeat in range(repeats):
        for pick in order:
            action = actions[int(pick)]
            noble_cycles += int(action.get("noble") is not None)
            cycle_start = time.perf_counter()
            snapshot = transactor.apply(work, action, seat)
            transactor.undo(work, action, seat, snapshot)
            elapsed += time.perf_counter() - cycle_start
            if state_fingerprint(state) != before:
                mismatches += 1
    cycles = repeats * len(order)
    return {
        "cycles": cycles,
        "apply_undo_us_mean": elapsed / cycles * 1e6 if cycles else 0.0,
        "fingerprint_us": bench_fingerprint(rule, repeats),
        "noble_cycles": noble_cycles,
        "fingerprint_mismatches": mismatches,
        "legal_actions": len(actions),
    }


def bench_search(
    rule: SplendorGameRule, sims: int, trees: int, seed: int
) -> dict[str, Any]:
    """End-to-end uniform-evaluator search cost at one position."""
    result = az_search(
        rule,
        UniformEvaluator(),
        np.random.default_rng(seed),
        simulations=sims,
        n_trees=trees,
    )
    stats = dict(result.stats)
    stats["moves_per_s"] = float(1.0 / max(stats["elapsed_s"], 1e-9))
    return stats


def bench_network(rule: SplendorGameRule, device: str) -> dict[str, Any] | None:
    """Single/batched policy_value forward latency on the AZ operating net."""
    try:
        import torch

        from splendor.agents.our_agents.dqn.network import QNetwork
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        return None
    if device == "cuda" and not torch.cuda.is_available():
        return None
    net = QNetwork(
        input_dim=312,
        feature_version="public-v2",
        auxiliary_heads=True,
        use_input_norm=False,
    ).to(device)
    from splendor.agents.our_agents.dqn.features import extract_observation

    state = rule.current_game_state
    seat = rule.getCurrentAgentIndex()
    start = time.perf_counter()
    for _ in range(50):
        extract_observation(state, seat, "public-v2")
    feature_us = (time.perf_counter() - start) / 50 * 1e6
    indices, _actions = legal_action_table(rule, seat)
    mask = np.zeros(3510, dtype=np.float32)
    mask[indices] = 1
    obs = extract_observation(state, seat, "public-v2")
    single = torch.from_numpy(obs).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    with torch.no_grad():
        for _ in range(5):
            net.policy_value(single, mask_t)  # warmup
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            net.policy_value(single, mask_t)
        if device == "cuda":
            torch.cuda.synchronize()
        batch1_us = (time.perf_counter() - start) / 50 * 1e6
        batch_obs = single.unsqueeze(0).repeat(64, 1)
        batch_mask = mask_t.unsqueeze(0).repeat(64, 1)
        start = time.perf_counter()
        for _ in range(10):
            net.policy_value(batch_obs, batch_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        batch64_us = (time.perf_counter() - start) / 10 * 1e6
    return {
        "device": device,
        "feature_extraction_us": feature_us,
        "policy_value_batch1_us": batch1_us,
        "policy_value_batch64_total_us": batch64_us,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/z0-benchmark"))
    parser.add_argument("--seed", type=int, default=830_000)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 12, 30])
    parser.add_argument("--sims", type=int, nargs="+", default=[32, 100])
    parser.add_argument("--trees", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--devices", type=str, nargs="+", default=["cpu", "cuda"])
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "seed": args.seed,
    }
    positions = sample_positions(args.checkpoints, args.seed)
    report["positions"] = {}
    for move_count, rule in sorted(positions.items()):
        entry: dict[str, Any] = {
            "deepcopy_us": bench_deepcopy(rule, args.repeats),
            "apply_undo": bench_apply_undo(rule, args.repeats, 8),
        }
        for sims in args.sims:
            entry[f"search_sims{sims}_trees{args.trees}"] = bench_search(
                rule, sims, args.trees, args.seed
            )
        report["positions"][f"move_{move_count}"] = entry
    report["network"] = {}
    mid = positions[max(positions)]
    for device in args.devices:
        net_report = bench_network(mid, device)
        if net_report is not None:
            report["network"][device] = net_report
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "z0_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Z0 report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
