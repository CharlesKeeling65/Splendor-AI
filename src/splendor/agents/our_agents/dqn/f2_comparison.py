"""Roadmap F2 harness: single-sample PUCT vs multi-determinized trees.

Equal-wall-clock paired comparison on sampled engine positions.  The harness
never mutates the engine or the global RNG state of the caller beyond the
declared seed; it is an offline analysis tool and produces a JSON report.

Prerequisites: a DQN checkpoint with trained auxiliary policy/value heads
(``auxiliary_heads=True``) - pass it via ``--teacher``.  Positions come from
random-play openings against the teacher's own move choices, so both search
modes see an identical position stream.

Usage::

    PYTHONHASHSEED=0 python -m splendor.agents.our_agents.dqn.f2_comparison \
        --teacher runs/f2-teacher/models/dqn_model.pth \
        --positions 40 --budget 3.0 --trees 4 8 16 --output runs/f2-comparison
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splendor.agents.our_agents.dqn.dqn_utils import load_dqn_template
from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import ACTION_DIM, QNetwork
from splendor.agents.our_agents.dqn.search import (
    multi_tree_search_policy,
    search_policy,
)
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule
from splendor.splendor.utils import LimitRoundsGameRule


def collect_positions(
    teacher: QNetwork,
    n_positions: int,
    seed: int,
    *,
    warmup_moves: int = 6,
) -> list[tuple[SplendorGameRule, int]]:
    """Sample mid-game two-player positions by teacher-greedy play."""
    positions: list[tuple[SplendorGameRule, int]] = []
    random.seed(seed)
    np.random.seed(seed)
    while len(positions) < n_positions:
        rule = LimitRoundsGameRule(2)
        for _move in range(warmup_moves):
            if rule.gameEnds():
                break
            state = rule.current_game_state
            seat = rule.getCurrentAgentIndex()
            legal = rule.getLegalActions(state, seat)
            mapping = create_action_mapping(legal, state, seat)
            indices = sorted(mapping)
            obs = teacher_forward(teacher, state, seat, indices)
            choice = int(np.argmax(obs))
            rule.update(mapping[indices[choice]])
        if rule.gameEnds():
            continue
        positions.append((rule, rule.getCurrentAgentIndex()))
    return positions


def teacher_forward(
    teacher: QNetwork,
    state: SplendorGameRule,
    seat: int,
    indices: list[int],
) -> np.ndarray:
    """Teacher logits over the sorted legal-action indices (numpy array)."""
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    mask[indices] = 1
    obs = extract_observation(state.current_game_state, seat, teacher.feature_version)
    with torch.no_grad():
        logits, _value = teacher.policy_value(
            torch.from_numpy(obs), torch.from_numpy(mask)
        )
    return logits[0, indices].cpu().numpy()


def timed_search(  # noqa: PLR0913 - one keyword per search knob
    mode: str,
    teacher: QNetwork,
    rule: SplendorGameRule,
    rng: np.random.Generator,
    *,
    wall_clock_budget_s: float,
    n_trees: int,
    allocation: str,
) -> dict[str, Any]:
    """Run one search mode until the budget expires; return the last policy."""
    stats: dict[str, float | int] = {}
    start = time.perf_counter()
    calls = 0
    policy = None
    while time.perf_counter() - start < wall_clock_budget_s:
        simulations = 4 * (calls + 1)  # escalate until the budget is spent
        if mode == "single":
            policy = search_policy(teacher, rule, simulations, rng, stats=stats)
        else:
            policy = multi_tree_search_policy(
                teacher,
                rule,
                simulations,
                rng,
                n_trees=n_trees,
                allocation=allocation,
                stats=stats,
            )
        calls += 1
    elapsed = time.perf_counter() - start
    return {
        "mode": mode,
        "n_trees": n_trees if mode != "single" else 1,
        "allocation": allocation if mode != "single" else "n/a",
        "calls": calls,
        "last_simulations": stats.get("simulations", 0),
        "tree_nodes": stats.get("tree_nodes", 0),
        "elapsed_s": elapsed,
        "policy": None if policy is None else policy.tolist(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument(
        "--budget", type=float, default=3.0, help="seconds per (position, mode)"
    )
    parser.add_argument("--trees", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=829_000)
    parser.add_argument("--output", type=Path, default=Path("runs/f2-comparison"))
    args = parser.parse_args(argv)

    teacher = load_dqn_template(args.teacher)
    if not teacher.auxiliary_heads:
        raise ValueError("teacher checkpoint has no auxiliary policy/value heads")
    positions = collect_positions(teacher, args.positions, args.seed)
    report: list[dict[str, Any]] = []
    for index, (rule, _seat) in enumerate(positions):
        for mode, n_trees, allocation in [
            ("single", 1, "n/a"),
            *[("multi", trees, "uniform") for trees in args.trees],
            ("multi", args.trees[0], "priority"),
        ]:
            rng = np.random.default_rng(args.seed + 1000 * index)
            record = timed_search(
                mode,
                teacher,
                rule,
                rng,
                wall_clock_budget_s=args.budget,
                n_trees=n_trees,
                allocation=allocation,
            )
            record["position"] = index
            del record["policy"]  # keep the report compact
            report.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "f2_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"F2 comparison written to {args.output / 'f2_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
