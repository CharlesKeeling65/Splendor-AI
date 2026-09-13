"""Roadmap F1: calibration of the critic's win prediction.

Plays the checkpoint against fixed opponents, records the value head's
estimate at every focal decision, and scores it against the game's final
outcome (+1 win / 0 loss, ties excluded).  Reports AUC, Brier score, expected
calibration error (ECE) and a reliability table so the caller can decide
whether the value function is usable as a search prior.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import ACTION_DIM
from splendor.agents.our_agents.policy_imitation.bc_training import DeviceName
from splendor.agents.our_agents.policy_imitation.policies import (
    build_fixed_baseline,
)
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    PolicyValueNetwork,
    load_ppo_checkpoint,
)
from splendor.agents.our_agents.policy_imitation.runner import select_action
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorState
from splendor.splendor.types import ActionType
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent

#: Acceptance threshold from the roadmap (search-prior prerequisite).
USABLE_AUC_THRESHOLD = 0.75
#: Reliability bins over the clipped [0, 1] prediction range.
CALIBRATION_BINS = 10
PAIR_SEATS = 2


def _value_head(
    model: PolicyValueNetwork,
    state: SplendorState,
    seat: int,
    actions: list[ActionType],
    device: torch.device,
) -> tuple[int, float]:
    """Greedy action index and the value estimate for one focal decision."""
    obs = extract_observation(state, seat, model.feature_version)
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    mapping = create_action_mapping(actions, state, seat)
    mask[np.asarray(sorted(mapping), dtype=np.int64)] = 1
    with torch.no_grad():
        _logits, value = model(
            torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device)
        )
    return int(np.argmax(mask)), float(value.item())


def play_labeled_game(
    model: PolicyValueNetwork,
    rival: Agent,
    seat: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]] | None:
    """Play one game; return (value, outcome) samples or None on failure."""
    random.seed(seed + seat * 7919)
    np.random.seed(seed + seat * 7919)
    rule = LimitRoundsGameRule(PAIR_SEATS)
    values: list[float] = []
    while not rule.gameEnds():
        state = rule.current_game_state
        turn = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(state, turn)
        try:
            if turn == seat:
                action_index, value = _value_head(model, state, seat, legal, device)
                values.append(value)
                rule.update(create_action_mapping(legal, state, seat)[action_index])
            else:
                decision = select_action(rival, legal, state, rule)
                rule.update(decision.action)
        except Exception:
            return None
    own = float(rule.calScore(rule.current_game_state, seat))
    best_rival = max(
        (
            float(rule.calScore(rule.current_game_state, agent.id))
            for agent in rule.current_game_state.agents
            if agent.id != seat
        ),
        default=own,
    )
    outcome = int(own > best_rival) - int(own < best_rival)
    if outcome == 0 or not values:
        return None
    return [{"value": value, "outcome": float(outcome > 0)} for value in values]


def auc_roc(values: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-aware."""
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_labels = labels[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = float(sorted_labels.sum())
    negatives = len(sorted_labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = float(ranks[sorted_labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def reliability_curve(
    predictions: np.ndarray, labels: np.ndarray
) -> tuple[list[dict[str, Any]], float]:
    """Reliability table and expected calibration error."""
    bins = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    table: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(CALIBRATION_BINS):
        lo, hi = float(bins[index]), float(bins[index + 1])
        in_bin = (
            (predictions >= lo) & (predictions < hi)
            if index < CALIBRATION_BINS - 1
            else (predictions >= lo) & (predictions <= hi)
        )
        if in_bin.sum() == 0:
            table.append({"bin": [lo, hi], "count": 0})
            continue
        mean_prediction = float(predictions[in_bin].mean())
        empirical = float(labels[in_bin].mean())
        ece += float(in_bin.sum()) / len(labels) * abs(mean_prediction - empirical)
        table.append(
            {
                "bin": [lo, hi],
                "count": int(in_bin.sum()),
                "mean_prediction": mean_prediction,
                "empirical_win_rate": empirical,
            }
        )
    return table, ece


def calibrate(
    checkpoint: Path,
    *,
    opponents: Sequence[str],
    games_per_opponent: int,
    seeds: Sequence[int],
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Play paired games and score the value head against final outcomes."""
    resolved: DeviceName = (
        "cuda"
        if device_name == "cuda"
        else "mps" if device_name == "mps" else "cpu"
    )
    model = load_ppo_checkpoint(checkpoint, device_name=resolved)
    assert isinstance(model, PolicyValueNetwork) and isinstance(model, nn.Module)
    torch_device = next(model.parameters()).device
    samples: list[dict[str, Any]] = []
    for opponent_name in opponents:
        template = build_fixed_baseline(opponent_name, device_name="cpu")
        for seed in seeds[:games_per_opponent]:
            for seat in range(PAIR_SEATS):
                game_samples = play_labeled_game(
                    model, template.build(1 - seat), seat, seed, torch_device
                )
                if game_samples is None:
                    continue
                for sample in game_samples:
                    sample.update(opponent=opponent_name, seed=seed, seat=seat)
                samples.extend(game_samples)
    values = np.asarray([s["value"] for s in samples], dtype=np.float64)
    labels = np.asarray([s["outcome"] for s in samples], dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < PAIR_SEATS:
        return {
            "samples": len(labels),
            "auc": None,
            "note": "insufficient label spread",
        }
    auc = auc_roc(values, labels)
    predictions = np.clip(values, 0.0, 1.0)
    table, ece = reliability_curve(predictions, labels)
    return {
        "samples": len(labels),
        "auc": auc,
        "brier": float(np.mean((predictions - labels) ** 2)),
        "ece": ece,
        "reliability": table,
        "mean_value": float(values.mean()),
        "base_rate": float(labels.mean()),
        "usable_as_search_prior": bool(auc >= USABLE_AUC_THRESHOLD),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponents",
        default="ga,heuristic,minimax",
        help="comma list of fixed baselines",
    )
    parser.add_argument("--games-per-opponent", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=854_010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/f1-calibration"))
    args = parser.parse_args(argv)
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    seeds = list(range(args.seed_start, args.seed_start + args.games_per_opponent))
    report = calibrate(
        args.checkpoint,
        opponents=opponents,
        games_per_opponent=args.games_per_opponent,
        seeds=seeds,
        device_name=args.device,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["opponents"] = opponents
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "f1_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    verdict = (
        "USABLE as search prior"
        if report.get("usable_as_search_prior")
        else "NOT usable as search prior"
    )
    print(
        f"F1 calibration ({verdict}): AUC={report.get('auc')}, "
        f"ECE={report.get('ece')}, n={report.get('samples')} "
        f"-> {args.output / 'f1_report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
