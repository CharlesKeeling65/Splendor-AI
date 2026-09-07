"""MCTS gate experiments for an existing auxiliary-head DQN snapshot.

The gate deliberately keeps search separate from policy training.  It compares
the same frozen weights with no search and several sampled-hidden-state PUCT
budgets, then reports value calibration and an information-set audit.  A gate
result is evidence for or against a later search experiment; it is not a new
deployable checkpoint.
"""

import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, override

import numpy as np
import torch

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.search import search_policy
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .evaluation import DecisionProbe, evaluate_matrix, play_game
from .policies import CandidateSpec

DeviceName = Literal["cpu", "cuda", "mps"]
SearchStat = float | int
SearchStatsSink = Callable[[dict[str, SearchStat]], None]


def _resolve_device(device_name: DeviceName) -> torch.device:
    """Use the requested accelerator only when this host actually provides it."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SearchDQNAgent(Agent):
    """Frozen DQN policy with an optional sampled-hidden-state search layer."""

    def __init__(
        self,
        _id: int,
        net: QNetwork,
        simulations: int,
        rng_seed: int,
        max_depth: int = 24,
        stats_sink: SearchStatsSink | None = None,
    ) -> None:
        super().__init__(_id)
        if simulations < 0:
            raise ValueError("simulations must be non-negative")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if simulations and not net.auxiliary_heads:
            raise ValueError("MCTS requires an auxiliary-head DQN checkpoint")
        self.net = net.eval()
        self.device = next(net.parameters()).device
        self.simulations = simulations
        self.max_depth = max_depth
        self._rng = np.random.default_rng(rng_seed + 1009 * _id)
        self._stats_sink = stats_sink

    def _emit_stats(self, stats: dict[str, SearchStat]) -> None:
        """Send a copy so a caller cannot mutate the agent's last query."""
        if self._stats_sink is not None:
            self._stats_sink(dict(stats))

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Select a legal action and record search work for this query."""
        began = time.perf_counter()
        mapping = create_action_mapping(actions, game_state, self.id)
        mask = create_legal_actions_mask(actions, game_state, self.id).astype(
            np.float32
        )
        observation = extract_observation(game_state, self.id, self.net.feature_version)
        obs_tensor = torch.from_numpy(observation).to(self.device)
        mask_tensor = torch.from_numpy(mask).to(self.device)
        stats: dict[str, SearchStat] = {
            "simulations": 0,
            "tree_nodes": 0,
            "search_fallback": 0,
        }
        if self.simulations == 0:
            action_index = self.net.act(obs_tensor, mask_tensor)
        else:
            pi = search_policy(
                self.net,
                game_rule,
                self.simulations,
                self._rng,
                max_depth=self.max_depth,
                stats=stats,
            )
            legal_indices = np.flatnonzero(mask).astype(np.int64)
            probabilities = pi[legal_indices]
            if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
                action_index = self.net.act(obs_tensor, mask_tensor)
                stats["search_fallback"] = 1
            else:
                action_index = int(legal_indices[int(np.argmax(probabilities))])
        stats["action_seconds"] = time.perf_counter() - began
        stats["action_index"] = action_index
        self._emit_stats(stats)
        return mapping[action_index]


def build_mcts_candidate(  # noqa: PLR0913 - gate factory options are explicit
    checkpoint: Path,
    *,
    simulations: int,
    device_name: DeviceName = "cpu",
    rng_seed: int = 1234,
    max_depth: int = 24,
    stats_sink: SearchStatsSink | None = None,
) -> CandidateSpec:
    """Build a frozen no-search or MCTS candidate from one DQN snapshot."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DQN snapshot does not exist: {checkpoint}")
    template = load_saved_dqn(checkpoint)
    if simulations and not template.auxiliary_heads:
        raise ValueError("MCTS gate requires a checkpoint with auxiliary heads")
    device = _resolve_device(device_name)
    frozen = deepcopy(template).to(device).eval()
    frozen.requires_grad_(False)
    name = "dqn-no-search" if simulations == 0 else f"dqn-mcts-{simulations}"

    def factory(agent_id: int) -> Agent:
        return SearchDQNAgent(
            agent_id,
            deepcopy(frozen).to(device).eval(),
            simulations,
            rng_seed,
            max_depth,
            stats_sink,
        )

    return CandidateSpec(
        name=name,
        role="teacher_candidate",
        factory=factory,
        feature_version=str(template.feature_version),
        snapshot=str(checkpoint),
    )


def summarize_search_cost(records: Sequence[dict[str, SearchStat]]) -> dict[str, Any]:
    """Summarize per-query search counters without hiding fallback queries."""
    if not records:
        return {
            "queries": 0,
            "search_queries": 0,
            "fallback_queries": 0,
            "simulations_total": 0,
            "tree_nodes_total": 0,
            "tree_nodes_mean": 0.0,
            "action_seconds_mean": 0.0,
            "action_seconds_p95": 0.0,
            "action_seconds_max": 0.0,
        }
    elapsed = np.asarray(
        [float(record.get("action_seconds", 0.0)) for record in records],
        dtype=np.float64,
    )
    return {
        "queries": len(records),
        "search_queries": sum(int(record.get("simulations", 0)) > 0 for record in records),
        "fallback_queries": sum(int(record.get("search_fallback", 0)) for record in records),
        "simulations_total": sum(int(record.get("simulations", 0)) for record in records),
        "tree_nodes_total": sum(int(record.get("tree_nodes", 0)) for record in records),
        "tree_nodes_mean": float(
            np.mean([int(record.get("tree_nodes", 0)) for record in records])
        ),
        "action_seconds_mean": float(np.mean(elapsed)),
        "action_seconds_p95": float(np.quantile(elapsed, 0.95)),
        "action_seconds_max": float(np.max(elapsed)),
    }


def _aggregate_gate_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate opponent rows while retaining their individual results."""
    rows = list(results.values())
    games = sum(int(row["games"]) for row in rows)
    wins = sum(int(row["wins"]) for row in rows)
    draws = sum(int(row["draws"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    failed = sum(int(row["failed_games"]) for row in rows)
    illegal = sum(int(row["candidate_illegal_actions"]) for row in rows)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "failed_games": failed,
        "candidate_illegal_actions": illegal,
        "win_rate": wins / games if games else None,
        "candidate_action_seconds_mean_per_game": float(
            np.mean(
                [
                    float(row["candidate_action_seconds_mean_per_game"])
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
    }


def run_search_gate(  # noqa: PLR0913 - gate protocol fields are explicit
    checkpoint: Path,
    seeds: Sequence[int],
    opponents: Sequence[CandidateSpec],
    simulations: Sequence[int],
    *,
    device_name: DeviceName = "cpu",
    rng_seed: int = 1234,
    max_depth: int = 24,
) -> dict[str, dict[str, Any]]:
    """Compare one checkpoint at each declared search budget."""
    budgets = sorted({int(value) for value in simulations})
    if not budgets or budgets[0] < 0:
        raise ValueError("search budgets must contain non-negative integers")
    gate: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        cost_records: list[dict[str, SearchStat]] = []
        candidate = build_mcts_candidate(
            checkpoint,
            simulations=budget,
            device_name=device_name,
            rng_seed=rng_seed,
            max_depth=max_depth,
            stats_sink=cost_records.append,
        )
        results = evaluate_matrix([candidate], opponents, seeds)[candidate.name]
        gate[str(budget)] = {
            "simulations": budget,
            "checkpoint": str(checkpoint),
            "results": results,
            "aggregate": _aggregate_gate_results(results),
            "search_cost": summarize_search_cost(cost_records),
            "teacher_queries": 0,
        }
    return gate


def calibrate_outcome_value(
    checkpoint: Path,
    opponent: CandidateSpec,
    seeds: Sequence[int],
    *,
    device_name: DeviceName = "cpu",
) -> dict[str, Any]:
    """Compare the auxiliary outcome head with signed terminal calScore."""
    template = load_saved_dqn(checkpoint)
    if not template.auxiliary_heads:
        raise ValueError("value calibration requires an auxiliary-head checkpoint")
    device = _resolve_device(device_name)
    net = deepcopy(template).to(device).eval()
    net.requires_grad_(False)
    candidate = build_mcts_candidate(
        checkpoint,
        simulations=0,
        device_name=device_name,
        rng_seed=9871,
    )
    predictions: list[float] = []
    targets: list[int] = []
    completed_games = 0
    failed_games = 0

    for seed in [int(value) for value in seeds]:
        for seat in (0, 1):
            probe_values: list[float] = []

            def sink(
                probe: DecisionProbe,
                target_values: list[float] = probe_values,
            ) -> None:
                observation = extract_observation(
                    probe.state, probe.seat, net.feature_version
                )
                mask = torch.from_numpy(
                    create_legal_actions_mask(
                        probe.actions, probe.state, probe.seat
                    ).astype(np.float32)
                ).to(device)
                with torch.no_grad():
                    _, value = net.policy_value(
                        torch.from_numpy(observation).to(device), mask
                    )
                target_values.append(float(value.item()))

            record, _ = play_game(
                candidate,
                opponent,
                seed=seed,
                seat=seat,
                probe_sink=sink,
            )
            if record["status"] != "completed" or record["outcome"] is None:
                failed_games += 1
                continue
            completed_games += 1
            predictions.extend(probe_values)
            targets.extend([int(record["outcome"])] * len(probe_values))

    if not predictions:
        raise RuntimeError("value calibration produced no completed decision states")
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    correlation: float | None = None
    if float(np.std(predicted)) > 0 and float(np.std(target)) > 0:
        correlation = float(np.corrcoef(predicted, target)[0, 1])
    signed_accuracy = float(np.mean(np.sign(predicted) == target))
    return {
        "checkpoint": str(checkpoint),
        "feature_version": str(net.feature_version),
        "games": len(seeds) * 2,
        "completed_games": completed_games,
        "failed_games": failed_games,
        "decision_samples": len(predictions),
        "target_definition": "sign(calScore(focal) - calScore(rival))",
        "prediction_range": [float(np.min(predicted)), float(np.max(predicted))],
        "mean_prediction": float(np.mean(predicted)),
        "mean_target": float(np.mean(target)),
        "mse": float(np.mean((predicted - target) ** 2)),
        "mae": float(np.mean(np.abs(predicted - target))),
        "signed_direction_accuracy": signed_accuracy,
        "pearson_correlation": correlation,
        "target_counts": {
            str(value): int(np.sum(target == value)) for value in (-1.0, 0.0, 1.0)
        },
    }


def assess_search_gate(  # noqa: PLR0913 - gate thresholds are reviewable inputs
    gate: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    information_audit: dict[str, Any],
    *,
    min_search_wins_delta: int = 1,
    max_latency_multiplier: float = 50.0,
    min_signed_direction_accuracy: float = 0.5,
) -> dict[str, Any]:
    """Apply the declared qualitative/numeric gate criteria explicitly."""
    if "0" not in gate:
        raise ValueError("gate comparison requires a zero-search baseline")
    baseline = gate["0"]["aggregate"]
    search_rows = [
        value for key, value in gate.items() if int(key) > 0
    ]
    if not search_rows:
        raise ValueError("gate comparison requires at least one positive budget")
    best = max(search_rows, key=lambda value: value["aggregate"]["wins"])
    best_aggregate = best["aggregate"]
    baseline_latency = float(
        gate["0"]["search_cost"]["action_seconds_mean"]
    )
    best_latency = float(best["search_cost"]["action_seconds_mean"])
    latency_multiplier = best_latency / baseline_latency if baseline_latency else None
    criteria = {
        "execution_clean": all(
            int(value["aggregate"]["failed_games"]) == 0
            and int(value["aggregate"]["candidate_illegal_actions"]) == 0
            and int(value["search_cost"]["fallback_queries"]) == 0
            for value in gate.values()
        ),
        "search_improves_wins": (
            int(best_aggregate["wins"]) - int(baseline["wins"])
            >= min_search_wins_delta
        ),
        "latency_within_budget": (
            latency_multiplier is not None
            and latency_multiplier <= max_latency_multiplier
        ),
        "value_directional_semantics": (
            float(calibration["signed_direction_accuracy"])
            >= min_signed_direction_accuracy
        ),
        "information_audit_pass": information_audit["status"] == "pass",
    }
    return {
        "status": "pass" if all(criteria.values()) else "blocked",
        "criteria": criteria,
        "baseline_budget": 0,
        "best_search_budget": int(best["simulations"]),
        "baseline_wins": int(baseline["wins"]),
        "best_search_wins": int(best_aggregate["wins"]),
        "wins_delta": int(best_aggregate["wins"]) - int(baseline["wins"]),
        "latency_multiplier": latency_multiplier,
        "thresholds": {
            "min_search_wins_delta": min_search_wins_delta,
            "max_latency_multiplier": max_latency_multiplier,
            "min_signed_direction_accuracy": min_signed_direction_accuracy,
        },
    }
