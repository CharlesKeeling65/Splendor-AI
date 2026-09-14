"""Unified paired-seat evaluation and teacher-trajectory collection."""

import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.splendor.gym.envs.utils import create_legal_actions_mask
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.splendor.utils import LimitRoundsGameRule

from .policies import CandidateSpec
from .protocol import isolated_seed
from .runner import TeacherDecisionError, select_action
from .scenario import ScenarioV1, rule_from_scenario
from .trajectory import TrajectoryDataset, TrajectoryStep

DecisionProbeSink = Callable[["DecisionProbe"], None]


@dataclass(frozen=True)
class DecisionProbe:
    """A copied state at which a teacher was queried."""

    state: SplendorState
    rule: SplendorGameRule
    actions: list[ActionType]
    seed: int
    seat: int
    ply: int
    action_index: int | None


def _outcome(score: float, rival_score: float) -> int:
    return int(score > rival_score) - int(score < rival_score)


def _cost_summary(latencies: list[float]) -> dict[str, float]:
    """Summarize decision cost without inventing a value for missing queries."""
    if not latencies:
        return {
            "action_seconds_mean": 0.0,
            "action_seconds_p95": 0.0,
            "action_seconds_max": 0.0,
        }
    return {
        "action_seconds_mean": float(np.mean(latencies)),
        "action_seconds_p95": float(np.quantile(latencies, 0.95)),
        "action_seconds_max": float(max(latencies)),
    }


def play_game(  # noqa: PLR0913,PLR0915 - one game owns all audit counters
    candidate: CandidateSpec,
    opponent: CandidateSpec,
    *,
    seed: int,
    seat: int,
    feature_version: str | None = None,
    collect_trajectory: bool = False,
    probe_sink: DecisionProbeSink | None = None,
    scenario: ScenarioV1 | None = None,
) -> tuple[dict[str, Any], list[TrajectoryStep]]:
    """Play one raw-engine game and retain failures in its denominator.

    ``scenario`` is an explicit opt-in.  The historical integer-seed path is
    unchanged; when a snapshot is supplied the seed can affect legacy agent
    randomness but can no longer redeal the opening board.
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    schema = feature_version or candidate.feature_version
    target = candidate.build(seat)
    rival = opponent.build(1 - seat)
    target_latencies: list[float] = []
    rival_latencies: list[float] = []
    trajectory: list[TrajectoryStep] = []
    target_queries = 0
    rival_queries = 0
    target_illegal = 0
    rival_illegal = 0
    target_search_nodes = 0
    rival_search_nodes = 0
    failure: dict[str, str] | None = None
    started = time.perf_counter()

    with isolated_seed(seed):
        rule = (
            LimitRoundsGameRule(2) if scenario is None else rule_from_scenario(scenario)
        )
        while not rule.gameEnds():
            state = rule.current_game_state
            turn = rule.current_agent_index
            legal_actions = rule.getLegalActions(state, turn)
            is_target = turn == seat
            if is_target:
                target_queries += 1
                observation = (
                    extract_observation(state, seat, schema)
                    if collect_trajectory
                    else None
                )
                legal_mask = (
                    create_legal_actions_mask(legal_actions, state, seat).astype(
                        np.uint8
                    )
                    if collect_trajectory
                    else None
                )
                score_before = state.agents[seat].score
                if probe_sink is not None:
                    probe_sink(
                        DecisionProbe(
                            state=deepcopy(state),
                            rule=deepcopy(rule),
                            actions=deepcopy(legal_actions),
                            seed=seed,
                            seat=seat,
                            ply=rule.action_counter,
                            action_index=None,
                        )
                    )
                try:
                    decision = select_action(target, legal_actions, state, rule)
                except TeacherDecisionError as exc:
                    target_illegal += int(exc.illegal)
                    target_latencies.append(exc.elapsed_seconds)
                    target_search_nodes += exc.search_nodes
                    failure = {
                        "side": "candidate",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                target_latencies.append(decision.elapsed_seconds)
                target_search_nodes += decision.search_nodes
                rule.update(decision.action)
                if collect_trajectory:
                    assert observation is not None
                    assert legal_mask is not None
                    trajectory.append(
                        TrajectoryStep(
                            observation=observation,
                            legal_mask=legal_mask,
                            action_index=decision.action_index,
                            deal_seed=(
                                scenario.source_seed if scenario is not None else seed
                            ),
                            seat=seat,
                            ply=rule.action_counter - 1,
                            step_in_episode=len(trajectory),
                            reward_delta=float(
                                rule.current_game_state.agents[seat].score
                                - score_before
                            ),
                            terminal=rule.gameEnds(),
                        )
                    )
            else:
                rival_queries += 1
                try:
                    decision = select_action(rival, legal_actions, state, rule)
                except TeacherDecisionError as exc:
                    rival_illegal += int(exc.illegal)
                    rival_latencies.append(exc.elapsed_seconds)
                    rival_search_nodes += exc.search_nodes
                    failure = {
                        "side": "opponent",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                rival_latencies.append(decision.elapsed_seconds)
                rival_search_nodes += decision.search_nodes
                rule.update(decision.action)

    final_state = rule.current_game_state
    if rule.gameEnds() and trajectory:
        # The opponent may end the game before the focal agent gets another
        # turn; keep the boundary on the last focal row rather than dropping
        # it from the dataset.
        trajectory[-1] = replace(trajectory[-1], terminal=True)
    score = float(rule.calScore(final_state, seat))
    rival_score = float(rule.calScore(final_state, 1 - seat))
    record: dict[str, Any] = {
        "seed": seed,
        "seat": seat,
        "candidate": candidate.name,
        "opponent": opponent.name,
        "status": "failed" if failure else "completed",
        "failure": failure,
        "outcome": _outcome(score, rival_score) if failure is None else None,
        "score": score,
        "rival_score": rival_score,
        "plies": rule.action_counter,
        "candidate_queries": target_queries,
        "opponent_queries": rival_queries,
        "candidate_illegal_actions": target_illegal,
        "opponent_illegal_actions": rival_illegal,
        "candidate_search_nodes": target_search_nodes,
        "opponent_search_nodes": rival_search_nodes,
        "elapsed_seconds": time.perf_counter() - started,
        **{
            f"candidate_{key}": value
            for key, value in _cost_summary(target_latencies).items()
        },
        "opponent_cost": _cost_summary(rival_latencies),
    }
    if scenario is not None:
        record.update(
            {
                "scenario_id": scenario.scenario_id,
                "canonical_state_sha256": scenario.canonical_state_sha256,
                "scenario_source_segment": scenario.source_segment,
                "scenario_source_seed": scenario.source_seed,
                "legacy_seed_argument": seed,
            }
        )
    # Keep the probe API useful while avoiding a second source of truth for
    # labels: callers can infer the label from the copied legal actions only
    # when they need to run a dedicated audit.
    return record, trajectory


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute W/D/L with both scheduled and completed denominators."""
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    completed = [record for record in records if record["status"] == "completed"]
    wins = sum(record.get("outcome") == 1 for record in completed)
    draws = sum(record.get("outcome") == 0 for record in completed)
    losses = sum(record.get("outcome") == -1 for record in completed)
    failures = len(records) - len(completed)
    summary: dict[str, Any] = {
        "games": len(records),
        "completed_games": len(completed),
        "failed_games": failures,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(records),
        "completed_win_rate": wins / len(completed) if completed else None,
        "mean_score": float(np.mean([record["score"] for record in completed]))
        if completed
        else None,
        "candidate_illegal_actions": sum(
            int(record["candidate_illegal_actions"]) for record in records
        ),
        "opponent_illegal_actions": sum(
            int(record["opponent_illegal_actions"]) for record in records
        ),
        "candidate_queries": sum(
            int(record["candidate_queries"]) for record in records
        ),
        "candidate_search_nodes": sum(
            int(record["candidate_search_nodes"]) for record in records
        ),
        "records": records,
    }
    latencies = [
        float(record["candidate_action_seconds_mean"])
        for record in completed
        if record["candidate_queries"]
    ]
    summary["candidate_action_seconds_mean_per_game"] = (
        float(np.mean(latencies)) if latencies else 0.0
    )
    return summary


def evaluate_candidate(
    candidate: CandidateSpec,
    opponent: CandidateSpec,
    seeds: Sequence[int],
    *,
    seats: Sequence[int] = (0, 1),
) -> dict[str, Any]:
    """Evaluate one candidate against one fixed opponent on paired games."""
    seed_list = [int(seed) for seed in seeds]
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("evaluation seeds must be nonempty and unique")
    if list(seats) != [0, 1]:
        raise ValueError("evaluation requires both seats [0, 1]")
    records = [
        play_game(candidate, opponent, seed=seed, seat=seat)[0]
        for seed in seed_list
        for seat in seats
    ]
    result = summarize_records(records)
    result.update(
        {
            "candidate": candidate.name,
            "candidate_snapshot": candidate.snapshot,
            "opponent": opponent.name,
            "feature_version": candidate.feature_version,
            "seeds": seed_list,
            "seats": list(seats),
        }
    )
    return result


def evaluate_matrix(
    candidates: Sequence[CandidateSpec],
    opponents: Sequence[CandidateSpec],
    seeds: Sequence[int],
    *,
    seats: Sequence[int] = (0, 1),
) -> dict[str, dict[str, Any]]:
    """Run a declared candidate x fixed-opponent matrix."""
    return {
        candidate.name: {
            opponent.name: evaluate_candidate(candidate, opponent, seeds, seats=seats)
            for opponent in opponents
        }
        for candidate in candidates
    }


def collect_teacher_dataset(  # noqa: PLR0913 - provenance fields are explicit
    candidate: CandidateSpec,
    opponent: CandidateSpec,
    seeds: Sequence[int],
    output: Path,
    *,
    feature_version: str,
    seats: Sequence[int] = (0, 1),
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Collect complete focal-agent trajectories and save failures separately."""
    seed_list = [int(seed) for seed in seeds]
    if list(seats) != [0, 1]:
        raise ValueError("trajectory collection requires both seats [0, 1]")
    records: list[dict[str, Any]] = []
    steps: list[TrajectoryStep] = []
    for seed in seed_list:
        for seat in seats:
            record, game_steps = play_game(
                candidate,
                opponent,
                seed=seed,
                seat=seat,
                feature_version=feature_version,
                collect_trajectory=True,
            )
            records.append(record)
            steps.extend(game_steps)
    metadata: dict[str, Any] = {
        "teacher": candidate.name,
        "teacher_snapshot": candidate.snapshot,
        "opponent": opponent.name,
        "feature_version": feature_version,
        "source_manifest": source_manifest,
        "seed_groups": {"collection": seed_list},
        "seats": list(seats),
        "games": len(records),
        "completed_games": sum(record["status"] == "completed" for record in records),
        "failed_games": sum(record["status"] == "failed" for record in records),
        "game_records": records,
    }
    dataset = TrajectoryDataset.from_steps(steps, metadata=metadata)
    dataset.save(output)
    return {
        "dataset": str(output),
        "dataset_hash": dataset.content_hash(),
        "samples": dataset.size,
        "summary": summarize_records(records),
        "metadata": dataset.metadata,
    }


def collect_decision_probes(
    candidate: CandidateSpec,
    opponent: CandidateSpec,
    seeds: Sequence[int],
    *,
    seats: Sequence[int] = (0, 1),
    limit: int | None = None,
) -> list[DecisionProbe]:
    """Collect copied candidate states for the information-set perturbation audit."""
    probes: list[DecisionProbe] = []

    def sink(probe: DecisionProbe) -> None:
        if limit is None or len(probes) < limit:
            probes.append(probe)

    for seed in seeds:
        for seat in seats:
            if limit is not None and len(probes) >= limit:
                return probes
            play_game(
                candidate,
                opponent,
                seed=int(seed),
                seat=int(seat),
                probe_sink=sink,
            )
    return probes
