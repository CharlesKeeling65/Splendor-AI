"""DAgger state aggregation with explicit teacher-query accounting."""

import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.splendor.gym.envs.utils import create_legal_actions_mask
from splendor.splendor.utils import LimitRoundsGameRule

from .policies import CandidateSpec
from .protocol import isolated_seed
from .runner import TeacherDecisionError, select_action
from .trajectory import (
    TrajectoryDataset,
    TrajectoryStep,
    concatenate_datasets,
)


def _cost_summary(latencies: list[float]) -> dict[str, float]:
    """Summarize query latency without inventing costs for missing calls."""
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


def _outcome(score: float, rival_score: float) -> int:
    """Return the engine score ordering used by all evaluation reports."""
    return int(score > rival_score) - int(score < rival_score)


def summarize_dagger_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize DAgger games while keeping failed games in the denominator."""
    if not records:
        raise ValueError("cannot summarize an empty DAgger run")
    completed = [record for record in records if record["status"] == "completed"]
    wins = sum(record.get("outcome") == 1 for record in completed)
    draws = sum(record.get("outcome") == 0 for record in completed)
    losses = sum(record.get("outcome") == -1 for record in completed)
    teacher_latencies = [
        float(record["teacher_action_seconds_mean"])
        for record in records
        if record["teacher_queries"]
    ]
    return {
        "games": len(records),
        "completed_games": len(completed),
        "failed_games": len(records) - len(completed),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(records),
        "completed_win_rate": wins / len(completed) if completed else None,
        "student_queries": sum(int(record["student_queries"]) for record in records),
        "teacher_queries": sum(int(record["teacher_queries"]) for record in records),
        "teacher_action_mismatches": sum(
            int(record["teacher_action_mismatches"]) for record in records
        ),
        "student_illegal_actions": sum(
            int(record["student_illegal_actions"]) for record in records
        ),
        "teacher_illegal_actions": sum(
            int(record["teacher_illegal_actions"]) for record in records
        ),
        "opponent_illegal_actions": sum(
            int(record["opponent_illegal_actions"]) for record in records
        ),
        "teacher_search_nodes": sum(
            int(record["teacher_search_nodes"]) for record in records
        ),
        "teacher_action_seconds_mean_per_game": (
            float(np.mean(teacher_latencies)) if teacher_latencies else 0.0
        ),
        "records": records,
    }


def play_dagger_game(  # noqa: PLR0913,PLR0915 - one game owns all audit counters
    teacher: CandidateSpec,
    student: CandidateSpec,
    opponent: CandidateSpec,
    *,
    seed: int,
    seat: int,
    feature_version: str,
) -> tuple[dict[str, Any], list[TrajectoryStep]]:
    """Run one student-visited game and label every focal state with teacher action.

    The raw engine is advanced with the student's action.  The teacher is
    queried on a defensive copy of the same state and never gets to advance
    the game.  Thus the collected labels describe student-state visitation,
    not a disguised teacher rollout.
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    student_agent = student.build(seat)
    teacher_agent = teacher.build(seat)
    rival_agent = opponent.build(1 - seat)
    steps: list[TrajectoryStep] = []
    step_details: list[dict[str, Any]] = []
    student_latencies: list[float] = []
    teacher_latencies: list[float] = []
    rival_latencies: list[float] = []
    student_queries = 0
    teacher_queries = 0
    rival_queries = 0
    student_illegal = 0
    teacher_illegal = 0
    rival_illegal = 0
    teacher_search_nodes = 0
    student_search_nodes = 0
    rival_search_nodes = 0
    failure: dict[str, str] | None = None
    started = time.perf_counter()

    with isolated_seed(seed):
        rule = LimitRoundsGameRule(2)
        while not rule.gameEnds():
            state = rule.current_game_state
            turn = rule.current_agent_index
            legal_actions = rule.getLegalActions(state, turn)
            if turn == seat:
                student_queries += 1
                teacher_queries += 1
                observation = extract_observation(state, seat, feature_version)
                legal_mask = create_legal_actions_mask(
                    legal_actions, state, seat
                ).astype(np.uint8)
                score_before = state.agents[seat].score
                try:
                    student_decision = select_action(
                        student_agent, legal_actions, state, rule
                    )
                except TeacherDecisionError as exc:
                    student_illegal += int(exc.illegal)
                    student_latencies.append(exc.elapsed_seconds)
                    student_search_nodes += exc.search_nodes
                    failure = {
                        "side": "student",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                student_latencies.append(student_decision.elapsed_seconds)
                student_search_nodes += student_decision.search_nodes
                try:
                    teacher_decision = select_action(
                        teacher_agent, legal_actions, state, rule
                    )
                except TeacherDecisionError as exc:
                    teacher_illegal += int(exc.illegal)
                    teacher_latencies.append(exc.elapsed_seconds)
                    teacher_search_nodes += exc.search_nodes
                    failure = {
                        "side": "teacher",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                teacher_latencies.append(teacher_decision.elapsed_seconds)
                teacher_search_nodes += teacher_decision.search_nodes
                rule.update(student_decision.action)
                steps.append(
                    TrajectoryStep(
                        observation=observation,
                        legal_mask=legal_mask,
                        action_index=teacher_decision.action_index,
                        deal_seed=seed,
                        seat=seat,
                        ply=rule.action_counter - 1,
                        step_in_episode=len(steps),
                        reward_delta=float(
                            rule.current_game_state.agents[seat].score - score_before
                        ),
                        terminal=rule.gameEnds(),
                    )
                )
                step_details.append(
                    {
                        "ply": rule.action_counter - 1,
                        "student_action_index": student_decision.action_index,
                        "teacher_action_index": teacher_decision.action_index,
                        "student_action_matches_teacher": (
                            student_decision.action_index
                            == teacher_decision.action_index
                        ),
                    }
                )
            else:
                rival_queries += 1
                try:
                    rival_decision = select_action(
                        rival_agent, legal_actions, state, rule
                    )
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
                rival_latencies.append(rival_decision.elapsed_seconds)
                rival_search_nodes += rival_decision.search_nodes
                rule.update(rival_decision.action)

    if rule.gameEnds() and steps:
        steps[-1] = replace(steps[-1], terminal=True)
    final_state = rule.current_game_state
    score = float(rule.calScore(final_state, seat))
    rival_score = float(rule.calScore(final_state, 1 - seat))
    record: dict[str, Any] = {
        "seed": seed,
        "seat": seat,
        "teacher": teacher.name,
        "teacher_snapshot": teacher.snapshot,
        "student": student.name,
        "student_snapshot": student.snapshot,
        "opponent": opponent.name,
        "status": "failed" if failure else "completed",
        "failure": failure,
        "outcome": _outcome(score, rival_score) if failure is None else None,
        "score": score,
        "rival_score": rival_score,
        "plies": rule.action_counter,
        "student_queries": student_queries,
        "teacher_queries": teacher_queries,
        "opponent_queries": rival_queries,
        "student_illegal_actions": student_illegal,
        "teacher_illegal_actions": teacher_illegal,
        "opponent_illegal_actions": rival_illegal,
        "student_search_nodes": student_search_nodes,
        "teacher_search_nodes": teacher_search_nodes,
        "opponent_search_nodes": rival_search_nodes,
        "teacher_action_mismatches": sum(
            not detail["student_action_matches_teacher"] for detail in step_details
        ),
        "teacher_action_match_rate": (
            float(
                np.mean(
                    [
                        detail["student_action_matches_teacher"]
                        for detail in step_details
                    ]
                )
            )
            if step_details
            else 0.0
        ),
        "student_action_seconds": _cost_summary(student_latencies),
        "teacher_action_seconds": _cost_summary(teacher_latencies),
        "opponent_action_seconds": _cost_summary(rival_latencies),
        "student_action_seconds_mean": _cost_summary(student_latencies)[
            "action_seconds_mean"
        ],
        "teacher_action_seconds_mean": _cost_summary(teacher_latencies)[
            "action_seconds_mean"
        ],
        "step_details": step_details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return record, steps


def collect_dagger_dataset(  # noqa: PLR0913 - provenance fields are explicit
    teacher: CandidateSpec,
    student: CandidateSpec,
    opponent: CandidateSpec,
    seeds: Sequence[int],
    output: Path,
    *,
    feature_version: str,
    round_index: int,
    seats: Sequence[int] = (0, 1),
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Collect one DAgger round and persist a fully auditable dataset."""
    seed_list = [int(seed) for seed in seeds]
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("DAgger seeds must be nonempty and unique")
    if list(seats) != [0, 1]:
        raise ValueError("DAgger collection requires both seats [0, 1]")
    records: list[dict[str, Any]] = []
    steps: list[TrajectoryStep] = []
    for seed in seed_list:
        for seat in seats:
            record, game_steps = play_dagger_game(
                teacher,
                student,
                opponent,
                seed=seed,
                seat=seat,
                feature_version=feature_version,
            )
            records.append(record)
            steps.extend(game_steps)
    metadata: dict[str, Any] = {
        "feature_version": feature_version,
        "dagger_round": round_index,
        "teacher": teacher.name,
        "teacher_snapshot": teacher.snapshot,
        "student": student.name,
        "student_snapshot": student.snapshot,
        "opponent": opponent.name,
        "source_manifest": source_manifest,
        "seed_groups": {"dagger_training": seed_list},
        "seats": list(seats),
        "games": len(records),
        "completed_games": sum(record["status"] == "completed" for record in records),
        "failed_games": sum(record["status"] == "failed" for record in records),
        "game_records": records,
        "teacher_queries": sum(int(record["teacher_queries"]) for record in records),
        "student_action_mismatches": sum(
            int(record["teacher_action_mismatches"]) for record in records
        ),
    }
    dataset = TrajectoryDataset.from_steps(steps, metadata=metadata)
    dataset.save(output)
    return {
        "dataset": str(output),
        "dataset_hash": dataset.content_hash(),
        "samples": dataset.size,
        "summary": summarize_dagger_records(records),
        "metadata": dataset.metadata,
    }


def aggregate_dagger_datasets(
    datasets: Sequence[Path],
    output: Path,
    *,
    round_index: int,
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Aggregate base/round datasets without dropping round provenance."""
    paths = [Path(path) for path in datasets]
    if not paths:
        raise ValueError("DAgger aggregation needs at least one dataset")
    loaded = [TrajectoryDataset.load(path) for path in paths]
    aggregate = concatenate_datasets(
        loaded,
        metadata={
            "dagger_round": round_index,
            "source_manifest": source_manifest,
            "source_datasets": [str(path) for path in paths],
            "aggregation": "base data plus student-visited teacher labels",
        },
    )
    aggregate.save(output)
    records = [
        record
        for dataset in loaded
        for record in dataset.metadata.get("game_records", [])
    ]
    return {
        "dataset": str(output),
        "dataset_hash": aggregate.content_hash(),
        "samples": aggregate.size,
        "source_datasets": [str(path) for path in paths],
        "source_hashes": [dataset.content_hash() for dataset in loaded],
        "round_index": round_index,
        "summary": summarize_dagger_records(records) if records else None,
        "metadata": aggregate.metadata,
    }
