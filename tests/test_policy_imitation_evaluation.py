"""Tests for paired evaluation, information audits, and trajectory replay."""

from pathlib import Path
from typing import override

import numpy as np

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.policy_imitation.evaluation import (
    collect_teacher_dataset,
    evaluate_candidate,
    summarize_records,
)
from splendor.agents.our_agents.policy_imitation.information_audit import (
    audit_candidate_information,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.trajectory import (
    TrajectoryDataset,
    split_by_seed,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


def _random_candidate(name: str = "random") -> CandidateSpec:
    return CandidateSpec(name, "teacher_candidate", RandomAgent)


def test_paired_evaluation_keeps_both_seats_and_costs() -> None:
    result = evaluate_candidate(_random_candidate(), _random_candidate("rival"), [800001])

    assert result["games"] == 2
    assert result["completed_games"] == 2
    assert result["failed_games"] == 0
    assert result["wins"] + result["draws"] + result["losses"] == 2
    assert result["candidate_queries"] > 0
    assert result["candidate_illegal_actions"] == 0
    assert result["candidate_action_seconds_mean_per_game"] >= 0
    assert {record["seat"] for record in result["records"]} == {0, 1}


def test_failed_games_remain_in_win_rate_denominator() -> None:
    records = [
        {"status": "completed", "outcome": 1, "score": 15, "candidate_illegal_actions": 0, "opponent_illegal_actions": 0, "candidate_queries": 1, "candidate_search_nodes": 0, "candidate_action_seconds_mean": 0.1},
        {"status": "failed", "outcome": None, "score": 0, "candidate_illegal_actions": 1, "opponent_illegal_actions": 0, "candidate_queries": 1, "candidate_search_nodes": 0, "candidate_action_seconds_mean": 0.1},
    ]

    result = summarize_records(records)

    assert result["games"] == 2
    assert result["completed_games"] == 1
    assert result["failed_games"] == 1
    assert result["win_rate"] == 0.5
    assert result["completed_win_rate"] == 1.0


def test_teacher_dataset_round_trips_and_splits_by_seed(tmp_path: Path) -> None:
    path = tmp_path / "teacher.npz"
    result = collect_teacher_dataset(
        _random_candidate(),
        _random_candidate("rival"),
        [800002],
        path,
        feature_version="v1",
        source_manifest="manifest.json",
    )
    dataset = TrajectoryDataset.load(path)
    splits = split_by_seed(
        dataset,
        {"train": [800002], "validation": [], "test": []},
    )

    assert result["samples"] == dataset.size > 0
    assert dataset.metadata["source_manifest"] == "manifest.json"
    assert dataset.legal_masks.dtype == np.uint8
    assert np.all(dataset.legal_masks[np.arange(dataset.size), dataset.actions] == 1)
    assert splits["train"].size == dataset.size
    assert splits["validation"].size == 0
    assert splits["test"].size == 0


class _DeckSensitiveAgent(Agent):
    """Agent used only to ensure the audit reports a potential hidden dependency."""

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_rule
        first_code = game_state.board.decks[0][0].code
        return actions[0] if first_code < "m" else actions[-1]


def test_candidate_spec_build_repairs_agent_id() -> None:
    candidate = CandidateSpec(
        "custom",
        "teacher_candidate",
        lambda _agent_id: _DeckSensitiveAgent(99),
    )

    agent = candidate.build(1)

    assert agent.id == 1


def test_random_teacher_passes_projection_audit() -> None:
    result = audit_candidate_information(
        _random_candidate(),
        _random_candidate("rival"),
        [800003],
        max_states=2,
    )

    assert result["states_checked"] == 2
    assert result["status"] == "pass"
    assert result["changed_decisions"] == 0
    assert result["information_stable_rate"] == 1.0
