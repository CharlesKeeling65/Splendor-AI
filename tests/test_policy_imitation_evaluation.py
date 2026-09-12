"""Tests for paired evaluation, information audits, and trajectory replay."""

from pathlib import Path
from typing import Any, override

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
from splendor.agents.our_agents.policy_imitation.runner import select_action
from splendor.agents.our_agents.policy_imitation.trajectory import (
    TrajectoryDataset,
    split_by_seed,
)
from splendor.splendor.gym.envs.utils import create_action_mapping
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
    records: list[dict[str, Any]] = [
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


class _RuleDeckSensitiveAgent(Agent):
    """Positive control that reads hidden deck order through the rule."""

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state
        deck = game_rule.current_game_state.board.decks[0]
        if len(deck) < 2:
            return actions[0]
        return actions[0] if deck[0].code < deck[-1].code else actions[-1]


class _PublicBoardAgent(Agent):
    """Deterministic control that uses only the displayed board."""

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_rule
        public_card = game_state.board.dealt[0][0]
        if public_card is None:
            return actions[0]
        return actions[0] if public_card.code < "m" else actions[-1]


class _CoherentMutatingAgent(Agent):
    """Mutate only defensive inputs while returning a copied legal action."""

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        assert game_rule.current_game_state is game_state
        assert game_rule.current_agent_index == 1
        assert game_rule.action_counter == 17
        selected = actions[1]
        actions.clear()
        game_state.board.decks[0].reverse()
        game_state.agents[0].score = 999
        game_rule.current_agent_index = 0
        game_rule.action_counter = 0
        return selected


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


def test_rule_state_hidden_dependency_is_flagged_by_projection_audit() -> None:
    result = audit_candidate_information(
        CandidateSpec(
            "rule-deck-sensitive", "teacher_candidate", _RuleDeckSensitiveAgent
        ),
        _random_candidate("rival"),
        [800004],
        max_states=2,
    )

    assert result["status"] == "risk"
    assert result["changed_decisions"] > 0
    assert any(
        item.get("changed_decision")
        for item in result["details"]["hidden_deck_order"]
    )
    assert any("not proof" in risk for risk in result["risks"])


def test_public_policy_passes_projection_audit() -> None:
    result = audit_candidate_information(
        CandidateSpec("public-board", "teacher_candidate", _PublicBoardAgent),
        _random_candidate("rival"),
        [800005],
        max_states=2,
    )

    assert result["status"] == "pass"
    assert result["changed_decisions"] == 0
    assert result["information_stable_rate"] == 1.0


def test_select_action_keeps_defensive_rule_state_coherent_and_validates_index() -> None:
    rule = SplendorGameRule(2)
    rule.current_agent_index = 1
    rule.action_counter = 17
    state = rule.current_game_state
    actions = rule.getLegalActions(state, 1)
    expected_index = next(
        index
        for index, action in create_action_mapping(actions, state, 1).items()
        if action == actions[1]
    )
    before_state = str(state)

    result = select_action(_CoherentMutatingAgent(1), actions, state, rule)

    assert result.action_index == expected_index
    assert rule.current_game_state is state
    assert str(rule.current_game_state) == before_state
    assert rule.current_agent_index == 1
    assert rule.action_counter == 17
