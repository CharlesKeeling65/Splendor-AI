"""Tests for student-state DAgger collection and history aggregation."""

from pathlib import Path
from typing import override

import numpy as np

from splendor.agents.our_agents.policy_imitation.dagger import (
    aggregate_dagger_datasets,
    collect_dagger_dataset,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.trajectory import TrajectoryDataset
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class _FirstActionAgent(Agent):
    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        return actions[0]


class _LastActionAgent(Agent):
    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        return actions[-1]


def _candidate(name: str, agent: type[Agent]) -> CandidateSpec:
    return CandidateSpec(name, "teacher_candidate", agent)


def test_dagger_labels_student_visited_states(tmp_path: Path) -> None:
    path = tmp_path / "round.npz"
    result = collect_dagger_dataset(
        _candidate("teacher", _LastActionAgent),
        _candidate("student", _FirstActionAgent),
        _candidate("opponent", _FirstActionAgent),
        [800501],
        path,
        feature_version="v1",
        round_index=1,
        source_manifest="dagger.json",
    )

    dataset = TrajectoryDataset.load(path)
    records = dataset.metadata["game_records"]
    assert result["samples"] == dataset.size > 0
    assert dataset.metadata["dagger_round"] == 1
    assert dataset.metadata["source_manifest"] == "dagger.json"
    assert sum(record["teacher_queries"] for record in records) > 0
    assert sum(record["student_queries"] for record in records) == dataset.size
    assert sum(record["teacher_action_mismatches"] for record in records) > 0
    assert np.all(dataset.legal_masks[np.arange(dataset.size), dataset.actions] == 1)


def test_dagger_aggregate_retains_round_hashes(tmp_path: Path) -> None:
    source = tmp_path / "round.npz"
    aggregate = tmp_path / "aggregate.npz"
    collect_dagger_dataset(
        _candidate("teacher", _LastActionAgent),
        _candidate("student", _FirstActionAgent),
        _candidate("opponent", _FirstActionAgent),
        [800502],
        source,
        feature_version="v1",
        round_index=1,
    )

    result = aggregate_dagger_datasets(
        [source, source],
        aggregate,
        round_index=2,
        source_manifest="dagger.json",
    )
    dataset = TrajectoryDataset.load(aggregate)
    assert result["samples"] == 2 * TrajectoryDataset.load(source).size
    assert len(dataset.metadata["source_hashes"]) == 2
    assert dataset.metadata["dagger_round"] == 2
    assert len(dataset.metadata["game_records"]) == 4
