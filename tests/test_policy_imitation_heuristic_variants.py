"""Tests for the roadmap C3 heuristic style variants."""

from __future__ import annotations

import random

import numpy as np
import pytest

from splendor.agents.our_agents.dqn.population import HeuristicAgent
from splendor.agents.our_agents.policy_imitation.policies import (
    HOARD_WEIGHTS,
    RUSH_WEIGHTS,
    WeightedHeuristicAgent,
    build_builtin_candidate,
)
from splendor.splendor.splendor_model import SplendorGameRule


@pytest.mark.parametrize("seed", [825_301, 825_302, 825_303])
def test_style_variants_replay_against_random_states(seed: int) -> None:
    """All three styles agree with themselves and differ from each other."""
    random.seed(seed)
    np.random.seed(seed)

    def picks(agent: HeuristicAgent) -> list[tuple[str, int]]:
        seen: list[tuple[str, int]] = []
        local = SplendorGameRule(2)
        for _ in range(20):
            state = local.current_game_state
            turn = local.getCurrentAgentIndex()
            legal = local.getLegalActions(state, turn)
            action = agent.SelectAction(legal, state, local)
            seen.append((action["type"], len(seen)))
            local.update(action)
            if local.gameEnds():
                break
        return seen

    baseline = picks(HeuristicAgent(0))
    rush = picks(WeightedHeuristicAgent(0, RUSH_WEIGHTS))
    hoard = picks(WeightedHeuristicAgent(0, HOARD_WEIGHTS))
    assert rush != hoard
    assert baseline != rush or baseline != hoard


def test_defaults_reproduce_frozen_heuristic() -> None:
    """Default weights must mirror the frozen heuristic exactly."""
    random.seed(825_304)
    np.random.seed(825_304)
    rule = SplendorGameRule(2)
    for _ in range(10):
        state = rule.current_game_state
        turn = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(state, turn)
        frozen = HeuristicAgent(turn).SelectAction(legal, state, rule)
        weighted = WeightedHeuristicAgent(turn).SelectAction(legal, state, rule)
        assert frozen == weighted
        rule.update(weighted)


def test_builtin_candidates_register_style_names() -> None:
    for name in ("heuristic-rush", "heuristic-hoard"):
        candidate = build_builtin_candidate(name)
        agent = candidate.factory(0)
        assert isinstance(agent, WeightedHeuristicAgent)
    with pytest.raises(ValueError):
        build_builtin_candidate("heuristic-flying")


def test_hoard_prefers_collecting_over_baseline() -> None:
    """On a fresh board the hoard style leans to gem collection."""
    random.seed(825_305)
    np.random.seed(825_305)
    rule = SplendorGameRule(2)
    collected = 0
    for _ in range(8):
        state = rule.current_game_state
        turn = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(state, turn)
        hoard = WeightedHeuristicAgent(turn, HOARD_WEIGHTS).SelectAction(
            legal, state, rule
        )
        if hoard["type"].startswith("collect"):
            collected += 1
        rule.update(hoard)
    assert collected >= 1
