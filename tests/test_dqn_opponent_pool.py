"""Tests for the DQN weighted opponent-pool curriculum helper."""

import random

import pytest

from splendor.agents.our_agents.dqn.opponent_pool import (
    OpponentPoolAgent,
    build_opponent_pool,
    parse_opponent_pool,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

ACTION = {"type": "pass", "noble": None}


class MarkerAgent(Agent):
    """Tiny policy that records every delegated call."""

    def __init__(self, _id: int, marker: str) -> None:
        super().__init__(_id)
        self.marker = marker
        self.calls = 0

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        self.calls += 1
        return actions[0]


def test_parse_weighted_pool() -> None:
    assert parse_opponent_pool("random:0.5, minimax:1") == [
        ("random", 0.5),
        ("minimax", 1.0),
    ]


@pytest.mark.parametrize("spec", ["", "random:0", "random:nan", ","])
def test_parse_rejects_invalid_pool(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_opponent_pool(spec)


def test_pool_keeps_one_policy_for_a_complete_game() -> None:
    random.seed(3)
    first = MarkerAgent(0, "first")
    second = MarkerAgent(0, "second")
    pool = OpponentPoolAgent(
        parse_opponent_pool("first:1,second:1"), [first, second], _id=0
    )
    state = SplendorState(2)

    pool.SelectAction([ACTION], state, None)  # type: ignore[arg-type]
    state.agents[0].agent_trace.action_reward.append((ACTION, 0))
    pool.SelectAction([ACTION], state, None)  # type: ignore[arg-type]

    assert sorted((first.calls, second.calls)) == [0, 2]


def test_build_pool_uses_factory_registry() -> None:
    first = MarkerAgent(0, "first")
    pool = build_opponent_pool("first", {"first": lambda _id: [first]})

    assert isinstance(pool, OpponentPoolAgent)
    assert pool.description == "first:1"


def test_build_pool_rejects_multi_agent_factory() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_opponent_pool(
            "bad", {"bad": lambda _id: [MarkerAgent(0, "a"), MarkerAgent(1, "b")]}
        )
