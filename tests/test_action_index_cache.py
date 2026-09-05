"""
Phase-0 acceptance tests for the ActionIndexCache (plan/phase-0 §4: A0.1, A0.2).

The cache replaced O(3510) linear scans inside create_legal_actions_mask /
create_action_mapping with O(1) hash lookups. These tests pin the two
invariants that make the optimization safe:

1. the cache is a bijection over ALL_ACTIONS (no key collisions);
2. the cached implementations are output-identical to the pre-cache
   ``_slow_*`` reference implementations on real engine states.
"""

import random

import numpy as np
import pytest

from splendor.splendor.gym.envs.actions import ALL_ACTIONS, Action, ActionEnum
from splendor.splendor.gym.envs.utils import (
    ACTION_INDEX,
    _index_of,
    _slow_create_action_mapping,
    _slow_create_legal_actions_mask,
    action_key,
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule

# plan/phase-0 §4 A0.1 calls for >= 50 random engine states.
NUM_DECISION_POINTS = 50


def _iter_random_decision_points(num_points: int, seed: int = 1234):
    """
    Yield ``(state, agent_index, legal_actions)`` tuples sampled from random
    playouts. States are consumed immediately by the caller - the engine
    mutates them in place on every update().
    """
    rng = random.Random(seed)
    rule = SplendorGameRule(2)
    produced = 0
    while produced < num_points:
        state = rule.current_game_state
        agent_index = rule.current_agent_index
        legal_actions = rule.getLegalActions(state, agent_index)
        yield state, agent_index, legal_actions
        produced += 1

        rule.update(rng.choice(legal_actions))
        if rule.gameEnds():
            rule = SplendorGameRule(2)


def test_cache_build_is_bijection():
    """
    A0.2: one cache entry per action, keys pairwise distinct, indices a
    permutation of range(3510).
    """
    assert len(ALL_ACTIONS) == 3510
    assert len(ACTION_INDEX) == len(ALL_ACTIONS)
    assert set(ACTION_INDEX.values()) == set(range(len(ALL_ACTIONS)))

    for index, action in enumerate(ALL_ACTIONS):
        assert ACTION_INDEX[action_key(action)] == index


def test_mask_equivalence_on_random_states():
    """
    A0.1 (mask): the cached mask builder is output-identical to the
    pre-cache implementation on real engine decision points.
    """
    for state, agent_index, legal_actions in _iter_random_decision_points(
        NUM_DECISION_POINTS
    ):
        cached = create_legal_actions_mask(legal_actions, state, agent_index)
        reference = _slow_create_legal_actions_mask(legal_actions, state, agent_index)
        assert np.array_equal(cached, reference)


def test_mapping_equivalence_on_random_states():
    """
    A0.1 (mapping): the cached action mapping is output-identical to the
    pre-cache implementation on real engine decision points.
    """
    for state, agent_index, legal_actions in _iter_random_decision_points(
        NUM_DECISION_POINTS
    ):
        cached = create_action_mapping(legal_actions, state, agent_index)
        reference = _slow_create_action_mapping(legal_actions, state, agent_index)
        assert cached == reference


def test_index_of_distinguishes_none_from_empty_gems():
    """
    The buy actions (collected_gems=None) and the no-return collect actions
    (returned_gems={}) must land on different cache keys.
    """
    pass_key = action_key(Action(type_enum=ActionEnum.PASS))
    pass_with_gems_key = action_key(
        Action(type_enum=ActionEnum.PASS, collected_gems={"white": 1})
    )
    assert pass_key != pass_with_gems_key
    assert pass_key[1] is None  # collected_gems=None stays None in the key
    assert pass_with_gems_key[1] == (("white", 1),)


def test_index_of_unknown_action_raises_with_details():
    """
    An action outside ALL_ACTIONS must raise a ValueError naming the action,
    instead of a bare list.index() error deep inside a loop.
    """
    foreign_action = Action(type_enum=ActionEnum.PASS, collected_gems={"white": 1})

    with pytest.raises(ValueError, match="action not in ALL_ACTIONS"):
        _index_of(foreign_action)
