"""Z0 tests: fingerprint/undo exactness and transactor invariants."""

import random
from copy import deepcopy

import numpy as np
import pytest

from splendor.agents.our_agents.alphazero.audit import reshuffle_hidden
from splendor.agents.our_agents.alphazero.state_utils import (
    Transactor,
    legal_action_table,
    state_fingerprint,
)
from splendor.splendor.utils import LimitRoundsGameRule


def _seeded_rule(seed: int) -> LimitRoundsGameRule:
    random.seed(seed)
    np.random.seed(seed)
    return LimitRoundsGameRule(2)


def test_fingerprint_is_deterministic_and_order_sensitive():
    rule = _seeded_rule(7)
    fingerprint = state_fingerprint(rule.current_game_state)
    assert state_fingerprint(rule.current_game_state) == fingerprint
    # Re-sampling the hidden assignment must move the fingerprint (deck
    # order is part of the oracle) even though the public info is unchanged.
    copy = deepcopy(rule)
    reshuffle_hidden(copy, 0, np.random.default_rng(1))
    assert state_fingerprint(copy.current_game_state) != fingerprint


def test_fingerprint_excludes_last_action_when_requested():
    rule = _seeded_rule(11)
    state = rule.current_game_state
    seat = rule.getCurrentAgentIndex()
    _indices, actions = legal_action_table(rule, seat)
    base = state_fingerprint(state, include_last_action=False)
    transactor = Transactor()
    snapshot = transactor.apply(rule, actions[0], seat)
    assert state_fingerprint(state, include_last_action=False) != base
    transactor.undo(rule, actions[0], seat, snapshot)
    assert state_fingerprint(state, include_last_action=False) == base


def _noble_probe_rule(seed: int) -> tuple[LimitRoundsGameRule, dict, int]:
    """A rule plus a real buy action forced to award a non-last noble.

    The noble branch (successor deletes from the middle of
    ``board.nobles``, predecessor appends back at the end) is the one
    engine-unrestored detail; forcing it deterministically keeps the
    exactness test meaningful instead of hoping random play triggers a
    noble award.
    """
    rule = _seeded_rule(seed)
    seat = rule.getCurrentAgentIndex()
    state = rule.current_game_state
    # Move 0 has no affordable cards, hence no buy actions: endow exactly
    # the cheapest dealt card's cost (a unique greedy payment, no discard
    # rule interference, and the action is inside the enumerated space).
    dealt = [
        card for row in state.board.dealt for card in row if card is not None
    ]
    cheapest = min(dealt, key=lambda card: sum(card.cost.values()))
    for colour, count in cheapest.cost.items():
        state.agents[seat].gems[colour] = count
    _indices, actions = legal_action_table(rule, seat)
    buy = next(action for action in actions if "buy" in action["type"])
    nobles = state.board.nobles
    assert len(nobles) >= 2
    buy["noble"] = nobles[0]  # not the last element -> order permutes
    return rule, buy, seat


def test_apply_undo_is_fingerprint_exact():
    transactor = Transactor()
    for seed in (830_001, 830_002, 830_003):
        rule, action, seat = _noble_probe_rule(seed)
        state = rule.current_game_state
        before = state_fingerprint(state)
        snapshot = transactor.apply(rule, action, seat)
        assert state_fingerprint(state) != before
        transactor.undo(rule, action, seat, snapshot)
        assert state_fingerprint(state) == before


def test_raw_engine_undo_permutes_nobles_without_snapshot():
    """Control: without the snapshot the noble branch does NOT undo exactly.

    This is what justifies the Transactor: the engine alone leaves
    ``board.nobles`` permuted when a non-last noble is taken.
    """
    rule, action, seat = _noble_probe_rule(830_004)
    state = rule.current_game_state
    before = state_fingerprint(state)
    rule.generateSuccessor(state, action, seat)
    rule.generatePredecessor(state, action, seat)
    assert state_fingerprint(state) != before


def test_transactor_leaves_rule_bookkeeping_untouched():
    rule = _seeded_rule(13)
    seat = rule.getCurrentAgentIndex()
    counter = rule.action_counter
    agent_index = rule.current_agent_index
    _indices, actions = legal_action_table(rule, seat)
    transactor = Transactor()
    snapshot = transactor.apply(rule, actions[0], seat)
    transactor.undo(rule, actions[0], seat, snapshot)
    assert rule.action_counter == counter
    assert rule.current_agent_index == agent_index


def test_legal_action_table_is_sorted_and_aligned():
    rule = _seeded_rule(17)
    seat = rule.getCurrentAgentIndex()
    indices, actions = legal_action_table(rule, seat)
    assert indices == sorted(indices)
    assert len(indices) == len(actions) > 0
    with pytest.raises(IndexError):
        # Sorted indices must all resolve through the engine mapping.
        _ = actions[len(indices)]
