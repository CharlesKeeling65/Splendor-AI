"""Z0 tests: the information audit must pass cleanly and catch leaks."""

from copy import deepcopy

import numpy as np

from splendor.agents.our_agents.alphazero.audit import (
    DeckPeekingEvaluator,
    reshuffle_hidden,
    run_information_audit,
)
from splendor.agents.our_agents.alphazero.benchmark import sample_positions
from splendor.agents.our_agents.alphazero.evaluator import UniformEvaluator


def _positions() -> list:
    return list(sample_positions([4, 8, 12], seed=830_100).values())


def test_clean_evaluator_is_hidden_order_invariant():
    positions = _positions()
    report = run_information_audit(
        UniformEvaluator(), positions, seed=831_000, simulations=8, n_trees=2
    )
    assert report["check"] == "clean"
    assert report["clean_equal"] is True
    assert all(entry["equal"] for entry in report["positions"])


def test_deck_peeking_control_breaks_the_invariance():
    positions = _positions()
    report = run_information_audit(
        DeckPeekingEvaluator(positions[0]),
        positions,
        seed=831_000,
        simulations=8,
        n_trees=2,
    )
    assert report["check"] == "positive_control"
    # The leak is deterministic but must actually fire on at least one
    # position - otherwise the control itself is broken.
    assert report["control_differs"] is True


def test_reshuffle_hidden_preserves_public_information():
    rule = sample_positions([6], seed=830_101)[6]
    copy = deepcopy(rule)
    before_dealt = [list(row) for row in copy.current_game_state.board.dealt]
    before_scores = [a.score for a in copy.current_game_state.agents]
    reshuffle_hidden(copy, 0, np.random.default_rng(5))
    assert [list(row) for row in copy.current_game_state.board.dealt] == before_dealt
    assert [a.score for a in copy.current_game_state.agents] == before_scores
    # Hidden assignment: same multiset of unseen cards, different order.
    for tier in range(3):
        old = sorted(c.code for c in rule.current_game_state.board.decks[tier])
        new = sorted(c.code for c in copy.current_game_state.board.decks[tier])
        assert old == new
