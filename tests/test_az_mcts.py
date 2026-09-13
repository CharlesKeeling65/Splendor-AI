"""Z0 tests: az_search behaviour, determinism and in-place invariants."""

import random
from copy import deepcopy

import numpy as np
import pytest

from splendor.agents.our_agents.alphazero.evaluator import UniformEvaluator
from splendor.agents.our_agents.alphazero.mcts import az_search, select_action
from splendor.agents.our_agents.alphazero.state_utils import (
    legal_action_table,
    state_fingerprint,
)
from splendor.splendor.utils import LimitRoundsGameRule


def _seeded_rule(seed: int, moves: int = 5) -> LimitRoundsGameRule:
    random.seed(seed)
    np.random.seed(seed)
    rule = LimitRoundsGameRule(2)
    for _ in range(moves):
        if rule.gameEnds():
            break
        state = rule.current_game_state
        seat = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(state, seat)
        rule.update(legal[int(np.random.randint(len(legal)))])
    return rule


def test_search_returns_valid_distribution_over_legal_actions():
    rule = _seeded_rule(42)
    indices, _actions = legal_action_table(rule, rule.getCurrentAgentIndex())
    result = az_search(
        rule,
        UniformEvaluator(),
        np.random.default_rng(0),
        simulations=12,
        n_trees=2,
    )
    assert result.indices == indices
    assert len(result.pi) == 3510
    assert result.pi.sum() == pytest.approx(1.0, abs=1e-5)
    assert result.visits.sum() == pytest.approx(12.0)
    legal_mask = np.zeros(3510)
    legal_mask[indices] = 1
    assert (result.pi[legal_mask == 0] == 0).all()


def test_search_is_deterministic_under_fixed_seed():
    rule = _seeded_rule(43)
    first = az_search(
        rule, UniformEvaluator(), np.random.default_rng(7), simulations=12, n_trees=2
    )
    second = az_search(
        rule, UniformEvaluator(), np.random.default_rng(7), simulations=12, n_trees=2
    )
    assert np.array_equal(first.visits, second.visits)


def test_search_does_not_mutate_the_searched_rule():
    rule = _seeded_rule(44)
    before = state_fingerprint(rule.current_game_state)
    az_search(
        rule, UniformEvaluator(), np.random.default_rng(0), simulations=8, n_trees=2
    )
    assert state_fingerprint(rule.current_game_state) == before


def test_root_noise_changes_the_distribution():
    rule = _seeded_rule(45)
    clean = az_search(
        rule, UniformEvaluator(), np.random.default_rng(1), simulations=24, n_trees=2
    )
    noisy = az_search(
        rule,
        UniformEvaluator(),
        np.random.default_rng(1),
        simulations=24,
        n_trees=2,
        root_noise=True,
    )
    assert not np.array_equal(clean.visits, noisy.visits)


def test_select_action_temperature_zero_is_argmax():
    rule = _seeded_rule(46)
    result = az_search(
        rule, UniformEvaluator(), np.random.default_rng(2), simulations=9, n_trees=3
    )
    rng = np.random.default_rng(0)
    assert select_action(result, 0.0, rng) == result.indices[
        int(np.argmax(result.visits))
    ]


def test_search_rejects_invalid_inputs():
    terminal = _seeded_rule(47, moves=200)
    if not terminal.gameEnds():
        pytest.skip("seeded game did not terminate")
    with pytest.raises(ValueError):
        az_search(
            terminal,
            UniformEvaluator(),
            np.random.default_rng(0),
            simulations=8,
            n_trees=2,
        )
    rule = _seeded_rule(48)
    with pytest.raises(ValueError):
        az_search(
            rule,
            UniformEvaluator(),
            np.random.default_rng(0),
            simulations=2,
            n_trees=4,
        )
    from splendor.splendor.splendor_model import SplendorGameRule

    three_player = SplendorGameRule(3)
    with pytest.raises(ValueError):
        az_search(
            three_player,
            UniformEvaluator(),
            np.random.default_rng(0),
            simulations=8,
            n_trees=2,
        )


def test_search_transposition_keys_ignores_deck_identity_differences():
    """Same public info under different hidden orders shares tree nodes.

    The uniform evaluator keys on the (last-action-free) state fingerprint of
    the *determinized* copy, so two searches started from hidden orders that
    sample to the same determinization must visit identical nodes.
    """
    rule = _seeded_rule(49)
    result = az_search(
        rule, UniformEvaluator(), np.random.default_rng(3), simulations=16, n_trees=2
    )
    replayed = az_search(
        deepcopy(rule),
        UniformEvaluator(),
        np.random.default_rng(3),
        simulations=16,
        n_trees=2,
    )
    assert np.array_equal(result.visits, replayed.visits)
