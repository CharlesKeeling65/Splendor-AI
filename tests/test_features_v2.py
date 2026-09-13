"""Tests for the shared public-v2 feature schemas (roadmap B1)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from splendor.splendor.constants import NORMAL_COLORS
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.features_v2 import (
    PANEL_DIM,
    V1_DIM,
    V2_DIM,
    V2_MULTI_DIM,
    extract_observation_public_v2,
    extract_observation_public_v2_multi,
    extract_observation_public_v2_multi_prefix,
    observation_dim,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState


def _random_state(seed: int, players: int, moves: int = 12):
    """Deal and play a few random moves to get a non-trivial mid-game state."""
    random.seed(seed)
    np.random.seed(seed)
    rule = SplendorGameRule(players)
    for _ in range(moves):
        agent_index = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(rule.current_game_state, agent_index)
        if not legal:
            break
        rule.update(random.choice(legal))
        if rule.gameEnds():
            break
    return rule.current_game_state


def _legacy_reference(state: SplendorState, seat: int) -> np.ndarray:
    """Independent recomputation of the original two-seat DQN formula."""
    base = extract_metrics_with_cards(state, seat).astype(np.float32)
    colors = (*NORMAL_COLORS, "yellow")
    extra: list[float] = [float(state.board.gems.get(c, 0)) for c in colors]
    for index in (seat, 1 - seat):
        agent = state.agents[index]
        extra.extend(float(agent.gems.get(c, 0)) for c in colors)
        extra.extend(float(len(agent.cards[c])) for c in NORMAL_COLORS)
        extra.append(float(len(agent.cards["yellow"])))
    nobles = state.board.nobles
    for index in range(3):
        cost = nobles[index][1] if index < len(nobles) else {}
        extra.extend(float(cost.get(c, 0)) for c in NORMAL_COLORS)
    extra.extend(
        [
            float(seat),
            float(any(a.score >= 15 for a in state.agents)),
        ]
    )
    return np.concatenate((base, np.asarray(extra, dtype=np.float32)))


def test_observation_dim_validates_schemas() -> None:
    assert observation_dim("v1") == V1_DIM == 265
    assert observation_dim("public-v2") == V2_DIM == 312
    assert observation_dim("public-v2-multi") == V2_MULTI_DIM == 337
    with pytest.raises(ValueError):
        observation_dim("public-v3")


def test_legacy_public_v2_matches_original_formula() -> None:
    """v1 must stay untouched and public-v2 must stay byte-identical."""
    for seed in range(5):
        state = _random_state(seed, players=2)
        for seat in (0, 1):
            got = extract_observation_public_v2(state, seat)
            want = _legacy_reference(state, seat)
            assert got.dtype == np.float32
            assert got.shape == (V2_DIM,)
            assert np.array_equal(got, want), f"seed={seed} seat={seat}"


def test_legacy_public_v2_rejects_other_seat_counts() -> None:
    state = _random_state(7, players=3, moves=2)
    with pytest.raises(ValueError):
        extract_observation_public_v2(state, 0)


def test_multi_schema_dimension_is_seat_count_invariant() -> None:
    for players in (2, 3, 4):
        state = _random_state(11 + players, players, moves=6)
        for seat in range(players):
            got = extract_observation_public_v2_multi(state, seat)
            assert got.shape == (V2_MULTI_DIM,)


def test_multi_two_seat_is_legacy_prefix_plus_explicit_padding() -> None:
    state = _random_state(21, players=2)
    for seat in (0, 1):
        legacy = extract_observation_public_v2(state, seat)
        multi = extract_observation_public_v2_multi(state, seat)
        assert np.array_equal(multi[:V2_DIM], legacy)
        assert multi[V2_DIM] == 2.0  # explicit seat-count feature
        assert np.array_equal(multi[V2_DIM + 1 :], np.zeros(V2_MULTI_DIM - V2_DIM - 1))


def test_multi_rival_slots_follow_cyclic_seat_order() -> None:
    state = _random_state(31, players=4, moves=8)
    seat = 1
    colors = (*NORMAL_COLORS, "yellow")
    prefix = extract_observation_public_v2_multi_prefix(state, seat)

    def _panel(index: int) -> list[float]:
        agent = state.agents[index]
        panel = [float(agent.gems.get(c, 0)) for c in colors]
        panel.extend(float(len(agent.cards[c])) for c in NORMAL_COLORS)
        panel.append(float(len(agent.cards["yellow"])))
        return panel

    expected_self = _panel(seat)
    expected_rival0 = _panel((seat + 1) % 4)
    got_self = prefix[V1_DIM + 6 : V1_DIM + 6 + PANEL_DIM]
    got_rival0 = prefix[V1_DIM + 6 + PANEL_DIM : V1_DIM + 6 + 2 * PANEL_DIM]
    assert np.array_equal(got_self, np.asarray(expected_self, dtype=np.float32))
    assert np.array_equal(got_rival0, np.asarray(expected_rival0, dtype=np.float32))

    multi = extract_observation_public_v2_multi(state, seat)
    expected_rival1 = _panel((seat + 2) % 4)
    expected_rival2 = _panel((seat + 3) % 4)
    assert np.array_equal(
        multi[V2_DIM + 1 : V2_DIM + 1 + PANEL_DIM],
        np.asarray(expected_rival1, dtype=np.float32),
    )
    assert np.array_equal(
        multi[V2_DIM + 1 + PANEL_DIM :],
        np.asarray(expected_rival2, dtype=np.float32),
    )


def test_multi_three_seat_zero_pads_last_rival_slot() -> None:
    state = _random_state(41, players=3, moves=6)
    seat = 2
    multi = extract_observation_public_v2_multi(state, seat)
    assert multi[V2_DIM] == 3.0
    # Two real rivals fill slots 0-1 (the latter in the tail block); slot 2 pads.
    assert np.array_equal(
        multi[V2_DIM + 1 + PANEL_DIM :], np.zeros(V2_MULTI_DIM - V2_DIM - 1 - PANEL_DIM)
    )


def test_multi_rejects_out_of_range_seat() -> None:
    state = _random_state(43, players=3, moves=2)
    with pytest.raises(ValueError):
        extract_observation_public_v2_multi(state, 3)


def test_known_state_hand_computed_supply_block() -> None:
    """The six supply gems land at [V1_DIM:V1_DIM+6) in declared colour order."""
    state = _random_state(47, players=2, moves=4)
    seat = 0
    obs = extract_observation_public_v2(state, seat)
    colors = (*NORMAL_COLORS, "yellow")
    expected = np.asarray(
        [float(state.board.gems.get(c, 0)) for c in colors], dtype=np.float32
    )
    assert np.array_equal(obs[V1_DIM : V1_DIM + 6], expected)
