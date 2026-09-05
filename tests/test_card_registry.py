"""
Phase-0 acceptance tests for the card/noble identity registry
(plan/phase-0 §4: A0.3).

The registry is the only identity bridge between web DOM card faces and
engine objects, so it must cover every card/noble exactly once and hand
back objects identical to the engine's own.
"""

import pytest

from splendor.browser.card_registry import (
    CARD_REGISTRY,
    NOBLE_REGISTRY,
    card_key,
    lookup_card,
    lookup_noble,
    web_row_to_deck_id,
)
from splendor.splendor.splendor_model import SplendorGameRule
from splendor.splendor.splendor_utils import CARDS, NOBLES


def test_all_cards_hit_and_are_unique():
    """A0.3: 90/90 cards hit the registry, with 90 pairwise-distinct keys."""
    assert len(CARDS) == 90
    assert len(CARD_REGISTRY) == 90

    for code, (colour, cost, deck_id, points) in CARDS.items():
        engine_deck_id = deck_id - 1
        card = lookup_card(engine_deck_id, colour, points, cost)
        assert card.code == code


def test_all_nobles_hit_and_are_unique():
    """A0.3: 10/10 nobles hit the registry, with 10 pairwise-distinct keys."""
    assert len(NOBLES) == 10
    assert len(NOBLE_REGISTRY) == 10

    for code, cost in NOBLES:
        assert lookup_noble(cost) == (code, cost)


def test_card_fields_match_engine():
    """
    A0.3: every card the engine deals (decks + dealt rows) is field-by-field
    identical to its registry counterpart - including the code string, which
    drives Card.__eq__.
    """
    state = SplendorGameRule(2).initialGameState()

    engine_cards = [
        card
        for deck in (*state.board.decks, *state.board.dealt)
        for card in deck
        if card is not None
    ]
    assert len(engine_cards) == 90

    for engine_card in engine_cards:
        registry_card = lookup_card(
            engine_card.deck_id,
            engine_card.colour,
            engine_card.points,
            engine_card.cost,
        )
        assert registry_card.code == engine_card.code
        assert registry_card.colour == engine_card.colour
        assert registry_card.cost == engine_card.cost
        assert registry_card.deck_id == engine_card.deck_id
        assert registry_card.points == engine_card.points
        assert registry_card == engine_card  # Card.__eq__ (code + points)


def test_lookup_card_miss_raises_with_quadruple():
    with pytest.raises(KeyError, match="no card matches face"):
        lookup_card(0, "white", 99, {"white": 99})


def test_web_row_to_deck_id_direction():
    """Web renders tiers top-down as 3/2/1; engine deck ids are 0-indexed."""
    assert web_row_to_deck_id(0) == 2
    assert web_row_to_deck_id(1) == 1
    assert web_row_to_deck_id(2) == 0


def test_card_key_normalizes_cost_order():
    """dict key order must not affect the identity of a card face."""
    assert card_key(0, "black", 0, {"green": 1, "white": 1}) == card_key(
        0, "black", 0, {"white": 1, "green": 1}
    )
