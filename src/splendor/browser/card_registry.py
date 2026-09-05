"""
Card & noble identity registry: the only bridge between web DOM identities
and engine objects.

The web page exposes a card by its face contents (colour, points, cost) and
its row on the table. This module maps that content tuple back to the exact
engine ``Card`` object (and a noble cost vector to the exact ``(code, cost)``
tuple), so everything downstream (feature extraction, legal-action
computation) operates on objects identical to the engine's own.

Backed by two verified facts:
  * the quadruple ``(deck_id, colour, points, cost)`` is unique across all
    90 cards;
  * the 10 nobles all have pairwise-distinct cost vectors.

The web tier-row direction conversion (top row = tier 3 = deck_id 2, bottom
row = tier 1 = deck_id 0) is deliberately funnelled through this module -
it is the single most off-by-one-prone spot of the whole browser layer.
"""

from splendor.splendor.splendor_model import Card
from splendor.splendor.splendor_utils import CARDS, NOBLES

# (deck_id, colour, points, sorted cost items)
CardKey = tuple[int, str, int, tuple[tuple[str, int], ...]]
# sorted cost items
NobleKey = tuple[tuple[str, int], ...]

# Values are engine-identical Card objects; the code string is taken from the
# CARDS key so the built Card compares equal to the engine's own cards
# (Card.__eq__ is code + points based).
CARD_REGISTRY: dict[CardKey, Card] = {}
NOBLE_REGISTRY: dict[NobleKey, tuple[str, dict]] = {}


def _build_registries() -> None:
    """
    Build both registries from the engine's static data, once at import.
    """
    for code, (colour, cost, deck_id, points) in CARDS.items():
        # the engine decrements deck_id when constructing its cards
        # (splendor_model.BoardState.__init__), so 0-index it the same way.
        engine_card = Card(colour, code, cost, deck_id - 1, points)
        CARD_REGISTRY[card_key(deck_id - 1, colour, points, cost)] = engine_card

    for code, cost in NOBLES:
        NOBLE_REGISTRY[noble_key(cost)] = (code, cost)


def card_key(deck_id: int, colour: str, points: int, cost: dict[str, int]) -> CardKey:
    """
    Build the registry key of a card from its face contents.
    """
    return (deck_id, colour, points, tuple(sorted(cost.items())))


def noble_key(cost: dict[str, int]) -> NobleKey:
    """
    Build the registry key of a noble from its cost vector.
    """
    return tuple(sorted(cost.items()))


def lookup_card(deck_id: int, colour: str, points: int, cost: dict[str, int]) -> Card:
    """
    Resolve a card face (as read from the DOM) into the engine Card object.

    :raises KeyError: carrying the input quadruple, when no card matches -
                      which indicates either a DOM extraction bug or a page
                      redesign, and should surface as early as possible.
    """
    key = card_key(deck_id, colour, points, cost)
    if key not in CARD_REGISTRY:
        raise KeyError(f"no card matches face (deck_id={deck_id}, colour={colour}, "
                       f"points={points}, cost={cost})")
    return CARD_REGISTRY[key]


def lookup_noble(cost: dict[str, int]) -> tuple[str, dict]:
    """
    Resolve a noble cost vector (as read from the DOM) into the engine
    ``(code, cost)`` tuple.

    :raises KeyError: carrying the input cost vector.
    """
    key = noble_key(cost)
    if key not in NOBLE_REGISTRY:
        raise KeyError(f"no noble matches cost={cost}")
    return NOBLE_REGISTRY[key]


def web_row_to_deck_id(row_index: int) -> int:
    """
    Convert a web table row (0-based, top to bottom) into the engine's
    0-indexed deck id. The web renders tiers top-down as 3/2/1.
    """
    return 2 - row_index


_build_registries()
