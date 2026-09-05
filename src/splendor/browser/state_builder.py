"""
Pseudo-state builder: DOM snapshot -> engine-isomorphic SplendorState.

This module is the pivot of the whole browser layer (plan phase-2 §3.3):
feature extraction (``features.extract_metrics_with_cards``) and legal-action
generation (``SplendorGameRule.getLegalActions``) both run on the state built
here, so the browser layer contains *zero* rule or feature code - everything
is delegated to the engine (single-source ruling, IMPLEMENTATION_SPEC §0.3-1).

Why the placeholder-card trick is sound (verified engine facts F7/F9):
``resources_sufficient`` and ``agent_buying_power`` only read
``len(agent.cards[colour])`` plus ``agent.gems``; ``getLegalActions`` only
reads ``board.dealt / board.gems / board.nobles`` and the acting agent's
``gems/cards`` - never the deck contents. So my permanent cards can be
count-correct placeholders, while table cards and my own reserved cards are
restored as real, registry-identical ``Card`` objects.

Why ``object.__new__``: ``BoardState.__init__`` runs ``random.sample`` and
``random.shuffle``; going through it would waste work on random initialisation
we immediately overwrite *and* silently perturb the global RNG streams that
reproducibility depends on (AGENTS.md fact 6).
"""

from splendor.splendor.constants import (
    NORMAL_COLORS,
    NUMBER_OF_TIERS,
    RESERVED,
    WILDCARD,
)
from splendor.splendor.splendor_model import Card, SplendorState
from splendor.splendor.splendor_utils import COLOURS, AgentTrace

from .card_registry import lookup_card, lookup_noble
from .dom_extractor import CardInfo, PanelInfo, Snapshot, validate_snapshot

# Placeholder cards only ever answer ``len()``; their other fields are inert.
# The synthetic code is deliberately outside CARDS so that an accidental
# identity comparison (Card.__eq__ looks codes up in CARDS) fails loudly
# instead of silently matching a real card.
_PLACEHOLDER_CODE_TEMPLATE = "pseudo-owned-{}-{}"


class SnapshotFaceError(KeyError):
    """A card face from the DOM matches no engine card (extraction bug)."""


def build_pseudo_state(
    snapshot: Snapshot,
    my_index: int,
    turns: int | None = None,
) -> SplendorState:
    """
    Build the engine-isomorphic pseudo state visible from my seat.

    :param snapshot: DOM snapshot (already in engine orientation).
    :param my_index: my position in ``snapshot["panels"]`` / ``state.agents``.
    :param turns: number of actions *I* have taken this game, mirrored into
        ``agent_trace.action_reward`` because ``turns_made_by_agent`` (a live
        feature) derives from it. The DOM exposes no turn counter, so the
        caller must maintain it (the browser env counts one per executed
        action); ``None`` means "start of my participation" (0). The parity
        test passes the engine's own count explicitly.
    """
    validate_snapshot(snapshot)
    panels = snapshot["panels"]
    if my_index not in range(len(panels)):
        raise ValueError(
            f"my_index {my_index} outside 0..{len(panels) - 1} "
            f"(snapshot has {len(panels)} panels)"
        )

    state = object.__new__(SplendorState)
    state.board = _build_board(snapshot)
    state.agents = [
        _build_agent(index, panel, snapshot, my_index, turns)
        for index, panel in enumerate(panels)
    ]
    state.agent_to_move = my_index
    return state


def _build_board(snapshot: Snapshot) -> SplendorState.BoardState:
    """Board visible to every player; decks stay empty (nothing reads them)."""
    board = object.__new__(SplendorState.BoardState)
    # F9: getLegalActions never touches board.decks; deal() (the only other
    # reader) is never invoked on a pseudo state. Empty decks keep the
    # invariant "dealt cards are not in decks" true and fail loudly (deal
    # would return None) if that engine assumption ever changes.
    board.decks = [[] for _ in range(NUMBER_OF_TIERS)]
    board.dealt = _build_dealt(snapshot)
    board.gems = dict(snapshot["supply"])
    board.nobles = [
        lookup_noble(noble["requirements"]) for noble in snapshot["nobles"]
    ]
    return board


def _build_dealt(snapshot: Snapshot) -> list[list[Card | None]]:
    """Restore real Card objects through the identity registry."""
    dealt: list[list[Card | None]] = []
    for deck_id in range(NUMBER_OF_TIERS):
        row: list[Card | None] = []
        for card_info in snapshot["dealt"][deck_id]:
            if card_info is None:
                row.append(None)
            else:
                row.append(_registry_card(card_info, deck_id))
        dealt.append(row)
    return dealt


def _registry_card(info: CardInfo, deck_id: int) -> Card:
    """
    Resolve a table card face, enforcing row/tier consistency.

    ``Action.to_action_element`` locates a table card via
    ``board.dealt[card.deck_id].index(card)``, so a card sitting in row
    ``deck_id`` whose registry deck_id differs would silently corrupt the
    action mapping. That mismatch means the DOM face disagrees with its row
    position - an extraction bug - and must fail here, not in the executor.
    """
    try:
        card = lookup_card(info["tier"], info["colour"], info["points"], info["cost"])
    except KeyError as error:
        raise SnapshotFaceError(
            f"table card face matches no engine card: {info} "
            "(DOM extraction bug or page redesign)"
        ) from error
    if card.deck_id != deck_id:
        raise SnapshotFaceError(
            f"card face {info} sits in row deck_id={deck_id} but its registry "
            f"deck_id is {card.deck_id} (DOM face/row mismatch)"
        )
    return card


def _build_agent(
    index: int,
    panel: PanelInfo,
    snapshot: Snapshot,
    my_index: int,
    turns: int | None,
) -> SplendorState.AgentState:
    """My agent gets full panel data; rivals only what features/rules read."""
    agent = object.__new__(SplendorState.AgentState)
    agent.id = index
    agent.score = panel["score"]
    agent.nobles = []
    agent.passed = False
    agent.last_action = None
    if index == my_index:
        agent.gems = dict(panel["gems"])
        agent.gems.setdefault(WILDCARD, 0)
        agent.cards = _my_cards(panel["card_counts"], snapshot["my_reserved"])
        agent.agent_trace = AgentTrace(index)
        # turns_made_by_agent reads len(action_reward); entries are unused.
        agent.agent_trace.action_reward = [None] * (turns if turns is not None else 0)
    else:
        # Rivals: features read only their score (F7); getLegalActions only
        # computes my actions. Zero gems / empty card lists keep every other
        # code path that might touch a rival well-defined.
        agent.gems = dict.fromkeys(COLOURS.values(), 0)
        agent.cards = {colour: [] for colour in COLOURS.values()}
        agent.agent_trace = AgentTrace(index)
        agent.agent_trace.action_reward = []
    return agent


def _my_cards(card_counts: dict[str, int], my_reserved: list[CardInfo]) -> dict[str, list[Card]]:
    """
    Permanent cards as count-correct placeholders; reserved cards as real
    registry objects (their cost/points feed features and buying).
    """
    cards: dict[str, list[Card]] = {colour: [] for colour in COLOURS.values()}
    for colour in NORMAL_COLORS:
        cards[colour] = _placeholder_cards(colour, card_counts.get(colour, 0))
    cards[RESERVED] = [_reserved_card(info) for info in my_reserved]
    return cards


def _placeholder_cards(colour: str, count: int) -> list[Card]:
    return [
        Card(
            colour=colour,
            code=_PLACEHOLDER_CODE_TEMPLATE.format(colour, position),
            cost={},
            deck_id=-1,
            points=0,
        )
        for position in range(count)
    ]


def _reserved_card(info: CardInfo) -> Card:
    """
    Restore one of my reserved cards.

    A face card's DOM tier is unreliable (ccbs-img-N on a face is an art
    number - only type-5 backs leak the tier), but the triple
    (colour, points, cost) is unique across all 90 cards, so exactly one
    deck_id matches. Probing the three ids keeps the P0 registry untouched
    while still failing loudly (SnapshotFaceError) on impossible faces.
    """
    last_error: KeyError | None = None
    for deck_id in range(NUMBER_OF_TIERS):
        try:
            return lookup_card(deck_id, info["colour"], info["points"], info["cost"])
        except KeyError as error:
            last_error = error
    raise SnapshotFaceError(
        f"reserved card face matches no engine card: {info}"
    ) from last_error
