"""
Offline tests for the advisor's reservation tracker (plan phase-7 T7.3):
synthetic Snapshot sequences exercising the full diff-semantics table -
mid-join init, reserve-from-table identification, reserve-from-deck grey
bucket + evidence hook, purchase retirement, new-game auto-reset and the
seen-face accumulation.
"""

import copy
from typing import Any

from splendor.browser.advisor.tracker import (
    EVENT_INIT,
    EVENT_PURCHASE,
    EVENT_RESERVE_DECK,
    EVENT_RESERVE_TABLE,
    EVENT_RESERVE_UNKNOWN,
    EVENT_RESET,
    ReservationTracker,
)
from splendor.browser.card_registry import CARD_REGISTRY
from splendor.browser.dom_extractor import (
    COLOR_INDEX_TO_NAME,
    FACE_INDEX_TO_NAME,
    Snapshot,
)
from splendor.splendor.splendor_model import Card

# One real card per interesting tier/colour, straight from the registry.
_T0_CARDS = [c for c in CARD_REGISTRY.values() if c.deck_id == 0]
_BLUE_T0 = next(c for c in _T0_CARDS if c.colour == "blue")
_RED_T0 = next(c for c in _T0_CARDS if c.colour == "red")
_T2_CARD = next(c for c in CARD_REGISTRY.values() if c.deck_id == 2)


def _info(card: Card) -> dict[str, Any]:
    """Registry Card -> DOM CardInfo (same face the page would show)."""
    return {
        "tier": card.deck_id,
        "colour": card.colour,
        "points": card.points,
        "cost": dict(card.cost),
    }


def _panel(
    seat: int,
    *,
    score: int = 0,
    card_counts: dict[str, int] | None = None,
    reserved_tiers: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "seat": seat,
        "score": score,
        "card_counts": card_counts
        or dict.fromkeys(FACE_INDEX_TO_NAME, 0),
        "gems": dict.fromkeys(COLOR_INDEX_TO_NAME, 0),
        "reserved_tiers": reserved_tiers or [],
    }


def _snapshot(  # noqa: PLR0913 - fixture builder mirrors the Snapshot shape
    panels: list[dict[str, Any]],
    *,
    dealt: list[list[Any]] | None = None,
    deck_counts: tuple[int, int, int] = (20, 21, 22),
    my_seat: int = 1,
    my_reserved: list[dict[str, Any]] | None = None,
    status: str = "等待玩家2操作",
) -> dict[str, Any]:
    # Deep-copy on purpose: the tracker diffs consecutive snapshots, and the
    # real extract hands it a fresh object every frame - tests must not get
    # away with mutating one shared dict between frames.
    return copy.deepcopy(
        {
            "dealt": dealt or [[None] * 4 for _ in range(3)],
            "deck_counts": list(deck_counts),
            "nobles": [],
            "supply": dict.fromkeys(COLOR_INDEX_TO_NAME, 4),
            "panels": panels,
            "my_seat": my_seat,
            "my_reserved": my_reserved or [],
            "status": status,
            "payment_options": None,
            "noble_options": None,
        }
    )


def _empty_row() -> list[Any]:
    return [None] * 4


def test_mid_join_initialises_unknown_entries() -> None:
    tracker = ReservationTracker()
    delta = tracker.update(
        _snapshot([_panel(1), _panel(2, reserved_tiers=[1, 2])]), frame_seq=1
    )
    assert delta.changed
    assert [event.kind for event in delta.events] == [EVENT_INIT, EVENT_INIT]
    assert [entry.tier for entry in tracker.reserved(2)] == [1, 2]
    assert all(entry.card is None for entry in tracker.reserved(2))
    assert tracker.unknown_reserved_count() == 2


def test_reserve_from_table_identifies_the_vanished_card() -> None:
    tracker = ReservationTracker()
    dealt_before = [[_info(_BLUE_T0), None, None, None], _empty_row(), _empty_row()]
    dealt_after = [_empty_row(), _empty_row(), _empty_row()]
    panels = [_panel(1), _panel(2)]
    tracker.update(_snapshot(panels, dealt=dealt_before), frame_seq=1)

    # Opponent reserves the blue table card: the slot empties while a
    # tier-0 back appears in their panel.
    panels[1]["reserved_tiers"] = [0]
    delta = tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=2)
    assert [event.kind for event in delta.events] == [EVENT_RESERVE_TABLE]
    assert delta.events[0].card is not None
    assert delta.events[0].card.code == _BLUE_T0.code
    entry = tracker.reserved(2)[0]
    assert entry.card is not None and entry.card.code == _BLUE_T0.code
    assert _BLUE_T0.code in tracker.seen_face_codes()


def test_reserve_from_table_rejects_tier_mismatch() -> None:
    # A tier-0 card vanished but the leaked back says tier 2: not table
    # evidence for that back - must degrade to the grey bucket.
    tracker = ReservationTracker()
    dealt_before = [[_info(_BLUE_T0), None, None, None], _empty_row(), _empty_row()]
    dealt_after = [_empty_row(), _empty_row(), _empty_row()]
    panels = [_panel(1), _panel(2)]
    tracker.update(_snapshot(panels, dealt=dealt_before), frame_seq=1)
    panels[1]["reserved_tiers"] = [2]
    delta = tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=2)
    assert [event.kind for event in delta.events] == [EVENT_RESERVE_UNKNOWN]
    assert tracker.reserved(2)[0].card is None


def test_reserve_from_deck_unknown_without_evidence() -> None:
    tracker = ReservationTracker()
    panels = [_panel(1), _panel(2)]
    tracker.update(_snapshot(panels), frame_seq=1)
    panels[1]["reserved_tiers"] = [2]
    delta = tracker.update(_snapshot(panels), frame_seq=2)
    assert [event.kind for event in delta.events] == [EVENT_RESERVE_UNKNOWN]
    assert tracker.reserved(2)[0].card is None
    assert tracker.reserved(2)[0].tier == 2


def test_reserve_from_deck_uses_evidence_hook() -> None:
    def evidence(
        _prev: Snapshot, _curr: Snapshot, _seat: int, tier: int
    ) -> Card | None:
        return _T2_CARD if tier == 2 else None

    tracker = ReservationTracker(evidence=evidence)
    panels = [_panel(1), _panel(2)]
    tracker.update(_snapshot(panels), frame_seq=1)
    panels[1]["reserved_tiers"] = [2]
    delta = tracker.update(_snapshot(panels), frame_seq=2)
    assert [event.kind for event in delta.events] == [EVENT_RESERVE_DECK]
    entry = tracker.reserved(2)[0]
    assert entry.card is not None and entry.card.code == _T2_CARD.code


def test_evidence_wrong_tier_is_rejected() -> None:
    def evidence(
        _prev: Snapshot, _curr: Snapshot, _seat: int, _tier: int
    ) -> Card | None:
        return _BLUE_T0  # a tier-0 face, whatever back was leaked

    tracker = ReservationTracker(evidence=evidence)
    panels = [_panel(1), _panel(2)]
    tracker.update(_snapshot(panels), frame_seq=1)
    panels[1]["reserved_tiers"] = [2]
    tracker.update(_snapshot(panels), frame_seq=2)
    assert tracker.reserved(2)[0].card is None  # 0-tier face for a 2-back


def test_purchase_retires_the_matching_known_entry() -> None:
    tracker = ReservationTracker()
    dealt_before = [[_info(_BLUE_T0), None, None, None], _empty_row(), _empty_row()]
    dealt_after = [_empty_row(), _empty_row(), _empty_row()]
    counts = dict.fromkeys(FACE_INDEX_TO_NAME, 0)
    panels = [_panel(1), _panel(2, card_counts=counts)]
    tracker.update(_snapshot(panels, dealt=dealt_before), frame_seq=1)
    panels[1]["reserved_tiers"] = [0]
    tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=2)
    assert tracker.reserved(2)[0].card is not None

    # Opponent buys the reserved blue card: back gone, blue count +1.
    counts["blue"] = 1
    panels[1]["reserved_tiers"] = []
    delta = tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=3)
    assert [event.kind for event in delta.events] == [EVENT_PURCHASE]
    assert tracker.reserved(2) == []
    # The face stays seen - reserved cards never return to the deck.
    assert _BLUE_T0.code in tracker.seen_face_codes()


def test_purchase_prefers_unknown_when_colours_do_not_match() -> None:
    tracker = ReservationTracker()
    dealt_before = [[_info(_BLUE_T0), None, None, None], _empty_row(), _empty_row()]
    dealt_after = [_empty_row(), _empty_row(), _empty_row()]
    counts = dict.fromkeys(FACE_INDEX_TO_NAME, 0)
    panels = [_panel(1), _panel(2, card_counts=counts)]
    tracker.update(_snapshot(panels, dealt=dealt_before), frame_seq=1)
    # One frame, two reserve actions: blue from the table (identified) and
    # a deck top card (grey bucket).
    panels[1]["reserved_tiers"] = [0, 1]
    tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=2)
    assert [entry.card is not None for entry in tracker.reserved(2)] == [True, False]

    # A red purchase lands: no tracked card is red -> retire the unknown.
    counts["red"] = 1
    panels[1]["reserved_tiers"] = [0]
    delta = tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=3)
    assert [event.kind for event in delta.events] == [EVENT_PURCHASE]
    remaining = tracker.reserved(2)
    assert len(remaining) == 1
    assert remaining[0].card is not None  # the known blue entry survived


def test_new_game_resets_memory() -> None:
    tracker = ReservationTracker()
    panels = [_panel(1, score=5), _panel(2, reserved_tiers=[0])]
    tracker.update(_snapshot(panels), frame_seq=1)
    assert len(tracker.reserved(2)) == 1

    reset_panels = [_panel(1), _panel(2)]
    delta = tracker.update(_snapshot(reset_panels), frame_seq=2)
    assert any(event.kind == EVENT_RESET for event in delta.events)
    assert tracker.reserved(2) == []
    assert tracker.seen_face_codes() == set()

def test_my_own_seat_is_never_tracked() -> None:
    tracker = ReservationTracker()
    panels = [_panel(1, reserved_tiers=[2]), _panel(2)]
    tracker.update(_snapshot(panels, my_seat=1), frame_seq=1)
    assert tracker.reserved(1) == []
    assert tracker.unknown_reserved_count() == 0


def test_seen_faces_survive_a_purchase() -> None:
    tracker = ReservationTracker()
    dealt_before = [[_info(_RED_T0), None, None, None], _empty_row(), _empty_row()]
    dealt_after = [_empty_row(), _empty_row(), _empty_row()]
    counts = dict.fromkeys(FACE_INDEX_TO_NAME, 0)
    counts["red"] = 1
    panels = [_panel(1), _panel(2, card_counts=counts)]
    tracker.update(_snapshot(panels, dealt=dealt_before), frame_seq=1)
    # Opponent buys the dealt red card straight off the table: no reserve
    # involved, the face must remain in the seen set.
    tracker.update(_snapshot(panels, dealt=dealt_after), frame_seq=2)
    assert _RED_T0.code in tracker.seen_face_codes()


def test_my_reserved_faces_enter_the_seen_set() -> None:
    tracker = ReservationTracker()
    delta = tracker.update(
        _snapshot(
            [_panel(1), _panel(2)],
            my_reserved=[_info(_RED_T0)],
            status="等待你操作",
        ),
        frame_seq=1,
    )
    assert not delta.changed
    assert _RED_T0.code in tracker.seen_face_codes()


def test_events_log_survives_reset() -> None:
    tracker = ReservationTracker()
    tracker.update(
        _snapshot([_panel(1, score=5), _panel(2, reserved_tiers=[0])]), frame_seq=1
    )
    tracker.update(_snapshot([_panel(1), _panel(2)]), frame_seq=2)  # all zero
    kinds = [event.kind for event in tracker.events_log()]
    assert EVENT_INIT in kinds
    assert EVENT_RESET in kinds
