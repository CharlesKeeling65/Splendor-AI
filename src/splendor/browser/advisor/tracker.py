"""
Opponent-reservation memory (plan phase-7 §3.2) - the advisory-side memory
reconstruction.

Driving page fact (user-confirmed 2026-09-14, to be anchored by the T7.2/E7
experiment): the moment an opponent reserves, the reserved card's face is
observable somewhere on the page. This module diffs consecutive snapshots
and turns those moments into persistent per-seat memory:

* **reserve-from-table** - a dealt card vanished in the same diff: the face
  is known from the board position alone, no selector needed.
* **reserve-from-deck** - the face can only come from the evidence hook
  (:class:`ReserveEvidence`, supplied once E7 anchors the DOM region).
  Until then the hook is ``None`` and the entry degrades to "unknown, tier
  known" - an explicit grey bucket, never a guess.
* **purchase of a reserved card** - the back count drops; the entry is
  retired (preferably cross-checked against the card-colour delta), so a
  missed identification cannot accumulate: it self-heals at purchase time.

The tracker also maintains ``seen_face_codes``: every card face ever
observed (dealt, my reserved, identified rival reserves). Together with the
90-card registry that set *is* the deck composition - the base of the
phase-7 deck histogram (plan §3.4) and the tightness upgrade over
``remote/rollout.py``'s blind per-tier sampling (whose blind picks this
tracker's identities replace inside the advisor's determinized
reconstruction).

The tracker is pure event memory: it never re-derives rules (single-source
discipline) and never touches the driver - it consumes already-extracted
:class:`Snapshot` values.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from splendor.browser.card_registry import CARD_REGISTRY, lookup_card
from splendor.browser.dom_extractor import (
    CardInfo,
    PanelInfo,
    Snapshot,
)
from splendor.splendor.splendor_model import Card

# A reserve-reveal evidence provider: given the previous and current
# snapshots, the reserving seat (1-based) and the leaked tier (0-based),
# return the identified card - or None when the reveal is not readable.
# Implemented by T7.2/E7 once the transient DOM region is measured; a None
# return keeps behaviour honest (grey bucket) meanwhile.
ReserveEvidence = Callable[[Snapshot, Snapshot, int, int], Card | None]

EVENT_INIT = "init"
EVENT_RESET = "reset"
EVENT_RESERVE_TABLE = "reserve_table"
EVENT_RESERVE_DECK = "reserve_deck"
EVENT_RESERVE_UNKNOWN = "reserve_unknown"
EVENT_PURCHASE = "purchase"

# code -> engine Card, built once (the registry is import-time static).
_CODE_TO_CARD: dict[str, Card] = {
    card.code: card for card in CARD_REGISTRY.values()
}


@dataclass
class TrackedReserved:
    """One rival-reserved card as memory represents it."""

    tier: int  # 0-based deck id, leaked by the card back (ccbs-img-N)
    card: Card | None  # None = face unknown (reserve-from-deck, no evidence)
    frame_seq: int  # adopted frame the entry was created at


@dataclass(frozen=True)
class ReservedEvent:
    """One observed memory transition, for logs / UI honesty lines."""

    seat: int  # 1-based page seat; 0 = tracker-level (init/reset)
    kind: str  # EVENT_* constant
    tier: int | None
    card: Card | None


@dataclass
class TrackerDelta:
    """What one :meth:`ReservationTracker.update` changed."""

    frame_seq: int
    changed: bool
    events: list[ReservedEvent] = field(default_factory=list)


def _face_code(info: CardInfo, deck_id: int | None) -> str | None:
    """
    Registry code of a DOM face, or None when it matches no engine card.

    Table cards know their tier (the row they sit in); my reserved cards
    must be probed across the three tiers - the (colour, points, cost)
    triple is unique across all 90 cards (P0 registry ruling).
    """
    tiers = range(3) if deck_id is None else (deck_id,)
    for tier in tiers:
        try:
            return lookup_card(
                tier, info["colour"], info["points"], info["cost"]
            ).code
        except KeyError:
            continue
    return None


def _multiset_added(prev: list[int], curr: list[int]) -> list[int]:
    """Tiers added between the two back lists (multiset difference)."""
    remaining = list(prev)
    added: list[int] = []
    for tier in curr:
        if tier in remaining:
            remaining.remove(tier)
        else:
            added.append(tier)
    return added


def _panel_of(snapshot: Snapshot, seat: int) -> PanelInfo | None:
    for panel in snapshot["panels"]:
        if panel["seat"] == seat:
            return panel
    return None


def _card_count_increases(
    prev: Snapshot | None, curr: Snapshot, seat: int
) -> set[str]:
    """Colours whose permanent-card count rose for ``seat`` in this diff."""
    before = _panel_of(prev, seat) if prev is not None else None
    after = _panel_of(curr, seat)
    if before is None or after is None:
        return set()
    return {
        colour
        for colour, count in after["card_counts"].items()
        if count > before["card_counts"].get(colour, 0)
    }


def _pick_retired(entries: list[TrackedReserved], gained: set[str]) -> int | None:
    """
    Which entry a purchase retired: a known face whose colour matches the
    panel's card-count increase, else the oldest unknown, else the oldest.
    """
    for index, entry in enumerate(entries):
        if entry.card is not None and entry.card.colour in gained:
            return index
    for index, entry in enumerate(entries):
        if entry.card is None:
            return index
    return 0 if entries else None


class ReservationTracker:
    """
    Per-seat memory of rivals' reserved cards plus the global seen-face set.

    Feed every adopted frame (advisor observer) through :meth:`update`; the
    engine consumes the view via :meth:`reserved` / :meth:`seen_face_codes`.
    """

    def __init__(self, evidence: ReserveEvidence | None = None) -> None:
        self._evidence = evidence
        self._tracked: dict[int, list[TrackedReserved]] = {}
        self._seen: set[str] = set()
        self._events: list[ReservedEvent] = []
        self._prev: Snapshot | None = None
        self._saw_score = False

    # ----- consumption API (engine / UI) -----------------------------------
    def reserved(self, seat: int) -> list[TrackedReserved]:
        """Memory entries for one 1-based rival seat (empty when unknown)."""
        return list(self._tracked.get(seat, []))

    def known_reserved_faces(self) -> list[Card]:
        """Identified rival-reserved faces across all seats."""
        return [
            entry.card
            for entries in self._tracked.values()
            for entry in entries
            if entry.card is not None
        ]

    def unknown_reserved_count(self) -> int:
        """Entries whose face never became known (the grey bucket)."""
        return sum(
            1
            for entries in self._tracked.values()
            for entry in entries
            if entry.card is None
        )

    def seen_face_codes(self) -> set[str]:
        """Every card face ever observed - the complement of the deck."""
        return set(self._seen)

    def events_log(self) -> list[ReservedEvent]:
        """Full event history (survives resets, for the honesty lines)."""
        return list(self._events)

    def reset(self, reason: str) -> None:
        """
        Clear all memory (dashboard's manual reset).

        ``reason`` labels the reset in the event log; the seen-face set is
        cleared with the entries because a new game reshuffles everything.
        """
        self._clear_memory()
        self._events.append(
            ReservedEvent(seat=0, kind=EVENT_RESET, tier=None, card=None)
        )
        _ = reason  # kept for caller-side logs / future structured events

    def _clear_memory(self) -> None:
        self._tracked.clear()
        self._seen.clear()
        self._prev = None
        self._saw_score = False

    # ----- ingestion --------------------------------------------------------
    def update(self, snapshot: Snapshot, frame_seq: int) -> TrackerDelta:
        """
        Fold one adopted frame into memory.

        Mid-join safe: the first frame containing a seat initialises its
        entries as unknown (count + tier are all the DOM leaks). A frame
        where every score returned to 0 after any positive score is judged
        a new game and resets memory.
        """
        events: list[ReservedEvent] = []
        self._record_seen_faces(snapshot)

        if self._saw_score and all(p["score"] == 0 for p in snapshot["panels"]):
            self._clear_memory()
            events.append(
                ReservedEvent(seat=0, kind=EVENT_RESET, tier=None, card=None)
            )
        self._saw_score = any(p["score"] > 0 for p in snapshot["panels"])

        my_seat = snapshot["my_seat"]
        for panel in snapshot["panels"]:
            seat = panel["seat"]
            if seat == my_seat:
                continue  # my reserved cards are read directly from the DOM
            curr = list(panel["reserved_tiers"])
            if seat not in self._tracked:
                self._tracked[seat] = []
                for tier in curr:
                    self._tracked[seat].append(
                        TrackedReserved(tier=tier, card=None, frame_seq=frame_seq)
                    )
                    events.append(
                        ReservedEvent(seat=seat, kind=EVENT_INIT, tier=tier, card=None)
                    )
            else:
                prev = self._prev_tiers(seat)
                new_tiers = _multiset_added(prev, curr)
                gone = len(prev) - len(curr)
                if new_tiers:
                    events.extend(
                        self._track_reserves(snapshot, seat, new_tiers, frame_seq)
                    )
                if gone > 0:
                    events.extend(self._track_purchase(snapshot, seat, gone))
        self._prev = snapshot
        self._events.extend(events)
        return TrackerDelta(frame_seq=frame_seq, changed=bool(events), events=events)

    # ----- internals ----------------------------------------------------------
    def _record_seen_faces(self, snapshot: Snapshot) -> None:
        for deck_id, row in enumerate(snapshot["dealt"]):
            for info in row:
                if info is not None:
                    code = _face_code(info, deck_id)
                    if code is not None:
                        self._seen.add(code)
        for info in snapshot["my_reserved"]:
            code = _face_code(info, None)
            if code is not None:
                self._seen.add(code)

    def _prev_tiers(self, seat: int) -> list[int]:
        panel = _panel_of(self._prev, seat) if self._prev is not None else None
        return list(panel["reserved_tiers"]) if panel is not None else []

    def _vanished_faces(
        self, prev: Snapshot, curr: Snapshot
    ) -> list[tuple[int, CardInfo]]:
        """Dealt faces present in ``prev`` but gone in ``curr``, by tier."""
        vanished: list[tuple[int, CardInfo]] = []
        for deck_id in range(3):
            for before, after in zip(
                prev["dealt"][deck_id], curr["dealt"][deck_id], strict=True
            ):
                if before is not None and after is None:
                    vanished.append((deck_id, before))
        return vanished

    def _track_reserves(
        self, snapshot: Snapshot, seat: int, new_tiers: list[int], frame_seq: int
    ) -> list[ReservedEvent]:
        events: list[ReservedEvent] = []
        prev = self._prev
        assert prev is not None  # callers: the seat was already tracked
        vanished = self._vanished_faces(prev, snapshot)
        for tier in new_tiers:
            # Table reserve: exactly one vanished dealt face of this tier.
            matching = [info for (t, info) in vanished if t == tier]
            card: Card | None = None
            kind = EVENT_RESERVE_UNKNOWN
            if len(matching) == 1:
                code = _face_code(matching[0], tier)
                if code is not None:
                    card, kind = _CODE_TO_CARD[code], EVENT_RESERVE_TABLE
            if card is None and not matching:
                # Deck reserve: the face comes only from the reveal evidence.
                candidate = (
                    self._evidence(prev, snapshot, seat, tier)
                    if self._evidence is not None
                    else None
                )
                if candidate is not None and candidate.deck_id == tier:
                    card, kind = candidate, EVENT_RESERVE_DECK
            self._tracked.setdefault(seat, []).append(
                TrackedReserved(tier=tier, card=card, frame_seq=frame_seq)
            )
            if card is not None:
                self._seen.add(card.code)
            events.append(ReservedEvent(seat=seat, kind=kind, tier=tier, card=card))
        return events

    def _track_purchase(
        self, snapshot: Snapshot, seat: int, gone: int
    ) -> list[ReservedEvent]:
        events: list[ReservedEvent] = []
        entries = self._tracked.setdefault(seat, [])
        gained_colours = _card_count_increases(self._prev, snapshot, seat)
        for _ in range(min(gone, len(entries))):
            index = _pick_retired(entries, gained_colours)
            if index is None:
                break
            retired = entries.pop(index)
            events.append(
                ReservedEvent(
                    seat=seat, kind=EVENT_PURCHASE, tier=retired.tier, card=retired.card
                )
            )
        return events
