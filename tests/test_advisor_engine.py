"""
Offline tests for the advisor's advice engine (plan phase-7 T7.4):

* GA top-1 must equal ``GeneAlgoAgent.SelectAction`` on the same state;
* the determinized reconstruction must conserve the page's deck counts and
  place tracked reserved faces exactly;
* minimax top-k must be deterministic and its top-1 value must equal the
  plain (deepcopy + successor chain) computation;
* deck histogram, affordability and noble progress arithmetic.

All offline: real engine states from seeded ``SplendorGameRule`` instances
plus synthetic snapshots - no browser, no network.
"""

import copy
import random
from typing import Any

import numpy as np

from splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent import (
    GeneAlgoAgent,
)
from splendor.agents.our_agents.minmax import MiniMaxAgent
from splendor.browser.advisor.engine import AdvisorEngine
from splendor.browser.advisor.tracker import ReservationTracker
from splendor.browser.card_registry import CARD_REGISTRY
from splendor.browser.dom_extractor import COLOR_INDEX_TO_NAME, FACE_INDEX_TO_NAME
from splendor.splendor.splendor_model import Card, SplendorGameRule

_T0_CARDS = sorted((c for c in CARD_REGISTRY.values() if c.deck_id == 0), key=lambda c: c.code)
_BLUE_T0 = next(c for c in _T0_CARDS if c.colour == "blue")
_RED_T0 = next(c for c in _T0_CARDS if c.colour == "red")
_T1_CARDS = sorted((c for c in CARD_REGISTRY.values() if c.deck_id == 1), key=lambda c: c.code)
_GREEN_T1 = next(c for c in _T1_CARDS if c.colour == "green")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _info(card: Card) -> dict[str, Any]:
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
    gems: dict[str, int] | None = None,
    reserved_tiers: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "seat": seat,
        "score": score,
        "card_counts": card_counts or dict.fromkeys(FACE_INDEX_TO_NAME, 0),
        "gems": gems or dict.fromkeys(COLOR_INDEX_TO_NAME, 0),
        "reserved_tiers": reserved_tiers or [],
    }


def _snapshot(  # noqa: PLR0913 - fixture builder mirrors the Snapshot shape
    panels: list[dict[str, Any]],
    *,
    dealt: list[list[Any]] | None = None,
    deck_counts: tuple[int, int, int] = (36, 26, 16),
    my_seat: int = 1,
    my_reserved: list[dict[str, Any]] | None = None,
    status: str = "等待你操作",
) -> dict[str, Any]:
    # Deep-copy on purpose: the tracker/engine diff consecutive snapshots,
    # and the real extract hands them a fresh object every frame - tests
    # must not get away with mutating one shared dict between frames.
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


def _opening_snapshot() -> dict[str, Any]:
    """A plausible 2-player opening: 12 dealt cards, full decks minus dealt."""
    dealt = [
        [_info(_BLUE_T0), _info(_RED_T0), None, None],
        [_info(_GREEN_T1), None, None, None],
        [None, None, None, None],
    ]
    return _snapshot(
        [_panel(1), _panel(2)],
        dealt=dealt,
        deck_counts=(38, 29, 20),
    )


# ----- GA consistency --------------------------------------------------------
def test_ga_top1_matches_gene_algo_agent() -> None:
    _seed_all(11)
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    legal = rule.getLegalActions(state, 0)

    agent = GeneAlgoAgent(0)
    expected = agent.SelectAction(list(legal), state, rule)

    engine = AdvisorEngine(2, 0, seed=3)
    advice = engine.ga_top_k(state)
    assert advice
    assert advice[0].action == expected


def test_ga_ranking_is_deterministic_and_sorted() -> None:
    _seed_all(11)
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    engine = AdvisorEngine(2, 0, seed=3)
    first = engine.ga_top_k(state)
    second = engine.ga_top_k(state)
    assert [(a.text, a.value) for a in first] == [(a.text, a.value) for a in second]
    values = [a.value for a in first]
    assert values == sorted(values, reverse=True)
    assert first[0].reasons  # top-1 carries its feature attribution


def test_reconstruction_conserves_deck_counts() -> None:
    tracker = ReservationTracker()
    snapshot_any: dict[str, Any] = _opening_snapshot()
    tracker.update(snapshot_any, frame_seq=1)
    engine = AdvisorEngine(2, 0, seed=5)
    state = engine.build_reconstruction(snapshot_any, tracker)  # type: ignore[arg-type]
    for tier in range(3):
        assert len(state.board.decks[tier]) == snapshot_any["deck_counts"][tier]
    # Dealt cards restored at their slots, excluded from the decks.
    assert state.board.dealt[0][0] is not None
    assert state.board.dealt[0][0].code == _BLUE_T0.code
    dealt_codes = {
        card.code for row in state.board.dealt for card in row if card is not None
    }
    deck_codes = {card.code for tier in state.board.decks for card in tier}
    assert dealt_codes.isdisjoint(deck_codes)
    # Same state -> same reconstruction (fingerprint equality).
    state_again = engine.build_reconstruction(snapshot_any, tracker)  # type: ignore[arg-type]
    from splendor.agents.our_agents.alphazero.state_utils import state_fingerprint

    assert state_fingerprint(state) == state_fingerprint(state_again)


def test_reconstruction_places_tracked_faces() -> None:
    tracker = ReservationTracker()
    # Frame 1: one tier-0 card dealt (deck 40-1=39), nothing reserved.
    dealt_before = [
        [_info(_BLUE_T0), None, None, None],
        [None, None, None, None],
        [None, None, None, None],
    ]
    # Frame 2: the blue card moved to the rival's reserve (table reserve,
    # face known) plus a deck-top reserve from tier 2 (deck 20-1=19).
    dealt_after = [
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
    ]
    panels = [_panel(1), _panel(2)]
    tracker.update(_snapshot(panels, dealt=dealt_before, deck_counts=(39, 30, 20)), frame_seq=1)
    panels[1]["reserved_tiers"] = [0, 2]
    tracker.update(_snapshot(panels, dealt=dealt_after, deck_counts=(39, 30, 19)), frame_seq=2)

    snapshot_now: dict[str, Any] = _snapshot(
        panels, dealt=dealt_after, deck_counts=(39, 30, 19)
    )
    engine = AdvisorEngine(2, 0, seed=5)
    state = engine.build_reconstruction(snapshot_now, tracker)  # type: ignore[arg-type]
    rival = state.agents[1]
    yellow_codes = [card.code for card in rival.cards["yellow"]]
    assert _BLUE_T0.code in yellow_codes  # the tracked table reserve
    # The unknown tier-2 back was sampled *out of* the tier-2 unseen pool,
    # so every deck conserves its page count exactly.
    assert [len(state.board.decks[t]) for t in range(3)] == [39, 30, 19]
    assert all(card.deck_id == 2 for card in state.board.decks[2])


def test_minimax_top_k_matches_plain_computation() -> None:
    _seed_all(21)
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    engine = AdvisorEngine(2, 0, seed=9)
    advice = engine.minimax_top_k(state, depth=2)
    assert advice
    values = [a.value for a in advice]
    assert values == sorted(values, reverse=True)

    # Plain reference: deepcopy + successor chain + min over replies.
    best_action = advice[0].action
    after_root = copy.deepcopy(state)
    rule.generateSuccessor(after_root, best_action, 0)
    evaluator = MiniMaxAgent(0)._evaluation_function  # noqa: SLF001
    replies = rule.getLegalActions(after_root, 1)
    expected = min(
        evaluator(rule.generateSuccessor(copy.deepcopy(after_root), reply, 1))
        for reply in replies
    )
    assert advice[0].value == expected
    assert advice[0].reasons and advice[0].reasons[0].startswith("对手最狠回应")


def test_minimax_is_deterministic_across_runs() -> None:
    _seed_all(21)
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    engine = AdvisorEngine(2, 0, seed=9)
    first = engine.minimax_top_k(state, depth=2)
    second = engine.minimax_top_k(state, depth=2)
    assert [(a.text, a.value) for a in first] == [(a.text, a.value) for a in second]


def test_minimax_rejects_non_two_seat_states() -> None:
    _seed_all(31)
    rule3 = SplendorGameRule(3)
    engine = AdvisorEngine(3, 0, seed=1)
    try:
        engine.minimax_top_k(rule3.current_game_state)
    except ValueError as error:
        assert "2 seats" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for 3 seats")


def test_deck_histogram_conserves_and_warns() -> None:
    tracker = ReservationTracker()
    snapshot_any: dict[str, Any] = _snapshot(
        [_panel(1), _panel(2)], deck_counts=(40, 30, 20)
    )
    tracker.update(snapshot_any, frame_seq=1)
    engine = AdvisorEngine(2, 0, seed=1)

    histogram = engine.deck_histogram(snapshot_any, tracker)  # type: ignore[arg-type]
    assert not histogram.warnings
    # No faces seen: registry totals (40/30/20) with grey 0.
    assert sum(histogram.rows[0].counts.values()) == 40
    assert sum(histogram.rows[1].counts.values()) == 30
    assert sum(histogram.rows[2].counts.values()) == 20
    assert all(row.grey == 0 for row in histogram.rows)

    # Page deck count above the accounted unseen -> conservation warning.
    inflated: dict[str, Any] = _snapshot(
        [_panel(1), _panel(2)], deck_counts=(45, 30, 20)
    )
    histogram_bad = engine.deck_histogram(inflated, tracker)  # type: ignore[arg-type]
    assert histogram_bad.warnings
    assert histogram_bad.rows[0].grey == 0


def test_deck_histogram_grey_bucket_counts_unknown_reserves() -> None:
    tracker = ReservationTracker()
    # One tier-2 card left the deck into a hidden reserve slot: the page
    # reads 19 while 20 unseen faces remain unaccounted for.
    panels = [_panel(1), _panel(2, reserved_tiers=[2])]
    snapshot_any: dict[str, Any] = _snapshot(panels, deck_counts=(40, 30, 19))
    tracker.update(snapshot_any, frame_seq=1)
    engine = AdvisorEngine(2, 0, seed=1)
    histogram = engine.deck_histogram(snapshot_any, tracker)  # type: ignore[arg-type]
    assert histogram.rows[2].grey == 1
    assert not histogram.warnings


def test_affordability_rows() -> None:
    # I produce 2 white, hold 1 white + 1 gold.
    my_panel = _panel(
        1,
        card_counts={"white": 2, "blue": 0, "green": 0, "red": 0, "black": 0},
        gems={"white": 1, "yellow": 1, "blue": 0, "green": 0, "red": 0, "black": 0},
    )
    cost3 = {**_info(_RED_T0), "cost": {"white": 3}}  # affordable via production+gem
    cost4 = {**_info(_GREEN_T1), "cost": {"white": 4}}  # short 1, gold covers
    cost5 = {**_info(_BLUE_T0), "cost": {"white": 5}}  # short 2 > gold 1
    dealt = [[cost3, cost4, cost5, None], [None] * 4, [None] * 4]
    snapshot_any: dict[str, Any] = _snapshot([my_panel, _panel(2)], dealt=dealt)

    engine = AdvisorEngine(2, 0, seed=1)
    rows = engine.affordability(snapshot_any)  # type: ignore[arg-type]
    assert [row.affordable for row in rows] == [True, True, False]
    assert rows[1].missing == {"white": 1} and rows[1].gold_covers == 1
    assert rows[2].missing == {"white": 2} and rows[2].gold_covers == 1
    assert rows[0].source == "dealt"


def test_affordability_counts_my_reserved_cards() -> None:
    my_panel = _panel(1)
    snapshot_any: dict[str, Any] = _snapshot(
        [my_panel, _panel(2)], my_reserved=[_info(_RED_T0)]
    )
    engine = AdvisorEngine(2, 0, seed=1)
    rows = engine.affordability(snapshot_any)  # type: ignore[arg-type]
    assert rows[-1].source == "reserved"
    assert rows[-1].text.endswith("卡")


def test_noble_progress_missing_colours() -> None:
    my_panel = _panel(1, card_counts={"white": 1, "blue": 0, "green": 0, "red": 0, "black": 0})
    snapshot_any: dict[str, Any] = _snapshot(
        [my_panel, _panel(2)],
    )
    snapshot_any["nobles"] = [{"requirements": {"white": 3, "blue": 2}}]
    engine = AdvisorEngine(2, 0, seed=1)
    progress = engine.noble_progress(snapshot_any)  # type: ignore[arg-type]
    assert progress[0]["missing"] == {"white": 2, "blue": 2}
