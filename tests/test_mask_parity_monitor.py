"""
MaskParityMonitor attribution tests (plan phase-2 A2.3): one constructed
difference per category - known rule difference (E5/E6 registry), DOM
extraction bug, page redesign - each attributed correctly, plus the
agreement and over-approximation residuals.
"""

import numpy as np
from numpy.typing import NDArray

from splendor.browser.dom_extractor import Snapshot
from splendor.browser.monitor import (
    DEFAULT_KNOWN_DIFFERENCES,
    MaskParityMonitor,
    dom_affordances,
)
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, ActionEnum

ACTION_SPACE = len(ALL_ACTIONS)


def _mask(indices: set[int]) -> NDArray:
    mask = np.zeros(ACTION_SPACE)
    for index in indices:
        mask[index] = 1
    return mask


def _indices_of(*types: ActionEnum, limit: int = 6) -> set[int]:
    found = {
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum in types and action.noble_index is None
    }
    return set(sorted(found)[:limit])


def _collect_index(collected: dict[str, int]) -> int:
    return next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.COLLECT_DIFF
        and action.collected_gems == collected
        and action.noble_index is None
    )


def _buy_index(tier: int, column: int) -> int:
    return next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.BUY_AVAILABLE
        and action.position is not None
        and action.position.tier == tier
        and action.position.card_index == column
        and action.noble_index is None
    )


PASS_INDEX = 0


def test_agreement_reported_as_ok() -> None:
    engine_legal = {PASS_INDEX, _collect_index({"white": 1, "blue": 1, "green": 1})}
    report = MaskParityMonitor().check(_mask(engine_legal), engine_legal)
    assert any("OK" in line for line in report)


def test_known_rule_difference_attributed_to_registry() -> None:
    """
    E5: the DOM affords a single-colour take the engine forbids (its forced
    minimum when the hand is light); E6: the DOM affords a buy the engine
    forbids (same-colour cap). Both land on the registered whitelist.

    The buckets are type-matched, so the report says "class ... bucket, not
    confirmed instances": a count there is a candidate-class size, not a
    verified divergence count.
    """
    engine_legal = {PASS_INDEX, _collect_index({"white": 1, "blue": 1, "green": 1})}
    dom = engine_legal | {_collect_index({"white": 1}), _buy_index(0, 0)}

    report = MaskParityMonitor().check(_mask(engine_legal), dom)
    joined = "\n".join(report)
    assert "E5" in joined
    assert "E6" in joined
    assert "KNOWN-DIFFERENCE CLASS" in joined
    assert "not confirmed instances" in joined
    # every engine-legal action is DOM-supported here, and the header says so
    assert "direction safe" in joined


def test_direction_note_absent_when_the_engine_asks_the_impossible() -> None:
    """The header must NOT claim safety while engine-only is non-zero."""
    engine_legal = {PASS_INDEX, _buy_index(0, 0)}
    report = MaskParityMonitor().check(_mask(engine_legal), {PASS_INDEX})
    joined = "\n".join(report)
    assert "engine-only=1" in joined
    assert "direction safe" not in joined
    assert "DOM EXTRACTION BUG" in joined


def test_dom_extraction_bug_attributed() -> None:
    """
    Engine-legal actions the measured DOM cannot support: the extraction is
    dropping facts (e.g. chip counts read as zero).
    """
    engine_legal = {
        PASS_INDEX,
        _collect_index({"white": 1, "blue": 1, "green": 1}),
        _buy_index(0, 0),
    }
    dom = {PASS_INDEX}  # only the pass button is found

    report = MaskParityMonitor().check(_mask(engine_legal), dom)
    joined = "\n".join(report)
    assert "DOM EXTRACTION BUG" in joined
    assert "PAGE REDESIGN" not in joined


def test_page_redesign_attributed_when_dom_is_empty() -> None:
    engine_legal = {PASS_INDEX, _collect_index({"white": 1, "blue": 1, "green": 1})}
    report = MaskParityMonitor().check(_mask(engine_legal), set())
    assert any("PAGE REDESIGN" in line for line in report)


def test_page_redesign_attributed_when_pass_button_vanishes() -> None:
    engine_legal = {PASS_INDEX, _collect_index({"white": 1, "blue": 1, "green": 1})}
    dom = engine_legal - {PASS_INDEX}
    report = MaskParityMonitor().check(_mask(engine_legal), dom)
    joined = "\n".join(report)
    assert "PAGE REDESIGN" in joined
    # the remaining dom-only residual must not be mislabelled as a bug
    assert "DOM EXTRACTION BUG" not in joined


def test_unregistered_differences_surface_for_triage() -> None:
    """A custom registry without E5 must leave collect extras visible."""
    monitor = MaskParityMonitor(known_differences=[])
    engine_legal = {PASS_INDEX, _collect_index({"white": 1, "blue": 1, "green": 1})}
    dom = engine_legal | {_collect_index({"white": 1})}
    report = monitor.check(_mask(engine_legal), dom)
    joined = "\n".join(report)
    assert "over-approximation" in joined
    assert "COLLECT_DIFF x1" in joined


def test_dom_affordances_over_approximate_legality() -> None:
    """
    Mechanical sanity of the affordance collector: a snapshot with full
    supply and a card at tier 0/col 0 affords the matching collect/buy, and
    everything with an absent target stays out.
    """
    from splendor.browser.card_registry import lookup_card

    assert lookup_card(0, "white", 0, {"blue": 3}).deck_id == 0  # real face
    snapshot: Snapshot = {
        "dealt": [
            [
                {"tier": 0, "colour": "white", "points": 0, "cost": {"blue": 3}},
                None,
                None,
                None,
            ],
            [None] * 4,
            [None] * 4,
        ],
        "deck_counts": [0, 0, 0],
        "nobles": [],
        "supply": {"white": 4, "blue": 4, "green": 0, "red": 0, "black": 0, "yellow": 5},
        "panels": [
            {
                "seat": 1,
                "score": 0,
                "card_counts": {},
                "gems": {},
                "reserved_tiers": [],
            }
        ],
        "my_seat": 1,
        "my_reserved": [],
        "status": "等待你操作",
        "payment_options": None,
        "noble_options": None,
    }
    afford = dom_affordances(snapshot)

    assert PASS_INDEX in afford
    assert _collect_index({"white": 1}) in afford
    assert _buy_index(0, 0) in afford
    assert _collect_index({"green": 1}) not in afford  # no green chips
    assert _buy_index(1, 0) not in afford  # no card at tier 1


def test_default_registry_covers_e5_and_e6() -> None:
    codes = {known.code for known in DEFAULT_KNOWN_DIFFERENCES}
    assert {"E5", "E6"} <= codes
    for known in DEFAULT_KNOWN_DIFFERENCES:
        assert known.description


def test_check_tolerates_empty_affordance_set() -> None:
    engine_legal = {PASS_INDEX}
    report = MaskParityMonitor().check(_mask(engine_legal), set())
    assert isinstance(report, list)
    assert report
