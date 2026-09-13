"""
Offline tests for the advisor's read-only observation loop (plan phase-7
T7.1): fixture HTML -> MockBrowserDriver -> AdvisorSession frames, phase
classification, the debounce rule and the package-level zero-click
invariant - all without touching the network.
"""

import ast
from pathlib import Path

from splendor.browser.advisor.observer import (
    DEFAULT_FAST_POLL_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    AdvisorSession,
    Phase,
    classify_phase,
)
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import SNAPSHOT_JS_MARKER, MockBrowserDriver

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"
REPO = Path(__file__).parent.parent
OPENING = (FIXTURES / "opening.html").read_text(encoding="utf-8")
PAYMENT = (FIXTURES / "payment_pills.html").read_text(encoding="utf-8")
GAME_OVER = (FIXTURES / "game_over.html").read_text(encoding="utf-8")

# The advisor package surface (package modules + the console entry): none of
# it may reference the action executor - the zero-click invariant is a
# source-level property, checked here so a future "convenience" import fails
# CI instead of clicking the real page.
ADVISOR_SOURCES = [
    *sorted((REPO / "src" / "splendor" / "browser" / "advisor").glob("*.py")),
    REPO / "src" / "splendor" / "play_advisor.py",
]

_OPPONENT_TURN = OPENING.replace("等待你操作", "等待玩家2操作", 1)


class _RotatingDriver(MockBrowserDriver):
    """Serves the next page of a list on every snapshot read (holds last)."""

    def __init__(self, pages: list[str]) -> None:
        super().__init__()
        # Not the base class's _pages (its url->html map): this is the
        # temporal page sequence for snapshot reads.
        self._sequence = pages
        self._read_count = 0

    def evaluate(self, js: str) -> object:
        if SNAPSHOT_JS_MARKER in js:
            index = min(self._read_count, len(self._sequence) - 1)
            self._read_count += 1
            self.set_html(self._sequence[index])
        return super().evaluate(js)


def _extract(page: str) -> Snapshot:
    driver = MockBrowserDriver()
    driver.set_html(page)
    return extract_snapshot(driver)


def test_adopt_requires_two_identical_reads() -> None:
    session = AdvisorSession(_RotatingDriver([OPENING]))
    assert session.poll_once() is False  # first read only arms the debounce
    assert session.poll_once() is True  # identical second read adopts
    assert session.frame is not None
    assert session.frame.frame_seq == 1
    assert session.frame.phase is Phase.MY_TURN
    # opening.html: my panel is the first my-2 block -> seat 1
    assert session.frame.my_index == 0
    assert session.poll_once() is False  # unchanged page: no new frame


def test_transient_page_is_never_adopted() -> None:
    # my-turn -> opponent-acting -> opponent-acting: the first (transient)
    # read must never surface as a frame; only the stable page is adopted.
    session = AdvisorSession(_RotatingDriver([OPENING, _OPPONENT_TURN]))
    assert session.poll_once() is False
    assert session.poll_once() is False
    assert session.poll_once() is True
    assert session.frame is not None
    assert session.frame.phase is Phase.OPPONENT_TURN
    assert session.frame.waiting_seat == 2
    assert session.frame.my_index == 0


def test_opponent_turn_polls_faster() -> None:
    session = AdvisorSession(_RotatingDriver([OPENING, _OPPONENT_TURN]))
    assert session.sleep_interval() == DEFAULT_POLL_INTERVAL
    assert session.poll_once() is False  # read 1: opening (arm)
    assert session.poll_once() is False  # read 2: opponent page (arm)
    assert session.poll_once() is True  # read 3: opponent page stable
    assert session.frame is not None
    assert session.sleep_interval() == DEFAULT_FAST_POLL_INTERVAL


def test_phase_payment_pauses_advice() -> None:
    assert classify_phase(_extract(PAYMENT)) is Phase.PAYMENT


def test_phase_discard_subflow() -> None:
    page = OPENING.replace(
        "等待你操作", "等待你丢弃多余宝石（每人最多持有10个）", 1
    )
    assert classify_phase(_extract(page)) is Phase.MY_DISCARD


def test_phase_noble_subflow() -> None:
    page = OPENING.replace("等待你操作", "等待你选择要获得的贵族卡", 1)
    assert classify_phase(_extract(page)) is Phase.MY_NOBLE


def test_phase_no_board_on_room_page() -> None:
    assert classify_phase(_extract(GAME_OVER)) is Phase.NO_BOARD


def test_unseated_snapshot_has_no_my_index() -> None:
    # game_over.html keeps panels but no board; dropping the 我 marker makes
    # the page a spectator view -> my_seat 0 -> my_index None.
    session = AdvisorSession(_RotatingDriver([GAME_OVER.replace("我", "", 1)]))
    assert session.poll_once() is False
    assert session.poll_once() is True
    assert session.frame is not None
    assert session.frame.my_index is None


def test_advisor_never_references_the_executor() -> None:
    for path in ADVISOR_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                assert "action_executor" not in module, (
                    f"{path} imports {module!r} - the advisor must stay "
                    "zero-click (plan phase-7 §6)"
                )


def test_session_stop_after_reaches_max_frames() -> None:
    session = AdvisorSession(_RotatingDriver([OPENING]))
    stop = session.stop_after(1)
    assert stop() is False
    assert session.poll_once() is False
    assert session.poll_once() is True
    assert stop() is True
