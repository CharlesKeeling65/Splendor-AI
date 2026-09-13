"""
Read-only observation loop for the browser advisor (plan phase-7 §3.1).

The advisor's contract with the real page is strictly one-way: every
interaction is :func:`extract_snapshot` - a passive CDP ``evaluate`` that
reads DOM facts and writes nothing. No executor import, no clicks, at any
phase (the package docstring and a source-level test enforce this).

Frames are *debounced*: a snapshot is adopted only after two consecutive
reads agree, so advice is never computed from an animation mid-state. While
the page waits on an opponent the poll runs faster - the opponent's
reserve-moment reveal that the tracker wants to catch (T7.2/E7) is
transient, and a passive read at 0.25s stays polite (zero writes).
"""

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

from splendor.browser.dom_extractor import (
    PHASE_DISCARD_TEXT,
    PHASE_NOBLE_TEXT,
    Snapshot,
    SnapshotSchemaError,
    extract_snapshot,
    is_my_turn,
    waiting_seat,
)
from splendor.browser.driver import BrowserDriver

DEFAULT_POLL_INTERVAL = 0.4
# Opponent acting: the reserve reveal (if E7 confirms one) is transient.
# A passive read-only evaluate at this cadence is far below any human pace.
DEFAULT_FAST_POLL_INTERVAL = 0.25


class Phase(enum.Enum):
    """What the page is waiting for, from the advisor's point of view."""

    NO_BOARD = "no_board"  # room page: game not started, or already over
    MY_TURN = "my_turn"  # 等待你操作 - the ordinary decision point
    MY_DISCARD = "my_discard"  # >10-gem discard sub-flow (human must click)
    MY_NOBLE = "my_noble"  # multi-noble choice sub-flow (human must click)
    PAYMENT = "payment"  # payment pill sub-flow (human must choose)
    OPPONENT_TURN = "opponent_turn"
    UNKNOWN = "unknown"  # transitional copy the grammar does not cover


PHASE_LABELS: dict[Phase, str] = {
    Phase.NO_BOARD: "未检测到对局棋盘（开局前或终局）",
    Phase.MY_TURN: "我的回合",
    Phase.MY_DISCARD: "我的丢弃子流程（请手动选择要丢弃的宝石）",
    Phase.MY_NOBLE: "我的贵族选择子流程（请手动点选贵族）",
    Phase.PAYMENT: "我的支付选择子流程（请手动选择支付方式；建议暂停）",
    Phase.OPPONENT_TURN: "对手回合",
    Phase.UNKNOWN: "过渡状态",
}


def _board_absent(snapshot: Snapshot) -> bool:
    """The measured room view (E3): no dealt cards and no deck piles."""
    rows_empty = all(
        card is None for row in snapshot["dealt"] for card in row
    )
    decks_empty = not any(snapshot["deck_counts"])
    return rows_empty and decks_empty


def classify_phase(snapshot: Snapshot) -> Phase:
    """
    Classify the snapshot into an advisory phase.

    The payment/noble flags are only trusted alongside my-turn status:
    candidate nobles and payment pills render on the *chooser's* screen,
    and the advisor only ever advises its own seat.
    """
    status = snapshot["status"]
    if (
        _board_absent(snapshot)
        and not is_my_turn(status)
        and waiting_seat(status) is None
    ):
        return Phase.NO_BOARD
    if is_my_turn(status):
        return _my_phase(snapshot)
    if waiting_seat(status) is not None:
        return Phase.OPPONENT_TURN
    return Phase.UNKNOWN


def _my_phase(snapshot: Snapshot) -> Phase:
    """Which of my own sub-flows (if any) the page is waiting on."""
    status = snapshot["status"]
    if snapshot["payment_options"] is not None:
        return Phase.PAYMENT
    if snapshot["noble_options"] is not None or PHASE_NOBLE_TEXT in status:
        return Phase.MY_NOBLE
    if PHASE_DISCARD_TEXT in status:
        return Phase.MY_DISCARD
    return Phase.MY_TURN


@dataclass(frozen=True)
class AdvisorFrame:
    """One adopted (debounce-stable) observation of the page."""

    snapshot: Snapshot
    phase: Phase
    my_index: int | None  # None while not seated (my_seat == 0)
    waiting_seat: int | None  # the seat the status text says is acting
    frame_seq: int  # monotonically increasing adopted-frame counter


class AdvisorSession:
    """
    Debounced read-only observation over one browser driver.

    ``poll_once`` performs a single ``extract_snapshot``; a snapshot is only
    *adopted* (becomes ``self.frame``, fires the consumer callback) when two
    consecutive reads agree byte-for-byte - the animation mid-state guard.
    Consumers get frames through :meth:`run`; tests drive ``poll_once``
    directly.
    """

    def __init__(
        self,
        driver: BrowserDriver,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        fast_poll_interval: float = DEFAULT_FAST_POLL_INTERVAL,
    ) -> None:
        self._driver = driver
        self._poll_interval = poll_interval
        self._fast_poll_interval = fast_poll_interval
        self._pending: Snapshot | None = None
        self.frame: AdvisorFrame | None = None
        self.reads = 0

    def poll_once(self) -> bool:
        """
        One extract + debounce step.

        :returns: True when a new stable frame was adopted this call.
        :raises SnapshotSchemaError: propagation is deliberate - the schema
            is the advisor's contract with the page, and a redesign must
            surface with field names (same ruling as the browser env).
        """
        self.reads += 1
        snapshot = extract_snapshot(self._driver)
        if self._pending is not None and snapshot == self._pending:
            self._pending = None
            self._adopt(snapshot)
            return True
        self._pending = snapshot
        return False

    def _adopt(self, snapshot: Snapshot) -> None:
        my_seat = snapshot["my_seat"]
        self.frame = AdvisorFrame(
            snapshot=snapshot,
            phase=classify_phase(snapshot),
            my_index=(my_seat - 1) if my_seat > 0 else None,
            waiting_seat=waiting_seat(snapshot["status"]),
            frame_seq=(self.frame.frame_seq if self.frame else 0) + 1,
        )

    def sleep_interval(self) -> float:
        """Base cadence, faster while an opponent is acting."""
        if self.frame is not None and self.frame.phase is Phase.OPPONENT_TURN:
            return self._fast_poll_interval
        return self._poll_interval

    def run(
        self,
        on_frame: Callable[[AdvisorFrame], None],
        *,
        should_stop: Callable[[], bool] | None = None,
        on_schema_error: Callable[[SnapshotSchemaError], bool] | None = None,
    ) -> None:
        """
        Poll until ``should_stop`` (default: never) or Ctrl+C.

        ``on_frame`` fires once per adopted frame. A
        :class:`SnapshotSchemaError` (page transitioning or redesigned) is
        offered to ``on_schema_error`` - returning True continues polling,
        False (or omitting the callback) re-raises: the fail-loud ruling
        from the browser env applies to the advisor too.
        """
        while should_stop is None or not should_stop():
            try:
                if self.poll_once():
                    if self.frame is None:  # pragma: no cover - adoption guarantees it
                        raise RuntimeError("adopted frame missing")
                    on_frame(self.frame)
            except SnapshotSchemaError as error:
                if on_schema_error is None or not on_schema_error(error):
                    raise
            try:
                time.sleep(self.sleep_interval())
            except KeyboardInterrupt:  # Ctrl+C during the sleep
                return

    def stop_after(self, frames: int) -> Callable[[], bool]:
        """Convenience stop predicate for offline smoke runs."""

        def _stop() -> bool:
            return self.frame is not None and self.frame.frame_seq >= frames

        return _stop


def re_raise(error: SnapshotSchemaError) -> NoReturn:
    """``on_schema_error`` callback that keeps the fail-loud default."""
    raise error
