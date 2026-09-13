"""
``play-advisor`` console script: a read-only Splendor advisor (plan
phase-7). Launches (or attaches to) one ego-browser task space, the human
plays *manually* in that window, and this process only polls the DOM and
ranks the human's legal moves.

Zero-click contract: the advisor never imports the action executor and
never mutates page state - the etiquette constraint (plan phase-2 M2.5)
is satisfied by construction, not by pacing.

Per adopted (debounce-stable) frame the CLI prints:

* the ranked GA advice (最优/次优 marked, feature attributions indented),
  plus the minimax deep ranking when ``--depth`` is set;
* the undealt-deck colour histogram per tier (the tracker's grey bucket
  included) and the affordability ruler;
* the reservation-memory coverage with its latest events.

Run against an offline fixture for a no-network smoke:
``play-advisor --fixture-html <file> --max-frames 2``.
"""

import argparse
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from splendor.browser.advisor.engine import (
    AdvisorEngine,
    DeckHistogram,
    _missing_text,
)
from splendor.browser.advisor.observer import (
    PHASE_LABELS,
    AdvisorFrame,
    AdvisorSession,
    Phase,
)
from splendor.browser.advisor.tracker import (
    ReservationTracker,
    ReservedEvent,
)
from splendor.browser.dom_extractor import (
    Snapshot,
    SnapshotSchemaError,
    looks_like_game_over,
)
from splendor.browser.driver import BrowserDriver, MockBrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.splendor.action_text import COLOR_CN

DEFAULT_TASK_SPACE = "splendor-advisor"

_ADVICE_ICONS = ("★", "☆", " ")
_MIN_BAR_SAMPLES = 2
_MIN_GAME_PANELS = 2  # below this the page is not an in-game view


def _format_scores(snapshot: Snapshot, my_seat: int) -> str:
    """``我=5 | 座2=3`` from the live panel snapshot."""
    if not snapshot["panels"]:
        return "(无面板)"
    parts = []
    for panel in snapshot["panels"]:
        label = "我" if panel["seat"] == my_seat else f"座{panel['seat']}"
        parts.append(f"{label}={panel['score']}")
    return " | ".join(parts)


def _bar(values: list[float], value: float, width: int = 10) -> str:
    """Relative-score bar inside one advice list (min..max of the set)."""
    if len(values) < _MIN_BAR_SAMPLES or max(values) == min(values):
        return "█" * width
    span = max(values) - min(values)
    filled = round((value - min(values)) / span * width)
    return "█" * filled + "·" * (width - filled)


def _histogram_lines(histogram: DeckHistogram) -> list[str]:
    lines = []
    for row in histogram.rows:
        counts = " ".join(
            f"{COLOR_CN.get(colour, colour)}{count}"
            for colour, count in sorted(row.counts.items())
            if count
        )
        grey = f" | 未见去向(灰) {row.grey}" if row.grey else ""
        lines.append(f"   T{row.tier + 1}(堆余 {row.deck_count}): {counts or '空'}{grey}")
    for warning in histogram.warnings:
        lines.append(f"   ⚠ {warning}")
    return lines


def _event_text(event: ReservedEvent) -> str:
    seat = f"座{event.seat}" if event.seat else "-"
    kind_cn = {
        "init": "接入(未知)",
        "reset": "新局重置",
        "reserve_table": "预留桌面牌(已识别)",
        "reserve_deck": "预留牌堆(已识别)",
        "reserve_unknown": "预留牌堆(未知)",
        "purchase": "购买预留牌",
    }.get(event.kind, event.kind)
    card = f" {event.card.code}" if event.card is not None else ""
    return f"{seat} {kind_cn}{card}"


class AdvisorCli:
    """
    Frame consumer behind the v0 console output.

    Wires observer frames through the tracker and the engine and emits one
    text block per relevant change. ``emit`` is injectable so tests can
    collect lines instead of watching stdout.
    """

    def __init__(
        self,
        *,
        depth: int = 0,
        top_k: int = 5,
        seed: int = 0,
        emit: Callable[[str], None] = print,
    ) -> None:
        self._depth = depth
        self._top_k = top_k
        self._seed = seed
        self._emit = emit
        self._tracker = ReservationTracker()
        self._engine: AdvisorEngine | None = None
        self._engine_key: tuple[int, int] | None = None
        self._last_key: tuple[str, str] | None = None
        self._prev_phase: Phase | None = None
        self._my_turns = 0
        self._standby_announced = False

    # ----- frame handling ------------------------------------------------------
    def on_frame(self, frame: AdvisorFrame) -> None:
        snapshot = frame.snapshot
        if frame.phase is Phase.NO_BOARD:
            self._announce_standby(frame)
            return
        self._standby_announced = False
        if not snapshot["panels"]:
            return
        if len(snapshot["panels"]) < _MIN_GAME_PANELS:
            return  # not an in-game view (single-panel fixture / lobby)
        self._tracker.update(snapshot, frame.frame_seq)
        my_index = frame.my_index
        if my_index is None:
            self._emit("   （未入座：仅展示局面与牌堆，不产生建议）")
            self._emit_block(frame, None, None)
            return
        self._ensure_engine(len(snapshot["panels"]), my_index)
        if frame.phase is Phase.MY_TURN:
            self._advise(frame)
        else:
            self._emit_block(frame, None, None)
        self._prev_phase = frame.phase

    # ----- internals -----------------------------------------------------------
    def _announce_standby(self, frame: AdvisorFrame) -> None:
        if self._standby_announced:
            return
        self._standby_announced = True
        status = frame.snapshot["status"]
        suffix = "（终局）" if looks_like_game_over(status, board_present=False) else ""
        self._emit(f"⏸ 未检测到对局棋盘{suffix}——待机轮询中")

    def _ensure_engine(self, panel_count: int, my_index: int) -> None:
        key = (panel_count, my_index)
        if self._engine is None or self._engine_key != key:
            self._engine = AdvisorEngine(
                panel_count, my_index, seed=self._seed, top_k=self._top_k
            )
            self._engine_key = key

    def _advise(self, frame: AdvisorFrame) -> None:
        assert self._engine is not None
        if self._prev_phase is not Phase.MY_TURN:
            self._my_turns += 1
        state = self._engine.build_reconstruction(
            frame.snapshot, self._tracker, turns=self._my_turns
        )
        advice = self._engine.ga_top_k(state)
        deep = (
            self._engine.minimax_top_k(state, depth=self._depth)
            if self._depth > 0
            else None
        )
        self._emit_block(frame, advice, deep)

    def _emit_block(
        self,
        frame: AdvisorFrame,
        advice: list | None,
        deep: list | None,
    ) -> None:
        assert self._engine is not None
        snapshot = frame.snapshot
        key = (self._block_key(snapshot), frame.phase.value)
        if key == self._last_key:
            return
        self._last_key = key

        self._emit(
            f"▶ {snapshot['status']} | 比分: "
            f"{_format_scores(snapshot, snapshot['my_seat'])} | "
            f"{PHASE_LABELS[frame.phase]}"
        )
        if advice is not None:
            self._emit_advice("建议（GA 快评，★最优 ☆次优）:", advice)
        if deep is not None:
            self._emit_advice(f"深算（minimax depth={self._depth}）:", deep)
        if frame.phase in (Phase.MY_DISCARD, Phase.MY_NOBLE, Phase.PAYMENT):
            self._emit("   （子流程需要手动点击，建议暂停）")
        histogram = self._engine.deck_histogram(snapshot, self._tracker)
        self._emit("  牌堆剩余（未翻开部分，按颜色）:")
        for line in _histogram_lines(histogram):
            self._emit(line)
        self._emit_afford(snapshot)
        known = len(self._tracker.known_reserved_faces())
        unknown = self._tracker.unknown_reserved_count()
        events = self._tracker.events_log()[-3:]
        self._emit(
            f"  预留记忆: 已识别 {known} / 未知 {unknown}"
            + (f" | 最近: {'; '.join(_event_text(e) for e in events)}" if events else "")
        )

    def _block_key(self, snapshot: Snapshot) -> str:
        """Cheap state signature for print dedupe (score+dealt+backs)."""
        dealt = tuple(
            (info["colour"], info["points"], tuple(sorted(info["cost"].items())))
            if info is not None
            else ()
            for row in snapshot["dealt"]
            for info in row
        )
        backs = tuple(
            tuple(panel["reserved_tiers"]) for panel in snapshot["panels"]
        )
        scores = tuple(panel["score"] for panel in snapshot["panels"])
        supply = tuple(snapshot["supply"][c] for c in sorted(snapshot["supply"]))
        return f"{scores}|{supply}|{dealt}|{backs}"

    def _emit_advice(self, title: str, advice: list) -> None:
        self._emit(f"  {title}")
        values = [a.value for a in advice]
        for rank, item in enumerate(advice, start=1):
            icon = _ADVICE_ICONS[rank - 1] if rank <= len(_ADVICE_ICONS) else " "
            self._emit(
                f"   {rank}. {icon} {item.text}  {_bar(values, item.value)} "
                f"{item.value:+.2f}"
            )
            for reason in item.reasons:
                self._emit(f"      · {reason}")

    def _emit_afford(self, snapshot: Snapshot) -> None:
        assert self._engine is not None
        rows = self._engine.affordability(snapshot)
        affordable = [row for row in rows if row.affordable]
        self._emit(
            f"  可负担: {len(affordable)}/{len(rows)} 张"
            + (
                "（" + "；".join(row.text for row in affordable[:3]) + "）"
                if affordable
                else ""
            )
        )
        for row in rows:
            if not row.affordable and row.missing:
                self._emit(
                    f"   · {row.text}（{'我的预留' if row.source == 'reserved' else '桌面'}）"
                    f" {_missing_text(row.missing, row.gold_covers)}"
                )
                break  # one closest-gap line is enough on the console


def _parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="play-advisor",
        description=(
            "Read-only Splendor advisor: watch the seat you play manually "
            "and print ranked move advice (plan phase-7)."
        ),
    )
    parser.add_argument(
        "--room-url", type=str, default=None,
        help="Room to navigate to on start (omit to navigate manually).",
    )
    parser.add_argument(
        "--task-space", type=str, default=DEFAULT_TASK_SPACE,
        help="ego-browser task space to attach to (your play window).",
    )
    parser.add_argument(
        "--profile-id", type=str, default=None,
        help="ego-browser profile pinning the login identity.",
    )
    parser.add_argument(
        "--poll", type=float, default=0.4, help="Base poll interval (s).",
    )
    parser.add_argument(
        "--fast-poll", type=float, default=0.25,
        help="Poll interval while an opponent is acting (s).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for every sampling path (deck reconstruction).",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="How many advice rows to show.",
    )
    parser.add_argument(
        "--depth", type=int, default=0,
        help="minimax deep mode depth on my turns (0 = off, 2-3 sensible).",
    )
    parser.add_argument(
        "--fixture-html", type=Path, default=None,
        help="Offline smoke: serve this fixture file via MockBrowserDriver.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N adopted frames (0 = run until Ctrl+C).",
    )
    options = vars(parser.parse_args())
    return options


def _build_driver(options: dict[str, Any]) -> BrowserDriver:
    """Mock driver for offline fixtures; ego-browser otherwise."""
    fixture = options["fixture_html"]
    if fixture is not None:
        mock = MockBrowserDriver()
        mock.set_html(fixture.read_text(encoding="utf-8"))
        return mock
    return EgoBrowserDriver(
        options["task_space"],
        room_url=options["room_url"],
        profile_id=options["profile_id"],
    )


def main() -> None:
    """Entry point of the ``play-advisor`` console script."""
    options = _parse_args()
    # Reproducibility (AGENTS.md fact 6): no torch in this process, so the
    # seed pair covers every sampling path the advisor owns.
    random.seed(options["seed"])
    np.random.seed(options["seed"])

    driver = _build_driver(options)
    if options["room_url"] is not None and options["fixture_html"] is None:
        driver.navigate(options["room_url"])

    session = AdvisorSession(
        driver,
        poll_interval=options["poll"],
        fast_poll_interval=options["fast_poll"],
    )
    cli = AdvisorCli(
        depth=options["depth"], top_k=options["topk"], seed=options["seed"]
    )
    print(
        "▶ play-advisor 只读辅助已启动（零点击：本进程从不执行页面操作）"
        + (f" | 房间 {options['room_url']}" if options["room_url"] else ""),
        flush=True,
    )

    should_stop = (
        session.stop_after(options["max_frames"]) if options["max_frames"] else None
    )
    schema_warned = False

    def on_schema_error(_error: SnapshotSchemaError) -> bool:
        nonlocal schema_warned
        if not schema_warned:
            print("⚠ 页面快照不符合 schema（过渡中或改版），继续轮询…", flush=True)
            schema_warned = True
        return True

    start = time.monotonic()

    def on_frame(frame: AdvisorFrame) -> None:
        cli.on_frame(frame)

    try:
        session.run(cli.on_frame, should_stop=should_stop, on_schema_error=on_schema_error)
    except SnapshotSchemaError as error:
        print(f"✖ 快照 schema 持续失败，退出: {error}", flush=True)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        pass
    if should_stop is not None:
        print(
            f"⏹ 已达帧数上限（{session.reads} 次读取）| {time.monotonic() - start:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
