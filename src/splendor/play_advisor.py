"""
``play-advisor`` console script: a read-only Splendor advisor (plan
phase-7). Launches (or attaches to) one ego-browser task space, the human
plays *manually* in that window, and this process only polls the DOM and
ranks the human's legal moves.

Zero-click contract: the advisor never imports the action executor and
never mutates page state - the etiquette constraint (plan phase-2 M2.5)
is satisfied by construction, not by pacing.

Per adopted (debounce-stable) frame :class:`AdvisorCli` computes one
JSON-ready *view* (tracker fold-in, GA top-k on my turn, deck histogram,
affordability, parity attribution) and then

* renders it as the v0 console block, and
* publishes it to the localhost dashboard (``AdvisorStore``), whose deep
  button runs minimax in a worker thread against the latest my-turn
  reconstruction.

Offline smoke: ``play-advisor --fixture-html <file> --max-frames 2``.
"""

import argparse
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from splendor.browser.advisor.engine import (
    TWO_PLAYER_SEATS,
    AdvisorEngine,
    DeckHistogram,
    DeckRow,
    _missing_text,
)
from splendor.browser.advisor.observer import (
    PHASE_LABELS,
    AdvisorFrame,
    AdvisorSession,
    Phase,
)
from splendor.browser.advisor.server import AdvisorStore, start_server
from splendor.browser.advisor.tracker import (
    ReservationTracker,
    ReservedEvent,
)
from splendor.browser.dom_extractor import (
    CardInfo,
    PanelInfo,
    Snapshot,
    SnapshotSchemaError,
    looks_like_game_over,
)
from splendor.browser.driver import BrowserDriver, MockBrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.splendor.action_text import COLOR_CN
from splendor.splendor.constants import NUMBER_OF_TIERS

DEFAULT_TASK_SPACE = "splendor-advisor"
DEFAULT_PORT = 8900  # play-dashboard uses 8899
_MIN_GAME_PANELS = TWO_PLAYER_SEATS  # below this the page is not in-game

_ADVICE_ICONS = ("★", "☆", " ")
_MIN_BAR_SAMPLES = 2

_EVENT_TEXT: dict[str, str] = {
    "init": "接入(未知)",
    "reset": "新局重置",
    "reserve_table": "预留桌面牌(已识别)",
    "reserve_deck": "预留牌堆(已识别)",
    "reserve_unknown": "预留牌堆(未知)",
    "purchase": "购买预留牌",
}


def _format_scores(snapshot: Snapshot, my_seat: int) -> str:
    """``我=5 | 座2=3`` from the live panel snapshot."""
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


def _event_text(event: ReservedEvent) -> str:
    seat = f"座{event.seat}" if event.seat else "-"
    card = f" {event.card.code}" if event.card is not None else ""
    return f"{seat} {_EVENT_TEXT.get(event.kind, event.kind)}{card}"


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


def _card_view(info: CardInfo) -> dict[str, Any]:
    """DOM CardInfo -> display dict (colour names kept in English keys)."""
    return {
        "colour": info["colour"],
        "points": info["points"],
        "cost": info["cost"],
        "text": (
            f"{info['tier'] + 1}级{COLOR_CN.get(info['colour'], info['colour'])}卡"
            + (f"{info['points']}分" if info["points"] else "")
        ),
    }


def _advice_rows(advice: list) -> list[dict[str, Any]]:
    """Advice dataclasses -> JSON-ready rows (actions stay engine-side)."""
    return [
        {
            "text": item.text,
            "value": item.value,
            "reasons": list(item.reasons),
            "source": item.source,
        }
        for item in advice
    ]


def _dealt_count(snapshot: Snapshot) -> int:
    return sum(1 for row in snapshot["dealt"] for info in row if info is not None)


def _dealt_key(snapshot: Snapshot, visible_index: int) -> str:
    """``tier:col`` of the visible_index-th non-None dealt card."""
    seen = 0
    for tier in range(NUMBER_OF_TIERS):
        for col, info in enumerate(snapshot["dealt"][tier]):
            if info is not None:
                if seen == visible_index:
                    return f"{tier}:{col}"
                seen += 1
    return f"{visible_index}:-"


def _reserved_summary(panel: PanelInfo) -> str:
    tiers = panel["reserved_tiers"]
    if not tiers:
        return "—"
    return ",".join(f"T{tier + 1}" for tier in tiers) + "（背）"


class AdvisorCli:
    """
    Frame consumer: compute one JSON-ready view per in-game frame, render
    it as console text and (optionally) publish it to the dashboard store.

    ``emit`` is injectable so tests can collect lines instead of watching
    stdout; ``store`` is None in pure-CLI mode.
    """

    def __init__(
        self,
        *,
        depth: int = 0,
        top_k: int = 5,
        seed: int = 0,
        store: AdvisorStore | None = None,
        emit: Callable[[str], None] = print,
    ) -> None:
        self._depth = depth
        self._top_k = top_k
        self._seed = seed
        self._store = store
        self._emit = emit
        self._tracker = ReservationTracker()
        self._engine: AdvisorEngine | None = None
        self._engine_key: tuple[int, int] | None = None
        self._last_key: tuple[str, str] | None = None
        self._prev_phase: Phase | None = None
        self._my_turns = 0
        self._standby_announced = False

    @property
    def engine(self) -> AdvisorEngine | None:
        """The live engine (the harness shares it with the deep worker)."""
        return self._engine

    # ----- frame handling ------------------------------------------------------
    def on_frame(self, frame: AdvisorFrame) -> None:
        if self._store is not None and self._store.pop_reset_request():
            self._tracker.reset("dashboard manual reset")
        snapshot = frame.snapshot
        if frame.phase is Phase.NO_BOARD:
            self._announce_standby(frame)
            return
        self._standby_announced = False
        if len(snapshot["panels"]) < _MIN_GAME_PANELS:
            return  # not an in-game view (single-panel close-up / lobby)
        view = self._compute(frame)
        if view is None:
            return
        if self._store is not None:
            self._store.set_state(view)
        self._render(view)

    # ----- computation (shared by console + dashboard) ---------------------------
    def _compute(self, frame: AdvisorFrame) -> dict[str, Any] | None:
        snapshot = frame.snapshot
        self._tracker.update(snapshot, frame.frame_seq)
        my_index = frame.my_index
        if my_index is None:
            return None  # unseated: no advice, no engine
        self._ensure_engine(len(snapshot["panels"]), my_index)
        assert self._engine is not None

        started = time.perf_counter()
        ga_rows: list[dict[str, Any]] = []
        deep_rows: list[dict[str, Any]] | None = None
        if frame.phase is Phase.MY_TURN:
            if self._prev_phase is not Phase.MY_TURN:
                self._my_turns += 1
            state = self._engine.build_reconstruction(
                snapshot, self._tracker, turns=self._my_turns
            )
            ga_rows = _advice_rows(self._engine.ga_top_k(state))
            if self._depth > 0:
                deep_rows = _advice_rows(
                    self._engine.minimax_top_k(state, depth=self._depth)
                )
            if self._store is not None:
                self._store.set_deep_source(state, self._depth)
        self._prev_phase = frame.phase
        elapsed_ms = (time.perf_counter() - started) * 1000

        histogram = self._engine.deck_histogram(snapshot, self._tracker)
        afford_rows = self._engine.affordability(snapshot)
        afford_map: dict[str, dict[str, Any]] = {}
        dealt_total = _dealt_count(snapshot)
        keys = [_dealt_key(snapshot, i) for i in range(dealt_total)] + [
            f"r:{j}" for j in range(len(snapshot["my_reserved"]))
        ]
        for key, row in zip(keys, afford_rows, strict=True):
            afford_map[key] = {
                "affordable": row.affordable,
                "missing_text": _missing_text(row.missing, row.gold_covers),
                "text": row.text,
                "source": row.source,
            }
        closest = next((row for row in reversed(afford_rows) if row.missing), None)
        return {
            "frame_seq": frame.frame_seq,
            "status": snapshot["status"],
            "phase": frame.phase.value,
            "phase_label": PHASE_LABELS[frame.phase],
            "my_seat": snapshot["my_seat"],
            "players": [
                {
                    "seat": panel["seat"],
                    "score": panel["score"],
                    "card_counts": panel["card_counts"],
                    "gems": panel["gems"],
                    "reserved_summary": _reserved_summary(panel),
                }
                for panel in snapshot["panels"]
            ],
            "dealt": [
                [_card_view(info) if info is not None else None for info in row]
                for row in snapshot["dealt"]
            ],
            "my_reserved": [_card_view(info) for info in snapshot["my_reserved"]],
            "supply": snapshot["supply"],
            "nobles": self._engine.noble_progress(snapshot),
            "advice": {"ga": ga_rows, "deep_sync": deep_rows},
            "deck_hist": {
                "rows": [
                    {
                        "tier": row.tier,
                        "counts": row.counts,
                        "grey": row.grey,
                        "deck_count": row.deck_count,
                    }
                    for row in histogram.rows
                ],
                "warnings": list(histogram.warnings),
            },
            "afford": {
                "affordable_count": sum(1 for r in afford_rows if r.affordable),
                "total": len(afford_rows),
                "map": afford_map,
                "closest": (
                    {
                        "text": closest.text,
                        "source": closest.source,
                        "missing_text": _missing_text(
                            closest.missing, closest.gold_covers
                        ),
                    }
                    if closest is not None
                    else None
                ),
            },
            "tracker": {
                "known": len(self._tracker.known_reserved_faces()),
                "unknown": self._tracker.unknown_reserved_count(),
                "recent_events": [
                    _event_text(event) for event in self._tracker.events_log()[-3:]
                ],
            },
            "parity": self._engine.parity_report(snapshot),
            "elapsed_ms": elapsed_ms,
        }

    def _ensure_engine(self, panel_count: int, my_index: int) -> None:
        key = (panel_count, my_index)
        if self._engine is None or self._engine_key != key:
            self._engine = AdvisorEngine(
                panel_count, my_index, seed=self._seed, top_k=self._top_k
            )
            self._engine_key = key

    # ----- console rendering -----------------------------------------------------
    def _announce_standby(self, frame: AdvisorFrame) -> None:
        if self._standby_announced:
            return
        self._standby_announced = True
        status = frame.snapshot["status"]
        suffix = "（终局）" if looks_like_game_over(status, board_present=False) else ""
        self._emit(f"⏸ 未检测到对局棋盘{suffix}——待机轮询中")

    def _render(self, view: dict[str, Any]) -> None:
        key = (self._state_key(view), view["phase"])
        if key == self._last_key:
            return
        self._last_key = key

        self._emit(
            f"▶ {view['status']} | 比分: {_scores_view(view)} | {view['phase_label']}"
        )
        if view["advice"]["ga"]:
            self._emit_advice("建议（GA 快评，★最优 ☆次优）:", view["advice"]["ga"])
        if view["advice"]["deep_sync"]:
            self._emit_advice(
                f"深算（minimax depth={self._depth}）:", view["advice"]["deep_sync"]
            )
        if view["phase"] in (Phase.MY_DISCARD.value, Phase.MY_NOBLE.value, Phase.PAYMENT.value):
            self._emit("   （子流程需要手动点击，建议暂停）")
        self._emit("  牌堆剩余（未翻开部分，按颜色）:")
        for line in _histogram_lines(
            DeckHistogram(
                rows=tuple(_row_from_view(row) for row in view["deck_hist"]["rows"]),
                warnings=tuple(view["deck_hist"]["warnings"]),
            )
        ):
            self._emit(line)
        self._emit_afford(view)
        tracker = view["tracker"]
        self._emit(
            f"  预留记忆: 已识别 {tracker['known']} / 未知 {tracker['unknown']}"
            + (
                f" | 最近: {'; '.join(tracker['recent_events'])}"
                if tracker["recent_events"]
                else ""
            )
        )

    def _state_key(self, view: dict[str, Any]) -> str:
        dealt = tuple(
            (card["colour"], card["points"], tuple(sorted(card["cost"].items())))
            if card is not None
            else ()
            for row in view["dealt"]
            for card in row
        )
        backs = tuple(p["reserved_summary"] for p in view["players"])
        scores = tuple(p["score"] for p in view["players"])
        supply = tuple(view["supply"][c] for c in sorted(view["supply"]))
        return f"{scores}|{supply}|{dealt}|{backs}"

    def _emit_advice(self, title: str, rows: list[dict[str, Any]]) -> None:
        self._emit(f"  {title}")
        values = [row["value"] for row in rows]
        for rank, row in enumerate(rows, start=1):
            icon = _ADVICE_ICONS[rank - 1] if rank <= len(_ADVICE_ICONS) else " "
            self._emit(
                f"   {rank}. {icon} {row['text']}  {_bar(values, row['value'])} "
                f"{row['value']:+.2f}"
            )
            for reason in row["reasons"]:
                self._emit(f"      · {reason}")

    def _emit_afford(self, view: dict[str, Any]) -> None:
        afford = view["afford"]
        affordable_texts = [
            entry["text"] for entry in afford["map"].values() if entry["affordable"]
        ]
        self._emit(
            f"  可负担: {afford['affordable_count']}/{afford['total']} 张"
            + ("（" + "；".join(affordable_texts[:3]) + "）" if affordable_texts else "")
        )
        closest = afford["closest"]
        if closest is not None:
            self._emit(
                f"   · {closest['text']}"
                f"（{'我的预留' if closest['source'] == 'reserved' else '桌面'}）"
                f" {closest['missing_text']}"
            )


def _row_from_view(row: dict[str, Any]) -> DeckRow:
    return DeckRow(
        tier=row["tier"],
        counts=dict(row["counts"]),
        grey=row["grey"],
        deck_count=row["deck_count"],
    )


def _scores_view(view: dict[str, Any]) -> str:
    parts = []
    for player in view["players"]:
        label = "我" if player["seat"] == view["my_seat"] else f"座{player['seat']}"
        parts.append(f"{label}={player['score']}")
    return " | ".join(parts)


# ----- deep worker (dashboard button) ---------------------------------------------
def deep_worker(store: AdvisorStore, engine_getter: Callable[[], AdvisorEngine | None]) -> None:
    """
    Wait for deep requests and run minimax on the latest reconstruction.

    The engine lock (inside ``minimax_top_k``) serialises against the poll
    thread's GA scoring; failures degrade to a payload error, never a crash.
    """
    while True:
        job = store.wait_deep_request(timeout=2.0)
        if job is None:
            continue
        current = engine_getter()
        if current is None or job.state is None:
            store.set_deep_error("暂无可深算的局面（等待我的回合）")
            continue
        started = time.perf_counter()
        try:
            rows = _advice_rows(current.minimax_top_k(job.state, depth=job.depth))
            store.set_deep_result(
                rows, elapsed_ms=(time.perf_counter() - started) * 1000
            )
        except Exception as error:  # deep must never kill the advisor
            store.set_deep_error(str(error))


# ----- entry point ------------------------------------------------------------------
def _parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="play-advisor",
        description=(
            "Read-only Splendor advisor: watch the seat you play manually "
            "and rank your legal moves (plan phase-7)."
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
    parser.add_argument("--poll", type=float, default=0.4, help="Base poll interval (s).")
    parser.add_argument(
        "--fast-poll", type=float, default=0.25,
        help="Poll interval while an opponent is acting (s).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for every sampling path (deck reconstruction).",
    )
    parser.add_argument("--topk", type=int, default=5, help="Advice rows to show.")
    parser.add_argument(
        "--depth", type=int, default=0,
        help="Synchronous minimax depth on my turns (0 = off; the dashboard's "
        "deep button works regardless).",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help="Dashboard port (0 = disable dashboard).",
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

    store: AdvisorStore | None = None
    engine_holder: dict[str, AdvisorEngine | None] = {"engine": None}
    if options["port"]:
        store = AdvisorStore()
        server = start_server(store, options["port"])
        threading.Thread(
            target=server.serve_forever, daemon=True, name="advisor-dashboard"
        ).start()
        threading.Thread(
            target=deep_worker,
            args=(store, lambda: engine_holder["engine"]),
            daemon=True,
            name="advisor-deep",
        ).start()
        print(f"📊 仪表盘: http://localhost:{options['port']}", flush=True)

    cli = AdvisorCli(
        depth=options["depth"],
        top_k=options["topk"],
        seed=options["seed"],
        store=store,
    )

    def on_frame(frame: AdvisorFrame) -> None:
        cli.on_frame(frame)
        if cli.engine is not None:
            engine_holder["engine"] = cli.engine

    session = AdvisorSession(
        driver,
        poll_interval=options["poll"],
        fast_poll_interval=options["fast_poll"],
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
    try:
        session.run(on_frame, should_stop=should_stop, on_schema_error=on_schema_error)
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
