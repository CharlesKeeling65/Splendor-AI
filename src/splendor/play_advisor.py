"""
``play-advisor`` console script: a read-only Splendor advisor (plan
phase-7). Launches (or attaches to) one ego-browser task space, the human
plays *manually* in that window, and this process only polls the DOM and
ranks the human's legal moves.

Zero-click contract: the advisor never imports the action executor and
never mutates page state - the etiquette constraint (plan phase-2 M2.5)
is satisfied by construction, not by pacing.

v0 (T7.1-T7.5) prints a text panel per adopted frame; the localhost
dashboard (T7.6) replaces the printout. Run against an offline fixture
for a no-network smoke: ``play-advisor --fixture-html <file> --max-frames 2``.
"""

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from splendor.browser.advisor.observer import (
    PHASE_LABELS,
    AdvisorFrame,
    AdvisorSession,
    Phase,
)
from splendor.browser.dom_extractor import SnapshotSchemaError, looks_like_game_over
from splendor.browser.driver import BrowserDriver, MockBrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver

DEFAULT_TASK_SPACE = "splendor-advisor"


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

    def on_frame(frame: AdvisorFrame) -> None:
        print(
            f"▶ [帧 {frame.frame_seq}] {frame.snapshot['status']} "
            f"| 阶段: {PHASE_LABELS[frame.phase]}",
            flush=True,
        )
        if frame.phase is Phase.NO_BOARD:
            status = frame.snapshot["status"]
            if looks_like_game_over(status, board_present=False):
                print("🏁 终局待机——同一房间开新局后将自动继续", flush=True)

    start = time.monotonic()
    try:
        session.run(on_frame, should_stop=should_stop, on_schema_error=on_schema_error)
    except SnapshotSchemaError as error:
        print(f"✖ 快照 schema 持续失败，退出: {error}", flush=True)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        pass
    if should_stop is not None:
        print(f"⏹ 已达帧数上限（{session.reads} 次读取）| {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
