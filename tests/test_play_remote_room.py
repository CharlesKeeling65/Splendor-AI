"""
Offline tests for play-web-remote's room coordination (self-play plumbing).

The live failure these pin down (web_events/bot0.jsonl, 2026-09-11): bot 0
never left the lobby, so it never published a room URL and bot 1 sat in
``_room_for`` until it gave up. Each test below targets one link of the
create -> publish -> seat -> start chain, plus the multi-game invariant that
a run stays inside the one room it created.
"""

import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.driver import MockBrowserDriver
from splendor.browser.session import BASE_URL, LABEL_RECONNECT, SessionManager
from splendor.play_remote import (
    ROOM_FILE,
    EventWriter,
    _announce_seated,
    _coordinate_room,
    _room_for,
    _start_until_running,
    _wait_for_predecessor_seat,
    _wait_for_seats,
)


@pytest.fixture
def events_dir() -> Iterator[Path]:
    """Scratch events directory next to the tests (sandbox-safe events_dir)."""
    directory = Path(__file__).parent / "_tmp_room_coordination"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)

ROOM_URL = f"{BASE_URL}/gt42"
# Measured live lobby markup: the link carries an emoji prefix, so the session
# layer matches it in contains mode.
LOBBY_PAGE = f"<html><body><a href='{ROOM_URL}'>👥 创建房间</a></body></html>"
ROOM_PAGE = "<html><body><button>加入</button><button>开始游戏</button></body></html>"
# Measured live interstitial: entering a room with a rotated identity yields a
# page whose only control is 重连.
INTERSTITIAL_PAGE = "<html><body><button>重连</button></body></html>"


class _RoomDriver(MockBrowserDriver):
    """
    Mock driver whose settled URL is the room page (create_room polls it).

    ``seat`` is what the seat-map probe (``SessionManager.MY_SEAT_JS``)
    answers: 1 means the 加入 click is believed, 0 means this browser is only
    watching - the live-measured outcome of clicking a stale free seat.
    """

    def __init__(self, *, seat: int = 1) -> None:
        super().__init__()
        self.nav_log: list[str] = []
        self.register_evaluate_override("location.href", ROOM_URL)
        self.register_evaluate_override("userseat", seat)

    def navigate(self, url: str) -> None:
        self.nav_log.append(url)
        super().navigate(url)


class _WatcherDriver(_RoomDriver):
    """Room where the 加入 click never sticks: the page stays 观战中."""

    def __init__(self) -> None:
        super().__init__(seat=0)


class _HealingRoomDriver(_RoomDriver):
    """
    Room that renders only 重连 until that button is clicked.

    Measured live 2026-09-11: entering a room with a rotated identity yields an
    interstitial whose *only* control is 重连; clicking it re-loads the room and
    both 加入 buttons come back. The mock models the button's effect - a reload
    that lands on the healthy room - so the click is what flips the page, not
    the load count (recover() navigates *before* it looks for 重连, so a
    load-count model would have healed the room too early and never exercised
    the click at all).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_page(ROOM_URL, INTERSTITIAL_PAGE)

    def click_labelled(
        self,
        label: str,
        *,
        exact: bool = True,
        index: int = 0,
        container_selector: str | None = None,
        container_index: int = 0,
    ) -> None:
        super().click_labelled(
            label,
            exact=exact,
            index=index,
            container_selector=container_selector,
            container_index=container_index,
        )
        if label == LABEL_RECONNECT:
            self.register_page(ROOM_URL, ROOM_PAGE)
            self.navigate(ROOM_URL)  # 重连 re-loads the room server-side


def _owner_options(events_dir: Path, bots: int = 1) -> dict[str, Any]:
    return {
        "bots": bots,
        "events_dir": str(events_dir),
        "room_url": None,
        # coordination files are believed only when newer than this stamp
        "run_started": time.time() - 1.0,
    }


def _owner_env(events_dir: Path, options: dict[str, Any]) -> BrowserSplendorEnv:
    driver = _RoomDriver()
    driver.register_page(BASE_URL, LOBBY_PAGE)
    driver.register_page(ROOM_URL, ROOM_PAGE)
    session = SessionManager(driver)
    return BrowserSplendorEnv(driver, session, feature_version="v1")


# ----- room reuse across games ------------------------------------------------
def test_owner_creates_the_room_once_for_the_whole_run(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Game 2+ must stay in game 1's room: peers resolve the published file
    exactly once (at driver construction), so a fresh room per game would
    leave the two bots in different rooms.
    """
    options = _owner_options(events_dir)
    env = _owner_env(events_dir, options)
    events = EventWriter(events_dir, 0)
    monkeypatch.setattr(
        "splendor.play_remote._start_until_running", lambda *_args: None
    )

    _coordinate_room(0, 0, env, options, events)
    published = (events_dir / "room_url.txt").read_text(encoding="utf-8")
    _coordinate_room(0, 1, env, options, events)

    assert published == ROOM_URL
    assert (events_dir / "room_url.txt").read_text(encoding="utf-8") == ROOM_URL
    # click_log is a MockBrowserDriver attribute, not part of the BrowserDriver
    # protocol env.driver is typed as.
    clicks = env.driver.click_log  # type: ignore[attr-defined]
    assert clicks.count(("label:创建房间", 0)) == 1


# ----- peer handshake ---------------------------------------------------------
def test_peer_waits_for_its_predecessor_seat(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Seats are chained: bot 1 must not load the room before bot 0 sits, or its
    加入 click addresses seat 1 (the seat bot 0 just took) and displaces it.
    """
    monkeypatch.setattr("splendor.play_remote.JOIN_WAIT_SECONDS", 0.0)
    options = _owner_options(events_dir, bots=2)

    with pytest.raises(TimeoutError, match=r"bot1 waited"):
        _wait_for_predecessor_seat(1, options)


def test_peer_navigates_to_the_room_before_reporting_its_seat(
    events_dir: Path,
) -> None:
    """
    Regression: the peer used to click 加入 while still on ``about:blank``
    (nothing navigated it to the room), then report a seat it never took.
    """
    driver = _RoomDriver()
    driver.register_page(ROOM_URL, ROOM_PAGE)
    session = SessionManager(driver, room_url=ROOM_URL)
    env = BrowserSplendorEnv(driver, session, feature_version="v1")
    events = EventWriter(events_dir, 1)
    options = _owner_options(events_dir, bots=2)
    _announce_seated(options, 0, events)  # the owner sat first (chain)

    _coordinate_room(1, 0, env, options, events)

    assert driver.nav_log == [ROOM_URL]
    assert ("label:加入", 0) in driver.click_log
    assert (events_dir / "seat1.ready").exists()


def test_owner_waits_for_every_peer_seat(events_dir: Path) -> None:
    options = _owner_options(events_dir, bots=3)
    events = EventWriter(events_dir, 0)

    _announce_seated(options, 1, events)
    _announce_seated(options, 2, events)
    _wait_for_seats(options, events)  # returns immediately: no sleeps involved


def test_wait_for_seats_gives_up_when_a_peer_never_sits(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("splendor.play_remote.JOIN_WAIT_SECONDS", 0.0)
    options = _owner_options(events_dir, bots=2)
    events = EventWriter(events_dir, 0)

    with pytest.raises(TimeoutError, match=r"seat1\.ready"):
        _wait_for_seats(options, events)


def _age(path: Path, seconds: float) -> None:
    """Backdate a coordination file so it looks like a previous run's."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_room_file_from_a_previous_run_is_ignored(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Live-measured 2026-09-11: a leftover room URL sent the peer into the
    previous room (it reached game_start there) while the owner was creating
    a new one - so a stale publish must not count as this run's room.
    """
    monkeypatch.setattr("splendor.play_remote.JOIN_WAIT_SECONDS", 0.0)
    options = _owner_options(events_dir, bots=2)
    room_file = events_dir / ROOM_FILE
    room_file.write_text(f"{BASE_URL}/stale", encoding="utf-8")
    _age(room_file, 3600.0)

    with pytest.raises(TimeoutError, match="never published"):
        _room_for(1, options)


def test_seat_marker_from_a_previous_run_is_ignored(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale seat marker must not satisfy the seat chain (or the owner wait)."""
    monkeypatch.setattr("splendor.play_remote.JOIN_WAIT_SECONDS", 0.0)
    options = _owner_options(events_dir, bots=2)
    marker = events_dir / "seat0.ready"
    marker.write_text("stale", encoding="utf-8")
    _age(marker, 3600.0)

    with pytest.raises(TimeoutError, match=r"bot1 waited"):
        _wait_for_predecessor_seat(1, options)


def test_peer_recovers_the_measured_reconnect_interstitial(
    events_dir: Path,
) -> None:
    """
    Live-measured 2026-09-11: a room entered with a rotated identity renders
    only 重连, so the seat click has to heal it before it can work.
    """
    driver = _HealingRoomDriver()
    session = SessionManager(driver, room_url=ROOM_URL)
    env = BrowserSplendorEnv(driver, session, feature_version="v1")
    events = EventWriter(events_dir, 0)
    options = _owner_options(events_dir, bots=2)
    _announce_seated(options, 0, events)  # the owner sat first (chain)

    _coordinate_room(1, 0, env, options, events)

    assert driver.nav_log == [
        ROOM_URL,  # _coordinate_room's navigate, landing on the interstitial
        ROOM_URL,  # recover()'s navigate, which still finds the interstitial
        ROOM_URL,  # the 重连 click's re-load, now a healthy room
    ]
    assert driver.click_log[:2] == [("label:重连", 0), ("label:加入", 0)]
    assert (events_dir / "seat1.ready").exists()


def test_spectator_after_the_click_is_retried_then_reported(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Live-measured 2026-09-11: clicking 加入 on a page rendered before a peer's
    join leaves the clicker in the 观战中 row - the click "succeeds" and no
    seat is held. The seat map read must catch that, never announce it.
    """
    monkeypatch.setattr("splendor.play_remote.SEAT_VERIFY_SECONDS", 0.0)
    driver = _WatcherDriver()
    driver.register_page(ROOM_URL, ROOM_PAGE)
    session = SessionManager(driver, room_url=ROOM_URL)
    env = BrowserSplendorEnv(driver, session, feature_version="v1")
    events = EventWriter(events_dir, 1)
    options = _owner_options(events_dir, bots=2)
    _announce_seated(options, 0, events)

    with pytest.raises(TimeoutError, match=r"观战中"):
        _coordinate_room(1, 0, env, options, events)
    assert not (events_dir / "seat1.ready").exists()
    # the reload of recover() is what the retry rides on
    assert driver.nav_log.count(ROOM_URL) >= 2


def test_owner_gives_up_loudly_when_the_seat_never_opens(events_dir: Path) -> None:
    """A seat that stays unreachable must not be reported as taken."""
    driver = _RoomDriver()
    driver.register_page(ROOM_URL, INTERSTITIAL_PAGE)
    session = SessionManager(driver, room_url=ROOM_URL)
    env = BrowserSplendorEnv(driver, session, feature_version="v1")
    events = EventWriter(events_dir, 0)
    options = _owner_options(events_dir, bots=2)
    _announce_seated(options, 0, events)  # the owner sat first (chain)

    with pytest.raises(ValueError, match="加入"):
        _coordinate_room(1, 0, env, options, events)
    assert not (events_dir / "seat1.ready").exists()


# ----- start-game retry -------------------------------------------------------
def test_start_until_running_retries_until_the_table_is_live(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The room page gives no feedback when 开始游戏 is refused, so the owner
    has to keep pressing until a turn-status sentence appears.
    """
    driver = _RoomDriver()
    driver.register_page(ROOM_URL, ROOM_PAGE)
    session = SessionManager(driver, room_url=ROOM_URL)
    env = BrowserSplendorEnv(driver, session, feature_version="v1")
    # SessionManager only navigates when it creates or re-joins a room, and
    # this test drives _start_until_running directly - without this the driver
    # still holds the blank initial page and 开始游戏 matches nothing.
    driver.navigate(ROOM_URL)
    events = EventWriter(events_dir, 0)
    # The mock page cannot turn into a board, so the liveness probe is what
    # gets stubbed; the click sequence is the real one.
    probes = {"count": 0}

    def running(_snapshot: dict[str, Any]) -> bool:
        probes["count"] += 1
        return probes["count"] >= 3

    monkeypatch.setattr("splendor.play_remote._game_running", running)
    monkeypatch.setattr("splendor.play_remote.START_RETRY_SECONDS", 0.0)

    _start_until_running(env, events)

    assert driver.click_log.count(("label:开始游戏", 0)) == 3


def test_start_until_running_gives_up_loudly(
    events_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _RoomDriver()
    driver.register_page(ROOM_URL, ROOM_PAGE)
    session = SessionManager(driver, room_url=ROOM_URL)
    env = BrowserSplendorEnv(driver, session, feature_version="v1")
    driver.navigate(ROOM_URL)
    events = EventWriter(events_dir, 0)
    monkeypatch.setattr("splendor.play_remote._game_running", lambda _snapshot: False)
    monkeypatch.setattr("splendor.play_remote.START_RETRY_SECONDS", 0.0)
    monkeypatch.setattr("splendor.play_remote.START_WAIT_SECONDS", 0.0)

    with pytest.raises(TimeoutError, match="never started"):
        _start_until_running(env, events)

    # the click did happen - the refusal is the room's, not a missing button
    assert driver.click_log.count(("label:开始游戏", 0)) == 1
