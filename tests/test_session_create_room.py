"""Offline tests for SessionManager.create_room's URL-settle polling."""

import pytest

from splendor.browser.driver import MockBrowserDriver
from splendor.browser.session import BASE_URL, SessionManager

LOBBY_HTML = "<a href='#'>创建房间</a>"
ROOM_URL = f"{BASE_URL}/f437"


def _lobby_driver() -> MockBrowserDriver:
    driver = MockBrowserDriver()
    driver.register_page(BASE_URL, LOBBY_HTML)
    return driver


def test_create_room_polls_until_room_url_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("splendor.browser.session._CREATE_ROOM_TIMEOUT", 1.0)
    driver = _lobby_driver()
    calls: dict[str, int] = {"n": 0}

    def url_override(_js: str) -> str:
        calls["n"] += 1
        # The SPA pushes the room URL asynchronously: first reads still
        # show the lobby.
        return BASE_URL if calls["n"] <= 2 else ROOM_URL

    driver.register_evaluate_override("location.href", url_override)
    session = SessionManager(driver)
    assert session.create_room() == ROOM_URL
    assert session.room_url == ROOM_URL
    assert calls["n"] >= 2


def test_create_room_raises_instead_of_returning_lobby_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("splendor.browser.session._CREATE_ROOM_TIMEOUT", 1.0)
    driver = _lobby_driver()
    driver.register_evaluate_override("location.href", BASE_URL)
    session = SessionManager(driver)
    with pytest.raises(TimeoutError, match="创建房间"):
        session.create_room()
    # The poisoned value must not be published as the room URL either.
    assert session.room_url is None


@pytest.mark.parametrize(
    "settled",
    [
        BASE_URL,  # the lobby itself - the measured live failure
        f"{BASE_URL}/",  # lobby with a trailing slash
        f"{BASE_URL}#",  # a hash-only "navigation"
        "https://game.hullqin.cn/",  # the site root, outside the game prefix
    ],
)
def test_create_room_rejects_urls_that_are_not_room_paths(
    monkeypatch: pytest.MonkeyPatch, settled: str
) -> None:
    """
    Landing "somewhere else" is not enough: the base URL with a trailing slash
    would have been accepted by the old "did the URL change" check and
    published as a room the second bot could never join.

    (Any non-empty segment after the prefix counts as a room id - the SPA's own
    route ``/ccbs/:roomId`` accepts it - so the guard is "left the lobby for a
    path under it", not "the segment looks like a known room code".)
    """
    monkeypatch.setattr("splendor.browser.session._CREATE_ROOM_TIMEOUT", 1.0)
    driver = _lobby_driver()
    driver.register_evaluate_override("location.href", settled)
    session = SessionManager(driver)
    with pytest.raises(TimeoutError, match="room URL did not appear"):
        session.create_room()


def test_pin_room_makes_later_games_reuse_that_room() -> None:
    """Self-play creates one room per run; peers resolve the file only once."""
    driver = _lobby_driver()
    driver.register_evaluate_override("location.href", ROOM_URL)
    session = SessionManager(driver)

    session.pin_room(ROOM_URL)
    assert session.room_url == ROOM_URL
    # new_game() now re-joins the pinned room instead of creating a new one.
    driver.click_log.clear()
    session.new_game()
    assert not [entry for entry in driver.click_log if entry[0] == "label:创建房间"]
