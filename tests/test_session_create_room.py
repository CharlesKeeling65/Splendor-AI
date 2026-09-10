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
