"""
Session management for game.hullqin.cn/ccbs rooms (BROWSER_RL_MAPPING §2).

The session recipes encode two measured pitfalls of the double-identity
self-play setup:

* **double-domain cookie deletion**: the identity cookie ``gid`` may live on
  ``game.hullqin.cn`` *or* ``.game.hullqin.cn``; deleting only one variant
  resurrects the old identity on the next page load;
* **swap-after-load ordering**: a page's websocket identity is fixed at the
  handshake, so a page must finish loading *before* cookies are rotated, and
  the rotation is followed by a reload that picks up the fresh ``gid``.

This class drives only "my" seat; the opposing seat belongs to another
process / driver instance running its own SessionManager + env.
"""

import time

from .driver import BrowserDriver

# Measured page facts (BROWSER_RL_MAPPING §2 + T0.4 experiments 2026-09-05).
# The lobby's 创建房间 is an <a> link; 加入 / 开始游戏 / N人 / 重连 are plain
# buttons with no stable ccbs-* class - text is their identity, hence the
# click_labelled primitives.
BASE_URL = "https://game.hullqin.cn/ccbs"
GID_COOKIE = "gid"
IDENTITY_DOMAINS: tuple[str, ...] = ("game.hullqin.cn", ".game.hullqin.cn")

DEFAULT_SEATS = 2  # the room's default; the toggle only exists for owners

LABEL_CREATE_ROOM = "创建房间"  # lobby link (exact text)
LABEL_JOIN_SEAT = "加入"  # one button per free seat, document order = seat order
LABEL_START_GAME = "开始游戏"
LABEL_RECONNECT = "重连"  # shown when the server flags a cookie problem

# Waiting game start needs the same patience class as any UI migration.
_START_TIMEOUT = 30.0

# The SPA pushes the room URL asynchronously after 创建房间 (measured
# 2026-09-11: an immediate location.href read returned the lobby URL,
# poisoning the published room file). Poll this long before giving up.
_CREATE_ROOM_TIMEOUT = 10.0


class SessionManager:
    """Room lifecycle: create, seat, start, identity swap, recover."""

    def __init__(
        self,
        driver: BrowserDriver,
        base_url: str = BASE_URL,
        room_url: str | None = None,
        seats: int = DEFAULT_SEATS,
    ) -> None:
        """
        :param room_url: pin the session to an existing room (e.g. one the
            user created to watch or play against the agent); new_game()
            then re-joins this room instead of creating fresh ones.
        :param seats: room size used when creating rooms.
        """
        self._driver = driver
        self._base_url = base_url
        self._room_url: str | None = room_url
        self._seats = seats

    @property
    def room_url(self) -> str | None:
        return self._room_url

    def create_room(self, seats: int = 2) -> str:
        """
        Create a room with ``seats`` seats and return its URL.

        The creator becomes the room owner and still must join a seat before
        the game can start. Only the owner sees the 2人/3人/4人 toggle, and
        2 is the default - the click is skipped for it.
        """
        if seats not in {2, 3, 4}:
            raise ValueError(f"unsupported seat count {seats}")
        self._driver.navigate(self._base_url)
        self._driver.click_labelled(LABEL_CREATE_ROOM, exact=False)
        if seats != DEFAULT_SEATS:
            self._driver.click_labelled(f"{seats}人")
        base = self._base_url.rstrip("/")
        deadline = time.monotonic() + _CREATE_ROOM_TIMEOUT
        url = str(self._driver.evaluate("location.href"))
        while time.monotonic() < deadline and url.rstrip("/") == base:
            time.sleep(0.5)
            url = str(self._driver.evaluate("location.href"))
        if url.rstrip("/") == base:
            # Loud failure beats a poisoned room file: the second bot would
            # otherwise "join" the lobby and wait for a game that never
            # starts. Common cause: the identity is still seated in an old
            # room server-side, so the create is silently refused.
            raise TimeoutError(
                "room URL did not appear after clicking 创建房间 "
                f"(still on {url!r}); the create may have been refused - "
                "check for a stale identity in an old room and re-run"
            )
        self._room_url = url
        return url

    def join_seat(self, seat: int) -> None:
        """
        Take ``seat`` (1-based page numbering) in the current room.

        The page renders one 加入 button per *free* seat in seat order, so
        ``index = seat - 1`` is correct while every seat before ``seat`` is
        already taken (the normal self-play flow: owner sits first, the
        second identity joins the next free seat). For "just take whatever
        is open" prefer :meth:`join_first_free_seat`.
        """
        if seat < 1:
            raise ValueError(f"seat numbers are 1-based, got {seat}")
        self._driver.click_labelled(LABEL_JOIN_SEAT, index=seat - 1)

    def join_first_free_seat(self) -> None:
        """
        Take the lowest-numbered free seat, whatever it is.

        This is the robust entry for scripted play: on a fresh room it takes
        seat 1 (ownership); in a room where a human or another process
        already sits lower, it takes the next free seat automatically.
        """
        self._driver.click_labelled(LABEL_JOIN_SEAT, index=0)

    def start_game(self) -> None:
        """Start the game; requires at least two occupied seats (owner only)."""
        self._driver.click_labelled(LABEL_START_GAME)

    def new_game(self) -> None:
        """
        Begin a fresh game for reset().

        Pinned-room mode (``room_url`` given): re-join the room, take the
        first free seat, and try to start - tolerated to fail because the
        room owner (e.g. a human watching) presses 开始游戏 themselves.

        Measured end-of-game reality (E3, T0.4): after a game ends the page
        returns to the room view where both seats are still occupied and the
        owner's 开始游戏 button is back - starting again is the common path.
        Falling back to a fresh room covers a lost room (kick, expiry).
        """
        if self._room_url is not None:
            self._driver.navigate(self._room_url)
            try:
                self.join_first_free_seat()
            except ValueError:
                pass  # already seated (the common case between games)
            try:
                self.start_game()
            except ValueError:
                pass  # not the owner - the human/owner seat starts the game
            return
        try:
            self.start_game()
            return
        except ValueError:
            pass
        self.create_room(seats=self._seats)
        self.join_first_free_seat()

    def switch_identity(self) -> None:
        """
        Rotate the ``gid`` identity cookie (double-identity self-play recipe).

        Ordering is load-bearing (BROWSER_RL_MAPPING §2 + T0.4 re-verification):
        the current page is loaded *first* because its websocket identity was
        fixed at handshake; cookies are then deleted on **both** domain
        variants (the measured double-domain pitfall); the final reload makes
        the server assign a fresh gid, which the *next* page load will use.
        Only the identity that navigates *after* the rotation picks it up -
        every other tab must already sit on its final page before rotating.
        """
        self._driver.navigate(self._base_url)  # load page with old identity
        for domain in IDENTITY_DOMAINS:
            self._driver.delete_cookies(GID_COOKIE, domain)
        self._driver.navigate(self._base_url)  # reload -> server issues new gid

    def recover(self) -> None:
        """
        Rebuild a playable session after a disconnect or broken room.

        Recovery is a first-class operation, not exception handling garnish:
        unattended web deployment (phase 3, 50 games) lives and dies by it.
        Strategy: navigate back to the room (reconnection is automatic on
        load); when the server answers with the measured "cookies disabled"
        interstitial, click its 重连 button; without a known room, fall back
        to a fresh room.
        """
        self._driver.navigate(self._room_url or self._base_url)
        try:
            self._driver.click_labelled(LABEL_RECONNECT)
        except ValueError:
            pass  # no interstitial - the load itself reconnected
        self._driver.wait_for("true", _START_TIMEOUT)
