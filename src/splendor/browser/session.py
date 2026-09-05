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

from .driver import BrowserDriver

# Measured page facts (BROWSER_RL_MAPPING §2). The lobby URL is measured;
# the in-lobby button selectors are structural assumptions pending T0.4.
BASE_URL = "https://game.hullqin.cn/ccbs"
GID_COOKIE = "gid"
IDENTITY_DOMAINS: tuple[str, ...] = ("game.hullqin.cn", ".game.hullqin.cn")

SELECTOR_CREATE_ROOM = "button.ccbs-create-room"  # [ASSUMED] 创建房间
SELECTOR_SEATS_TEMPLATE = "button.ccbs-seats-{seats}"  # [ASSUMED] 修改人数
SELECTOR_JOIN_SEAT_TEMPLATE = ".ccbs-seat-{seat} .ccbs-join"  # [ASSUMED] 加入
SELECTOR_START_GAME = "button.ccbs-start-game"  # [ASSUMED] 开始游戏
SELECTOR_REMATCH = "button.ccbs-rematch"  # [ASSUMED] 再来一局

# Waiting game start needs the same patience class as any UI migration.
_START_TIMEOUT = 30.0


class SessionManager:
    """Room lifecycle: create, seat, start, rematch, identity swap, recover."""

    def __init__(self, driver: BrowserDriver, base_url: str = BASE_URL) -> None:
        self._driver = driver
        self._base_url = base_url
        self._room_url: str | None = None

    @property
    def room_url(self) -> str | None:
        return self._room_url

    def create_room(self, seats: int = 2) -> str:
        """
        Create a room with ``seats`` seats and return its URL.

        The creator becomes the room owner and still must join a seat before
        the game can start.
        """
        self._driver.navigate(self._base_url)
        self._driver.click(SELECTOR_CREATE_ROOM)
        if seats not in (2, 3, 4):
            raise ValueError(f"unsupported seat count {seats}")
        self._driver.click(SELECTOR_SEATS_TEMPLATE.format(seats=seats))
        self._room_url = str(self._driver.evaluate("location.href"))
        return self._room_url

    def join_seat(self, seat: int) -> None:
        """Take ``seat`` (1-based page numbering) in the current room."""
        if seat < 1:
            raise ValueError(f"seat numbers are 1-based, got {seat}")
        self._driver.click(SELECTOR_JOIN_SEAT_TEMPLATE.format(seat=seat))

    def start_game(self) -> None:
        """Start the game; requires every seat but one to be occupied."""
        self._driver.click(SELECTOR_START_GAME)

    def new_game(self) -> None:
        """
        Begin a fresh game for reset(): prefer the rematch button of a ended
        game, otherwise create a new room and re-join my seat.
        """
        try:
            self._driver.click(SELECTOR_REMATCH)
            return
        except ValueError:
            pass
        self.create_room()
        self.join_seat(1)

    def switch_identity(self) -> None:
        """
        Rotate the ``gid`` identity cookie (double-identity self-play recipe).

        Ordering is load-bearing (BROWSER_RL_MAPPING §2): the current page is
        loaded *first* because its websocket identity was fixed at handshake;
        cookies are then deleted on **both** domain variants (the measured
        double-domain pitfall); the final reload makes the server assign a
        fresh gid, which the *next* page load will use.
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
        load); without a known room, fall back to a fresh room.
        """
        self._driver.navigate(self._room_url or self._base_url)
        self._driver.wait_for("true", _START_TIMEOUT)
