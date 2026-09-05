"""
Offline browser-adapter tests (plan phase-2 A2.2): HTML fixtures ->
MockBrowserDriver -> extract_snapshot -> build_pseudo_state -> obs pipeline,
plus executor click sequences, payment strategy and session recipes - all
without touching the network.
"""

from pathlib import Path

import numpy as np
import pytest

import splendor.splendor.gym  # noqa: F401  # registers gym envs (idempotent)
from splendor.browser.action_executor import (
    SELECTOR_CONFIRM_PASS,
    SELECTOR_CONFIRM_TAKE,
    SELECTOR_MODE_PASS,
    SELECTOR_MODE_TAKE_GEMS,
    ActionExecutionError,
    ActionExecutor,
    parse_pill,
)
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import (
    SnapshotSchemaError,
    extract_snapshot,
    is_my_turn,
    looks_like_game_over,
    snapshot_from_raw,
)
from splendor.browser.driver import MockBrowserDriver
from splendor.browser.session import SessionManager
from splendor.browser.state_builder import build_pseudo_state
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, ActionEnum

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"
ALL_FIXTURES = sorted(FIXTURES.glob("*.html"))
BASE_URL = "https://game.hullqin.cn/ccbs"


def _driver_for(fixture: str) -> MockBrowserDriver:
    driver = MockBrowserDriver()
    driver.set_html((FIXTURES / fixture).read_text(encoding="utf-8"), url=BASE_URL)
    return driver


class _StubSession(SessionManager):
    """
    Session whose new_game() simply (re)loads a fixed page. ``html=None``
    keeps whatever page the driver currently holds (mirrors reconnecting
    into a room that is still rendering).
    """

    def __init__(self, driver: MockBrowserDriver, html: str | None = None) -> None:
        super().__init__(driver)
        self._html = html
        self._mock_driver = driver

    def new_game(self) -> None:
        if self._html is not None:
            self._mock_driver.set_html(self._html, url=f"{BASE_URL}/gt01")


# ---------------------------------------------------------------------------
# A2.2: every fixture through the whole pipeline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_fixture_pipeline(fixture: Path) -> None:
    driver = _driver_for(fixture.name)
    snapshot = extract_snapshot(driver)
    my_index = snapshot["my_seat"] - 1
    pseudo_state = build_pseudo_state(snapshot, my_index, turns=3)
    obs = extract_metrics_with_cards(pseudo_state, my_index)

    assert obs.shape == (265,)
    assert np.all(np.isfinite(obs))
    assert snapshot["supply"]["yellow"] >= 0
    assert 1 <= len(snapshot["panels"]) <= 4


def test_opening_fixture_matches_engine_opening_shape() -> None:
    """2-player opening: decks 36/26/16 (measured, BROWSER_RL_MAPPING §3.1)."""
    snapshot = extract_snapshot(_driver_for("opening.html"))
    assert snapshot["deck_counts"] == [36, 26, 16]
    assert len(snapshot["panels"]) == 2
    assert is_my_turn(snapshot["status"])
    assert snapshot["payment_options"] is None
    # 12 table cards, all four slots filled on a fresh opening
    assert all(all(card is not None for card in row) for row in snapshot["dealt"])


def test_empty_deck_fixture_has_trailing_empty_slots() -> None:
    snapshot = extract_snapshot(_driver_for("empty_deck.html"))
    assert snapshot["deck_counts"] == [0, 0, 0]
    assert any(card is None for row in snapshot["dealt"] for card in row)


def test_game_over_fixture_signals_terminal() -> None:
    snapshot = extract_snapshot(_driver_for("game_over.html"))
    assert not is_my_turn(snapshot["status"])
    assert looks_like_game_over(snapshot["status"])
    assert snapshot["panels"][0]["score"] == 15


def test_noble_fixture_exposes_choice_ui() -> None:
    snapshot = extract_snapshot(_driver_for("noble_available.html"))
    assert snapshot["noble_options"] is not None
    assert snapshot["noble_options"][0]["requirements"] == {"green": 4, "red": 4}


# ---------------------------------------------------------------------------
# Payment pills: parsing, env accessor, and the greedy strategy
# ---------------------------------------------------------------------------
def test_parse_pill_color_characters() -> None:
    assert parse_pill("2白1金") == {"white": 2, "yellow": 1}
    assert parse_pill("3白") == {"white": 3}
    assert parse_pill("1黑") == {"black": 1}
    with pytest.raises(ValueError, match="unparseable"):
        parse_pill("白金")


def test_env_get_payment_options_reads_pending_pills() -> None:
    payment_env = BrowserSplendorEnv(
        _driver_for("payment_pills.html"),
        _StubSession(_driver_for("payment_pills.html")),
        click_delay=(0, 0),
    )
    options = payment_env.get_payment_options(0)
    assert options == [{"white": 3}, {"white": 2, "yellow": 1}]

    plain_driver = _driver_for("opening.html")
    plain_env = BrowserSplendorEnv(
        plain_driver, _StubSession(plain_driver), click_delay=(0, 0)
    )
    assert plain_env.get_payment_options(0) is None


def test_select_payment_greedy_matches_engine_semantics() -> None:
    """
    My panel holds 3 white gems and no white card; the target card costs
    {white: 3}. The engine's greedy payment spends 0 gold -> the executor
    must pick "3白" (index 0), keeping the live state in the training
    distribution (coloured-first, gold-fills-shortfall semantics).
    """
    driver = _driver_for("payment_pills.html")
    snapshot = extract_snapshot(driver)
    pseudo_state = build_pseudo_state(snapshot, 0, turns=2)

    buy_action_index = next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.BUY_AVAILABLE
        and action.position is not None
        and action.position.tier == 0
        and action.position.card_index == 0
        and action.noble_index is None
    )
    action = ALL_ACTIONS[buy_action_index]
    executor = ActionExecutor(driver, click_delay=(0, 0))
    choice = executor.select_payment_greedy(
        ["3白", "2白1金"], action, pseudo_state
    )
    assert choice == 0

    # A panel short of one white gem would flip the greedy choice to gold.
    from splendor.browser.dom_extractor import PanelInfo

    panels = snapshot["panels"]
    panels[0] = PanelInfo(
        seat=1,
        score=panels[0]["score"],
        card_counts=panels[0]["card_counts"],
        gems={**panels[0]["gems"], "white": 2},
        reserved_tiers=panels[0]["reserved_tiers"],
    )
    short_state = build_pseudo_state({**snapshot, "panels": panels}, 0, turns=2)
    assert executor.select_payment_greedy(["3白", "2白1金"], action, short_state) == 1


# ---------------------------------------------------------------------------
# Executor click sequences against real fixture markup
# ---------------------------------------------------------------------------
def test_executor_pass_sequence() -> None:
    driver = _driver_for("opening.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))
    executor.execute(0, snapshot, None)  # ALL_ACTIONS[0] = PASS

    assert driver.click_log == [
        (SELECTOR_MODE_PASS, 0),
        (SELECTOR_CONFIRM_PASS, 0),
    ]


def test_executor_collect_sequence() -> None:
    driver = _driver_for("opening.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))

    collect_index = next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.COLLECT_DIFF
        and action.collected_gems == {"white": 1, "blue": 1, "green": 1}
        and action.noble_index is None
    )
    executor.execute(collect_index, snapshot, None)

    assert driver.click_log[0] == (SELECTOR_MODE_TAKE_GEMS, 0)
    assert driver.click_log[-1] == (SELECTOR_CONFIRM_TAKE, 0)
    clicked_chips = sorted(selector for selector, _ in driver.click_log[1:-1])
    assert clicked_chips == sorted(
        [
            "button.ccbs-circle.ccbs-color-0",  # white
            "button.ccbs-circle.ccbs-color-1",  # blue
            "button.ccbs-circle.ccbs-color-2",  # green
        ]
    )  # click order follows the action's dict order, which is not semantic


def test_executor_buy_sequence_clicks_card_overlay() -> None:
    driver = _driver_for("opening.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))

    # Buy the first card of the bottom row (deck 0, web row 2); the unique
    # payment means no pill click follows.
    buy_index = next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.BUY_AVAILABLE
        and action.position is not None
        and action.position.tier == 0
        and action.position.card_index == 0
        and action.noble_index is None
    )
    executor.execute(buy_index, snapshot, None)
    assert driver.click_log[0] == ("button.ccbs-buy-card", 0)
    assert driver.click_log[1] == (".ccbs-row-2 .ccbs-overlay-buy", 0)


def test_executor_reserve_on_deck_top_offsets_by_deck_wrapper() -> None:
    driver = _driver_for("opening.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))

    # Reserve from the deck stack itself = column index -1 in the overlay
    # list; reserving card slot 0 of tier 0 lands on overlay index 1 because
    # the deck wrapper carries the first reserve overlay in the row.
    reserve_card = next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.RESERVE
        and action.position is not None
        and action.position.tier == 0
        and action.position.card_index == 0
        and action.noble_index is None
        and action.collected_gems == {"yellow": 1}
    )
    executor.execute(reserve_card, snapshot, None)
    assert driver.click_log == [
        ("button.ccbs-reserve-card", 0),
        (".ccbs-row-2 .ccbs-overlay-reserve", 1),
    ]


_NO_BUTTONS_PAGE = (
    "<html><body>"
    '<div class="ccbs-status">等待你操作</div>'
    + "".join(
        f'<div class="flex justify-center origin-top ccbs-row-{r}">'
        '<div class="ccbs-card ccbs-type-5 ccbs-img-0">'
        '<div class="ccbs-left-count">0</div></div></div>'
        for r in range(3)
    )
    + '<div class="ccbs-supply">'
    '<button class="ccbs-circle ccbs-color-0">0</button>'
    "</div>"
    '<div class="ccbs-player ccbs-me"><span class="ccbs-score">0分</span></div>'
    "</body></html>"
)


def test_executor_fails_loudly_on_missing_page_elements() -> None:
    """A page that validates but lost its action buttons must fail the
    executor (twice) instead of clicking into the void."""
    driver = MockBrowserDriver()
    driver.set_html(_NO_BUTTONS_PAGE)
    executor = ActionExecutor(driver, click_delay=(0, 0))
    with pytest.raises(ActionExecutionError, match="failed twice"):
        executor.execute(0, extract_snapshot(driver), None)
    assert driver.click_log == []


# ---------------------------------------------------------------------------
# Environment: protocol conformance (A2.4), reset, step, terminal detection
# ---------------------------------------------------------------------------
def test_env_satisfies_protocol_and_resets() -> None:
    driver = _driver_for("opening.html")
    env = BrowserSplendorEnv(driver, _StubSession(driver), click_delay=(0, 0))
    assert isinstance(env, SplendorEnvBase)
    assert env.observation_space.shape == (265,)
    assert env.action_space.n == len(ALL_ACTIONS)

    obs, info = env.reset(seed=42)
    assert obs.shape == (265,)
    assert obs.dtype == np.float32
    assert info["my_id"] == 1  # my panel is the first seat in the fixture
    mask = env.get_legal_actions_mask()
    assert mask.shape == (len(ALL_ACTIONS),)
    assert mask.sum() > 0


_TERMINAL_PAGE = """
<html><body>
<div class="ccbs-status">游戏结束 本局结算</div>
<div class="ccbs-supply">
<button class="ccbs-circle ccbs-color-0">1</button><button class="ccbs-circle ccbs-color-1">1</button>
<button class="ccbs-circle ccbs-color-2">1</button><button class="ccbs-circle ccbs-color-3">1</button>
<button class="ccbs-circle ccbs-color-4">1</button><button class="ccbs-circle ccbs-color-5">1</button>
</div>
<div class="ccbs-player ccbs-me"><span class="ccbs-score">12分</span></div>
<div class="ccbs-player"><span class="ccbs-score">15分</span></div>
<div class="flex justify-center origin-top ccbs-row-0"><div class="ccbs-card ccbs-type-5 ccbs-img-2"><div class="ccbs-left-count">1</div></div></div>
<div class="flex justify-center origin-top ccbs-row-1"><div class="ccbs-card ccbs-type-5 ccbs-img-1"><div class="ccbs-left-count">1</div></div></div>
<div class="flex justify-center origin-top ccbs-row-2"><div class="ccbs-card ccbs-type-5 ccbs-img-0"><div class="ccbs-left-count">1</div></div></div>
<button class="ccbs-pass">放弃</button><button class="ccbs-confirm-pass">确认放弃</button>
<button class="ccbs-take-gems">取宝石</button><button class="ccbs-confirm-take">确认拿这些</button>
<button class="ccbs-buy-card">购买</button><button class="ccbs-reserve-card">预定</button>
</body></html>
"""


def test_env_step_detects_terminal_via_status_text() -> None:
    """Terminal detection unit (E3 marker text) through the public step()."""
    driver = _driver_for("opening.html")
    driver.set_html(_TERMINAL_PAGE, url=f"{BASE_URL}/gt01")
    env = BrowserSplendorEnv(
        driver, _StubSession(driver, _TERMINAL_PAGE), click_delay=(0, 0)
    )
    env.reset(seed=1)
    _obs, reward, terminated, truncated, _info = env.step(0)  # PASS
    assert terminated is True
    assert truncated is False
    assert reward == 0.0  # passing never scores


class _RotatingDriver(MockBrowserDriver):
    """
    Serves the opponent page on the first snapshot read after :meth:`arm`
    and the scored page on the second - an offline simulation of the turn
    rotation (my action -> opponent acting -> my turn again, +3 points).
    Arming after reset() decouples the rotation from reset's own reads.
    """

    def __init__(self, opponent_page: str, scored_page: str) -> None:
        super().__init__()
        self._opponent_page = opponent_page
        self._scored_page = scored_page
        self._armed = False
        self._flips = 0

    def arm(self) -> None:
        self._armed = True

    def evaluate(self, js: str) -> object:
        from splendor.browser.driver import SNAPSHOT_JS_MARKER

        if self._armed and SNAPSHOT_JS_MARKER in js:
            self._flips += 1
            # armed reads: 1 = step()'s pre-action extract (page still shows
            # the pre-action state), 2 = first wait poll (opponent acting),
            # 3+ = my turn again with the score applied.
            if self._flips <= 2:
                self.set_html(self._opponent_page)
            else:
                self.set_html(self._scored_page)
        return super().evaluate(js)


def test_env_step_reward_is_panel_score_differential() -> None:
    """
    reward = my panel "N分" delta across the step, exactly the local env's
    action_reward semantics: pass (0 points) while a page rotation gives my
    panel +3 points must return reward 3.0, terminated False.

    Page rotation: opening (my turn, 0分) -> opponent acting (same score) ->
    my turn again with 3分 on my panel.
    """
    opening = (FIXTURES / "opening.html").read_text(encoding="utf-8")
    opponent_page = opening.replace("等待你操作", "等待玩家2操作", 1)
    scored_page = opponent_page.replace(
        '<span class="ccbs-score">0分</span>', '<span class="ccbs-score">3分</span>', 1
    ).replace("等待玩家2操作", "等待你操作", 1)

    rotating = _RotatingDriver(opponent_page, scored_page)
    rotating.set_html(opening, url=f"{BASE_URL}/gt01")
    env = BrowserSplendorEnv(
        rotating,
        _StubSession(rotating, opening),
        click_delay=(0, 0),
        poll_interval=0.0,
    )
    env.reset(seed=1)
    rotating.arm()  # the next read is step()'s poll loop
    _obs, reward, terminated, truncated, _info = env.step(0)  # PASS
    assert reward == 3.0
    assert terminated is False
    assert truncated is False


# ---------------------------------------------------------------------------
# Session recipes (BROWSER_RL_MAPPING §2)
# ---------------------------------------------------------------------------
_LOBBY_PAGE = """
<html><body>
<button class="ccbs-create-room">创建房间</button>
<button class="ccbs-seats-2">2人</button>
<button class="ccbs-seats-4">4人</button>
<div class="ccbs-seat-1"><button class="ccbs-join">加入</button></div>
<div class="ccbs-seat-2"><button class="ccbs-join">加入</button></div>
<button class="ccbs-start-game">开始游戏</button>
<button class="ccbs-rematch">再来一局</button>
</body></html>
"""


class _NavigatingDriver(MockBrowserDriver):
    def __init__(self) -> None:
        super().__init__()
        self.nav_log: list[str] = []

    def navigate(self, url: str) -> None:
        self.nav_log.append(url)
        super().navigate(url)


def test_session_room_lifecycle_clicks() -> None:
    driver = _NavigatingDriver()
    driver.register_page(BASE_URL, _LOBBY_PAGE)
    session = SessionManager(driver)

    room_url = session.create_room(seats=2)
    assert room_url == BASE_URL  # the mock cannot navigate client-side
    assert ("button.ccbs-create-room", 0) in driver.click_log
    assert ("button.ccbs-seats-2", 0) in driver.click_log

    session.join_seat(1)
    session.start_game()
    assert (".ccbs-seat-1 .ccbs-join", 0) in driver.click_log
    assert ("button.ccbs-start-game", 0) in driver.click_log

    session.new_game()  # rematch button present in the lobby page
    assert ("button.ccbs-rematch", 0) in driver.click_log


def test_switch_identity_deletes_gid_on_both_domains() -> None:
    """
    The measured double-domain pitfall: a leftover gid on either
    game.hullqin.cn or .game.hullqin.cn resurrects the old identity.
    """
    driver = _NavigatingDriver()
    driver.register_page(BASE_URL, _LOBBY_PAGE)
    driver.set_cookie({"name": "gid", "value": "A", "domain": "game.hullqin.cn"})
    driver.set_cookie({"name": "gid", "value": "A", "domain": ".game.hullqin.cn"})
    driver.set_cookie({"name": "other", "value": "x", "domain": "game.hullqin.cn"})

    session = SessionManager(driver)
    session.switch_identity()

    for domain in ("game.hullqin.cn", ".game.hullqin.cn"):
        remaining = driver.get_cookies(domain)
        assert all(cookie["name"] != "gid" for cookie in remaining)
    assert any(
        cookie["name"] == "other" for cookie in driver.get_cookies("game.hullqin.cn")
    )  # unrelated cookies survive
    # Ordering: load page -> rotate -> reload (ws identity fixed at handshake)
    assert driver.nav_log == [BASE_URL, BASE_URL]


# ---------------------------------------------------------------------------
# Schema validation fail-fast behaviour
# ---------------------------------------------------------------------------
def test_snapshot_schema_error_names_the_field() -> None:
    bad: dict[str, object] = {
        "rows": "not-a-list",
        "nobles": [],
        "supply": [],
        "panels": [],
        "my_panel_index": None,
        "my_reserved": [],
        "status_text": "",
        "body_text": "",
        "payment_pill_texts": None,
        "noble_option_rects": None,
    }
    with pytest.raises(SnapshotSchemaError, match="rows"):
        snapshot_from_raw(bad)


def test_snapshot_schema_error_on_missing_top_level_field() -> None:
    driver = _driver_for("opening.html")
    raw = _raw_from(driver)
    del raw["panels"]
    with pytest.raises(SnapshotSchemaError, match="panels"):
        snapshot_from_raw(raw)


def test_snapshot_schema_error_when_my_panel_marker_missing() -> None:
    html = (
        "<html><body>"
        '<div class="ccbs-status">等待你操作</div>'
        + "".join(
            f'<div class="flex justify-center origin-top ccbs-row-{r}"></div>'
            for r in range(3)
        )
        + '<div class="ccbs-supply"></div>'
        '<div class="ccbs-player"><span class="ccbs-score">1分</span></div>'
        "</body></html>"
    )
    driver = MockBrowserDriver()
    driver.set_html(html)
    with pytest.raises(SnapshotSchemaError, match="my panel"):
        extract_snapshot(driver)


def _raw_from(driver: MockBrowserDriver) -> dict:
    from splendor.browser.dom_extractor import EXTRACT_SNAPSHOT_JS
    from splendor.browser.driver import SNAPSHOT_JS_MARKER

    assert SNAPSHOT_JS_MARKER in EXTRACT_SNAPSHOT_JS
    return driver.evaluate(EXTRACT_SNAPSHOT_JS)
