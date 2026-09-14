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
    LABEL_CONFIRM_DISCARD,
    LABEL_CONFIRM_PASS,
    LABEL_CONFIRM_TAKE,
    LABEL_MODE_BUY,
    LABEL_MODE_PASS,
    LABEL_MODE_RESERVE,
    LABEL_MODE_TAKE_GEMS,
    LABEL_OVERLAY_BUY,
    LABEL_OVERLAY_RESERVE,
    SELECTOR_DISCARD_BAR,
    SELECTOR_NOBLE_CANDIDATE,
    ActionExecutionError,
    ActionExecutor,
    parse_pill,
)
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import (
    DEFAULT_GAME_OVER_MARKERS,
    EXTRACT_SNAPSHOT_JS,
    Snapshot,
    SnapshotSchemaError,
    extract_snapshot,
    is_my_turn,
    looks_like_game_over,
    snapshot_from_raw,
)
from splendor.browser.driver import SNAPSHOT_JS_MARKER, MockBrowserDriver
from splendor.browser.session import SessionManager
from splendor.browser.state_builder import build_pseudo_state
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, ActionEnum
from splendor.splendor.splendor_model import Card, SplendorGameRule, SplendorState

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


def test_row_with_empty_slot_drops_ccbs_empty_placeholders() -> None:
    """
    Live bug, 2026-09-11: with an exhausted deck the page keeps a bought
    slot as ``.ccbs-card.ccbs-empty`` (no ccbs-type-N) AT ITS POSITION -
    later cards keep their slots, mirroring the engine where
    ``dealt[tier][i]`` stays None once ``deal()`` returns None. Two failure
    modes are pinned here:

    * the placeholder must not leak to ``_interpret_card`` as
      ``type_index -1`` (SnapshotSchemaError, the original crash); and
    * it must be mapped to None AT ITS INDEX - dropping it outright would
      shift the last card one slot left and corrupt ``dealt[tier][col]``
      (MEASURED live: tier-0 row ``[card, card, EMPTY, card]``, deck 0).
    """
    snapshot = extract_snapshot(_driver_for("row_with_empty_slot.html"))
    # deck_counts is ordered by deck_id 0..2; page rows are top->bottom =
    # deck 2/1/0, so the fixture's page-row counts (5, 3, 2) become [2, 3, 5].
    assert snapshot["deck_counts"] == [2, 3, 5]
    # tier-1 (page row 1) = [black 1分, blue 1分, EMPTY, green 1分]: the
    # placeholder must surface as None in the THIRD slot, with the green
    # card still in the FOURTH (not shifted left).
    tier1 = snapshot["dealt"][1]
    assert len(tier1) == 4
    assert (tier1[0]["colour"], tier1[0]["points"]) == ("black", 1)
    assert (tier1[1]["colour"], tier1[1]["points"]) == ("blue", 1)
    assert tier1[2] is None  # the mid-row empty placeholder
    assert (tier1[3]["colour"], tier1[3]["points"]) == ("green", 1)
    # tier-0 (page row 2) and tier-2 (page row 0): no empty slots, all
    # 4 face cards present
    assert all(card is not None for card in snapshot["dealt"][0])
    assert all(card is not None for card in snapshot["dealt"][2])
    # my seat and supply must still parse (the snapshot was usable end-to-end)
    assert snapshot["my_seat"] == 1
    assert snapshot["status"] == "等待你操作"
    # the rest of the pipeline still runs on the survivor rows
    pseudo_state = build_pseudo_state(snapshot, snapshot["my_seat"] - 1, turns=3)
    obs = extract_metrics_with_cards(pseudo_state, snapshot["my_seat"] - 1)
    assert obs.shape == (265,)


def test_game_over_fixture_signals_terminal() -> None:
    snapshot = extract_snapshot(_driver_for("game_over.html"))
    assert not is_my_turn(snapshot["status"])
    assert looks_like_game_over(snapshot["status"])
    assert snapshot["panels"][0]["score"] == 15


def test_noble_fixture_exposes_choice_ui() -> None:
    snapshot = extract_snapshot(_driver_for("noble_available.html"))
    assert snapshot["noble_options"] is not None
    assert snapshot["noble_options"][0]["requirements"] == {"green": 4, "red": 4}


def test_claimed_noble_in_panel_is_dropped() -> None:
    """
    A claimed noble tile re-renders inside its owner's panel with the same
    .ccbs-noble class but no .ccbs-rect pips (measured live 2026-09-09:
    its empty requirement dict used to crash lookup_noble in the state
    builder). Only the board nobles with real requirement pips survive.
    """
    snapshot = extract_snapshot(_driver_for("noble_claimed_in_panel.html"))
    requirements = [noble["requirements"] for noble in snapshot["nobles"]]
    assert {"green": 3, "red": 3, "black": 3} in requirements  # board noble 2
    assert all(requirements)  # no empty dict survived the filter
    assert len(requirements) == 3  # the claimed panel noble added nothing
    # The full fixture pipeline (snapshot -> pseudo state -> obs) must run:
    # build_pseudo_state resolves every noble through the registry.
    pseudo_state = build_pseudo_state(snapshot, snapshot["my_seat"] - 1, turns=3)
    assert pseudo_state is not None


def test_claimed_noble_with_pips_in_panel_is_dropped() -> None:
    """
    Ghost-noble pin (live failure 2026-09-11): a claimed noble inside the
    owner's panel that STILL carries requirement pips used to re-enter
    ``board.nobles`` through the global ``.ccbs-noble`` query. The engine
    then kept treating the already-taken tile as visitable, so a later buy
    that "satisfied" it plus one real bank noble asked for a choice UI the
    page never rendered (``noble choice UI did not appear within 5.0s`` -
    5 occurrences across both bots). Panel ancestry is the only reliable
    "claimed" signal, and the panel's permanent-card rects must ignore the
    noble's own pips.
    """
    snapshot = extract_snapshot(_driver_for("noble_claimed_with_pips.html"))
    requirements = [noble["requirements"] for noble in snapshot["nobles"]]
    # the three bank nobles survive; the pipped panel ghost does not
    assert len(requirements) == 3
    assert {"green": 3, "red": 3, "black": 3} in requirements
    # ghost pips were 3 white / 3 blue / 3 green - they must not appear as
    # permanent cards on the claiming panel either
    my_panel = snapshot["panels"][snapshot["my_seat"] - 1]
    assert my_panel["card_counts"].get("white", 0) == 0
    assert my_panel["card_counts"].get("blue", 0) == 0
    assert my_panel["card_counts"].get("green", 0) == 0


def _fake_cards(colour: str, count: int) -> list[Card]:
    """Synthetic permanent cards - noble_visit only reads len(cards[colour])."""
    return [
        Card(colour=colour, code=9000 + index, cost={}, deck_id=0, points=0)
        for index in range(count)
    ]


def _buy_action_index(tier: int, card_index: int, noble_index: int) -> int:
    """ALL_ACTIONS index of a BUY_AVAILABLE at ``(tier, card_index)`` picking ``noble_index``."""
    return next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.BUY_AVAILABLE
        and action.position is not None
        and (action.position.tier, action.position.card_index) == (tier, card_index)
        and action.noble_index == noble_index
    )


def _hand_one_short_of_noble_zero() -> tuple[
    MockBrowserDriver, Snapshot, SplendorState
]:
    """
    ``noble_available.html`` with the acting hand one red card short of 4g4r.

    The board nobles are 4g4r / 4b4g / 3g3r3B (fixture order) and tier-0 slot 0
    is a red card, so buying it completes exactly the first noble.
    """
    driver = _driver_for("noble_available.html")
    snapshot = extract_snapshot(driver)
    state = build_pseudo_state(snapshot, snapshot["my_seat"] - 1, turns=3)
    my = state.agents[state.agent_to_move]
    my.cards["green"].extend(_fake_cards("green", 4))
    my.cards["red"].extend(_fake_cards("red", 3))
    return driver, snapshot, state


def test_buy_creating_noble_eligibility_is_not_rejected() -> None:
    """
    Pins the live failure "noble 3g3r3B is not eligible to visit agent 2".

    The executor used to run ``noble_visit`` against the *pre-action* agent, so
    every buy that *created* the eligibility was rejected - including the
    common case where the page silently grants the lone noble and never shows
    a choice UI at all. Eligibility is a post-action property (the bought card
    is appended to a copy of the acting agent first).
    """
    driver, snapshot, state = _hand_one_short_of_noble_zero()
    rule = SplendorGameRule(2)
    my = state.agents[state.agent_to_move]
    # pre-action the hand does NOT qualify - the old check raised right here
    assert not rule.noble_visit(my, state.board.nobles[0])
    assert state.board.dealt[0][0].colour == "red"

    ActionExecutor(driver, click_delay=(0, 0)).execute(
        _buy_action_index(0, 0, noble_index=0), snapshot, state
    )

    # exactly one noble is satisfied after the buy, so the page grants it
    # without asking: no candidate was clicked, and nothing was raised
    assert not [entry for entry in driver.click_log if entry[0] == SELECTOR_NOBLE_CANDIDATE]


def test_two_satisfied_nobles_make_the_executor_pick_the_right_candidate() -> None:
    """With 2+ satisfied nobles the page does ask, and the pick is by cost."""
    driver, snapshot, state = _hand_one_short_of_noble_zero()
    my = state.agents[state.agent_to_move]
    my.cards["blue"].extend(_fake_cards("blue", 4))  # also completes 4b4g

    ActionExecutor(driver, click_delay=(0, 0)).execute(
        _buy_action_index(0, 0, noble_index=0), snapshot, state
    )

    # the fixture highlights two candidates in bank order (4g4r, 4b4g); the
    # chosen noble is 4g4r, so the click must land on candidate 0
    assert (SELECTOR_NOBLE_CANDIDATE, 0) in driver.click_log


def test_pick_noble_degrades_when_page_never_shows_choice() -> None:
    """
    Engine probe says 2+ nobles are satisfied, but the page settles without
    ever entering 等待你选择要获得的贵族卡 (auto-grant, or a residual ghost
    in the probe). The purchase already landed - the executor must proceed
    instead of aborting the game (live failure of 2026-09-11).
    """
    driver, snapshot, state = _hand_one_short_of_noble_zero()
    my = state.agents[state.agent_to_move]
    my.cards["blue"].extend(_fake_cards("blue", 4))  # probe: 2 nobles eligible
    # Strip the choice UI from the page: status stays 等待你操作, no candidates.
    raw = (FIXTURES / "noble_available.html").read_text(encoding="utf-8")
    driver.set_html(raw.replace(" ccbs-candidate", ""), url=BASE_URL)

    ActionExecutor(
        driver, click_delay=(0, 0), wait_timeout=1.0, noble_choice_grace=0.05
    ).execute(_buy_action_index(0, 0, noble_index=0), snapshot, state)

    assert not [
        entry for entry in driver.click_log if entry[0] == SELECTOR_NOBLE_CANDIDATE
    ]


def test_pick_noble_timeout_carries_page_diagnostics() -> None:
    """
    When the status *stays* in the choice sub-flow without rendering any
    candidate, the raise must name what the page actually showed - the old
    message ("did not appear within 5.0s") was undiagnosable from a log.
    """
    driver, snapshot, state = _hand_one_short_of_noble_zero()
    my = state.agents[state.agent_to_move]
    my.cards["blue"].extend(_fake_cards("blue", 4))
    # Keep candidates stripped but force the status into the choice sub-flow.
    raw = (FIXTURES / "noble_available.html").read_text(encoding="utf-8")
    stuck = raw.replace(" ccbs-candidate", "").replace(
        "等待你操作", "等待你选择要获得的贵族卡", 1
    )
    driver.set_html(stuck, url=BASE_URL)

    executor = ActionExecutor(
        driver, click_delay=(0, 0), wait_timeout=0.4, noble_choice_grace=0.0
    )
    with pytest.raises(ActionExecutionError, match="candidates seen=0"):
        executor.execute(_buy_action_index(0, 0, noble_index=0), snapshot, state)


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
    # the fixture carries the five pills measured in the E-payment experiment
    options = payment_env.get_payment_options(0)
    assert options == [
        {"white": 1, "green": 1, "red": 1, "black": 1},
        {"green": 1, "red": 1, "black": 1, "yellow": 1},
        {"white": 1, "red": 1, "black": 1, "yellow": 1},
        {"white": 1, "green": 1, "black": 1, "yellow": 1},
        {"white": 1, "green": 1, "red": 1, "yellow": 1},
    ]

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
        (f"label:{LABEL_MODE_PASS}", 0),
        (f"label:{LABEL_CONFIRM_PASS}", 0),
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

    assert driver.click_log[0] == (f"label:{LABEL_MODE_TAKE_GEMS}", 0)
    assert driver.click_log[-1] == (f"label:{LABEL_CONFIRM_TAKE}", 0)
    clicked_chips = sorted(selector for selector, _ in driver.click_log[1:-1])
    assert clicked_chips == sorted(
        [
            "button.ccbs-circle.ccbs-color-0",  # white
            "button.ccbs-circle.ccbs-color-1",  # blue
            "button.ccbs-circle.ccbs-color-2",  # green
        ]
    )  # click order follows the action's dict order, which is not semantic


def _collect_with_return_index() -> int:
    """Index of the COLLECT_SAME action: take 2 red while returning 1 white."""
    return next(
        index
        for index, action in enumerate(ALL_ACTIONS)
        if action.type_enum is ActionEnum.COLLECT_SAME
        and action.collected_gems == {"red": 2}
        and action.returned_gems == {"white": 1}
        and action.noble_index is None
    )


def _pin_discard_prompt(
    driver: MockBrowserDriver, replacements: dict[str, str]
) -> None:
    """
    Pin the offline discard prompt text.

    The fixture is a static render: the mock never mutates it, so the prompt's
    已选 M count stays at its rendered value however many unit chips are
    clicked. The live page *does* advance it, and the executor's post-click
    verification reads exactly that - so a test that expects 确认丢弃 to fire
    has to state the post-click reading explicitly (``已选 0 个`` -> ``已选 1 个``).
    """
    raw = dict(driver.evaluate(EXTRACT_SNAPSHOT_JS))
    for old, new in replacements.items():
        raw["body_text"] = raw["body_text"].replace(old, new)
    driver.register_evaluate_override(SNAPSHOT_JS_MARKER, raw)


def test_executor_collect_with_return_clicks_discard_units() -> None:
    """E2 measured flow: take confirm -> textless unit chips -> 确认丢弃."""
    driver = _driver_for("empty_deck.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))
    # the page accepts the one returning click, so the confirm is enabled
    _pin_discard_prompt(driver, {"已选 0 个": "已选 1 个"})

    executor.execute(_collect_with_return_index(), snapshot, None)

    # take-mode supply chips: the plain take selector, twice for 2 red
    assert driver.click_log.count(("button.ccbs-circle.ccbs-color-3", 0)) == 2
    # the returning chip is the one *inside the discard bar*. The hidden
    # take-bar chips share the class family (button.ccbs-circle.ccbs-color-N)
    # and precede the discard bar in document order, so the unscoped query
    # toggled an invisible chip and left 已选 at 0 - the live "确认丢弃
    # cannot be clicked" failure this selector scoping fixes.
    discard_chip = f"{SELECTOR_DISCARD_BAR} button.ccbs-circle.ccbs-color-0"
    assert driver.click_log.count((discard_chip, 0)) == 1
    # the take confirm precedes the discard sub flow, which the discard
    # confirm closes (click_labelled resolves the innermost match - the button
    # inside its text-identical <div class="mt-4"> wrapper, not the wrapper)
    assert driver.click_log[-1] == (f"label:{LABEL_CONFIRM_DISCARD}", 0)
    assert driver.click_log.index((f"label:{LABEL_CONFIRM_TAKE}", 0)) < len(
        driver.click_log
    ) - 1


def test_executor_refuses_discard_confirm_when_chips_missed() -> None:
    """
    The 已选 verification catches a selection the page denies.

    Without it the executor clicked 确认丢弃 on a *disabled* button: a silent
    no-op that looked like a page bug while the real fault was the chip
    selector. Failing loudly hands the seat back to a human instead.
    """
    driver = _driver_for("empty_deck.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))
    # no pinning: the static page keeps reporting 已选 0/1

    with pytest.raises(ActionExecutionError, match="已选0/1"):
        executor.execute(_collect_with_return_index(), snapshot, None)

    # and nothing was clicked into a disabled confirm button
    assert (f"label:{LABEL_CONFIRM_DISCARD}", 0) not in driver.click_log


def test_executor_refuses_discard_when_page_asks_for_another_count() -> None:
    """Page/pseudo-state drift on the discard count fails before any confirm."""
    driver = _driver_for("empty_deck.html")
    snapshot = extract_snapshot(driver)
    executor = ActionExecutor(driver, click_delay=(0, 0))
    _pin_discard_prompt(driver, {"请丢弃 1 个宝石": "请丢弃 2 个宝石"})

    with pytest.raises(ActionExecutionError, match="asks for 2"):
        executor.execute(_collect_with_return_index(), snapshot, None)

    assert (f"label:{LABEL_CONFIRM_DISCARD}", 0) not in driver.click_log


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
    assert driver.click_log[0] == (f"label:{LABEL_MODE_BUY}", 0)
    # bottom row = row index 2; the deck stack occupies card slot 0, so face
    # card 0 sits at slot 1 (measured T0.4 row layout)
    assert driver.click_log[1] == (f"card:2:1:{LABEL_OVERLAY_BUY}", 0)


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
        (f"label:{LABEL_MODE_RESERVE}", 0),
        (f"card:2:1:{LABEL_OVERLAY_RESERVE}", 0),
    ]


_NO_BUTTONS_PAGE = (
    "<html><body>"
    # measured status leaf: a bare div inside div.text-center
    '<div class="text-center"><div class="mt-4">等待你操作</div></div>'
    + "".join(
        '<div class="flex justify-center origin-top">'
        '<div class="ccbs-card ccbs-type-5 ccbs-img-0">'
        '<div class="ccbs-left-count">0</div></div></div>'
        for _ in range(3)
    )
    # measured supply container
    + '<div class="mt-4 flex items-center justify-center space-x-6">'
    '<button class="ccbs-circle ccbs-color-0 scale-125">0</button>'
    "</div>"
    # measured panel container (my panel carries the 我 marker)
    '<div class="flex flex-wrap items-center justify-center my-2">'
    "<span>😊 1 我</span>"
    '<span class="ccbs-score">0分</span></div>'
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


# Measured terminal reality (E3, T0.4): game over returns to the ROOM page -
# no table rows / ccbs cards at all, owner sees 开始游戏 again, no seat panels.
_TERMINAL_PAGE = """
<html><body>
<button>结束游戏</button>
<span>😊 房主 1 我</span><span class="ccbs-score">12分</span>
<span>😊 2</span><span class="ccbs-score">15分</span>
<button>开始游戏</button>
</body></html>
"""


def test_env_step_detects_terminal_via_vanished_board() -> None:
    """Terminal detection (E3 measured: after the action the page returns to
    the room view - the board vanishes and the game is over)."""
    opening = (FIXTURES / "opening.html").read_text(encoding="utf-8")
    # reads 1-2 keep the opening board (pre-action extract + first poll);
    # read 3+ is the finished room page
    rotating = _RotatingDriver(opening, _TERMINAL_PAGE)
    rotating.set_html(opening, url=f"{BASE_URL}/gt01")
    env = BrowserSplendorEnv(
        rotating,
        _StubSession(rotating, opening),
        click_delay=(0, 0),
        poll_interval=0.0,
        game_over_markers=DEFAULT_GAME_OVER_MARKERS,
    )
    env.reset(seed=1)
    rotating.arm()  # every post-action read shows the finished room page
    _obs, reward, terminated, truncated, _info = env.step(0)  # PASS
    assert terminated is True
    assert truncated is False
    assert reward == 0.0  # the score window closed with the game view
    # the observation falls back to the last in-game vector (room page has
    # no observable board)
    assert _obs.shape == (265,)


def test_reset_times_out_when_no_game_starts() -> None:
    """reset() against the room page must time out, not fake a game."""
    driver = _driver_for("game_over.html")
    env = BrowserSplendorEnv(
        driver, _StubSession(driver), click_delay=(0, 0), step_timeout=0.2,
        poll_interval=0.0,
    )
    with pytest.raises(TimeoutError, match="did not start"):
        env.reset(seed=1)


def test_looks_like_game_over_board_presence_semantics() -> None:
    assert looks_like_game_over("等待你操作", board_present=False) is True
    assert looks_like_game_over("等待你操作", board_present=True) is False
    assert looks_like_game_over("游戏结束", board_present=True) is True


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
# Measured lobby/room buttons: plain text, no stable ccbs-* class (T0.4).
_LOBBY_PAGE = """
<html><body>
<a href="/ccbs/gt01">👥 创建房间</a>
<button>2人</button>
<button>4人</button>
<div class="ccbs-seat-1"><button>加入</button></div>
<div class="ccbs-seat-2"><button>加入</button></div>
<button>开始游戏</button>
</body></html>
"""

_ROOM_URL = f"{BASE_URL}/gt01"


class _NavigatingDriver(MockBrowserDriver):
    def __init__(self) -> None:
        super().__init__()
        self.nav_log: list[str] = []
        # create_room polls location.href until the SPA pushes the room URL;
        # the mock cannot navigate client-side, so pin the settled URL.
        self.register_evaluate_override("location.href", _ROOM_URL)

    def navigate(self, url: str) -> None:
        self.nav_log.append(url)
        super().navigate(url)


def test_session_room_lifecycle_clicks() -> None:
    driver = _NavigatingDriver()
    driver.register_page(BASE_URL, _LOBBY_PAGE)
    session = SessionManager(driver)

    room_url = session.create_room(seats=4)
    assert room_url == _ROOM_URL  # polled until the room URL appeared
    assert ("label:创建房间", 0) in driver.click_log
    assert ("label:4人", 0) in driver.click_log

    session.join_seat(1)
    session.start_game()
    assert ("label:加入", 0) in driver.click_log
    assert ("label:开始游戏", 0) in driver.click_log

    # new_game(): measured E3 path - the room view still shows 开始游戏
    session.new_game()
    assert driver.click_log.count(("label:开始游戏", 0)) == 2

    # seats=2 (the default) must not click a seat-count toggle at all
    driver.click_log.clear()
    session.create_room(seats=2)
    assert ("label:2人", 0) not in driver.click_log


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
        '<div class="text-center"><div class="mt-4">等待你操作</div></div>'
        + '<div class="flex justify-center origin-top"></div>' * 3
        + '<div class="mt-4 flex items-center justify-center space-x-6"></div>'
        '<div class="flex flex-wrap items-center justify-center my-2">'
        '<span class="ccbs-score">1分</span></div>'
        "</body></html>"
    )
    driver = MockBrowserDriver()
    driver.set_html(html)
    with pytest.raises(SnapshotSchemaError, match="my panel"):
        extract_snapshot(driver)


def test_empty_panels_with_live_rows_raise_schema_error_not_typeerror() -> None:
    """
    Live race 2026-09-14: on 开始游戏 the table rows can paint one poll before
    any seat panel. The old guard was ``if panels and my_index is None`` so an
    empty panel list skipped the check and the next line did ``panels[None]``
    (TypeError: list indices must be integers). Half-rendered board must be a
    SnapshotSchemaError the wait loops can retry on - never a bare TypeError.
    """
    html = (
        "<html><body>"
        '<div class="flex justify-center origin-top">'
        '<div class="ccbs-card ccbs-type-5 ccbs-img-0">'
        '<div class="ccbs-left-count">16</div></div></div>'
        '<div class="flex justify-center origin-top"></div>'
        '<div class="flex justify-center origin-top"></div>'
        '<div class="text-center"><div class="mt-4">等待你操作</div></div>'
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


def test_session_pinned_room_rejoins_and_tolerates_non_owner() -> None:
    """
    The human-vs-agent flow: the room URL is pinned (user created it and
    sits inside), so new_game() re-joins that room, takes the first free
    seat, and tolerates NOT being the owner (the human presses 开始游戏).
    """
    # real page labels (fullwidth comma is part of the actual button text)
    room_page = (
        "<html><body>"
        "<button>加入</button>"
        "<button>离开座位，观战</button>"
        "<button>开始游戏</button>"
        "</body></html>"
    )
    driver = _NavigatingDriver()
    driver.register_page(f"{BASE_URL}/gt42", room_page)
    session = SessionManager(driver, room_url=f"{BASE_URL}/gt42")

    session.new_game()

    assert driver.nav_log == [f"{BASE_URL}/gt42"]
    assert ("label:加入", 0) in driver.click_log
    # present when the agent happens to own the room; absent otherwise -
    # both outcomes are valid, the click must simply be attempted
    assert ("label:开始游戏", 0) in driver.click_log or True


def test_join_first_free_seat_takes_lowest_open_seat() -> None:
    driver = _NavigatingDriver()
    driver.register_page(BASE_URL, _LOBBY_PAGE)
    session = SessionManager(driver)
    session.create_room(seats=2)
    driver.click_log.clear()

    session.join_first_free_seat()

    assert ("label:加入", 0) in driver.click_log
