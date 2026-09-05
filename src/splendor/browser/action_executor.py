"""
Action executor: policy intent -> web click sequences (BROWSER_RL_MAPPING §4.1).

Every click is surrounded by two safeguards:

* *etiquette*: a randomised human-pace delay before each click
  (``HUMAN_CLICK_DELAY``) - a hard constraint of this project, not a nicety
  (roadmap R7);
* *confirmation*: ``wait_for`` after each click asserting the UI actually
  migrated; one retry covers transient jank, a second failure raises
  ``ActionExecutionError`` and hands control back to a human.

Selector constants are centralised here: when the page is redesigned, this
module is the single place to fix. Selectors marked ``[B3.1]`` were measured
on the real page; ``[ASSUMED]`` ones are pending the T0.4 experiments.
"""

import random
import re
import time

from splendor.splendor.constants import MAX_TIER_CARDS, NUMBER_OF_TIERS, RESERVED
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, Action, ActionEnum
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState

from .dom_extractor import (
    COLOR_NAME_TO_INDEX,
    Snapshot,
    extract_snapshot,
)
from .driver import BrowserDriver

# --- selectors ---------------------------------------------------------------
# Top action buttons; page labels are emoji-prefixed texts (take gems / buy /
# reserve / pass). Classes assumed pending T0.4. [ASSUMED]
SELECTOR_MODE_TAKE_GEMS = "button.ccbs-take-gems"
SELECTOR_MODE_BUY = "button.ccbs-buy-card"
SELECTOR_MODE_RESERVE = "button.ccbs-reserve-card"
SELECTOR_MODE_PASS = "button.ccbs-pass"

# Supply chips: button.ccbs-circle.ccbs-color-{c} on my turn. [B3.1]
SELECTOR_SUPPLY_CHIP_TEMPLATE = "button.ccbs-circle.ccbs-color-{color_index}"

# Overlay buttons rendered on cards while in buy/reserve mode. [ASSUMED]
SELECTOR_OVERLAY_BUY = ".ccbs-overlay-buy"
SELECTOR_OVERLAY_RESERVE = ".ccbs-overlay-reserve"

# Table rows: ccbs-row-{0,1,2} top->bottom, scoped descendant selectors for
# targeting a specific card's overlay. [ASSUMED]
SELECTOR_ROW_OVERLAY_TEMPLATE = ".ccbs-row-{row} {overlay}"

# My reserved band (face-up cards): the gray strip. [B3.1 band, ASSUMED class]
SELECTOR_MY_RESERVED_BUY = "div.bg-gray-400.ccbs-my-reserved .ccbs-overlay-buy"

# Confirm/cancel area: gray bar div.mt-2.p-2.bg-gray-400 with 确认拿这些. [B3.1]
SELECTOR_CONFIRM_TAKE = "button.ccbs-confirm-take"
# Pass dialog confirm button (确认放弃). [ASSUMED]
SELECTOR_CONFIRM_PASS = "button.ccbs-confirm-pass"
# Return-gems confirm button inside the >10-gems sub flow. [ASSUMED][E2 pending]
SELECTOR_CONFIRM_RETURN = "button.ccbs-confirm-return"

# Payment pills inside the 请选择支付方式 selector. [B3.1 pills, ASSUMED class]
SELECTOR_PAYMENT_PILL = ".ccbs-payment-options .ccbs-pill"

# Multi-noble choice UI. [ASSUMED][E1 pending]
SELECTOR_NOBLE_CHOICE = ".ccbs-noble-options .ccbs-noble-choice"

# --- etiquette ----------------------------------------------------------------
HUMAN_CLICK_DELAY: tuple[float, float] = (0.2, 0.5)

# Pill texts are digit + Chinese colour-character pairs, e.g. "2白1金".
_PILL_RE = re.compile(r"([0-9]+)([白蓝绿红黑金])")
# Page colour characters -> engine colour names, aligned with ccbs-color order
# (白/蓝/绿/红/黑/金, BROWSER_RL_MAPPING §3.1).
_PILL_CHAR_TO_COLOR = {
    "白": "white",
    "蓝": "blue",
    "绿": "green",
    "红": "red",
    "黑": "black",
    "金": "yellow",
}


class ActionExecutionError(RuntimeError):
    """A click sequence failed twice; a human should take over."""


def parse_pill(pill_text: str) -> dict[str, int]:
    """
    Parse a payment pill label like ``"2白1金"`` into a gems dict.

    :raises ValueError: when the text contains no recognisable count/colour
                        pair (a pill format change breaks the executor and
                        must surface here, not as a wrong payment).
    """
    payment: dict[str, int] = {}
    for count_text, color_char in _PILL_RE.findall(pill_text):
        payment[_PILL_CHAR_TO_COLOR[color_char]] = int(count_text)
    if not payment:
        raise ValueError(f"unparseable payment pill text: {pill_text!r}")
    return payment


class ActionExecutor:
    """Executes ALL_ACTIONS indices as measured click sequences."""

    def __init__(
        self,
        driver: BrowserDriver,
        click_delay: tuple[float, float] = HUMAN_CLICK_DELAY,
        wait_timeout: float = 5.0,
    ) -> None:
        self._driver = driver
        self._click_delay = click_delay
        self._wait_timeout = wait_timeout
        # Engine instance used only to *reuse* payment/noble semantics (see
        # select_payment_greedy); its throwaway random initial state is the
        # price of keeping those semantics single-sourced in the engine.
        self._rule = SplendorGameRule(2)

    # ----- entry point --------------------------------------------------------
    def execute(
        self,
        action_index: int,
        snapshot: Snapshot,
        pseudo_state: SplendorState | None = None,
    ) -> None:
        """
        Perform the click sequence of ``ALL_ACTIONS[action_index]``.

        :param snapshot: the snapshot this action was decided on (used for
            DOM preconditions); the executor re-reads the page where the UI
            state changes mid-sequence (payment pills).
        :param pseudo_state: required for actions whose semantics need engine
            data (BUY payment strategy, noble eligibility); PASS/COLLECT/
            RESERVE sequences never touch it, and tests may pass None there.
        :raises ActionExecutionError: when a step fails twice.
        :raises ValueError: on an out-of-range index or unmet preconditions.
        """
        if action_index not in range(len(ALL_ACTIONS)):
            raise ValueError(f"action index {action_index} outside ALL_ACTIONS")
        action = ALL_ACTIONS[action_index]

        if action.type_enum is ActionEnum.PASS:
            self._execute_pass()
        elif action.type_enum in (ActionEnum.COLLECT_SAME, ActionEnum.COLLECT_DIFF):
            self._execute_collect(action)
        elif action.type_enum is ActionEnum.RESERVE:
            self._execute_reserve(action, snapshot)
        elif action.type_enum in (ActionEnum.BUY_AVAILABLE, ActionEnum.BUY_RESERVE):
            self._execute_buy(action, snapshot, pseudo_state)
        else:  # pragma: no cover - ActionEnum is a closed set
            raise ValueError(f"unsupported action type {action.type_enum}")

        if action.noble_index is not None:
            self._pick_noble(action, pseudo_state)

    # ----- per-type sequences (BROWSER_RL_MAPPING §4.1) -----------------------
    def _execute_pass(self) -> None:
        # PASS: pass button -> confirm dialog (2 steps).
        self._click_confirmed(SELECTOR_MODE_PASS, 0)
        self._click_confirmed(SELECTOR_CONFIRM_PASS, 0)

    def _execute_collect(self, action: Action) -> None:
        # COLLECT: take-gems mode -> per-colour chip clicks -> confirm (3 steps).
        collected = action.collected_gems or {}
        if not collected:
            raise ValueError("collect action without collected_gems")
        self._click_confirmed(SELECTOR_MODE_TAKE_GEMS, 0)
        for colour, count in collected.items():
            selector = _supply_chip_selector(colour)
            for _ in range(count):
                self._click_confirmed(selector, 0)
        self._click_confirmed(SELECTOR_CONFIRM_TAKE, 0)
        self._return_gems_if_needed(action)

    def _return_gems_if_needed(self, action: Action) -> None:
        """
        The >10-gems return sub flow (E2 unmeasured): click the chips to give
        back, then confirm. Implemented against assumed selectors so the flow
        exists end to end; T0.4 confirms the exact interaction.
        """
        returned = action.returned_gems or {}
        if not returned:
            return
        for colour, count in returned.items():
            selector = _supply_chip_selector(colour)
            for _ in range(count):
                self._click_confirmed(selector, 0)
        self._click_confirmed(SELECTOR_CONFIRM_RETURN, 0)

    def _execute_reserve(self, action: Action, snapshot: Snapshot) -> None:
        # RESERVE: reserve mode -> target card's reserve overlay (2 steps; the
        # page grants the yellow gem automatically).
        position = action.position
        if position is None or position.tier not in range(NUMBER_OF_TIERS):
            raise ValueError(f"reserve action without a board position: {action}")
        if not _card_exists(snapshot, position.tier, position.card_index):
            raise ValueError(
                f"reserve target (tier={position.tier}, "
                f"col={position.card_index}) has no card on the page"
            )
        self._click_confirmed(SELECTOR_MODE_RESERVE, 0)
        selector = SELECTOR_ROW_OVERLAY_TEMPLATE.format(
            row=_row_of_deck_id(position.tier), overlay=SELECTOR_OVERLAY_RESERVE
        )
        # In reserve mode the deck wrapper (row's first element) also carries a
        # reserve overlay - reserving from the deck top - so card slots shift
        # by one whenever that deck still has cards. [ASSUMED ordering]
        deck_offset = 1 if snapshot["deck_counts"][position.tier] > 0 else 0
        self._click_confirmed(selector, deck_offset + position.card_index)

    def _execute_buy(
        self, action: Action, snapshot: Snapshot, pseudo_state: SplendorState | None
    ) -> None:
        # BUY: buy mode -> target card's buy overlay -> optional payment pill
        # (2-3 steps; a unique payment settles immediately).
        position = action.position
        if position is None:
            raise ValueError(f"buy action without a position: {action}")
        self._click_confirmed(SELECTOR_MODE_BUY, 0)
        if action.type_enum is ActionEnum.BUY_RESERVE:
            reserved = snapshot["my_reserved"]
            if position.reserved_index not in range(len(reserved)):
                raise ValueError(
                    f"buy_reserve index {position.reserved_index} outside my "
                    f"{len(reserved)} reserved card(s)"
                )
            selector = SELECTOR_MY_RESERVED_BUY
            overlay_index = position.reserved_index
        else:
            if not _card_exists(snapshot, position.tier, position.card_index):
                raise ValueError(f"buy target not on the page: {action}")
            selector = SELECTOR_ROW_OVERLAY_TEMPLATE.format(
                row=_row_of_deck_id(position.tier), overlay=SELECTOR_OVERLAY_BUY
            )
            # Buy overlays never appear on the deck wrapper, so card slots map
            # 1:1 onto the overlay match list. [ASSUMED ordering]
            overlay_index = position.card_index
        self._click_confirmed(selector, overlay_index)
        self._settle_payment(action, pseudo_state)

    def _settle_payment(
        self, action: Action, pseudo_state: SplendorState | None
    ) -> None:
        """
        Settle a purchase when the page asks for a payment choice.

        The pills only render *after* the buy-overlay click, so the page is
        re-read here; a snapshot without pills means the purchase already
        settled (unique payment). Re-reading once more on ambiguity is left
        to the real-page integration pass (pending T0.4 timing data).
        """
        fresh = extract_snapshot(self._driver)
        pills = fresh["payment_options"]
        if not pills:
            return
        if pseudo_state is None:
            raise ValueError(
                "payment pills pending but no pseudo state supplied - the "
                "greedy payment strategy needs the engine view of the seat"
            )
        choice = self.select_payment_greedy(pills, action, pseudo_state)
        self._click_confirmed(SELECTOR_PAYMENT_PILL, choice)

    def _pick_noble(self, action: Action, pseudo_state: SplendorState | None) -> None:
        """
        Claim the noble this action selected when the choice UI is up (E1
        pending). The UI is assumed to list the *eligible* nobles; eligibility
        is engine code (``noble_visit``), so the click index is derived by
        engine rule reuse instead of assuming board order.
        """
        if pseudo_state is None:
            raise ValueError("noble claim needs the pseudo state (eligibility)")
        nobles = pseudo_state.board.nobles
        if action.noble_index not in range(len(nobles)):
            raise ValueError(
                f"noble_index {action.noble_index} outside the "
                f"{len(nobles)} board nobles"
            )
        noble = nobles[action.noble_index]
        agent = pseudo_state.agents[pseudo_state.agent_to_move]
        eligible = [n for n in nobles if self._rule.noble_visit(agent, n)]
        if noble not in eligible:
            raise ValueError(
                f"noble {noble[0]} is not eligible to visit agent "
                f"{agent.id} (engine rule says no)"
            )
        self._click_confirmed(SELECTOR_NOBLE_CHOICE, eligible.index(noble))

    # ----- payment strategy -----------------------------------------------------
    def select_payment_greedy(
        self, pills: list[str], action: Action, pseudo_state: SplendorState | None
    ) -> int:
        """
        Tier-1 payment policy: imitate the engine's greedy payment.

        The engine settles every purchase with one greedy payment (coloured
        gems first, gold only fills the shortfall - ``resources_sufficient``,
        splendor_model.py:371-394). Local training never saw states produced
        by other payment splits, so consistency beats local optimality: pick
        the pill whose gold usage matches the engine's (minimal), preferring
        the exact same split. Any other choice would push the live game out
        of the training distribution. The explicit payment dimension is
        phase-4 (tier-2) work.

        :returns: the index into ``pills`` to click.
        """
        if pseudo_state is None:
            raise ValueError("payment strategy needs the pseudo state")
        cost = self._cost_of(action, pseudo_state)
        agent = pseudo_state.agents[pseudo_state.agent_to_move]
        greedy = self._rule.resources_sufficient(agent, cost)

        parsed = [parse_pill(pill) for pill in pills]
        greedy_gold = greedy.get("yellow", 0)
        candidates = [
            index
            for index, payment in enumerate(parsed)
            if payment.get("yellow", 0) == greedy_gold
        ]
        if not candidates:
            # Defensive: the page's options disagree with the pseudo state's
            # greedy gold usage - fall back to minimal gold usage.
            candidates = sorted(
                range(len(parsed)), key=lambda i: parsed[i].get("yellow", 0)
            )
        for index in candidates:
            if parsed[index] == greedy:
                return index
        return candidates[0]

    def _cost_of(
        self, action: Action, pseudo_state: SplendorState | None
    ) -> dict[str, int]:
        """Cost dict of the card a buy action targets."""
        position = action.position
        if position is None:
            raise ValueError(f"buy action without a position: {action}")
        if pseudo_state is None:  # pragma: no cover - guarded by caller
            raise ValueError("buy cost lookup needs the pseudo state")
        if action.type_enum is ActionEnum.BUY_RESERVE:
            my = pseudo_state.agents[pseudo_state.agent_to_move]
            if position.reserved_index not in range(len(my.cards[RESERVED])):
                raise ValueError(
                    f"buy_reserve index {position.reserved_index} has no card"
                )
            card = my.cards[RESERVED][position.reserved_index]
        else:
            valid_position = position.tier in range(
                NUMBER_OF_TIERS
            ) and position.card_index in range(MAX_TIER_CARDS)
            if not valid_position:  # pragma: no cover - guarded in execute()
                raise ValueError(f"buy position outside the board: {action}")
            card = pseudo_state.board.dealt[position.tier][position.card_index]
            if card is None:
                raise ValueError(f"no card at {action.position} to buy")
        return dict(card.cost)

    # ----- click plumbing --------------------------------------------------------
    def _click_confirmed(self, selector: str, index: int) -> None:
        """
        One etiquette-paced click followed by a UI-migration wait.

        Failures are retried exactly once (transient render jank); a second
        failure raises - blindly clicking into an unknown page state is the
        race-condition accident source this design exists to avoid.
        """
        last_error: Exception | None = None
        for _ in range(2):
            try:
                if self._click_delay:
                    time.sleep(random.uniform(*self._click_delay))
                self._driver.click(selector, index)
                self._driver.wait_for("true", self._wait_timeout)
                return
            except Exception as error:  # retried once, then raise
                last_error = error
        raise ActionExecutionError(
            f"click {selector!r}[{index}] failed twice: {last_error}"
        ) from last_error

    def force_pass(self) -> None:
        """
        Degraded rescue used on step timeouts: pass to keep the seat alive.
        Losing one action is always better than a forfeit on hard timeout.
        """
        self._execute_pass()


def _supply_chip_selector(colour: str) -> str:
    if colour not in COLOR_NAME_TO_INDEX:
        raise ValueError(f"unknown gem colour {colour!r}")
    return SELECTOR_SUPPLY_CHIP_TEMPLATE.format(color_index=COLOR_NAME_TO_INDEX[colour])


def _row_of_deck_id(deck_id: int) -> int:
    """Web rows run top->bottom = deck_id 2/1/0 (BROWSER_RL_MAPPING §3.1)."""
    return NUMBER_OF_TIERS - 1 - deck_id


def _card_exists(snapshot: Snapshot, tier: int, column: int) -> bool:
    if tier not in range(NUMBER_OF_TIERS) or column not in range(MAX_TIER_CARDS):
        return False
    return snapshot["dealt"][tier][column] is not None
