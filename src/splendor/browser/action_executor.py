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
on the real page; the multi-noble and discard-row selectors were read out of
the shipped ccbs bundle on 2026-09-11 (the page is a static SPA, so the
bundle is a reliable source of truth for classes and labels).
"""

import copy
import random
import re
import time
from collections.abc import Mapping

from splendor.splendor.constants import MAX_TIER_CARDS, NUMBER_OF_TIERS, RESERVED
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, Action, ActionEnum
from splendor.splendor.splendor_model import Card, SplendorGameRule, SplendorState

from .dom_extractor import (
    COLOR_NAME_TO_INDEX,
    DISCARD_SELECTION_RE,
    PHASE_NOBLE_TEXT,
    NobleInfo,
    Snapshot,
    extract_snapshot,
)
from .driver import BrowserDriver

# --- selectors & labels ------------------------------------------------------
# All identities below were measured on the live page (T0.4 experiments,
# 2026-09-05, docs/web_experiments.md). The action mode buttons, the
# confirm/cancel buttons and the card overlay buttons carry **no stable
# ccbs-* class** - their trimmed text is the only stable identity, so they
# are addressed through click_labelled / click_card_button.

# Mode buttons (emoji-prefixed; exact trimmed texts). [MEASURED]
LABEL_MODE_TAKE_GEMS = "💎取宝石"
LABEL_MODE_BUY = "💰购买发展卡"
LABEL_MODE_RESERVE = "💳预定发展卡"
LABEL_MODE_PASS = "❌放弃"

# Supply chips: button.ccbs-circle.ccbs-color-{c} while in take-gems mode.
# [B3.1, MEASURED] - outside take mode the same chips render as <div>, which
# conveniently de-clutters the selector during the discard sub flow.
SELECTOR_SUPPLY_CHIP_TEMPLATE = "button.ccbs-circle.ccbs-color-{color_index}"

# Table rows: div.flex.justify-center.origin-top (top row = deck_id 2). The
# my-reserved band shares the classes plus bg-gray-400. [B3.1, MEASURED]
SELECTOR_TABLE_ROW = "div.flex.justify-center.origin-top"
SELECTOR_MY_RESERVED_BAND = "div.flex.justify-center.origin-top.bg-gray-400"

# Card overlay labels inside .ccbs-card (no classes - text only). [MEASURED]
LABEL_OVERLAY_BUY = "购买"
LABEL_OVERLAY_RESERVE = "预定"

# Confirm / cancel labels of the gray bar (div.mt-2.p-2.bg-gray-400).
# [B3.1 bar, MEASURED labels incl. the E2 discard step]
LABEL_CONFIRM_TAKE = "确认拿这些"
LABEL_CONFIRM_PASS = "确认放弃"
LABEL_CONFIRM_DISCARD = "确认丢弃"
SELECTOR_GRAY_BAR = "div.mt-2.p-2.bg-gray-400"

# Discard/return unit chips (E2 measured 2026-09-05; selector re-derived from
# the ccbs bundle 2026-09-11). In the discard sub flow each held gem renders
# as an individual textless button.ccbs-circle.ccbs-color-{c} - and those
# chips share their class family with the *take-gems* confirm bar's held
# chips, which survive the take (hidden, `-mt-12 opacity-0`) and precede the
# discard UI in document order. An unscoped query therefore toggled the
# hidden chips and left 已选 at 0, so 确认丢弃 stayed disabled. Scoping to the
# discard chip row fixes it: div.mt-2.space-x-2.p-2.bg-gray-400 is the only
# element with that class combination.
SELECTOR_DISCARD_BAR = "div.space-x-2.p-2.bg-gray-400"
SELECTOR_DISCARD_CHIP_TEMPLATE = (
    SELECTOR_DISCARD_BAR + " button.ccbs-circle.ccbs-color-{color_index}"
)

# Multi-noble choice UI (measured from the ccbs bundle 2026-09-11, replacing
# the earlier [ASSUMED] class guess): while two or more nobles are
# simultaneously satisfied, the page marks each *candidate* bank noble with
# ccbs-candidate and wraps it in a clickable button. There is no
# .ccbs-noble-options container - that selector never matched.
SELECTOR_NOBLE_CANDIDATE = ".ccbs-noble.ccbs-candidate"

# --- etiquette ----------------------------------------------------------------
HUMAN_CLICK_DELAY: tuple[float, float] = (0.2, 0.5)
# After a buy the server needs a beat to settle the purchase and either grant
# a lone noble silently or flip the status into 等待你选择要获得的贵族卡. The
# grace window lets that transition happen before "no choice UI" is trusted.
NOBLE_CHOICE_GRACE_SECONDS = 0.8

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
        noble_choice_grace: float = NOBLE_CHOICE_GRACE_SECONDS,
    ) -> None:
        self._driver = driver
        self._click_delay = click_delay
        self._wait_timeout = wait_timeout
        # How long after the buy click the page gets to *enter* the noble-choice
        # state before the executor concludes it resolved without a UI (auto-
        # grant, or the engine probe over-counted). Live failures showed the
        # page settling in well under a second when no choice is required.
        self._noble_choice_grace = noble_choice_grace
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

    # ----- per-type sequences (BROWSER_RL_MAPPING §4.1, measured T0.4) --------
    def _execute_pass(self) -> None:
        # PASS: pass button -> confirm dialog (2 steps, both measured).
        self._click_labelled(LABEL_MODE_PASS)
        self._click_labelled(LABEL_CONFIRM_PASS)

    def _execute_collect(self, action: Action) -> None:
        # COLLECT: take-gems mode -> per-colour chip clicks -> confirm (3 steps).
        # Measured pitfall (T0.4): clicks must be paced one by one - a burst
        # of synchronous clicks loses all but the last selection.
        collected = action.collected_gems or {}
        if not collected:
            raise ValueError("collect action without collected_gems")
        self._click_labelled(LABEL_MODE_TAKE_GEMS)
        for colour, count in collected.items():
            selector = _supply_chip_selector(colour)
            for _ in range(count):
                self._click_confirmed(selector, 0)
        self._click_labelled(LABEL_CONFIRM_TAKE)
        self._return_gems_if_needed(action)

    def _return_gems_if_needed(self, action: Action) -> None:
        """
        The >10-gems return sub flow (E2 measured, 2026-09-05): after the take
        confirm the page enters a discard step ("请丢弃 N 个宝石，已选 M 个");
        each held gem is an individual textless chip button, toggled by
        clicking, and 确认丢弃 settles the step.
        """
        returned = action.returned_gems or {}
        if not returned:
            return
        for colour, count in returned.items():
            selector = _discard_chip_selector(colour)
            # Each held gem is its own toggle chip: clicking the SAME chip
            # twice cancels the selection, so the i-th gem of a colour needs
            # the i-th chip - not index 0 repeatedly.
            for position in range(count):
                self._click_confirmed(selector, position)
        self._verify_discard_selection(sum(returned.values()))
        self._click_labelled(LABEL_CONFIRM_DISCARD)

    def _verify_discard_selection(self, expected: int) -> None:
        """
        Fail loudly when the unit-chip clicks did not land.

        确认丢弃 is ``disabled`` until 已选 M == 请丢弃 N, and clicking a
        disabled button is a silent no-op - the pre-fix failure mode looked
        like "the confirm button cannot be clicked" while the real cause was a
        chip selector that hit the hidden take-gems chips. The extractor folds
        the prompt's 已选 M/N pair into ``status`` precisely so this check can
        exist; a page that renders no such prompt skips it.
        """
        progress = DISCARD_SELECTION_RE.search(
            extract_snapshot(self._driver)["status"]
        )
        if progress is None:
            return
        selected, required = int(progress.group(1)), int(progress.group(2))
        if required != expected:
            raise ActionExecutionError(
                f"discard sub-flow asks for {required} gem(s) but this action "
                f"returns {expected}: the pseudo state disagrees with the page"
            )
        if selected != expected:
            raise ActionExecutionError(
                f"discard sub-flow shows 已选{selected}/{required} after "
                f"clicking {expected} unit chip(s): the clicks did not land "
                f"(check {SELECTOR_DISCARD_BAR!r} against the live page)"
            )

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
        self._click_labelled(LABEL_MODE_RESERVE)
        # Measured row layout (T0.4): the deck stack is the row's card 0 and
        # carries its own 预定 overlay, so face cards shift by one whenever
        # that deck still has cards.
        deck_offset = 1 if snapshot["deck_counts"][position.tier] > 0 else 0
        self._click_card_button(
            SELECTOR_TABLE_ROW,
            _row_of_deck_id(position.tier),
            deck_offset + position.card_index,
            LABEL_OVERLAY_RESERVE,
        )

    def _execute_buy(
        self, action: Action, snapshot: Snapshot, pseudo_state: SplendorState | None
    ) -> None:
        # BUY: buy mode -> target card's buy overlay -> optional payment pill
        # (2-3 steps; a unique payment settles immediately).
        position = action.position
        if position is None:
            raise ValueError(f"buy action without a position: {action}")
        self._click_labelled(LABEL_MODE_BUY)
        if action.type_enum is ActionEnum.BUY_RESERVE:
            reserved = snapshot["my_reserved"]
            if position.reserved_index not in range(len(reserved)):
                raise ValueError(
                    f"buy_reserve index {position.reserved_index} outside my "
                    f"{len(reserved)} reserved card(s)"
                )
            # The band lists my reserved face cards without a deck stack.
            self._click_card_button(
                SELECTOR_MY_RESERVED_BAND, 0, position.reserved_index, LABEL_OVERLAY_BUY
            )
        else:
            if not _card_exists(snapshot, position.tier, position.card_index):
                raise ValueError(f"buy target not on the page: {action}")
            deck_offset = 1 if snapshot["deck_counts"][position.tier] > 0 else 0
            self._click_card_button(
                SELECTOR_TABLE_ROW,
                _row_of_deck_id(position.tier),
                deck_offset + position.card_index,
                LABEL_OVERLAY_BUY,
            )
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
        # Pills are plain text buttons inside the gray bar (measured T0.4);
        # the splits are pairwise distinct so the text identifies the pill.
        self._click_labelled(pills[choice], container_selector=SELECTOR_GRAY_BAR)

    def _pick_noble(self, action: Action, pseudo_state: SplendorState | None) -> None:
        """
        Claim the noble this action selected - only when the *page* asks.

        Two authorities, deliberately split:

        * the engine probe (``_noble_choices_expected``) is a cheap fast path:
          0-1 satisfied nobles means the page grants silently, so return
          without touching the DOM (the common case, measured live);
        * once the probe expects a choice, the **page status** decides. The
          probe reads ``board.nobles`` from the DOM and can over-count when a
          claimed tile leaks into the bank list; the status line
          (``等待你…选择要获得的贵族卡``) is what the server actually renders
          when a pick is required. If the page settles without ever entering
          that state, the purchase already landed - proceed rather than abort
          a legal action over a reporting mismatch (live failure mode of
          2026-09-11: ``noble choice UI did not appear within 5.0s`` killed
          the game after a fine buy).

        The candidate is located by its cost vector rather than by bank order:
        the page's candidate order is the bank order, but matching on cost
        keeps the executor correct under any re-ordering (noble cost vectors
        are pairwise distinct - card_registry fact).
        """
        if pseudo_state is None:
            raise ValueError("noble claim needs the pseudo state (eligibility)")
        nobles = pseudo_state.board.nobles
        if action.noble_index not in range(len(nobles)):
            raise ValueError(
                f"noble_index {action.noble_index} outside the "
                f"{len(nobles)} board nobles"
            )
        chosen = nobles[action.noble_index]
        expected = _cost_key(chosen[1])
        if not self._noble_choices_expected(action, pseudo_state):
            return  # 0 or 1 satisfied noble: the page grants it without a click
        self._await_noble_choice(chosen, expected)

    def _await_noble_choice(
        self,
        chosen: tuple[str, dict[str, int]],
        expected: tuple[tuple[str, int], ...],
    ) -> None:
        """Poll the page until it picks, settles without asking, or times out."""
        started = time.monotonic()
        deadline = started + self._wait_timeout
        grace_deadline = started + self._noble_choice_grace
        saw_choice_status = False
        candidates_seen = 0
        while True:
            fresh = extract_snapshot(self._driver)
            status = fresh["status"]
            options = fresh["noble_options"] or []
            candidates_seen = max(candidates_seen, len(options))
            if options:
                self._click_noble_candidate(chosen, expected, options)
                return
            if PHASE_NOBLE_TEXT in status:
                saw_choice_status = True  # choice is up: keep the full budget
            elif saw_choice_status or time.monotonic() > grace_deadline:
                # The page left the choice state (or never entered it after
                # the grace window): it resolved the visit without a UI.
                return
            if time.monotonic() >= deadline:
                raise ActionExecutionError(
                    "noble choice UI did not appear within "
                    f"{self._wait_timeout}s while the page still asks for a "
                    f"pick (status={status!r}; expected cost="
                    f"{dict(chosen[1])!r}; candidates seen={candidates_seen}; "
                    f"bank nobles="
                    f"{[dict(n['requirements']) for n in fresh['nobles']]!r})"
                )
            time.sleep(0.2)

    def _click_noble_candidate(
        self,
        chosen: tuple[str, dict[str, int]],
        expected: tuple[tuple[str, int], ...],
        options: list[NobleInfo],
    ) -> None:
        for index, option in enumerate(options):
            if _cost_key(option["requirements"]) == expected:
                self._click_confirmed(SELECTOR_NOBLE_CANDIDATE, index)
                return
        raise ActionExecutionError(
            f"noble {chosen[0]} is not among the {len(options)} "
            "candidate(s) the page highlights (engine/page mismatch); "
            f"expected cost={dict(chosen[1])!r}, "
            f"page candidates={[dict(o['requirements']) for o in options]!r}"
        )

    def _noble_choices_expected(
        self, action: Action, pseudo_state: SplendorState
    ) -> bool:
        """
        Whether the page will block this action on a noble choice.

        Mirrors the page's post-action handler: it re-derives the satisfied
        noble set from the acting player's permanent cards and, when that set
        holds 2+ nobles, waits for a pick; a single satisfied noble is granted
        automatically. Buying a card adds it to a copy of the agent first,
        because eligibility is a post-action property.
        """
        probe = copy.deepcopy(pseudo_state.agents[pseudo_state.agent_to_move])
        if action.type_enum in (ActionEnum.BUY_AVAILABLE, ActionEnum.BUY_RESERVE):
            card = self._bought_card(action, pseudo_state)
            probe.cards[card.colour].append(card)
        satisfied = [
            noble
            for noble in pseudo_state.board.nobles
            if self._rule.noble_visit(probe, noble)
        ]
        return len(satisfied) > 1

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
        return dict(self._bought_card(action, pseudo_state).cost)

    def _bought_card(
        self, action: Action, pseudo_state: SplendorState | None
    ) -> Card:
        """The card a buy action targets (raises for non-buy actions)."""
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
            return my.cards[RESERVED][position.reserved_index]
        valid_position = position.tier in range(
            NUMBER_OF_TIERS
        ) and position.card_index in range(MAX_TIER_CARDS)
        if not valid_position:  # pragma: no cover - guarded in execute()
            raise ValueError(f"buy position outside the board: {action}")
        card = pseudo_state.board.dealt[position.tier][position.card_index]
        if card is None:
            raise ValueError(f"no card at {action.position} to buy")
        return card

    # ----- click plumbing --------------------------------------------------------
    def _click_confirmed(self, selector: str, index: int) -> None:
        """
        One etiquette-paced CSS click followed by a UI-migration wait.

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

    def _click_labelled(
        self,
        label: str,
        *,
        container_selector: str | None = None,
    ) -> None:
        """
        One etiquette-paced text click followed by a UI-migration wait, with
        the same retry-once discipline as the other click helpers.
        """
        last_error: Exception | None = None
        for _ in range(2):
            try:
                if self._click_delay:
                    time.sleep(random.uniform(*self._click_delay))
                self._driver.click_labelled(
                    label, container_selector=container_selector
                )
                self._driver.wait_for("true", self._wait_timeout)
                return
            except Exception as error:  # retried once, then raise
                last_error = error
        raise ActionExecutionError(
            f"labelled click {label!r} failed twice: {last_error}"
        ) from last_error

    def _click_card_button(
        self, container_selector: str, container_index: int, card_index: int, label: str
    ) -> None:
        """Etiquette-paced overlay click (card first, then its button)."""
        last_error: Exception | None = None
        for _ in range(2):
            try:
                if self._click_delay:
                    time.sleep(random.uniform(*self._click_delay))
                self._driver.click_card_button(
                    container_selector, container_index, card_index, label
                )
                self._driver.wait_for("true", self._wait_timeout)
                return
            except Exception as error:  # retried once, then raise
                last_error = error
        raise ActionExecutionError(
            f"card button click {container_selector!r}[{container_index}]:"
            f"{card_index}:{label!r} failed twice: {last_error}"
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


def _discard_chip_selector(colour: str) -> str:
    if colour not in COLOR_NAME_TO_INDEX:
        raise ValueError(f"unknown gem colour {colour!r}")
    return SELECTOR_DISCARD_CHIP_TEMPLATE.format(
        color_index=COLOR_NAME_TO_INDEX[colour]
    )


def _cost_key(cost: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    """
    Order-insensitive identity of a cost vector.

    Noble cost vectors are pairwise distinct (card_registry fact), so the
    sorted item tuple identifies a noble across the engine and the DOM
    without depending on either side's ordering.
    """
    return tuple(sorted((colour, int(count)) for colour, count in cost.items()))


def _row_of_deck_id(deck_id: int) -> int:
    """Web rows run top->bottom = deck_id 2/1/0 (BROWSER_RL_MAPPING §3.1)."""
    return NUMBER_OF_TIERS - 1 - deck_id


def _card_exists(snapshot: Snapshot, tier: int, column: int) -> bool:
    if tier not in range(NUMBER_OF_TIERS) or column not in range(MAX_TIER_CARDS):
        return False
    return snapshot["dealt"][tier][column] is not None
