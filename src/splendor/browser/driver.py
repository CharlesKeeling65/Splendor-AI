"""
Browser driver abstraction: the seawall in front of browser-automation tools.

The concrete driver (ego-browser, playwright, raw CDP, ...) lives *below*
this protocol. Browser-automation tool APIs are far less stable than this
repository's own code, so every browser interaction in the layer above
(extraction, execution, session) is written against the eight methods here;
migrating tools then costs exactly one adapter, not a rewrite.

The protocol also exists so tests can inject ``MockBrowserDriver``: all
browser logic runs offline against in-memory HTML fixtures, which is the
prerequisite for CI that never touches the network (plan phase-2 §3.1).

Selection-support note: the mock implements the small selector subset the
browser layer actually uses (tag + ``.class`` compounds, space = descendant
combinator, flat document-order match list addressed by ``index``). Real
drivers natively support CSS selectors, a superset of that subset.
"""

from __future__ import annotations

import re
import xml.sax.saxutils as _sax
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Marker embedded in the first line of dom_extractor.EXTRACT_SNAPSHOT_JS.
# The mock dispatches on this marker instead of importing dom_extractor
# (which imports this module - importing back would be a cycle).
SNAPSHOT_JS_MARKER = "__CCBS_EXTRACT_SNAPSHOT__"

# Special-cased by MockBrowserDriver (and natively supported by every real
# driver): reading the current URL without extending the protocol.
CURRENT_URL_JS = "location.href"

# Payment pill label: digit + Chinese colour-character pairs, e.g. "2白1金"
# (measured T0.4). Used by the mock reader to tell pills apart from the
# confirm/cancel buttons sharing the gray bar.
_PILL_TEXT_RE = re.compile(r"^(?:[0-9]+[白蓝绿红黑金])+$")

# Measured turn-status texts (T0.4 + ccbs bundle 2026-09-11): a leaf element
# flips between these while the game runs; all vanish on game over. Kept here
# because the module must not import dom_extractor (import cycle).
_STATUS_TEXT_RE = re.compile(
    r"等待(你|玩家[0-9]+)"
    r"(?:操作|丢弃多余宝石（每人最多持有10个）|选择要获得的贵族卡)"
)


def _read_status_text(root: _Element) -> str:
    """
    Text of the turn-status leaf, or "".

    The status element has no stable class (measured T0.4: a bare ``div.mt-4``
    inside ``div.text-center``), so the reading keys on the measured *texts*
    instead: the unique childless element matching them. Matching is anchored
    at the start because the page appends 【最后一回合】 on the final round.
    """
    for element in root.iter_tree():
        if any(isinstance(child, _Element) for child in element.children):
            continue  # leaves only, mirroring the JS children.length === 0
        text = element.text_content().strip()
        if _STATUS_TEXT_RE.match(text):
            return text
    return ""

# The card-back type index in the ccbs-type encoding (BROWSER_RL_MAPPING
# §3.1: ccbs-type-5 is only used for deck piles / card backs). Mirrored here
# because this module must not import dom_extractor (import cycle).
CARD_BACK_TYPE_INDEX = 5
# Face cards live in ccbs-type-0..LAST_FACE_TYPE_INDEX (inclusive); the live
# page also renders bare `.ccbs-card.ccbs-empty` placeholders (no
# ccbs-type-N at all) for unoccupied row slots, so the explicit range
# check is required to drop them (live bug, 2026-09-11).
LAST_FACE_TYPE_INDEX = 4


@runtime_checkable
class BrowserDriver(Protocol):
    """
    The eight browser operations the whole browser layer is allowed to use.

    Everything is deliberately low-level and tool-agnostic: one JS string
    evaluator, one clicker, one waiter, and the cookie/screenshot/navigate
    primitives needed by the session manager.
    """

    def evaluate(self, js: str) -> Any:  # noqa: ANN401  # JSON payload by design
        """Execute JS and return the JSON-serializable result (decoded)."""
        ...

    def click(self, selector: str, index: int = 0) -> None:
        """
        Click the ``index``-th element matching ``selector`` (document order).

        :raises ValueError: when no element matches (fail fast instead of
                            clicking into a stale page).
        """
        ...

    def click_labelled(
        self,
        label: str,
        *,
        exact: bool = True,
        index: int = 0,
        container_selector: str | None = None,
        container_index: int = 0,
    ) -> None:
        """
        Click the ``index``-th element whose trimmed text matches ``label``
        (optionally scoped inside the ``container_index``-th container).

        Measured page reality (T0.4 experiments, 2026-09-05): the action mode
        buttons (``💎取宝石`` ...), the confirm/cancel buttons (``确认拿这些``,
        ``确认放弃``, ``确认丢弃``), the payment pills and the room buttons
        (``创建房间`` / ``加入`` / ``开始游戏`` / ``重连``) carry **no stable
        ccbs-* class** - text is their only stable identity, so the protocol
        must expose text-based clicking.

        **Innermost-match rule** (measured 2026-09-11 on the ccbs chunk): a
        text-identical wrapper swallows the click. The discard step renders
        ``<div class="mt-4"><button>确认丢弃</button></div>`` - both elements
        have textContent ``确认丢弃``, and document order puts the *div*
        first, so clicking "the first match" hit a non-interactive div and
        the button was never pressed. Only matches that contain no other
        match in the same scope are candidates, which also keeps ``index``
        meaningful for repeated plain buttons (``加入``, ``预定``).

        :param label: the text to match (trimmed comparison).
        :param exact: when False, any element whose text *contains* the label
                      matches (for emoji-prefixed labels where the emoji glyph
                      may vary across font renders).
        :param index: which match to click (document order within container,
                      after the innermost-match filter).
        :param container_selector: optional CSS scope (e.g. the gray confirm
                                   bar for payment pills).
        :param container_index: which container match to scope into.

        :raises ValueError: when no element matches.
        """
        ...

    def click_card_button(
        self,
        container_selector: str,
        container_index: int,
        card_index: int,
        label: str,
    ) -> None:
        """
        Click the button labelled ``label`` inside the ``card_index``-th
        ``.ccbs-card`` of the ``container_index``-th container match.

        Buy/reserve overlay buttons carry no ccbs-* class either (measured,
        T0.4: pure Tailwind utilities + the texts ``购买`` / ``预定``), and the
        overlay set depends on *affordability* - so the card is addressed
        first (stable board position) and its overlay resolved second, never
        the other way around. Rows list the deck stack as card 0 (measured).

        :raises ValueError: when the container, card or button is missing.
        """
        ...

    def wait_for(self, condition_js: str, timeout: float) -> None:
        """
        Poll ``condition_js`` until it evaluates truthy or ``timeout`` elapses.

        :raises TimeoutError: when the condition never becomes truthy.
        """
        ...

    def navigate(self, url: str) -> None:
        """Load ``url`` in the driver's page."""
        ...

    def get_cookies(self, domain: str) -> list[dict]:
        """Return cookies visible for ``domain`` (host and domain cookies)."""
        ...

    def set_cookie(self, cookie: dict) -> None:
        """Store one cookie (dict with at least name/value/domain)."""
        ...

    def delete_cookies(self, name: str, domain: str) -> None:
        """
        Delete cookies named ``name`` under ``domain``.

        The caller must invoke this once per domain variant: the web game's
        identity cookie may live on either ``game.hullqin.cn`` or
        ``.game.hullqin.cn``, and a leftover on either re-introduces the old
        identity (the double-domain pitfall, BROWSER_RL_MAPPING §2).
        """
        ...

    def screenshot(self, path: str) -> None:
        """Save a screenshot (or page dump, for offline drivers) to ``path``."""
        ...


class _Element:
    """Minimal DOM element node for the mock's in-memory page."""

    __slots__ = ("attrs", "children", "classes", "parent", "tag")

    def __init__(
        self,
        tag: str,
        classes: list[str],
        attrs: dict[str, str],
        parent: _Element | None,
    ) -> None:
        self.tag = tag
        self.classes = classes
        self.attrs = attrs
        self.parent = parent
        self.children: list[_Element | str] = []

    def text_content(self) -> str:
        """Concatenation of all descendant text (textContent semantics)."""
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            else:
                parts.append(child.text_content())
        return "".join(parts)

    def iter_tree(self) -> Iterator[_Element]:
        """Pre-order (document order) traversal of this subtree."""
        yield self
        for child in self.children:
            if isinstance(child, _Element):
                yield from child.iter_tree()

    def class_index(self, prefix: str) -> int | None:
        """Parse ``f"{prefix}<int>"`` out of the class list, if present."""
        for cls in self.classes:
            if cls.startswith(prefix):
                tail = cls[len(prefix) :]
                if tail.isdigit():
                    return int(tail)
        return None


class _TreeBuilder(HTMLParser):
    """Builds an ``_Element`` tree; fixtures are strictly well-formed HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.root = _Element("#root", [], {}, None)
        self._stack: list[_Element] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {key: value or "" for key, value in attrs}
        classes = attr_dict.get("class", "").split()
        element = _Element(tag, classes, attr_dict, self._stack[-1])
        self._stack[-1].children.append(element)
        self._stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {key: value or "" for key, value in attrs}
        classes = attr_dict.get("class", "").split()
        element = _Element(tag, classes, attr_dict, self._stack[-1])
        self._stack[-1].children.append(element)

    def handle_endtag(self, tag: str) -> None:
        for position in range(len(self._stack) - 1, 0, -1):
            if self._stack[position].tag == tag:
                del self._stack[position:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def parse_html(html: str) -> _Element:
    """Parse an HTML document into the mock's element tree."""
    builder = _TreeBuilder()
    builder.feed(html)
    return builder.root


def _matches_compound(element: _Element, tag: str, classes: list[str]) -> bool:
    if tag and element.tag != tag:
        return False
    return all(cls in element.classes for cls in classes)


def _label_matches(element: _Element, label: str, exact: bool) -> bool:
    """Text identity used by click_labelled (trimmed comparison)."""
    text = element.text_content().strip()
    return text == label if exact else label in text


def _is_descendant(element: _Element, ancestor: _Element) -> bool:
    """Whether ``element`` sits anywhere below ``ancestor`` in the tree."""
    parent = element.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _innermost_matches(matches: list[_Element]) -> list[_Element]:
    """
    Keep only matches that contain no other match (``click_labelled`` rule).

    Mirrors the JS side: a wrapper whose textContent is identical to the
    label (e.g. ``<div class="mt-4"><button>确认丢弃</button></div>``) precedes
    its own button in document order and would otherwise absorb the click.

    The same rule is what makes the *contains* mode usable at all. Measured
    2026-09-11 on the live lobby: ``<a href="/ccbs/c496">👥 创建房间</a>``
    (emoji prefix, hence contains mode) has eight matches - ``HTML``, ``BODY``,
    ``#root``, three layout ``DIV``s and finally the ``A``. ``matches[0]`` is
    therefore ``<html>``, whose ``click()`` is a no-op: no navigation, and
    ``SessionManager.create_room`` timed out with a bogus "the create may
    have been refused" hint (web_events/bot0.jsonl).
    """
    return [
        element
        for element in matches
        if not any(
            other is not element and _is_descendant(other, element)
            for other in matches
        )
    ]


def query_selector_all(root: _Element, selector: str) -> list[_Element]:
    """
    Evaluate the mock's selector subset, returning document-order matches.

    Supported: whitespace-separated compounds (descendant combinator); each
    compound is an optional tag name plus ``.class`` tokens. As in CSS, the
    returned elements are the matches of the *last* compound that have the
    required ancestor chain (``".row .buy"`` returns the buy buttons inside
    rows, not the rows themselves).
    """
    compounds: list[tuple[str, list[str]]] = []
    for token in selector.split():
        parts = token.split(".")
        compounds.append((parts[0] or "", [cls for cls in parts[1:] if cls]))
    if not compounds:
        return []

    def matches_chain(element: _Element, compound_index: int) -> bool:
        tag, classes = compounds[compound_index]
        if not _matches_compound(element, tag, classes):
            return False
        if compound_index == 0:
            return True
        ancestor = element.parent
        while ancestor is not None:
            if matches_chain(ancestor, compound_index - 1):
                return True
            ancestor = ancestor.parent
        return False

    last = len(compounds) - 1
    return [
        element for element in root.iter_tree() if matches_chain(element, last)
    ]


class MockBrowserDriver:
    """
    In-memory BrowserDriver for offline tests (plan phase-2 §3.1).

    Capabilities, in the order tests usually need them:

    * ``set_html`` injects a fixture page; ``click`` then validates selectors
      against the real fixture markup and appends to ``click_log``;
    * ``evaluate`` serves registered overrides first, then - when the JS
      carries the snapshot marker - derives the raw DOM reading from the
      parsed fixture with the same *mechanical* selection semantics the
      real EXTRACT_SNAPSHOT_JS uses (the interpretive logic lives only in
      ``dom_extractor``, so there is exactly one source of semantics);
    * ``wait_for`` returns registered condition results, defaulting to
      "condition satisfied" (offline there is no asynchronous UI);
    * a plain cookie jar implements the double-domain delete semantics.
    """

    def __init__(self) -> None:
        self._root: _Element = parse_html("<html><body></body></html>")
        self._url = "about:blank"
        self._pages: dict[str, str] = {}
        self._evaluate_overrides: list[tuple[str, Any]] = []
        self._wait_results: list[tuple[str, bool]] = []
        self._cookies: dict[tuple[str, str], dict] = {}
        self.click_log: list[tuple[str, int]] = []

    # ----- page injection -------------------------------------------------
    def set_html(self, html: str, url: str | None = None) -> None:
        """Replace the current page with ``html`` (a fixture snapshot)."""
        self._root = parse_html(html)
        if url is not None:
            self._url = url

    def register_page(self, url: str, html: str) -> None:
        """Map ``url`` to ``html`` for subsequent ``navigate`` calls."""
        self._pages[url] = html

    # ----- BrowserDriver protocol ----------------------------------------
    def evaluate(self, js: str) -> Any:  # noqa: ANN401  # JSON payload by design
        for pattern, value in self._evaluate_overrides:
            if pattern in js:
                return value(js) if callable(value) else value
        if SNAPSHOT_JS_MARKER in js:
            return read_raw_snapshot(self._root)
        if js.strip() == CURRENT_URL_JS:
            return self._url
        raise ValueError(
            "MockBrowserDriver cannot evaluate arbitrary JS; register an "
            f"evaluate override for it: {js[:80]!r}..."
        )

    def click(self, selector: str, index: int = 0) -> None:
        matches = query_selector_all(self._root, selector)
        if not matches:
            raise ValueError(f"no element matches selector {selector!r}")
        if index not in range(len(matches)):
            raise ValueError(
                f"selector {selector!r} matched {len(matches)} element(s); "
                f"index {index} is out of range"
            )
        self.click_log.append((selector, index))

    def click_labelled(
        self,
        label: str,
        *,
        exact: bool = True,
        index: int = 0,
        container_selector: str | None = None,
        container_index: int = 0,
    ) -> None:
        scope = self._root
        if container_selector is not None:
            containers = query_selector_all(self._root, container_selector)
            if container_index not in range(len(containers)):
                raise ValueError(
                    f"container {container_selector!r} matched "
                    f"{len(containers)} element(s); index {container_index} "
                    "is out of range"
                )
            scope = containers[container_index]

        matches = _innermost_matches(
            [
                element
                for element in scope.iter_tree()
                if _label_matches(element, label, exact)
            ]
        )
        if index not in range(len(matches)):
            raise ValueError(
                f"label {label!r} matched {len(matches)} element(s); "
                f"index {index} is out of range"
            )
        self.click_log.append((f"label:{label}", index))

    def click_card_button(
        self,
        container_selector: str,
        container_index: int,
        card_index: int,
        label: str,
    ) -> None:
        containers = query_selector_all(self._root, container_selector)
        if container_index not in range(len(containers)):
            raise ValueError(
                f"container {container_selector!r} matched "
                f"{len(containers)} element(s); index {container_index} "
                "is out of range"
            )
        cards = query_selector_all(containers[container_index], ".ccbs-card")
        if card_index not in range(len(cards)):
            raise ValueError(
                f"card index {card_index} outside the {len(cards)} card(s) of "
                f"{container_selector!r}[{container_index}]"
            )
        # 2026-09-14 live: buy overlay is "购买?" (trailing ?); fixtures still
        # use plain "购买". Accept either so offline and live stay aligned.
        accepted = {label, label + "?"}
        buttons = [
            element
            for element in cards[card_index].iter_tree()
            if element.tag == "button" and element.text_content().strip() in accepted
        ]
        if not buttons:
            raise ValueError(
                f"no button labelled {label!r} inside card {card_index} of "
                f"{container_selector!r}[{container_index}]"
            )
        self.click_log.append((f"card:{container_index}:{card_index}:{label}", 0))

    def wait_for(self, condition_js: str, timeout: float) -> None:
        for pattern, satisfied in self._wait_results:
            if pattern in condition_js:
                if satisfied:
                    return
                raise TimeoutError(f"mock condition never satisfied: {condition_js!r}")
        # Default: no asynchronous UI offline, conditions are immediately met.

    def navigate(self, url: str) -> None:
        self._url = url
        if url in self._pages:
            self.set_html(self._pages[url], url=url)

    def get_cookies(self, domain: str) -> list[dict]:
        visible = []
        for (_name, cookie_domain), cookie in self._cookies.items():
            if _domain_matches(domain, cookie_domain):
                visible.append(dict(cookie))
        return visible

    def set_cookie(self, cookie: dict) -> None:
        key = (str(cookie["name"]), str(cookie.get("domain", "")))
        self._cookies[key] = dict(cookie)

    def delete_cookies(self, name: str, domain: str) -> None:
        for key in [key for key in self._cookies if _domain_matches(domain, key[1])]:
            if key[0] == name:
                del self._cookies[key]

    def screenshot(self, path: str) -> None:
        # Offline "screenshot": dump the current page text, which is what a
        # human would eyeball in the real browser.
        Path(path).write_text(
            self._root.text_content() or "<empty page>", encoding="utf-8"
        )

    # ----- test hooks -----------------------------------------------------
    def register_evaluate_override(
        self, js_substring: str, value: Any  # noqa: ANN401  # JSON payload
    ) -> None:
        """Serve ``value`` (or ``value(js)`` when callable) for matching JS."""
        self._evaluate_overrides.append((js_substring, value))

    def register_wait_result(self, condition_substring: str, satisfied: bool) -> None:
        """Pin the outcome of ``wait_for`` for matching conditions."""
        self._wait_results.append((condition_substring, satisfied))

    @property
    def current_url(self) -> str:
        return self._url


# ----- raw snapshot reading (shared mechanical selection semantics) --------
def _class_or_minus_one(element: _Element, prefix: str) -> int:
    index = element.class_index(prefix)
    return -1 if index is None else index


def _read_counts(scope: _Element, selector: str) -> list[dict]:
    return [
        {
            "color_index": _class_or_minus_one(el, "ccbs-color-"),
            "text": el.text_content().strip(),
        }
        for el in query_selector_all(scope, selector)
    ]


def _read_card(element: _Element) -> dict:
    scores = query_selector_all(element, ".ccbs-score")
    return {
        "type_index": _class_or_minus_one(element, "ccbs-type-"),
        "img_index": _class_or_minus_one(element, "ccbs-img-"),
        "score_text": scores[0].text_content().strip() if scores else "",
        "circles": _read_counts(element, ".ccbs-circle"),
    }


def _is_face_card(element: _Element) -> bool:
    """
    A face card carries ``ccbs-type-0..4``. Used only for the my-reserved
    band (a variable-length list with no fixed slots): deck backs
    (``ccbs-type-5``) and bare ``.ccbs-empty`` placeholders are dropped
    outright there. Table rows use ``_read_row_card`` instead, which maps
    empty placeholders positionally (see its docstring).
    """
    type_index = _class_or_minus_one(element, "ccbs-type-")
    return 0 <= type_index <= LAST_FACE_TYPE_INDEX


def _read_row_card(element: _Element) -> dict:
    """
    Read one table-row card entry, mapping empty slot placeholders
    positionally.

    When a deck is exhausted the page keeps a bought slot as a bare
    ``.ccbs-card.ccbs-empty`` div AT ITS POSITION (later cards keep their
    slots), mirroring the engine where ``dealt[tier][i]`` stays None once
    ``deal()`` returns None. The placeholder becomes ``{"empty": True, ...}``
    so ``_interpret_rows`` can place None at that index; dropping it would
    shift every later card one slot left and corrupt ``dealt[tier][col]``
    (MEASURED live 2026-09-11: tier-0 row ``[card, card, EMPTY, card]`` with
    deck count 0).
    """
    if "ccbs-empty" in element.classes:
        return {
            "empty": True,
            "type_index": -1,
            "img_index": -1,
            "score_text": "",
            "circles": [],
        }
    return _read_card(element)


def _read_rows(root: _Element) -> list[dict]:
    """
    The three table rows: BROWSER_RL_MAPPING §3.1 documents them as
    ``div.flex.justify-center.origin-top``; the my-reserved band shares those
    classes plus ``bg-gray-400`` and is filtered out.
    """
    rows: list[dict] = []
    for row in query_selector_all(root, "div.flex.justify-center.origin-top"):
        if "bg-gray-400" in row.classes:
            continue
        deck_count = None
        backs = query_selector_all(row, ".ccbs-card.ccbs-type-5")
        if backs:
            counts = query_selector_all(backs[0], ".ccbs-left-count")
            if counts:
                deck_count = counts[0].text_content().strip()
        cards = [
            _read_row_card(el)
            for el in query_selector_all(row, ".ccbs-card")
            if _class_or_minus_one(el, "ccbs-type-") != CARD_BACK_TYPE_INDEX
        ]
        rows.append({"deck_count_text": deck_count, "cards": cards})
    return rows


def _read_nobles(root: _Element) -> list[dict]:
    """
    Bank nobles only: tiles inside a seat panel are *claimed* copies.

    A claimed noble re-renders under its owner's panel with the same
    ``.ccbs-noble`` class (measured 2026-09-09 without requirement pips; a
    later live run showed panel copies that still carried pips - those used
    to re-enter ``board.nobles`` as ghosts and made every later buy look
    noble-eligible). Panel membership is the only reliable "already taken"
    signal the page exposes, so the global query is filtered by ancestry.
    """
    panels = query_selector_all(
        root, "div.flex.flex-wrap.items-center.justify-center.my-2"
    )
    readings: list[dict] = []
    for noble in query_selector_all(root, ".ccbs-noble"):
        if any(_is_descendant(noble, panel) for panel in panels):
            continue
        scores = query_selector_all(noble, ".ccbs-score")
        readings.append(
            {
                "score_text": scores[0].text_content().strip() if scores else "",
                "rects": _read_counts(noble, ".ccbs-rect"),
            }
        )
    return readings


def _read_panels(root: _Element) -> tuple[list[dict], int | None]:
    """
    Read the per-seat panels; also report the position of my own panel.

    Measured container (T0.4, 2026-09-05; redesign 2026-09-14 dropped
    ``my-2``): one ``div.flex.flex-wrap.items-center.justify-center`` per
    seat, in seat order; my own panel is the one whose text carries the ``我``
    marker (now an orange pill badge - innerText still contains it).
    """
    panel_selector = "div.flex.flex-wrap.items-center.justify-center"
    panels: list[dict] = []
    my_panel_index: int | None = None
    for seat_offset, panel in enumerate(query_selector_all(root, panel_selector)):
        if "我" in panel.text_content():
            my_panel_index = seat_offset
        scores = query_selector_all(panel, ".ccbs-score")
        # Card pips only: a claimed .ccbs-noble inside the panel may carry
        # requirement rects that must not inflate permanent-card counts.
        panel_nobles = query_selector_all(panel, ".ccbs-noble")
        card_rects = [
            el
            for el in query_selector_all(panel, ".ccbs-rect")
            if not any(_is_descendant(el, noble) for noble in panel_nobles)
        ]
        # 2026-09-14: panels may have no .ccbs-score element at all; leave
        # score_text empty and let _parse_score fall back to panel text.
        panels.append(
            {
                "text": panel.text_content(),
                "score_text": scores[0].text_content().strip() if scores else "",
                "rects": [
                    {
                        "color_index": _class_or_minus_one(el, "ccbs-color-"),
                        "text": el.text_content().strip(),
                    }
                    for el in card_rects
                ],
                "circles": _read_counts(panel, ".ccbs-circle"),
                "reserved_backs": [
                    _class_or_minus_one(el, "ccbs-img-")
                    for el in query_selector_all(panel, ".ccbs-card.ccbs-type-5")
                ],
            }
        )
    return panels, my_panel_index


def read_raw_snapshot(root: _Element) -> dict:
    """
    Mechanical DOM reading shared with EXTRACT_SNAPSHOT_JS.

    This is intentionally *the same dumb selection* the real JS performs
    (find elements by ccbs-* classes, read class indices and text); all
    interpretation (tier row conversion, colour naming, status parsing)
    happens once, in dom_extractor - both for the real browser and here.
    """
    panels, my_panel_index = _read_panels(root)

    band = query_selector_all(root, "div.flex.justify-center.origin-top.bg-gray-400")
    my_reserved = (
        [
            _read_card(el)
            for el in query_selector_all(band[0], ".ccbs-card")
            if _is_face_card(el)
        ]
        if band
        else []
    )

    status_text = _read_status_text(root)

    # Measured supply container (T0.4): a single row of six chips below the
    # table - div.ccbs-circle.ccbs-color-N.scale-125 when idle, the same
    # elements as <button> while in take-gems mode. A bare ".ccbs-circle"
    # query would hit card-cost pips first (they precede the supply row in
    # document order), so the container is part of the selector.
    supply_area = query_selector_all(
        root, "div.mt-4.flex.items-center.justify-center.space-x-6"
    )
    supply = _read_counts(supply_area[0], ".ccbs-circle") if supply_area else []

    # Measured payment UI (T0.4): pending pills render inside the gray
    # confirm bar (div.mt-2.p-2.bg-gray-400) next to a 请选择支付方式 heading;
    # pill buttons are plain <button> elements (no ccbs-pill class).
    payment_area = query_selector_all(root, "div.mt-2.p-2.bg-gray-400")
    payment_pills = None
    if payment_area and "请选择支付方式" in payment_area[0].text_content():
        payment_pills = [
            el.text_content().strip()
            for el in query_selector_all(payment_area[0], "button")
            if _PILL_TEXT_RE.match(el.text_content().strip())
        ]

    # Multi-noble choice UI (measured 2026-09-11 from the ccbs bundle): when
    # 2+ nobles are simultaneously satisfied the page wraps each *candidate*
    # bank noble in a clickable button and tags the noble tile itself with
    # ccbs-candidate. There is no .ccbs-noble-options container.
    noble_candidates = query_selector_all(root, ".ccbs-noble.ccbs-candidate")
    noble_options = (
        [_read_counts(choice, ".ccbs-rect") for choice in noble_candidates]
        or None
    )

    return {
        "rows": _read_rows(root),
        "nobles": _read_nobles(root),
        "supply": supply,
        "panels": panels,
        "my_panel_index": my_panel_index,
        "my_reserved": my_reserved,
        "status_text": status_text,
        "body_text": root.text_content(),
        "payment_pill_texts": payment_pills,
        "noble_option_rects": noble_options,
    }


def _domain_matches(requested: str, stored: str) -> bool:
    """
    Domain matching for the mock cookie jar: leading dots are ignored and a
    host matches its own domain cookie (approximates cookie scoping closely
    enough for the double-domain delete recipe to be exercised).
    """
    requested_host = requested.lstrip(".")
    stored_host = stored.lstrip(".")
    return requested_host == stored_host or requested_host.endswith(
        f".{stored_host}"
    ) or stored_host.endswith(f".{requested_host}")


def escape_html_text(text: str) -> str:
    """Escape text for fixture generation (kept next to the mock on purpose)."""
    return _sax.escape(text)
