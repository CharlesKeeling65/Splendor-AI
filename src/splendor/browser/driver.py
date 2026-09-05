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

# The card-back type index in the ccbs-type encoding (BROWSER_RL_MAPPING
# §3.1: ccbs-type-5 is only used for deck piles / card backs). Mirrored here
# because this module must not import dom_extractor (import cycle).
CARD_BACK_TYPE_INDEX = 5


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


def query_selector_all(root: _Element, selector: str) -> list[_Element]:
    """
    Evaluate the mock's selector subset, returning document-order matches.

    Supported: whitespace-separated compounds (descendant combinator); each
    compound is an optional tag name plus ``.class`` tokens.
    """
    compounds: list[tuple[str, list[str]]] = []
    for token in selector.split():
        parts = token.split(".")
        compounds.append((parts[0] or "", [cls for cls in parts[1:] if cls]))
    if not compounds:
        return []

    matches: list[_Element] = []

    def matches_from(element: _Element, compound_index: int) -> bool:
        tag, classes = compounds[compound_index]
        if not _matches_compound(element, tag, classes):
            return False
        return compound_index == len(compounds) - 1 or any(
            matches_from(descendant, compound_index + 1)
            for descendant in element.iter_tree()
            if descendant is not element
        )

    for element in root.iter_tree():
        if matches_from(element, 0):
            matches.append(element)
    return matches


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
            _read_card(el)
            for el in query_selector_all(row, ".ccbs-card")
            if _class_or_minus_one(el, "ccbs-type-") != CARD_BACK_TYPE_INDEX
        ]
        rows.append({"deck_count_text": deck_count, "cards": cards})
    return rows


def _read_nobles(root: _Element) -> list[dict]:
    readings: list[dict] = []
    for noble in query_selector_all(root, ".ccbs-noble"):
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
    Read the per-seat panels; also report the position of my own panel
    (the one carrying the ccbs-me marker), or None when absent.
    """
    panels: list[dict] = []
    my_panel_index: int | None = None
    for seat_offset, panel in enumerate(query_selector_all(root, ".ccbs-player")):
        if "ccbs-me" in panel.classes:
            my_panel_index = seat_offset
        panels.append(
            {
                "text": panel.text_content(),
                "rects": _read_counts(panel, ".ccbs-rect"),
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
            if _class_or_minus_one(el, "ccbs-type-") != CARD_BACK_TYPE_INDEX
        ]
        if band
        else []
    )

    status_elements = query_selector_all(root, ".ccbs-status")
    status_text = status_elements[0].text_content().strip() if status_elements else ""

    supply_area = query_selector_all(root, ".ccbs-supply")
    supply = _read_counts(supply_area[0], ".ccbs-circle") if supply_area else []

    payment_area = query_selector_all(root, ".ccbs-payment-options")
    payment_pills = (
        [
            el.text_content().strip()
            for el in query_selector_all(payment_area[0], ".ccbs-pill")
        ]
        if payment_area
        else None
    )

    noble_area = query_selector_all(root, ".ccbs-noble-options")
    noble_options = (
        [
            _read_counts(choice, ".ccbs-rect")
            for choice in query_selector_all(noble_area[0], ".ccbs-noble-choice")
        ]
        if noble_area
        else None
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
