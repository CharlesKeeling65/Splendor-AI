"""
DOM snapshot extraction: the single doorway from the web world to Python.

Design (plan phase-2 §3.2):

* ``EXTRACT_SNAPSHOT_JS`` is a single IIFE evaluated in one CDP round trip -
  dozens of decisions per game times dozens of data items make per-field queries
  (each a few tens of ms) a real latency cost.
* The JS is deliberately *mechanical*: it only selects ``ccbs-*`` elements and
  reads class indices and text. Every interpretation (web row order ->
  deck_id conversion, colour-index -> colour-name, "N分" parsing, status
  classification) happens once, in Python, in :func:`snapshot_from_raw`.
  ``MockBrowserDriver`` replays the same mechanical selection over offline
  HTML fixtures, so there is exactly one source of interpretation semantics.
* ``extract_snapshot`` validates both the raw reading and the resulting
  ``Snapshot``. A page redesign (renamed ``ccbs-*`` classes) must explode
  here with a field-named error, not downstream as "the agent got worse".
* The schema deliberately excludes opponents' reserved-card slots: the 265-d
  feature only uses *own* reserved cards and rival scores (source ruling,
  UPGRADE_ROADMAP §1.2), so the memory-reconstruction module of
  BROWSER_RL_MAPPING §3.2/§3.3 is out of the minimal loop.

Selector provenance, cited per selector in the JS: ``[B3.1]`` entries were
measured on the real page (BROWSER_RL_MAPPING §3.1); ``[ASSUMED]`` entries
are structural anchors not recorded in §3.1 (panel/supply containers, mode
buttons, ...) and are pending confirmation by the T0.4 page experiments -
they are centralised here and in ``action_executor`` so a fix lands in one
spot.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from splendor.splendor.constants import MAX_TIER_CARDS, NUMBER_OF_TIERS

from .driver import CURRENT_URL_JS, SNAPSHOT_JS_MARKER, BrowserDriver

# ccbs-color-0..5 = 白/蓝/绿/红/黑/金 (measured, BROWSER_RL_MAPPING §3.1);
# ccbs-type-0..4 uses the same order for card colours (type-5 is a card back).
COLOR_INDEX_TO_NAME: tuple[str, ...] = (
    "white",
    "blue",
    "green",
    "red",
    "black",
    "yellow",
)
FACE_INDEX_TO_NAME: tuple[str, ...] = COLOR_INDEX_TO_NAME[:5]
COLOR_NAME_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(COLOR_INDEX_TO_NAME)
}

# Turn status texts (measured, BROWSER_RL_MAPPING §3.1/§5.2).
MY_TURN_TEXT = "等待你操作"
_WAITING_RE = re.compile(r"等待玩家(\d+)操作")

# Terminal-state text markers (E3 experiment still pending - BROWSER_RL_MAPPING
# §5.3 lists candidate signals but the settled DOM feature is unmeasured).
# Keep configurable: the browser env accepts overrides at construction.
DEFAULT_GAME_OVER_MARKERS: tuple[str, ...] = ("游戏结束",)


class SnapshotSchemaError(ValueError):
    """Raised when the page's DOM no longer matches the expected schema."""


# ---------------------------------------------------------------------------
# Raw reading contract (produced by EXTRACT_SNAPSHOT_JS and, offline, by
# MockBrowserDriver). Mechanical values only - class indices and texts.
# ---------------------------------------------------------------------------
class RawCountReading(TypedDict):
    """One ``ccbs-color-N`` element: its colour index and its text."""

    color_index: int
    text: str


class RawCardReading(TypedDict):
    type_index: int  # ccbs-type-N, 0..4 for face cards
    img_index: int  # ccbs-img-N (art number; leaks tier only on type-5 backs)
    score_text: str  # .ccbs-score text, "" when the card shows no points
    circles: list[RawCountReading]  # cost pips


class RawRowReading(TypedDict):
    deck_count_text: str | None  # .ccbs-left-count of the row's deck wrapper
    cards: list[RawCardReading]  # face cards, left to right


class RawNobleReading(TypedDict):
    score_text: str
    rects: list[RawCountReading]


class RawPanelReading(TypedDict):
    text: str  # panel inner text (carries the seat number)
    score_text: str  # .ccbs-score text ("N分"); scoped read so the seat
    # number in `text` can never glue onto the score ("115分" bug)
    rects: list[RawCountReading]  # permanent cards per colour
    circles: list[RawCountReading]  # gems in hand
    reserved_backs: list[int]  # ccbs-img-N of type-5 backs (N leaks tier)


class RawSnapshotReading(TypedDict):
    rows: list[RawRowReading]  # exactly 3, top -> bottom
    nobles: list[RawNobleReading]
    supply: list[RawCountReading]
    panels: list[RawPanelReading]  # DOM order = seat order
    my_panel_index: int | None  # position of my panel (the 我-marked one)
    my_reserved: list[RawCardReading]  # face-up cards in my gray band
    status_text: str
    body_text: str
    payment_pill_texts: list[str] | None
    noble_option_rects: list[list[RawCountReading]] | None


# ---------------------------------------------------------------------------
# Public snapshot schema (IMPLEMENTATION_SPEC §3 T2.1).
# ---------------------------------------------------------------------------
class CardInfo(TypedDict):
    """One card face as the page shows it (tier is the 0-based deck id)."""

    tier: int
    colour: str
    points: int
    cost: dict[str, int]


class NobleInfo(TypedDict):
    """A noble; identity is recovered via the noble registry (§0.3-2)."""

    requirements: dict[str, int]


class PanelInfo(TypedDict):
    """One seat's panel: everything the DOM shows about one player."""

    seat: int
    score: int
    card_counts: dict[str, int]  # permanent cards per colour
    gems: dict[str, int]  # gems in hand, 6 colours incl. yellow
    reserved_tiers: list[int]  # tiers leaked by the reserved card backs


class Snapshot(TypedDict):
    """The full DOM-visible game state, already in engine orientation."""

    dealt: list[list[CardInfo | None]]  # 3x4, indexed by deck_id 0..2
    deck_counts: list[int]  # per deck_id 0..2
    nobles: list[NobleInfo]
    supply: dict[str, int]  # public gem supply, 6 colours
    panels: list[PanelInfo]  # by seat
    my_seat: int  # 1-based page seat number
    my_reserved: list[CardInfo]  # <= 3 face-up reserved cards
    status: str  # 等待你操作 / 等待玩家N操作 / terminal copy (E3 pending)
    payment_options: list[str] | None  # pending payment pill texts
    noble_options: list[NobleInfo] | None  # multi-noble choice UI (E1 pending)


# ---------------------------------------------------------------------------
# EXTRACT_SNAPSHOT_JS - one IIFE, mechanical reading only.
# ---------------------------------------------------------------------------
EXTRACT_SNAPSHOT_JS: str = (
    "// "
    + SNAPSHOT_JS_MARKER
    + """
// Single-round-trip DOM reading for the Splendor browser layer.
// Selector provenance: [B3.1] = plan/reference/BROWSER_RL_MAPPING.md section
// 3.1 (measured on the real page); [MEASURED 2026-09-05] = verified during
// the T0.4 page experiments (docs/web_experiments.md); [ASSUMED] = pending
// live confirmation (multi-noble choice UI only). All interpretation happens in
// Python (dom_extractor.snapshot_from_raw); this script only selects ccbs-*
// elements and reads class indices / text.
(() => {
  const classIndexOf = (el, prefix) => {
    for (const cls of el.classList) {
      if (cls.startsWith(prefix)) {
        const n = parseInt(cls.slice(prefix.length), 10);
        if (!Number.isNaN(n)) { return n; }
      }
    }
    return -1;
  };
  const colorIndexOf = (el) => classIndexOf(el, "ccbs-color-");  // [B3.1]
  const typeIndexOf = (el) => classIndexOf(el, "ccbs-type-");    // [B3.1]
  const imgIndexOf = (el) => classIndexOf(el, "ccbs-img-");
  const readCounts = (scope, selector) =>
    Array.from(scope.querySelectorAll(selector)).map((el) => ({
      color_index: colorIndexOf(el),
      text: (el.textContent || "").trim(),
    }));
  const readCard = (el) => {
    const score = el.querySelector(".ccbs-score");  // [B3.1] absent when 0 pts
    return {
      type_index: typeIndexOf(el),
      img_index: imgIndexOf(el),
      score_text: score ? (score.textContent || "").trim() : "",
      circles: readCounts(el, ".ccbs-circle"),  // [B3.1] cost pips
    };
  };
  // [B3.1] 3 table rows = div.flex.justify-center.origin-top; the my-reserved
  // band shares those classes plus bg-gray-400 and is filtered out.
  const rows = Array.from(
    document.querySelectorAll("div.flex.justify-center.origin-top")
  )
    .filter((el) => !el.classList.contains("bg-gray-400"))
    .map((row) => {
      const back = row.querySelector(".ccbs-card.ccbs-type-5");  // [B3.1] deck
      const count = back
        ? back.querySelector(".ccbs-left-count")  // [B3.1] deck remaining
        : null;
      const cards = Array.from(row.querySelectorAll(".ccbs-card"))
        .filter((el) => typeIndexOf(el) !== 5)
        .map(readCard);
      return {
        deck_count_text: count ? (count.textContent || "").trim() : null,
        cards: cards,
      };
    });
  const nobles = Array.from(document.querySelectorAll(".ccbs-noble")).map(  // [B3.1]
    (noble) => ({
      score_text: noble.querySelector(".ccbs-score")
        ? noble.querySelector(".ccbs-score").textContent.trim()
        : "",
      rects: readCounts(noble, ".ccbs-rect"),  // [B3.1] noble requirements
    })
  );
  // [MEASURED 2026-09-05] supply container: a single row of six chips below
  // the table (div.ccbs-circle.ccbs-color-N.scale-125, <button> only in
  // take-gems mode). A bare ".ccbs-circle" query would hit card-cost pips
  // first - they precede the supply row in document order.
  const supplyArea = document.querySelector(
    "div.mt-4.flex.items-center.justify-center.space-x-6"
  );
  const supply = supplyArea ? readCounts(supplyArea, ".ccbs-circle") : [];
  // [MEASURED 2026-09-05] one div.flex.flex-wrap.items-center.justify-center.my-2
  // per seat, in seat order; my own panel is the one whose text carries 我.
  const panelEls = Array.from(
    document.querySelectorAll("div.flex.flex-wrap.items-center.justify-center.my-2")
  );
  const myPanelIdx = panelEls.findIndex((el) =>
    (el.innerText || el.textContent || "").includes("我")
  );
  const myPanelIndex = myPanelIdx >= 0 ? myPanelIdx : null;
  const panels = panelEls.map((p) => {
    const score = p.querySelector(".ccbs-score");  // [B3.1] "N分" element
    return {
    text: p.innerText || p.textContent || "",
    score_text: score ? (score.textContent || "").trim() : "",
    rects: readCounts(p, ".ccbs-rect"),     // [B3.1] permanent cards
    circles: readCounts(p, ".ccbs-circle"), // [B3.1] gems in hand
    reserved_backs: Array.from(
      p.querySelectorAll(".ccbs-card.ccbs-type-5")  // [B3.1] backs leak tier
    ).map(imgIndexOf),
    };
  });
  // [B3.1] my reserved band: face-up .ccbs-card inside the gray strip.
  const band = document.querySelector(
    "div.flex.justify-center.origin-top.bg-gray-400"
  );
  const myReserved = band
    ? Array.from(band.querySelectorAll(".ccbs-card"))
        .filter((el) => typeIndexOf(el) !== 5)
        .map(readCard)
    : [];
  // [MEASURED 2026-09-05] the turn-status element has no stable class (a
  // bare div.mt-4 inside div.text-center); the *texts* 等待你操作 /
  // 等待玩家N操作 are the stable identity - pick the unique leaf matching.
  const statusEl = Array.from(document.querySelectorAll("*")).find(
    (el) =>
      el.children.length === 0 &&
      /^等待(你|玩家[0-9]+)操作$/.test((el.textContent || "").trim())
  );
  // [MEASURED 2026-09-05] pending payment pills render inside the gray
  // confirm bar next to a 请选择支付方式 heading; pill buttons are plain
  // <button> elements whose text is a digit+colour-character sequence.
  const paymentArea = document.querySelector("div.mt-2.p-2.bg-gray-400");
  const pillRe = /^(?:[0-9]+[白蓝绿红黑金])+$/;
  // [ASSUMED][E1 pending] multi-noble choice UI.
  const nobleArea = document.querySelector(".ccbs-noble-options");
  return {
    rows: rows,
    nobles: nobles,
    supply: supply,
    panels: panels,
    my_panel_index: myPanelIndex,
    my_reserved: myReserved,
    status_text: statusEl ? (statusEl.textContent || "").trim() : "",
    body_text: document.body ? document.body.innerText || "" : "",
    payment_pill_texts:
      paymentArea && (paymentArea.textContent || "").includes("请选择支付方式")
        ? Array.from(paymentArea.querySelectorAll("button"))
            .map((el) => (el.textContent || "").trim())
            .filter((text) => pillRe.test(text))
        : null,
    noble_option_rects: nobleArea
      ? Array.from(nobleArea.querySelectorAll(".ccbs-noble-choice")).map(
          (choice) => readCounts(choice, ".ccbs-rect")
        )
      : null,
  };
})();
"""
)


# ---------------------------------------------------------------------------
# Extraction + validation
# ---------------------------------------------------------------------------
def extract_snapshot(driver: BrowserDriver) -> Snapshot:
    """
    Evaluate EXTRACT_SNAPSHOT_JS once and interpret it into a Snapshot.

    :raises SnapshotSchemaError: when the raw reading or the interpreted
        snapshot violates the schema - i.e. the page changed under us.
    """
    raw = driver.evaluate(EXTRACT_SNAPSHOT_JS)
    if not isinstance(raw, Mapping):
        raise SnapshotSchemaError(
            f"evaluate(EXTRACT_SNAPSHOT_JS) must return an object, got {type(raw)}"
        )
    return snapshot_from_raw(raw)


def _interpret_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[list[CardInfo | None]], list[int]]:
    """
    Interpret the three table rows into engine orientation.

    Web rows run top -> bottom = deck_id 2/1/0 (the tier direction conversion
    is funnelled through this single loop - the most off-by-one-prone spot).
    Empty slots (deck exhausted) are trailing Nones on the page; a mid-row
    empty slot is not producible by the engine's deal order.
    """
    dealt: list[list[CardInfo | None]] = []
    deck_counts: list[int] = []
    for deck_id in range(NUMBER_OF_TIERS):
        row = rows[NUMBER_OF_TIERS - 1 - deck_id]
        cards: list[CardInfo | None] = [
            _interpret_card(raw_card, deck_id)
            for raw_card in row["cards"][:MAX_TIER_CARDS]
        ]
        while len(cards) < MAX_TIER_CARDS:
            cards.append(None)
        dealt.append(cards)
        deck_counts.append(_parse_count(row["deck_count_text"], fallback=0))
    return dealt, deck_counts


def _room_page_snapshot(raw: Mapping[str, Any]) -> Snapshot:
    """Interpret the measured post-game room page (E3): no board at all."""
    panels: list[PanelInfo] = []
    for index, raw_panel in enumerate(raw["panels"]):
        rects = _counts_to_dict(raw_panel["rects"])
        circles = _counts_to_dict(raw_panel["circles"])
        panels.append(
            {
                "seat": index + 1,
                "score": _parse_score(raw_panel["score_text"], raw_panel["text"]),
                "card_counts": {name: rects.get(name, 0) for name in FACE_INDEX_TO_NAME},
                "gems": {name: circles.get(name, 0) for name in COLOR_INDEX_TO_NAME},
                "reserved_tiers": [
                    tier for tier in raw_panel["reserved_backs"] if tier >= 0
                ],
            }
        )
    my_panel_index = raw["my_panel_index"]
    status = raw["status_text"] or _status_from_body(raw["body_text"])
    return {
        "dealt": [[None] * MAX_TIER_CARDS for _ in range(NUMBER_OF_TIERS)],
        "deck_counts": [0, 0, 0],
        "nobles": [],
        "supply": dict.fromkeys(COLOR_INDEX_TO_NAME, 0),
        "panels": panels,
        "my_seat": (my_panel_index + 1) if my_panel_index is not None else 0,
        "my_reserved": [],
        "status": status,
        "payment_options": None,
        "noble_options": None,
    }


def snapshot_from_raw(raw: Mapping[str, Any]) -> Snapshot:
    """Validate a raw reading and interpret it into the public Snapshot."""
    _validate_raw_reading(raw)

    rows = raw["rows"]
    # An empty rows list is the measured room page after game over (E3): the
    # board is simply gone.
    if not rows:
        return _room_page_snapshot(raw)
    dealt, deck_counts = _interpret_rows(rows)

    # Board nobles always render 3-5 requirement pips, so their requirement
    # dict is never empty. A *claimed* noble tile is re-rendered with the
    # same .ccbs-noble class (in its owner's panel area) but without
    # .ccbs-rect pips - such a reading yields an empty requirement dict that
    # no registry entry matches (measured live 2026-09-09: KeyError on {}).
    # Drop empty readings; keep the global query so the board area needs no
    # container assumption.
    nobles: list[NobleInfo] = [
        {"requirements": requirements}
        for noble in raw["nobles"]
        if (requirements := _counts_to_dict(noble["rects"]))
    ]

    supply = dict.fromkeys(COLOR_INDEX_TO_NAME, 0)
    for reading in raw["supply"]:
        name = _color_name(reading["color_index"], "supply")
        supply[name] = _parse_count(reading["text"], fallback=0)

    panels: list[PanelInfo] = []
    for index, raw_panel in enumerate(raw["panels"]):
        rects = _counts_to_dict(raw_panel["rects"])
        circles = _counts_to_dict(raw_panel["circles"])
        panels.append(
            {
                "seat": index + 1,
                "score": _parse_score(raw_panel["score_text"], raw_panel["text"]),
                "card_counts": {name: rects.get(name, 0) for name in FACE_INDEX_TO_NAME},
                "gems": {name: circles.get(name, 0) for name in COLOR_INDEX_TO_NAME},
                "reserved_tiers": [
                    tier for tier in raw_panel["reserved_backs"] if tier >= 0
                ],
            }
        )

    my_panel_index = raw["my_panel_index"]
    if raw["panels"] and (
        my_panel_index is None or my_panel_index not in range(len(raw["panels"]))
    ):
        raise SnapshotSchemaError(
            "my panel not found: no seat panel text carries the 我 marker "
            "(extraction bug or page redesign)"
        )
    my_reserved: list[CardInfo] = []
    panel_backs = raw["panels"][my_panel_index]["reserved_backs"]
    for offset, raw_card in enumerate(raw["my_reserved"]):
        tier = -1
        if len(panel_backs) == len(raw["my_reserved"]):
            # Best-effort tier recovery: my panel shows the same reserved
            # cards as backs whose img index leaks the tier. Order equality
            # across the two regions is an assumption (pending T0.4); the
            # state builder does not depend on it (faces resolve by their
            # unique (colour, points, cost) triple).
            tier = panel_backs[offset]
        info = _interpret_card(raw_card, tier)
        my_reserved.append(info)

    status = raw["status_text"] or _status_from_body(raw["body_text"])

    payment_options = raw["payment_pill_texts"] or None
    raw_noble_rects = raw["noble_option_rects"]
    noble_option_list: list[NobleInfo] | None = None
    if raw_noble_rects:
        noble_option_list = [
            {"requirements": _counts_to_dict(rects)} for rects in raw_noble_rects
        ]

    snapshot: Snapshot = {
        "dealt": dealt,
        "deck_counts": deck_counts,
        "nobles": nobles,
        "supply": supply,
        "panels": panels,
        # 0 = "no seat" (room page after game over; the env treats it as
        # terminal and never builds a pseudo state from such a snapshot)
        "my_seat": (my_panel_index + 1) if my_panel_index is not None else 0,
        "my_reserved": my_reserved,
        "status": status,
        "payment_options": payment_options,
        "noble_options": noble_option_list,
    }
    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    """
    Validate the public Snapshot schema; raise SnapshotSchemaError with the
    offending field path on the first violation.
    """
    for key in Snapshot.__annotations__:
        if key not in snapshot:
            raise SnapshotSchemaError(f"snapshot missing required field {key!r}")

    dealt = snapshot["dealt"]
    _check(
        dealt,
        lambda v: _is_list_of_len(v, NUMBER_OF_TIERS),
        "dealt",
        "list of 3 tier rows",
    )
    for deck_id, row in enumerate(dealt):
        _check(
            row,
            lambda v: _is_list_of_len(v, MAX_TIER_CARDS),
            f"dealt[{deck_id}]",
            "list of 4 card slots",
        )
        for col, card in enumerate(row):
            if card is not None:
                _check_card(card, f"dealt[{deck_id}][{col}]")

    _check(
        snapshot["deck_counts"],
        lambda v: _is_int_list_of_len(v, NUMBER_OF_TIERS),
        "deck_counts",
        "list of 3 ints",
    )

    _check(snapshot["nobles"], _is_list, "nobles", "list")
    for index, noble in enumerate(snapshot["nobles"]):
        _check(
            noble,
            lambda v: isinstance(v, Mapping)
            and _is_cost(v.get("requirements", None)),
            f"nobles[{index}]",
            "NobleInfo with requirements dict[str, int]",
        )

    _check(
        snapshot["supply"],
        lambda v: isinstance(v, Mapping)
        and all(isinstance(key, str) and isinstance(val, int) for key, val in v.items()),
        "supply",
        "dict[str, int]",
    )

    _check(snapshot["panels"], _is_list, "panels", "list")
    for index, panel in enumerate(snapshot["panels"]):
        _check(
            panel,
            lambda v: isinstance(v, Mapping)
            and isinstance(v.get("seat", None), int)
            and isinstance(v.get("score", None), int)
            and _is_cost(v.get("card_counts", None))
            and _is_cost(v.get("gems", None))
            and _is_list(v.get("reserved_tiers", None)),
            f"panels[{index}]",
            "PanelInfo (seat/score ints, card_counts/gems dicts, tiers list)",
        )

    _check(snapshot["my_seat"], lambda v: isinstance(v, int), "my_seat", "int")
    _check(snapshot["my_reserved"], _is_list, "my_reserved", "list")
    for index, card in enumerate(snapshot["my_reserved"]):
        _check_card(card, f"my_reserved[{index}]")

    _check(snapshot["status"], lambda v: isinstance(v, str), "status", "str")
    _check(
        snapshot["payment_options"],
        lambda v: v is None or _is_str_list(v),
        "payment_options",
        "list[str] | None",
    )
    _check(
        snapshot["noble_options"],
        lambda v: v is None or _is_list(v),
        "noble_options",
        "list[NobleInfo] | None",
    )


# ---------------------------------------------------------------------------
# Status classification helpers (page-text semantics, consumed by the env)
# ---------------------------------------------------------------------------
def is_my_turn(status: str) -> bool:
    """Whether the status text says it is my turn to act."""
    return MY_TURN_TEXT in status


def waiting_seat(status: str) -> int | None:
    """The seat number the status text says is acting, when waiting."""
    match = _WAITING_RE.search(status)
    return int(match.group(1)) if match else None


def looks_like_game_over(
    text: str,
    markers: Sequence[str] = DEFAULT_GAME_OVER_MARKERS,
    *,
    board_present: bool = True,
) -> bool:
    """
    Whether the page signals a finished game.

    Measured terminal signal (E3, T0.4 2026-09-05): on game over the page
    returns to the room view - the table rows disappear entirely - so an
    absent board is terminal by itself. Text markers (e.g. 游戏结束) stay
    configurable as a secondary signal for the natural-end screen, whose
    exact DOM lands with the phase-3 live deployment.
    """
    if not board_present:
        return True
    return any(marker in text for marker in markers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _interpret_card(raw_card: Mapping[str, Any], tier: int) -> CardInfo:
    colour_index = raw_card["type_index"]
    if colour_index not in range(len(FACE_INDEX_TO_NAME)):
        raise SnapshotSchemaError(
            f"card type_index {colour_index} outside 0..4 (ccbs-type class bug)"
        )
    cost: dict[str, int] = {}
    for reading in raw_card["circles"]:
        name = _color_name(reading["color_index"], "card cost")
        count = _parse_count(reading["text"], fallback=0)
        if count > 0:
            cost[name] = count
    return {
        "tier": tier,
        "colour": FACE_INDEX_TO_NAME[colour_index],
        "points": _parse_count(raw_card["score_text"], fallback=0),
        "cost": cost,
    }


def _counts_to_dict(readings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reading in readings:
        name = _color_name(reading["color_index"], "ccbs-color element")
        count = _parse_count(reading["text"], fallback=0)
        if count > 0:
            counts[name] = count
    return counts


def _color_name(index: int, context: str) -> str:
    if index not in range(len(COLOR_INDEX_TO_NAME)):
        raise SnapshotSchemaError(
            f"ccbs-color index {index} outside 0..5 in {context} "
            "(colour encoding changed - BROWSER_RL_MAPPING §3.1)"
        )
    return COLOR_INDEX_TO_NAME[index]


def _parse_count(text: str | None, fallback: int) -> int:
    if text is None:
        return fallback
    stripped = text.strip()
    return int(stripped) if stripped.isdigit() else fallback


_SCORE_RE = re.compile(r"(\d+)分")


def _parse_score(score_text: str, panel_text: str) -> int:
    """
    Panel score from the scoped ``.ccbs-score`` element text ("15分" -> 15),
    falling back to the whole panel text when the element is missing.

    The scoped read matters: the raw panel text glues the seat number onto
    the score ("座位1" + "15分" reads as "115分").
    """
    match = _SCORE_RE.search(score_text) or _SCORE_RE.search(panel_text)
    return int(match.group(1)) if match else 0


def _status_from_body(body_text: str) -> str:
    # Fallback when the JS-side status scan came up empty (e.g. game over):
    # fish the measured status phrases out of the whole page text.
    if MY_TURN_TEXT in body_text:
        return MY_TURN_TEXT
    match = _WAITING_RE.search(body_text)
    if match:
        return match.group(0)
    return body_text.strip()


# --- raw-reading validation ------------------------------------------------
def _validate_raw_reading(raw: Mapping[str, Any]) -> None:
    _require_keys(
        raw,
        (
            "rows",
            "nobles",
            "supply",
            "panels",
            "my_panel_index",
            "my_reserved",
            "status_text",
            "body_text",
            "payment_pill_texts",
            "noble_option_rects",
        ),
        "raw reading",
    )
    rows = raw["rows"]
    _check(
        rows,
        lambda v: isinstance(v, list) and len(v) in (0, NUMBER_OF_TIERS),
        "rows",
        "list of exactly 3 rows (in game) or empty (room page after game over)",
    )
    for index, row in enumerate(rows):
        _check(row, _is_mapping, f"rows[{index}]", "RawRowReading object")
        _require_keys(row, ("deck_count_text", "cards"), f"rows[{index}]")
        _check(
            row["deck_count_text"],
            lambda v: v is None or isinstance(v, str),
            f"rows[{index}].deck_count_text",
            "str | None",
        )
        _check(
            row["cards"],
            _is_card_list,
            f"rows[{index}].cards",
            "list of RawCardReading",
        )

    _check(raw["nobles"], _is_list, "nobles", "list")
    for index, noble in enumerate(raw["nobles"]):
        _check(
            noble,
            lambda v: isinstance(v, Mapping)
            and isinstance(v.get("score_text", None), str)
            and _is_count_list(v.get("rects", None)),
            f"nobles[{index}]",
            "RawNobleReading",
        )

    _check(raw["supply"], _is_count_list, "supply", "list of RawCountReading")
    _check(raw["panels"], _is_list, "panels", "list")
    for index, panel in enumerate(raw["panels"]):
        _check(
            panel,
            lambda v: isinstance(v, Mapping)
            and isinstance(v.get("text", None), str)
            and isinstance(v.get("score_text", None), str)
            and _is_count_list(v.get("rects", None))
            and _is_count_list(v.get("circles", None))
            and _is_int_list(v.get("reserved_backs", None)),
            f"panels[{index}]",
            "RawPanelReading",
        )

    _check(
        raw["my_panel_index"],
        lambda v: v is None or isinstance(v, int),
        "my_panel_index",
        "int | None",
    )
    _check(raw["my_reserved"], _is_card_list, "my_reserved", "list of RawCardReading")
    _check(raw["status_text"], lambda v: isinstance(v, str), "status_text", "str")
    _check(raw["body_text"], lambda v: isinstance(v, str), "body_text", "str")
    _check(
        raw["payment_pill_texts"],
        lambda v: v is None or _is_str_list(v),
        "payment_pill_texts",
        "list[str] | None",
    )
    _check(
        raw["noble_option_rects"],
        lambda v: v is None or _is_count_list_list(v),
        "noble_option_rects",
        "list[list[RawCountReading]] | None",
    )


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], what: str) -> None:
    for key in keys:
        if key not in mapping:
            raise SnapshotSchemaError(f"{what} missing required field {key!r}")


def _check(
    value: object, predicate: Callable[[object], bool], path: str, expected: str
) -> None:
    if not predicate(value):
        raise SnapshotSchemaError(
            f"schema violation at {path}: expected {expected}, got {value!r}"
        )


def _is_list(value: object) -> bool:
    return isinstance(value, list)


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_list_of_len(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length


def _is_int_list_of_len(value: object, length: int) -> bool:
    return _is_int_list(value) and len(value) == length  # type: ignore[arg-type]


def _is_count_list_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_count_list(item) for item in value)


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_int_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def _is_count_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, Mapping)
        and isinstance(item.get("color_index", None), int)
        and isinstance(item.get("text", None), str)
        for item in value
    )


def _is_card_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_card(item) for item in value)


def _is_card(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("type_index", None), int)
        and isinstance(value.get("img_index", None), int)
        and isinstance(value.get("score_text", None), str)
        and _is_count_list(value.get("circles", None))
    )


def _is_cost(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(count, int) for key, count in value.items()
    )


def _check_card(card: object, path: str) -> None:
    _check(
        card,
        lambda v: isinstance(v, Mapping)
        and isinstance(v.get("tier", None), int)
        and isinstance(v.get("colour", None), str)
        and isinstance(v.get("points", None), int)
        and _is_cost(v.get("cost", None)),
        path,
        "CardInfo (tier/points int, colour str, cost dict[str, int])",
    )


# Re-exported so consumers (tests, session) can reference the URL-reading JS
# without importing driver internals beyond the public names.
__all__ = [
    "COLOR_INDEX_TO_NAME",
    "CURRENT_URL_JS",
    "DEFAULT_GAME_OVER_MARKERS",
    "EXTRACT_SNAPSHOT_JS",
    "FACE_INDEX_TO_NAME",
    "MY_TURN_TEXT",
    "CardInfo",
    "NobleInfo",
    "PanelInfo",
    "RawCardReading",
    "RawCountReading",
    "RawNobleReading",
    "RawPanelReading",
    "RawRowReading",
    "RawSnapshotReading",
    "Snapshot",
    "SnapshotSchemaError",
    "extract_snapshot",
    "is_my_turn",
    "looks_like_game_over",
    "snapshot_from_raw",
    "validate_snapshot",
    "waiting_seat",
]
