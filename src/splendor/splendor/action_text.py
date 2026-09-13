"""
Human-readable rendering of engine-format action dicts (ActionType).

Shared text for every consumer that shows a move to a human: the
``play_vs_humans`` live reporter, ``play-web``'s decision log and the
phase-7 browser advisor (CLI + dashboard). Lives in the engine package -
deliberately free of torch / browser imports - so a lightweight consumer
can render advice without paying for an inference stack.
"""

from collections.abc import Mapping
from typing import Any

from splendor.splendor.splendor_model import Card

COLOR_CN: dict[str, str] = {
    "white": "白",
    "blue": "蓝",
    "green": "绿",
    "red": "红",
    "black": "黑",
    "yellow": "金",
}


def _gems_cn(counts: Mapping[str, int] | None) -> str:
    """``{'blue': 1, 'red': 2}`` -> ``蓝x1/红x2`` (zero counts dropped)."""
    if not counts:
        return ""
    return "/".join(
        f"{COLOR_CN.get(str(colour), str(colour))}x{count}"
        for colour, count in counts.items()
        if count
    )


def _card_cn(card: Card | None) -> str:
    tier = getattr(card, "deck_id", None)
    colour = COLOR_CN.get(str(getattr(card, "colour", "")), "?")
    points = getattr(card, "points", 0)
    tier_cn = f"{tier + 1}级" if isinstance(tier, int) else ""
    return f"{tier_cn}{colour}卡" + (f"{points}分" if points else "")


def describe_action(action: Mapping[str, Any]) -> str:
    """One-line Chinese description of an engine-format action dict."""
    atype = str(action.get("type", "?"))
    if atype in ("collect_diff", "collect_same"):
        text = f"拿取宝石 {_gems_cn(action.get('collected_gems'))}"
    elif atype == "reserve":
        yellow = (action.get("collected_gems") or {}).get("yellow", 0)
        text = f"预留 {_card_cn(action.get('card'))}" + ("(+1金)" if yellow else "")
    elif atype in ("buy_available", "buy_reserve"):
        where = "桌面" if atype == "buy_available" else "预留"
        text = f"购买{where} {_card_cn(action.get('card'))}"
        payment = _gems_cn(action.get("returned_gems"))
        if payment:
            text += f"(支付 {payment})"
    elif atype == "pass":
        text = "跳过"
    else:
        text = atype
    if action.get("noble"):
        text += ",并获得贵族"
    return text
