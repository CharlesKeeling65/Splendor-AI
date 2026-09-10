"""
Status-fallback regression tests (phase-6 fix): the transitional-page
fallback in ``dom_extractor._status_from_body`` must never return the raw
body text. The pre-fix behaviour (whole innerText - dozens of
newline-separated numbers during the discard sub-flow and other
transitional states) garbled every log line and TimeoutError carrying a
status ("很多带换行的数字" symptom).
"""

import pytest

from splendor.browser.dom_extractor import (
    DEFAULT_GAME_OVER_MARKERS,
    MY_TURN_TEXT,
    _status_from_body,  # noqa: PLC2701 - regression target is the private path
)


def test_discard_phrase_becomes_compact_label() -> None:
    body = "宝石不足\n💎取宝石\n请丢弃 2 个宝石\n白 x3\n确认丢弃\n座位1\n3分"
    status = _status_from_body(body)
    assert status == "等待丢弃2个宝石"
    assert "\n" not in status


def test_discard_phrase_without_space() -> None:
    assert _status_from_body("请丢弃4个宝石") == "等待丢弃4个宝石"


def test_payment_phrase_becomes_compact_label() -> None:
    body = "请选择支付方式\n2白1金\n1白\n1金"
    assert _status_from_body(body) == "等待选择支付方式"


def test_game_over_marker_becomes_compact_label() -> None:
    body = "游戏结束\n座位1 15分\n再来一局"
    assert _status_from_body(body) in DEFAULT_GAME_OVER_MARKERS


def test_my_turn_and_waiting_take_priority() -> None:
    assert _status_from_body("座位1 3分 等待你操作") == MY_TURN_TEXT
    assert _status_from_body("等待玩家2操作 等待丢弃1个宝石") == "等待玩家2操作"


def test_bare_numbers_collapse_to_one_line() -> None:
    # The reported symptom: dozens of newline-separated numbers.
    body = "\n".join(str(n) for n in range(50))
    status = _status_from_body(body)
    assert "\n" not in status
    assert "  " not in status


def test_long_text_truncated() -> None:
    status = _status_from_body("字" * 500)
    assert len(status) <= 120
    assert status.endswith("...")


def test_short_text_returned_intact() -> None:
    assert _status_from_body("  座位1 3分 白 x2  ") == "座位1 3分 白 x2"


@pytest.mark.parametrize("bad", ["", "   \n  "])
def test_empty_body_degrades_to_empty_status(bad: str) -> None:
    assert not _status_from_body(bad)
