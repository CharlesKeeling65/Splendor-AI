"""
Offline tests for the advisor's v0 console renderer (plan phase-7 T7.5):
AdvisorCli over fixture snapshots with a collecting emit - advice blocks,
sub-flow pauses, standby announcements and print dedupe.
"""

from pathlib import Path

from splendor.browser.advisor.observer import AdvisorFrame, classify_phase
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import MockBrowserDriver
from splendor.play_advisor import AdvisorCli

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"
OPENING = (FIXTURES / "opening.html").read_text(encoding="utf-8")
PAYMENT = (FIXTURES / "payment_pills.html").read_text(encoding="utf-8")
GAME_OVER = (FIXTURES / "game_over.html").read_text(encoding="utf-8")


def _extract(page: str) -> Snapshot:
    driver = MockBrowserDriver()
    driver.set_html(page)
    return extract_snapshot(driver)


def _frame(page: str, seq: int = 1) -> AdvisorFrame:
    snapshot = _extract(page)
    my_seat = snapshot["my_seat"]
    return AdvisorFrame(
        snapshot=snapshot,
        phase=classify_phase(snapshot),
        my_index=(my_seat - 1) if my_seat > 0 else None,
        waiting_seat=None,
        frame_seq=seq,
    )


def _cli(lines: list[str], *, depth: int = 0) -> AdvisorCli:
    return AdvisorCli(depth=depth, top_k=5, emit=lines.append)


def test_my_turn_block_contains_all_sections() -> None:
    lines: list[str] = []
    _cli(lines).on_frame(_frame(OPENING))
    text = "\n".join(lines)
    assert "我的回合" in text
    assert "建议（GA 快评" in text
    assert "★" in text and "☆" in text  # 最优 / 次优 marks
    assert "牌堆剩余" in text
    assert "T1(堆余 36)" in text  # opening fixture: 40 - 4 dealt
    assert "可负担" in text
    assert "预留记忆" in text


def test_deep_mode_adds_ranking_when_enabled() -> None:
    lines_off: list[str] = []
    lines_on: list[str] = []
    _cli(lines_off).on_frame(_frame(OPENING))
    _cli(lines_on, depth=2).on_frame(_frame(OPENING))
    assert "深算（minimax depth=2）" not in "\n".join(lines_off)
    assert "深算（minimax depth=2）" in "\n".join(lines_on)


def test_identical_frame_is_not_printed_twice() -> None:
    lines: list[str] = []
    cli = _cli(lines)
    cli.on_frame(_frame(OPENING, seq=1))
    count = len(lines)
    cli.on_frame(_frame(OPENING, seq=2))  # same board: dedupe key matches
    assert len(lines) == count


def test_payment_subflow_pauses_advice() -> None:
    # Inject the payment pills on the 2-panel opening page: the real
    # payment fixture is a single-panel close-up, not an in-game view.
    snapshot = _extract(OPENING)
    snapshot["payment_options"] = ["2白1金"]
    frame = AdvisorFrame(
        snapshot=snapshot,
        phase=classify_phase(snapshot),
        my_index=0,
        waiting_seat=None,
        frame_seq=1,
    )
    lines: list[str] = []
    _cli(lines).on_frame(frame)
    text = "\n".join(lines)
    assert "建议（GA 快评" not in text
    assert "子流程需要手动点击" in text


def test_single_panel_page_is_ignored_quietly() -> None:
    lines: list[str] = []
    _cli(lines).on_frame(_frame(PAYMENT))  # close-up fixture: 1 panel
    assert lines == []


def test_unseated_frame_skips_advice() -> None:
    page = GAME_OVER.replace("我", "", 1)
    # Spectator view has no board at all -> standby branch, not advice.
    lines: list[str] = []
    _cli(lines).on_frame(_frame(page))
    text = "\n".join(lines)
    assert "建议（GA 快评" not in text


def test_standby_announced_once() -> None:
    lines: list[str] = []
    cli = _cli(lines)
    cli.on_frame(_frame(GAME_OVER, seq=1))
    cli.on_frame(_frame(GAME_OVER, seq=2))
    standby = [line for line in lines if "待机" in line]
    assert len(standby) == 1
