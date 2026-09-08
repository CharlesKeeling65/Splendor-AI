"""Unit tests for the play_vs_humans helpers (pure logic, offline)."""

from splendor.play_vs_humans import (
    _card_cn,
    _gems_cn,
    _judge,
    describe_action,
)


class TestDescribeAction:
    def test_collect_diff(self) -> None:
        action = {"type": "collect_diff", "collected_gems": {"blue": 1, "red": 2}}
        assert describe_action(action) == "拿取宝石 蓝x1/红x2"

    def test_reserve_with_yellow(self) -> None:
        card = _FakeCard(tier=1, colour="green", points=2)
        action = {
            "type": "reserve",
            "card": card,
            "collected_gems": {"yellow": 1},
        }
        text = describe_action(action)
        assert "预留" in text
        assert "2级绿卡2分" in text
        assert "+1金" in text

    def test_buy_available_with_payment_and_noble(self) -> None:
        card = _FakeCard(tier=0, colour="red", points=0)
        action = {
            "type": "buy_available",
            "card": card,
            "returned_gems": {"white": 2, "blue": 1},
            "noble": {"white": 3},
        }
        text = describe_action(action)
        assert "购买桌面" in text
        assert "1级红卡" in text
        assert "白x2/蓝x1" in text
        assert "贵族" in text

    def test_buy_reserve_has_no_payment_text(self) -> None:
        card = _FakeCard(tier=-1, colour="blue", points=3)
        action = {"type": "buy_reserve", "card": card}
        text = describe_action(action)
        assert "购买预留" in text
        assert "支付" not in text


class TestGemsAndCard:
    def test_zero_counts_dropped(self) -> None:
        assert _gems_cn({"blue": 1, "red": 0}) == "蓝x1"

    def test_none_and_empty(self) -> None:
        assert _gems_cn(None) == ""
        assert _gems_cn({}) == ""

    def test_card_without_points(self) -> None:
        card = _FakeCard(tier=2, colour="black", points=0)
        assert _card_cn(card) == "3级黑卡"


class TestJudge:
    def test_win(self) -> None:
        result, ranking = _judge({1: 10, 4: 15}, my_seat=4)
        assert result == "win"
        assert ranking == [(4, 15), (1, 10)]

    def test_loss(self) -> None:
        result, _ = _judge({1: 16, 4: 15}, my_seat=4)
        assert result == "loss"

    def test_draw(self) -> None:
        result, _ = _judge({1: 15, 4: 15}, my_seat=4)
        assert result == "draw"

    def test_no_data(self) -> None:
        result, ranking = _judge({}, my_seat=4)
        assert result == "unknown"
        assert ranking == []


class _FakeCard:
    """Minimal stand-in for the engine Card (attribute shape only)."""

    def __init__(self, tier: int, colour: str, points: int) -> None:
        self.deck_id = tier
        self.colour = colour
        self.points = points
