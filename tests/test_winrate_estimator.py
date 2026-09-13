"""
Offline win-rate estimator tests (phase-6): fixture snapshot -> engine state
reconstruction -> batched Monte-Carlo rollouts, entirely offline (random
initial weights - the estimator is deployment plumbing, not a policy test).
"""

from pathlib import Path

import pytest

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import MockBrowserDriver
from splendor.remote.rollout import WinRateEstimator

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"


def _snapshot_of(fixture: str) -> Snapshot:
    driver = MockBrowserDriver()
    driver.set_html((FIXTURES / fixture).read_text(encoding="utf-8"))
    return extract_snapshot(driver)


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    return _snapshot_of("opening.html")


@pytest.fixture(scope="module")
def estimator() -> WinRateEstimator:
    model = QNetwork(input_dim=265, output_dim=3510, feature_version="v1")
    model.eval()
    return WinRateEstimator(model)


def test_unseen_cards_exclude_visible_faces(snapshot: Snapshot) -> None:
    from splendor.remote.rollout import _unseen_cards

    unseen = _unseen_cards(snapshot)
    visible_keys = {
        (
            info["tier"],
            info["colour"],
            info["points"],
            tuple(sorted(info["cost"].items())),
        )
        for row in snapshot["dealt"]
        for info in row
        if info is not None
    }
    assert all(
        (card.deck_id, card.colour, card.points, tuple(sorted(card.cost.items())))
        not in visible_keys
        for card in unseen
    )
    # 90 total minus the dealt faces visible in the fixture.
    assert len(unseen) < 90


def test_two_player_smoke(snapshot: Snapshot, estimator: WinRateEstimator) -> None:
    result = estimator.estimate(
        snapshot,
        actor_seat=snapshot["my_seat"] or 1,
        n_rollouts=2,
        max_steps=400,
        seed=11,
    )
    assert len(result.win_rates) == 2
    assert all(0.0 <= rate <= 1.0 for rate in result.win_rates)
    assert 0.0 <= result.draw_rate <= 1.0
    # Every finished rollout distributes exactly one win credit (draws
    # split it), aborted rollouts distribute none.
    assert sum(result.win_rates) == pytest.approx(
        (result.rollouts - result.aborted) / result.rollouts
    )
    assert result.elapsed_s > 0.0


def test_deterministic_under_seed(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    first = estimator.estimate(
        snapshot, actor_seat=1, n_rollouts=2, max_steps=400, seed=42
    )
    second = estimator.estimate(
        snapshot, actor_seat=1, n_rollouts=2, max_steps=400, seed=42
    )
    assert first.win_rates == second.win_rates
    assert first.draw_rate == second.draw_rate


def _board_with_seats(snapshot: Snapshot, seats: int) -> Snapshot:
    """Synthetic board: cycle the fixture's seat panels up to ``seats`` seats."""
    base = list(snapshot["panels"])
    panels = [base[index % len(base)] for index in range(seats)]
    return {**snapshot, "panels": panels}


def test_accepts_two_to_four_player_boards(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    """
    v1's metric block reserves MAX_RIVALS (=3) rival-score slots on each side
    of the observer, so 2-4 seats all vectorize; the public-v2 schema is the
    one that encodes exactly two.

    The previous check hard-required exactly 2 seats, so a 3-player self-play
    run aborted with "supports 2 seats, got 3" and the dashboard never got a
    win-rate point at all. Values above 2 seats are an out-of-distribution
    proxy (local DQN training is 2-player) but structurally sound.
    """
    for seats in (3, 4):
        result = estimator.estimate(
            _board_with_seats(snapshot, seats),
            actor_seat=1,
            n_rollouts=1,
            max_steps=400,
            seed=7,
        )
        assert len(result.win_rates) == seats


def test_rejects_boards_beyond_the_rival_block(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    # 5 seats: the observer's rival-score block runs out at MAX_RIVALS + 1 = 4
    with pytest.raises(ValueError, match=r"supports 2\.\.4 seats"):
        estimator.estimate(
            _board_with_seats(snapshot, 5), actor_seat=1, n_rollouts=1, max_steps=5
        )


def test_actor_seat_mismatch_fails_loudly(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    with pytest.raises(ValueError):
        estimator.estimate(snapshot, actor_seat=9, n_rollouts=1, max_steps=5, seed=1)


@pytest.fixture(scope="module")
def multi_estimator() -> WinRateEstimator:
    model = QNetwork(input_dim=337, output_dim=3510, feature_version="public-v2-multi")
    model.eval()
    return WinRateEstimator(model)


@pytest.mark.parametrize("seats", [2, 3, 4])
def test_public_v2_multi_estimator_accepts_two_to_four_seats(
    snapshot: Snapshot, multi_estimator: WinRateEstimator, seats: int
) -> None:
    """Roadmap E4: the multi-seat schema unlocks 2..4-seat estimation."""
    result = multi_estimator.estimate(
        _board_with_seats(snapshot, seats),
        actor_seat=1,
        n_rollouts=1,
        max_steps=400,
        seed=11,
    )
    assert len(result.win_rates) == seats


def test_legacy_public_v2_estimator_still_rejects_three_seats(
    snapshot: Snapshot,
) -> None:
    model = QNetwork(input_dim=312, output_dim=3510, feature_version="public-v2")
    model.eval()
    legacy = WinRateEstimator(model)
    with pytest.raises(ValueError, match="exactly 2 seats"):
        legacy.estimate(
            _board_with_seats(snapshot, 3), actor_seat=1, n_rollouts=1, max_steps=5
        )
