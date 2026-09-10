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
    from splendor.remote.rollout import _unseen_cards  # noqa: PLC2701

    unseen = _unseen_cards(snapshot)
    visible_keys = {
        (info["tier"], info["colour"], info["points"], tuple(sorted(info["cost"].items())))
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


def test_two_player_smoke(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    result = estimator.estimate(
        snapshot, actor_seat=snapshot["my_seat"] or 1,
        n_rollouts=2, max_steps=400, seed=11,
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


def test_rejects_non_two_player_boards(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    three = dict(snapshot)
    three["panels"] = list(snapshot["panels"]) * 2  # synthetic 4 panels
    with pytest.raises(ValueError, match="seats"):
        estimator.estimate(three, actor_seat=1, n_rollouts=1, max_steps=5)


def test_actor_seat_mismatch_fails_loudly(
    snapshot: Snapshot, estimator: WinRateEstimator
) -> None:
    with pytest.raises(ValueError):
        estimator.estimate(
            snapshot, actor_seat=9, n_rollouts=1, max_steps=5, seed=1
        )
