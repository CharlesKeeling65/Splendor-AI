"""CLI current-bucket weights must survive argument parsing."""

import pytest

from splendor.agents.our_agents.policy_imitation.cli import _current_pool_weight


@pytest.mark.parametrize(
    ("pool", "expected"),
    [
        ("ga:1,current:0", 0.0),
        ("ga:1,current:2.5", 2.5),
        ("ga:1,current", 1.0),
        ("ga:1", 0.0),
    ],
)
def test_current_weight(pool: str, expected: float) -> None:
    assert _current_pool_weight(pool) == expected


def test_duplicate_current_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        _current_pool_weight("current:1,current:2")
