"""Task-1 tests for explicit, canonical opponent-pool probabilities."""

import pytest

from splendor.agents.generic.first_move import FirstActionAgent
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    OpponentPoolEntry,
    WeightedPoolSnapshot,
)


def _entry(name: str, weight: float) -> OpponentPoolEntry:
    return OpponentPoolEntry(
        name,
        CandidateSpec(name, "fixed_baseline", FirstActionAgent, snapshot=f"{name}.pth"),
        weight,
    )


def test_weighted_pool_normalizes_sorted_cdf_and_counts() -> None:
    entries = [_entry("rush", 2), _entry("ga", 1), _entry("heuristic", 2)]
    snapshot = WeightedPoolSnapshot.from_entries(entries)

    assert [item.name for item in snapshot.items] == ["ga", "heuristic", "rush"]
    assert [item.probability for item in snapshot.items] == pytest.approx(
        [0.2, 0.4, 0.4]
    )
    assert [item.cumulative_probability for item in snapshot.items] == pytest.approx(
        [0.2, 0.6, 1.0]
    )
    assert snapshot.select(entries, 0.0).name == "ga"
    assert snapshot.select(entries, 0.2).name == "heuristic"
    assert snapshot.select(entries, 0.999999).name == "rush"
    report = snapshot.as_dict(["rush", "rush", "ga"])
    assert report["actual_counts"] == {"ga": 1, "heuristic": 0, "rush": 2}
    assert report["draws"] == 3


def test_weighted_pool_snapshot_is_input_order_invariant() -> None:
    entries = [_entry("b", 3), _entry("a", 1)]
    left = WeightedPoolSnapshot.from_entries(entries)
    right = WeightedPoolSnapshot.from_entries(tuple(reversed(entries)))
    assert left == right


def test_history_is_one_explicit_bucket_mass() -> None:
    entries = [
        _entry("current", 1),
        _entry("history-1", 0.5),
        _entry("history-2", 0.5),
        _entry("heuristic", 2),
        _entry("rush", 2),
        _entry("ga", 1),
        _entry("minimax", 1),
    ]
    snapshot = WeightedPoolSnapshot.from_entries(entries)
    probabilities = {item.name: item.probability for item in snapshot.items}
    assert probabilities["heuristic"] == 0.25
    assert probabilities["rush"] == 0.25
    assert probabilities["ga"] == 0.125
    assert probabilities["minimax"] == 0.125
    assert probabilities["current"] == 0.125
    assert probabilities["history-1"] + probabilities["history-2"] == 0.125


def test_weighted_pool_rejects_duplicate_or_zero_mass() -> None:
    with pytest.raises(ValueError, match="unique"):
        WeightedPoolSnapshot.from_entries([_entry("same", 1), _entry("same", 2)])
    with pytest.raises(ValueError, match="positive mass"):
        WeightedPoolSnapshot.from_entries([_entry("zero", 0)])


def test_weighted_pool_rejects_stale_weights_or_snapshot_identity() -> None:
    entries = [_entry("a", 1), _entry("b", 2)]
    snapshot = WeightedPoolSnapshot.from_entries(entries)
    with pytest.raises(ValueError, match="do not match"):
        snapshot.select([_entry("a", 2), _entry("b", 1)], 0.3)
    changed_identity = [
        _entry("a", 1),
        OpponentPoolEntry(
            "b",
            CandidateSpec(
                "b", "fixed_baseline", FirstActionAgent, snapshot="changed.pth"
            ),
            2,
        ),
    ]
    with pytest.raises(ValueError, match="do not match"):
        snapshot.select(changed_identity, 0.3)
