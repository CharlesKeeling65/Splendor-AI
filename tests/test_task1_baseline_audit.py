"""On-disk baseline hashes and historical denominator evidence for T1.0."""

from __future__ import annotations

from pathlib import Path

from splendor.agents.our_agents.policy_imitation.baseline_audit import (
    EXPECTED_OFFICIAL_SHA256,
    EXPECTED_PARENT_BC_SHA256,
    EXPECTED_PARENT_DATA_HASH,
    build_baseline_audit,
)

REPO = Path(__file__).resolve().parents[1]


def test_frozen_task1_baseline_and_parent_lineage_match_disk() -> None:
    audit = build_baseline_audit(REPO)
    official = audit["official"]

    assert official["checkpoint_sha256"] == EXPECTED_OFFICIAL_SHA256
    assert official["parent_bc"]["sha256"] == EXPECTED_PARENT_BC_SHA256
    assert official["parent_dataset"]["content_sha256"] == EXPECTED_PARENT_DATA_HASH
    assert official["contract"] == {
        "feature_version": "public-v2",
        "input_dim": 312,
        "action_dim": 3510,
        "hidden_layers": [128, 128, 128, 128],
        "inference": "eval + masked greedy argmax",
        "training_action": "masked categorical (legacy global torch RNG)",
        "reward": "terminal +/-10/0 + nonzero-terminal potential kappa=0.05",
    }
    assert set(audit["opponents"]) == {
        "random",
        "heuristic",
        "heuristic-rush",
        "ga",
        "minimax",
    }


def test_c4r2_audit_exposes_wrapped_cluster_denominators() -> None:
    audit = build_baseline_audit(REPO)["historical_c4r2"]

    assert audit["scheduled_rows_total"] == 3150
    assert audit["declared_seed_count"] == 3150
    assert audit["unique_source_seeds_total"] == 50
    assert audit["error_rows_total"] == 0
    for opponent in audit["matchups"].values():
        assert opponent["scheduled_rows"] == 150
        assert opponent["unique_source_seeds"] == 50
        assert opponent["unique_seed_seat_cells"] == 100
        assert opponent["duplicated_cell_count"] == 50
        assert opponent["rows_in_duplicated_cells"] == 100
        assert opponent["duplicate_excess_rows"] == 50
