"""Tests for the policy-imitation run protocol."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest,
    load_manifest,
    require_approved,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    isolated_seed,
    seed_everything,
)


def _groups() -> dict[str, list[int]]:
    return {"training": [42], "validation": [700001], "final_test": [800001]}


def test_seed_everything_and_isolation() -> None:
    seed_everything(7)
    expected = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(19)
    with isolated_seed(7):
        actual = (random.random(), np.random.random(), torch.rand(1).item())
    assert actual == pytest.approx(expected)
    after = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(19)
    assert after == pytest.approx(
        (random.random(), np.random.random(), torch.rand(1).item())
    )


def test_manifest_requires_disjoint_seed_groups(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="leaks"):
        create_manifest(
            path,
            experiment_id="test",
            phase="3.0",
            purpose="smoke",
            seed_groups={
                "training": [42],
                "validation": [42],
                "final_test": [800001],
            },
            budget={"games": 1},
            success_thresholds={"max_failures": 0},
        )


def test_manifest_rejects_reserved_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        create_manifest(
            tmp_path / "manifest.json",
            experiment_id="test",
            phase="3.0",
            purpose="formal",
            seed_groups={
                "training": [42],
                "validation": [910000],
                "final_test": [800001],
            },
            budget={"games": 1},
            success_thresholds={"max_failures": 0},
        )


def test_manifest_approval_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    create_manifest(
        path,
        experiment_id="test",
        phase="3.0",
        purpose="formal",
        seed_groups=_groups(),
        budget={"games": 1},
        success_thresholds={"max_failures": 0},
    )
    manifest = load_manifest(path)
    with pytest.raises(RuntimeError, match="not approved"):
        require_approved(manifest)
    approved = approve_manifest(path, "reviewer", "budget and seed groups checked")
    require_approved(approved)
    assert load_manifest(path)["review"]["reviewer"] == "reviewer"


def test_manifest_requires_both_seats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both seats"):
        create_manifest(
            tmp_path / "manifest.json",
            experiment_id="test",
            phase="3.0",
            purpose="smoke",
            seed_groups=_groups(),
            seats=(0,),
            budget={"games": 1},
            success_thresholds={"max_failures": 0},
        )
