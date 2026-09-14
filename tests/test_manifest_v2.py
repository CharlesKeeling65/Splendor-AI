"""Manifest-v2 safety gates for task-1 formal experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest_v2,
    load_manifest,
    require_approved,
    transition_manifest,
)
from splendor.seed_registry import registry_sha256

REPO = Path(__file__).resolve().parents[1]


def _create(path: Path) -> dict[str, object]:
    return create_manifest_v2(
        path,
        experiment_id="task1-test",
        phase="T1.0",
        purpose="protocol smoke test",
        seed_segments={
            "training": "training",
            "validation": "validation",
            "final_test": "independent_test",
        },
        budget={"fixed_n": 50, "replicates": 3},
        hypotheses={"H-pool": "targeted pool improves target styles"},
        estimands={"primary": "paired scenario score-rate difference"},
        decision_rule={"selection": "pre-registered lexicographic rule"},
        artifact_contract={"episodes": "episodes.jsonl.zst"},
        baselines={"ppo-best": {"sha256": "a" * 64}},
        repo=REPO,
    )


def test_v2_declaration_is_hashed_and_creation_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _create(path)

    assert manifest["schema_version"] == 2
    assert manifest["status"] == "proposed"
    assert manifest["provenance"]["seed_registry"]["sha256"] == registry_sha256()  # type: ignore[index]
    assert len(str(manifest["declaration_sha256"])) == 64
    with pytest.raises(FileExistsError):
        _create(path)


def test_v2_rejects_declaration_or_registry_tampering(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _create(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["declaration"]["budget"]["fixed_n"] = 51
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declaration SHA-256"):
        load_manifest(path)

    path.unlink()
    payload = _create(path)
    payload["provenance"]["seed_registry"]["sha256"] = "0" * 64  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance SHA-256"):
        load_manifest(path)


def test_v2_lifecycle_is_strict_and_requires_human_approval(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    proposed = _create(path)
    with pytest.raises(RuntimeError, match="not approved"):
        require_approved(proposed)
    with pytest.raises(ValueError, match="invalid manifest transition"):
        transition_manifest(path, "running", actor="runner", note="skip approval")

    approved = approve_manifest(path, "reviewer", "declaration checked")
    require_approved(approved)
    running = transition_manifest(path, "running", actor="runner", note="start")
    assert running["status"] == "running"
    completed = transition_manifest(path, "completed", actor="runner", note="done")
    assert completed["status"] == "completed"
    assert [event["to"] for event in completed["lifecycle"]] == [
        "proposed",
        "approved",
        "running",
        "completed",
    ]
    with pytest.raises(ValueError, match="invalid manifest transition"):
        transition_manifest(path, "running", actor="runner", note="restart")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("code", "commit", "forged"),
        ("runtime", "python_version", "0.0"),
        ("baselines", "ppo-best", {"sha256": "0" * 64}),
    ],
)
def test_v2_provenance_tampering_is_detected(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    path = tmp_path / f"{section}.json"
    _create(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"][section][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance SHA-256"):
        load_manifest(path)


def test_v2_lifecycle_actor_and_dependency_rehash_tampering_are_detected(
    tmp_path: Path,
) -> None:
    lifecycle_path = tmp_path / "lifecycle.json"
    _create(lifecycle_path)
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["lifecycle"][0]["actor"] = "forged"
    lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle event SHA-256"):
        load_manifest(lifecycle_path)

    dependency_path = tmp_path / "dependencies.json"
    _create(dependency_path)
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    files = dependency["provenance"]["dependencies"]["files"]
    files["uv.lock"] = "0" * 64
    dependency["provenance"]["dependencies"]["aggregate_sha256"] = "0" * 64
    dependency_path.write_text(json.dumps(dependency), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance SHA-256"):
        load_manifest(dependency_path)


def test_v2_requires_distinct_registered_segments(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="distinct seed segments"):
        create_manifest_v2(
            path,
            experiment_id="bad",
            phase="T1.0",
            purpose="test",
            seed_segments={
                "training": "training",
                "validation": "training",
                "final_test": "independent_test",
            },
            budget={"games": 1},
            hypotheses={"H": "x"},
            estimands={"d": "x"},
            decision_rule={"select": "x"},
            artifact_contract={"episodes": "x"},
            baselines={"ppo-best": {"sha256": "a" * 64}},
            repo=REPO,
        )


def test_v2_enforces_split_sealing_roles(tmp_path: Path) -> None:
    common: dict[str, Any] = {
        "experiment_id": "bad",
        "phase": "T1.0",
        "purpose": "test",
        "budget": {"games": 1},
        "hypotheses": {"H": "x"},
        "estimands": {"d": "x"},
        "decision_rule": {"select": "x"},
        "artifact_contract": {"episodes": "x"},
        "baselines": {"ppo-best": {"sha256": "a" * 64}},
        "repo": REPO,
    }
    with pytest.raises(ValueError, match="missing required seed splits"):
        create_manifest_v2(
            tmp_path / "missing.json",
            seed_segments={"training": "training", "validation": "validation"},
            **common,
        )
    with pytest.raises(ValueError, match="training split cannot use a sealed"):
        create_manifest_v2(
            tmp_path / "sealed-training.json",
            seed_segments={
                "training": "independent_test",
                "validation": "validation",
                "final_test": "c2_test",
            },
            **common,
        )
    with pytest.raises(ValueError, match="final_test split must use a sealed"):
        create_manifest_v2(
            tmp_path / "open-test.json",
            seed_segments={
                "training": "training",
                "validation": "validation",
                "final_test": "ci_smoke",
            },
            **common,
        )
