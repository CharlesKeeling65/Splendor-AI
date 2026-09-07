"""Declared experiment manifests and execution gates for imitation learning."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .protocol import RuntimeSnapshot, capture_code_revision

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_STATUSES = frozenset(
    {"proposed", "approved", "running", "completed", "blocked"}
)
DEFAULT_FORBIDDEN_SEED_RANGES: tuple[tuple[int, int], ...] = (
    (910000, 910010),
    (920000, 920050),
    (930000, 930025),
)
REQUIRED_SEED_GROUPS = ("training", "validation", "final_test")


def _normalise_seeds(name: str, values: Sequence[int]) -> list[int]:
    """Validate and normalize one named seed group."""
    result = [int(value) for value in values]
    if any(value < 0 for value in result):
        raise ValueError(f"{name} contains a negative seed")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate seeds")
    return result


def _validate_seed_groups(
    seed_groups: Mapping[str, Sequence[int]],
    forbidden_ranges: Sequence[tuple[int, int]],
) -> dict[str, list[int]]:
    """Ensure train/validation/test groups cannot leak into one another."""
    missing = [name for name in REQUIRED_SEED_GROUPS if name not in seed_groups]
    if missing:
        raise ValueError(f"missing required seed groups: {', '.join(missing)}")
    normalized = {
        name: _normalise_seeds(name, values) for name, values in seed_groups.items()
    }
    ownership: dict[int, str] = {}
    for name, seeds in normalized.items():
        for seed in seeds:
            previous = ownership.setdefault(seed, name)
            if previous != name:
                raise ValueError(f"seed {seed} leaks between {previous} and {name}")
            if any(start <= seed < end for start, end in forbidden_ranges):
                raise ValueError(f"seed {seed} is reserved historical test data")
    return normalized


def validate_manifest(manifest: Mapping[str, Any]) -> None:  # noqa: C901 - protocol checks are intentionally explicit
    """Validate all protocol fields needed before a run may start."""
    required = (
        "experiment_id",
        "phase",
        "purpose",
        "status",
        "seed_plan",
        "budget",
        "success_thresholds",
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest is missing {key}")
    if manifest["status"] not in MANIFEST_STATUSES:
        raise ValueError(f"invalid manifest status {manifest['status']!r}")
    seed_plan = manifest["seed_plan"]
    if not isinstance(seed_plan, Mapping):
        raise ValueError("seed_plan must be a mapping")
    forbidden_ranges = [
        tuple(pair)
        for pair in seed_plan.get("forbidden_ranges", DEFAULT_FORBIDDEN_SEED_RANGES)
    ]
    seeds = _validate_seed_groups(seed_plan.get("groups", {}), forbidden_ranges)
    seats = [int(seat) for seat in seed_plan.get("seats", [])]
    if seats != [0, 1]:
        raise ValueError("the first implementation requires both seats [0, 1]")
    if any(start >= end for start, end in forbidden_ranges):
        raise ValueError("forbidden seed ranges must be non-empty half-open intervals")
    if not isinstance(manifest["budget"], Mapping) or not manifest["budget"]:
        raise ValueError("manifest budget must be a non-empty mapping")
    if (
        not isinstance(manifest["success_thresholds"], Mapping)
        or not manifest["success_thresholds"]
    ):
        raise ValueError("manifest success_thresholds must be a non-empty mapping")
    if seed_plan.get("groups") != seeds:
        raise ValueError("seed groups must contain unique integer lists")


def create_manifest(  # noqa: PLR0913 - manifest fields must be explicit at the call site
    path: Path,
    *,
    experiment_id: str,
    phase: str,
    purpose: str,
    seed_groups: Mapping[str, Sequence[int]],
    budget: Mapping[str, Any],
    success_thresholds: Mapping[str, Any],
    seats: Sequence[int] = (0, 1),
    forbidden_ranges: Sequence[tuple[int, int]] = DEFAULT_FORBIDDEN_SEED_RANGES,
    requested_device: str = "cpu",
    repo: Path | None = None,
    status: str = "proposed",
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Create and persist a declared manifest; formal commands must approve it."""
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "phase": phase,
        "purpose": purpose,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed_plan": {
            "groups": {name: list(values) for name, values in seed_groups.items()},
            "seats": list(seats),
            "forbidden_ranges": [list(pair) for pair in forbidden_ranges],
            "rng_protocol": "random.seed + numpy.random.seed + torch.manual_seed",
            "seat_mapping": "one game per (deal_seed, seat); raw engine IDs 0 and 1",
        },
        "budget": dict(budget),
        "success_thresholds": dict(success_thresholds),
        "runtime": RuntimeSnapshot.collect(requested_device).as_dict(),
        "code": capture_code_revision(repo),
        "notes": list(notes),
    }
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a manifest before it is consumed by an experiment."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")
    validate_manifest(manifest)
    return manifest


def approve_manifest(path: Path, reviewer: str, note: str) -> dict[str, Any]:
    """Record explicit human approval without changing the declared protocol."""
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    manifest = load_manifest(path)
    if manifest["status"] != "proposed":
        raise ValueError(
            f"only proposed manifests can be approved, got {manifest['status']!r}"
        )
    manifest["status"] = "approved"
    manifest["review"] = {
        "reviewer": reviewer,
        "note": note,
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    validate_manifest(manifest)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def require_approved(manifest: Mapping[str, Any]) -> None:
    """Stop formal execution until a human has reviewed the manifest."""
    validate_manifest(manifest)
    if manifest["status"] != "approved":
        raise RuntimeError(
            "manifest is not approved; review the declared budget/thresholds before running"
        )
