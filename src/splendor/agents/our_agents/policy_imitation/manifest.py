"""Versioned experiment manifests and explicit lifecycle gates.

Schema v1 remains readable for historical policy-imitation runs. Task-1
formal paths opt in to schema v2, whose immutable declaration is content
addressed and whose seed facts come directly from :mod:`splendor.seed_registry`.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from splendor.seed_registry import (
    LEGACY_MANIFEST_FORBIDDEN_RANGES,
    SEED_REGISTRY_SCHEMA_VERSION,
    registry_sha256,
    registry_snapshot,
    resolve_segment,
)

from .protocol import (
    RuntimeSnapshot,
    capture_code_revision,
    capture_dependency_provenance,
    require_reproducible_code,
    sha256_canonical_json,
)

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_V2 = 2
MANIFEST_V2_PROTOCOL = "splendor-experiment-manifest/2"
RNG_PROTOCOL_V1 = "splendor-rng-v1"
MANIFEST_STATUSES = frozenset(
    {"proposed", "approved", "running", "completed", "blocked"}
)
MANIFEST_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"approved"}),
    "approved": frozenset({"running"}),
    "running": frozenset({"completed", "blocked"}),
    "completed": frozenset(),
    "blocked": frozenset(),
}
# Compatibility alias only. Bounds are declared in seed_registry.py.
DEFAULT_FORBIDDEN_SEED_RANGES = LEGACY_MANIFEST_FORBIDDEN_RANGES
REQUIRED_SEED_GROUPS = ("training", "validation", "final_test")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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


def _validate_manifest_v1(manifest: Mapping[str, Any]) -> None:
    required = (
        "experiment_id",
        "phase",
        "purpose",
        "status",
        "seed_plan",
        "budget",
        "success_thresholds",
    )
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


def _require_mapping(
    container: Mapping[str, Any], name: str, *, nonempty: bool = True
) -> Mapping[str, Any]:
    value = container.get(name)
    if not isinstance(value, Mapping) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}mapping")
    return value


def _validate_seed_plan_v2(seed_plan: Mapping[str, Any]) -> None:
    segments = _require_mapping(seed_plan, "segments")
    missing_splits = sorted(set(REQUIRED_SEED_GROUPS) - set(segments))
    if missing_splits:
        raise ValueError(
            f"manifest v2 is missing required seed splits: {', '.join(missing_splits)}"
        )
    segment_names = [str(value) for value in segments.values()]
    if len(segment_names) != len(set(segment_names)):
        raise ValueError("manifest v2 logical splits must use distinct seed segments")
    resolved = {name: resolve_segment(str(value)) for name, value in segments.items()}
    for name in ("training", "validation"):
        if resolved[name].sealed:
            raise ValueError(f"manifest v2 {name} split cannot use a sealed segment")
    if not resolved["final_test"].sealed:
        raise ValueError("manifest v2 final_test split must use a sealed segment")
    for name, segment in resolved.items():
        if name != "final_test" and segment.sealed:
            raise ValueError(
                f"manifest v2 sealed segment {segment.name!r} is assigned to {name!r}"
            )
    if seed_plan.get("seats") != [0, 1]:
        raise ValueError("manifest v2 2p protocol requires seats [0, 1]")


def _validate_declaration_v2(declaration: Mapping[str, Any]) -> None:
    required = (
        "experiment_id",
        "phase",
        "purpose",
        "protocol_versions",
        "hypotheses",
        "estimands",
        "seed_plan",
        "budget",
        "decision_rule",
        "artifact_contract",
    )
    missing = [name for name in required if not declaration.get(name)]
    if missing:
        raise ValueError(f"manifest v2 declaration is missing {missing[0]}")
    protocol_versions = _require_mapping(declaration, "protocol_versions")
    if protocol_versions.get("rng") != RNG_PROTOCOL_V1:
        raise ValueError("manifest v2 must declare splendor-rng-v1")
    _validate_seed_plan_v2(_require_mapping(declaration, "seed_plan"))


def _validate_provenance_v2(provenance: Mapping[str, Any]) -> None:
    registry = _require_mapping(provenance, "seed_registry")
    if registry.get("schema") != SEED_REGISTRY_SCHEMA_VERSION:
        raise ValueError("manifest v2 seed-registry schema mismatch")
    snapshot = _require_mapping(registry, "snapshot")
    if registry.get("sha256") != sha256_canonical_json(snapshot):
        raise ValueError("manifest v2 seed-registry SHA-256 mismatch")
    dependencies = _require_mapping(provenance, "dependencies")
    files = _require_mapping(dependencies, "files")
    if dependencies.get("aggregate_sha256") != sha256_canonical_json(files):
        raise ValueError("manifest v2 dependency digest mismatch")
    _require_mapping(provenance, "code")
    _require_mapping(provenance, "runtime")
    _require_mapping(provenance, "baselines")


def _validate_lifecycle_event(  # noqa: PLR0913 - explicit hash-chain inputs
    event: Mapping[str, Any],
    *,
    initial: bool,
    expected_source: str,
    previous_event_sha256: str | None,
    declaration_sha256: object,
    provenance_sha256: object,
) -> tuple[str, str]:
    target = str(event.get("to"))
    if initial:
        if event.get("from") is not None or target != "proposed":
            raise ValueError("manifest v2 lifecycle must begin at proposed")
    elif (
        event.get("from") != expected_source
        or target not in MANIFEST_TRANSITIONS[expected_source]
    ):
        raise ValueError(
            f"invalid manifest transition {expected_source!r} -> {target!r}"
        )
    missing = [name for name in ("actor", "note", "at") if not event.get(name)]
    if missing:
        raise ValueError(f"manifest v2 lifecycle event is missing {missing[0]}")
    if event.get("declaration_sha256") != declaration_sha256:
        raise ValueError("manifest v2 lifecycle declaration binding mismatch")
    if event.get("provenance_sha256") != provenance_sha256:
        raise ValueError("manifest v2 lifecycle provenance binding mismatch")
    if event.get("previous_event_sha256") != previous_event_sha256:
        raise ValueError("manifest v2 lifecycle hash chain is discontinuous")
    event_body = {key: value for key, value in event.items() if key != "event_sha256"}
    event_sha256 = str(event.get("event_sha256"))
    if event_sha256 != sha256_canonical_json(event_body):
        raise ValueError("manifest v2 lifecycle event SHA-256 mismatch")
    return target, event_sha256


def _validate_lifecycle_v2(manifest: Mapping[str, Any]) -> None:
    status = manifest.get("status")
    if status not in MANIFEST_STATUSES:
        raise ValueError(f"invalid manifest status {status!r}")
    lifecycle = manifest.get("lifecycle")
    if not isinstance(lifecycle, list) or not lifecycle:
        raise ValueError("manifest v2 lifecycle must be a non-empty list")
    expected = "proposed"
    previous_event_sha256: str | None = None
    for index, raw_event in enumerate(lifecycle):
        if not isinstance(raw_event, Mapping):
            raise ValueError("manifest v2 lifecycle events must be mappings")
        expected, previous_event_sha256 = _validate_lifecycle_event(
            raw_event,
            initial=index == 0,
            expected_source=expected,
            previous_event_sha256=previous_event_sha256,
            declaration_sha256=manifest.get("declaration_sha256"),
            provenance_sha256=manifest.get("provenance_sha256"),
        )
    if status != expected:
        raise ValueError("manifest v2 status disagrees with lifecycle")


def _validate_manifest_v2(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != MANIFEST_V2_PROTOCOL:
        raise ValueError("manifest v2 has an invalid protocol identifier")
    declaration = _require_mapping(manifest, "declaration")
    declared_hash = manifest.get("declaration_sha256")
    if declared_hash != sha256_canonical_json(declaration):
        raise ValueError("manifest v2 declaration SHA-256 mismatch")
    _validate_declaration_v2(declaration)
    provenance = _require_mapping(manifest, "provenance")
    if manifest.get("provenance_sha256") != sha256_canonical_json(provenance):
        raise ValueError("manifest v2 provenance SHA-256 mismatch")
    _validate_provenance_v2(provenance)
    _validate_lifecycle_v2(manifest)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a historical v1 or opt-in task-1 v2 manifest."""
    version = manifest.get("schema_version")
    if version == MANIFEST_SCHEMA_VERSION:
        _validate_manifest_v1(manifest)
        return
    if version == MANIFEST_SCHEMA_V2:
        _validate_manifest_v2(manifest)
        return
    raise ValueError("unsupported manifest schema version")


def create_manifest(  # noqa: PLR0913 - legacy fields stay explicit
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
    """Create a schema-v1 manifest for legacy callers."""
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "phase": phase,
        "purpose": purpose,
        "status": status,
        "created_at": _now(),
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
    _write_json(path, manifest, exclusive=False)
    return manifest


def create_manifest_v2(  # noqa: PLR0913 - protocol fields are explicit
    path: Path,
    *,
    experiment_id: str,
    phase: str,
    purpose: str,
    seed_segments: Mapping[str, str],
    budget: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    estimands: Mapping[str, Any],
    decision_rule: Mapping[str, Any],
    artifact_contract: Mapping[str, Any],
    baselines: Mapping[str, Any],
    requested_device: str = "cpu",
    repo: Path | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Create a proposed, immutable-declaration task-1 manifest."""
    root = (repo or Path.cwd()).resolve()
    declaration: dict[str, Any] = {
        "experiment_id": experiment_id,
        "phase": phase,
        "purpose": purpose,
        "protocol_versions": {
            "manifest": MANIFEST_V2_PROTOCOL,
            "rng": RNG_PROTOCOL_V1,
            "scenario": "ScenarioV1",
        },
        "hypotheses": dict(hypotheses),
        "estimands": dict(estimands),
        "seed_plan": {
            "segments": dict(seed_segments),
            "seats": [0, 1],
            "scenario_identity": "canonical ScenarioV1 SHA-256",
        },
        "budget": dict(budget),
        "decision_rule": dict(decision_rule),
        "artifact_contract": dict(artifact_contract),
    }
    created_at = _now()
    provenance: dict[str, Any] = {
        "created_at": created_at,
        "code": capture_code_revision(root),
        "dependencies": capture_dependency_provenance(root),
        "runtime": RuntimeSnapshot.collect(requested_device).as_dict(),
        "seed_registry": {
            "schema": SEED_REGISTRY_SCHEMA_VERSION,
            "sha256": registry_sha256(),
            "snapshot": registry_snapshot(),
        },
        "baselines": dict(baselines),
        "notes": list(notes),
    }
    provenance_sha256 = sha256_canonical_json(provenance)
    initial_event: dict[str, Any] = {
        "from": None,
        "to": "proposed",
        "actor": "manifest-author",
        "note": "immutable declaration created",
        "at": created_at,
        "declaration_sha256": sha256_canonical_json(declaration),
        "provenance_sha256": provenance_sha256,
        "previous_event_sha256": None,
    }
    initial_event["event_sha256"] = sha256_canonical_json(initial_event)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_V2,
        "protocol": MANIFEST_V2_PROTOCOL,
        "declaration": declaration,
        "declaration_sha256": sha256_canonical_json(declaration),
        "provenance_sha256": provenance_sha256,
        "status": "proposed",
        "lifecycle": [initial_event],
        "provenance": provenance,
    }
    validate_manifest(manifest)
    _write_json(path, manifest, exclusive=True)
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a manifest before it is consumed."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")
    validate_manifest(manifest)
    return manifest


def transition_manifest(
    path: Path,
    target: str,
    *,
    actor: str,
    note: str,
    declaration_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply one legal v2 lifecycle transition without changing declaration."""
    if not actor.strip() or not note.strip():
        raise ValueError("manifest transition actor and note must not be empty")
    manifest = load_manifest(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_V2:
        raise ValueError("transition_manifest requires a schema-v2 manifest")
    current = str(manifest["status"])
    if target not in MANIFEST_TRANSITIONS[current]:
        raise ValueError(f"invalid manifest transition {current!r} -> {target!r}")
    if (
        declaration_sha256 is not None
        and declaration_sha256 != manifest["declaration_sha256"]
    ):
        raise ValueError("approved declaration SHA-256 does not match manifest")
    event: dict[str, Any] = {
        "from": current,
        "to": target,
        "actor": actor,
        "note": note,
        "at": _now(),
        "declaration_sha256": manifest["declaration_sha256"],
        "provenance_sha256": manifest["provenance_sha256"],
        "previous_event_sha256": manifest["lifecycle"][-1]["event_sha256"],
    }
    event["event_sha256"] = sha256_canonical_json(event)
    manifest["status"] = target
    manifest["lifecycle"].append(event)
    validate_manifest(manifest)
    _write_json(path, manifest, exclusive=False)
    return manifest


def approve_manifest(path: Path, reviewer: str, note: str) -> dict[str, Any]:
    """Record explicit human approval for v1 or v2 manifests."""
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    manifest = load_manifest(path)
    if manifest.get("schema_version") == MANIFEST_SCHEMA_V2:
        return transition_manifest(
            path,
            "approved",
            actor=reviewer,
            note=note,
            declaration_sha256=str(manifest["declaration_sha256"]),
        )
    if manifest["status"] != "proposed":
        raise ValueError(
            f"only proposed manifests can be approved, got {manifest['status']!r}"
        )
    manifest["status"] = "approved"
    manifest["review"] = {
        "reviewer": reviewer,
        "note": note,
        "approved_at": _now(),
    }
    validate_manifest(manifest)
    _write_json(path, manifest, exclusive=False)
    return manifest


def require_approved(manifest: Mapping[str, Any]) -> None:
    """Stop formal execution until the immutable declaration is approved."""
    validate_manifest(manifest)
    if manifest["status"] != "approved":
        raise RuntimeError(
            "manifest is not approved; review the declared budget/thresholds before running"
        )
    if manifest.get("schema_version") == MANIFEST_SCHEMA_V2:
        provenance = _require_mapping(manifest, "provenance")
        require_reproducible_code(_require_mapping(provenance, "code"))
