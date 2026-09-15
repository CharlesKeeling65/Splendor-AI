"""Versioned experiment manifests and explicit lifecycle gates.

Schema v1 remains readable for historical policy-imitation runs. Task-1
formal paths opt in to schema v2, whose immutable declaration is content
addressed and whose seed facts come directly from :mod:`splendor.seed_registry`.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from splendor.seed_registry import (
    LEGACY_MANIFEST_FORBIDDEN_RANGES,
    SEED_REGISTRY_SCHEMA_VERSION,
    TASK1_SCENARIO_SPLITS,
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
SHA256_HEX_LENGTH = 64


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _is_utc_timestamp(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)


def _real_manifest_path(path: Path, *, create_parent: bool) -> Path:
    """Canonicalize a manifest path while refusing symlink traversal."""
    lexical = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    parent = lexical.parent
    if parent.resolve(strict=False) != parent:
        raise ValueError("manifest parent path traverses a symlink")
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_metadata = os.lstat(parent)
    except OSError as exc:
        raise ValueError(f"cannot access manifest parent directory: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise ValueError("manifest parent path is not a real directory")
    if parent.resolve(strict=True) != parent:
        raise ValueError("manifest parent path traverses a symlink")
    if os.path.lexists(lexical):
        try:
            metadata = os.lstat(lexical)
        except OSError as exc:
            raise ValueError(f"cannot inspect manifest path: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("manifest path is not a real regular file")
        if lexical.resolve(strict=True) != lexical:
            raise ValueError("manifest path traverses a symlink")
    return lexical


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path = _real_manifest_path(path, create_parent=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    """Serialize lifecycle read-modify-write operations across processes."""
    path = _real_manifest_path(path, create_parent=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"cannot open manifest lifecycle lock: {exc}") from exc
    with os.fdopen(descriptor, "a+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("manifest lifecycle lock is not a regular file")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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


def _validate_seed_plan_v2(  # noqa: C901,PLR0912 - all split roles fail closed
    seed_plan: Mapping[str, Any],
) -> None:
    raw_seed_roll = seed_plan.get("seed_roll")
    if raw_seed_roll is not None:
        from .seed_roll import (  # noqa: PLC0415 - avoid manifest import cycle
            SeedRollError,
            validate_seed_roll_binding,
        )

        try:
            validate_seed_roll_binding(raw_seed_roll)
        except SeedRollError as exc:
            raise ValueError(f"manifest v2 seed roll is invalid: {exc}") from exc
    segments = _require_mapping(seed_plan, "segments")
    task1_names = set(TASK1_SCENARIO_SPLITS)
    if set(segments) & task1_names:
        missing_task1 = sorted(task1_names - set(segments))
        if missing_task1:
            raise ValueError(
                "manifest v2 is missing Task-1 scenario splits: "
                f"{', '.join(missing_task1)}"
            )
        unexpected = sorted(set(segments) - task1_names)
        if unexpected:
            raise ValueError(
                "Task-1 scenario seed plan has unexpected splits: "
                f"{', '.join(unexpected)}"
            )
        for logical_name, expected_segment in TASK1_SCENARIO_SPLITS.items():
            declared = resolve_segment(str(segments[logical_name]))
            if declared != expected_segment:
                raise ValueError(
                    f"Task-1 split {logical_name!r} must use {expected_segment.name!r}"
                )
        if seed_plan.get("seats") != [0, 1]:
            raise ValueError("manifest v2 2p protocol requires seats [0, 1]")
        return
    if raw_seed_roll is not None:
        raise ValueError("manifest v2 seed roll requires Task-1 scenario splits")
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


def _validate_declaration_v2(  # noqa: C901,PLR0912,PLR0915 - fail closed
    declaration: Mapping[str, Any],
) -> None:
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
    if protocol_versions.get("scenario") not in {
        "ScenarioV1",  # T1.0 manifest compatibility alias
        "splendor-scenario/1",
    }:
        raise ValueError("manifest v2 must declare ScenarioV1")
    seed_plan = _require_mapping(declaration, "seed_plan")
    _validate_seed_plan_v2(seed_plan)
    has_seed_roll = seed_plan.get("seed_roll") is not None
    if has_seed_roll != (protocol_versions.get("seed_roll") == "splendor-seed-roll/1"):
        raise ValueError("manifest v2 seed-roll protocol declaration mismatch")
    artifact_contract = _require_mapping(declaration, "artifact_contract")
    scenario_banks = artifact_contract.get("scenario_banks")
    if scenario_banks is not None:
        if not isinstance(scenario_banks, Mapping) or not scenario_banks:
            raise ValueError("artifact_contract scenario_banks must be a mapping")
        payload_hashes: list[str] = []
        for logical_split, payload_sha256 in scenario_banks.items():
            if type(logical_split) is not str or not logical_split:
                raise ValueError("scenario bank split names must be non-empty strings")
            if (
                type(payload_sha256) is not str
                or len(payload_sha256) != SHA256_HEX_LENGTH
                or any(
                    character not in "0123456789abcdef" for character in payload_sha256
                )
            ):
                raise ValueError(
                    f"scenario bank {logical_split!r} payload SHA-256 is invalid"
                )
            payload_hashes.append(payload_sha256)
        if len(payload_hashes) != len(set(payload_hashes)):
            raise ValueError("scenario bank bindings must have unique payload hashes")
    raw_formal_training = declaration.get("formal_training")
    if raw_formal_training is not None:
        formal_training = _require_mapping(declaration, "formal_training")
        training_protocol = formal_training.get("protocol")
        if training_protocol not in {"paired-training-v1", "paired-training-v2"}:
            raise ValueError(
                "formal training must declare paired-training-v1 or paired-training-v2"
            )
        schedule_hash = formal_training.get("paired_schedule_sha256")
        if (
            type(schedule_hash) is not str
            or len(schedule_hash) != SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in schedule_hash)
        ):
            raise ValueError("formal training schedule SHA-256 is invalid")
        treatments = formal_training.get("expected_treatments")
        if (
            not isinstance(treatments, list)
            or not treatments
            or any(type(value) is not str or not value for value in treatments)
            or treatments != sorted(set(treatments))
        ):
            raise ValueError(
                "formal training expected_treatments must be a sorted unique list"
            )
        replicate_ids = formal_training.get("replicate_ids")
        if (
            not isinstance(replicate_ids, list)
            or not replicate_ids
            or any(type(value) is not int or value < 0 for value in replicate_ids)
            or replicate_ids != sorted(set(replicate_ids))
        ):
            raise ValueError(
                "formal training replicate_ids must be a sorted unique list"
            )
        worker_count = formal_training.get("worker_count")
        if type(worker_count) is not int or worker_count < 1:
            raise ValueError("formal training worker_count must be positive")
        if formal_training.get("worker_scope") != "independent-training-jobs":
            raise ValueError("formal training worker_scope is invalid")
        scenario_bank_hash = formal_training.get("scenario_bank_sha256")
        if scenario_bank_hash is not None and (
            type(scenario_bank_hash) is not str
            or len(scenario_bank_hash) != SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in scenario_bank_hash
            )
        ):
            raise ValueError("formal training scenario-bank SHA-256 is invalid")
        raw_seed_roll = _require_mapping(declaration, "seed_plan").get("seed_roll")
        if training_protocol == "paired-training-v2":
            required_v2_fields = {
                "protocol",
                "paired_schedule_sha256",
                "expected_treatments",
                "replicate_ids",
                "worker_count",
                "worker_scope",
                "scenario_bank_sha256",
                "seed_roll_payload_sha256",
                "randomization_root_sha256",
                "treatment_contracts",
                "job_outputs",
                "replicate_stage",
                "activation_artifact_path",
                "activation_artifact_sha256",
            }
            if set(formal_training) != required_v2_fields:
                raise ValueError("paired-training-v2 binding schema mismatch")
            if raw_seed_roll is None:
                raise ValueError("paired-training-v2 requires a manifest seed roll")
            if type(scenario_bank_hash) is not str:
                raise ValueError("paired-training-v2 requires a scenario-bank SHA-256")
            from .seed_roll import (  # noqa: PLC0415 - avoid manifest import cycle
                load_confirmatory_activation_artifact,
                validate_seed_roll_binding,
            )

            seed_roll = validate_seed_roll_binding(raw_seed_roll)
            if (
                seed_roll["experiment_id"] != declaration.get("experiment_id")
                or seed_roll["phase"] != declaration.get("phase")
                or formal_training.get("seed_roll_payload_sha256")
                != seed_roll["payload_sha256"]
                or formal_training.get("randomization_root_sha256")
                != seed_roll["randomization_root_sha256"]
            ):
                raise ValueError("formal training seed-roll binding mismatch")
            roll_replicates = set(cast(list[int], seed_roll["replicate_ids"]))
            if not set(cast(list[int], replicate_ids)) <= roll_replicates:
                raise ValueError(
                    "formal training replicate IDs are not reserved by the seed roll"
                )
            replicate_stage = formal_training.get("replicate_stage")
            activation_path = formal_training.get("activation_artifact_path")
            activation_sha256 = formal_training.get("activation_artifact_sha256")
            expected_stage_replicates = (
                seed_roll["pilot_replicate_ids"]
                if replicate_stage == "pilot"
                else seed_roll["confirmatory_reserve_replicate_ids"]
            )
            if (
                replicate_stage not in {"pilot", "confirmatory-reserve"}
                or replicate_ids != expected_stage_replicates
            ):
                raise ValueError(
                    "paired-training-v2 replicate IDs do not match their "
                    "pre-registered stage"
                )
            if replicate_stage == "pilot":
                if activation_path is not None or activation_sha256 is not None:
                    raise ValueError(
                        "paired-training-v2 pilot cannot declare activation evidence"
                    )
                if (
                    artifact_contract.get("confirmatory_activation_path") is not None
                    or artifact_contract.get("confirmatory_activation_sha256")
                    is not None
                ):
                    raise ValueError(
                        "paired-training-v2 pilot cannot bind confirmatory activation"
                    )
            else:
                if (
                    type(activation_path) is not str
                    or not Path(activation_path).is_absolute()
                    or os.path.normpath(activation_path) != activation_path
                    or Path(activation_path).resolve(strict=False)
                    != Path(activation_path)
                    or not _is_sha256(activation_sha256)
                ):
                    raise ValueError(
                        "paired-training-v2 confirmatory reserve requires a "
                        "normalized activation path and SHA-256"
                    )
                if (
                    artifact_contract.get("confirmatory_activation_path")
                    != activation_path
                    or artifact_contract.get("confirmatory_activation_sha256")
                    != activation_sha256
                ):
                    raise ValueError(
                        "paired-training-v2 confirmatory activation is not bound by "
                        "the artifact contract"
                    )
                activation = load_confirmatory_activation_artifact(
                    Path(activation_path),
                    expected_experiment_id=cast(str, declaration.get("experiment_id")),
                    expected_phase=cast(str, declaration.get("phase")),
                    expected_seed_roll_payload_sha256=cast(
                        str, seed_roll["payload_sha256"]
                    ),
                    expected_pilot_replicate_ids=cast(
                        list[int], seed_roll["pilot_replicate_ids"]
                    ),
                    expected_confirmatory_replicate_ids=cast(
                        list[int], seed_roll["confirmatory_reserve_replicate_ids"]
                    ),
                )
                if activation.artifact_sha256 != activation_sha256:
                    raise ValueError(
                        "paired-training-v2 confirmatory activation SHA-256 mismatch"
                    )
            if (
                not isinstance(scenario_banks, Mapping)
                or scenario_banks.get("train-schedule") != scenario_bank_hash
            ):
                raise ValueError(
                    "paired-training-v2 scenario bank is not bound by the artifact contract"
                )
            contracts = formal_training.get("treatment_contracts")
            if not isinstance(contracts, list) or len(contracts) != len(treatments):
                raise ValueError(
                    "paired-training-v2 treatment contracts are incomplete"
                )
            contract_ids: list[str] = []
            for contract in contracts:
                if not isinstance(contract, Mapping) or set(contract) != {
                    "treatment_id",
                    "initial_checkpoint_sha256",
                    "trainer_config_sha256",
                    "opponent_pool_sha256",
                }:
                    raise ValueError(
                        "paired-training-v2 treatment contract schema mismatch"
                    )
                treatment_id = contract.get("treatment_id")
                if type(treatment_id) is not str:
                    raise ValueError(
                        "paired-training-v2 treatment contract ID is invalid"
                    )
                contract_ids.append(treatment_id)
                if any(
                    not _is_sha256(contract.get(field))
                    for field in (
                        "initial_checkpoint_sha256",
                        "trainer_config_sha256",
                        "opponent_pool_sha256",
                    )
                ):
                    raise ValueError(
                        "paired-training-v2 treatment contract hash is invalid"
                    )
            if contract_ids != treatments:
                raise ValueError(
                    "paired-training-v2 contracts do not match expected treatments"
                )
            outputs = formal_training.get("job_outputs")
            if not isinstance(outputs, list):
                raise ValueError("paired-training-v2 job outputs are invalid")
            output_coordinates: list[tuple[int, str]] = []
            output_paths: list[str] = []
            for output in outputs:
                if not isinstance(output, Mapping) or set(output) != {
                    "replicate_id",
                    "treatment_id",
                    "output_dir",
                }:
                    raise ValueError("paired-training-v2 job output schema mismatch")
                replicate_id = output.get("replicate_id")
                treatment_id = output.get("treatment_id")
                output_dir = output.get("output_dir")
                if (
                    type(replicate_id) is not int
                    or type(treatment_id) is not str
                    or type(output_dir) is not str
                    or not Path(output_dir).is_absolute()
                    or os.path.normpath(output_dir) != output_dir
                    or Path(output_dir).resolve(strict=False) != Path(output_dir)
                ):
                    raise ValueError("paired-training-v2 job output is invalid")
                output_coordinates.append((replicate_id, treatment_id))
                output_paths.append(output_dir)
            expected_output_coordinates = [
                (replicate_id, treatment_id)
                for replicate_id in replicate_ids
                for treatment_id in treatments
            ]
            if output_coordinates != expected_output_coordinates or len(
                output_paths
            ) != len(set(output_paths)):
                raise ValueError(
                    "paired-training-v2 job output matrix is incomplete or duplicated"
                )
        elif raw_seed_roll is not None:
            raise ValueError("paired-training-v1 cannot declare a seed roll")
    raw_statistics = declaration.get("statistical_protocol")
    statistical_protocol = None
    if raw_statistics is not None:
        from .statistics import (  # noqa: PLC0415 - avoid manifest import cycle
            StatisticsError,
            validate_analysis_plan_binding,
            validate_statistical_protocol_binding,
        )

        try:
            statistical_protocol = validate_statistical_protocol_binding(raw_statistics)
        except StatisticsError as exc:
            raise ValueError(
                f"manifest statistical protocol is invalid: {exc}"
            ) from exc
    raw_evaluation = declaration.get("paired_evaluation")
    if raw_evaluation is not None:
        from .paired_evaluation import (  # noqa: PLC0415 - avoid import cycle
            PairedEvaluationError,
            validate_paired_evaluation_binding,
        )

        if raw_statistics is None:
            raise ValueError("formal paired evaluation requires a statistical_protocol")
        try:
            validate_paired_evaluation_binding(raw_evaluation)
            assert statistical_protocol is not None
            validate_analysis_plan_binding(raw_evaluation, statistical_protocol)
        except PairedEvaluationError as exc:
            raise ValueError(f"manifest paired evaluation is invalid: {exc}") from exc
        except StatisticsError as exc:
            raise ValueError(
                f"manifest evaluation/statistics binding is invalid: {exc}"
            ) from exc


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


def _validate_lifecycle_event(  # noqa: C901,PLR0913 - explicit chain inputs
    event: Mapping[str, Any],
    *,
    initial: bool,
    expected_source: str,
    previous_event_sha256: str | None,
    declaration_sha256: object,
    provenance_sha256: object,
) -> tuple[str, str]:
    required = {
        "from",
        "to",
        "actor",
        "note",
        "at",
        "declaration_sha256",
        "provenance_sha256",
        "previous_event_sha256",
        "event_sha256",
    }
    if set(event) != required:
        raise ValueError("manifest v2 lifecycle event schema mismatch")
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
    for field in ("actor", "note"):
        value = event.get(field)
        if type(value) is not str or not value.strip():
            raise ValueError(f"manifest v2 lifecycle event {field} is invalid")
    if not _is_utc_timestamp(event.get("at")):
        raise ValueError("manifest v2 lifecycle event timestamp is invalid")
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
    formal_training: Mapping[str, Any] | None = None,
    seed_roll: Mapping[str, Any] | None = None,
    paired_evaluation: Mapping[str, Any] | None = None,
    statistical_protocol: Mapping[str, Any] | None = None,
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
            "scenario": "splendor-scenario/1",
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
    if seed_roll is not None:
        cast(dict[str, Any], declaration["protocol_versions"])["seed_roll"] = (
            "splendor-seed-roll/1"
        )
        cast(dict[str, Any], declaration["seed_plan"])["seed_roll"] = dict(seed_roll)
    if formal_training is not None:
        declaration["formal_training"] = dict(formal_training)
    if paired_evaluation is not None:
        declaration["paired_evaluation"] = dict(paired_evaluation)
    if statistical_protocol is not None:
        declaration["statistical_protocol"] = dict(statistical_protocol)
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
    path = _real_manifest_path(path, create_parent=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open manifest: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("manifest path is not a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            manifest = json.load(stream)
    finally:
        os.close(descriptor)
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
    with _manifest_lock(path):
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
