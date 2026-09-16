"""Read-only progress monitor for the Task-1 T1.4 crossed pilot.

The production orchestrator is intentionally fail-closed and writes a small
amount of lifecycle state beside the declaration.  This module is an operator
view over those files; it never transitions a manifest, resumes a job, or
creates an artifact.  In particular, an old r3 run may have a multi-gigabyte
``result.json`` but no ``updates.jsonl.zst``.  Result files and journals are read
with bounded work where possible so that observing a run does not compete with
the trainer for disk or memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

JsonObject = dict[str, object]

MONITOR_SCHEMA: Final = "splendor-t14-pilot-monitor/1"
ORCHESTRATOR_STATUS_NAME: Final = "orchestrator-status.json"
LAUNCH_DECLARATION_NAME: Final = "launch-declaration.json"
COMPLETION_NAME: Final = "completion.json"
JOB_STATUS_NAME: Final = "status.json"
JOB_RESULT_NAME: Final = "result.json"
# The formal trainer currently writes concatenated zstd frames.  The plain
# ``updates.jsonl`` fallback is retained solely for pre-zstd local fixtures and
# historical runs; the r3 runner itself has no ``updates.jsonl.zst`` journal.
UPDATE_JOURNAL_NAME: Final = "updates.jsonl.zst"
LEGACY_UPDATE_JOURNAL_NAME: Final = "updates.jsonl"
_MAX_FULL_RESULT_BYTES: Final = 8 * 1024 * 1024
_MAX_FULL_JOURNAL_BYTES: Final = 8 * 1024 * 1024
_EDGE_READ_BYTES: Final = 2 * 1024 * 1024
_TMUX_TIMEOUT_SECONDS: Final = 2.0
_PILOT_JOB_COUNT: Final = 6
_SESSION_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_SCHEMA_PATTERN: Final = re.compile(
    r"^splendor-t14-(?:pilot-orchestration|pilot-status)/[0-9]+$"
)
_TOP_LEVEL_LINE_PATTERN: Final = re.compile(
    r'^  "(?P<key>[A-Za-z_][A-Za-z0-9_]*)":\s*(?P<value>.*?)(?:,)?$'
)

_TERMINAL_JOB_STATUSES: Final = frozenset(
    {
        "completed",
        "failed",
        "validation-failed",
        "interrupted",
        "cancelled",
        "blocked",
        "not-started",
    }
)
_ACTIVE_JOB_STATUSES: Final = frozenset({"running", "initial"})
_QUEUED_JOB_STATUSES: Final = frozenset({"queued", "pending"})
_ORCHESTRATOR_STATES: Final = frozenset(
    {"launching", "running", "completed", "failed", "blocked"}
)


class PilotMonitorError(RuntimeError):
    """Raised when a monitor input is missing, unsafe, or internally invalid."""


@dataclass(frozen=True)
class _Declaration:
    path: Path
    digest: str
    raw: JsonObject
    experiment_id: str
    phase: str
    manifest_path: Path
    control_dir: Path
    output_root: Path
    tmux_session: str
    treatment_targets: dict[str, int]
    limits: dict[str, int]

    @property
    def status_path(self) -> Path:
        return self.control_dir / ORCHESTRATOR_STATUS_NAME

    @property
    def launch_path(self) -> Path:
        return self.control_dir / LAUNCH_DECLARATION_NAME

    @property
    def completion_path(self) -> Path:
        return self.control_dir / COMPLETION_NAME


@dataclass(frozen=True)
class _Manifest:
    path: Path
    digest: str
    status: str
    raw: JsonObject
    jobs: tuple[tuple[int, str, Path], ...]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise PilotMonitorError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotMonitorError(f"value is not canonical JSON: {exc}") from exc


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise PilotMonitorError(f"{label} must be an absolute normalized path")
    if "\x00" in value:
        raise PilotMonitorError(f"{label} contains a NUL byte")
    path = Path(value)
    if (
        not path.is_absolute()
        or os.path.normpath(value) != value
        or Path(os.path.abspath(value)) != path  # noqa: PTH100
    ):
        raise PilotMonitorError(f"{label} must be an absolute normalized path")
    try:
        if path.resolve(strict=False) != path:
            raise PilotMonitorError(f"{label} traverses a symlink")
    except OSError as exc:
        raise PilotMonitorError(f"cannot inspect {label}: {exc}") from exc
    return path


def _read_json(
    path: Path,
    *,
    label: str,
    missing_ok: bool = False,
) -> tuple[JsonObject | None, str | None]:
    """Read one regular JSON file without following symlinks."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None, None
        raise PilotMonitorError(f"{label} is missing: {path}") from None
    except OSError as exc:
        raise PilotMonitorError(f"cannot inspect {label} {path}: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise PilotMonitorError(f"{label} must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, PilotMonitorError) as exc:
        raise PilotMonitorError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PilotMonitorError(f"{label} root must be a JSON object: {path}")
    return cast(JsonObject, payload), _canonical_digest(payload)


def _read_optional_file_metadata(path: Path, *, label: str) -> os.stat_result | None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PilotMonitorError(f"cannot inspect {label} {path}: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise PilotMonitorError(f"{label} must be a regular non-symlink file: {path}")
    return metadata


def _read_treatment_targets(raw: object) -> dict[str, int]:
    targets: dict[str, int] = {}
    if raw is None:
        return targets
    if not isinstance(raw, list):
        raise PilotMonitorError("declaration treatments must be a list")
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise PilotMonitorError(f"declaration treatments[{index}] must be an object")
        treatment_id = item.get("treatment_id")
        if type(treatment_id) is not str or not treatment_id:
            raise PilotMonitorError(
                f"declaration treatments[{index}].treatment_id is invalid"
            )
        config = item.get("config")
        if not isinstance(config, Mapping):
            continue
        updates = config.get("updates")
        if updates is None:
            continue
        if type(updates) is not int or updates <= 0:
            raise PilotMonitorError(
                f"declaration treatment {treatment_id!r} has invalid updates"
            )
        previous = targets.setdefault(treatment_id, updates)
        if previous != updates:
            raise PilotMonitorError(
                f"declaration treatment {treatment_id!r} has inconsistent updates"
            )
    return targets


def _read_limits(raw: object) -> dict[str, int]:
    """Read declaration-owned resource limits without supplying defaults."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PilotMonitorError("declaration limits must be an object")
    limits: dict[str, int] = {}
    for name in ("wall_seconds", "max_output_bytes", "min_free_bytes"):
        value = raw.get(name)
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise PilotMonitorError(f"declaration limit {name} is invalid")
        limits[name] = value
    return limits


def _load_declaration(path: Path) -> _Declaration:
    source = _canonical_path(str(path), label="declaration path")
    raw, digest = _read_json(source, label="orchestration declaration")
    if raw is None or digest is None:  # pragma: no cover - missing_ok is false
        raise PilotMonitorError("orchestration declaration is missing")
    schema = raw.get("schema_version")
    if type(schema) is not str or _SCHEMA_PATTERN.fullmatch(schema) is None:
        raise PilotMonitorError("orchestration declaration schema_version is invalid")
    if not schema.startswith("splendor-t14-pilot-orchestration/"):
        raise PilotMonitorError("orchestration declaration schema is not T1.4")
    experiment_id = raw.get("experiment_id")
    phase = raw.get("phase")
    if type(experiment_id) is not str or not experiment_id:
        raise PilotMonitorError("orchestration declaration experiment_id is invalid")
    if phase != "T1.4":
        raise PilotMonitorError("orchestration declaration phase must be T1.4")
    manifest_path = _canonical_path(raw.get("manifest_path"), label="manifest_path")
    control_dir = _canonical_path(raw.get("control_dir"), label="control_dir")
    output_root = _canonical_path(raw.get("output_root"), label="output_root")
    if control_dir != source.parent:
        raise PilotMonitorError("control_dir must be the declaration parent directory")
    if output_root == control_dir or control_dir in output_root.parents:
        raise PilotMonitorError("output_root must not contain control_dir")
    tmux_session = raw.get("tmux_session")
    if type(tmux_session) is not str or _SESSION_PATTERN.fullmatch(tmux_session) is None:
        raise PilotMonitorError("tmux_session is not a canonical session name")
    return _Declaration(
        path=source,
        digest=digest,
        raw=raw,
        experiment_id=experiment_id,
        phase=phase,
        manifest_path=manifest_path,
        control_dir=control_dir,
        output_root=output_root,
        tmux_session=tmux_session,
        treatment_targets=_read_treatment_targets(raw.get("treatments")),
        limits=_read_limits(raw.get("limits")),
    )


def _load_manifest(declaration: _Declaration) -> _Manifest:  # noqa: C901,PLR0912
    raw, digest = _read_json(declaration.manifest_path, label="manifest")
    if raw is None or digest is None:  # pragma: no cover - missing_ok is false
        raise PilotMonitorError("manifest is missing")
    status = raw.get("status")
    if type(status) is not str or not status:
        raise PilotMonitorError("manifest status is missing or invalid")
    manifest_declaration = raw.get("declaration")
    if isinstance(manifest_declaration, Mapping):
        manifest_experiment = manifest_declaration.get("experiment_id")
        if manifest_experiment is not None and manifest_experiment != declaration.experiment_id:
            raise PilotMonitorError("manifest experiment_id does not match declaration")
        formal = manifest_declaration.get("formal_training")
    else:
        formal = None
    if not isinstance(formal, Mapping):
        raise PilotMonitorError("manifest declaration.formal_training is missing")
    raw_jobs = formal.get("job_outputs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise PilotMonitorError("manifest formal_training.job_outputs is missing")
    jobs: list[tuple[int, str, Path]] = []
    coordinates: set[tuple[int, str]] = set()
    output_paths: set[Path] = set()
    for index, item in enumerate(raw_jobs):
        if not isinstance(item, Mapping):
            raise PilotMonitorError(f"manifest job_outputs[{index}] must be an object")
        replicate_id = item.get("replicate_id")
        treatment_id = item.get("treatment_id")
        if type(replicate_id) is not int or replicate_id < 0:
            raise PilotMonitorError(f"manifest job_outputs[{index}].replicate_id is invalid")
        if type(treatment_id) is not str or not treatment_id:
            raise PilotMonitorError(f"manifest job_outputs[{index}].treatment_id is invalid")
        output_dir = _canonical_path(
            item.get("output_dir"),
            label=f"manifest job_outputs[{index}].output_dir",
        )
        if output_dir.parent != declaration.output_root:
            raise PilotMonitorError(
                f"manifest job {replicate_id}/{treatment_id} is not a direct output child"
            )
        coordinate = (replicate_id, treatment_id)
        if coordinate in coordinates:
            raise PilotMonitorError(f"manifest contains duplicate job {coordinate}")
        if output_dir in output_paths:
            raise PilotMonitorError(f"manifest contains duplicate output path {output_dir}")
        coordinates.add(coordinate)
        output_paths.add(output_dir)
        jobs.append((replicate_id, treatment_id, output_dir))
    return _Manifest(
        path=declaration.manifest_path,
        digest=digest,
        status=status,
        raw=raw,
        jobs=tuple(jobs),
    )


def _validate_status_identity(  # noqa: C901,PLR0912,PLR0915
    declaration: _Declaration,
    payload: JsonObject,
    *,
    launch_digest: str | None,
) -> None:
    schema = payload.get("schema_version")
    if schema is not None and (
        type(schema) is not str or _SCHEMA_PATTERN.fullmatch(schema) is None
    ):
        raise PilotMonitorError("orchestrator status schema_version is invalid")
    state = payload.get("state")
    if state not in _ORCHESTRATOR_STATES:
        raise PilotMonitorError(f"orchestrator status state is invalid: {state!r}")
    status_hash = payload.get("status_sha256")
    if status_hash is not None:
        if type(status_hash) is not str:
            raise PilotMonitorError("orchestrator status hash is invalid")
        body = dict(payload)
        body.pop("status_sha256", None)
        if status_hash != _canonical_digest(body):
            raise PilotMonitorError("orchestrator status hash is invalid")
    for key, expected in (
        ("experiment_id", declaration.experiment_id),
        ("tmux_session", declaration.tmux_session),
    ):
        value = payload.get(key)
        if value is not None and value != expected:
            raise PilotMonitorError(f"orchestrator status {key} does not match declaration")
    status_path_value = payload.get("declaration_path")
    if status_path_value is not None:
        expected_paths = {str(declaration.path)}
        if launch_digest is not None:
            expected_paths.add(str(declaration.launch_path))
        if status_path_value not in expected_paths:
            raise PilotMonitorError("orchestrator status declaration_path is not bound")
    declaration_sha = payload.get("declaration_sha256")
    allowed_declaration_hashes = {declaration.digest}
    if launch_digest is not None:
        allowed_declaration_hashes.add(launch_digest)
    if declaration_sha is not None and declaration_sha not in allowed_declaration_hashes:
        raise PilotMonitorError("orchestrator status declaration hash is not bound")
    for key in ("updated_unix_seconds", "elapsed_seconds"):
        value = payload.get(key)
        if value is not None:
            if type(value) not in {int, float}:
                raise PilotMonitorError(f"orchestrator status {key} is invalid")
            numeric_value = float(cast(int | float, value))
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise PilotMonitorError(f"orchestrator status {key} is invalid")
    resources = payload.get("resources")
    if resources is not None:
        if not isinstance(resources, Mapping) or set(resources) != {
            "output_bytes",
            "free_bytes",
        }:
            raise PilotMonitorError("orchestrator status resources are invalid")
        for key in ("output_bytes", "free_bytes"):
            value = resources.get(key)
            if type(value) is not int or value < 0:
                raise PilotMonitorError(f"orchestrator status resources.{key} is invalid")
    status_limits = payload.get("limits")
    if status_limits is not None:
        if not isinstance(status_limits, Mapping):
            raise PilotMonitorError("orchestrator status limits are invalid")
        for limit_name, declared_limit in declaration.limits.items():
            observed = status_limits.get(limit_name)
            if observed is not None and observed != declared_limit:
                raise PilotMonitorError(
                    f"orchestrator status limit {limit_name} does not match declaration"
                )


def _load_orchestrator_status(declaration: _Declaration) -> JsonObject | None:
    launch_raw, launch_digest = _read_json(
        declaration.launch_path,
        label="launch declaration",
        missing_ok=True,
    )
    del launch_raw  # The digest is the only identity needed here.
    payload, _ = _read_json(
        declaration.status_path,
        label="orchestrator status",
        missing_ok=True,
    )
    if payload is not None:
        _validate_status_identity(
            declaration,
            payload,
            launch_digest=launch_digest,
        )
    return payload


def _sha256_regular_file(path: Path) -> str:
    """Hash one regular file without following a replacement symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PilotMonitorError(f"cannot open completion artifact {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PilotMonitorError(
                f"completion artifact is not a regular file: {path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _completion_file_names() -> set[str]:
    """Return the exact formal files attested for each completed pilot job."""
    validation_updates = range(0, 2001, 50)
    checkpoint_updates = range(50, 2001, 50)
    return {
        "formal-output-reservation.json",
        "initial.pth",
        "best.pth",
        "final.pth",
        JOB_RESULT_NAME,
        JOB_STATUS_NAME,
        UPDATE_JOURNAL_NAME,
        "checkpoint-selection.json",
        *(f"validation-update-{update}.json" for update in validation_updates),
        *(f"update-{update}.pth" for update in checkpoint_updates),
    }


def _completion_summary(  # noqa: C901,PLR0912 - terminal receipt attestation
    declaration: _Declaration,
    manifest: _Manifest,
) -> JsonObject:
    """Verify the receipt and stream-hash every declared terminal artifact."""
    payload, _ = _read_json(
        declaration.completion_path,
        label="completion evidence",
        missing_ok=True,
    )
    if payload is None:
        return {
            "present": False,
            "valid": False,
            "sha256": None,
            "files_verified": 0,
            "error": None,
        }
    body = dict(payload)
    completion_sha256 = body.pop("completion_sha256", None)
    try:
        if set(payload) != {
            "schema_version",
            "declaration_path",
            "declaration_sha256",
            "experiment_id",
            "manifest_path",
            "seed_roll_payload_sha256",
            "scenario_bank_payload_sha256",
            "validation_bank_payload_sha256",
            "formal_training_sha256",
            "jobs",
            "completion_sha256",
        }:
            raise PilotMonitorError("completion evidence schema is invalid")
        if (
            payload.get("schema_version") != "splendor-t14-pilot-completion/2"
            or payload.get("declaration_sha256") != declaration.digest
            or payload.get("declaration_path")
            not in {str(declaration.path), str(declaration.launch_path)}
            or payload.get("experiment_id") != declaration.experiment_id
            or payload.get("manifest_path") != str(manifest.path)
            or type(completion_sha256) is not str
            or completion_sha256 != _canonical_digest(body)
        ):
            raise PilotMonitorError("completion evidence hash or identity is invalid")
        for key in (
            "seed_roll_payload_sha256",
            "scenario_bank_payload_sha256",
            "validation_bank_payload_sha256",
            "formal_training_sha256",
        ):
            digest = payload.get(key)
            if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
                raise PilotMonitorError(f"completion evidence {key} is invalid")
        expected_jobs = {
            (replicate_id, treatment_id): output_dir
            for replicate_id, treatment_id, output_dir in manifest.jobs
        }
        if len(expected_jobs) != _PILOT_JOB_COUNT:
            raise PilotMonitorError("completion requires exactly six manifest jobs")
        raw_jobs = payload.get("jobs")
        if not isinstance(raw_jobs, list) or len(raw_jobs) != len(expected_jobs):
            raise PilotMonitorError("completion evidence job count is invalid")
        seen: set[tuple[int, str]] = set()
        files_verified = 0
        expected_file_names = _completion_file_names()
        for raw_job in raw_jobs:
            if not isinstance(raw_job, Mapping) or set(raw_job) != {
                "replicate_id",
                "treatment_id",
                "output_dir",
                "files",
            }:
                raise PilotMonitorError("completion job record schema is invalid")
            replicate_id = raw_job.get("replicate_id")
            treatment_id = raw_job.get("treatment_id")
            if type(replicate_id) is not int or type(treatment_id) is not str:
                raise PilotMonitorError("completion job coordinate is invalid")
            coordinate = (replicate_id, treatment_id)
            output_dir = expected_jobs.get(coordinate)
            if (
                output_dir is None
                or coordinate in seen
                or raw_job.get("output_dir") != str(output_dir)
            ):
                raise PilotMonitorError("completion job does not match the manifest")
            seen.add(coordinate)
            files = raw_job.get("files")
            if not isinstance(files, Mapping) or set(files) != expected_file_names:
                raise PilotMonitorError("completion job file inventory is invalid")
            for name, expected_sha256 in files.items():
                if (
                    type(name) is not str
                    or type(expected_sha256) is not str
                    or _SHA256_PATTERN.fullmatch(expected_sha256) is None
                ):
                    raise PilotMonitorError("completion artifact hash is invalid")
                artifact = output_dir / name
                if _sha256_regular_file(artifact) != expected_sha256:
                    raise PilotMonitorError(
                        f"completion artifact hash changed: {coordinate}/{name}"
                    )
                files_verified += 1
        if set(expected_jobs) != seen:
            raise PilotMonitorError("completion evidence coordinates are incomplete")
    except PilotMonitorError as exc:
        return {
            "present": True,
            "valid": False,
            "sha256": (
                completion_sha256 if type(completion_sha256) is str else None
            ),
            "files_verified": 0,
            "error": str(exc),
        }
    return {
        "present": True,
        "valid": True,
        "sha256": completion_sha256,
        "files_verified": files_verified,
        "error": None,
    }


def _read_top_level_edges(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(_EDGE_READ_BYTES)
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _EDGE_READ_BYTES), os.SEEK_SET)
            suffix = stream.read(_EDGE_READ_BYTES)
    except (OSError, ValueError) as exc:
        raise PilotMonitorError(f"cannot read result edges {path}: {exc}") from exc
    try:
        return prefix.decode("utf-8"), suffix.decode("utf-8")
    except UnicodeError as exc:
        raise PilotMonitorError(f"result file is not UTF-8 JSON: {path}") from exc


def _extract_top_level_values(text: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in text.splitlines():
        match = _TOP_LEVEL_LINE_PATTERN.match(line)
        if match is None:
            continue
        key = match.group("key")
        if key in values:
            continue
        encoded = match.group("value").strip()
        try:
            values[key] = json.loads(encoded)
        except json.JSONDecodeError:
            # A value can be split at the bounded edge.  It is simply not
            # available in the summary; the status file remains authoritative.
            continue
    return values


def _result_summary(path: Path) -> JsonObject:  # noqa: C901
    metadata = _read_optional_file_metadata(path, label="job result")
    if metadata is None:
        return {
            "present": False,
            "status": None,
            "size_bytes": 0,
            "read_mode": "missing",
        }
    size = metadata.st_size
    payload: JsonObject | None = None
    read_mode = "full"
    if size <= _MAX_FULL_RESULT_BYTES:
        payload, _ = _read_json(path, label="job result")
    else:
        read_mode = "bounded-edges"
        prefix, suffix = _read_top_level_edges(path)
        values = _extract_top_level_values(prefix)
        values.update({key: value for key, value in _extract_top_level_values(suffix).items() if key not in values})
        payload = values
    assert payload is not None
    status = payload.get("status")
    if status is not None and type(status) is not str:
        raise PilotMonitorError(f"job result status is invalid: {path}")
    result: JsonObject = {
        "present": True,
        "status": status,
        "size_bytes": size,
        "read_mode": read_mode,
    }
    for key in ("best_update", "elapsed_seconds", "error"):
        if key in payload:
            result[key] = payload[key]
    config = payload.get("config")
    if isinstance(config, Mapping) and type(config.get("updates")) is int:
        result["target_updates"] = config["updates"]
    journal_metadata = payload.get("formal_update_journal")
    if isinstance(journal_metadata, Mapping):
        if type(journal_metadata.get("entries")) is int:
            result["journal_entries"] = journal_metadata["entries"]
        journal_path = journal_metadata.get("path")
        if type(journal_path) is str:
            result["journal_path"] = journal_path
    aggregate_phases = payload.get("aggregate_phase_seconds")
    if isinstance(aggregate_phases, Mapping):
        result["aggregate_phase_seconds"] = dict(aggregate_phases)
    return result


def _journal_entry_digest(body: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _validate_journal_entry(
    raw: object,
    *,
    expected_index: int,
    previous_digest: str | None,
) -> tuple[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "index",
        "update",
        "previous_entry_sha256",
        "record",
        "entry_sha256",
    }:
        raise PilotMonitorError("formal update journal entry schema is invalid")
    entry = dict(raw)
    entry_digest = entry.pop("entry_sha256")
    if (
        entry.get("schema_version") != "splendor-formal-ppo-update-journal/1"
        or entry.get("index") != expected_index
        or entry.get("update") != expected_index
        or entry.get("previous_entry_sha256") != previous_digest
        or type(entry_digest) is not str
        or entry_digest != _journal_entry_digest(entry)
    ):
        raise PilotMonitorError("formal update journal chain is invalid")
    record = entry.get("record")
    if not isinstance(record, Mapping) or record.get("update") != expected_index:
        raise PilotMonitorError("formal update journal record coordinate is invalid")
    return cast(str, entry_digest), expected_index


def _journal_summary(  # noqa: C901,PLR0912
    path: Path,
    *,
    expected_entries: int | None = None,
    expected_last_update: int | None = None,
) -> JsonObject:
    metadata = _read_optional_file_metadata(path, label="update journal")
    if metadata is None:
        return {
            "present": False,
            "compatibility": "legacy-no-journal",
            "entries": 0,
            "last_update": None,
            "size_bytes": 0,
            "read_mode": "missing",
            "error": None,
        }
    size = metadata.st_size
    if path.suffix == ".zst":
        # Formal v2 journals are concatenated zstd frames.  Decompressing every
        # frame on every watch tick would defeat the purpose of this monitor;
        # status.json is the durable index/entry count, while the journal's
        # presence and byte size are still observed here.  We intentionally do
        # not invent an update number if the status metadata is absent.
        return {
            "present": True,
            "compatibility": "formal-journal-zstd",
            "entries": expected_entries,
            "last_update": expected_last_update,
            "size_bytes": size,
            "read_mode": "status-metadata",
            "error": None
            if expected_entries is not None
            else "compressed journal requires status journal_entries metadata",
        }
    if size == 0:
        return {
            "present": True,
            "compatibility": "formal-journal",
            "entries": 0,
            "last_update": None,
            "size_bytes": 0,
            "read_mode": "full",
            "error": None,
        }
    if size > _MAX_FULL_JOURNAL_BYTES:
        try:
            with path.open("rb") as stream:
                newline_count = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    newline_count += chunk.count(b"\n")
                stream.seek(max(0, size - _EDGE_READ_BYTES), os.SEEK_SET)
                tail = stream.read(_EDGE_READ_BYTES)
        except OSError as exc:
            raise PilotMonitorError(f"cannot read update journal {path}: {exc}") from exc
        complete_entries = newline_count
        if not tail.endswith(b"\n"):
            complete_entries = max(0, complete_entries)
            last_newline = tail.rfind(b"\n")
            tail = tail[: last_newline + 1] if last_newline >= 0 else b""
        last_line = tail.splitlines()[-1] if tail.splitlines() else b""
        if not last_line:
            raise PilotMonitorError("update journal has no readable final entry")
        try:
            last_raw = json.loads(last_line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PilotMonitorError("update journal final entry is invalid") from exc
        if not isinstance(last_raw, Mapping) or type(last_raw.get("update")) is not int:
            raise PilotMonitorError("update journal final entry coordinate is invalid")
        return {
            "present": True,
            "compatibility": "formal-journal",
            "entries": complete_entries,
            "last_update": last_raw["update"],
            "size_bytes": size,
            "read_mode": "bounded-tail",
            "error": "chain not fully revalidated for a large live journal",
        }
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PilotMonitorError(f"cannot read update journal {path}: {exc}") from exc
    lines = data.splitlines(keepends=True)
    partial = bool(lines and not lines[-1].endswith(b"\n"))
    if partial:
        lines = lines[:-1]
    previous: str | None = None
    last_update: int | None = None
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            raise PilotMonitorError("update journal contains a truncated entry")
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PilotMonitorError("update journal contains invalid JSON") from exc
        previous, last_update = _validate_journal_entry(
            raw,
            expected_index=index,
            previous_digest=previous,
        )
    return {
        "present": True,
        "compatibility": "formal-journal",
        "entries": len(lines),
        "last_update": last_update,
        "size_bytes": size,
        "read_mode": "full",
        "partial": partial,
        "error": "last line is incomplete; showing durable prefix" if partial else None,
    }


def _read_job_status(path: Path) -> JsonObject | None:
    raw, _ = _read_json(path, label="job status", missing_ok=True)
    if raw is None:
        return None
    status = raw.get("status")
    if type(status) is not str or not status:
        raise PilotMonitorError(f"job status is missing or invalid: {path}")
    for key in ("update", "updates"):
        value = raw.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise PilotMonitorError(f"job status {key} is invalid: {path}")
    for key in ("elapsed_seconds", "update_seconds"):
        value = raw.get(key)
        if value is not None:
            if type(value) not in {int, float}:
                raise PilotMonitorError(f"job status {key} is invalid: {path}")
            numeric_value = float(cast(int | float, value))
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise PilotMonitorError(f"job status {key} is invalid: {path}")
    return raw


def _job_directory_state(path: Path) -> str:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        raise PilotMonitorError(f"cannot inspect job output directory {path}: {exc}") from exc
    if path.is_symlink() or not path.is_dir():
        raise PilotMonitorError(f"job output path must be a real directory: {path}")
    del metadata
    return "present"


def _job_snapshot(  # noqa: C901,PLR0912,PLR0915
    declaration: _Declaration,
    replicate_id: int,
    treatment_id: str,
    output_dir: Path,
) -> JsonObject:
    directory_state = _job_directory_state(output_dir)
    status_raw: JsonObject | None = None
    result: JsonObject
    journal: JsonObject
    if directory_state == "present":
        status_raw = _read_job_status(output_dir / JOB_STATUS_NAME)
        result = _result_summary(output_dir / JOB_RESULT_NAME)
        journal_path = output_dir / UPDATE_JOURNAL_NAME
        result_journal_path = result.get("journal_path")
        if type(result_journal_path) is str:
            candidate_journal_path = _canonical_path(
                result_journal_path,
                label="result formal_update_journal.path",
            )
            if candidate_journal_path.parent != output_dir:
                raise PilotMonitorError(
                    "result formal_update_journal.path must be inside its job directory"
                )
            journal_path = candidate_journal_path
        if not os.path.lexists(journal_path):
            legacy_path = output_dir / LEGACY_UPDATE_JOURNAL_NAME
            if os.path.lexists(legacy_path):
                journal_path = legacy_path
        journal = _journal_summary(
            journal_path,
            expected_entries=(
                cast(int, status_raw["journal_entries"])
                if status_raw is not None
                and type(status_raw.get("journal_entries")) is int
                else (
                    cast(int, result["journal_entries"])
                    if type(result.get("journal_entries")) is int
                    else None
                )
            ),
            expected_last_update=(
                cast(int, status_raw["update"])
                if status_raw is not None and type(status_raw.get("update")) is int
                else None
            ),
        )
    else:
        result = _result_summary(output_dir / JOB_RESULT_NAME)
        journal_path = output_dir / UPDATE_JOURNAL_NAME
        if not os.path.lexists(journal_path):
            legacy_path = output_dir / LEGACY_UPDATE_JOURNAL_NAME
            if os.path.lexists(legacy_path):
                journal_path = legacy_path
        journal = _journal_summary(journal_path)
    status_name = status_raw.get("status") if status_raw is not None else "missing"
    target: int | None = None
    if status_raw is not None and type(status_raw.get("updates")) is int:
        target = cast(int, status_raw["updates"])
    if target is None:
        target = declaration.treatment_targets.get(treatment_id)
    if target is None and type(result.get("target_updates")) is int:
        target = cast(int, result["target_updates"])
    current: int | None = None
    if status_raw is not None and type(status_raw.get("update")) is int:
        current = cast(int, status_raw["update"])
    if current is None and type(journal.get("last_update")) is int:
        current = cast(int, journal["last_update"])
    if current is None:
        current = 0
    if status_raw is None:
        classification = "queued"
    elif status_name in _TERMINAL_JOB_STATUSES:
        classification = "terminal"
    elif status_name in _ACTIVE_JOB_STATUSES:
        classification = "active"
    elif status_name in _QUEUED_JOB_STATUSES:
        classification = "queued"
    else:
        classification = "unknown"
    progress = (
        None
        if target is None or target == 0
        else current / cast(int, target)
    )
    issues: list[str] = []
    result_status = result.get("status")
    if status_raw is not None and result.get("present") and result_status != status_name:
        issues.append("status/result status mismatch")
    journal_last = journal.get("last_update")
    if (
        status_raw is not None
        and type(journal_last) is int
        and type(status_raw.get("update")) is int
        and journal_last != status_raw["update"]
    ):
        issues.append("status/journal update mismatch")
    if target is not None and current > target:
        issues.append("update exceeds target")
    if journal.get("error") is not None and journal.get("present"):
        issues.append(str(journal["error"]))
    return {
        "key": f"r{replicate_id}-{treatment_id}",
        "replicate_id": replicate_id,
        "treatment_id": treatment_id,
        "output_dir": str(output_dir),
        "directory": directory_state,
        "status": status_name,
        "classification": classification,
        "update": current,
        "target_updates": target,
        "progress": progress,
        "elapsed_seconds": (
            status_raw.get("elapsed_seconds") if status_raw is not None else None
        ),
        "result": result,
        "journal": journal,
        "issues": issues,
    }


def _tmux_snapshot(session: str) -> JsonObject:
    command = ["tmux", "has-session", "-t", f"={session}"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_TMUX_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return {
            "session": session,
            "available": False,
            "alive": None,
            "error": "tmux executable is unavailable",
        }
    except subprocess.TimeoutExpired:
        return {
            "session": session,
            "available": True,
            "alive": None,
            "error": "tmux liveness query timed out",
        }
    except OSError as exc:
        return {
            "session": session,
            "available": False,
            "alive": None,
            "error": f"tmux liveness query failed: {exc}",
        }
    return {
        "session": session,
        "available": True,
        "alive": completed.returncode == 0,
        "error": None,
    }


def _resource_snapshot(
    declaration: _Declaration,
    orchestrator: Mapping[str, object] | None,
) -> JsonObject:
    resources = orchestrator.get("resources") if orchestrator is not None else None
    output_bytes: int | None = None
    free_bytes: int | None = None
    source = "unavailable"
    if isinstance(resources, Mapping):
        if type(resources.get("output_bytes")) is int:
            output_bytes = cast(int, resources["output_bytes"])
        if type(resources.get("free_bytes")) is int:
            free_bytes = cast(int, resources["free_bytes"])
        source = "orchestrator-status"
    if free_bytes is None:
        ancestor = declaration.output_root
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        try:
            free_bytes = shutil.disk_usage(ancestor).free
            source = "filesystem-fallback" if source == "unavailable" else source
        except OSError:
            free_bytes = None
    return {
        "output_bytes": output_bytes,
        "free_bytes": free_bytes,
        "source": source,
    }


def _risk_budget(
    limits: Mapping[str, object] | None,
    resources: Mapping[str, object],
    elapsed_seconds: float | None,
) -> JsonObject:
    wall_limit = limits.get("wall_seconds") if limits is not None else None
    output_limit = limits.get("max_output_bytes") if limits is not None else None
    free_floor = limits.get("min_free_bytes") if limits is not None else None
    output_bytes = resources.get("output_bytes")
    free_bytes = resources.get("free_bytes")

    def budget(
        limit: object,
        used: object,
        *,
        direction: str,
        unit: str,
    ) -> JsonObject:
        valid_limit = type(limit) is int and cast(int, limit) >= 0
        valid_used = type(used) in {int, float} and float(cast(Any, used)) >= 0
        if not valid_limit or not valid_used:
            return {
                "limit": limit if valid_limit else None,
                "used": used if valid_used else None,
                "headroom": None,
                "unit": unit,
                "status": "unknown",
            }
        limit_number = cast(int, limit)
        used_number = float(cast(Any, used))
        if direction == "minimum":
            headroom = used_number - limit_number
        else:
            headroom = limit_number - used_number
        return {
            "limit": limit_number,
            "used": used_number,
            "headroom": headroom,
            "unit": unit,
            "status": "ok" if headroom >= 0 else "breached",
        }

    return {
        "wall": budget(wall_limit, elapsed_seconds, direction="maximum", unit="seconds"),
        "output": budget(
            output_limit,
            output_bytes,
            direction="maximum",
            unit="bytes",
        ),
        "free": budget(free_floor, free_bytes, direction="minimum", unit="bytes"),
    }


def _completion_verified(
    orchestrator_state: str | None,
    manifest_status: str,
    jobs: Sequence[Mapping[str, object]],
    completion_valid: bool,
) -> bool:
    if (
        orchestrator_state != "completed"
        or manifest_status != "completed"
        or not completion_valid
        or not jobs
    ):
        return False
    for job in jobs:
        if (
            job.get("status") != "completed"
            or job.get("classification") != "terminal"
            or type(job.get("target_updates")) is not int
            or type(job.get("update")) is not int
            or job["update"] != job["target_updates"]
            or bool(job.get("issues"))
        ):
            return False
        result = job.get("result")
        if not isinstance(result, Mapping) or result.get("status") != "completed":
            return False
    return True


def snapshot(declaration_path: Path) -> JsonObject:
    """Read one complete, JSON-serializable pilot progress snapshot."""
    declaration = _load_declaration(declaration_path)
    manifest = _load_manifest(declaration)
    orchestrator = _load_orchestrator_status(declaration)
    completion = _completion_summary(declaration, manifest)
    jobs = [
        _job_snapshot(declaration, replicate, treatment, output_dir)
        for replicate, treatment, output_dir in manifest.jobs
    ]
    counts = {
        "total": len(jobs),
        "active": sum(job["classification"] == "active" for job in jobs),
        "queued": sum(job["classification"] == "queued" for job in jobs),
        "terminal": sum(job["classification"] == "terminal" for job in jobs),
        "unknown": sum(job["classification"] == "unknown" for job in jobs),
    }
    current_updates = sum(
        cast(int, job["update"])
        for job in jobs
        if type(job.get("update")) is int
    )
    known_targets = [
        cast(int, job["target_updates"])
        for job in jobs
        if type(job.get("target_updates")) is int
    ]
    target_updates = sum(known_targets) if len(known_targets) == len(jobs) else None
    progress_fraction = (
        None
        if target_updates in (None, 0)
        else current_updates / cast(int, target_updates)
    )
    status_state = orchestrator.get("state") if orchestrator is not None else None
    status_state_str = status_state if type(status_state) is str else None
    elapsed = (
        orchestrator.get("elapsed_seconds")
        if orchestrator is not None
        else None
    )
    if type(elapsed) not in {int, float} or not math.isfinite(float(cast(Any, elapsed))):
        elapsed_values = [
            cast(float, job["elapsed_seconds"])
            for job in jobs
            if type(job.get("elapsed_seconds")) in {int, float}
        ]
        elapsed = max(elapsed_values) if elapsed_values else None
    elapsed_value = cast(float | None, elapsed)
    resources = _resource_snapshot(declaration, orchestrator)
    limits_mapping: Mapping[str, object] | None = declaration.limits
    risk = _risk_budget(limits_mapping, resources, elapsed_value)
    tmux = _tmux_snapshot(declaration.tmux_session)
    issues = [
        issue
        for job in jobs
        for issue in cast(list[object], job.get("issues", []))
    ]
    if status_state_str == "running" and tmux.get("alive") is False:
        issues.append("orchestrator is running but tmux session is not alive")
    if manifest.status == "completed" and status_state_str != "completed":
        issues.append("manifest is completed without a completed orchestrator status")
    if status_state_str in {"failed", "blocked"} and counts["active"]:
        issues.append("terminal orchestrator has stale active child statuses")
    if completion.get("present") and not completion.get("valid"):
        issues.append("completion evidence is present but invalid")
    if (
        status_state_str == "completed" or manifest.status == "completed"
    ) and not completion.get("valid"):
        issues.append("completed lifecycle lacks valid completion evidence")
    complete = _completion_verified(
        status_state_str,
        manifest.status,
        jobs,
        completion.get("valid") is True,
    )
    state = status_state_str or manifest.status or "not-started"
    return {
        "schema_version": MONITOR_SCHEMA,
        "state": state,
        "complete": complete,
        "declaration": {
            "path": str(declaration.path),
            "sha256": declaration.digest,
            "schema_version": declaration.raw.get("schema_version"),
            "experiment_id": declaration.experiment_id,
            "phase": declaration.phase,
            "manifest_path": str(declaration.manifest_path),
            "control_dir": str(declaration.control_dir),
            "output_root": str(declaration.output_root),
            "tmux_session": declaration.tmux_session,
            "limits": dict(declaration.limits),
        },
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.digest,
            "status": manifest.status,
            "job_count": len(manifest.jobs),
        },
        "orchestrator": {
            "status_path": str(declaration.status_path),
            "present": orchestrator is not None,
            "state": status_state_str,
            "updated_unix_seconds": (
                orchestrator.get("updated_unix_seconds")
                if orchestrator is not None
                else None
            ),
            "elapsed_seconds": elapsed_value,
            "detail": orchestrator.get("detail") if orchestrator is not None else None,
        },
        "completion": completion,
        "tmux": tmux,
        "jobs": jobs,
        "counts": counts,
        "progress": {
            "current_updates": current_updates,
            "target_updates": target_updates,
            "fraction": progress_fraction,
            "percent": None if progress_fraction is None else progress_fraction * 100.0,
            "complete": complete,
        },
        "elapsed_seconds": elapsed_value,
        "output_bytes": resources.get("output_bytes"),
        "free_bytes": resources.get("free_bytes"),
        "resources": resources,
        "risk_budget": risk,
        "issues": sorted({str(issue) for issue in issues}),
    }


# Descriptive aliases make the read-only API convenient for tests and small
# operator scripts without introducing a second implementation.
load_snapshot = snapshot
build_snapshot = snapshot
read_snapshot = snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task1-pilot-monitor")
    parser.add_argument("declaration", type=Path)
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="repeat snapshots after this many seconds until the run is terminal",
    )
    return parser


def _print_snapshot(payload: Mapping[str, object]) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Print one JSON snapshot or poll until the pilot becomes terminal."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.watch is not None and (
        not math.isfinite(args.watch) or args.watch <= 0
    ):
        parser.error("--watch interval must be a finite positive number")
    try:
        while True:
            payload = snapshot(args.declaration)
            _print_snapshot(payload)
            if args.watch is None:
                break
            if payload.get("state") in {"completed", "failed", "blocked"}:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return
    except PilotMonitorError as exc:
        parser.exit(2, f"task1-pilot-monitor: error: {exc}\n")


if __name__ == "__main__":
    main()
