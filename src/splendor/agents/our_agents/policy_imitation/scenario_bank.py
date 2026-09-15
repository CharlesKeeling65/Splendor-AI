"""Content-addressed ScenarioV1 banks, split audits, and sealed access gates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import IO, Any, Final, Literal, cast

from splendor.seed_registry import (
    CI_SMOKE,
    STABILIZATION_TEST,
    TASK1_SCENARIO_SPLITS,
    SeedSegment,
    allocate_seeds,
    registry_sha256,
    resolve_task1_scenario_split,
    validate_registry_disjointness,
)

from .protocol import (
    PairedTrainingRow,
    canonical_json_bytes,
    sha256_canonical_json,
    sha256_file,
)
from .scenario import (
    ScenarioV1,
    ScenarioValidationError,
    SelectionKind,
    generate_scenario,
    scenario_collection_sha256,
    scenario_state_set_sha256,
    validate_scenario,
    with_sampling_metadata,
)
from .seed_roll import (
    RolledScenarioSelection,
    SeedRollArtifact,
    SeedRollError,
    SeedRollSelectionDesign,
    iter_seed_roll_scenarios,
    parse_seed_roll_selection_design,
    seed_roll_selection_design,
    validate_rolled_scenario_selection,
    validate_scenarios_against_seed_roll,
    validate_seed_rolled_training_schedule,
)

BANK_SCHEMA_VERSION: Final = "splendor-scenario-bank/1"
STRESS_SELECTION_SCHEMA_VERSION: Final = "splendor-stress-selection/1"
CANDIDATE_FREEZE_SCHEMA_VERSION: Final = "splendor-candidate-freeze/1"
CONSUMPTION_LEDGER_SCHEMA_VERSION: Final = "splendor-sealed-consumption/1"
FINAL_REPORT_PURPOSE: Final = "final_report"
SHA256_HEX_LENGTH: Final = 64
Compression = Literal["none", "zstd"]

# These two routes exercise ordinary/sealed code in CI without touching any
# Task-1 source that could later enter a formal conclusion.  The sealed route
# deliberately points at an already-consumed historical segment.
_FIXTURE_SPLITS = {
    "ci-fixture": CI_SMOKE,
    "ci-sealed-fixture": STABILIZATION_TEST,
}


class ScenarioBankError(ValueError):
    """Raised when a bank, split, or immutable artifact fails validation."""


@dataclass(frozen=True)
class ScenarioBank:
    """A verified bank loaded from a content-addressed artifact."""

    logical_split: str
    source_segment: str
    selection_kind: str
    scenarios: tuple[ScenarioV1, ...]
    payload_sha256: str
    state_set_sha256: str
    artifact_sha256: str
    artifact_path: Path
    manifest_path: Path
    sealed: bool
    scenario_count: int
    selection_design: StressSelectionDesign | SeedRollSelectionDesign | None


@dataclass(frozen=True)
class StressSelectionDesign:
    """Replayable, pre-keyed stratified lottery declaration."""

    descriptor: str
    bin_edges: tuple[float, ...]
    per_bin: int
    selection_key: str
    candidate_count: int
    candidate_collection_sha256: str
    candidate_state_set_sha256: str
    bin_candidate_counts: tuple[int, ...]
    bin_selected_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STRESS_SELECTION_SCHEMA_VERSION,
            "descriptor": self.descriptor,
            "bin_edges": list(self.bin_edges),
            "per_bin": self.per_bin,
            "selection_key": self.selection_key,
            "candidate_count": self.candidate_count,
            "candidate_collection_sha256": self.candidate_collection_sha256,
            "candidate_state_set_sha256": self.candidate_state_set_sha256,
            "bin_candidate_counts": list(self.bin_candidate_counts),
            "bin_selected_counts": list(self.bin_selected_counts),
            "lottery": "ascending SHA256(protocol, key, design, scenario_id)",
        }

    @property
    def sha256(self) -> str:
        return sha256_canonical_json(self.to_dict())


@dataclass(frozen=True)
class BalancedStressSelection:
    """Selected rows plus the design needed to reproduce their probabilities."""

    scenarios: tuple[ScenarioV1, ...]
    design: StressSelectionDesign


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_bank_split(logical_split: str) -> SeedSegment:
    if logical_split in _FIXTURE_SPLITS:
        return _FIXTURE_SPLITS[logical_split]
    return resolve_task1_scenario_split(logical_split)


def validate_task1_split_registry() -> None:
    """Audit the seven required split names, interval isolation, and sealing."""
    validate_registry_disjointness()
    required = {
        "train-schedule",
        "dagger-rollout",
        "teacher-validation",
        "validation-A",
        "validation-B",
        "sealed-test-iid",
        "stress",
    }
    if set(TASK1_SCENARIO_SPLITS) != required:
        raise ScenarioBankError("Task-1 scenario split registry is incomplete")
    sealed = {name for name, segment in TASK1_SCENARIO_SPLITS.items() if segment.sealed}
    if sealed != {"sealed-test-iid"}:
        raise ScenarioBankError("only sealed-test-iid may be sealed in Task-1")


def generate_iid_scenarios(
    logical_split: str,
    count: int,
    *,
    offset: int = 0,
) -> tuple[ScenarioV1, ...]:
    """Generate a natural-deal prefix from one registered logical split."""
    if count < 1 or offset < 0:
        raise ScenarioBankError(
            "scenario count must be positive and offset non-negative"
        )
    segment = _resolve_bank_split(logical_split)
    if logical_split == "stress":
        raise ScenarioBankError("stress banks require explicit balanced selection")
    capacity = segment.end - segment.start
    if offset + count > capacity:
        raise ScenarioBankError(
            f"split {logical_split!r} cannot supply offset={offset}, count={count}"
        )
    seeds = allocate_seeds(segment, offset + count)[offset:]
    kind: SelectionKind = "ci-fixture" if logical_split in _FIXTURE_SPLITS else "iid"
    return tuple(
        generate_scenario(segment.name, seed, selection_kind=kind) for seed in seeds
    )


def _bin_index(value: float, edges: Sequence[float]) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def _stress_lottery_rank(
    scenario_id: str,
    *,
    design_scope: Mapping[str, object],
) -> str:
    return sha256_canonical_json(
        {
            "protocol": STRESS_SELECTION_SCHEMA_VERSION,
            "design": dict(design_scope),
            "scenario_id": scenario_id,
        }
    )


def select_balanced_stress_scenarios(  # noqa: C901,PLR0912 - full design gate
    candidates: Sequence[ScenarioV1],
    *,
    descriptor: str,
    bin_edges: Sequence[float],
    per_bin: int,
    selection_key: str,
) -> BalancedStressSelection:
    """Select equal counts per bin through a replayable keyed-hash lottery.

    Treating the pre-registered key as the design randomization, every row in
    a bin has first-order inclusion probability ``per_bin / bin_size``.  The
    candidate-pool hashes, key, bin edges and counts are persisted with the
    resulting bank; choosing a key after inspecting ranks is protocol misuse.
    """
    if not candidates or per_bin < 1:
        raise ScenarioBankError("stress selection needs candidates and per_bin > 0")
    if not _is_sha256(selection_key):
        raise ScenarioBankError("stress selection_key must be a pre-registered SHA-256")
    if type(descriptor) is not str or not descriptor:
        raise ScenarioBankError("stress descriptor must be a non-empty string")
    edges = tuple(float(edge) for edge in bin_edges)
    if any(not math.isfinite(edge) for edge in edges) or any(
        left >= right for left, right in pairwise(edges)
    ):
        raise ScenarioBankError("stress bin edges must be finite and increasing")
    # Collection hashing validates every row and refuses repeated states.
    try:
        candidate_collection_hash = scenario_collection_sha256(candidates)
        candidate_state_set_hash = scenario_state_set_sha256(candidates)
    except ScenarioValidationError as exc:
        raise ScenarioBankError(f"invalid stress candidate pool: {exc}") from exc
    source_seeds = [scenario.source_seed for scenario in candidates]
    if len(source_seeds) != len(set(source_seeds)):
        raise ScenarioBankError("stress candidate pool repeats a source seed")

    groups: dict[int, list[ScenarioV1]] = defaultdict(list)
    for scenario in candidates:
        if scenario.source_segment not in {
            TASK1_SCENARIO_SPLITS["stress"].name,
            CI_SMOKE.name,
        }:
            raise ScenarioBankError("stress candidates must use the stress segment")
        if scenario.selection_kind not in {"iid", "ci-fixture"}:
            raise ScenarioBankError("stress candidates must be unselected source rows")
        strata = dict(scenario.strata_v1)
        if descriptor not in strata:
            raise ScenarioBankError(f"unknown stress descriptor {descriptor!r}")
        groups[_bin_index(float(strata[descriptor]), edges)].append(scenario)
    expected_bins = set(range(len(edges) + 1))
    if set(groups) != expected_bins:
        raise ScenarioBankError("stress candidate pool does not cover every bin")
    bin_candidate_counts = tuple(len(groups[index]) for index in sorted(groups))
    design_scope: dict[str, object] = {
        "selection_key": selection_key,
        "descriptor": descriptor,
        "bin_edges": list(edges),
        "per_bin": per_bin,
        "candidate_collection_sha256": candidate_collection_hash,
        "candidate_state_set_sha256": candidate_state_set_hash,
        "bin_candidate_counts": list(bin_candidate_counts),
    }
    design = StressSelectionDesign(
        descriptor=descriptor,
        bin_edges=edges,
        per_bin=per_bin,
        selection_key=selection_key,
        candidate_count=len(candidates),
        candidate_collection_sha256=candidate_collection_hash,
        candidate_state_set_sha256=candidate_state_set_hash,
        bin_candidate_counts=bin_candidate_counts,
        bin_selected_counts=tuple(per_bin for _ in sorted(groups)),
    )
    selected: list[ScenarioV1] = []
    for index in sorted(groups):
        group = sorted(
            groups[index],
            key=lambda scenario: (
                _stress_lottery_rank(
                    scenario.scenario_id,
                    design_scope=design_scope,
                ),
                scenario.scenario_id,
            ),
        )
        if len(group) < per_bin:
            raise ScenarioBankError(
                f"stress bin {index} has {len(group)} candidates, needs {per_bin}"
            )
        probability = per_bin / len(group)
        selected.extend(
            with_sampling_metadata(
                scenario,
                selection_kind="stress-balanced",
                inclusion_probability=probability,
                selection_stratum=f"{descriptor}:bin-{index}",
                selection_design_sha256=design.sha256,
            )
            for scenario in group[:per_bin]
        )
    return BalancedStressSelection(
        scenarios=tuple(sorted(selected, key=lambda scenario: scenario.source_seed)),
        design=design,
    )


def _parse_stress_selection_design(  # noqa: C901,PLR0912 - strict schema parser
    raw: object,
) -> StressSelectionDesign:
    if not isinstance(raw, Mapping):
        raise ScenarioBankError("stress bank requires a selection design")
    required = {
        "schema_version",
        "descriptor",
        "bin_edges",
        "per_bin",
        "selection_key",
        "candidate_count",
        "candidate_collection_sha256",
        "candidate_state_set_sha256",
        "bin_candidate_counts",
        "bin_selected_counts",
        "lottery",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != STRESS_SELECTION_SCHEMA_VERSION
    ):
        raise ScenarioBankError("stress selection design schema mismatch")
    descriptor = raw.get("descriptor")
    if type(descriptor) is not str or not descriptor:
        raise ScenarioBankError("stress selection descriptor is invalid")
    edges_raw = raw.get("bin_edges")
    if not isinstance(edges_raw, list) or any(
        type(edge) is not float for edge in edges_raw
    ):
        raise ScenarioBankError("stress selection bin edges must be float values")
    edges = tuple(cast(list[float], edges_raw))
    if any(not math.isfinite(edge) for edge in edges) or any(
        left >= right for left, right in pairwise(edges)
    ):
        raise ScenarioBankError("stress selection bin edges are invalid")
    per_bin = raw.get("per_bin")
    candidate_count = raw.get("candidate_count")
    if type(per_bin) is not int or per_bin < 1:
        raise ScenarioBankError("stress selection per_bin is invalid")
    if type(candidate_count) is not int or candidate_count < 1:
        raise ScenarioBankError("stress selection candidate_count is invalid")
    if not _is_sha256(raw.get("selection_key")):
        raise ScenarioBankError("stress selection key is invalid")
    if not _is_sha256(raw.get("candidate_collection_sha256")) or not _is_sha256(
        raw.get("candidate_state_set_sha256")
    ):
        raise ScenarioBankError("stress selection candidate-pool hash is invalid")

    def integer_counts(field: str) -> tuple[int, ...]:
        value = raw.get(field)
        if not isinstance(value, list) or any(
            type(count) is not int or count < 1 for count in value
        ):
            raise ScenarioBankError(f"stress selection {field} is invalid")
        return tuple(cast(list[int], value))

    candidate_counts = integer_counts("bin_candidate_counts")
    selected_counts = integer_counts("bin_selected_counts")
    bin_count = len(edges) + 1
    if len(candidate_counts) != bin_count or len(selected_counts) != bin_count:
        raise ScenarioBankError("stress selection bin-count shape mismatch")
    if any(count < per_bin for count in candidate_counts) or any(
        count != per_bin for count in selected_counts
    ):
        raise ScenarioBankError("stress selection per-bin counts are inconsistent")
    if sum(candidate_counts) != candidate_count:
        raise ScenarioBankError("stress selection candidate counts do not add up")
    if raw.get("lottery") != "ascending SHA256(protocol, key, design, scenario_id)":
        raise ScenarioBankError("stress selection lottery protocol is invalid")
    return StressSelectionDesign(
        descriptor=descriptor,
        bin_edges=edges,
        per_bin=per_bin,
        selection_key=cast(str, raw["selection_key"]),
        candidate_count=candidate_count,
        candidate_collection_sha256=cast(str, raw["candidate_collection_sha256"]),
        candidate_state_set_sha256=cast(str, raw["candidate_state_set_sha256"]),
        bin_candidate_counts=candidate_counts,
        bin_selected_counts=selected_counts,
    )


def _validate_stress_selection_rows(
    scenarios: Sequence[ScenarioV1],
    design: StressSelectionDesign,
) -> None:
    expected_design_hash = design.sha256
    observed_counts = [0] * (len(design.bin_edges) + 1)
    for scenario in scenarios:
        if scenario.selection_design_sha256 != expected_design_hash:
            raise ScenarioBankError("stress row selection-design hash mismatch")
        value = dict(scenario.strata_v1).get(design.descriptor)
        if value is None:
            raise ScenarioBankError("stress row omits its selection descriptor")
        bin_index = _bin_index(float(value), design.bin_edges)
        if scenario.selection_stratum != f"{design.descriptor}:bin-{bin_index}":
            raise ScenarioBankError("stress row selection stratum mismatch")
        expected_probability = design.per_bin / design.bin_candidate_counts[bin_index]
        if scenario.inclusion_probability != expected_probability:
            raise ScenarioBankError("stress row inclusion probability mismatch")
        observed_counts[bin_index] += 1
    if tuple(observed_counts) != design.bin_selected_counts:
        raise ScenarioBankError("stress bank selected counts do not match its design")


def validate_split_disjointness(
    banks: Mapping[str, Sequence[ScenarioV1]],
) -> None:
    """Reject source-seed or state identity reuse within/across logical splits."""
    if not banks:
        raise ScenarioBankError("split audit requires at least one bank")
    seed_owner: dict[int, str] = {}
    id_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    for logical_split, scenarios in banks.items():
        if not scenarios:
            raise ScenarioBankError(f"split {logical_split!r} is empty")
        expected_segment = _resolve_bank_split(logical_split)
        for scenario in scenarios:
            validate_scenario(scenario)
            if scenario.source_segment != expected_segment.name:
                raise ScenarioBankError(
                    f"split {logical_split!r} uses source segment "
                    f"{scenario.source_segment!r}, expected {expected_segment.name!r}"
                )
            seed_key = scenario.source_seed
            if seed_key in seed_owner:
                raise ScenarioBankError(
                    f"source seed {seed_key} is duplicated across scenario splits"
                )
            seed_owner[seed_key] = logical_split
            if scenario.scenario_id in id_owner:
                raise ScenarioBankError(
                    f"scenario_id {scenario.scenario_id} is duplicated across splits"
                )
            id_owner[scenario.scenario_id] = logical_split
            if scenario.canonical_state_sha256 in hash_owner:
                raise ScenarioBankError("canonical state is duplicated across splits")
            hash_owner[scenario.canonical_state_sha256] = logical_split


def _payload_bytes(scenarios: Sequence[ScenarioV1]) -> bytes:
    ordered = sorted(scenarios, key=lambda scenario: scenario.source_seed)
    return b"".join(
        canonical_json_bytes(scenario.to_dict()) + b"\n" for scenario in ordered
    )


def _compress_zstd(payload: bytes) -> bytes:
    executable = shutil.which("zstd")
    if executable is None:
        raise ScenarioBankError("writing .jsonl.zst banks requires the zstd executable")
    result = subprocess.run(
        [executable, "--quiet", "--compress", "--stdout", "--threads=1", "-19"],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ScenarioBankError(f"zstd compression failed: {message}")
    return result.stdout


@contextmanager
def _open_payload_stream(
    artifact_path: Path,
    compression: object,
) -> Iterator[IO[bytes]]:
    """Yield decompressed bytes without buffering the whole bank in memory."""
    if compression == "none":
        with artifact_path.open("rb") as stream:
            yield stream
        return
    if compression != "zstd":
        raise ScenarioBankError("scenario bank compression is invalid")
    executable = shutil.which("zstd")
    if executable is None:
        raise ScenarioBankError("reading .jsonl.zst banks requires the zstd executable")
    process = subprocess.Popen(
        [
            executable,
            "--quiet",
            "--decompress",
            "--stdout",
            "--",
            str(artifact_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        yield process.stdout
    except BaseException:
        process.stdout.close()
        process.kill()
        process.wait()
        process.stderr.close()
        raise
    else:
        process.stdout.close()
        error = process.stderr.read()
        return_code = process.wait()
        process.stderr.close()
        if return_code != 0:
            message = error.decode("utf-8", errors="replace").strip()
            raise ScenarioBankError(f"zstd decompression failed: {message}")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _bank_manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.name}.manifest.json")


def write_scenario_bank(  # noqa: C901,PLR0912,PLR0915 - selection/artifact gates
    output_dir: Path,
    logical_split: str,
    scenarios: Sequence[ScenarioV1] | BalancedStressSelection | RolledScenarioSelection,
    *,
    compression: Compression = "zstd",
) -> dict[str, object]:
    """Write a never-overwritten bank and sidecar under content-derived names."""
    stress_selection: BalancedStressSelection | None
    rolled_selection: RolledScenarioSelection | None
    scenario_rows: Sequence[ScenarioV1]
    if isinstance(scenarios, BalancedStressSelection):
        stress_selection = scenarios
        rolled_selection = None
        scenario_rows = scenarios.scenarios
    elif isinstance(scenarios, RolledScenarioSelection):
        stress_selection = None
        rolled_selection = scenarios
        scenario_rows = scenarios.scenarios
    else:
        stress_selection = None
        rolled_selection = None
        scenario_rows = scenarios
    if not scenario_rows:
        raise ScenarioBankError("cannot write an empty scenario bank")
    validate_split_disjointness({logical_split: scenario_rows})
    segment = _resolve_bank_split(logical_split)
    kinds = {scenario.selection_kind for scenario in scenario_rows}
    if len(kinds) != 1:
        raise ScenarioBankError("one bank cannot mix selection kinds")
    selection_kind = next(iter(kinds))
    if logical_split == "stress" and selection_kind != "stress-balanced":
        raise ScenarioBankError("stress bank must preserve balanced-selection weights")
    if logical_split == "sealed-test-iid" and selection_kind != "iid":
        raise ScenarioBankError("sealed-test-iid must be a natural-deal IID bank")
    selection_design: dict[str, object] | None = None
    selection_design_sha256: str | None = None
    if selection_kind == "stress-balanced":
        if stress_selection is None:
            raise ScenarioBankError(
                "stress-balanced banks require their complete selection design"
            )
        _validate_stress_selection_rows(scenario_rows, stress_selection.design)
        selection_design = stress_selection.design.to_dict()
        selection_design_sha256 = stress_selection.design.sha256
    elif selection_kind == "natural-deal-srswor":
        if rolled_selection is None:
            raise ScenarioBankError(
                "natural-deal SRSWOR banks require their complete seed-roll selection design"
            )
        try:
            validate_rolled_scenario_selection(
                scenario_rows,
                rolled_selection.design,
            )
        except SeedRollError as exc:
            raise ScenarioBankError(
                f"invalid natural-deal SRSWOR selection: {exc}"
            ) from exc
        selection_design = rolled_selection.design.to_dict()
        selection_design_sha256 = rolled_selection.design.sha256
    elif stress_selection is not None:
        raise ScenarioBankError("stress selection rows have the wrong selection kind")
    elif rolled_selection is not None:
        raise ScenarioBankError("rolled selection rows have the wrong selection kind")

    raw_payload = _payload_bytes(scenario_rows)
    payload_sha256 = hashlib.sha256(raw_payload).hexdigest()
    if compression == "zstd":
        artifact_payload = _compress_zstd(raw_payload)
        suffix = ".jsonl.zst"
    elif compression == "none":
        artifact_payload = raw_payload
        suffix = ".jsonl"
    else:  # pragma: no cover - Literal callers, runtime corruption guard
        raise ScenarioBankError(f"unsupported bank compression {compression!r}")
    artifact_sha256 = hashlib.sha256(artifact_payload).hexdigest()
    artifact_path = Path(output_dir) / f"{logical_split}-{payload_sha256}{suffix}"
    manifest_path = _bank_manifest_path(artifact_path)
    manifest: dict[str, object] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "logical_split": logical_split,
        "source_segment": segment.name,
        "source_range": [segment.start, segment.end],
        "sealed": segment.sealed,
        "selection_kind": selection_kind,
        "scenario_count": len(scenario_rows),
        "scenario_collection_sha256": scenario_collection_sha256(scenario_rows),
        "state_set_sha256": scenario_state_set_sha256(scenario_rows),
        "payload_sha256": payload_sha256,
        "artifact_sha256": artifact_sha256,
        "compression": compression,
        "artifact": artifact_path.name,
        "seed_registry_sha256": registry_sha256(),
        "selection_design": selection_design,
        "selection_design_sha256": selection_design_sha256,
    }
    _write_exclusive(artifact_path, artifact_payload)
    try:
        _write_exclusive(
            manifest_path,
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )
    except Exception:
        # The content-addressed data file is intentionally left in place.  It
        # must never be silently overwritten; an operator can inspect and
        # either complete the missing sidecar or choose a new output directory.
        raise
    return {
        **manifest,
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path),
    }


def _publish_temporary_file(source: Path, destination: Path) -> None:
    """Atomically publish same-filesystem bytes without overwriting a bank."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise ScenarioBankError(
            f"scenario bank artifact already exists: {destination}"
        ) from exc
    destination.chmod(0o444)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _compress_zstd_file(source: Path, destination: Path) -> None:
    executable = shutil.which("zstd")
    if executable is None:
        raise ScenarioBankError("writing .jsonl.zst banks requires the zstd executable")
    with source.open("rb") as source_stream, destination.open("xb") as output_stream:
        result = subprocess.run(
            [executable, "--quiet", "--compress", "--stdout", "--threads=1", "-19"],
            stdin=source_stream,
            stdout=output_stream,
            stderr=subprocess.PIPE,
            check=False,
        )
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ScenarioBankError(f"zstd compression failed: {message}")


def write_seed_roll_scenario_bank(  # noqa: PLR0915 - streaming lifecycle is explicit
    output_dir: Path,
    artifact: SeedRollArtifact,
    active_replicate_ids: Sequence[int],
    *,
    compression: Compression = "zstd",
) -> dict[str, object]:
    """Stream a root-derived training bank without retaining every ScenarioV1."""
    if compression not in {"none", "zstd"}:
        raise ScenarioBankError(f"unsupported bank compression {compression!r}")
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        design = seed_roll_selection_design(artifact, active_replicate_ids)
    except SeedRollError as exc:
        raise ScenarioBankError(f"invalid seed-roll authority: {exc}") from exc
    payload_digest = hashlib.sha256()
    row_hashes: list[tuple[str, str]] = []
    state_ids: list[str] = []
    seen_state_ids: set[str] = set()
    counts: Counter[str] = Counter()
    previous_source_seed: int | None = None
    raw_temporary: Path | None = None
    artifact_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination_dir,
            prefix=".rolled-scenarios.",
            suffix=".jsonl.tmp",
            delete=False,
        ) as stream:
            raw_temporary = Path(stream.name)
            for scenario in iter_seed_roll_scenarios(artifact, design):
                validate_scenario(scenario)
                if (
                    previous_source_seed is not None
                    and scenario.source_seed <= previous_source_seed
                ):
                    raise ScenarioBankError(
                        "seed-roll scenario stream is not in canonical source order"
                    )
                previous_source_seed = scenario.source_seed
                if scenario.scenario_id in seen_state_ids:
                    raise ScenarioBankError("seed-roll scenario stream repeats a state")
                assert scenario.selection_stratum is not None
                counts[scenario.selection_stratum] += 1
                line = canonical_json_bytes(scenario.to_dict()) + b"\n"
                stream.write(line)
                payload_digest.update(line)
                row_hashes.append(
                    (scenario.scenario_id, hashlib.sha256(line[:-1]).hexdigest())
                )
                state_ids.append(scenario.scenario_id)
                seen_state_ids.add(scenario.scenario_id)
            stream.flush()
            os.fsync(stream.fileno())
        expected_counts = {
            f"replicate-{replicate_id}": design.games_per_replicate
            for replicate_id in design.active_replicate_ids
        }
        if len(state_ids) != design.selected_count or dict(counts) != expected_counts:
            raise ScenarioBankError(
                "seed-roll scenario stream does not match its replicate design"
            )
        payload_sha256 = payload_digest.hexdigest()
        if compression == "zstd":
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination_dir,
                prefix=".rolled-scenarios.",
                suffix=".jsonl.zst.tmp",
            )
            os.close(descriptor)
            artifact_temporary = Path(temporary_name)
            artifact_temporary.unlink()
            _compress_zstd_file(raw_temporary, artifact_temporary)
            suffix = ".jsonl.zst"
        else:
            artifact_temporary = raw_temporary
            suffix = ".jsonl"
        artifact_sha256 = sha256_file(artifact_temporary)
        artifact_path = destination_dir / f"train-schedule-{payload_sha256}{suffix}"
        manifest_path = _bank_manifest_path(artifact_path)
        manifest: dict[str, object] = {
            "schema_version": BANK_SCHEMA_VERSION,
            "logical_split": "train-schedule",
            "source_segment": TASK1_SCENARIO_SPLITS["train-schedule"].name,
            "source_range": [
                TASK1_SCENARIO_SPLITS["train-schedule"].start,
                TASK1_SCENARIO_SPLITS["train-schedule"].end,
            ],
            "sealed": False,
            "selection_kind": "natural-deal-srswor",
            "scenario_count": len(state_ids),
            "scenario_collection_sha256": sha256_canonical_json(sorted(row_hashes)),
            "state_set_sha256": sha256_canonical_json(sorted(state_ids)),
            "payload_sha256": payload_sha256,
            "artifact_sha256": artifact_sha256,
            "compression": compression,
            "artifact": artifact_path.name,
            "seed_registry_sha256": registry_sha256(),
            "selection_design": design.to_dict(),
            "selection_design_sha256": design.sha256,
        }
        _publish_temporary_file(artifact_temporary, artifact_path)
        _write_exclusive(
            manifest_path,
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )
        return {
            **manifest,
            "artifact_path": str(artifact_path),
            "manifest_path": str(manifest_path),
        }
    finally:
        for temporary in (raw_temporary, artifact_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _read_bank_manifest(  # noqa: C901,PLR0912,PLR0915 - immutable field gate
    artifact_path: Path,
) -> dict[str, object]:
    manifest_path = _bank_manifest_path(artifact_path)
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioBankError(f"cannot read scenario bank manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioBankError("scenario bank manifest root must be an object")
    required = {
        "schema_version",
        "logical_split",
        "source_segment",
        "source_range",
        "sealed",
        "selection_kind",
        "scenario_count",
        "scenario_collection_sha256",
        "state_set_sha256",
        "payload_sha256",
        "artifact_sha256",
        "compression",
        "artifact",
        "seed_registry_sha256",
        "selection_design",
        "selection_design_sha256",
    }
    if set(raw) != required or raw.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ScenarioBankError("scenario bank manifest schema mismatch")
    if raw.get("artifact") != artifact_path.name:
        raise ScenarioBankError("scenario bank manifest points to another artifact")
    if raw.get("seed_registry_sha256") != registry_sha256():
        raise ScenarioBankError("scenario bank seed-registry hash mismatch")
    for field in (
        "scenario_collection_sha256",
        "state_set_sha256",
        "payload_sha256",
        "artifact_sha256",
    ):
        if not _is_sha256(raw.get(field)):
            raise ScenarioBankError(f"scenario bank {field} is invalid")
    logical_split = raw.get("logical_split")
    if type(logical_split) is not str:
        raise ScenarioBankError("scenario bank logical_split is invalid")
    segment = _resolve_bank_split(logical_split)
    if raw.get("source_segment") != segment.name or raw.get("source_range") != [
        segment.start,
        segment.end,
    ]:
        raise ScenarioBankError("scenario bank source segment metadata mismatch")
    if raw.get("sealed") is not segment.sealed:
        raise ScenarioBankError("scenario bank sealed metadata mismatch")
    if (
        type(raw.get("scenario_count")) is not int
        or cast(int, raw["scenario_count"]) < 1
    ):
        raise ScenarioBankError("scenario bank scenario_count is invalid")
    compression = raw.get("compression")
    if compression not in {"none", "zstd"}:
        raise ScenarioBankError("scenario bank compression is invalid")
    expected_suffix = ".jsonl.zst" if compression == "zstd" else ".jsonl"
    if not artifact_path.name.endswith(expected_suffix):
        raise ScenarioBankError("scenario bank compression suffix mismatch")
    expected_name = f"{logical_split}-{raw['payload_sha256']}{expected_suffix}"
    if artifact_path.name != expected_name:
        raise ScenarioBankError(
            "scenario bank filename does not match its content address"
        )
    selection_kind = raw.get("selection_kind")
    if type(selection_kind) is not str:
        raise ScenarioBankError("scenario bank selection_kind is invalid")
    if selection_kind == "stress-balanced":
        stress_design = _parse_stress_selection_design(raw.get("selection_design"))
        if raw.get("selection_design_sha256") != stress_design.sha256:
            raise ScenarioBankError("stress selection-design SHA-256 mismatch")
    elif selection_kind == "natural-deal-srswor":
        try:
            rolled_design = parse_seed_roll_selection_design(
                raw.get("selection_design")
            )
        except SeedRollError as exc:
            raise ScenarioBankError(
                f"invalid natural-deal SRSWOR selection design: {exc}"
            ) from exc
        if raw.get("selection_design_sha256") != rolled_design.sha256:
            raise ScenarioBankError(
                "natural-deal SRSWOR selection-design SHA-256 mismatch"
            )
        if raw.get("scenario_count") != rolled_design.selected_count:
            raise ScenarioBankError(
                "natural-deal SRSWOR scenario count/design mismatch"
            )
    elif (
        raw.get("selection_design") is not None
        or raw.get("selection_design_sha256") is not None
    ):
        raise ScenarioBankError("non-stress bank cannot declare a selection design")
    return cast(dict[str, object], raw)


def inspect_scenario_bank(artifact_path: Path) -> dict[str, object]:
    """Verify public metadata and artifact bytes without exposing sealed rows."""
    path = Path(artifact_path)
    manifest = _read_bank_manifest(path)
    if sha256_file(path) != manifest["artifact_sha256"]:
        raise ScenarioBankError("scenario bank artifact SHA-256 mismatch")
    return manifest


def _load_scenario_bank(  # noqa: C901,PLR0912,PLR0915 - every layer is verified
    artifact_path: Path,
    *,
    allow_sealed: bool,
    scenario_ids: Collection[str] | None = None,
) -> ScenarioBank:
    path = Path(artifact_path)
    manifest = inspect_scenario_bank(path)
    sealed = manifest["sealed"]
    if type(sealed) is not bool:
        raise ScenarioBankError("scenario bank sealed flag is invalid")
    if sealed and not allow_sealed:
        raise ScenarioBankError(
            "sealed scenario rows require the final-report consumption gate"
        )
    requested_values = None if scenario_ids is None else tuple(scenario_ids)
    requested_ids = None if requested_values is None else frozenset(requested_values)
    if requested_ids is not None and (
        not requested_ids
        or len(requested_ids) != len(requested_values or ())
        or any(not _is_sha256(value) for value in requested_ids)
    ):
        raise ScenarioBankError(
            "scenario subset IDs must be unique non-empty SHA-256 values"
        )
    if requested_ids is not None and manifest["selection_kind"] == "stress-balanced":
        raise ScenarioBankError(
            "stress banks must be loaded with their complete design"
        )

    payload_digest = hashlib.sha256()
    scenarios: list[ScenarioV1] = []
    logical_split = str(manifest["logical_split"])
    expected_segment = _resolve_bank_split(logical_split)
    row_hashes: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    rolled_design = (
        parse_seed_roll_selection_design(manifest["selection_design"])
        if manifest["selection_kind"] == "natural-deal-srswor"
        else None
    )
    rolled_counts: Counter[str] = Counter()
    rolled_strata = (
        {
            f"replicate-{replicate_id}"
            for replicate_id in rolled_design.active_replicate_ids
        }
        if rolled_design is not None
        else set()
    )
    previous_source_seed: int | None = None
    row_count = 0
    with _open_payload_stream(path, manifest["compression"]) as stream:
        for line_number, line in enumerate(stream, start=1):
            payload_digest.update(line)
            try:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ScenarioValidationError("scenario row must be an object")
                scenario = ScenarioV1.from_dict(raw)
            except (json.JSONDecodeError, ScenarioValidationError) as exc:
                raise ScenarioBankError(
                    f"invalid scenario bank row {line_number}: {exc}"
                ) from exc
            canonical_line = canonical_json_bytes(scenario.to_dict()) + b"\n"
            if line != canonical_line:
                raise ScenarioBankError(
                    "scenario bank payload is not canonical JSONL/source-seed order"
                )
            if (
                previous_source_seed is not None
                and scenario.source_seed <= previous_source_seed
            ):
                raise ScenarioBankError(
                    "scenario bank payload is not canonical JSONL/source-seed order"
                )
            previous_source_seed = scenario.source_seed
            if scenario.source_segment != expected_segment.name:
                raise ScenarioBankError(
                    f"split {logical_split!r} uses source segment "
                    f"{scenario.source_segment!r}, expected {expected_segment.name!r}"
                )
            if scenario.scenario_id in seen_ids:
                raise ScenarioBankError("scenario bank contains a duplicate state")
            seen_ids.add(scenario.scenario_id)
            if scenario.selection_kind != manifest["selection_kind"]:
                raise ScenarioBankError("scenario bank selection-kind mismatch")
            if rolled_design is not None:
                if (
                    scenario.selection_design_sha256 != rolled_design.sha256
                    or scenario.inclusion_probability
                    != rolled_design.inclusion_probability
                    or scenario.selection_stratum not in rolled_strata
                ):
                    raise ScenarioBankError(
                        "natural-deal SRSWOR scenario row does not match its design"
                    )
                assert scenario.selection_stratum is not None
                rolled_counts[scenario.selection_stratum] += 1
            row_hashes.append(
                (
                    scenario.scenario_id,
                    hashlib.sha256(canonical_line[:-1]).hexdigest(),
                )
            )
            row_count += 1
            if requested_ids is None or scenario.scenario_id in requested_ids:
                scenarios.append(scenario)

    if payload_digest.hexdigest() != manifest["payload_sha256"]:
        raise ScenarioBankError("scenario bank payload SHA-256 mismatch")
    if row_count != manifest["scenario_count"]:
        raise ScenarioBankError("scenario bank row count mismatch")
    if rolled_design is not None and dict(rolled_counts) != dict.fromkeys(
        rolled_strata,
        rolled_design.games_per_replicate,
    ):
        raise ScenarioBankError("natural-deal SRSWOR bank replicate counts mismatch")
    if (
        sha256_canonical_json(sorted(row_hashes))
        != manifest["scenario_collection_sha256"]
    ):
        raise ScenarioBankError("scenario collection SHA-256 mismatch")
    if sha256_canonical_json(sorted(seen_ids)) != manifest["state_set_sha256"]:
        raise ScenarioBankError("scenario state-set SHA-256 mismatch")
    if requested_ids is not None:
        missing = sorted(requested_ids - seen_ids)
        if missing:
            raise ScenarioBankError(
                f"scenario bank is missing requested states: {missing[:3]}"
            )
    selection_design: StressSelectionDesign | SeedRollSelectionDesign | None = None
    if manifest["selection_kind"] == "stress-balanced":
        design = _parse_stress_selection_design(manifest["selection_design"])
        _validate_stress_selection_rows(scenarios, design)
        selection_design = design
    elif manifest["selection_kind"] == "natural-deal-srswor":
        assert rolled_design is not None
        selection_design = rolled_design
    return ScenarioBank(
        logical_split=logical_split,
        source_segment=str(manifest["source_segment"]),
        selection_kind=str(manifest["selection_kind"]),
        scenarios=tuple(scenarios),
        payload_sha256=str(manifest["payload_sha256"]),
        state_set_sha256=str(manifest["state_set_sha256"]),
        artifact_sha256=str(manifest["artifact_sha256"]),
        artifact_path=path,
        manifest_path=_bank_manifest_path(path),
        sealed=sealed,
        scenario_count=row_count,
        selection_design=selection_design,
    )


def load_scenario_bank(artifact_path: Path) -> ScenarioBank:
    """Load an ordinary bank; sealed row access always fails closed."""
    return _load_scenario_bank(artifact_path, allow_sealed=False)


def load_scenario_bank_subset(
    artifact_path: Path,
    scenario_ids: Collection[str],
) -> ScenarioBank:
    """Verify a whole ordinary bank while retaining only requested scenarios."""
    return _load_scenario_bank(
        artifact_path,
        allow_sealed=False,
        scenario_ids=scenario_ids,
    )


def audit_seed_roll_scenario_bank(
    artifact_path: Path,
    seed_roll: SeedRollArtifact,
    active_replicate_ids: Sequence[int],
    *,
    schedule: Sequence[PairedTrainingRow] | None = None,
    expected_treatments: Sequence[str] = (),
) -> dict[str, object]:
    """Replay every rolled bank row against its registered source before training.

    Formal training performs this complete parent-side audit before spawning any
    optimizer.  Worker subset loading remains an additional per-job check; it is
    not the first time an unused row is authenticated.
    """
    bank = load_scenario_bank(artifact_path)
    if not isinstance(bank.selection_design, SeedRollSelectionDesign):
        raise ScenarioBankError(
            "formal seed-roll audit requires a natural-deal SRSWOR bank"
        )
    if schedule is None and expected_treatments:
        raise ScenarioBankError(
            "seed-roll bank audit received treatments without a schedule"
        )
    if schedule is not None and not expected_treatments:
        raise ScenarioBankError("seed-roll schedule audit requires expected treatments")
    try:
        if schedule is None:
            validate_scenarios_against_seed_roll(
                seed_roll,
                bank.scenarios,
                active_replicate_ids,
                selection_design=bank.selection_design,
            )
        else:
            validate_seed_rolled_training_schedule(
                seed_roll,
                bank.scenarios,
                schedule,
                active_replicate_ids,
                expected_treatments=expected_treatments,
                selection_design=bank.selection_design,
            )
    except SeedRollError as exc:
        raise ScenarioBankError(f"seed-roll bank replay failed: {exc}") from exc
    return {
        "payload_sha256": bank.payload_sha256,
        "artifact_sha256": bank.artifact_sha256,
        "state_set_sha256": bank.state_set_sha256,
        "scenario_count": bank.scenario_count,
        "selection_design_sha256": bank.selection_design.sha256,
    }


def freeze_candidate_set(
    path: Path,
    candidates: Mapping[str, str],
    *,
    manifest_declaration_sha256: str,
    frozen_by: str,
) -> dict[str, object]:
    """Write an exclusive candidate-set freeze bound to an approved declaration."""
    if not candidates or not frozen_by.strip():
        raise ScenarioBankError("candidate freeze requires candidates and an actor")
    if not _is_sha256(manifest_declaration_sha256):
        raise ScenarioBankError("candidate freeze manifest hash is invalid")
    ordered = sorted(candidates.items())
    if any(not name or not _is_sha256(digest) for name, digest in ordered):
        raise ScenarioBankError("candidate freeze contains an invalid name or SHA-256")
    checkpoint_hashes = [digest for _name, digest in ordered]
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise ScenarioBankError(
            "candidate freeze must deduplicate checkpoints by SHA-256"
        )
    candidate_rows = [
        {"name": name, "checkpoint_sha256": digest} for name, digest in ordered
    ]
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_FREEZE_SCHEMA_VERSION,
        "manifest_declaration_sha256": manifest_declaration_sha256,
        "candidates": candidate_rows,
        "candidate_set_sha256": sha256_canonical_json(candidate_rows),
        "frozen_by": frozen_by,
        "frozen_at": _now(),
    }
    _write_exclusive(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return payload


def _load_candidate_freeze(  # noqa: C901 - strict frozen schema parser
    path: Path,
) -> dict[str, object]:
    try:
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioBankError(f"cannot read candidate freeze: {exc}") from exc
    required = {
        "schema_version",
        "manifest_declaration_sha256",
        "candidates",
        "candidate_set_sha256",
        "frozen_by",
        "frozen_at",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw.get("schema_version") != CANDIDATE_FREEZE_SCHEMA_VERSION
    ):
        raise ScenarioBankError("candidate freeze schema mismatch")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ScenarioBankError("candidate freeze has no candidates")
    if raw.get("candidate_set_sha256") != sha256_canonical_json(candidates):
        raise ScenarioBankError("candidate freeze SHA-256 mismatch")
    if not _is_sha256(raw.get("manifest_declaration_sha256")):
        raise ScenarioBankError("candidate freeze manifest hash is invalid")
    names: list[str] = []
    checkpoint_hashes: list[str] = []
    for row in candidates:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "checkpoint_sha256"}
            or type(row["name"]) is not str
            or not row["name"]
            or not _is_sha256(row["checkpoint_sha256"])
        ):
            raise ScenarioBankError("candidate freeze contains an invalid row")
        names.append(cast(str, row["name"]))
        checkpoint_hashes.append(cast(str, row["checkpoint_sha256"]))
    if names != sorted(set(names)):
        raise ScenarioBankError("candidate freeze names must be sorted and unique")
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise ScenarioBankError(
            "candidate freeze must deduplicate checkpoints by SHA-256"
        )
    if type(raw.get("frozen_by")) is not str or not str(raw["frozen_by"]).strip():
        raise ScenarioBankError("candidate freeze actor is invalid")
    if type(raw.get("frozen_at")) is not str or not raw["frozen_at"]:
        raise ScenarioBankError("candidate freeze timestamp is invalid")
    return cast(dict[str, object], raw)


def _read_ledger(  # noqa: C901,PLR0912,PLR0915 - strict event/state parser
    stream: IO[str],
) -> tuple[list[dict[str, object]], str | None]:
    stream.seek(0)
    events: list[dict[str, object]] = []
    previous_hash: str | None = None
    attempts: dict[str, dict[str, object]] = {}
    common_fields = {
        "bank_payload_sha256",
        "bank_artifact_sha256",
        "candidate_set_sha256",
        "candidate_freeze_sha256",
        "manifest_declaration_sha256",
        "purpose",
        "actor",
    }
    base_fields = {
        "schema_version",
        "sequence",
        "previous_event_sha256",
        "event",
        "status",
        "at",
        "event_sha256",
        *common_fields,
    }
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioBankError(
                f"consumption ledger row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(raw, dict):
            raise ScenarioBankError("consumption ledger event must be an object")
        if line != canonical_json_bytes(raw).decode("utf-8") + "\n":
            raise ScenarioBankError("consumption ledger event is not canonical JSON")
        if raw.get("schema_version") != CONSUMPTION_LEDGER_SCHEMA_VERSION:
            raise ScenarioBankError("consumption ledger schema mismatch")
        event_kind = raw.get("event")
        expected_fields = (
            base_fields
            if event_kind == "started"
            else base_fields | {"error_type", "error"}
            if event_kind == "finished"
            else set()
        )
        if not expected_fields or set(raw) != expected_fields:
            raise ScenarioBankError("consumption ledger event fields mismatch")
        event_hash = raw.get("event_sha256")
        body = {key: value for key, value in raw.items() if key != "event_sha256"}
        if not _is_sha256(event_hash) or event_hash != sha256_canonical_json(body):
            raise ScenarioBankError("consumption ledger event SHA-256 mismatch")
        if raw.get("previous_event_sha256") != previous_hash:
            raise ScenarioBankError("consumption ledger hash chain is discontinuous")
        if type(raw.get("sequence")) is not int or raw.get("sequence") != len(events):
            raise ScenarioBankError("consumption ledger sequence is discontinuous")
        if any(
            not _is_sha256(raw.get(field))
            for field in common_fields
            if field.endswith("sha256")
        ):
            raise ScenarioBankError("consumption ledger contains an invalid SHA-256")
        if raw.get("purpose") != FINAL_REPORT_PURPOSE:
            raise ScenarioBankError("consumption ledger purpose is invalid")
        if type(raw.get("actor")) is not str or not str(raw["actor"]).strip():
            raise ScenarioBankError("consumption ledger actor is invalid")
        if type(raw.get("at")) is not str or not raw["at"]:
            raise ScenarioBankError("consumption ledger timestamp is invalid")

        bank_hash = cast(str, raw["bank_payload_sha256"])
        if event_kind == "started":
            if raw.get("status") != "running":
                raise ScenarioBankError("consumption start status must be running")
            if bank_hash in attempts:
                raise ScenarioBankError("sealed bank has multiple consumption attempts")
            attempts[bank_hash] = cast(dict[str, object], raw)
        else:
            if raw.get("status") not in {"completed", "failed"}:
                raise ScenarioBankError("consumption finish status is invalid")
            started = attempts.get(bank_hash)
            if started is None:
                raise ScenarioBankError("consumption ledger finishes before it starts")
            if started.get("terminal_event_sha256") is not None:
                raise ScenarioBankError(
                    "consumption attempt has multiple terminal events"
                )
            if any(raw[field] != started[field] for field in common_fields):
                raise ScenarioBankError(
                    "consumption finish binding changed after start"
                )
            if raw["status"] == "completed" and (
                raw["error_type"] is not None or raw["error"] is not None
            ):
                raise ScenarioBankError("completed consumption cannot carry an error")
            if raw["status"] == "failed" and (
                type(raw["error_type"]) is not str
                or not raw["error_type"]
                or type(raw["error"]) is not str
            ):
                raise ScenarioBankError("failed consumption must describe its error")
            # Keep terminal state outside the serialized start schema while
            # auditing subsequent events in this read.
            attempts[bank_hash] = {
                **started,
                "terminal_event_sha256": event_hash,
            }
        previous_hash = cast(str, event_hash)
        events.append(cast(dict[str, object], raw))
    return events, previous_hash


def _append_ledger_event(
    ledger_path: Path,
    body: Mapping[str, object],
    *,
    require_unconsumed_bank: str | None = None,
) -> dict[str, object]:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        events, previous_hash = _read_ledger(stream)
        if require_unconsumed_bank is not None and any(
            event.get("bank_payload_sha256") == require_unconsumed_bank
            and event.get("event") == "started"
            for event in events
        ):
            raise ScenarioBankError("sealed bank already has a consumption attempt")
        event = {
            "schema_version": CONSUMPTION_LEDGER_SCHEMA_VERSION,
            "sequence": len(events),
            "previous_event_sha256": previous_hash,
            **dict(body),
        }
        event["event_sha256"] = sha256_canonical_json(event)
        stream.seek(0, os.SEEK_END)
        stream.write(canonical_json_bytes(event).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return event


def _validate_sealed_authority(
    *,
    purpose: str,
    candidate_freeze_path: Path,
    experiment_manifest_path: Path,
    logical_split: str,
    bank_payload_sha256: str,
) -> tuple[dict[str, object], dict[str, Any]]:
    if purpose != FINAL_REPORT_PURPOSE:
        raise ScenarioBankError("sealed bank purpose must be final_report")
    candidate_freeze = _load_candidate_freeze(candidate_freeze_path)
    from .manifest import (  # noqa: PLC0415 - avoids module cycle
        MANIFEST_SCHEMA_V2,
        load_manifest,
    )

    manifest = load_manifest(experiment_manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_V2 or manifest.get(
        "status"
    ) not in {
        "approved",
        "running",
    }:
        raise ScenarioBankError(
            "sealed access requires an approved/running manifest v2"
        )
    if candidate_freeze["manifest_declaration_sha256"] != manifest.get(
        "declaration_sha256"
    ):
        raise ScenarioBankError("candidate freeze is bound to another manifest")
    declaration = manifest.get("declaration")
    artifact_contract = (
        declaration.get("artifact_contract")
        if isinstance(declaration, Mapping)
        else None
    )
    scenario_banks = (
        artifact_contract.get("scenario_banks")
        if isinstance(artifact_contract, Mapping)
        else None
    )
    if (
        not isinstance(scenario_banks, Mapping)
        or scenario_banks.get(logical_split) != bank_payload_sha256
    ):
        raise ScenarioBankError("sealed bank is not bound to this manifest declaration")
    lifecycle = manifest.get("lifecycle")
    if not isinstance(lifecycle, list) or not any(
        isinstance(event, Mapping) and event.get("to") == "approved"
        for event in lifecycle
    ):
        raise ScenarioBankError("sealed access manifest has no approval event")
    return candidate_freeze, manifest


@contextmanager
def consume_sealed_scenario_bank(  # noqa: PLR0913 - authority is explicit
    artifact_path: Path,
    ledger_path: Path,
    *,
    purpose: str,
    candidate_freeze_path: Path,
    experiment_manifest_path: Path,
    actor: str,
) -> Iterator[ScenarioBank]:
    """Open one sealed bank exactly once and append start/terminal events."""
    if not actor.strip():
        raise ScenarioBankError("sealed bank consumer actor must not be empty")
    public_manifest = inspect_scenario_bank(artifact_path)
    if public_manifest.get("logical_split") not in {
        "sealed-test-iid",
        "ci-sealed-fixture",
    }:
        raise ScenarioBankError("consumption gate only accepts a sealed test bank")
    if public_manifest.get("sealed") is not True:
        raise ScenarioBankError("consumption gate refuses an unsealed bank")
    candidate_freeze, experiment_manifest = _validate_sealed_authority(
        purpose=purpose,
        candidate_freeze_path=candidate_freeze_path,
        experiment_manifest_path=experiment_manifest_path,
        logical_split=str(public_manifest["logical_split"]),
        bank_payload_sha256=str(public_manifest["payload_sha256"]),
    )
    bank_hash = str(public_manifest["payload_sha256"])
    common: dict[str, object] = {
        "bank_payload_sha256": bank_hash,
        "bank_artifact_sha256": public_manifest["artifact_sha256"],
        "candidate_set_sha256": candidate_freeze["candidate_set_sha256"],
        "candidate_freeze_sha256": sha256_file(candidate_freeze_path),
        "manifest_declaration_sha256": experiment_manifest["declaration_sha256"],
        "purpose": purpose,
        "actor": actor,
    }
    _append_ledger_event(
        ledger_path,
        {**common, "event": "started", "status": "running", "at": _now()},
        require_unconsumed_bank=bank_hash,
    )
    try:
        bank = _load_scenario_bank(artifact_path, allow_sealed=True)
        yield bank
    except BaseException as exc:
        _append_ledger_event(
            ledger_path,
            {
                **common,
                "event": "finished",
                "status": "failed",
                "at": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    else:
        _append_ledger_event(
            ledger_path,
            {
                **common,
                "event": "finished",
                "status": "completed",
                "at": _now(),
                "error_type": None,
                "error": None,
            },
        )


def scenario_lookup(scenarios: Sequence[ScenarioV1]) -> dict[str, ScenarioV1]:
    """Build a duplicate-refusing lookup for runner injection."""
    scenario_collection_sha256(scenarios)
    return {scenario.scenario_id: scenario for scenario in scenarios}


__all__ = [
    "BANK_SCHEMA_VERSION",
    "CANDIDATE_FREEZE_SCHEMA_VERSION",
    "CONSUMPTION_LEDGER_SCHEMA_VERSION",
    "FINAL_REPORT_PURPOSE",
    "BalancedStressSelection",
    "ScenarioBank",
    "ScenarioBankError",
    "StressSelectionDesign",
    "audit_seed_roll_scenario_bank",
    "consume_sealed_scenario_bank",
    "freeze_candidate_set",
    "generate_iid_scenarios",
    "inspect_scenario_bank",
    "load_scenario_bank",
    "load_scenario_bank_subset",
    "scenario_lookup",
    "select_balanced_stress_scenarios",
    "validate_split_disjointness",
    "validate_task1_split_registry",
    "write_scenario_bank",
    "write_seed_roll_scenario_bank",
]
