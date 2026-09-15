"""Auditable one-shot randomization for Task-1 formal scenario schedules.

The legacy project used memorable integers both as model seeds and as deal
seeds.  A formal seed roll instead draws one 256-bit root exactly once, uses a
domain-separated keyed permutation to allocate registered source seeds without
replacement, and records the complete allocation before any training outcome
exists.  The root controls *addresses* of random streams; immutable ScenarioV1
snapshots remain the identity of dealt games.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

from splendor.seed_registry import (
    TASK1_TRAIN_SCHEDULE,
    registry_sha256,
    resolve_task1_scenario_split,
)

from .protocol import (
    PairedTrainingRow,
    RngKey,
    SeedLineage,
    canonical_json_bytes,
    derive_seed,
    make_paired_training_schedule,
    sha256_canonical_json,
    validate_paired_training_schedule,
)
from .scenario import (
    ScenarioV1,
    ScenarioValidationError,
    generate_scenario,
    validate_scenario,
    with_sampling_metadata,
)

SEED_ROLL_SCHEMA_VERSION: Final = "splendor-seed-roll/1"
SRSWOR_SELECTION_SCHEMA_VERSION: Final = "splendor-natural-deal-srswor/1"
SEED_ROLL_RESERVATION_SCHEMA_VERSION: Final = "splendor-seed-roll-reservation/1"
SEED_ROLL_AUTHORITY_SCHEMA_VERSION: Final = "splendor-seed-roll-authority/1"
CONFIRMATORY_ACTIVATION_SCHEMA_VERSION: Final = "splendor-confirmatory-activation/1"
SEED_ROLL_ROOT_BYTES: Final = 32
SOURCE_SELECTION_METHOD: Final = (
    "ascending HMAC-SHA256(root, canonical protocol/experiment/phase/registry/"
    "split/range/source-seed); "
    "global permutation without replacement"
)
REPLICATE_ALLOCATION_METHOD: Final = (
    "contiguous equal blocks of the keyed global permutation in replicate-id order"
)
SEAT_ASSIGNMENT_METHOD: Final = "independent HMAC-SHA256 seat rank per replicate; lower half seat 0, upper half seat 1"
ROOT_GENERATOR: Final = "python-secrets.token_bytes(32)/operating-system-csprng"
MODEL_SEED_METHOD: Final = (
    "splendor-rng-v1 model_init keyed by randomization-root commitment and replicate"
)
SHA256_HEX_LENGTH: Final = 64
MIN_BALANCED_GAMES: Final = 2
TASK1_PILOT_REPLICATE_COUNT: Final = 3
TASK1_FORMAL_REPLICATE_IDS: Final = (0, 1, 2, 3, 4)
TASK1_FORMAL_GAMES_PER_REPLICATE: Final = 32_000
TASK1_FORMAL_DESIGN_PROFILE: Final = "task1-formal-5x32000"
CI_DESIGN_PROFILE: Final = "ci-fixture"
ACTIVATION_BOUND_PATH_COUNT: Final = 3
ASCII_CONTROL_LIMIT: Final = 32
HISTORICAL_MODEL_SEED_ALIASES: Final = frozenset({42, 1234, 2024})
AUTHORITY_LEDGER_NAME: Final = "seed-roll-authority.jsonl"
AUTHORITY_LOCK_NAME: Final = ".seed-roll-authority.lock"
AUTHORITY_RESERVATIONS_DIR: Final = "reservations"
AUTHORITY_ARTIFACTS_DIR: Final = "artifacts"


class SeedRollError(ValueError):
    """Raised when a seed roll or its immutable binding is invalid."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _root_bytes(root_hex: str) -> bytes:
    if (
        type(root_hex) is not str
        or len(root_hex) != SEED_ROLL_ROOT_BYTES * 2
        or any(character not in "0123456789abcdef" for character in root_hex)
    ):
        raise SeedRollError(
            "randomization root must be exactly 32 bytes of lowercase hexadecimal"
        )
    return bytes.fromhex(root_hex)


def randomization_root_sha256(root_hex: str) -> str:
    """Return the commitment to one canonical 256-bit randomization root."""
    return hashlib.sha256(_root_bytes(root_hex)).hexdigest()


def _rank_digest(  # noqa: PLR0913 - every frozen selection axis is explicit
    root: bytes,
    *,
    experiment_id: str,
    phase: str,
    logical_split: str,
    source_segment: str,
    source_range: tuple[int, int],
    seed_registry_sha256: str,
    source_seed: int,
) -> bytes:
    message = canonical_json_bytes(
        {
            "protocol": SEED_ROLL_SCHEMA_VERSION,
            "domain": "source-seed-permutation",
            "experiment_id": experiment_id,
            "phase": phase,
            "logical_split": logical_split,
            "source_segment": source_segment,
            "source_range": list(source_range),
            "seed_registry_sha256": seed_registry_sha256,
            "source_seed": source_seed,
        }
    )
    return hmac.new(root, message, hashlib.sha256).digest()


def _seat_rank_digest(  # noqa: PLR0913 - every frozen seat axis is explicit
    root: bytes,
    *,
    experiment_id: str,
    phase: str,
    randomization_root_sha256: str,
    replicate_id: int,
    source_seed: int,
) -> bytes:
    message = canonical_json_bytes(
        {
            "protocol": SEED_ROLL_SCHEMA_VERSION,
            "domain": "seat-assignment",
            "experiment_id": experiment_id,
            "phase": phase,
            "logical_split": "train-schedule",
            "randomization_root_sha256": randomization_root_sha256,
            "replicate_id": replicate_id,
            "source_seed": source_seed,
        }
    )
    return hmac.new(root, message, hashlib.sha256).digest()


def _model_init_lineage(
    *,
    experiment_id: str,
    phase: str,
    root_sha256: str,
    replicate_id: int,
) -> SeedLineage:
    return derive_seed(
        RngKey(
            experiment_id=experiment_id,
            phase=phase,
            randomization_root_sha256=root_sha256,
            coupling_group=f"replicate-{replicate_id}",
            stream_name="model_init",
            replicate_id=replicate_id,
        )
    )


@dataclass(frozen=True)
class ReplicateSeedRoll:
    """One preallocated training replicate within a complete seed roll."""

    replicate_id: int
    source_seeds: tuple[int, ...]
    seats: tuple[int, ...]
    model_init_lineage: SeedLineage

    def __post_init__(self) -> None:
        if type(self.replicate_id) is not int or self.replicate_id < 0:
            raise SeedRollError("replicate_id must be a non-negative integer")
        if not self.source_seeds or len(self.source_seeds) != len(self.seats):
            raise SeedRollError("replicate source seeds and seats must be non-empty")
        if any(type(seed) is not int or seed < 0 for seed in self.source_seeds):
            raise SeedRollError("replicate source seeds must be non-negative integers")
        if len(self.source_seeds) != len(set(self.source_seeds)):
            raise SeedRollError(
                "replicate source seeds must be sampled without replacement"
            )
        if any(type(seat) is not int or seat not in (0, 1) for seat in self.seats):
            raise SeedRollError("replicate seat schedule must contain only integer 0/1")
        seat_counts = Counter(self.seats)
        if seat_counts != {0: len(self.seats) // 2, 1: len(self.seats) // 2}:
            raise SeedRollError("replicate seat schedule must be exactly balanced")

    @property
    def source_seed_order_sha256(self) -> str:
        return sha256_canonical_json(list(self.source_seeds))

    @property
    def seat_schedule_sha256(self) -> str:
        return sha256_canonical_json(list(self.seats))

    @property
    def coordinate_sha256(self) -> str:
        return sha256_canonical_json(
            [
                {"ordinal": ordinal, "source_seed": source_seed, "seat": seat}
                for ordinal, (source_seed, seat) in enumerate(
                    zip(self.source_seeds, self.seats, strict=True)
                )
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "model_init_lineage": self.model_init_lineage.as_dict(),
            "model_seed63": self.model_init_lineage.seed63,
            "model_seed_digest": self.model_init_lineage.digest_hex,
            "source_seeds": list(self.source_seeds),
            "source_seed_order_sha256": self.source_seed_order_sha256,
            "seats": list(self.seats),
            "seat_counts": {"0": self.seats.count(0), "1": self.seats.count(1)},
            "seat_schedule_sha256": self.seat_schedule_sha256,
            "coordinate_sha256": self.coordinate_sha256,
        }


@dataclass(frozen=True)
class SeedRollPlan:
    """Complete deterministic allocation reconstructed from one root."""

    experiment_id: str
    phase: str
    logical_split: str
    randomization_root_hex: str
    replicates: tuple[ReplicateSeedRoll, ...]

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - fail-closed axes
        for field_name, value in (
            ("experiment_id", self.experiment_id),
            ("phase", self.phase),
        ):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or any(ord(character) < ASCII_CONTROL_LIMIT for character in value)
            ):
                raise SeedRollError(
                    f"seed-roll {field_name} must be a canonical identifier"
                )
        _root_bytes(self.randomization_root_hex)
        if self.logical_split != "train-schedule":
            raise SeedRollError("formal training seed roll requires train-schedule")
        if not self.replicates:
            raise SeedRollError("seed roll requires at least one replicate")
        replicate_ids = tuple(item.replicate_id for item in self.replicates)
        if replicate_ids != tuple(range(len(self.replicates))):
            raise SeedRollError(
                "seed-roll replicate IDs must be canonical contiguous integers from zero"
            )
        counts = {len(item.source_seeds) for item in self.replicates}
        if len(counts) != 1:
            raise SeedRollError("seed-roll replicates must have equal scenario counts")
        all_seeds = [seed for item in self.replicates for seed in item.source_seeds]
        if len(all_seeds) != len(set(all_seeds)):
            raise SeedRollError("source seeds overlap across replicates")
        segment = resolve_task1_scenario_split(self.logical_split)
        if segment != TASK1_TRAIN_SCHEDULE:
            raise SeedRollError("seed roll resolved an unexpected training segment")
        if any(seed not in range(segment.start, segment.end) for seed in all_seeds):
            raise SeedRollError(
                "seed roll contains a source outside its registered segment"
            )
        expected_root_sha256 = self.randomization_root_sha256
        model_seeds: list[int] = []
        for item in self.replicates:
            expected = _model_init_lineage(
                experiment_id=self.experiment_id,
                phase=self.phase,
                root_sha256=expected_root_sha256,
                replicate_id=item.replicate_id,
            )
            if item.model_init_lineage != expected:
                raise SeedRollError("model-init lineage does not match the seed roll")
            model_seeds.append(item.model_init_lineage.seed63)
        if len(model_seeds) != len(set(model_seeds)):
            raise SeedRollError("model-init seeds collide across reserved replicates")
        if set(model_seeds) & HISTORICAL_MODEL_SEED_ALIASES:
            raise SeedRollError(
                "formal model-init seeds must not reuse historical seed aliases"
            )

    @property
    def randomization_root_sha256(self) -> str:
        return randomization_root_sha256(self.randomization_root_hex)

    @property
    def replicate_ids(self) -> tuple[int, ...]:
        return tuple(item.replicate_id for item in self.replicates)

    @property
    def games_per_replicate(self) -> int:
        return len(self.replicates[0].source_seeds)

    @property
    def selected_source_seed_count(self) -> int:
        return len(self.replicates) * self.games_per_replicate

    @property
    def pilot_replicate_ids(self) -> tuple[int, ...]:
        return self.replicate_ids[:TASK1_PILOT_REPLICATE_COUNT]

    @property
    def confirmatory_reserve_replicate_ids(self) -> tuple[int, ...]:
        return self.replicate_ids[TASK1_PILOT_REPLICATE_COUNT:]

    @property
    def design_profile(self) -> str:
        if (
            self.phase == "T1.4"
            and self.replicate_ids == TASK1_FORMAL_REPLICATE_IDS
            and self.games_per_replicate == TASK1_FORMAL_GAMES_PER_REPLICATE
        ):
            return TASK1_FORMAL_DESIGN_PROFILE
        return CI_DESIGN_PROFILE

    @property
    def source_seed_union_sha256(self) -> str:
        return sha256_canonical_json(
            sorted(seed for item in self.replicates for seed in item.source_seeds)
        )

    @property
    def payload_sha256(self) -> str:
        return sha256_canonical_json(self.to_dict())

    def replicate(self, replicate_id: int) -> ReplicateSeedRoll:
        if type(replicate_id) is not int or replicate_id not in self.replicate_ids:
            raise SeedRollError(
                f"replicate {replicate_id!r} is not reserved by this roll"
            )
        return self.replicates[replicate_id]

    def to_dict(self) -> dict[str, object]:
        segment = resolve_task1_scenario_split(self.logical_split)
        return {
            "schema_version": SEED_ROLL_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "logical_split": self.logical_split,
            "source_segment": segment.name,
            "source_range": [segment.start, segment.end],
            "source_population": segment.end - segment.start,
            "seed_registry_sha256": registry_sha256(),
            "root_generator": ROOT_GENERATOR,
            "randomization_root_hex": self.randomization_root_hex,
            "randomization_root_sha256": self.randomization_root_sha256,
            "source_selection_method": SOURCE_SELECTION_METHOD,
            "replicate_allocation_method": REPLICATE_ALLOCATION_METHOD,
            "seat_assignment_method": SEAT_ASSIGNMENT_METHOD,
            "model_seed_method": MODEL_SEED_METHOD,
            "replicate_ids": list(self.replicate_ids),
            "design_profile": self.design_profile,
            "pilot_replicate_ids": list(self.pilot_replicate_ids),
            "confirmatory_reserve_replicate_ids": list(
                self.confirmatory_reserve_replicate_ids
            ),
            "replicate_roles": {
                str(replicate_id): (
                    "pilot"
                    if replicate_id in self.pilot_replicate_ids
                    else "confirmatory-reserve"
                )
                for replicate_id in self.replicate_ids
            },
            "games_per_replicate": self.games_per_replicate,
            "selected_source_seed_count": self.selected_source_seed_count,
            "first_order_inclusion_probability": {
                "numerator": self.selected_source_seed_count,
                "denominator": segment.end - segment.start,
            },
            "per_replicate_first_order_inclusion_probability": {
                "numerator": self.games_per_replicate,
                "denominator": segment.end - segment.start,
            },
            "pooled_seat_specific_inclusion_probability": {
                "numerator": self.selected_source_seed_count,
                "denominator": 2 * (segment.end - segment.start),
            },
            "per_replicate_seat_specific_inclusion_probability": {
                "numerator": self.games_per_replicate,
                "denominator": 2 * (segment.end - segment.start),
            },
            "source_seed_union_sha256": self.source_seed_union_sha256,
            "replicates": [item.to_dict() for item in self.replicates],
        }


@dataclass(frozen=True)
class SeedRollArtifact:
    """A verified canonical seed-roll file and its parsed plan."""

    path: Path
    authority_dir: Path
    plan: SeedRollPlan
    payload_sha256: str
    artifact_sha256: str
    roll_identity_sha256: str
    authority_event_sha256: str
    authority_event: Mapping[str, object]

    def manifest_binding(self) -> dict[str, object]:
        """Return the self-contained declaration frozen by manifest v2."""
        return {
            "protocol": SEED_ROLL_SCHEMA_VERSION,
            "experiment_id": self.plan.experiment_id,
            "phase": self.plan.phase,
            "authority_dir": str(self.authority_dir),
            "payload_sha256": self.payload_sha256,
            "artifact_sha256": self.artifact_sha256,
            "roll_identity_sha256": self.roll_identity_sha256,
            "authority_event_sha256": self.authority_event_sha256,
            "authority_event": dict(self.authority_event),
            "randomization_root_hex": self.plan.randomization_root_hex,
            "randomization_root_sha256": self.plan.randomization_root_sha256,
            "logical_split": self.plan.logical_split,
            "source_segment": TASK1_TRAIN_SCHEDULE.name,
            "source_range": [TASK1_TRAIN_SCHEDULE.start, TASK1_TRAIN_SCHEDULE.end],
            "source_population": (
                TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start
            ),
            "seed_registry_sha256": registry_sha256(),
            "source_selection_method": SOURCE_SELECTION_METHOD,
            "replicate_allocation_method": REPLICATE_ALLOCATION_METHOD,
            "seat_assignment_method": SEAT_ASSIGNMENT_METHOD,
            "model_seed_method": MODEL_SEED_METHOD,
            "replicate_ids": list(self.plan.replicate_ids),
            "design_profile": self.plan.design_profile,
            "pilot_replicate_ids": list(self.plan.pilot_replicate_ids),
            "confirmatory_reserve_replicate_ids": list(
                self.plan.confirmatory_reserve_replicate_ids
            ),
            "replicate_roles": {
                str(replicate_id): (
                    "pilot"
                    if replicate_id in self.plan.pilot_replicate_ids
                    else "confirmatory-reserve"
                )
                for replicate_id in self.plan.replicate_ids
            },
            "games_per_replicate": self.plan.games_per_replicate,
            "selected_source_seed_count": self.plan.selected_source_seed_count,
            "first_order_inclusion_probability": {
                "numerator": self.plan.selected_source_seed_count,
                "denominator": (TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start),
            },
            "per_replicate_first_order_inclusion_probability": {
                "numerator": self.plan.games_per_replicate,
                "denominator": (TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start),
            },
            "pooled_seat_specific_inclusion_probability": {
                "numerator": self.plan.selected_source_seed_count,
                "denominator": (
                    2 * (TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start)
                ),
            },
            "per_replicate_seat_specific_inclusion_probability": {
                "numerator": self.plan.games_per_replicate,
                "denominator": (
                    2 * (TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start)
                ),
            },
            "source_seed_union_sha256": self.plan.source_seed_union_sha256,
            "model_seed63_by_replicate": {
                str(item.replicate_id): item.model_init_lineage.seed63
                for item in self.plan.replicates
            },
            "model_seed_digest_by_replicate": {
                str(item.replicate_id): item.model_init_lineage.digest_hex
                for item in self.plan.replicates
            },
        }


@dataclass(frozen=True)
class ConfirmatoryActivationArtifact:
    """Verified post-pilot authorization for the reserved replicates.

    The activation is deliberately a separate immutable file.  It binds the
    completed pilot manifest and one pilot evidence artifact before a new
    confirmatory manifest may move the reserved replicates into execution.
    """

    path: Path
    artifact_sha256: str
    experiment_id: str
    phase: str
    seed_roll_payload_sha256: str
    pilot_replicate_ids: tuple[int, ...]
    confirmatory_replicate_ids: tuple[int, ...]
    pilot_manifest_path: Path
    pilot_manifest_sha256: str
    pilot_manifest_declaration_sha256: str
    pilot_completed_event_sha256: str
    pilot_evidence_path: Path
    pilot_evidence_sha256: str
    decision: str
    actor: str
    note: str
    at: str


@dataclass(frozen=True)
class SeedRollSelectionDesign:
    """Sampling metadata for a materialized subset of reserved replicates."""

    seed_roll_payload_sha256: str
    randomization_root_sha256: str
    active_replicate_ids: tuple[int, ...]
    games_per_replicate: int
    source_population: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.seed_roll_payload_sha256) or not _is_sha256(
            self.randomization_root_sha256
        ):
            raise SeedRollError("natural-deal SRSWOR design hashes are invalid")
        if (
            not self.active_replicate_ids
            or self.active_replicate_ids
            != tuple(sorted(set(self.active_replicate_ids)))
            or any(
                type(value) is not int or value < 0
                for value in self.active_replicate_ids
            )
        ):
            raise SeedRollError("active replicate IDs must be sorted unique integers")
        expected_population = TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start
        if (
            self.games_per_replicate < MIN_BALANCED_GAMES
            or self.games_per_replicate % 2
        ):
            raise SeedRollError(
                "natural-deal SRSWOR games per replicate must be even and at least two"
            )
        if self.source_population != expected_population:
            raise SeedRollError(
                "natural-deal SRSWOR source population must equal the registered "
                "train-schedule segment"
            )
        if self.selected_count > self.source_population:
            raise SeedRollError(
                "natural-deal SRSWOR sample exceeds its source population"
            )

    @property
    def selected_count(self) -> int:
        return len(self.active_replicate_ids) * self.games_per_replicate

    @property
    def inclusion_probability(self) -> float:
        return self.selected_count / self.source_population

    @property
    def per_replicate_inclusion_probability(self) -> float:
        return self.games_per_replicate / self.source_population

    @property
    def sha256(self) -> str:
        return sha256_canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SRSWOR_SELECTION_SCHEMA_VERSION,
            "seed_roll_payload_sha256": self.seed_roll_payload_sha256,
            "randomization_root_sha256": self.randomization_root_sha256,
            "active_replicate_ids": list(self.active_replicate_ids),
            "games_per_replicate": self.games_per_replicate,
            "source_population": self.source_population,
            "selected_count": self.selected_count,
            "first_order_inclusion_probability": {
                "numerator": self.selected_count,
                "denominator": self.source_population,
            },
            "per_replicate_first_order_inclusion_probability": {
                "numerator": self.games_per_replicate,
                "denominator": self.source_population,
            },
            "pooled_seat_specific_inclusion_probability": {
                "numerator": self.selected_count,
                "denominator": 2 * self.source_population,
            },
            "per_replicate_seat_specific_inclusion_probability": {
                "numerator": self.games_per_replicate,
                "denominator": 2 * self.source_population,
            },
        }


@dataclass(frozen=True)
class RolledScenarioSelection:
    """Materialized natural-deal SRSWOR scenarios plus their finite-population design."""

    scenarios: tuple[ScenarioV1, ...]
    design: SeedRollSelectionDesign


def parse_seed_roll_selection_design(raw: object) -> SeedRollSelectionDesign:
    """Parse an exact natural-deal SRSWOR bank design without consulting mutable state."""
    if not isinstance(raw, Mapping):
        raise SeedRollError("natural-deal SRSWOR bank requires a selection design")
    required = {
        "schema_version",
        "seed_roll_payload_sha256",
        "randomization_root_sha256",
        "active_replicate_ids",
        "games_per_replicate",
        "source_population",
        "selected_count",
        "first_order_inclusion_probability",
        "per_replicate_first_order_inclusion_probability",
        "pooled_seat_specific_inclusion_probability",
        "per_replicate_seat_specific_inclusion_probability",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != SRSWOR_SELECTION_SCHEMA_VERSION
    ):
        raise SeedRollError("natural-deal SRSWOR selection design schema mismatch")
    active = raw.get("active_replicate_ids")
    games = raw.get("games_per_replicate")
    population = raw.get("source_population")
    if (
        not isinstance(active, list)
        or any(type(value) is not int for value in active)
        or type(games) is not int
        or type(population) is not int
        or type(raw.get("seed_roll_payload_sha256")) is not str
        or type(raw.get("randomization_root_sha256")) is not str
    ):
        raise SeedRollError("natural-deal SRSWOR selection design fields are invalid")
    design = SeedRollSelectionDesign(
        seed_roll_payload_sha256=cast(str, raw["seed_roll_payload_sha256"]),
        randomization_root_sha256=cast(str, raw["randomization_root_sha256"]),
        active_replicate_ids=tuple(cast(list[int], active)),
        games_per_replicate=games,
        source_population=population,
    )
    if dict(raw) != design.to_dict():
        raise SeedRollError(
            "natural-deal SRSWOR selection design is internally inconsistent"
        )
    return design


def _validate_roll_request(
    experiment_id: str,
    phase: str,
    replicate_ids: Sequence[int],
    games_per_replicate: int,
) -> tuple[int, ...]:
    for field_name, value in (("experiment_id", experiment_id), ("phase", phase)):
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or any(ord(character) < ASCII_CONTROL_LIMIT for character in value)
        ):
            raise SeedRollError(
                f"seed-roll {field_name} must be a canonical identifier"
            )
    identities = tuple(replicate_ids)
    if identities != tuple(range(len(identities))) or not identities:
        raise SeedRollError(
            "replicate IDs must be canonical contiguous integers from zero"
        )
    if type(games_per_replicate) is not int or games_per_replicate < MIN_BALANCED_GAMES:
        raise SeedRollError("games_per_replicate must be an integer of at least two")
    if games_per_replicate % 2:
        raise SeedRollError("games_per_replicate must be even for exact seat balance")
    capacity = TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start
    if len(identities) * games_per_replicate > capacity:
        raise SeedRollError("seed-roll request exceeds the training source segment")
    return identities


def build_seed_roll_plan(
    *,
    experiment_id: str,
    phase: str,
    replicate_ids: Sequence[int],
    games_per_replicate: int,
    randomization_root_hex: str,
) -> SeedRollPlan:
    """Deterministically reconstruct the complete allocation from one root."""
    identities = _validate_roll_request(
        experiment_id,
        phase,
        replicate_ids,
        games_per_replicate,
    )
    _root_bytes(randomization_root_hex)
    return _build_seed_roll_plan_cached(
        experiment_id,
        phase,
        identities,
        games_per_replicate,
        randomization_root_hex,
        registry_sha256(),
    )


@lru_cache(maxsize=8)
def _build_seed_roll_plan_cached(  # noqa: PLR0913 - cache binds protocol facts
    experiment_id: str,
    phase: str,
    identities: tuple[int, ...],
    games_per_replicate: int,
    randomization_root_hex: str,
    registry_digest: str,
) -> SeedRollPlan:
    """Cache immutable reconstructions used repeatedly by manifest gates."""
    root = _root_bytes(randomization_root_hex)
    root_sha256 = hashlib.sha256(root).hexdigest()
    segment = TASK1_TRAIN_SCHEDULE
    ranked = sorted(
        range(segment.start, segment.end),
        key=lambda source_seed: (
            _rank_digest(
                root,
                experiment_id=experiment_id,
                phase=phase,
                logical_split="train-schedule",
                source_segment=segment.name,
                source_range=(segment.start, segment.end),
                seed_registry_sha256=registry_digest,
                source_seed=source_seed,
            ),
            source_seed,
        ),
    )
    selected = ranked[: len(identities) * games_per_replicate]
    replicates: list[ReplicateSeedRoll] = []
    for block_index, replicate_id in enumerate(identities):
        start = block_index * games_per_replicate
        source_seeds = tuple(selected[start : start + games_per_replicate])
        seat_order = sorted(
            source_seeds,
            key=lambda source_seed: (
                _seat_rank_digest(
                    root,
                    experiment_id=experiment_id,
                    phase=phase,
                    randomization_root_sha256=root_sha256,
                    replicate_id=replicate_id,
                    source_seed=source_seed,
                ),
                source_seed,
            ),
        )
        seat_zero = set(seat_order[: games_per_replicate // 2])
        seats = tuple(
            0 if source_seed in seat_zero else 1 for source_seed in source_seeds
        )
        replicates.append(
            ReplicateSeedRoll(
                replicate_id=replicate_id,
                source_seeds=source_seeds,
                seats=seats,
                model_init_lineage=_model_init_lineage(
                    experiment_id=experiment_id,
                    phase=phase,
                    root_sha256=root_sha256,
                    replicate_id=replicate_id,
                ),
            )
        )
    return SeedRollPlan(
        experiment_id=experiment_id,
        phase=phase,
        logical_split="train-schedule",
        randomization_root_hex=randomization_root_hex,
        replicates=tuple(replicates),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _real_directory(path: Path, *, create: bool) -> Path:
    """Return one canonical directory while refusing symlink traversal."""
    # ``resolve`` alone would erase the evidence that the caller supplied a
    # symlink, so preserve the lexical absolute path for the comparison.
    lexical = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    if lexical.resolve(strict=False) != lexical:
        raise SeedRollError(f"seed-roll directory traverses a symlink: {lexical}")
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    try:
        metadata = os.lstat(lexical)
    except OSError as exc:
        raise SeedRollError(
            f"cannot access seed-roll directory {lexical}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SeedRollError(f"seed-roll path is not a real directory: {lexical}")
    if lexical.resolve(strict=True) != lexical:
        raise SeedRollError(f"seed-roll directory traverses a symlink: {lexical}")
    return lexical


def _read_regular_bytes(path: Path) -> bytes:
    """Read an authority file without following a replaceable symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SeedRollError(
            f"cannot open seed-roll authority file {path}: {exc}"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SeedRollError(f"seed-roll authority path is not a file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    _real_directory(path.parent, create=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SeedRollError(
            f"cannot create seed-roll authority file {path}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), mode)
    _fsync_directory(path.parent)


def _roll_identity_sha256(experiment_id: str, phase: str) -> str:
    return sha256_canonical_json(
        {
            "protocol": SEED_ROLL_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "phase": phase,
        }
    )


def _roll_request(
    *,
    authority_dir: Path,
    experiment_id: str,
    phase: str,
    replicate_ids: Sequence[int],
    games_per_replicate: int,
) -> dict[str, object]:
    identities = _validate_roll_request(
        experiment_id,
        phase,
        replicate_ids,
        games_per_replicate,
    )
    segment = TASK1_TRAIN_SCHEDULE
    return {
        "protocol": SEED_ROLL_SCHEMA_VERSION,
        "authority_dir": str(authority_dir),
        "experiment_id": experiment_id,
        "phase": phase,
        "logical_split": "train-schedule",
        "source_segment": segment.name,
        "source_range": [segment.start, segment.end],
        "source_population": segment.end - segment.start,
        "seed_registry_sha256": registry_sha256(),
        "root_generator": ROOT_GENERATOR,
        "source_selection_method": SOURCE_SELECTION_METHOD,
        "replicate_allocation_method": REPLICATE_ALLOCATION_METHOD,
        "seat_assignment_method": SEAT_ASSIGNMENT_METHOD,
        "model_seed_method": MODEL_SEED_METHOD,
        "replicate_ids": list(identities),
        "design_profile": (
            TASK1_FORMAL_DESIGN_PROFILE
            if phase == "T1.4"
            and identities == TASK1_FORMAL_REPLICATE_IDS
            and games_per_replicate == TASK1_FORMAL_GAMES_PER_REPLICATE
            else CI_DESIGN_PROFILE
        ),
        "pilot_replicate_ids": list(identities[:TASK1_PILOT_REPLICATE_COUNT]),
        "confirmatory_reserve_replicate_ids": list(
            identities[TASK1_PILOT_REPLICATE_COUNT:]
        ),
        "replicate_roles": {
            str(replicate_id): (
                "pilot"
                if replicate_id in identities[:TASK1_PILOT_REPLICATE_COUNT]
                else "confirmatory-reserve"
            )
            for replicate_id in identities
        },
        "games_per_replicate": games_per_replicate,
        "selected_source_seed_count": len(identities) * games_per_replicate,
    }


def _reservation_path(authority_dir: Path, roll_identity_sha256: str) -> Path:
    return authority_dir / AUTHORITY_RESERVATIONS_DIR / f"{roll_identity_sha256}.json"


def _artifact_path(authority_dir: Path, payload_sha256: str) -> Path:
    return authority_dir / AUTHORITY_ARTIFACTS_DIR / f"seed-roll-{payload_sha256}.json"


@contextmanager
def _authority_lock(authority_dir: Path) -> Iterator[None]:
    authority_dir = _real_directory(authority_dir, create=True)
    lock_path = authority_dir / AUTHORITY_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SeedRollError(f"cannot open seed-roll authority lock: {exc}") from exc
    with os.fdopen(descriptor, "a+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise SeedRollError("seed-roll authority lock is not a regular file")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_reservation(
    raw: object,
    *,
    roll_identity_sha256: str,
    request: Mapping[str, object],
) -> str:
    if not isinstance(raw, Mapping):
        raise SeedRollError("seed-roll reservation root must be an object")
    required = {
        "schema_version",
        "roll_identity_sha256",
        "request",
        "request_sha256",
        "randomization_root_hex",
        "randomization_root_sha256",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != SEED_ROLL_RESERVATION_SCHEMA_VERSION
        or raw.get("roll_identity_sha256") != roll_identity_sha256
        or raw.get("request") != dict(request)
        or raw.get("request_sha256") != sha256_canonical_json(request)
    ):
        raise SeedRollError("seed-roll reservation does not match its identity/design")
    root_hex = raw.get("randomization_root_hex")
    if type(root_hex) is not str or raw.get(
        "randomization_root_sha256"
    ) != randomization_root_sha256(root_hex):
        raise SeedRollError("seed-roll reservation root commitment mismatch")
    return root_hex


def _load_reservation(
    path: Path,
    *,
    roll_identity_sha256: str,
    request: Mapping[str, object],
) -> str:
    try:
        payload = _read_regular_bytes(path)
        raw: Any = json.loads(payload)
    except (SeedRollError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedRollError(f"cannot read seed-roll reservation: {exc}") from exc
    root_hex = _validate_reservation(
        raw,
        roll_identity_sha256=roll_identity_sha256,
        request=request,
    )
    if payload != canonical_json_bytes(raw) + b"\n":
        raise SeedRollError("seed-roll reservation is not canonical JSON")
    return root_hex


def _create_or_load_reservation(
    authority_dir: Path,
    *,
    roll_identity_sha256: str,
    request: Mapping[str, object],
) -> str:
    path = _reservation_path(authority_dir, roll_identity_sha256)
    _real_directory(path.parent, create=True)
    if os.path.lexists(path):
        return _load_reservation(
            path,
            roll_identity_sha256=roll_identity_sha256,
            request=request,
        )
    # Reserve the identity before drawing.  A crash after ``open('xb')`` leaves
    # an invalid, blocking file rather than silently permitting another roll.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SeedRollError(f"cannot create seed-roll reservation: {exc}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        root_hex = secrets.token_bytes(SEED_ROLL_ROOT_BYTES).hex()
        reservation = {
            "schema_version": SEED_ROLL_RESERVATION_SCHEMA_VERSION,
            "roll_identity_sha256": roll_identity_sha256,
            "request": dict(request),
            "request_sha256": sha256_canonical_json(request),
            "randomization_root_hex": root_hex,
            "randomization_root_sha256": randomization_root_sha256(root_hex),
        }
        stream.write(canonical_json_bytes(reservation) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)
    _fsync_directory(path.parent)
    return root_hex


def _validate_authority_event(  # noqa: C901 - complete authority event gate
    raw: object,
    *,
    previous_event_sha256: str | None,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise SeedRollError("seed-roll authority event must be an object")
    required = {
        "schema_version",
        "event_type",
        "roll_identity_sha256",
        "request",
        "request_sha256",
        "randomization_root_sha256",
        "payload_sha256",
        "artifact_sha256",
        "artifact_relative_path",
        "previous_event_sha256",
        "at",
        "event_sha256",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != SEED_ROLL_AUTHORITY_SCHEMA_VERSION
        or raw.get("event_type") != "roll-created"
    ):
        raise SeedRollError("seed-roll authority event schema mismatch")
    for field in (
        "roll_identity_sha256",
        "request_sha256",
        "randomization_root_sha256",
        "payload_sha256",
        "artifact_sha256",
        "event_sha256",
    ):
        if not _is_sha256(raw.get(field)):
            raise SeedRollError(f"seed-roll authority {field} is invalid")
    request = raw.get("request")
    if not isinstance(request, Mapping) or raw.get(
        "request_sha256"
    ) != sha256_canonical_json(request):
        raise SeedRollError("seed-roll authority request binding mismatch")
    if raw.get("previous_event_sha256") != previous_event_sha256:
        raise SeedRollError("seed-roll authority hash chain is discontinuous")
    timestamp = raw.get("at")
    try:
        parsed_timestamp = (
            datetime.fromisoformat(timestamp) if type(timestamp) is str else None
        )
    except ValueError:
        parsed_timestamp = None
    if (
        parsed_timestamp is None
        or parsed_timestamp.tzinfo is None
        or parsed_timestamp.utcoffset() != UTC.utcoffset(parsed_timestamp)
    ):
        raise SeedRollError("seed-roll authority timestamp is invalid")
    prior = raw.get("previous_event_sha256")
    if prior is not None and not _is_sha256(prior):
        raise SeedRollError("seed-roll authority previous-event hash is invalid")
    relative = raw.get("artifact_relative_path")
    expected_relative = (
        f"{AUTHORITY_ARTIFACTS_DIR}/seed-roll-{raw['payload_sha256']}.json"
    )
    if relative != expected_relative:
        raise SeedRollError("seed-roll authority artifact path is invalid")
    event_body = {key: value for key, value in raw.items() if key != "event_sha256"}
    if raw.get("event_sha256") != sha256_canonical_json(event_body):
        raise SeedRollError("seed-roll authority event hash mismatch")
    return dict(raw)


def _load_authority_events(authority_dir: Path) -> list[dict[str, object]]:
    path = authority_dir / AUTHORITY_LEDGER_NAME
    if not os.path.lexists(path):
        return []
    events: list[dict[str, object]] = []
    seen_identities: set[str] = set()
    previous: str | None = None
    try:
        payload = _read_regular_bytes(path)
        if not payload:
            raise SeedRollError("seed-roll authority ledger is empty")
        for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
            raw: Any = json.loads(line)
            if line != canonical_json_bytes(raw) + b"\n":
                raise SeedRollError(
                    f"seed-roll authority line {line_number} is not canonical JSON"
                )
            event = _validate_authority_event(
                raw,
                previous_event_sha256=previous,
            )
            identity = cast(str, event["roll_identity_sha256"])
            if identity in seen_identities:
                raise SeedRollError(
                    "seed-roll authority repeats an experiment/phase identity"
                )
            seen_identities.add(identity)
            previous = cast(str, event["event_sha256"])
            events.append(event)
    except (SeedRollError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedRollError(f"cannot read seed-roll authority ledger: {exc}") from exc
    return events


def _append_authority_event(authority_dir: Path, event: Mapping[str, object]) -> None:
    path = authority_dir / AUTHORITY_LEDGER_NAME
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SeedRollError(f"cannot append seed-roll authority ledger: {exc}") from exc
    with os.fdopen(descriptor, "ab") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SeedRollError("seed-roll authority ledger is not a regular file")
        stream.write(canonical_json_bytes(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(authority_dir)


def _parse_plan_payload(payload: bytes) -> SeedRollPlan:
    try:
        raw: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedRollError(f"cannot parse seed-roll artifact: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SeedRollError("seed-roll artifact root must be an object")
    required = {
        "schema_version",
        "experiment_id",
        "phase",
        "logical_split",
        "source_segment",
        "source_range",
        "source_population",
        "seed_registry_sha256",
        "root_generator",
        "randomization_root_hex",
        "randomization_root_sha256",
        "source_selection_method",
        "replicate_allocation_method",
        "seat_assignment_method",
        "model_seed_method",
        "replicate_ids",
        "design_profile",
        "pilot_replicate_ids",
        "confirmatory_reserve_replicate_ids",
        "replicate_roles",
        "games_per_replicate",
        "selected_source_seed_count",
        "first_order_inclusion_probability",
        "per_replicate_first_order_inclusion_probability",
        "pooled_seat_specific_inclusion_probability",
        "per_replicate_seat_specific_inclusion_probability",
        "source_seed_union_sha256",
        "replicates",
    }
    if set(raw) != required or raw.get("schema_version") != SEED_ROLL_SCHEMA_VERSION:
        raise SeedRollError("seed-roll artifact schema mismatch")
    experiment_id = raw.get("experiment_id")
    phase = raw.get("phase")
    root_hex = raw.get("randomization_root_hex")
    replicate_ids = raw.get("replicate_ids")
    games = raw.get("games_per_replicate")
    if (
        type(experiment_id) is not str
        or type(phase) is not str
        or type(root_hex) is not str
        or not isinstance(replicate_ids, list)
        or any(type(value) is not int for value in replicate_ids)
        or type(games) is not int
    ):
        raise SeedRollError("seed-roll artifact has invalid reconstruction fields")
    expected = build_seed_roll_plan(
        experiment_id=experiment_id,
        phase=phase,
        replicate_ids=cast(list[int], replicate_ids),
        games_per_replicate=games,
        randomization_root_hex=root_hex,
    )
    if dict(raw) != expected.to_dict():
        raise SeedRollError("seed-roll artifact does not match its root-derived plan")
    if payload != canonical_json_bytes(expected.to_dict()) + b"\n":
        raise SeedRollError("seed-roll artifact is not canonical JSON")
    return expected


def _authorized_artifact(
    source: Path,
    authority_dir: Path,
    plan: SeedRollPlan,
    payload: bytes,
    event: Mapping[str, object],
) -> SeedRollArtifact:
    lexical_source = Path(os.path.abspath(os.fspath(source)))  # noqa: PTH100
    artifact_directory = _real_directory(
        authority_dir / AUTHORITY_ARTIFACTS_DIR,
        create=False,
    )
    if (
        lexical_source != source
        or lexical_source.resolve(strict=False) != lexical_source
        or lexical_source.parent != artifact_directory
    ):
        raise SeedRollError(
            "seed-roll artifact path traverses a symlink or leaves its authority"
        )
    source = lexical_source
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    roll_identity_sha256 = _roll_identity_sha256(plan.experiment_id, plan.phase)
    request = _roll_request(
        authority_dir=authority_dir,
        experiment_id=plan.experiment_id,
        phase=plan.phase,
        replicate_ids=plan.replicate_ids,
        games_per_replicate=plan.games_per_replicate,
    )
    reserved_root = _load_reservation(
        _reservation_path(authority_dir, roll_identity_sha256),
        roll_identity_sha256=roll_identity_sha256,
        request=request,
    )
    if reserved_root != plan.randomization_root_hex:
        raise SeedRollError(
            "seed-roll artifact root does not match its one-shot reservation"
        )
    if (
        event.get("roll_identity_sha256") != roll_identity_sha256
        or event.get("request") != request
        or event.get("randomization_root_sha256") != plan.randomization_root_sha256
        or event.get("payload_sha256") != plan.payload_sha256
        or event.get("artifact_sha256") != artifact_sha256
        or (authority_dir / cast(str, event["artifact_relative_path"])).resolve()
        != source.resolve()
    ):
        raise SeedRollError("seed-roll artifact does not match its authority event")
    if artifact_sha256 != hashlib.sha256(_read_regular_bytes(source)).hexdigest():
        raise SeedRollError("seed-roll artifact changed while being read")
    return SeedRollArtifact(
        path=source,
        authority_dir=authority_dir,
        plan=plan,
        payload_sha256=plan.payload_sha256,
        artifact_sha256=artifact_sha256,
        roll_identity_sha256=roll_identity_sha256,
        authority_event_sha256=cast(str, event["event_sha256"]),
        authority_event=dict(event),
    )


def create_seed_roll_artifact(
    authority_dir: Path,
    *,
    experiment_id: str,
    phase: str,
    replicate_ids: Sequence[int],
    games_per_replicate: int,
) -> SeedRollArtifact:
    """Create or recover the sole authorized roll for an experiment/phase.

    A fixed identity reservation is written before the CSPRNG root is exposed.
    The append-only, hash-chained authority ledger then prevents another path or
    filename within the same authority from being used to reroll that identity.
    """
    identities = _validate_roll_request(
        experiment_id,
        phase,
        replicate_ids,
        games_per_replicate,
    )
    authority = _real_directory(Path(authority_dir), create=True)
    request = _roll_request(
        authority_dir=authority,
        experiment_id=experiment_id,
        phase=phase,
        replicate_ids=identities,
        games_per_replicate=games_per_replicate,
    )
    roll_identity_sha256 = _roll_identity_sha256(experiment_id, phase)
    with _authority_lock(authority):
        events = _load_authority_events(authority)
        existing = next(
            (
                event
                for event in events
                if event["roll_identity_sha256"] == roll_identity_sha256
            ),
            None,
        )
        if existing is not None:
            if existing.get("request") != request:
                raise SeedRollError(
                    "seed-roll identity is already committed to another design"
                )
            source = authority / cast(str, existing["artifact_relative_path"])
            artifact_directory = _real_directory(
                authority / AUTHORITY_ARTIFACTS_DIR,
                create=False,
            )
            if source.parent != artifact_directory:
                raise SeedRollError(
                    "seed-roll authority event leaves its real artifact directory"
                )
            payload = _read_regular_bytes(source)
            plan = _parse_plan_payload(payload)
            return _authorized_artifact(source, authority, plan, payload, existing)

        root_hex = _create_or_load_reservation(
            authority,
            roll_identity_sha256=roll_identity_sha256,
            request=request,
        )
        plan = build_seed_roll_plan(
            experiment_id=experiment_id,
            phase=phase,
            replicate_ids=identities,
            games_per_replicate=games_per_replicate,
            randomization_root_hex=root_hex,
        )
        payload = canonical_json_bytes(plan.to_dict()) + b"\n"
        destination = _artifact_path(authority, plan.payload_sha256)
        if os.path.lexists(destination):
            if _read_regular_bytes(destination) != payload:
                raise SeedRollError(
                    "seed-roll content address is occupied by different bytes"
                )
        else:
            _write_exclusive(destination, payload, mode=0o444)
        event_body: dict[str, object] = {
            "schema_version": SEED_ROLL_AUTHORITY_SCHEMA_VERSION,
            "event_type": "roll-created",
            "roll_identity_sha256": roll_identity_sha256,
            "request": request,
            "request_sha256": sha256_canonical_json(request),
            "randomization_root_sha256": plan.randomization_root_sha256,
            "payload_sha256": plan.payload_sha256,
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "artifact_relative_path": str(destination.relative_to(authority)),
            "previous_event_sha256": (events[-1]["event_sha256"] if events else None),
            "at": _now(),
        }
        event_body["event_sha256"] = sha256_canonical_json(event_body)
        event = _validate_authority_event(
            event_body,
            previous_event_sha256=cast(
                str | None,
                events[-1]["event_sha256"] if events else None,
            ),
        )
        _append_authority_event(authority, event)
        return _authorized_artifact(destination, authority, plan, payload, event)


def load_seed_roll_artifact(
    path: Path,
    *,
    authority_dir: Path | None = None,
) -> SeedRollArtifact:
    """Load a root-derived artifact and verify its one-shot authority event."""
    lexical_source = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    if lexical_source.resolve(strict=False) != lexical_source:
        raise SeedRollError("seed-roll artifact path traverses a symlink")
    source = lexical_source
    authority = _real_directory(
        Path(authority_dir) if authority_dir is not None else source.parent.parent,
        create=False,
    )
    if source.parent != authority / AUTHORITY_ARTIFACTS_DIR:
        raise SeedRollError("seed-roll artifact is outside its authority directory")
    with _authority_lock(authority):
        payload = _read_regular_bytes(source)
        plan = _parse_plan_payload(payload)
        roll_identity_sha256 = _roll_identity_sha256(plan.experiment_id, plan.phase)
        events = _load_authority_events(authority)
        matches = [
            event
            for event in events
            if event["roll_identity_sha256"] == roll_identity_sha256
        ]
        if len(matches) != 1:
            raise SeedRollError(
                "seed-roll authority must contain exactly one matching event"
            )
        return _authorized_artifact(source, authority, plan, payload, matches[0])


def _reload_authorized_artifact(artifact: SeedRollArtifact) -> SeedRollArtifact:
    reloaded = load_seed_roll_artifact(
        artifact.path,
        authority_dir=artifact.authority_dir,
    )
    if reloaded.manifest_binding() != artifact.manifest_binding():
        raise SeedRollError("seed-roll artifact changed after it was loaded")
    return reloaded


def validate_seed_roll_binding(  # noqa: C901 - every authority field fails closed
    raw: object,
) -> dict[str, object]:
    """Strictly validate the self-contained manifest representation."""
    if not isinstance(raw, Mapping):
        raise SeedRollError("manifest seed_roll must be a mapping")
    required = {
        "protocol",
        "experiment_id",
        "phase",
        "authority_dir",
        "payload_sha256",
        "artifact_sha256",
        "roll_identity_sha256",
        "authority_event_sha256",
        "authority_event",
        "randomization_root_hex",
        "randomization_root_sha256",
        "logical_split",
        "source_segment",
        "source_range",
        "source_population",
        "seed_registry_sha256",
        "source_selection_method",
        "replicate_allocation_method",
        "seat_assignment_method",
        "model_seed_method",
        "replicate_ids",
        "design_profile",
        "pilot_replicate_ids",
        "confirmatory_reserve_replicate_ids",
        "replicate_roles",
        "games_per_replicate",
        "selected_source_seed_count",
        "first_order_inclusion_probability",
        "per_replicate_first_order_inclusion_probability",
        "pooled_seat_specific_inclusion_probability",
        "per_replicate_seat_specific_inclusion_probability",
        "source_seed_union_sha256",
        "model_seed63_by_replicate",
        "model_seed_digest_by_replicate",
    }
    if set(raw) != required or raw.get("protocol") != SEED_ROLL_SCHEMA_VERSION:
        raise SeedRollError("manifest seed_roll schema mismatch")
    if any(
        not _is_sha256(raw.get(field))
        for field in (
            "payload_sha256",
            "artifact_sha256",
            "roll_identity_sha256",
            "authority_event_sha256",
            "randomization_root_sha256",
            "seed_registry_sha256",
            "source_seed_union_sha256",
        )
    ):
        raise SeedRollError("manifest seed_roll contains an invalid SHA-256")
    root_hex = raw.get("randomization_root_hex")
    if type(root_hex) is not str or randomization_root_sha256(root_hex) != raw.get(
        "randomization_root_sha256"
    ):
        raise SeedRollError("manifest seed_roll root commitment mismatch")
    experiment_id = raw.get("experiment_id")
    phase = raw.get("phase")
    authority_dir = raw.get("authority_dir")
    replicate_ids = raw.get("replicate_ids")
    games = raw.get("games_per_replicate")
    if (
        type(experiment_id) is not str
        or not experiment_id
        or type(phase) is not str
        or not phase
        or type(authority_dir) is not str
        or not Path(authority_dir).is_absolute()
        or os.path.normpath(authority_dir) != authority_dir
        or Path(authority_dir).resolve(strict=False) != Path(authority_dir)
        or not isinstance(replicate_ids, list)
        or any(type(value) is not int for value in replicate_ids)
        or type(games) is not int
    ):
        raise SeedRollError("manifest seed_roll replicate declaration is invalid")
    expected_plan = build_seed_roll_plan(
        experiment_id=experiment_id,
        phase=phase,
        replicate_ids=cast(list[int], replicate_ids),
        games_per_replicate=games,
        randomization_root_hex=cast(str, root_hex),
    )
    canonical_payload = canonical_json_bytes(expected_plan.to_dict()) + b"\n"
    raw_authority_event = raw.get("authority_event")
    if not isinstance(raw_authority_event, Mapping):
        raise SeedRollError("manifest seed_roll authority event is invalid")
    authority_event = _validate_authority_event(
        raw_authority_event,
        previous_event_sha256=cast(
            str | None,
            raw_authority_event.get("previous_event_sha256"),
        ),
    )
    if authority_event["event_sha256"] != raw.get("authority_event_sha256"):
        raise SeedRollError("manifest seed_roll authority-event binding mismatch")
    expected_request = _roll_request(
        authority_dir=Path(cast(str, authority_dir)),
        experiment_id=experiment_id,
        phase=phase,
        replicate_ids=expected_plan.replicate_ids,
        games_per_replicate=expected_plan.games_per_replicate,
    )
    expected_artifact_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    if (
        authority_event.get("roll_identity_sha256")
        != _roll_identity_sha256(experiment_id, phase)
        or authority_event.get("request") != expected_request
        or authority_event.get("request_sha256")
        != sha256_canonical_json(expected_request)
        or authority_event.get("randomization_root_sha256")
        != expected_plan.randomization_root_sha256
        or authority_event.get("payload_sha256") != expected_plan.payload_sha256
        or authority_event.get("artifact_sha256") != expected_artifact_sha256
    ):
        raise SeedRollError(
            "manifest seed_roll authority event does not match the root-derived plan"
        )
    expected = SeedRollArtifact(
        path=Path("seed-roll.json"),
        authority_dir=Path(cast(str, authority_dir)),
        plan=expected_plan,
        payload_sha256=expected_plan.payload_sha256,
        artifact_sha256=expected_artifact_sha256,
        roll_identity_sha256=_roll_identity_sha256(experiment_id, phase),
        authority_event_sha256=cast(str, raw["authority_event_sha256"]),
        authority_event=authority_event,
    ).manifest_binding()
    if dict(raw) != expected:
        raise SeedRollError("manifest seed_roll does not match its root-derived plan")
    # A manifest approval is an authority decision, not merely a check that an
    # embedded event can hash itself.  Re-read the bound reservation, ledger,
    # and content-addressed artifact so a fabricated-but-self-consistent event
    # cannot enter an approved declaration.  Runtime gates repeat this check.
    authority = _real_directory(Path(cast(str, authority_dir)), create=False)
    authorized = load_seed_roll_artifact(
        _artifact_path(authority, expected_plan.payload_sha256),
        authority_dir=authority,
    )
    if authorized.manifest_binding() != expected:
        raise SeedRollError(
            "manifest seed_roll is not anchored in its declared authority"
        )
    return dict(raw)


def require_seed_roll_binding(
    artifact: SeedRollArtifact,
    raw: object,
) -> None:
    """Require one manifest binding to equal the verified artifact exactly."""
    binding = validate_seed_roll_binding(raw)
    reloaded = _reload_authorized_artifact(artifact)
    if (
        reloaded.manifest_binding() != artifact.manifest_binding()
        or binding != reloaded.manifest_binding()
    ):
        raise SeedRollError("manifest seed_roll does not match its immutable artifact")


def require_task1_formal_seed_roll(artifact: SeedRollArtifact) -> SeedRollArtifact:
    """Require the exact pre-registered 5 x 32,000 production design.

    Smaller deterministic rolls remain useful for CI and carry the explicit
    ``ci-fixture`` profile, but no optimizer may consume them as a Task-1
    formal run.
    """
    reloaded = _reload_authorized_artifact(artifact)
    if (
        reloaded.plan.phase != "T1.4"
        or reloaded.plan.design_profile != TASK1_FORMAL_DESIGN_PROFILE
        or reloaded.plan.replicate_ids != TASK1_FORMAL_REPLICATE_IDS
        or reloaded.plan.games_per_replicate != TASK1_FORMAL_GAMES_PER_REPLICATE
        or reloaded.plan.pilot_replicate_ids
        != TASK1_FORMAL_REPLICATE_IDS[:TASK1_PILOT_REPLICATE_COUNT]
        or reloaded.plan.confirmatory_reserve_replicate_ids
        != TASK1_FORMAL_REPLICATE_IDS[TASK1_PILOT_REPLICATE_COUNT:]
    ):
        raise SeedRollError(
            "Task-1 formal training requires the pre-registered 5 x 32,000 "
            "seed-roll design"
        )
    return reloaded


def materialize_seed_roll_scenarios(
    artifact: SeedRollArtifact,
    active_replicate_ids: Sequence[int],
) -> RolledScenarioSelection:
    """Generate immutable ScenarioV1 rows for an approved subset of replicates."""
    design = seed_roll_selection_design(artifact, active_replicate_ids)
    return RolledScenarioSelection(
        tuple(iter_seed_roll_scenarios(artifact, design)),
        design,
    )


def seed_roll_selection_design(
    artifact: SeedRollArtifact,
    active_replicate_ids: Sequence[int],
) -> SeedRollSelectionDesign:
    """Create the finite-population design for one frozen replicate subset."""
    artifact = _reload_authorized_artifact(artifact)
    identities = tuple(active_replicate_ids)
    if (
        not identities
        or identities != tuple(sorted(set(identities)))
        or any(value not in artifact.plan.replicate_ids for value in identities)
    ):
        raise SeedRollError("active replicate IDs must be a sorted reserved subset")
    return SeedRollSelectionDesign(
        seed_roll_payload_sha256=artifact.payload_sha256,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
        active_replicate_ids=identities,
        games_per_replicate=artifact.plan.games_per_replicate,
        source_population=TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start,
    )


def iter_seed_roll_scenarios(
    artifact: SeedRollArtifact,
    design: SeedRollSelectionDesign,
) -> Iterator[ScenarioV1]:
    """Yield a rolled bank in canonical source-seed order with bounded memory."""
    if (
        design.seed_roll_payload_sha256 != artifact.payload_sha256
        or design.randomization_root_sha256 != artifact.plan.randomization_root_sha256
        or design.games_per_replicate != artifact.plan.games_per_replicate
        or any(
            replicate_id not in artifact.plan.replicate_ids
            for replicate_id in design.active_replicate_ids
        )
    ):
        raise SeedRollError("natural-deal SRSWOR design does not match its seed roll")
    assignments = sorted(
        (
            source_seed,
            replicate_id,
        )
        for replicate_id in design.active_replicate_ids
        for source_seed in artifact.plan.replicate(replicate_id).source_seeds
    )
    for source_seed, replicate_id in assignments:
        source = generate_scenario(
            TASK1_TRAIN_SCHEDULE.name,
            source_seed,
            selection_kind="iid",
        )
        yield with_sampling_metadata(
            source,
            selection_kind="natural-deal-srswor",
            inclusion_probability=design.inclusion_probability,
            selection_stratum=f"replicate-{replicate_id}",
            selection_design_sha256=design.sha256,
        )


def validate_rolled_scenario_selection(
    scenarios: Sequence[ScenarioV1],
    design: SeedRollSelectionDesign,
) -> None:
    """Validate row-level design metadata and exact replicate counts."""
    if len(scenarios) != design.selected_count:
        raise SeedRollError(
            "natural-deal SRSWOR scenario count does not match its design"
        )
    counts: Counter[str] = Counter()
    source_seeds: set[int] = set()
    scenario_ids: set[str] = set()
    for scenario in scenarios:
        if scenario.selection_kind != "natural-deal-srswor":
            raise SeedRollError(
                "natural-deal SRSWOR design contains a non-rolled scenario"
            )
        if scenario.selection_design_sha256 != design.sha256:
            raise SeedRollError("natural-deal SRSWOR scenario design hash mismatch")
        if scenario.inclusion_probability != design.inclusion_probability:
            raise SeedRollError(
                "natural-deal SRSWOR scenario inclusion probability mismatch"
            )
        if scenario.selection_stratum is None:
            raise SeedRollError("natural-deal SRSWOR scenario has no replicate stratum")
        counts[scenario.selection_stratum] += 1
        if scenario.source_seed in source_seeds or scenario.scenario_id in scenario_ids:
            raise SeedRollError("natural-deal SRSWOR scenario rows are not unique")
        source_seeds.add(scenario.source_seed)
        scenario_ids.add(scenario.scenario_id)
    expected = {
        f"replicate-{replicate_id}": design.games_per_replicate
        for replicate_id in design.active_replicate_ids
    }
    if dict(counts) != expected:
        raise SeedRollError(
            "natural-deal SRSWOR replicate strata do not match the design"
        )


def validate_scenarios_against_seed_roll(  # noqa: C901,PLR0912 - full replay gate
    artifact: SeedRollArtifact,
    scenarios: Sequence[ScenarioV1],
    active_replicate_ids: Sequence[int],
    *,
    selection_design: SeedRollSelectionDesign | None = None,
    allow_active_replicate_subset: bool = False,
) -> SeedRollSelectionDesign:
    """Require materialized rows to be exactly the root-derived assignments."""
    artifact = _reload_authorized_artifact(artifact)
    identities = tuple(active_replicate_ids)
    if (
        not identities
        or identities != tuple(sorted(set(identities)))
        or any(value not in artifact.plan.replicate_ids for value in identities)
    ):
        raise SeedRollError("scenario validation requires a sorted reserved subset")
    design = selection_design or SeedRollSelectionDesign(
        seed_roll_payload_sha256=artifact.payload_sha256,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
        active_replicate_ids=identities,
        games_per_replicate=artifact.plan.games_per_replicate,
        source_population=TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start,
    )
    active_ids_match = design.active_replicate_ids == identities
    if allow_active_replicate_subset:
        active_ids_match = set(identities) <= set(design.active_replicate_ids)
    if (
        design.seed_roll_payload_sha256 != artifact.payload_sha256
        or design.randomization_root_sha256 != artifact.plan.randomization_root_sha256
        or design.games_per_replicate != artifact.plan.games_per_replicate
        or design.source_population
        != TASK1_TRAIN_SCHEDULE.end - TASK1_TRAIN_SCHEDULE.start
        or any(
            replicate_id not in artifact.plan.replicate_ids
            for replicate_id in design.active_replicate_ids
        )
        or not active_ids_match
        or len(scenarios) != len(identities) * artifact.plan.games_per_replicate
    ):
        raise SeedRollError("scenario bank design does not match the seed roll")
    actual: dict[int, set[int]] = {replicate_id: set() for replicate_id in identities}
    seen_sources: set[int] = set()
    seen_scenario_ids: set[str] = set()
    seen_state_hashes: set[str] = set()
    for scenario in scenarios:
        try:
            validate_scenario(scenario)
        except ScenarioValidationError as exc:
            raise SeedRollError(
                f"natural-deal SRSWOR ScenarioV1 is invalid: {exc}"
            ) from exc
        if (
            scenario.source_segment != TASK1_TRAIN_SCHEDULE.name
            or scenario.source_seed
            not in range(TASK1_TRAIN_SCHEDULE.start, TASK1_TRAIN_SCHEDULE.end)
            or scenario.selection_kind != "natural-deal-srswor"
            or scenario.selection_design_sha256 != design.sha256
            or scenario.inclusion_probability != design.inclusion_probability
            or scenario.selection_stratum is None
        ):
            raise SeedRollError(
                "natural-deal SRSWOR scenario metadata does not match the design"
            )
        try:
            replicate_id = int(scenario.selection_stratum.removeprefix("replicate-"))
        except ValueError as exc:
            raise SeedRollError(
                "natural-deal SRSWOR scenario has an invalid replicate stratum"
            ) from exc
        if replicate_id not in actual:
            raise SeedRollError(
                "natural-deal SRSWOR scenario uses an inactive replicate"
            )
        if scenario.selection_stratum != f"replicate-{replicate_id}":
            raise SeedRollError("natural-deal SRSWOR scenario stratum is not canonical")
        if (
            scenario.source_seed in seen_sources
            or scenario.scenario_id in seen_scenario_ids
            or scenario.canonical_state_sha256 in seen_state_hashes
        ):
            raise SeedRollError(
                "natural-deal SRSWOR scenarios repeat a source or state"
            )
        seen_sources.add(scenario.source_seed)
        seen_scenario_ids.add(scenario.scenario_id)
        seen_state_hashes.add(scenario.canonical_state_sha256)
        regenerated = generate_scenario(
            TASK1_TRAIN_SCHEDULE.name,
            scenario.source_seed,
            selection_kind="iid",
        )
        if (
            regenerated.scenario_id != scenario.scenario_id
            or regenerated.canonical_state_sha256 != scenario.canonical_state_sha256
            or regenerated.state_payload() != scenario.state_payload()
        ):
            raise SeedRollError(
                "natural-deal SRSWOR ScenarioV1 does not match its registered source seed"
            )
        actual[replicate_id].add(scenario.source_seed)
    for replicate_id in identities:
        expected = set(artifact.plan.replicate(replicate_id).source_seeds)
        if actual[replicate_id] != expected:
            raise SeedRollError(
                f"replicate {replicate_id} scenarios do not match the seed roll"
            )
    return design


def validate_seed_rolled_training_schedule(  # noqa: PLR0913 - protocol axes explicit
    artifact: SeedRollArtifact,
    scenarios: Sequence[ScenarioV1],
    rows: Sequence[PairedTrainingRow],
    active_replicate_ids: Sequence[int],
    *,
    expected_treatments: Sequence[str],
    selection_design: SeedRollSelectionDesign | None = None,
    allow_active_replicate_subset: bool = False,
) -> SeedRollSelectionDesign:
    """Bind schedule coordinates to the root-derived scenario and seat order.

    Generic paired-schedule validation proves CRN consistency, but cannot know
    which registered source seed belongs at a particular ordinal.  This gate
    closes that gap by comparing every treatment row with the immutable roll.
    """
    artifact = _reload_authorized_artifact(artifact)
    try:
        validate_paired_training_schedule(
            rows,
            expected_treatments=expected_treatments,
        )
    except ValueError as exc:
        raise SeedRollError(f"seed-rolled training schedule is invalid: {exc}") from exc
    identities = tuple(active_replicate_ids)
    declared_treatments = tuple(expected_treatments)
    treatments = tuple(sorted(declared_treatments))
    if (
        not declared_treatments
        or len(set(declared_treatments)) != len(declared_treatments)
        or {row.replicate_id for row in rows} != set(identities)
        or {row.treatment_id for row in rows} != set(treatments)
        or {(row.experiment_id, row.phase) for row in rows}
        != {(artifact.plan.experiment_id, artifact.plan.phase)}
        or {row.randomization_root_sha256 for row in rows}
        != {artifact.plan.randomization_root_sha256}
    ):
        raise SeedRollError("training schedule identity does not match the seed roll")
    design = validate_scenarios_against_seed_roll(
        artifact,
        scenarios,
        identities,
        selection_design=selection_design,
        allow_active_replicate_subset=allow_active_replicate_subset,
    )
    scenario_by_source = {scenario.source_seed: scenario for scenario in scenarios}
    for replicate_id in identities:
        reserved = artifact.plan.replicate(replicate_id)
        expected = tuple(
            (scenario_by_source[source_seed].scenario_id, seat)
            for source_seed, seat in zip(
                reserved.source_seeds,
                reserved.seats,
                strict=True,
            )
        )
        for treatment_id in treatments:
            treatment_rows = tuple(
                sorted(
                    (
                        row
                        for row in rows
                        if row.replicate_id == replicate_id
                        and row.treatment_id == treatment_id
                    ),
                    key=lambda row: (row.update, row.game_index),
                )
            )
            actual = tuple((row.scenario_id, row.seat) for row in treatment_rows)
            if actual != expected:
                raise SeedRollError(
                    "training schedule scenario/seat order does not match the "
                    f"seed roll for replicate {replicate_id}"
                )
    return design


def make_seed_rolled_training_schedule(  # noqa: PLR0913 - protocol axes explicit
    artifact: SeedRollArtifact,
    scenarios: Sequence[ScenarioV1],
    *,
    active_replicate_ids: Sequence[int],
    treatment_ids: Sequence[str],
    updates: int,
    games_per_update: int,
    selection_design: SeedRollSelectionDesign | None = None,
) -> tuple[PairedTrainingRow, ...]:
    """Build the only accepted schedule from root-derived rows and seats."""
    identities = tuple(active_replicate_ids)
    design = validate_scenarios_against_seed_roll(
        artifact,
        scenarios,
        identities,
        selection_design=selection_design,
    )
    if artifact.plan.games_per_replicate != updates * games_per_update:
        raise SeedRollError("seed roll does not match the requested training budget")
    scenario_by_source = {scenario.source_seed: scenario for scenario in scenarios}
    scenarios_by_replicate = {
        replicate_id: tuple(
            scenario_by_source[source_seed].scenario_id
            for source_seed in artifact.plan.replicate(replicate_id).source_seeds
        )
        for replicate_id in identities
    }
    seats_by_replicate = {
        replicate_id: artifact.plan.replicate(replicate_id).seats
        for replicate_id in identities
    }
    rows = make_paired_training_schedule(
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        treatment_ids=treatment_ids,
        scenarios_by_replicate=scenarios_by_replicate,
        seats_by_replicate=seats_by_replicate,
        updates=updates,
        games_per_update=games_per_update,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
    )
    if design.active_replicate_ids != identities:
        raise SeedRollError(
            "training schedule must consume the bank's exact active replicates"
        )
    return rows


def _activation_file_path(path: Path, *, label: str, create_parent: bool) -> Path:
    """Return one absolute, normalized path with no symlink traversal."""
    supplied = Path(path)
    lexical = Path(os.path.abspath(os.fspath(supplied)))  # noqa: PTH100
    if (
        not supplied.is_absolute()
        or os.path.normpath(os.fspath(supplied)) != os.fspath(supplied)
        or lexical != supplied
        or lexical.resolve(strict=False) != lexical
    ):
        raise SeedRollError(f"{label} must be a normalized absolute non-symlink path")
    _real_directory(lexical.parent, create=create_parent)
    return lexical


def _utc_datetime(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value) if type(value) is str else None
    except ValueError:
        parsed = None
    if (
        parsed is None
        or parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
    ):
        raise SeedRollError(f"{label} timestamp is invalid")
    return parsed


def _validate_activation_pilot_manifest(
    path: Path,
    *,
    experiment_id: str,
    phase: str,
    seed_roll_payload_sha256: str,
    pilot_replicate_ids: tuple[int, ...],
) -> tuple[str, str, str, datetime]:
    """Verify the exact completed pilot manifest bound by an activation."""
    source = _activation_file_path(
        path,
        label="pilot manifest",
        create_parent=False,
    )
    payload = _read_regular_bytes(source)
    try:
        raw: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedRollError(f"cannot parse activation pilot manifest: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SeedRollError("activation pilot manifest root must be an object")
    # Keep the import local: manifest validation itself lazily imports this
    # module for seed-roll checks.
    from .manifest import validate_manifest  # noqa: PLC0415

    try:
        validate_manifest(raw)
    except ValueError as exc:
        raise SeedRollError(f"activation pilot manifest is invalid: {exc}") from exc
    declaration = raw.get("declaration")
    lifecycle = raw.get("lifecycle")
    if not isinstance(declaration, Mapping) or not isinstance(lifecycle, list):
        raise SeedRollError("activation pilot manifest is missing v2 declarations")
    formal_training = declaration.get("formal_training")
    seed_plan = declaration.get("seed_plan")
    roll_binding = (
        seed_plan.get("seed_roll") if isinstance(seed_plan, Mapping) else None
    )
    if (
        raw.get("status") != "completed"
        or declaration.get("experiment_id") != experiment_id
        or declaration.get("phase") != phase
        or not isinstance(formal_training, Mapping)
        or formal_training.get("protocol") != "paired-training-v2"
        or formal_training.get("replicate_stage") != "pilot"
        or formal_training.get("replicate_ids") != list(pilot_replicate_ids)
        or formal_training.get("activation_artifact_path") is not None
        or formal_training.get("activation_artifact_sha256") is not None
        or not isinstance(roll_binding, Mapping)
        or roll_binding.get("payload_sha256") != seed_roll_payload_sha256
        or not lifecycle
        or not isinstance(lifecycle[-1], Mapping)
        or lifecycle[-1].get("to") != "completed"
    ):
        raise SeedRollError(
            "activation requires the completed pilot manifest for the same seed roll"
        )
    declaration_sha256 = raw.get("declaration_sha256")
    completed_event_sha256 = lifecycle[-1].get("event_sha256")
    if not _is_sha256(declaration_sha256) or not _is_sha256(completed_event_sha256):
        raise SeedRollError("activation pilot manifest hashes are invalid")
    completed_at = _utc_datetime(
        lifecycle[-1].get("at"),
        label="pilot completion",
    )
    return (
        hashlib.sha256(payload).hexdigest(),
        cast(str, declaration_sha256),
        cast(str, completed_event_sha256),
        completed_at,
    )


def create_confirmatory_activation_artifact(  # noqa: PLR0913 - evidence axes explicit
    path: Path,
    *,
    seed_roll: SeedRollArtifact,
    pilot_manifest_path: Path,
    pilot_evidence_path: Path,
    actor: str,
    note: str,
) -> ConfirmatoryActivationArtifact:
    """Write a one-shot post-pilot decision that activates reserved replicates."""
    artifact = _reload_authorized_artifact(seed_roll)
    if type(actor) is not str or not actor or type(note) is not str or not note:
        raise SeedRollError(
            "confirmatory activation requires a non-empty actor and note"
        )
    destination = _activation_file_path(
        path,
        label="confirmatory activation artifact",
        create_parent=True,
    )
    pilot_manifest = _activation_file_path(
        pilot_manifest_path,
        label="pilot manifest",
        create_parent=False,
    )
    evidence = _activation_file_path(
        pilot_evidence_path,
        label="pilot evidence",
        create_parent=False,
    )
    if len({destination, pilot_manifest, evidence}) != ACTIVATION_BOUND_PATH_COUNT:
        raise SeedRollError("activation, manifest, and evidence paths must be distinct")
    (
        pilot_manifest_sha256,
        pilot_declaration_sha256,
        pilot_completed_event_sha256,
        pilot_completed_at,
    ) = _validate_activation_pilot_manifest(
        pilot_manifest,
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        seed_roll_payload_sha256=artifact.payload_sha256,
        pilot_replicate_ids=artifact.plan.pilot_replicate_ids,
    )
    evidence_bytes = _read_regular_bytes(evidence)
    if not evidence_bytes:
        raise SeedRollError("confirmatory activation pilot evidence is empty")
    activated_at = _now()
    if _utc_datetime(activated_at, label="activation") < pilot_completed_at:
        raise SeedRollError("confirmatory activation predates pilot completion")
    body: dict[str, object] = {
        "protocol": CONFIRMATORY_ACTIVATION_SCHEMA_VERSION,
        "experiment_id": artifact.plan.experiment_id,
        "phase": artifact.plan.phase,
        "seed_roll_payload_sha256": artifact.payload_sha256,
        "pilot_replicate_ids": list(artifact.plan.pilot_replicate_ids),
        "confirmatory_replicate_ids": list(
            artifact.plan.confirmatory_reserve_replicate_ids
        ),
        "pilot_manifest_path": str(pilot_manifest),
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "pilot_manifest_declaration_sha256": pilot_declaration_sha256,
        "pilot_completed_event_sha256": pilot_completed_event_sha256,
        "pilot_evidence_path": str(evidence),
        "pilot_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "decision": "activate-confirmatory-reserve",
        "actor": actor,
        "note": note,
        "at": activated_at,
    }
    payload = canonical_json_bytes(body) + b"\n"
    _write_exclusive(destination, payload, mode=0o444)
    return load_confirmatory_activation_artifact(
        destination,
        expected_experiment_id=artifact.plan.experiment_id,
        expected_phase=artifact.plan.phase,
        expected_seed_roll_payload_sha256=artifact.payload_sha256,
        expected_pilot_replicate_ids=artifact.plan.pilot_replicate_ids,
        expected_confirmatory_replicate_ids=(
            artifact.plan.confirmatory_reserve_replicate_ids
        ),
    )


def load_confirmatory_activation_artifact(  # noqa: C901,PLR0913 - strict gate
    path: Path,
    *,
    expected_experiment_id: str,
    expected_phase: str,
    expected_seed_roll_payload_sha256: str,
    expected_pilot_replicate_ids: Sequence[int],
    expected_confirmatory_replicate_ids: Sequence[int],
) -> ConfirmatoryActivationArtifact:
    """Load and revalidate an activation plus its completed-pilot evidence."""
    source = _activation_file_path(
        path,
        label="confirmatory activation artifact",
        create_parent=False,
    )
    payload = _read_regular_bytes(source)
    try:
        raw: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedRollError(f"cannot parse confirmatory activation: {exc}") from exc
    required = {
        "protocol",
        "experiment_id",
        "phase",
        "seed_roll_payload_sha256",
        "pilot_replicate_ids",
        "confirmatory_replicate_ids",
        "pilot_manifest_path",
        "pilot_manifest_sha256",
        "pilot_manifest_declaration_sha256",
        "pilot_completed_event_sha256",
        "pilot_evidence_path",
        "pilot_evidence_sha256",
        "decision",
        "actor",
        "note",
        "at",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("protocol") != CONFIRMATORY_ACTIVATION_SCHEMA_VERSION
        or payload != canonical_json_bytes(raw) + b"\n"
    ):
        raise SeedRollError("confirmatory activation schema/canonical form mismatch")
    pilot_ids = tuple(expected_pilot_replicate_ids)
    reserve_ids = tuple(expected_confirmatory_replicate_ids)
    if (
        raw.get("experiment_id") != expected_experiment_id
        or raw.get("phase") != expected_phase
        or raw.get("seed_roll_payload_sha256") != expected_seed_roll_payload_sha256
        or raw.get("pilot_replicate_ids") != list(pilot_ids)
        or raw.get("confirmatory_replicate_ids") != list(reserve_ids)
        or not pilot_ids
        or not reserve_ids
        or raw.get("decision") != "activate-confirmatory-reserve"
        or type(raw.get("actor")) is not str
        or not raw.get("actor")
        or type(raw.get("note")) is not str
        or not raw.get("note")
        or type(raw.get("pilot_manifest_path")) is not str
        or type(raw.get("pilot_evidence_path")) is not str
    ):
        raise SeedRollError(
            "confirmatory activation does not match the reserved design"
        )
    for field in (
        "seed_roll_payload_sha256",
        "pilot_manifest_sha256",
        "pilot_manifest_declaration_sha256",
        "pilot_completed_event_sha256",
        "pilot_evidence_sha256",
    ):
        if not _is_sha256(raw.get(field)):
            raise SeedRollError(f"confirmatory activation {field} is invalid")
    pilot_manifest = _activation_file_path(
        Path(cast(str, raw.get("pilot_manifest_path"))),
        label="pilot manifest",
        create_parent=False,
    )
    evidence = _activation_file_path(
        Path(cast(str, raw.get("pilot_evidence_path"))),
        label="pilot evidence",
        create_parent=False,
    )
    if len({source, pilot_manifest, evidence}) != ACTIVATION_BOUND_PATH_COUNT:
        raise SeedRollError("activation, manifest, and evidence paths must be distinct")
    (
        actual_manifest_sha256,
        actual_declaration_sha256,
        actual_completed_event_sha256,
        pilot_completed_at,
    ) = _validate_activation_pilot_manifest(
        pilot_manifest,
        experiment_id=expected_experiment_id,
        phase=expected_phase,
        seed_roll_payload_sha256=expected_seed_roll_payload_sha256,
        pilot_replicate_ids=pilot_ids,
    )
    evidence_bytes = _read_regular_bytes(evidence)
    if not evidence_bytes:
        raise SeedRollError("confirmatory activation pilot evidence is empty")
    if (
        raw.get("pilot_manifest_sha256") != actual_manifest_sha256
        or raw.get("pilot_manifest_declaration_sha256") != actual_declaration_sha256
        or raw.get("pilot_completed_event_sha256") != actual_completed_event_sha256
        or raw.get("pilot_evidence_sha256")
        != hashlib.sha256(evidence_bytes).hexdigest()
    ):
        raise SeedRollError("confirmatory activation evidence changed after approval")
    activated_at = _utc_datetime(raw.get("at"), label="activation")
    if activated_at < pilot_completed_at:
        raise SeedRollError("confirmatory activation predates pilot completion")
    # Catch a replacement between the initial read and the completed evidence
    # audit.  This does not turn a local filesystem into an external WORM
    # authority, but it closes ordinary path/TOCTOU mistakes.
    if _read_regular_bytes(source) != payload:
        raise SeedRollError("confirmatory activation changed while being read")
    return ConfirmatoryActivationArtifact(
        path=source,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        experiment_id=expected_experiment_id,
        phase=expected_phase,
        seed_roll_payload_sha256=expected_seed_roll_payload_sha256,
        pilot_replicate_ids=pilot_ids,
        confirmatory_replicate_ids=reserve_ids,
        pilot_manifest_path=pilot_manifest,
        pilot_manifest_sha256=actual_manifest_sha256,
        pilot_manifest_declaration_sha256=actual_declaration_sha256,
        pilot_completed_event_sha256=actual_completed_event_sha256,
        pilot_evidence_path=evidence,
        pilot_evidence_sha256=cast(str, raw["pilot_evidence_sha256"]),
        decision="activate-confirmatory-reserve",
        actor=cast(str, raw["actor"]),
        note=cast(str, raw["note"]),
        at=cast(str, raw["at"]),
    )


def seed_roll_plan_sha256(path: Path) -> str:
    """Convenience accessor that verifies before returning the canonical hash."""
    return load_seed_roll_artifact(path).payload_sha256
