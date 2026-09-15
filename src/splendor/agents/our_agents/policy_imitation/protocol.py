"""Reproducibility and provenance helpers for policy-imitation runs."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import numpy as np
import torch

RNG_PROTOCOL_VERSION = "splendor-rng-v1"
RNG_PROTOCOL_PREFIX = b"splendor-rng-v1\0"
_SEED63_MASK = (1 << 63) - 1
_U53_DENOMINATOR = 1 << 53
_UINT32_MAX = (1 << 32) - 1
_SHA256_HEX_LENGTH = 64
_ASCII_CONTROL_LIMIT = 32
_FORMAL_WORKER_COUNT_ENV = "SPLENDOR_FORMAL_WORKER_COUNT"
_FORMAL_GAME_STREAMS = frozenset(
    {
        "scenario_source",
        "pool_draw",
        "policy_action",
        "opponent_init",
        "opponent_action",
    }
)
_JobT = TypeVar("_JobT")
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class RngKey:
    """Semantic address of one formal random event.

    ``treatment_id`` is lineage metadata rather than seed material.  Whether
    treatments share a draw is declared by ``coupling_group``: paired
    treatments use the same group, while independent treatments must use
    different groups.  This keeps the CRN decision explicit instead of
    relying on a caller accidentally omitting the treatment name.
    """

    stream_name: str
    experiment_id: str | None = None
    protocol_version: str = RNG_PROTOCOL_VERSION
    randomization_root_sha256: str | None = None
    phase: str | None = None
    coupling_group: str | None = None
    replicate_id: int | None = None
    treatment_id: str | None = None
    scenario_id: str | None = None
    seat: int | None = None
    opponent_id: str | None = None
    update: int | None = None
    game_index: int | None = None
    epoch: int | None = None
    focal_step: int | None = None
    opponent_step: int | None = None

    def __post_init__(self) -> None:
        if type(self.stream_name) is not str or not self.stream_name:
            raise ValueError("RNG stream_name must not be empty")
        for field_name in (
            "experiment_id",
            "phase",
            "coupling_group",
            "treatment_id",
            "scenario_id",
            "opponent_id",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"RNG key {field_name} must be a non-empty string")
        if self.protocol_version != RNG_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported RNG protocol {self.protocol_version!r}; "
                f"expected {RNG_PROTOCOL_VERSION!r}"
            )
        if self.randomization_root_sha256 is not None and (
            len(self.randomization_root_sha256) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef"
                for character in self.randomization_root_sha256
            )
        ):
            raise ValueError("RNG randomization_root_sha256 must be lowercase SHA-256")
        if self.treatment_id is not None and not self.coupling_group:
            raise ValueError(
                "RNG keys with treatment lineage require an explicit coupling_group"
            )
        for field_name in (
            "replicate_id",
            "seat",
            "update",
            "game_index",
            "epoch",
            "focal_step",
            "opponent_step",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"RNG key {field_name} must be a non-negative integer")

    def as_dict(self) -> dict[str, str | int]:
        """Return the complete, sparse lineage representation."""
        return {
            name: value for name, value in asdict(self).items() if value is not None
        }

    def derivation_dict(self) -> dict[str, str | int]:
        """Return seed material, with CRN sharing controlled by the group."""
        payload = self.as_dict()
        payload.pop("treatment_id", None)
        return payload


@dataclass(frozen=True)
class SeedLineage:
    """Full audit record for one event-keyed random value."""

    key: RngKey | Mapping[str, object]
    canonical_key_json: str
    digest_hex: str
    seed63: int
    u53: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe record including treatment lineage metadata."""
        key = self.key.as_dict() if isinstance(self.key, RngKey) else dict(self.key)
        return {
            "key": key,
            "canonical_key_json": self.canonical_key_json,
            "digest_hex": self.digest_hex,
            "seed63": self.seed63,
            "u53": self.u53,
        }


@dataclass
class RngBundle:
    """Actor-local RNG implementations initialized from one lineage record."""

    lineage: SeedLineage
    python: random.Random
    numpy: np.random.Generator
    torch_cpu: torch.Generator

    @classmethod
    def from_lineage(cls, lineage: SeedLineage) -> RngBundle:
        """Create local generators without reading or mutating global RNG state."""
        torch_generator = torch.Generator(device="cpu")
        torch_generator.manual_seed(lineage.seed63)
        return cls(
            lineage=lineage,
            python=random.Random(lineage.seed63),
            numpy=np.random.default_rng(lineage.seed63),
            torch_cpu=torch_generator,
        )


def derive_seed(key: RngKey | Mapping[str, object]) -> SeedLineage:
    """Derive the protocol-v1 digest, 63-bit seed and open-interval ``u53``."""
    if isinstance(key, RngKey):
        payload: Mapping[str, object] = key.derivation_dict()
    else:
        mapping_key = dict(key)
        for name, value in mapping_key.items():
            if type(name) is not str or not name:
                raise ValueError("RNG mapping keys must be non-empty strings")
            if value is not None and type(value) not in {str, int}:
                raise ValueError(
                    "RNG mapping values must be strings, integers, or null"
                )
        protocol_version = mapping_key.get("protocol_version", RNG_PROTOCOL_VERSION)
        if protocol_version != RNG_PROTOCOL_VERSION:
            raise ValueError(f"unsupported RNG protocol {protocol_version!r}")
        treatment_id = mapping_key.get("treatment_id")
        if treatment_id is not None:
            if type(treatment_id) is not str or not treatment_id:
                raise ValueError("RNG mapping treatment_id must be non-empty")
            coupling_group = mapping_key.get("coupling_group")
            if type(coupling_group) is not str or not coupling_group:
                raise ValueError(
                    "RNG mappings with treatment lineage require an explicit "
                    "coupling_group"
                )
        # Mapping callers receive exactly the same CRN semantics as RngKey:
        # treatment stays in SeedLineage.key for audit, but only the explicit
        # coupling group decides whether two branches share a random value.
        derivation_payload = dict(mapping_key)
        derivation_payload.pop("treatment_id", None)
        payload = derivation_payload
    canonical = canonical_json_bytes(payload)
    digest = hashlib.sha256(RNG_PROTOCOL_PREFIX + canonical).digest()
    seed63 = int.from_bytes(digest[:8], "big") & _SEED63_MASK
    u53 = ((int.from_bytes(digest[8:16], "big") >> 11) + 0.5) / _U53_DENOMINATOR
    return SeedLineage(
        key=key,
        canonical_key_json=canonical.decode("utf-8"),
        digest_hex=digest.hex(),
        seed63=seed63,
        u53=u53,
    )


def inverse_cdf_index(weights: Sequence[float], u53: float) -> int:
    """Map one stable ``[0, 1)`` variate onto non-negative weights."""
    values = np.asarray(tuple(weights), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("inverse-CDF sampling needs a non-empty 1-D weight list")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("inverse-CDF weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("inverse-CDF weights must contain positive mass")
    if not np.isfinite(u53) or not 0.0 <= u53 < 1.0:
        raise ValueError("inverse-CDF variate must lie in [0, 1)")
    cdf = np.cumsum(values / total)
    cdf[-1] = 1.0
    return min(int(np.searchsorted(cdf, u53, side="right")), len(cdf) - 1)


@dataclass(frozen=True)
class FormalGameRng:
    """Base namespace used to address every random event in one formal game."""

    experiment_id: str
    phase: str
    coupling_group: str
    replicate_id: int
    treatment_id: str
    scenario_id: str
    seat: int
    update: int
    game_index: int
    randomization_root_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "phase",
            "coupling_group",
            "treatment_id",
            "scenario_id",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"formal RNG {field_name} must not be empty")
        for field_name in ("replicate_id", "seat", "update", "game_index"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"formal RNG {field_name} must be non-negative")
        if self.randomization_root_sha256 is not None and (
            len(self.randomization_root_sha256) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef"
                for character in self.randomization_root_sha256
            )
        ):
            raise ValueError(
                "formal RNG randomization_root_sha256 must be lowercase SHA-256"
            )

    def key(
        self,
        stream_name: str,
        *,
        opponent_id: str | None = None,
        epoch: int | None = None,
        focal_step: int | None = None,
        opponent_step: int | None = None,
    ) -> RngKey:
        """Build a treatment-labelled key in this game's coupling namespace."""
        if self.randomization_root_sha256 is not None:
            if stream_name not in _FORMAL_GAME_STREAMS:
                raise ValueError(f"unsupported formal game RNG stream {stream_name!r}")
            if epoch is not None:
                raise ValueError("formal game RNG streams cannot declare an epoch")
            if stream_name in {"scenario_source", "pool_draw"} and any(
                value is not None for value in (opponent_id, focal_step, opponent_step)
            ):
                raise ValueError(
                    f"formal {stream_name} stream has irrelevant event axes"
                )
            if stream_name == "policy_action" and (
                focal_step is None
                or opponent_id is not None
                or opponent_step is not None
            ):
                raise ValueError("formal policy_action requires only focal_step")
            if stream_name in {"opponent_init", "opponent_action"} and (
                opponent_id is None or opponent_step is None or focal_step is not None
            ):
                raise ValueError(
                    f"formal {stream_name} requires opponent_id and opponent_step"
                )
        if stream_name == "scenario_source":
            # A scenario is a seat-independent immutable environment object.
            # Reusing its identity must never silently redeal because a runner
            # changed seat, update, game index, worker, or model replicate.
            return RngKey(
                experiment_id=self.experiment_id,
                phase=self.phase,
                randomization_root_sha256=self.randomization_root_sha256,
                coupling_group=f"scenario:{self.scenario_id}",
                stream_name=stream_name,
                treatment_id=self.treatment_id,
                scenario_id=self.scenario_id,
            )
        return RngKey(
            experiment_id=self.experiment_id,
            phase=self.phase,
            randomization_root_sha256=self.randomization_root_sha256,
            coupling_group=self.coupling_group,
            stream_name=stream_name,
            replicate_id=self.replicate_id,
            treatment_id=self.treatment_id,
            scenario_id=self.scenario_id,
            seat=self.seat,
            opponent_id=opponent_id,
            update=self.update,
            game_index=self.game_index,
            epoch=epoch,
            focal_step=focal_step,
            opponent_step=opponent_step,
        )

    def lineage(
        self,
        stream_name: str,
        *,
        opponent_id: str | None = None,
        epoch: int | None = None,
        focal_step: int | None = None,
        opponent_step: int | None = None,
    ) -> SeedLineage:
        """Derive a lineage record for one event within the game."""
        return derive_seed(
            self.key(
                stream_name,
                opponent_id=opponent_id,
                epoch=epoch,
                focal_step=focal_step,
                opponent_step=opponent_step,
            )
        )


@dataclass(frozen=True)
class PairedTrainingRow:
    """One immutable game row shared across paired treatment branches."""

    experiment_id: str
    phase: str
    replicate_id: int
    treatment_id: str
    update: int
    game_index: int
    scenario_id: str
    seat: int
    coupling_group: str
    scenario_source: SeedLineage
    pool_draw: SeedLineage
    randomization_root_sha256: str | None = None

    def game_rng(self) -> FormalGameRng:
        """Recover the event namespace used for action-time derivations."""
        return FormalGameRng(
            experiment_id=self.experiment_id,
            phase=self.phase,
            coupling_group=self.coupling_group,
            replicate_id=self.replicate_id,
            treatment_id=self.treatment_id,
            scenario_id=self.scenario_id,
            seat=self.seat,
            update=self.update,
            game_index=self.game_index,
            randomization_root_sha256=self.randomization_root_sha256,
        )

    def semantic_dict(self) -> dict[str, object]:
        """Return treatment-neutral CRN semantics for equality checks."""
        payload: dict[str, object] = {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "replicate_id": self.replicate_id,
            "update": self.update,
            "game_index": self.game_index,
            "scenario_id": self.scenario_id,
            "seat": self.seat,
            "coupling_group": self.coupling_group,
            "scenario_source_digest": self.scenario_source.digest_hex,
            "pool_draw_digest": self.pool_draw.digest_hex,
        }
        if self.randomization_root_sha256 is not None:
            payload["randomization_root_sha256"] = self.randomization_root_sha256
        return payload

    def as_dict(self) -> dict[str, object]:
        """Return the complete treatment-labelled schedule row."""
        payload: dict[str, object] = {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "replicate_id": self.replicate_id,
            "treatment_id": self.treatment_id,
            "update": self.update,
            "game_index": self.game_index,
            "scenario_id": self.scenario_id,
            "seat": self.seat,
            "coupling_group": self.coupling_group,
            "scenario_source": self.scenario_source.as_dict(),
            "pool_draw": self.pool_draw.as_dict(),
        }
        if self.randomization_root_sha256 is not None:
            payload["randomization_root_sha256"] = self.randomization_root_sha256
        return payload


def make_paired_training_schedule(  # noqa: C901,PLR0913 - protocol axes explicit
    *,
    experiment_id: str,
    phase: str,
    treatment_ids: Sequence[str],
    scenarios_by_replicate: Mapping[int, Sequence[str]],
    seats_by_replicate: Mapping[int, Sequence[int]],
    updates: int,
    games_per_update: int,
    randomization_root_sha256: str | None = None,
) -> tuple[PairedTrainingRow, ...]:
    """Create an order/worker-invariant paired training schedule.

    Scenario and seat arrays are explicit inputs rather than mutable draws.
    Each replicate gets its own coupling group; all listed treatments in that
    replicate consequently share the same event random variables.
    """
    if updates < 1 or games_per_update < 1:
        raise ValueError("schedule updates and games_per_update must be positive")
    treatments = tuple(treatment_ids)
    if not treatments or len(set(treatments)) != len(treatments):
        raise ValueError("treatment IDs must be non-empty and unique")
    if set(scenarios_by_replicate) != set(seats_by_replicate):
        raise ValueError("scenario and seat schedules must cover the same replicates")
    games = updates * games_per_update
    rows: list[PairedTrainingRow] = []
    for replicate_id in sorted(scenarios_by_replicate):
        scenarios = tuple(scenarios_by_replicate[replicate_id])
        seats = tuple(seats_by_replicate[replicate_id])
        if len(scenarios) != games or len(seats) != games:
            raise ValueError(
                f"replicate {replicate_id} needs exactly {games} scenarios and seats"
            )
        if len(set(scenarios)) != len(scenarios):
            raise ValueError(f"replicate {replicate_id} repeats a training scenario")
        if any(seat not in (0, 1) for seat in seats):
            raise ValueError("formal 2p training seats must be 0 or 1")
        if randomization_root_sha256 is not None and (
            len(seats) % 2 or seats.count(0) != seats.count(1)
        ):
            raise ValueError(
                "seed-rolled formal training requires exact per-replicate seat balance"
            )
        coupling_group = f"replicate-{replicate_id}"
        for treatment_id in treatments:
            for ordinal, (scenario_id, seat) in enumerate(
                zip(scenarios, seats, strict=True)
            ):
                update = ordinal // games_per_update + 1
                game_index = ordinal % games_per_update
                context = FormalGameRng(
                    experiment_id=experiment_id,
                    phase=phase,
                    coupling_group=coupling_group,
                    replicate_id=replicate_id,
                    treatment_id=treatment_id,
                    scenario_id=scenario_id,
                    seat=seat,
                    update=update,
                    game_index=game_index,
                    randomization_root_sha256=randomization_root_sha256,
                )
                rows.append(
                    PairedTrainingRow(
                        experiment_id=experiment_id,
                        phase=phase,
                        replicate_id=replicate_id,
                        treatment_id=treatment_id,
                        update=update,
                        game_index=game_index,
                        scenario_id=scenario_id,
                        seat=seat,
                        coupling_group=coupling_group,
                        scenario_source=context.lineage("scenario_source"),
                        pool_draw=context.lineage("pool_draw"),
                        randomization_root_sha256=randomization_root_sha256,
                    )
                )
    result = tuple(rows)
    validate_paired_training_schedule(result, expected_treatments=treatments)
    return result


def validate_paired_training_schedule(  # noqa: C901,PLR0912 - every schedule axis is audited
    rows: Sequence[PairedTrainingRow],
    *,
    expected_treatments: Sequence[str] | None = None,
) -> None:
    """Reject incomplete, duplicated, or non-paired formal schedule rows."""
    if not rows:
        raise ValueError("paired training schedule must not be empty")
    identities = [
        (row.replicate_id, row.treatment_id, row.update, row.game_index) for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("paired training schedule contains a duplicate row key")
    experiment_phases = {(row.experiment_id, row.phase) for row in rows}
    if len(experiment_phases) != 1:
        raise ValueError("paired schedule mixes experiment or phase identities")
    randomization_roots = {row.randomization_root_sha256 for row in rows}
    if len(randomization_roots) != 1:
        raise ValueError("paired schedule mixes randomization roots")
    replicate_ids = sorted({row.replicate_id for row in rows})
    expected = set(expected_treatments) if expected_treatments is not None else None
    if expected is not None and (
        not expected or len(expected) != len(tuple(expected_treatments or ()))
    ):
        raise ValueError("expected treatment IDs must be non-empty and unique")
    seen_scenarios: dict[str, int] = {}
    for replicate_id in replicate_ids:
        replicate_rows = [row for row in rows if row.replicate_id == replicate_id]
        coupling_groups = {row.coupling_group for row in replicate_rows}
        expected_group = f"replicate-{replicate_id}"
        if coupling_groups != {expected_group}:
            raise ValueError(
                f"replicate {replicate_id} must use coupling group {expected_group!r}"
            )
        treatments = {row.treatment_id for row in replicate_rows}
        if expected is not None and treatments != expected:
            raise ValueError(
                f"replicate {replicate_id} treatment coverage does not match expected"
            )
        coordinate_sets = {
            treatment: {
                (row.update, row.game_index): row.semantic_dict()
                for row in replicate_rows
                if row.treatment_id == treatment
            }
            for treatment in treatments
        }
        reference = next(iter(coordinate_sets.values()))
        if any(coordinates != reference for coordinates in coordinate_sets.values()):
            raise ValueError(
                f"replicate {replicate_id} treatments do not share CRN semantics"
            )
        scenario_ids = [str(semantic["scenario_id"]) for semantic in reference.values()]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError(f"replicate {replicate_id} repeats a training scenario")
        roots = {
            semantic.get("randomization_root_sha256") for semantic in reference.values()
        }
        if roots != {None}:
            seats = [cast(int, semantic["seat"]) for semantic in reference.values()]
            if len(seats) % 2 or seats.count(0) != seats.count(1):
                raise ValueError(
                    "seed-rolled formal training requires exact per-replicate seat balance"
                )
        for semantic in reference.values():
            scenario_id = str(semantic["scenario_id"])
            other_replicate = seen_scenarios.setdefault(scenario_id, replicate_id)
            if other_replicate != replicate_id:
                raise ValueError(
                    f"training scenario {scenario_id!r} crosses replicates"
                )
        for row in replicate_rows:
            context = row.game_rng()
            if row.scenario_source != context.lineage("scenario_source"):
                raise ValueError("schedule scenario-source lineage is inconsistent")
            if row.pool_draw != context.lineage("pool_draw"):
                raise ValueError("schedule pool-draw lineage is inconsistent")


def _validate_optional_sha256(value: object, field_name: str) -> None:
    if value is not None and (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"formal {field_name} SHA-256 is invalid")


@dataclass(frozen=True)
class FormalTreatmentContract:
    """Immutable trainer inputs that may legitimately differ by treatment."""

    treatment_id: str
    initial_checkpoint_sha256: str
    trainer_config_sha256: str
    opponent_pool_sha256: str

    def __post_init__(self) -> None:
        if type(self.treatment_id) is not str or not self.treatment_id:
            raise ValueError("formal treatment contract needs a treatment ID")
        for field_name, value in (
            ("initial checkpoint", self.initial_checkpoint_sha256),
            ("trainer config", self.trainer_config_sha256),
            ("opponent pool", self.opponent_pool_sha256),
        ):
            if value is None:
                raise ValueError(f"formal {field_name} SHA-256 is required")
            _validate_optional_sha256(value, field_name)

    def as_dict(self) -> dict[str, str]:
        return {
            "treatment_id": self.treatment_id,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "trainer_config_sha256": self.trainer_config_sha256,
            "opponent_pool_sha256": self.opponent_pool_sha256,
        }


@dataclass(frozen=True)
class FormalJobOutput:
    """One predeclared output location for a treatment/replicate job."""

    replicate_id: int
    treatment_id: str
    output_dir: str

    def __post_init__(self) -> None:
        if type(self.replicate_id) is not int or self.replicate_id < 0:
            raise ValueError("formal job output replicate_id is invalid")
        if type(self.treatment_id) is not str or not self.treatment_id:
            raise ValueError("formal job output treatment_id is invalid")
        if (
            type(self.output_dir) is not str
            or not self.output_dir
            or not Path(self.output_dir).is_absolute()
            or os.path.normpath(self.output_dir) != self.output_dir
            or Path(self.output_dir).resolve(strict=False) != Path(self.output_dir)
        ):
            raise ValueError(
                "formal job output must be a normalized absolute directory"
            )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "replicate_id": self.replicate_id,
            "treatment_id": self.treatment_id,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class FormalTrainingSpec:
    """Explicit opt-in contract binding one trainer to its paired schedule."""

    experiment_id: str
    phase: str
    replicate_id: int
    treatment_id: str
    expected_treatments: tuple[str, ...]
    schedule: tuple[PairedTrainingRow, ...]
    scenario_bank_sha256: str | None = None
    worker_count: int = 1
    seed_roll_payload_sha256: str | None = None
    randomization_root_sha256: str | None = None
    treatment_contracts: tuple[FormalTreatmentContract, ...] = ()
    job_outputs: tuple[FormalJobOutput, ...] = ()
    replicate_stage: Literal["pilot", "confirmatory-reserve"] | None = None
    activation_artifact_path: str | None = None
    activation_artifact_sha256: str | None = None

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - fail-closed contract
        for field_name, value in (
            ("experiment_id", self.experiment_id),
            ("phase", self.phase),
            ("treatment_id", self.treatment_id),
        ):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or any(ord(character) < _ASCII_CONTROL_LIMIT for character in value)
            ):
                raise ValueError(f"formal {field_name} must be a canonical identifier")
        if any(
            type(value) is not str
            or not value
            or value != value.strip()
            or any(ord(character) < _ASCII_CONTROL_LIMIT for character in value)
            for value in self.expected_treatments
        ):
            raise ValueError(
                "formal expected_treatments must use canonical identifiers"
            )
        if self.worker_count < 1:
            raise ValueError("formal worker_count must be positive")
        _validate_optional_sha256(self.scenario_bank_sha256, "scenario-bank")
        roll_values = (
            self.seed_roll_payload_sha256,
            self.randomization_root_sha256,
        )
        if any(value is None for value in roll_values) and any(
            value is not None for value in roll_values
        ):
            raise ValueError(
                "formal seed-roll payload and randomization root must be paired"
            )
        for digest_name, digest_value in (
            ("seed-roll payload", self.seed_roll_payload_sha256),
            ("randomization root", self.randomization_root_sha256),
        ):
            _validate_optional_sha256(digest_value, digest_name)
        if (
            self.seed_roll_payload_sha256 is not None
            and self.scenario_bank_sha256 is None
        ):
            raise ValueError("paired-training-v2 requires a scenario-bank SHA-256")
        validate_paired_training_schedule(
            self.schedule,
            expected_treatments=self.expected_treatments,
        )
        identities = {(row.experiment_id, row.phase) for row in self.schedule}
        if identities != {(self.experiment_id, self.phase)}:
            raise ValueError(
                "formal training spec does not match its schedule identity"
            )
        if self.treatment_id not in self.expected_treatments:
            raise ValueError("formal treatment is absent from expected_treatments")
        if self.replicate_id not in {row.replicate_id for row in self.schedule}:
            raise ValueError("formal replicate is absent from schedule")
        schedule_roots = {row.randomization_root_sha256 for row in self.schedule}
        if schedule_roots != {self.randomization_root_sha256}:
            raise ValueError(
                "formal training randomization root does not match its schedule"
            )
        if self.seed_roll_payload_sha256 is None:
            if (
                self.treatment_contracts
                or self.job_outputs
                or self.replicate_stage is not None
                or self.activation_artifact_path is not None
                or self.activation_artifact_sha256 is not None
            ):
                raise ValueError(
                    "paired-training-v1 cannot declare v2 treatment/job contracts"
                )
            return
        if self.replicate_stage not in {"pilot", "confirmatory-reserve"}:
            raise ValueError(
                "paired-training-v2 must declare pilot or confirmatory-reserve stage"
            )
        if self.replicate_stage == "pilot":
            if (
                self.activation_artifact_path is not None
                or self.activation_artifact_sha256 is not None
            ):
                raise ValueError(
                    "pilot training cannot declare a post-pilot activation artifact"
                )
        else:
            if (
                self.activation_artifact_path is None
                or self.activation_artifact_sha256 is None
            ):
                raise ValueError(
                    "confirmatory-reserve training requires a bound activation "
                    "artifact path and SHA-256"
                )
            if (
                not Path(self.activation_artifact_path).is_absolute()
                or os.path.normpath(self.activation_artifact_path)
                != self.activation_artifact_path
                or Path(self.activation_artifact_path).resolve(strict=False)
                != Path(self.activation_artifact_path)
            ):
                raise ValueError(
                    "confirmatory activation artifact must use a normalized "
                    "absolute non-symlink path"
                )
            _validate_optional_sha256(
                self.activation_artifact_sha256,
                "confirmatory activation artifact",
            )
        contract_ids = tuple(
            contract.treatment_id for contract in self.treatment_contracts
        )
        expected_contract_ids = tuple(sorted(self.expected_treatments))
        if contract_ids != expected_contract_ids:
            raise ValueError(
                "paired-training-v2 treatment contracts must exactly cover the "
                "sorted treatment set"
            )
        output_coordinates = tuple(
            (output.replicate_id, output.treatment_id) for output in self.job_outputs
        )
        expected_outputs = tuple(
            (replicate_id, treatment_id)
            for replicate_id in sorted({row.replicate_id for row in self.schedule})
            for treatment_id in expected_contract_ids
        )
        if output_coordinates != expected_outputs:
            raise ValueError(
                "paired-training-v2 job outputs must exactly cover the canonical "
                "replicate x treatment matrix"
            )
        if len({output.output_dir for output in self.job_outputs}) != len(
            self.job_outputs
        ):
            raise ValueError("paired-training-v2 job output directories must be unique")

    def manifest_binding(self) -> dict[str, object]:
        """Return the treatment-neutral declaration a manifest must freeze."""
        binding: dict[str, object] = {
            "protocol": (
                "paired-training-v2"
                if self.seed_roll_payload_sha256 is not None
                else "paired-training-v1"
            ),
            "paired_schedule_sha256": paired_schedule_hash(
                self.schedule,
                treatment_id=self.treatment_id,
            ),
            "expected_treatments": sorted(self.expected_treatments),
            "replicate_ids": sorted({row.replicate_id for row in self.schedule}),
            "worker_count": self.worker_count,
            "worker_scope": "independent-training-jobs",
        }
        if self.scenario_bank_sha256 is not None:
            binding["scenario_bank_sha256"] = self.scenario_bank_sha256
        if self.seed_roll_payload_sha256 is not None:
            assert self.randomization_root_sha256 is not None
            binding["seed_roll_payload_sha256"] = self.seed_roll_payload_sha256
            binding["randomization_root_sha256"] = self.randomization_root_sha256
            binding["treatment_contracts"] = [
                contract.as_dict() for contract in self.treatment_contracts
            ]
            binding["job_outputs"] = [output.as_dict() for output in self.job_outputs]
            binding["replicate_stage"] = self.replicate_stage
            binding["activation_artifact_path"] = self.activation_artifact_path
            binding["activation_artifact_sha256"] = self.activation_artifact_sha256
        return binding

    def treatment_contract(self, treatment_id: str) -> FormalTreatmentContract:
        """Return one exact v2 treatment contract."""
        for contract in self.treatment_contracts:
            if contract.treatment_id == treatment_id:
                return contract
        raise ValueError(f"no formal treatment contract for {treatment_id!r}")

    def output_for(self, replicate_id: int, treatment_id: str) -> FormalJobOutput:
        """Return one exact v2 job output declaration."""
        for output in self.job_outputs:
            if (
                output.replicate_id == replicate_id
                and output.treatment_id == treatment_id
            ):
                return output
        raise ValueError(
            f"no formal output for replicate={replicate_id}, treatment={treatment_id!r}"
        )

    def require_worker_runtime(self) -> None:
        """Reject an N-worker declaration executed directly in the parent.

        ``worker_count`` controls independent treatment/replicate jobs, not
        rollouts inside one optimizer.  A multi-worker formal spec therefore
        has to enter :func:`run_formal_spawn_jobs`; this avoids silently
        recording parallel execution while running the trainer in-process.
        """
        require_formal_spawn_context()
        if (
            self.seed_roll_payload_sha256 is not None
            and mp.current_process().name == "MainProcess"
        ):
            raise RuntimeError(
                "paired-training-v2 must run through the one-shot spawn job matrix"
            )
        if self.worker_count == 1:
            return
        if mp.current_process().name == "MainProcess":
            raise RuntimeError(
                "multi-worker formal training must run inside a spawned worker"
            )
        if mp.get_start_method(allow_none=True) != "spawn":
            raise RuntimeError("formal training worker was not started with spawn")
        declared_count = os.environ.get(_FORMAL_WORKER_COUNT_ENV)
        if declared_count != str(self.worker_count):
            raise RuntimeError(
                "formal training worker_count does not match its spawn executor"
            )

    def selected_rows(
        self,
        *,
        updates: int,
        games_per_update: int,
    ) -> tuple[PairedTrainingRow, ...]:
        """Return this run's rows and verify an exact rectangular budget."""
        selected = tuple(
            sorted(
                (
                    row
                    for row in self.schedule
                    if row.replicate_id == self.replicate_id
                    and row.treatment_id == self.treatment_id
                ),
                key=lambda row: (row.update, row.game_index),
            )
        )
        expected_coordinates = {
            (update, game_index)
            for update in range(1, updates + 1)
            for game_index in range(games_per_update)
        }
        coordinates = {(row.update, row.game_index) for row in selected}
        if coordinates != expected_coordinates or len(selected) != len(
            expected_coordinates
        ):
            raise ValueError("formal schedule does not match the PPO update budget")
        return selected

    def model_init_lineage(self) -> SeedLineage:
        """Derive paired model initialization for a replicate block."""
        return derive_seed(
            RngKey(
                experiment_id=self.experiment_id,
                phase=self.phase,
                randomization_root_sha256=self.randomization_root_sha256,
                coupling_group=f"replicate-{self.replicate_id}",
                stream_name="model_init",
                replicate_id=self.replicate_id,
                treatment_id=self.treatment_id,
            )
        )

    def minibatch_key(self, update: int) -> RngKey:
        """Return an update-level key; PPO derives one child per epoch."""
        return RngKey(
            experiment_id=self.experiment_id,
            phase=self.phase,
            randomization_root_sha256=self.randomization_root_sha256,
            coupling_group=f"replicate-{self.replicate_id}",
            stream_name="minibatch",
            replicate_id=self.replicate_id,
            treatment_id=self.treatment_id,
            update=update,
        )

    def worker_lineage(self, worker_index: int) -> SeedLineage:
        """Derive a non-semantic process-start stream for one spawn worker."""
        if worker_index not in range(self.worker_count):
            raise ValueError("worker index lies outside formal worker_count")
        return derive_seed(
            RngKey(
                experiment_id=self.experiment_id,
                phase=self.phase,
                randomization_root_sha256=self.randomization_root_sha256,
                coupling_group=f"replicate-{self.replicate_id}",
                stream_name="worker",
                replicate_id=self.replicate_id,
                treatment_id=self.treatment_id,
                focal_step=worker_index,
            )
        )

    def worker_partitions(
        self,
        *,
        updates: int,
        games_per_update: int,
    ) -> tuple[tuple[PairedTrainingRow, ...], ...]:
        """Partition rows only to audit worker-count semantic invariance.

        Optimizer rollouts remain sequential within one training job.  The
        actual worker pool runs independent treatment/replicate jobs via
        :func:`run_formal_spawn_jobs`.
        """
        selected = self.selected_rows(
            updates=updates,
            games_per_update=games_per_update,
        )
        return tuple(
            tuple(
                row
                for index, row in enumerate(selected)
                if index % self.worker_count == worker
            )
            for worker in range(self.worker_count)
        )

    def as_dict(self) -> dict[str, object]:
        """Return the run binding without duplicating the full schedule."""
        return {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "replicate_id": self.replicate_id,
            "treatment_id": self.treatment_id,
            "expected_treatments": list(self.expected_treatments),
            "worker_count": self.worker_count,
            "paired_schedule_sha256": paired_schedule_hash(
                self.schedule,
                treatment_id=self.treatment_id,
            ),
            "scenario_bank_sha256": self.scenario_bank_sha256,
            "seed_roll_payload_sha256": self.seed_roll_payload_sha256,
            "randomization_root_sha256": self.randomization_root_sha256,
            "treatment_contracts": [
                contract.as_dict() for contract in self.treatment_contracts
            ],
            "job_outputs": [output.as_dict() for output in self.job_outputs],
            "replicate_stage": self.replicate_stage,
            "activation_artifact_path": self.activation_artifact_path,
            "activation_artifact_sha256": self.activation_artifact_sha256,
            "manifest_binding": self.manifest_binding(),
            "model_init_lineage": self.model_init_lineage().as_dict(),
            "worker_lineages": [
                self.worker_lineage(index).as_dict()
                for index in range(self.worker_count)
            ],
        }


def paired_schedule_hash(
    rows: Sequence[PairedTrainingRow], *, treatment_id: str
) -> str:
    """Hash one treatment's treatment-neutral schedule in canonical order."""
    validate_paired_training_schedule(rows)
    selected = [row for row in rows if row.treatment_id == treatment_id]
    if not selected:
        raise ValueError(f"schedule contains no treatment {treatment_id!r}")
    payload = [
        row.semantic_dict()
        for row in sorted(
            selected,
            key=lambda row: (row.replicate_id, row.update, row.game_index),
        )
    ]
    return sha256_canonical_json(payload)


def _started_with_zero_hash_seed() -> bool:
    """Return whether hash randomization was disabled at interpreter startup."""
    return os.environ.get("PYTHONHASHSEED") == "0" and sys.flags.hash_randomization == 0


def require_formal_spawn_context() -> mp.context.BaseContext:
    """Require interpreter-level hash determinism and return spawn context."""
    if not _started_with_zero_hash_seed():
        raise RuntimeError(
            "formal workers require PYTHONHASHSEED=0 before Python starts"
        )
    context = mp.get_context("spawn")
    if context.get_start_method() != "spawn":  # pragma: no cover - stdlib guard
        raise RuntimeError("formal workers require multiprocessing spawn")
    return context


def run_formal_spawn_jobs(
    jobs: Sequence[_JobT],
    *,
    worker: Callable[[_JobT], _ResultT],
    worker_count: int,
) -> tuple[_ResultT, ...]:
    """Execute independent formal jobs with spawn and stable input ordering.

    Results are returned in input order, irrespective of future completion
    order.  Semantic schedule rows are carried by each job and are never
    derived from worker index or completion order.
    """
    if worker_count < 1:
        raise ValueError("formal worker_count must be positive")
    if not jobs:
        return ()
    context = require_formal_spawn_context()
    missing = object()
    results: list[object] = [missing] * len(jobs)
    previous_count = os.environ.get(_FORMAL_WORKER_COUNT_ENV)
    os.environ[_FORMAL_WORKER_COUNT_ENV] = str(worker_count)
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as pool:
            future_indexes: dict[Future[_ResultT], int] = {
                pool.submit(worker, job): index for index, job in enumerate(jobs)
            }
            for future in as_completed(future_indexes):
                results[future_indexes[future]] = future.result()
    finally:
        if previous_count is None:
            os.environ.pop(_FORMAL_WORKER_COUNT_ENV, None)
        else:
            os.environ[_FORMAL_WORKER_COUNT_ENV] = previous_count
    if any(
        result is missing for result in results
    ):  # pragma: no cover - executor guard
        raise RuntimeError("formal spawn executor returned an incomplete result set")
    return tuple(cast(_ResultT, result) for result in results)


def configure_formal_torch_determinism(
    requested_device: str,
    *,
    warn_only: bool = False,
) -> RuntimeSnapshot:
    """Enable and capture the deterministic settings for a formal process.

    Unlike the legacy device resolver, this function never falls back from a
    requested accelerator.  ``PYTHONHASHSEED`` must already have affected the
    running interpreter; CUDA's cuBLAS workspace setting must be installed
    before CUDA initialization.
    """
    require_formal_spawn_context()
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("formal Task-1 execution supports only cpu or cuda")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "formal CUDA execution requested but CUDA is unavailable"
            )
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {":4096:8", ":16:8"}:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization"
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = RuntimeSnapshot.collect(requested_device)
    if snapshot.resolved_device != requested_device:
        raise RuntimeError(
            f"formal device resolution changed {requested_device!r} to "
            f"{snapshot.resolved_device!r}"
        )
    return snapshot


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the repository's engine and neural agents."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    # Preserve every legacy uint32 sequence exactly.  Event-derived 63-bit
    # seeds use both words to initialize MT19937 instead of discarding the
    # upper 31 bits.
    if seed <= _UINT32_MAX:
        np.random.seed(seed)
    else:
        np.random.seed(
            np.asarray(
                [seed & _UINT32_MAX, (seed >> 32) & _UINT32_MAX],
                dtype=np.uint32,
            )
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_python_seed(seed: int) -> Iterator[None]:
    """Temporarily seed only Python's RNG, used by legacy engine dealing."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    python_state = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(python_state)


@contextmanager
def isolated_seed(seed: int) -> Iterator[None]:
    """Run a game or audit probe without consuming the caller's RNG stream."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        seed_everything(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Machine and library information that can affect an experiment."""

    python_version: str
    python_executable: str
    platform: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: tuple[str, ...]
    requested_device: str
    resolved_device: str
    cpu_count: int | None
    torch_num_threads: int
    torch_num_interop_threads: int
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cudnn_allow_tf32: bool
    matmul_allow_tf32: bool
    cublas_workspace_config: str | None
    python_hash_seed: str | None
    python_hash_randomization: int
    cuda_driver_version: str | None

    @classmethod
    def collect(cls, requested_device: str = "cpu") -> RuntimeSnapshot:
        """Collect a serializable snapshot and resolve unavailable accelerators."""
        if requested_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device {requested_device!r}")
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_devices = tuple(
            torch.cuda.get_device_name(index) for index in range(cuda_device_count)
        )
        if requested_device == "cuda" and cuda_available:
            resolved_device = "cuda"
        elif requested_device == "mps" and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
        deterministic_warn_only = False
        if torch.are_deterministic_algorithms_enabled():
            deterministic_warn_only = bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            )
        return cls(
            python_version=sys.version.split()[0],
            python_executable=sys.executable,
            platform=platform.platform(),
            numpy_version=np.__version__,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_devices=cuda_devices,
            requested_device=requested_device,
            resolved_device=resolved_device,
            cpu_count=os.cpu_count(),
            torch_num_threads=torch.get_num_threads(),
            torch_num_interop_threads=torch.get_num_interop_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=deterministic_warn_only,
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            python_hash_seed=os.environ.get("PYTHONHASHSEED"),
            python_hash_randomization=int(sys.flags.hash_randomization),
            cuda_driver_version=_cuda_driver_version(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fields."""
        return asdict(self)


def _git_output(repo: Path, *arguments: str) -> str | None:
    """Read one Git value without invoking a shell."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _command_output(*arguments: str) -> str | None:
    """Read a short, optional command value without invoking a shell."""
    try:
        result = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _cuda_driver_version() -> str | None:
    value = _command_output(
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )
    return value.splitlines()[0].strip() if value else None


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically for provenance digests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_canonical_json(value: object) -> str:
    """Hash a JSON-compatible value using the repository canonical form."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return b""
    return result.stdout if result.returncode == 0 else b""


def capture_dirty_patch(repo: Path) -> bytes:
    """Return a replayable tracked/untracked worktree snapshot."""
    tracked = _git_bytes(repo, "diff", "HEAD", "--binary")
    untracked_raw = _git_bytes(repo, "ls-files", "--others", "--exclude-standard")
    chunks = [b"# splendor uncommitted patch snapshot v1\n", tracked]
    for raw_relative in sorted(untracked_raw.splitlines()):
        relative = raw_relative.decode("utf-8")
        path = repo / relative
        if not path.is_file():
            continue
        chunks.extend(
            (
                b"\n# UNTRACKED_FILE_BEGIN " + raw_relative + b"\n",
                base64.b64encode(path.read_bytes()),
                b"\n# UNTRACKED_FILE_END\n",
            )
        )
    return b"".join(chunks)


def capture_code_revision(repo: Path | None = None) -> dict[str, Any]:
    """Capture commit, dirty paths, and the replayable patch digest."""
    root = (repo or Path.cwd()).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked = _git_output(root, "diff", "--name-only") or ""
    untracked = _git_output(root, "ls-files", "--others", "--exclude-standard") or ""
    dirty_files = sorted(
        {line for line in (*tracked.splitlines(), *untracked.splitlines()) if line}
    )
    patch = capture_dirty_patch(root)
    return {
        "repository": str(root),
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "patch_format": "tracked git diff plus base64 untracked files",
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }


def capture_dependency_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Hash dependency declarations that can change a formal run."""
    root = (repo or Path.cwd()).resolve()
    candidates = [root / "pyproject.toml", root / "uv.lock", root / "requirements.txt"]
    requirements = root / "requirements"
    if requirements.is_dir():
        candidates.extend(sorted(requirements.glob("*.txt")))
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }
    return {
        "files": files,
        "aggregate_sha256": sha256_canonical_json(files),
    }


def require_reproducible_code(code: Mapping[str, Any]) -> None:
    """Reject provenance that is neither clean nor backed by a patch digest."""
    if code.get("commit") is None:
        raise RuntimeError("formal execution requires a Git commit")
    if code.get("dirty") and not code.get("patch_sha256"):
        raise RuntimeError("dirty formal execution requires a replayable patch SHA-256")
