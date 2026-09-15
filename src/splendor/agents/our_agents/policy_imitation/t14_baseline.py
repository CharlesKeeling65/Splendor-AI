"""Auditable T1.4 frozen-baseline replay and variance decomposition.

This module deliberately sits above the general paired-evaluation runner.  It
does not introduce another game loop: it freezes the two admissible banks,
requires the same checkpoint/opponent axes in both batches, and turns their
episode ledgers into one content-addressed baseline report.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from .paired_evaluation import (
    EvaluationPolicy,
    PairedEvaluationError,
    PairedEvaluationSpec,
    audit_episode_batch,
    load_episode_records,
    require_paired_evaluation_manifest,
    run_paired_evaluation,
)
from .protocol import canonical_json_bytes, sha256_canonical_json, sha256_file
from .scenario_bank import ScenarioBank

T14_BASELINE_PROTOCOL: Final = "task1-t14-frozen-baseline-v1"
T14_BASELINE_REPORT_SCHEMA: Final = "task1-t14-baseline-report/1"
T14_VARIANCE_PROTOCOL: Final = "balanced-functional-anova-population-v1"
T14_BASELINE_STATISTICAL_PROTOCOL: Final = (
    "task1-t14-baseline-descriptive-statistics-v1"
)
T14_PPO_BEST_SHA256: Final = (
    "e225464c17a783bd91b51251336f917e9f867af4f475d8414372885fbb758102"
)
T14_PPO_BEST_SOURCE_SHA256: Final = (
    "5435d14160964dfcbe3791b7e9edbb43e92b9c9d69c7c41f10b74f0c4ece586b"
)
T14_PPO_BEST_CONFIG_SHA256: Final = (
    "45c75b4d6525127a24ce343b208b3b8ec22828e9eb7869bc473bfec0cd18a467"
)
T14_PPO_BEST_STATE_SHA256: Final = (
    "eae53bc714ef1e7b2f3287652185122cf0f123d7b97707a8937405b114d3db11"
)
T14_BASELINE_OPPONENTS: Final = (
    {
        "policy_id": "ga",
        "policy_sha256": "8f69edab941bea746e7a363afada68b070b137465bb46588fba91c76f4a04850",
        "source_sha256": "04eee10842a859d2d1d3ddaa54e7f5313e28560a3d2dd97bbae545cc2400d79c",
        "config_sha256": "82ba600b932de1f084bfa5fc88a38aca117c15c43e62d86455f3f029d8d88e2d",
        "checkpoint_sha256": None,
        "checkpoint_state_sha256": None,
        "snapshot": None,
        "role": "opponent",
        "treatment_id": None,
        "replicate_id": None,
        "model_seed": None,
        "feature_version": "v1",
    },
    {
        "policy_id": "heuristic",
        "policy_sha256": "c2fca8de365414b3d552366efaa83e1d2f707bbfe1edc7587de064287c95d35b",
        "source_sha256": "d3b03e35992a43eb7bd2a4dd07e4ffe3a61726fdd7fd5dacd72c213f86b0c198",
        "config_sha256": "db21aae5a269bdc446e15e4cec79e03d03d0f105e29aec070e38f9de2a2cf483",
        "checkpoint_sha256": None,
        "checkpoint_state_sha256": None,
        "snapshot": None,
        "role": "opponent",
        "treatment_id": None,
        "replicate_id": None,
        "model_seed": None,
        "feature_version": "v1",
    },
    {
        "policy_id": "minimax",
        "policy_sha256": "ecf6e5dba9c94eb9e8ffbba8b113d3172e08a5d77eee1ca8cf3abf05a8f14222",
        "source_sha256": "4049bc402bcaf7a01373c5f209a2137ee0b53fb23ba99a5381f9d71475b0aa77",
        "config_sha256": "91a1fcd5f57e5aa1cbb9ff074cb2a6ebee4002c767cd459726b0b8b0b43ff6bc",
        "checkpoint_sha256": None,
        "checkpoint_state_sha256": None,
        "snapshot": None,
        "role": "opponent",
        "treatment_id": None,
        "replicate_id": None,
        "model_seed": None,
        "feature_version": "v1",
    },
)
SHA256_HEX_LENGTH: Final = 64
MIN_CROSSED_AXIS_SIZE: Final = 2
Profile = Literal["production", "ci-fixture"]


class T14BaselineError(ValueError):
    """Raised when a T1.4 baseline input is not frozen or rectangular."""


def baseline_descriptive_statistical_binding(*, profile: Profile) -> dict[str, object]:
    """Return the only non-inferential statistics contract accepted here."""
    if profile not in {"production", "ci-fixture"}:
        raise T14BaselineError("unknown T1.4 baseline profile")
    return {
        "protocol": T14_BASELINE_STATISTICAL_PROTOCOL,
        "profile": profile,
        "estimand": "frozen-checkpoint scheduled score-rate cells",
        "score_rate": {"win": 1.0, "draw": 0.5, "loss": 0.0},
        "seat_axis": [0, 1],
        "failure_handling": "retain-and-invalidate-entire-bank-batch",
        "variance_protocol": T14_VARIANCE_PROTOCOL,
        "inference": "none",
        "confidence_intervals": False,
        "power_analysis": False,
        "selection_or_model_comparison": False,
    }


def validate_t14_baseline_statistical_binding(  # noqa: C901,PLR0912 - fail closed
    raw: object,
    evaluation_binding: object,
    *,
    phase: object,
) -> Profile:
    """Validate the narrow manifest exception for a one-policy baseline batch."""
    if not isinstance(raw, Mapping):
        raise T14BaselineError("baseline descriptive statistics must be a mapping")
    profile = raw.get("profile")
    if profile not in {"production", "ci-fixture"} or dict(raw) != (
        baseline_descriptive_statistical_binding(profile=cast(Profile, profile))
    ):
        raise T14BaselineError("baseline descriptive statistics schema mismatch")
    if not isinstance(evaluation_binding, Mapping):
        raise T14BaselineError("baseline paired evaluation binding is missing")
    candidates = evaluation_binding.get("candidates")
    opponents = evaluation_binding.get("opponents")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise T14BaselineError("baseline evaluation requires exactly one candidate")
    if not isinstance(opponents, list) or len(opponents) < MIN_CROSSED_AXIS_SIZE:
        raise T14BaselineError("baseline evaluation requires at least two opponents")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping) or any(
        not isinstance(opponent, Mapping) or opponent.get("role") != "opponent"
        for opponent in opponents
    ):
        raise T14BaselineError("baseline evaluation policy metadata is invalid")
    if evaluation_binding.get("seats") != [0, 1]:
        raise T14BaselineError("baseline evaluation requires both seats")
    scenario_count = evaluation_binding.get("scenario_count")
    if type(scenario_count) is not int or scenario_count < MIN_CROSSED_AXIS_SIZE:
        raise T14BaselineError("baseline variance needs at least two scenarios")
    split = evaluation_binding.get("scenario_bank_split")
    if profile == "production":
        if phase != "T1.4":
            raise T14BaselineError("production baseline manifest phase must be T1.4")
        if split not in {"validation-A", "stress"}:
            raise T14BaselineError(
                "production baseline permits only validation-A or stress"
            )
        if (
            candidate.get("role") != "frozen-baseline"
            or candidate.get("policy_id") != "ppo-best"
            or candidate.get("policy_sha256") != T14_PPO_BEST_SHA256
            or candidate.get("checkpoint_sha256") != T14_PPO_BEST_SHA256
            or candidate.get("source_sha256") != T14_PPO_BEST_SOURCE_SHA256
            or candidate.get("config_sha256") != T14_PPO_BEST_CONFIG_SHA256
            or candidate.get("checkpoint_state_sha256")
            != T14_PPO_BEST_STATE_SHA256
            or candidate.get("feature_version") != "public-v2"
        ):
            raise T14BaselineError(
                "production descriptive manifest must freeze public-v2 ppo-best"
            )
        observed_opponents = tuple(
            sorted(
                (dict(opponent) for opponent in opponents),
                key=lambda opponent: cast(str, opponent["policy_id"]),
            )
        )
        if observed_opponents != T14_BASELINE_OPPONENTS:
            raise T14BaselineError(
                "production baseline opponents must be the frozen "
                "ga/heuristic/minimax source-config suite"
            )
    elif (
        phase != "T1.4-baseline-ci"
        or split != "ci-fixture"
        or candidate.get("role") != "ci-fixture"
        or candidate.get("checkpoint_sha256") is not None
    ):
        raise T14BaselineError("CI descriptive manifest is not a sealed CI fixture")
    return cast(Profile, profile)


@dataclass(frozen=True)
class BaselineCell:
    """One completed cell in the crossed scenario x seat x opponent tensor."""

    scenario_id: str
    seat: int
    opponent_sha256: str
    score_rate: float

    def __post_init__(self) -> None:
        for field in ("scenario_id", "opponent_sha256"):
            value = getattr(self, field)
            if (
                type(value) is not str
                or len(value) != SHA256_HEX_LENGTH
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise T14BaselineError(f"baseline cell {field} is not a SHA-256")
        if self.seat not in (0, 1):
            raise T14BaselineError("baseline cell seat must be 0 or 1")
        if (
            type(self.score_rate) not in {int, float}
            or not math.isfinite(self.score_rate)
            or not 0.0 <= self.score_rate <= 1.0
        ):
            raise T14BaselineError("baseline cell score_rate must lie in [0, 1]")


def _bank_binding(bank: ScenarioBank) -> dict[str, object]:
    design_sha256 = (
        None if bank.selection_design is None else bank.selection_design.sha256
    )
    return {
        "logical_split": bank.logical_split,
        "selection_kind": bank.selection_kind,
        "scenario_count": bank.scenario_count,
        "payload_sha256": bank.payload_sha256,
        "state_set_sha256": bank.state_set_sha256,
        "artifact_sha256": bank.artifact_sha256,
        "selection_design_sha256": design_sha256,
    }


def validate_t14_bank_pair(
    iid_bank: ScenarioBank,
    stress_bank: ScenarioBank,
    *,
    profile: Profile,
) -> None:
    """Reject development/test banks and any sealed access in the baseline run."""
    if profile not in {"production", "ci-fixture"}:
        raise T14BaselineError("unknown T1.4 baseline profile")
    if iid_bank.sealed or stress_bank.sealed:
        raise T14BaselineError("T1.4 baseline must never read a sealed bank")
    if profile == "production":
        if (
            iid_bank.logical_split != "validation-A"
            or iid_bank.selection_kind != "iid"
        ):
            raise T14BaselineError(
                "production IID baseline requires exactly validation-A; "
                "validation-B and sealed-test-iid are forbidden"
            )
        if (
            stress_bank.logical_split != "stress"
            or stress_bank.selection_kind != "stress-balanced"
        ):
            raise T14BaselineError(
                "production stress baseline requires the pre-registered "
                "stress-balanced bank"
            )
    elif (
        iid_bank.logical_split != "ci-fixture"
        or iid_bank.selection_kind != "ci-fixture"
        or stress_bank.logical_split != "ci-fixture"
        or stress_bank.selection_kind != "stress-balanced"
    ):
        raise T14BaselineError(
            "CI profile requires separate ci-fixture IID and stress-balanced banks"
        )
    # Each loaded bank already attests its registered source segment; this is
    # the additional cross-bank identity audit (CI intentionally uses one
    # source segment for both fixture banks).
    iid_ids = {scenario.scenario_id for scenario in iid_bank.scenarios}
    stress_ids = {scenario.scenario_id for scenario in stress_bank.scenarios}
    iid_states = {scenario.canonical_state_sha256 for scenario in iid_bank.scenarios}
    stress_states = {
        scenario.canonical_state_sha256 for scenario in stress_bank.scenarios
    }
    iid_seeds = {scenario.source_seed for scenario in iid_bank.scenarios}
    stress_seeds = {scenario.source_seed for scenario in stress_bank.scenarios}
    if iid_ids & stress_ids or iid_states & stress_states or iid_seeds & stress_seeds:
        raise T14BaselineError("IID and stress banks overlap")


@dataclass(frozen=True)
class T14BaselineSpec:
    """Two manifest-bound batches that differ only in their scenario bank."""

    iid_evaluation: PairedEvaluationSpec
    stress_evaluation: PairedEvaluationSpec
    expected_checkpoint_sha256: str | None
    profile: Profile = "production"

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - frozen axes audited
        iid = self.iid_evaluation
        stress = self.stress_evaluation
        validate_t14_bank_pair(
            iid.scenario_bank,
            stress.scenario_bank,
            profile=self.profile,
        )
        if (
            iid.experiment_id != stress.experiment_id
            or iid.phase != stress.phase
            or iid.code_sha256 != stress.code_sha256
        ):
            raise T14BaselineError(
                "IID and stress evaluations must share experiment, phase, and code"
            )
        if iid.batch_id == stress.batch_id:
            raise T14BaselineError("IID and stress batches need distinct batch IDs")
        if iid.episodes_filename == stress.episodes_filename:
            raise T14BaselineError("IID and stress episode ledgers must be distinct")
        if len(iid.candidates) != 1 or len(stress.candidates) != 1:
            raise T14BaselineError("T1.4 baseline freezes exactly one checkpoint")
        iid_candidate = iid.candidates[0]
        stress_candidate = stress.candidates[0]
        if iid_candidate.metadata() != stress_candidate.metadata():
            raise T14BaselineError("IID and stress batches changed the baseline policy")
        iid_opponents = tuple(
            policy.metadata()
            for policy in sorted(iid.opponents, key=lambda item: item.policy_sha256)
        )
        stress_opponents = tuple(
            policy.metadata()
            for policy in sorted(stress.opponents, key=lambda item: item.policy_sha256)
        )
        if iid_opponents != stress_opponents:
            raise T14BaselineError("IID and stress batches changed opponent axes")
        if len(iid.opponents) < MIN_CROSSED_AXIS_SIZE:
            raise T14BaselineError(
                "opponent variance requires at least two frozen opponents"
            )
        if self.profile == "production":
            if self.expected_checkpoint_sha256 != T14_PPO_BEST_SHA256:
                raise T14BaselineError(
                    "production baseline must pin the registered ppo-best SHA-256"
                )
            if (
                iid_candidate.role != "frozen-baseline"
                or iid_candidate.policy_id != "ppo-best"
                or iid_candidate.candidate.feature_version != "public-v2"
                or iid_candidate.checkpoint_sha256
                != self.expected_checkpoint_sha256
                or iid_candidate.source_sha256 != T14_PPO_BEST_SOURCE_SHA256
                or iid_candidate.config_sha256 != T14_PPO_BEST_CONFIG_SHA256
                or iid_candidate.checkpoint_state_sha256
                != T14_PPO_BEST_STATE_SHA256
            ):
                raise T14BaselineError(
                    "production baseline policy is not the frozen public-v2 ppo-best"
                )
            observed_opponents = tuple(
                sorted(
                    (policy.metadata() for policy in iid.opponents),
                    key=lambda opponent: cast(str, opponent["policy_id"]),
                )
            )
            if observed_opponents != T14_BASELINE_OPPONENTS:
                raise T14BaselineError(
                    "production baseline opponents are not the frozen "
                    "ga/heuristic/minimax suite"
                )
        elif (
            self.expected_checkpoint_sha256 is not None
            or iid_candidate.role != "ci-fixture"
        ):
            raise T14BaselineError(
                "CI baseline must be a snapshotless ci-fixture policy"
            )
        if self.profile == "production" and iid.phase != "T1.4":
            raise T14BaselineError("production baseline phase must be exactly T1.4")

    @property
    def baseline_policy(self) -> EvaluationPolicy:
        return self.iid_evaluation.candidates[0]

    def manifest_binding(self) -> dict[str, object]:
        """Return the canonical, hashable declaration for this two-bank run."""
        return {
            "protocol": T14_BASELINE_PROTOCOL,
            "profile": self.profile,
            "experiment_id": self.iid_evaluation.experiment_id,
            "phase": self.iid_evaluation.phase,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "baseline": self.baseline_policy.metadata(),
            "opponents": [
                policy.metadata()
                for policy in sorted(
                    self.iid_evaluation.opponents,
                    key=lambda item: item.policy_sha256,
                )
            ],
            "iid_bank": _bank_binding(self.iid_evaluation.scenario_bank),
            "stress_bank": _bank_binding(self.stress_evaluation.scenario_bank),
            "iid_evaluation": self.iid_evaluation.manifest_binding(),
            "iid_manifest_declaration_sha256": (
                self.iid_evaluation.manifest_declaration_sha256
            ),
            "stress_evaluation": self.stress_evaluation.manifest_binding(),
            "stress_manifest_declaration_sha256": (
                self.stress_evaluation.manifest_declaration_sha256
            ),
            "matrix": "one baseline x every scenario x seats[0,1] x every opponent",
            "score_rate": {"win": 1.0, "draw": 0.5, "loss": 0.0},
            "failure_handling": "retain-and-invalidate-bank-analysis",
            "variance_protocol": T14_VARIANCE_PROTOCOL,
            "stress_estimand": "unweighted-balanced-diagnostic",
        }

    @property
    def config_sha256(self) -> str:
        return sha256_canonical_json(self.manifest_binding())


def finite_population_variance_decomposition(
    cells: Sequence[BaselineCell],
) -> dict[str, object]:
    """Orthogonally decompose a complete balanced crossed score tensor.

    This is a deterministic finite-population functional ANOVA (population
    divisor, ``ddof=0``), not a fitted Gaussian random-effects model.  With one
    outcome per cell, all two- and three-way terms are pooled as interaction.
    """
    if not cells:
        raise T14BaselineError("variance decomposition needs completed cells")
    scenarios = tuple(sorted({cell.scenario_id for cell in cells}))
    seats = tuple(sorted({cell.seat for cell in cells}))
    opponents = tuple(sorted({cell.opponent_sha256 for cell in cells}))
    if (
        seats != (0, 1)
        or len(opponents) < MIN_CROSSED_AXIS_SIZE
        or len(scenarios) < MIN_CROSSED_AXIS_SIZE
    ):
        raise T14BaselineError(
            "variance tensor requires >=2 scenarios, both seats, and >=2 opponents"
        )
    values: dict[tuple[str, int, str], float] = {}
    for cell in cells:
        key = (cell.scenario_id, cell.seat, cell.opponent_sha256)
        if key in values:
            raise T14BaselineError("variance tensor contains a duplicate cell")
        values[key] = float(cell.score_rate)
    expected = {
        (scenario, seat, opponent)
        for scenario in scenarios
        for seat in seats
        for opponent in opponents
    }
    if set(values) != expected:
        raise T14BaselineError("variance tensor is not a complete crossed matrix")

    def mean(keys: Sequence[tuple[str, int, str]]) -> float:
        return math.fsum(values[key] for key in keys) / len(keys)

    grand = math.fsum(values.values()) / len(values)
    deal_effect = {
        scenario: mean(
            [(scenario, seat, opponent) for seat in seats for opponent in opponents]
        )
        - grand
        for scenario in scenarios
    }
    seat_effect = {
        seat: mean(
            [(scenario, seat, opponent) for scenario in scenarios for opponent in opponents]
        )
        - grand
        for seat in seats
    }
    opponent_effect = {
        opponent: mean(
            [(scenario, seat, opponent) for scenario in scenarios for seat in seats]
        )
        - grand
        for opponent in opponents
    }
    deal_seat = {
        (scenario, seat): mean(
            [(scenario, seat, opponent) for opponent in opponents]
        )
        - grand
        - deal_effect[scenario]
        - seat_effect[seat]
        for scenario in scenarios
        for seat in seats
    }
    deal_opponent = {
        (scenario, opponent): mean(
            [(scenario, seat, opponent) for seat in seats]
        )
        - grand
        - deal_effect[scenario]
        - opponent_effect[opponent]
        for scenario in scenarios
        for opponent in opponents
    }
    seat_opponent = {
        (seat, opponent): mean(
            [(scenario, seat, opponent) for scenario in scenarios]
        )
        - grand
        - seat_effect[seat]
        - opponent_effect[opponent]
        for seat in seats
        for opponent in opponents
    }
    three_way = {
        key: value
        - grand
        - deal_effect[key[0]]
        - seat_effect[key[1]]
        - opponent_effect[key[2]]
        - deal_seat[(key[0], key[1])]
        - deal_opponent[(key[0], key[2])]
        - seat_opponent[(key[1], key[2])]
        for key, value in values.items()
    }

    def variance(effect_values: Sequence[float]) -> float:
        return math.fsum(value * value for value in effect_values) / len(effect_values)

    breakdown = {
        "deal_x_seat": variance(tuple(deal_seat.values())),
        "deal_x_opponent": variance(tuple(deal_opponent.values())),
        "seat_x_opponent": variance(tuple(seat_opponent.values())),
        "deal_x_seat_x_opponent": variance(tuple(three_way.values())),
    }
    components = {
        "deal": variance(tuple(deal_effect.values())),
        "seat": variance(tuple(seat_effect.values())),
        "opponent": variance(tuple(opponent_effect.values())),
        "interaction": math.fsum(breakdown.values()),
    }
    total = math.fsum((value - grand) ** 2 for value in values.values()) / len(values)
    component_sum = math.fsum(components.values())
    if not math.isclose(total, component_sum, rel_tol=0.0, abs_tol=1e-12):
        raise T14BaselineError("variance decomposition failed its sum invariant")
    fractions = {
        name: (value / total if total else 0.0)
        for name, value in components.items()
    }
    return {
        "protocol": T14_VARIANCE_PROTOCOL,
        "estimand": "finite-population variance of scheduled score-rate cells",
        "ddof": 0,
        "scenario_count": len(scenarios),
        "seat_count": len(seats),
        "opponent_count": len(opponents),
        "cell_count": len(values),
        "grand_mean": grand,
        "total_variance": total,
        "component_variances": components,
        "component_fractions": fractions,
        "interaction_breakdown": breakdown,
        "identifiability_note": (
            "one observation per cell: interaction pools all crossed residual terms; "
            "components are descriptive finite-population energies, not Gaussian "
            "random-effect variance estimates"
        ),
    }


def _record_set_sha256(records: Sequence[Mapping[str, object]]) -> str:
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: sha256_canonical_json(record["episode_key"]),
    )
    return sha256_canonical_json(ordered)


def _summary_rows(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        completed = [row for row in rows if row["status"] == "completed"]
        wins = sum(row["outcome"] == 1 for row in completed)
        draws = sum(row["outcome"] == 0 for row in completed)
        losses = sum(row["outcome"] == -1 for row in completed)
        failures = len(rows) - len(completed)
        numerator = wins + 0.5 * draws
        return {
            "scheduled": len(rows),
            "completed": len(completed),
            "failed": failures,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": numerator / len(rows) if failures == 0 else None,
            "completed_score_rate": (
                numerator / len(completed) if completed else None
            ),
        }

    by_opponent: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_seat: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        opponent = cast(Mapping[str, object], record["opponent"])
        by_opponent[str(opponent["policy_id"])].append(record)
        by_seat[cast(int, record["seat"])].append(record)
    return {
        "overall": summarize(records),
        "by_opponent": {
            key: summarize(value) for key, value in sorted(by_opponent.items())
        },
        "by_seat": {
            str(key): summarize(value) for key, value in sorted(by_seat.items())
        },
    }


def _bank_analysis(
    evaluation: PairedEvaluationSpec,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    audit = audit_episode_batch(evaluation, records)
    failures = [
        {
            "episode_key": record["episode_key"],
            "failure": record["failure"],
        }
        for record in records
        if record["status"] == "failed"
    ]
    variance: dict[str, object] | None = None
    if audit["status"] == "valid":
        cells = [
            BaselineCell(
                scenario_id=cast(str, record["scenario_id"]),
                seat=cast(int, record["seat"]),
                opponent_sha256=cast(
                    str, cast(Mapping[str, object], record["opponent"])["policy_sha256"]
                ),
                score_rate=float(cast(float, record["score_rate"])),
            )
            for record in records
        ]
        variance = finite_population_variance_decomposition(cells)
    return {
        "bank": _bank_binding(evaluation.scenario_bank),
        "evaluation_batch_id": evaluation.batch_id,
        "schedule_sha256": evaluation.schedule_sha256,
        "episode_records_sha256": _record_set_sha256(records),
        "audit": audit,
        "wdl_and_score_rate": _summary_rows(records),
        "failures": failures,
        "variance_decomposition": variance,
    }


def build_t14_baseline_document(
    spec: T14BaselineSpec,
    *,
    iid_records: Sequence[Mapping[str, object]],
    stress_records: Sequence[Mapping[str, object]],
    episode_artifact_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a fully recomputable report without silently dropping failures."""
    artifacts = dict(episode_artifact_sha256 or {})
    if artifacts and (
        set(artifacts) != {"validation-A", "stress"}
        or any(
            len(value) != SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in value)
            for value in artifacts.values()
        )
    ):
        raise T14BaselineError("episode artifact hash binding is invalid")
    iid = _bank_analysis(spec.iid_evaluation, iid_records)
    stress = _bank_analysis(spec.stress_evaluation, stress_records)
    core: dict[str, object] = {
        "schema_version": T14_BASELINE_REPORT_SCHEMA,
        "protocol": T14_BASELINE_PROTOCOL,
        "config": spec.manifest_binding(),
        "config_sha256": spec.config_sha256,
        "episode_artifact_sha256": artifacts,
        "results": {"validation-A": iid, "stress": stress},
        "overall_status": (
            "valid"
            if iid["audit"]["status"] == stress["audit"]["status"] == "valid"  # type: ignore[index]
            else "invalid"
        ),
        "historical_comparison_rule": (
            "new-protocol baseline only; do not pool its interval or denominator "
            "with C4-R2 pseudo-replicated games"
        ),
        "artifact_hash_rule": (
            "SHA-256 of exact canonical JSON bytes; digest is encoded in the "
            "content-addressed report filename"
        ),
    }
    return core


def _write_content_addressed_json(
    output_dir: Path,
    document: Mapping[str, object],
    *,
    prefix: str,
) -> dict[str, object]:
    payload = canonical_json_bytes(dict(document))
    digest = hashlib.sha256(payload).hexdigest()
    path = Path(output_dir) / f"{prefix}-{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise T14BaselineError("existing baseline report fails content-address audit")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o444)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            raise
    return {"path": str(path), "content_sha256": digest}


def write_t14_baseline_document(
    output_dir: Path,
    document: Mapping[str, object],
) -> dict[str, object]:
    """Publish canonical bytes once under their SHA-256 content address."""
    return _write_content_addressed_json(
        output_dir,
        document,
        prefix="t14-baseline",
    )


def run_t14_baseline_evaluation(
    spec: T14BaselineSpec,
    output_dir: Path,
    *,
    iid_manifest: Path,
    stress_manifest: Path,
) -> dict[str, object]:
    """Preflight both running manifests, resume both matrices, and report."""
    try:
        require_paired_evaluation_manifest(iid_manifest, spec.iid_evaluation)
        require_paired_evaluation_manifest(stress_manifest, spec.stress_evaluation)
    except PairedEvaluationError as exc:
        raise T14BaselineError(f"T1.4 manifest preflight failed: {exc}") from exc
    iid_path = Path(output_dir) / spec.iid_evaluation.episodes_filename
    stress_path = Path(output_dir) / spec.stress_evaluation.episodes_filename
    run_paired_evaluation(
        spec.iid_evaluation,
        iid_path,
        source_manifest=iid_manifest,
    )
    run_paired_evaluation(
        spec.stress_evaluation,
        stress_path,
        source_manifest=stress_manifest,
    )
    document = build_t14_baseline_document(
        spec,
        iid_records=load_episode_records(iid_path),
        stress_records=load_episode_records(stress_path),
        episode_artifact_sha256={
            "validation-A": sha256_file(iid_path),
            "stress": sha256_file(stress_path),
        },
    )
    artifact = write_t14_baseline_document(output_dir, document)
    return {"document": document, "artifact": artifact}


__all__ = [
    "T14_BASELINE_OPPONENTS",
    "T14_BASELINE_PROTOCOL",
    "T14_BASELINE_REPORT_SCHEMA",
    "T14_BASELINE_STATISTICAL_PROTOCOL",
    "T14_PPO_BEST_CONFIG_SHA256",
    "T14_PPO_BEST_SHA256",
    "T14_PPO_BEST_SOURCE_SHA256",
    "T14_PPO_BEST_STATE_SHA256",
    "T14_VARIANCE_PROTOCOL",
    "BaselineCell",
    "T14BaselineError",
    "T14BaselineSpec",
    "baseline_descriptive_statistical_binding",
    "build_t14_baseline_document",
    "finite_population_variance_decomposition",
    "run_t14_baseline_evaluation",
    "validate_t14_bank_pair",
    "validate_t14_baseline_statistical_binding",
    "write_t14_baseline_document",
]
