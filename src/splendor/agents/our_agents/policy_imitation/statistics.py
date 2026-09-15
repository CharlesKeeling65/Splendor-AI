"""Paired, cluster-aware statistics for Task-1 evaluation episodes.

The functions here never treat seats or games as independent samples.  They
first collapse both seats inside each immutable scenario, preserve opponent
endpoints under one joint resample, and use a separate outer model-replicate
resample for method-level claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Final, Literal, cast

import numpy as np

from .paired_evaluation import (
    PairedEvaluationError,
    PairedEvaluationSpec,
    audit_episode_batch,
    require_paired_evaluation_manifest,
    validate_episode_record,
    validate_paired_evaluation_binding,
)
from .protocol import sha256_canonical_json

STATISTICS_SCHEMA_VERSION: Final = "splendor-paired-statistics/1"
CLUSTER_BOOTSTRAP_PROTOCOL: Final = "joint-scenario-cluster-bootstrap-v1"
NESTED_BOOTSTRAP_PROTOCOL: Final = "replicate-scenario-nested-bootstrap-v1"
POWER_PROTOCOL: Final = "paired-fixed-n-power-v1"
POWER_VARIANCE_TRANSFER_PROTOCOL: Final = "matched-unit-opponent-scale-v1"
MIN_NESTED_REPLICATES: Final = 5
MIN_PILOT_REPLICATES: Final = 3
MIN_RESAMPLES: Final = 1_000
MIN_POWER_N: Final = 2
SCENARIO_BLOCK_SIZE: Final = 50
MIN_FINAL_IID_SCENARIOS: Final = 200
SHA256_HEX_LENGTH: Final = 64
FORMAL_FAMILYWISE_ALPHA: Final = 0.05
MINIMUM_TARGET_POWER: Final = 0.8
Weighting = Literal["iid", "unweighted-diagnostic", "inverse-inclusion"]
Alternative = Literal["two-sided", "greater"]
PowerUnit = Literal["scenario", "model_replicate"]
ContrastKind = Literal["deployment", "method"]
HypothesisKind = Literal["superiority", "noninferiority", "exploratory"]
PowerDataRole = Literal["current-batch-pilot", "approved-pilot-fixed-n"]
EndpointKey = tuple[str, str]


class StatisticsError(ValueError):
    """Raised when an estimand would use an invalid independence structure."""


def _number(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise StatisticsError(f"{field} must be a finite number")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric):
        raise StatisticsError(f"{field} must be a finite number")
    return numeric


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise StatisticsError(f"{field} must be an integer")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class HypothesisFamily:
    """One pre-registered family with an explicit direction and threshold."""

    family_id: str
    kind: HypothesisKind
    endpoints: tuple[str, ...]
    alternative: Alternative
    minimum_effect: float | None = None
    noninferiority_margin: float | None = None

    def __post_init__(self) -> None:
        if type(self.family_id) is not str or not self.family_id:
            raise StatisticsError("hypothesis family_id must not be empty")
        if self.kind not in {"superiority", "noninferiority", "exploratory"}:
            raise StatisticsError("hypothesis family kind is invalid")
        if (
            not self.endpoints
            or any(
                type(endpoint) is not str or not endpoint for endpoint in self.endpoints
            )
            or tuple(sorted(set(self.endpoints))) != self.endpoints
        ):
            raise StatisticsError(
                "hypothesis endpoints must be a sorted unique non-empty tuple"
            )
        if self.alternative not in {"two-sided", "greater"}:
            raise StatisticsError("hypothesis alternative is invalid")
        if self.kind == "superiority":
            if (
                self.alternative != "greater"
                or self.minimum_effect is None
                or type(self.minimum_effect) not in {int, float}
                or not math.isfinite(self.minimum_effect)
                or not 0.0 < self.minimum_effect <= 1.0
                or self.noninferiority_margin is not None
            ):
                raise StatisticsError("superiority family threshold is invalid")
        elif self.kind == "noninferiority":
            if (
                self.alternative != "greater"
                or self.noninferiority_margin is None
                or type(self.noninferiority_margin) not in {int, float}
                or not math.isfinite(self.noninferiority_margin)
                or not 0.0 < self.noninferiority_margin <= 1.0
                or self.minimum_effect is not None
            ):
                raise StatisticsError("noninferiority family margin is invalid")
        elif (
            self.alternative != "two-sided"
            or self.minimum_effect is not None
            or self.noninferiority_margin is not None
        ):
            raise StatisticsError("exploratory families cannot carry a decision margin")

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "kind": self.kind,
            "endpoints": list(self.endpoints),
            "alternative": self.alternative,
            "minimum_effect": self.minimum_effect,
            "noninferiority_margin": self.noninferiority_margin,
        }


@dataclass(frozen=True)
class AnalysisContrast:
    """A manifest-bound linear contrast computed from episode score rates."""

    contrast_id: str
    kind: ContrastKind
    family_id: str
    components: tuple[tuple[str, float], ...]
    power_unit: PowerUnit | None = None

    def __post_init__(self) -> None:
        if type(self.contrast_id) is not str or not self.contrast_id:
            raise StatisticsError("analysis contrast_id must not be empty")
        if self.kind not in {"deployment", "method"}:
            raise StatisticsError("analysis contrast kind is invalid")
        if type(self.family_id) is not str or not self.family_id:
            raise StatisticsError("analysis contrast family_id must not be empty")
        names = [name for name, _coefficient in self.components]
        if (
            len(self.components) < MIN_POWER_N
            or names != sorted(set(names))
            or any(
                type(name) is not str
                or not name
                or type(coefficient) not in {int, float}
                or not math.isfinite(coefficient)
                or coefficient == 0
                for name, coefficient in self.components
            )
            or not math.isclose(
                math.fsum(coefficient for _name, coefficient in self.components),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise StatisticsError(
                "analysis components must be sorted unique finite zero-sum contrasts"
            )
        if self.kind == "deployment" and any(not _is_sha256(name) for name in names):
            raise StatisticsError(
                "deployment contrast components must be checkpoint/source SHA-256s"
            )
        if self.power_unit is not None and (
            (self.kind == "deployment" and self.power_unit != "scenario")
            or (self.kind == "method" and self.power_unit != "model_replicate")
        ):
            raise StatisticsError("analysis contrast uses the wrong power unit")
        if (
            self.power_unit is not None
            and max(
                math.fsum(
                    coefficient
                    for _name, coefficient in self.components
                    if coefficient > 0
                ),
                -math.fsum(
                    coefficient
                    for _name, coefficient in self.components
                    if coefficient < 0
                ),
            )
            > 1.0
        ):
            raise StatisticsError(
                "power-selected contrast must remain a score-rate difference in [-1, 1]"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "kind": self.kind,
            "family_id": self.family_id,
            "components": [
                {"identity": name, "coefficient": coefficient}
                for name, coefficient in self.components
            ],
            "power_unit": self.power_unit,
        }


@dataclass(frozen=True)
class ScenarioScore:
    """A candidate's score rate after collapsing both seats in one scenario."""

    candidate_id: str
    candidate_sha256: str
    treatment_id: str | None
    replicate_id: int | None
    model_seed: int | None
    opponent_id: str
    opponent_sha256: str
    scenario_id: str
    score_rate: float
    selection_kind: str
    inclusion_probability: float
    selection_stratum: str | None


@dataclass(frozen=True)
class PairedDifference:
    """One paired scenario-level contrast for one opponent endpoint."""

    contrast_id: str
    replicate_id: int | None
    opponent_id: str
    opponent_sha256: str
    scenario_id: str
    value: float
    component_scores: tuple[tuple[str, float], ...]
    component_policies: tuple[tuple[str, str], ...]
    component_coefficients: tuple[tuple[str, float], ...]
    contrast_kind: ContrastKind = "deployment"
    model_seed: int | None = None
    selection_kind: str = "iid"
    inclusion_probability: float = 1.0
    selection_stratum: str | None = None

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - fail closed
        if (
            type(self.contrast_id) is not str
            or not self.contrast_id
            or type(self.opponent_id) is not str
            or not self.opponent_id
        ):
            raise StatisticsError("paired difference identities must not be empty")
        if not _is_sha256(self.opponent_sha256) or not _is_sha256(self.scenario_id):
            raise StatisticsError("paired difference hashes are invalid")
        if self.replicate_id is not None and (
            type(self.replicate_id) is not int or self.replicate_id < 0
        ):
            raise StatisticsError("paired difference replicate_id is invalid")
        if self.contrast_kind == "deployment":
            if self.replicate_id is not None or self.model_seed is not None:
                raise StatisticsError(
                    "deployment difference cannot carry replicate/model seed"
                )
        elif self.contrast_kind == "method":
            if (
                self.replicate_id is None
                or type(self.model_seed) is not int
                or self.model_seed < 0
            ):
                raise StatisticsError(
                    "method difference requires a replicate and model seed"
                )
        else:
            raise StatisticsError("paired difference contrast kind is invalid")
        if type(self.value) not in {int, float} or not math.isfinite(self.value):
            raise StatisticsError("paired difference must be finite")
        if self.selection_kind not in {"iid", "ci-fixture", "stress-balanced"}:
            raise StatisticsError("paired difference selection kind is invalid")
        if (
            type(self.inclusion_probability)
            not in {
                int,
                float,
            }
            or not math.isfinite(self.inclusion_probability)
            or not (0.0 < self.inclusion_probability <= 1.0)
        ):
            raise StatisticsError("paired difference inclusion probability is invalid")
        component_names = [name for name, _value in self.component_scores]
        if (
            not self.component_scores
            or component_names != sorted(set(component_names))
            or any(
                type(name) is not str
                or not name
                or type(value) not in {int, float}
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
                for name, value in self.component_scores
            )
        ):
            raise StatisticsError("paired difference component scores are invalid")
        if (
            not self.component_policies
            or len(self.component_policies) != len(self.component_scores)
            or [name for name, _digest in self.component_policies]
            != [name for name, _score in self.component_scores]
            or any(
                type(name) is not str or not name or not _is_sha256(digest)
                for name, digest in self.component_policies
            )
        ):
            raise StatisticsError("paired difference component policies are invalid")
        if (
            len(self.component_coefficients) != len(self.component_scores)
            or [name for name, _coefficient in self.component_coefficients]
            != component_names
            or any(
                type(coefficient) not in {int, float}
                or not math.isfinite(coefficient)
                or coefficient == 0
                for _name, coefficient in self.component_coefficients
            )
            or not math.isclose(
                math.fsum(
                    coefficient for _name, coefficient in self.component_coefficients
                ),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise StatisticsError(
                "paired difference component coefficients are invalid"
            )
        scores = dict(self.component_scores)
        expected_value = math.fsum(
            coefficient * scores[name]
            for name, coefficient in self.component_coefficients
        )
        if not math.isclose(self.value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise StatisticsError(
                "paired difference value disagrees with its linear contrast"
            )
        if self.selection_kind in {"iid", "ci-fixture"} and (
            self.inclusion_probability != 1.0 or self.selection_stratum is not None
        ):
            raise StatisticsError("IID difference carries stress sampling metadata")
        if self.selection_kind == "stress-balanced" and (
            type(self.selection_stratum) is not str or not self.selection_stratum
        ):
            raise StatisticsError("stress difference lacks its selection stratum")

    def audit_dict(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "replicate_id": self.replicate_id,
            "opponent_id": self.opponent_id,
            "opponent_sha256": self.opponent_sha256,
            "scenario_id": self.scenario_id,
            "value": self.value,
            "component_scores": dict(self.component_scores),
            "component_policies": dict(self.component_policies),
            "component_coefficients": dict(self.component_coefficients),
            "contrast_kind": self.contrast_kind,
            "model_seed": self.model_seed,
            "selection_kind": self.selection_kind,
            "inclusion_probability": self.inclusion_probability,
            "selection_stratum": self.selection_stratum,
        }


@dataclass(frozen=True)
class BootstrapConfig:
    """Frozen simultaneous percentile-bootstrap settings."""

    confidence_level: float = 0.95
    resamples: int = 10_000
    seed: int = 0
    alternative: Alternative = "two-sided"
    correction: str = "bonferroni"
    weighting: Weighting = "iid"

    def __post_init__(self) -> None:
        if type(self.confidence_level) not in {int, float} or not (
            0.0 < self.confidence_level < 1.0
        ):
            raise StatisticsError("bootstrap confidence_level must lie in (0, 1)")
        if type(self.resamples) is not int or self.resamples < MIN_RESAMPLES:
            raise StatisticsError("bootstrap requires at least 1,000 resamples")
        if type(self.seed) is not int or self.seed < 0:
            raise StatisticsError("bootstrap seed must be a non-negative integer")
        if self.alternative not in {"two-sided", "greater"}:
            raise StatisticsError("bootstrap alternative is invalid")
        if self.correction != "bonferroni":
            raise StatisticsError(
                "only pre-registered Bonferroni intervals are supported"
            )
        if self.weighting not in {
            "iid",
            "unweighted-diagnostic",
            "inverse-inclusion",
        }:
            raise StatisticsError("bootstrap weighting mode is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "resamples": self.resamples,
            "seed": self.seed,
            "alternative": self.alternative,
            "correction": self.correction,
            "weighting": self.weighting,
        }


@dataclass(frozen=True)
class PowerAnalysisConfig:
    """A fixed-N, one-sided paired power design for one independent unit."""

    unit: PowerUnit
    minimum_effect: float
    familywise_alpha: float = 0.05
    family_size: int = 1
    target_power: float = 0.8
    max_n: int = 500
    round_to: int = 1
    simulations: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - fail closed
        if self.unit not in {"scenario", "model_replicate"}:
            raise StatisticsError("power unit must be scenario or model_replicate")
        if (
            type(self.minimum_effect) not in {int, float}
            or not math.isfinite(self.minimum_effect)
            or not (0.0 < self.minimum_effect <= 1.0)
        ):
            raise StatisticsError("minimum_effect must be a score-rate proportion")
        if type(self.familywise_alpha) not in {int, float} or (
            self.familywise_alpha != FORMAL_FAMILYWISE_ALPHA
        ):
            raise StatisticsError("formal familywise_alpha must equal 0.05")
        if type(self.family_size) is not int or self.family_size < 1:
            raise StatisticsError("power family_size must be positive")
        if type(self.target_power) not in {int, float} or not (
            MINIMUM_TARGET_POWER <= self.target_power < 1.0
        ):
            raise StatisticsError("target_power must lie in [0.8, 1)")
        if type(self.max_n) is not int or self.max_n < MIN_POWER_N:
            raise StatisticsError("power max_n must be at least two")
        if type(self.round_to) is not int or self.round_to < 1:
            raise StatisticsError("power round_to must be positive")
        if self.max_n % self.round_to:
            raise StatisticsError("power max_n must be divisible by round_to")
        if type(self.simulations) is not int or self.simulations < MIN_RESAMPLES:
            raise StatisticsError("power analysis requires at least 1,000 simulations")
        if type(self.seed) is not int or self.seed < 0:
            raise StatisticsError("power seed must be a non-negative integer")
        if self.unit == "scenario" and self.round_to != SCENARIO_BLOCK_SIZE:
            raise StatisticsError("formal scenario N must use fixed 50-scenario blocks")
        if self.unit == "scenario" and self.max_n < MIN_FINAL_IID_SCENARIOS:
            raise StatisticsError(
                "formal scenario power max_n cannot be below the 200-scenario floor"
            )
        if self.unit == "model_replicate" and self.max_n < MIN_NESTED_REPLICATES:
            raise StatisticsError(
                "method power max_n cannot be below five model replicates"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "minimum_effect": self.minimum_effect,
            "familywise_alpha": self.familywise_alpha,
            "family_size": self.family_size,
            "target_power": self.target_power,
            "max_n": self.max_n,
            "round_to": self.round_to,
            "simulations": self.simulations,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PowerSourceSpec:
    """Bind prospective power either to this pilot or to an approved pilot."""

    data_role: PowerDataRole
    pilot_manifest_declaration_sha256: str | None = None
    pilot_statistics_content_sha256: str | None = None
    fixed_scenario_n: int | None = None
    fixed_model_replicates: int | None = None
    pilot_power_design_sha256: str | None = None
    variance_transfer_protocol: str | None = None

    def __post_init__(self) -> None:
        fields = (
            self.pilot_manifest_declaration_sha256,
            self.pilot_statistics_content_sha256,
            self.fixed_scenario_n,
            self.fixed_model_replicates,
            self.pilot_power_design_sha256,
            self.variance_transfer_protocol,
        )
        if self.data_role == "current-batch-pilot":
            if any(value is not None for value in fields):
                raise StatisticsError(
                    "current-batch pilot cannot carry fixed-N or prior-artifact fields"
                )
            return
        if self.data_role != "approved-pilot-fixed-n":
            raise StatisticsError("power data role is invalid")
        if not _is_sha256(self.pilot_manifest_declaration_sha256) or not _is_sha256(
            self.pilot_statistics_content_sha256
        ):
            raise StatisticsError(
                "fixed-N power source requires pilot manifest/statistics hashes"
            )
        if not _is_sha256(self.pilot_power_design_sha256):
            raise StatisticsError(
                "fixed-N power source requires a pilot power-design SHA-256"
            )
        if self.variance_transfer_protocol != POWER_VARIANCE_TRANSFER_PROTOCOL:
            raise StatisticsError(
                "fixed-N power source requires the supported variance-transfer protocol"
            )
        if (
            type(self.fixed_scenario_n) is not int
            or self.fixed_scenario_n < MIN_FINAL_IID_SCENARIOS
            or self.fixed_scenario_n % SCENARIO_BLOCK_SIZE
        ):
            raise StatisticsError(
                "fixed scenario N must be at least 200 and use 50-scenario blocks"
            )
        if (
            type(self.fixed_model_replicates) is not int
            or self.fixed_model_replicates < MIN_NESTED_REPLICATES
        ):
            raise StatisticsError("fixed method design requires at least 5 replicates")

    def to_dict(self) -> dict[str, object]:
        return {
            "data_role": self.data_role,
            "pilot_manifest_declaration_sha256": (
                self.pilot_manifest_declaration_sha256
            ),
            "pilot_statistics_content_sha256": (self.pilot_statistics_content_sha256),
            "fixed_scenario_n": self.fixed_scenario_n,
            "fixed_model_replicates": self.fixed_model_replicates,
            "pilot_power_design_sha256": self.pilot_power_design_sha256,
            "variance_transfer_protocol": self.variance_transfer_protocol,
        }


@dataclass(frozen=True)
class StatisticalProtocolSpec:
    """The immutable statistical choices embedded in a manifest v2."""

    hypothesis_families: tuple[HypothesisFamily, ...]
    analysis_contrasts: tuple[AnalysisContrast, ...]
    scenario_bootstrap: BootstrapConfig
    nested_bootstrap: BootstrapConfig
    scenario_power: PowerAnalysisConfig
    replicate_power: PowerAnalysisConfig
    power_source: PowerSourceSpec
    stress_weighting: Literal["unweighted-diagnostic", "inverse-inclusion"] = (
        "unweighted-diagnostic"
    )

    def __post_init__(self) -> None:  # noqa: C901,PLR0912 - cross-field contract
        if not self.hypothesis_families:
            raise StatisticsError("statistical protocol needs hypothesis families")
        family_ids = [family.family_id for family in self.hypothesis_families]
        if family_ids != sorted(set(family_ids)):
            raise StatisticsError(
                "hypothesis families must be sorted and unique by family_id"
            )
        if not self.analysis_contrasts:
            raise StatisticsError("statistical protocol needs analysis contrasts")
        contrast_ids = [contrast.contrast_id for contrast in self.analysis_contrasts]
        if contrast_ids != sorted(set(contrast_ids)):
            raise StatisticsError(
                "analysis contrasts must be sorted and unique by contrast_id"
            )
        known_families = set(family_ids)
        used_families = {contrast.family_id for contrast in self.analysis_contrasts}
        if used_families != known_families:
            raise StatisticsError(
                "every hypothesis family must have one or more bound contrasts"
            )
        if self.scenario_bootstrap.weighting != "iid":
            raise StatisticsError("primary scenario bootstrap must use IID weighting")
        if self.nested_bootstrap.weighting != "iid":
            raise StatisticsError("primary nested bootstrap must use IID weighting")
        if self.scenario_power.unit != "scenario":
            raise StatisticsError("scenario power config has the wrong unit")
        if self.replicate_power.unit != "model_replicate":
            raise StatisticsError("replicate power config has the wrong unit")
        power_contrasts = {
            unit: [
                contrast
                for contrast in self.analysis_contrasts
                if contrast.power_unit == unit
            ]
            for unit in ("scenario", "model_replicate")
        }
        if any(len(contrasts) != 1 for contrasts in power_contrasts.values()):
            raise StatisticsError(
                "statistical protocol requires exactly one contrast per power unit"
            )
        family_by_id = {family.family_id: family for family in self.hypothesis_families}
        for unit, config in (
            ("scenario", self.scenario_power),
            ("model_replicate", self.replicate_power),
        ):
            family = family_by_id[power_contrasts[unit][0].family_id]
            expected_effect = (
                family.minimum_effect
                if family.kind == "superiority"
                else family.noninferiority_margin
            )
            if family.kind == "exploratory" or config.minimum_effect != expected_effect:
                raise StatisticsError(
                    f"{unit} power effect must match its decision-family threshold"
                )
            if config.family_size != len(family.endpoints):
                raise StatisticsError(
                    f"{unit} power family_size must equal its hypothesis-family "
                    "endpoint count"
                )
        if (
            self.scenario_power.familywise_alpha
            != self.replicate_power.familywise_alpha
        ):
            raise StatisticsError("power configs must share familywise alpha")
        if self.scenario_power.target_power != self.replicate_power.target_power:
            raise StatisticsError("power configs must share target power")
        if self.power_source.data_role == "approved-pilot-fixed-n":
            fixed_scenario_n = cast(int, self.power_source.fixed_scenario_n)
            fixed_replicates = cast(int, self.power_source.fixed_model_replicates)
            if (
                fixed_scenario_n > self.scenario_power.max_n
                or fixed_replicates > self.replicate_power.max_n
            ):
                raise StatisticsError("fixed N exceeds its pre-registered power cap")
            if fixed_replicates % self.replicate_power.round_to:
                raise StatisticsError(
                    "fixed model-replicate N must respect power round_to"
                )
        if self.stress_weighting not in {
            "unweighted-diagnostic",
            "inverse-inclusion",
        }:
            raise StatisticsError("stress weighting declaration is invalid")

    def manifest_binding(self) -> dict[str, object]:
        return {
            "protocol": "task1-statistical-protocol-v1",
            "score_rate": {"win": 1.0, "draw": 0.5, "loss": 0.0},
            "seat_aggregation": "mean-both-seats-before-contrast",
            "failure_handling": "retain-and-invalidate-entire-formal-batch",
            "missing_value_handling": "invalidate-entire-formal-batch",
            "wilson_usage": "descriptive-only",
            "hypothesis_families": [
                family.to_dict() for family in self.hypothesis_families
            ],
            "analysis_contrasts": [
                contrast.to_dict() for contrast in self.analysis_contrasts
            ],
            "bootstrap": {
                "scenario": self.scenario_bootstrap.to_dict(),
                "nested_replicate_scenario": self.nested_bootstrap.to_dict(),
                "joint_endpoint_resampling": True,
                "scenario_cluster": "scenario_id after two-seat aggregation",
            },
            "power": {
                "design": "fixed-N",
                "sequential_stopping": False,
                "source": self.power_source.to_dict(),
                "scenario": self.scenario_power.to_dict(),
                "model_replicate": self.replicate_power.to_dict(),
            },
            "stress_weighting": self.stress_weighting,
        }


def _exact_mapping(
    raw: object,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise StatisticsError(f"{label} schema mismatch")
    return cast(Mapping[str, object], raw)


def _bootstrap_config_from_dict(raw: object, label: str) -> BootstrapConfig:
    fields = {
        "confidence_level",
        "resamples",
        "seed",
        "alternative",
        "correction",
        "weighting",
    }
    values = _exact_mapping(raw, fields, label)
    alternative = values["alternative"]
    weighting = values["weighting"]
    if alternative not in {"two-sided", "greater"}:
        raise StatisticsError(f"{label} alternative is invalid")
    if weighting not in {"iid", "unweighted-diagnostic", "inverse-inclusion"}:
        raise StatisticsError(f"{label} weighting is invalid")
    return BootstrapConfig(
        confidence_level=_number(values["confidence_level"], "confidence_level"),
        resamples=_integer(values["resamples"], "resamples"),
        seed=_integer(values["seed"], "seed"),
        alternative=cast(Alternative, alternative),
        correction=str(values["correction"]),
        weighting=cast(Weighting, weighting),
    )


def _power_config_from_dict(raw: object, label: str) -> PowerAnalysisConfig:
    fields = {
        "unit",
        "minimum_effect",
        "familywise_alpha",
        "family_size",
        "target_power",
        "max_n",
        "round_to",
        "simulations",
        "seed",
    }
    values = _exact_mapping(raw, fields, label)
    unit = values["unit"]
    if unit not in {"scenario", "model_replicate"}:
        raise StatisticsError(f"{label} unit is invalid")
    return PowerAnalysisConfig(
        unit=cast(PowerUnit, unit),
        minimum_effect=_number(values["minimum_effect"], "minimum_effect"),
        familywise_alpha=_number(values["familywise_alpha"], "familywise_alpha"),
        family_size=_integer(values["family_size"], "family_size"),
        target_power=_number(values["target_power"], "target_power"),
        max_n=_integer(values["max_n"], "max_n"),
        round_to=_integer(values["round_to"], "round_to"),
        simulations=_integer(values["simulations"], "simulations"),
        seed=_integer(values["seed"], "seed"),
    )


def _power_source_from_dict(raw: object) -> PowerSourceSpec:
    values = _exact_mapping(
        raw,
        {
            "data_role",
            "pilot_manifest_declaration_sha256",
            "pilot_statistics_content_sha256",
            "fixed_scenario_n",
            "fixed_model_replicates",
            "pilot_power_design_sha256",
            "variance_transfer_protocol",
        },
        "power source",
    )
    role = values["data_role"]
    if role not in {"current-batch-pilot", "approved-pilot-fixed-n"}:
        raise StatisticsError("power source data_role is invalid")

    def optional_integer(value: object, field: str) -> int | None:
        return None if value is None else _integer(value, field)

    manifest_sha256 = values["pilot_manifest_declaration_sha256"]
    statistics_sha256 = values["pilot_statistics_content_sha256"]
    design_sha256 = values["pilot_power_design_sha256"]
    transfer_protocol = values["variance_transfer_protocol"]
    if manifest_sha256 is not None and type(manifest_sha256) is not str:
        raise StatisticsError("pilot manifest declaration hash must be a string")
    if statistics_sha256 is not None and type(statistics_sha256) is not str:
        raise StatisticsError("pilot statistics content hash must be a string")
    if design_sha256 is not None and type(design_sha256) is not str:
        raise StatisticsError("pilot power-design hash must be a string")
    if transfer_protocol is not None and type(transfer_protocol) is not str:
        raise StatisticsError("variance-transfer protocol must be a string")
    return PowerSourceSpec(
        data_role=cast(PowerDataRole, role),
        pilot_manifest_declaration_sha256=cast(str | None, manifest_sha256),
        pilot_statistics_content_sha256=cast(str | None, statistics_sha256),
        fixed_scenario_n=optional_integer(
            values["fixed_scenario_n"], "fixed_scenario_n"
        ),
        fixed_model_replicates=optional_integer(
            values["fixed_model_replicates"], "fixed_model_replicates"
        ),
        pilot_power_design_sha256=cast(str | None, design_sha256),
        variance_transfer_protocol=cast(str | None, transfer_protocol),
    )


def _analysis_contrast_from_dict(raw: object) -> AnalysisContrast:
    values = _exact_mapping(
        raw,
        {"contrast_id", "kind", "family_id", "components", "power_unit"},
        "analysis contrast",
    )
    kind = values["kind"]
    power_unit = values["power_unit"]
    if kind not in {"deployment", "method"}:
        raise StatisticsError("analysis contrast kind is invalid")
    if power_unit not in {None, "scenario", "model_replicate"}:
        raise StatisticsError("analysis contrast power unit is invalid")
    raw_components = values["components"]
    if not isinstance(raw_components, list):
        raise StatisticsError("analysis contrast components must be a list")
    components: list[tuple[str, float]] = []
    for raw_component in raw_components:
        component = _exact_mapping(
            raw_component,
            {"identity", "coefficient"},
            "analysis contrast component",
        )
        identity = component["identity"]
        if type(identity) is not str:
            raise StatisticsError("analysis contrast identity must be a string")
        components.append(
            (
                identity,
                _number(component["coefficient"], "analysis coefficient"),
            )
        )
    return AnalysisContrast(
        contrast_id=str(values["contrast_id"]),
        kind=cast(ContrastKind, kind),
        family_id=str(values["family_id"]),
        components=tuple(components),
        power_unit=cast(PowerUnit | None, power_unit),
    )


def validate_statistical_protocol_binding(  # noqa: C901,PLR0912 - manifest gate
    raw: object,
) -> StatisticalProtocolSpec:
    """Parse and canonicalize the exact statistics declaration in a manifest."""
    root_fields = {
        "protocol",
        "score_rate",
        "seat_aggregation",
        "failure_handling",
        "missing_value_handling",
        "wilson_usage",
        "hypothesis_families",
        "analysis_contrasts",
        "bootstrap",
        "power",
        "stress_weighting",
    }
    binding = _exact_mapping(raw, root_fields, "statistical protocol")
    if binding["protocol"] != "task1-statistical-protocol-v1":
        raise StatisticsError("statistical protocol identifier is invalid")
    if binding["score_rate"] != {"win": 1.0, "draw": 0.5, "loss": 0.0}:
        raise StatisticsError("statistical score-rate/tie rule is invalid")
    fixed_strings = {
        "seat_aggregation": "mean-both-seats-before-contrast",
        "failure_handling": "retain-and-invalidate-entire-formal-batch",
        "missing_value_handling": "invalidate-entire-formal-batch",
        "wilson_usage": "descriptive-only",
    }
    if any(binding[field] != value for field, value in fixed_strings.items()):
        raise StatisticsError("statistical aggregation/failure contract is invalid")

    raw_families = binding["hypothesis_families"]
    if not isinstance(raw_families, list) or not raw_families:
        raise StatisticsError("statistical hypothesis families must be a list")
    families: list[HypothesisFamily] = []
    family_fields = {
        "family_id",
        "kind",
        "endpoints",
        "alternative",
        "minimum_effect",
        "noninferiority_margin",
    }
    for raw_family in raw_families:
        family = _exact_mapping(raw_family, family_fields, "hypothesis family")
        endpoints = family["endpoints"]
        if not isinstance(endpoints, list) or any(
            type(endpoint) is not str for endpoint in endpoints
        ):
            raise StatisticsError("hypothesis family endpoints are invalid")
        kind = family["kind"]
        alternative = family["alternative"]
        if kind not in {"superiority", "noninferiority", "exploratory"}:
            raise StatisticsError("hypothesis family kind is invalid")
        if alternative not in {"two-sided", "greater"}:
            raise StatisticsError("hypothesis family alternative is invalid")
        minimum_effect = family["minimum_effect"]
        margin = family["noninferiority_margin"]
        families.append(
            HypothesisFamily(
                family_id=str(family["family_id"]),
                kind=cast(HypothesisKind, kind),
                endpoints=tuple(cast(list[str], endpoints)),
                alternative=cast(Alternative, alternative),
                minimum_effect=(
                    None
                    if minimum_effect is None
                    else _number(minimum_effect, "minimum_effect")
                ),
                noninferiority_margin=(
                    None if margin is None else _number(margin, "margin")
                ),
            )
        )

    raw_contrasts = binding["analysis_contrasts"]
    if not isinstance(raw_contrasts, list) or not raw_contrasts:
        raise StatisticsError("statistical analysis contrasts must be a list")
    contrasts = tuple(
        _analysis_contrast_from_dict(raw_contrast) for raw_contrast in raw_contrasts
    )

    bootstrap = _exact_mapping(
        binding["bootstrap"],
        {
            "scenario",
            "nested_replicate_scenario",
            "joint_endpoint_resampling",
            "scenario_cluster",
        },
        "bootstrap declaration",
    )
    if (
        bootstrap["joint_endpoint_resampling"] is not True
        or bootstrap["scenario_cluster"] != "scenario_id after two-seat aggregation"
    ):
        raise StatisticsError("bootstrap cluster/joint-resampling contract is invalid")
    power = _exact_mapping(
        binding["power"],
        {
            "design",
            "sequential_stopping",
            "source",
            "scenario",
            "model_replicate",
        },
        "power declaration",
    )
    if power["design"] != "fixed-N" or power["sequential_stopping"] is not False:
        raise StatisticsError(
            "power design must be fixed-N without sequential stopping"
        )
    stress_weighting = binding["stress_weighting"]
    if stress_weighting not in {"unweighted-diagnostic", "inverse-inclusion"}:
        raise StatisticsError("stress weighting declaration is invalid")
    spec = StatisticalProtocolSpec(
        hypothesis_families=tuple(families),
        analysis_contrasts=contrasts,
        scenario_bootstrap=_bootstrap_config_from_dict(
            bootstrap["scenario"], "scenario bootstrap"
        ),
        nested_bootstrap=_bootstrap_config_from_dict(
            bootstrap["nested_replicate_scenario"], "nested bootstrap"
        ),
        scenario_power=_power_config_from_dict(power["scenario"], "scenario power"),
        replicate_power=_power_config_from_dict(
            power["model_replicate"], "replicate power"
        ),
        power_source=_power_source_from_dict(power["source"]),
        stress_weighting=cast(
            Literal["unweighted-diagnostic", "inverse-inclusion"],
            stress_weighting,
        ),
    )
    if spec.manifest_binding() != dict(binding):
        raise StatisticsError("statistical protocol declaration is not canonical")
    return spec


def _policy_metadata(record: Mapping[str, object], side: str) -> Mapping[str, object]:
    value = record[side]
    if not isinstance(value, Mapping):  # validate_episode_record catches this too
        raise StatisticsError(f"episode {side} metadata is invalid")
    return cast(Mapping[str, object], value)


def aggregate_scenario_scores(  # noqa: C901,PLR0912,PLR0915 - rectangular audit
    records: Sequence[Mapping[str, object]],
) -> tuple[ScenarioScore, ...]:
    """Validate a complete matrix and average exactly seats 0/1 per scenario."""
    if not records:
        raise StatisticsError("scenario aggregation needs episode records")
    groups: dict[tuple[str, str, str, int | None], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    candidate_metadata: dict[str, dict[str, object]] = {}
    opponent_metadata: dict[str, dict[str, object]] = {}
    scenario_metadata: dict[str, tuple[object, ...]] = {}
    common_contract: tuple[object, ...] | None = None
    for record in records:
        try:
            key = validate_episode_record(record)
        except PairedEvaluationError as exc:
            raise StatisticsError(f"invalid episode row: {exc}") from exc
        if record["status"] != "completed":
            raise StatisticsError("formal statistics reject failed episode batches")
        candidate = _policy_metadata(record, "candidate")
        metadata = dict(candidate)
        previous = candidate_metadata.setdefault(key.candidate_sha256, metadata)
        if previous != metadata:
            raise StatisticsError(
                "one checkpoint digest has conflicting candidate/replicate metadata"
            )
        opponent = _policy_metadata(record, "opponent")
        previous_opponent = opponent_metadata.setdefault(
            key.opponent_sha256, dict(opponent)
        )
        if previous_opponent != dict(opponent):
            raise StatisticsError("one opponent digest has conflicting metadata")
        scenario_contract = (
            record["canonical_state_sha256"],
            record["scenario_source_segment"],
            record["scenario_source_seed"],
            record["scenario_selection_kind"],
            record["scenario_inclusion_probability"],
            record["scenario_selection_stratum"],
            record["scenario_selection_design_sha256"],
        )
        previous_scenario = scenario_metadata.setdefault(
            key.scenario_id, scenario_contract
        )
        if previous_scenario != scenario_contract:
            raise StatisticsError("one scenario identity has conflicting metadata")
        contract = (
            record["experiment_id"],
            record["phase"],
            record["batch_id"],
            record["manifest_declaration_sha256"],
            record["code_sha256"],
            record["scenario_bank_sha256"],
            record["scenario_bank_split"],
            record["schedule_sha256"],
        )
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise StatisticsError("episode rows mix evaluation contracts")
        groups[
            (
                key.candidate_sha256,
                key.opponent_sha256,
                key.scenario_id,
                key.replicate_id,
            )
        ].append(record)

    scores: list[ScenarioScore] = []
    candidate_cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for (
        candidate_hash,
        opponent_hash,
        scenario_id,
        replicate_id,
    ), rows in groups.items():
        seats = [cast(int, row["seat"]) for row in rows]
        if len(rows) != MIN_POWER_N or set(seats) != {0, 1}:
            raise StatisticsError(
                "each candidate/opponent/scenario requires exactly both seats"
            )
        candidate = _policy_metadata(rows[0], "candidate")
        opponent = _policy_metadata(rows[0], "opponent")
        for row in rows[1:]:
            if (
                _policy_metadata(row, "candidate") != candidate
                or _policy_metadata(row, "opponent") != opponent
            ):
                raise StatisticsError("seat-paired rows changed policy metadata")
        selection = {
            (
                str(row["scenario_selection_kind"]),
                _number(
                    row["scenario_inclusion_probability"],
                    "scenario inclusion probability",
                ),
                cast(str | None, row["scenario_selection_stratum"]),
            )
            for row in rows
        }
        if len(selection) != 1:
            raise StatisticsError(
                "seat-paired rows changed scenario selection metadata"
            )
        selection_kind, inclusion_probability, selection_stratum = next(iter(selection))
        score_rate = float(
            np.mean([_number(row["score_rate"], "episode score_rate") for row in rows])
        )
        scores.append(
            ScenarioScore(
                candidate_id=str(candidate["policy_id"]),
                candidate_sha256=candidate_hash,
                treatment_id=cast(str | None, candidate["treatment_id"]),
                replicate_id=replicate_id,
                model_seed=cast(int | None, candidate["model_seed"]),
                opponent_id=str(opponent["policy_id"]),
                opponent_sha256=opponent_hash,
                scenario_id=scenario_id,
                score_rate=score_rate,
                selection_kind=selection_kind,
                inclusion_probability=inclusion_probability,
                selection_stratum=selection_stratum,
            )
        )
        candidate_cells[candidate_hash].add((opponent_hash, scenario_id))
    reference_cells = next(iter(candidate_cells.values()))
    if any(cells != reference_cells for cells in candidate_cells.values()):
        raise StatisticsError("candidate matrices do not share complete paired cells")
    return tuple(
        sorted(
            scores,
            key=lambda row: (
                row.candidate_sha256,
                row.opponent_sha256,
                row.scenario_id,
            ),
        )
    )


def checkpoint_differences(
    records: Sequence[Mapping[str, object]],
    *,
    candidate_sha256: str,
    control_sha256: str,
    contrast_id: str = "d_deploy",
) -> tuple[PairedDifference, ...]:
    """Compute candidate-control differences paired by opponent and scenario."""
    if candidate_sha256 == control_sha256:
        raise StatisticsError("checkpoint contrast needs two different digests")
    return checkpoint_contrast_differences(
        records,
        coefficients={candidate_sha256: 1.0, control_sha256: -1.0},
        contrast_id=contrast_id,
    )


def checkpoint_contrast_differences(
    records: Sequence[Mapping[str, object]],
    *,
    coefficients: Mapping[str, float],
    contrast_id: str,
) -> tuple[PairedDifference, ...]:
    """Compute a manifest-style linear contrast over fixed checkpoints."""
    identities = sorted(coefficients)
    AnalysisContrast(
        contrast_id=contrast_id,
        kind="deployment",
        family_id="validation-only",
        components=tuple((name, coefficients[name]) for name in identities),
    )
    scores = aggregate_scenario_scores(records)
    by_identity = {
        identity: {
            (row.opponent_sha256, row.scenario_id): row
            for row in scores
            if row.candidate_sha256 == identity
        }
        for identity in identities
    }
    reference = next(iter(by_identity.values()))
    if not reference or any(
        rows.keys() != reference.keys() for rows in by_identity.values()
    ):
        raise StatisticsError("checkpoint contrast lacks a complete paired matrix")
    differences = []
    for cell in sorted(reference):
        component_rows = [by_identity[identity][cell] for identity in identities]
        labels = [row.candidate_id for row in component_rows]
        if len(labels) != len(set(labels)):
            raise StatisticsError("checkpoint contrast candidate names are ambiguous")
        component_data = sorted(
            (
                row.candidate_id,
                row.score_rate,
                row.candidate_sha256,
                coefficients[row.candidate_sha256],
            )
            for row in component_rows
        )
        differences.append(
            _difference_from_components(
                contrast_id=contrast_id,
                replicate_id=None,
                rows=component_rows,
                components=tuple(
                    (name, score) for name, score, _digest, _coef in component_data
                ),
                component_policies=tuple(
                    (name, digest) for name, _score, digest, _coef in component_data
                ),
                component_coefficients=tuple(
                    (name, coefficient)
                    for name, _score, _digest, coefficient in component_data
                ),
                value=math.fsum(
                    coefficient * score
                    for _name, score, _digest, coefficient in component_data
                ),
                contrast_kind="deployment",
            )
        )
    return tuple(differences)


def _difference_from_components(  # noqa: PLR0913 - provenance stays explicit
    *,
    contrast_id: str,
    replicate_id: int | None,
    rows: Sequence[ScenarioScore],
    components: Sequence[tuple[str, float]],
    component_policies: Sequence[tuple[str, str]],
    component_coefficients: Sequence[tuple[str, float]],
    value: float,
    contrast_kind: ContrastKind,
    model_seed: int | None = None,
) -> PairedDifference:
    first = rows[0]
    if any(
        row.opponent_sha256 != first.opponent_sha256
        or row.scenario_id != first.scenario_id
        or row.selection_kind != first.selection_kind
        or row.inclusion_probability != first.inclusion_probability
        or row.selection_stratum != first.selection_stratum
        for row in rows[1:]
    ):
        raise StatisticsError("contrast components are not scenario-paired")
    return PairedDifference(
        contrast_id=contrast_id,
        replicate_id=replicate_id,
        opponent_id=first.opponent_id,
        opponent_sha256=first.opponent_sha256,
        scenario_id=first.scenario_id,
        value=value,
        component_scores=tuple(components),
        component_policies=tuple(component_policies),
        component_coefficients=tuple(component_coefficients),
        contrast_kind=contrast_kind,
        model_seed=model_seed,
        selection_kind=first.selection_kind,
        inclusion_probability=first.inclusion_probability,
        selection_stratum=first.selection_stratum,
    )


def method_contrast_differences(  # noqa: C901,PLR0912 - crossed-arm audit
    records: Sequence[Mapping[str, object]],
    *,
    coefficients: Mapping[str, float],
    contrast_id: str,
) -> tuple[PairedDifference, ...]:
    """Compute a paired treatment contrast, including 2x2 interactions."""
    if len(coefficients) < MIN_POWER_N or not contrast_id:
        raise StatisticsError("method contrast needs a name and at least two arms")
    if any(
        not name
        or type(value) not in {int, float}
        or not math.isfinite(value)
        or value == 0
        for name, value in coefficients.items()
    ):
        raise StatisticsError("method contrast coefficients are invalid")
    scores = aggregate_scenario_scores(records)
    required = set(coefficients)
    by_treatment_replicate: dict[tuple[str, int], list[ScenarioScore]] = defaultdict(
        list
    )
    candidate_identity: dict[tuple[str, int], tuple[str, int]] = {}
    for row in scores:
        if row.treatment_id not in required:
            continue
        if row.replicate_id is None or row.model_seed is None:
            raise StatisticsError(
                "method contrasts require replicate/model-seed metadata"
            )
        key = (row.treatment_id, row.replicate_id)
        identity = (row.candidate_sha256, row.model_seed)
        previous = candidate_identity.setdefault(key, identity)
        if previous != identity:
            raise StatisticsError(
                "one treatment/replicate maps to multiple checkpoints"
            )
        by_treatment_replicate[key].append(row)
    replicate_sets = [
        {
            replicate
            for treatment, replicate in by_treatment_replicate
            if treatment == arm
        }
        for arm in required
    ]
    if (
        not replicate_sets
        or not replicate_sets[0]
        or any(values != replicate_sets[0] for values in replicate_sets[1:])
    ):
        raise StatisticsError("method arms do not cover the same replicates")
    replicate_ids = sorted(replicate_sets[0])
    # Reusing a checkpoint within one arm fabricates independent replicates.
    for arm in required:
        identities = [
            candidate_identity[(arm, replicate)] for replicate in replicate_ids
        ]
        checkpoint_hashes = [identity[0] for identity in identities]
        model_seeds = [identity[1] for identity in identities]
        if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
            raise StatisticsError("checkpoint digest is reused across replicates")
        if len(model_seeds) != len(set(model_seeds)):
            raise StatisticsError("model seed is reused across replicates")

    differences: list[PairedDifference] = []
    for replicate_id in replicate_ids:
        paired_model_seeds = {
            candidate_identity[(arm, replicate_id)][1] for arm in required
        }
        if len(paired_model_seeds) != 1:
            raise StatisticsError("paired method arms do not share the model seed")
        arm_maps = {
            arm: {
                (row.opponent_sha256, row.scenario_id): row
                for row in by_treatment_replicate[(arm, replicate_id)]
            }
            for arm in required
        }
        reference_cells = next(iter(arm_maps.values())).keys()
        if any(rows.keys() != reference_cells for rows in arm_maps.values()):
            raise StatisticsError("method arms lack a complete paired matrix")
        for cell in sorted(reference_cells):
            component_rows = [arm_maps[arm][cell] for arm in sorted(required)]
            component_values = [
                (arm, arm_maps[arm][cell].score_rate) for arm in sorted(required)
            ]
            component_policies = [
                (arm, arm_maps[arm][cell].candidate_sha256) for arm in sorted(required)
            ]
            value = math.fsum(
                coefficients[arm] * arm_maps[arm][cell].score_rate
                for arm in sorted(required)
            )
            differences.append(
                _difference_from_components(
                    contrast_id=contrast_id,
                    replicate_id=replicate_id,
                    rows=component_rows,
                    components=component_values,
                    component_policies=component_policies,
                    component_coefficients=[
                        (arm, coefficients[arm]) for arm in sorted(required)
                    ],
                    value=value,
                    contrast_kind="method",
                    model_seed=next(iter(paired_model_seeds)),
                )
            )
    return tuple(differences)


def method_differences(
    records: Sequence[Mapping[str, object]],
    *,
    treatment_id: str,
    control_treatment_id: str = "O",
) -> tuple[PairedDifference, ...]:
    return method_contrast_differences(
        records,
        coefficients={treatment_id: 1.0, control_treatment_id: -1.0},
        contrast_id=f"d_method:{treatment_id}-{control_treatment_id}",
    )


def factorial_interaction_differences(
    records: Sequence[Mapping[str, object]],
    *,
    control: str = "O",
    pool_only: str = "A",
    labels_only: str = "B",
    combined: str = "C",
) -> tuple[PairedDifference, ...]:
    """Return the pre-specified ``C - A - B + O`` interaction."""
    return method_contrast_differences(
        records,
        coefficients={combined: 1.0, pool_only: -1.0, labels_only: -1.0, control: 1.0},
        contrast_id=f"interaction:{combined}-{pool_only}-{labels_only}+{control}",
    )


def _weight_for(row: PairedDifference, weighting: Weighting) -> float:
    if weighting == "iid":
        if (
            row.selection_kind not in {"iid", "ci-fixture"}
            or row.inclusion_probability != 1.0
        ):
            raise StatisticsError("IID estimands reject stress/oversampled scenarios")
        return 1.0
    if weighting == "unweighted-diagnostic":
        return 1.0
    if row.selection_kind != "stress-balanced":
        raise StatisticsError("inverse-inclusion weighting requires a stress design")
    return 1.0 / row.inclusion_probability


def _identical_duplicate_value(rows: Sequence[PairedDifference]) -> float:
    """Collapse mechanical copies but reject conflicting duplicate coordinates."""
    if not rows:
        raise StatisticsError("paired-difference coordinate is empty")
    first = rows[0]
    if any(row != first for row in rows[1:]):
        raise StatisticsError(
            "paired-difference coordinate contains conflicting duplicate rows"
        )
    return float(first.value)


def _difference_input_hash(rows: Sequence[PairedDifference]) -> str:
    return sha256_canonical_json(
        [
            row.audit_dict()
            for row in sorted(
                rows,
                key=lambda item: (
                    item.replicate_id if item.replicate_id is not None else -1,
                    item.contrast_id,
                    item.opponent_sha256,
                    item.scenario_id,
                    item.value,
                    item.component_scores,
                    sha256_canonical_json(item.audit_dict()),
                ),
            )
        ]
    )


def _endpoint_label(endpoint: EndpointKey, names: Mapping[str, str]) -> str:
    contrast_id, opponent_sha256 = endpoint
    return f"{contrast_id}|{names[opponent_sha256]}:{opponent_sha256}"


def hypothesis_endpoint_id(
    contrast_id: str, opponent_id: str, opponent_sha256: str
) -> str:
    """Return the exact endpoint spelling required in a hypothesis family."""
    if type(contrast_id) is not str or not contrast_id:
        raise StatisticsError("hypothesis endpoint contrast_id is invalid")
    if type(opponent_id) is not str or not opponent_id:
        raise StatisticsError("hypothesis endpoint opponent_id is invalid")
    if not _is_sha256(opponent_sha256):
        raise StatisticsError("hypothesis endpoint opponent SHA-256 is invalid")
    return f"{contrast_id}|{opponent_id}:{opponent_sha256}"


def _audit_replicate_provenance(
    differences: Sequence[PairedDifference],
) -> None:
    """Reject relabelled seeds or checkpoints before replicate inference."""
    if not differences or any(row.replicate_id is None for row in differences):
        raise StatisticsError("replicate provenance requires replicate differences")
    identities: dict[tuple[int, str], tuple[int, tuple[tuple[str, str], ...]]] = {}
    seed_by_replicate: dict[int, int] = {}
    for row in differences:
        assert row.replicate_id is not None and row.model_seed is not None
        key = (row.replicate_id, row.contrast_id)
        identity = (row.model_seed, row.component_policies)
        previous = identities.setdefault(key, identity)
        if previous != identity:
            raise StatisticsError(
                "one replicate/contrast has conflicting model provenance"
            )
        previous_seed = seed_by_replicate.setdefault(row.replicate_id, row.model_seed)
        if previous_seed != row.model_seed:
            raise StatisticsError("one replicate has conflicting model seeds")
    if len(seed_by_replicate) != len(set(seed_by_replicate.values())):
        raise StatisticsError("model seed is reused across independent replicates")
    replicates = sorted(seed_by_replicate)
    reference_keys = {
        (contrast, component)
        for (replicate, contrast), (_seed, policies) in identities.items()
        if replicate == replicates[0]
        for component, _digest in policies
    }
    for replicate in replicates:
        keys = {
            (contrast, component)
            for (candidate_replicate, contrast), (_seed, policies) in identities.items()
            if candidate_replicate == replicate
            for component, _digest in policies
        }
        if keys != reference_keys:
            raise StatisticsError("replicates have different contrast components")
    for contrast, component in reference_keys:
        hashes = [
            dict(identities[(replicate, contrast)][1])[component]
            for replicate in replicates
        ]
        if len(hashes) != len(set(hashes)):
            raise StatisticsError("checkpoint digest is reused across replicates")


def _confidence_intervals(
    samples: np.ndarray,
    points: np.ndarray,
    endpoints: Sequence[str],
    config: BootstrapConfig,
) -> dict[str, dict[str, float | None]]:
    alpha = 1.0 - config.confidence_level
    family_size = len(endpoints)
    if config.alternative == "two-sided":
        lower_probability = alpha / (2 * family_size)
        upper_probability = 1.0 - lower_probability
    else:
        lower_probability = alpha / family_size
        upper_probability = None
    result: dict[str, dict[str, float | None]] = {}
    for endpoint_index, endpoint in enumerate(endpoints):
        endpoint_samples = samples[:, endpoint_index]
        result[endpoint] = {
            "estimate": float(points[endpoint_index]),
            "lower": float(np.quantile(endpoint_samples, lower_probability)),
            "upper": (
                float(np.quantile(endpoint_samples, upper_probability))
                if upper_probability is not None
                else None
            ),
        }
    return result


def _scenario_matrix(  # noqa: C901 - endpoint rectangle audit
    differences: Sequence[PairedDifference],
    weighting: Weighting,
) -> tuple[list[str], list[EndpointKey], np.ndarray, np.ndarray, dict[str, str]]:
    if not differences:
        raise StatisticsError("bootstrap needs paired differences")
    if any(row.contrast_kind != "deployment" for row in differences):
        raise StatisticsError("scenario bootstrap is reserved for deployment contrasts")
    replicate_ids = {row.replicate_id for row in differences}
    if len(replicate_ids) > 1:
        raise StatisticsError(
            "flat scenario bootstrap cannot combine model replicates; use nested bootstrap"
        )
    if None not in replicate_ids:
        _audit_replicate_provenance(differences)
    endpoint_names: dict[str, str] = {}
    grouped: dict[tuple[str, str, str], list[PairedDifference]] = defaultdict(list)
    for row in differences:
        previous = endpoint_names.setdefault(row.opponent_sha256, row.opponent_id)
        if previous != row.opponent_id:
            raise StatisticsError("one opponent digest has conflicting names")
        grouped[(row.scenario_id, row.contrast_id, row.opponent_sha256)].append(row)
    scenarios = sorted({scenario for scenario, _contrast, _opponent in grouped})
    endpoints = sorted(
        {(contrast, opponent) for _scenario, contrast, opponent in grouped}
    )
    expected = {
        (scenario, contrast, opponent)
        for scenario in scenarios
        for contrast, opponent in endpoints
    }
    if set(grouped) != expected:
        raise StatisticsError("bootstrap endpoints do not share the same scenarios")
    matrix = np.empty((len(scenarios), len(endpoints)), dtype=np.float64)
    weights = np.empty(len(scenarios), dtype=np.float64)
    for scenario_index, scenario in enumerate(scenarios):
        scenario_weights: set[float] = set()
        for endpoint_index, (contrast, opponent) in enumerate(endpoints):
            rows = grouped[(scenario, contrast, opponent)]
            matrix[scenario_index, endpoint_index] = _identical_duplicate_value(rows)
            scenario_weights.update(_weight_for(row, weighting) for row in rows)
        if len(scenario_weights) != 1:
            raise StatisticsError("scenario inclusion weights differ across endpoints")
        weights[scenario_index] = next(iter(scenario_weights))
    return scenarios, endpoints, matrix, weights, endpoint_names


def joint_scenario_cluster_bootstrap(
    differences: Sequence[PairedDifference],
    config: BootstrapConfig,
) -> dict[str, object]:
    """Jointly resample scenario clusters across every opponent endpoint."""
    scenarios, endpoints, matrix, weights, endpoint_names = _scenario_matrix(
        differences, config.weighting
    )
    points = np.average(matrix, axis=0, weights=weights)
    rng = np.random.default_rng(config.seed)
    samples = np.empty((config.resamples, len(endpoints)), dtype=np.float64)
    draw_digest = hashlib.sha256()
    chunk_size = 256
    for start in range(0, config.resamples, chunk_size):
        stop = min(start + chunk_size, config.resamples)
        indexes = rng.integers(
            0,
            len(scenarios),
            size=(stop - start, len(scenarios)),
            dtype=np.int64,
        )
        draw_digest.update(indexes.tobytes())
        selected_weights = weights[indexes]
        selected_values = matrix[indexes]
        samples[start:stop] = (selected_values * selected_weights[:, :, None]).sum(
            axis=1
        ) / selected_weights.sum(axis=1)[:, None]
    endpoint_labels = [
        _endpoint_label(endpoint, endpoint_names) for endpoint in endpoints
    ]
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "protocol": CLUSTER_BOOTSTRAP_PROTOCOL,
        "independent_unit": "scenario",
        "seat_aggregation": "mean-before-resampling",
        "joint_endpoints": endpoint_labels,
        "scenario_count": len(scenarios),
        "scenario_ids_sha256": sha256_canonical_json(scenarios),
        "input_sha256": _difference_input_hash(differences),
        "resample_index_sha256": draw_digest.hexdigest(),
        "config": config.to_dict(),
        "intervals": _confidence_intervals(samples, points, endpoint_labels, config),
        "estimand_label": (
            "IID score-rate difference"
            if config.weighting == "iid"
            else "stress diagnostic (not an unweighted IID claim)"
        ),
    }


def _nested_matrix(  # noqa: C901 - full replicate/scenario rectangle audit
    differences: Sequence[PairedDifference],
    weighting: Weighting,
) -> tuple[
    list[int],
    list[str],
    list[EndpointKey],
    np.ndarray,
    np.ndarray,
    dict[str, str],
]:
    if not differences:
        raise StatisticsError("nested bootstrap needs paired differences")
    if any(row.contrast_kind != "method" for row in differences):
        raise StatisticsError("nested bootstrap is reserved for method contrasts")
    if any(row.replicate_id is None for row in differences):
        raise StatisticsError("nested bootstrap requires model replicate IDs")
    _audit_replicate_provenance(differences)
    replicates = sorted({cast(int, row.replicate_id) for row in differences})
    if len(replicates) < MIN_NESTED_REPLICATES:
        raise StatisticsError(
            f"nested bootstrap requires at least {MIN_NESTED_REPLICATES} replicates"
        )
    endpoints = sorted({(row.contrast_id, row.opponent_sha256) for row in differences})
    endpoint_names: dict[str, str] = {}
    grouped: dict[tuple[int, str, str, str], list[PairedDifference]] = defaultdict(list)
    for row in differences:
        assert row.replicate_id is not None
        previous = endpoint_names.setdefault(row.opponent_sha256, row.opponent_id)
        if previous != row.opponent_id:
            raise StatisticsError("one opponent digest has conflicting names")
        grouped[
            (
                row.replicate_id,
                row.scenario_id,
                row.contrast_id,
                row.opponent_sha256,
            )
        ].append(row)
    scenario_sets = {
        replicate: {
            scenario
            for candidate_replicate, scenario, _contrast, _opponent in grouped
            if candidate_replicate == replicate
        }
        for replicate in replicates
    }
    reference_scenarios = next(iter(scenario_sets.values()))
    if any(scenarios != reference_scenarios for scenarios in scenario_sets.values()):
        raise StatisticsError("replicates must use the same scenario set")
    scenarios = sorted(reference_scenarios)
    expected = {
        (replicate, scenario, contrast, opponent)
        for replicate in replicates
        for scenario in scenarios
        for contrast, opponent in endpoints
    }
    if set(grouped) != expected:
        raise StatisticsError("nested bootstrap matrix is incomplete")
    matrix = np.empty(
        (len(replicates), len(scenarios), len(endpoints)), dtype=np.float64
    )
    weights = np.empty((len(replicates), len(scenarios)), dtype=np.float64)
    for replicate_index, replicate in enumerate(replicates):
        for scenario_index, scenario in enumerate(scenarios):
            cell_weights: set[float] = set()
            for endpoint_index, (contrast, opponent) in enumerate(endpoints):
                rows = grouped[(replicate, scenario, contrast, opponent)]
                matrix[replicate_index, scenario_index, endpoint_index] = (
                    _identical_duplicate_value(rows)
                )
                cell_weights.update(_weight_for(row, weighting) for row in rows)
            if len(cell_weights) != 1:
                raise StatisticsError("nested scenario weights differ across endpoints")
            weights[replicate_index, scenario_index] = next(iter(cell_weights))
    return replicates, scenarios, endpoints, matrix, weights, endpoint_names


def nested_replicate_scenario_bootstrap(
    differences: Sequence[PairedDifference],
    config: BootstrapConfig,
) -> dict[str, object]:
    """Resample model replicates outside and scenarios inside each replicate."""
    replicates, scenarios, endpoints, matrix, weights, endpoint_names = _nested_matrix(
        differences, config.weighting
    )
    replicate_points = np.stack(
        [
            np.average(matrix[index], axis=0, weights=weights[index])
            for index in range(len(replicates))
        ]
    )
    points = replicate_points.mean(axis=0)
    rng = np.random.default_rng(config.seed)
    samples = np.empty((config.resamples, len(endpoints)), dtype=np.float64)
    draw_digest = hashlib.sha256()
    for bootstrap_index in range(config.resamples):
        outer = rng.integers(0, len(replicates), size=len(replicates), dtype=np.int64)
        draw_digest.update(outer.tobytes())
        selected_replicate_values: list[np.ndarray] = []
        for replicate_index in outer:
            inner = rng.integers(0, len(scenarios), size=len(scenarios), dtype=np.int64)
            draw_digest.update(inner.tobytes())
            selected_replicate_values.append(
                np.average(
                    matrix[replicate_index, inner],
                    axis=0,
                    weights=weights[replicate_index, inner],
                )
            )
        samples[bootstrap_index] = np.stack(selected_replicate_values).mean(axis=0)
    endpoint_labels = [
        _endpoint_label(endpoint, endpoint_names) for endpoint in endpoints
    ]
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "protocol": NESTED_BOOTSTRAP_PROTOCOL,
        "independent_unit": "model_replicate",
        "outer_resample": "replicate",
        "inner_resample": "scenario-within-replicate",
        "seat_aggregation": "mean-before-resampling",
        "replicate_count": len(replicates),
        "scenario_count_per_replicate": len(scenarios),
        "replicate_ids": replicates,
        "scenario_ids_sha256": sha256_canonical_json(scenarios),
        "input_sha256": _difference_input_hash(differences),
        "resample_index_sha256": draw_digest.hexdigest(),
        "config": config.to_dict(),
        "intervals": _confidence_intervals(samples, points, endpoint_labels, config),
        "limitations": [
            "Five replicates remain a small outer sample; report every replicate.",
            "The interval describes the declared training-replicate population only.",
        ],
    }


def interquartile_mean(values: Sequence[float]) -> float:
    """Return the exact 25%-trimmed empirical mean with fractional boundaries."""
    array = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise StatisticsError("IQM needs a finite non-empty 1-D sample")
    lower = 0.25 * array.size
    upper = 0.75 * array.size
    weighted_sum = 0.0
    total_weight = 0.0
    for index, value in enumerate(array):
        weight = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        weighted_sum += weight * float(value)
        total_weight += weight
    return weighted_sum / total_weight


def summarize_method_differences(  # noqa: C901 - full summary audit
    differences: Sequence[PairedDifference],
    *,
    profile_thresholds: Sequence[float] = (-0.05, 0.0, 0.03, 0.05),
) -> dict[str, object]:
    """Report each replicate plus stratified IQM/profile/probability improvement."""
    if not differences or any(row.replicate_id is None for row in differences):
        raise StatisticsError("method summary requires replicate-level differences")
    if any(row.contrast_kind != "method" for row in differences):
        raise StatisticsError("method summary requires method contrasts")
    _audit_replicate_provenance(differences)
    contrast_ids = {row.contrast_id for row in differences}
    if len(contrast_ids) != 1:
        raise StatisticsError("method summary requires exactly one contrast")
    thresholds = tuple(float(value) for value in profile_thresholds)
    if not thresholds or any(not math.isfinite(value) for value in thresholds):
        raise StatisticsError("performance-profile thresholds must be finite")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise StatisticsError("performance-profile thresholds must be sorted/unique")
    by_coordinate: dict[tuple[int, str, str], list[PairedDifference]] = defaultdict(
        list
    )
    scenario_sets: dict[tuple[int, str], set[str]] = defaultdict(set)
    endpoint_names: dict[str, str] = {}
    for row in differences:
        assert row.replicate_id is not None
        previous_name = endpoint_names.setdefault(row.opponent_sha256, row.opponent_id)
        if previous_name != row.opponent_id:
            raise StatisticsError("one opponent digest has conflicting names")
        by_coordinate[(row.replicate_id, row.opponent_sha256, row.scenario_id)].append(
            row
        )
        scenario_sets[(row.replicate_id, row.opponent_sha256)].add(row.scenario_id)
    replicates = sorted({replicate for replicate, _endpoint in scenario_sets})
    endpoints = sorted(endpoint_names)
    expected = {
        (replicate, endpoint) for replicate in replicates for endpoint in endpoints
    }
    if set(scenario_sets) != expected:
        raise StatisticsError("method summary replicate/opponent matrix is incomplete")
    reference_scenarios = next(iter(scenario_sets.values()))
    if any(scenarios != reference_scenarios for scenarios in scenario_sets.values()):
        raise StatisticsError("method summary cells use different scenario sets")
    cells = {
        (replicate, endpoint): float(
            np.mean(
                [
                    _identical_duplicate_value(
                        by_coordinate[(replicate, endpoint, scenario)]
                    )
                    for scenario in sorted(reference_scenarios)
                ]
            )
        )
        for replicate, endpoint in sorted(expected)
    }
    per_replicate = []
    replicate_means: list[float] = []
    for replicate in replicates:
        endpoint_values = {
            f"{endpoint_names[endpoint]}:{endpoint[:12]}": cells[(replicate, endpoint)]
            for endpoint in endpoints
        }
        mean = float(np.mean(list(endpoint_values.values())))
        replicate_means.append(mean)
        per_replicate.append(
            {
                "replicate_id": replicate,
                "mean_difference": mean,
                "opponents": endpoint_values,
            }
        )
    stratified_iqm = float(
        np.mean(
            [
                interquartile_mean(
                    [cells[(replicate, endpoint)] for replicate in replicates]
                )
                for endpoint in endpoints
            ]
        )
    )
    cell_values = np.asarray(list(cells.values()), dtype=np.float64)
    probability_improvement = float(
        np.mean((cell_values > 0).astype(np.float64) + 0.5 * (cell_values == 0))
    )
    return {
        "replicate_count": len(replicates),
        "per_replicate": per_replicate,
        "mean_difference": float(np.mean(replicate_means)),
        "between_replicate_sample_sd": (
            float(np.std(replicate_means, ddof=1))
            if len(replicate_means) >= MIN_POWER_N
            else None
        ),
        "worst_replicate": float(min(replicate_means)),
        "opponent_stratified_iqm": stratified_iqm,
        "probability_of_improvement": probability_improvement,
        "performance_profile": {
            str(threshold): float(np.mean(cell_values >= threshold))
            for threshold in thresholds
        },
        "profile_unit": "replicate-by-opponent scenario mean",
    }


def replicate_effects(differences: Sequence[PairedDifference]) -> tuple[float, ...]:
    """Collapse all scenarios/opponents to one equally weighted value per model run."""
    if len({row.contrast_id for row in differences}) != 1:
        raise StatisticsError("replicate power requires exactly one contrast")
    if any(row.contrast_kind != "method" for row in differences):
        raise StatisticsError("replicate effects require a method contrast")
    if len({row.opponent_sha256 for row in differences}) != 1:
        raise StatisticsError("replicate effects require exactly one endpoint")
    _audit_replicate_provenance(differences)
    by_cell: dict[tuple[int, str, str], list[PairedDifference]] = defaultdict(list)
    cells_by_replicate: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for row in differences:
        if row.replicate_id is None:
            raise StatisticsError("replicate effects require replicate IDs")
        _weight_for(row, "iid")
        by_cell[(row.replicate_id, row.scenario_id, row.opponent_sha256)].append(row)
        cells_by_replicate[row.replicate_id].add((row.scenario_id, row.opponent_sha256))
    if not by_cell:
        raise StatisticsError("replicate effects need paired differences")
    reference_cells = next(iter(cells_by_replicate.values()))
    if any(cells != reference_cells for cells in cells_by_replicate.values()):
        raise StatisticsError("replicate power inputs do not share a complete matrix")
    return tuple(
        math.fsum(
            _identical_duplicate_value(by_cell[(replicate, scenario, opponent)])
            for scenario, opponent in sorted(reference_cells)
        )
        / len(reference_cells)
        for replicate in sorted(cells_by_replicate)
    )


def scenario_effects(differences: Sequence[PairedDifference]) -> tuple[float, ...]:
    """Return one value per deployment scenario for one opponent endpoint."""
    if len({row.contrast_id for row in differences}) != 1:
        raise StatisticsError("scenario power requires exactly one contrast")
    if any(row.contrast_kind != "deployment" for row in differences):
        raise StatisticsError("scenario effects require a deployment contrast")
    if len({row.opponent_sha256 for row in differences}) != 1:
        raise StatisticsError("scenario effects require exactly one endpoint")
    if any(row.replicate_id is not None for row in differences):
        raise StatisticsError("deployment scenario effects cannot contain replicates")
    by_cell: dict[tuple[str, str], list[PairedDifference]] = defaultdict(list)
    endpoints_by_scenario: dict[str, set[str]] = defaultdict(set)
    for row in differences:
        _weight_for(row, "iid")
        by_cell[(row.scenario_id, row.opponent_sha256)].append(row)
        endpoints_by_scenario[row.scenario_id].add(row.opponent_sha256)
    if not by_cell:
        raise StatisticsError("scenario effects need paired differences")
    reference_endpoints = next(iter(endpoints_by_scenario.values()))
    if any(
        endpoints != reference_endpoints for endpoints in endpoints_by_scenario.values()
    ):
        raise StatisticsError("scenario power inputs do not share all endpoints")
    return tuple(
        math.fsum(
            _identical_duplicate_value(by_cell[(scenario, opponent)])
            for opponent in sorted(reference_endpoints)
        )
        / len(reference_endpoints)
        for scenario in sorted(endpoints_by_scenario)
    )


def _round_up(value: int, block: int) -> int:
    return max(block, ((value + block - 1) // block) * block)


def paired_power_analysis(  # noqa: C901,PLR0915 - simulation design is explicit
    pilot_differences: Sequence[float],
    config: PowerAnalysisConfig,
) -> dict[str, object]:
    """Choose a fixed N from paired-unit variance, then validate by simulation."""
    values = np.sort(np.asarray(tuple(pilot_differences), dtype=np.float64))
    if values.ndim != 1 or values.size < MIN_POWER_N or not np.isfinite(values).all():
        raise StatisticsError("power pilot needs at least two finite paired units")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise StatisticsError(
            "power pilot values must be score-rate differences in [-1, 1]"
        )
    if np.all(values == values[0]):
        raise StatisticsError("power analysis cannot infer N from zero paired variance")
    sample_sd = float(np.std(values, ddof=1))
    if sample_sd <= 0.0:
        raise StatisticsError("power analysis cannot infer N from zero paired variance")
    adjusted_alpha = config.familywise_alpha / config.family_size
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - adjusted_alpha)
    target_quantile = normal.inv_cdf(config.target_power)
    raw_initial = math.ceil(
        ((critical + target_quantile) * sample_sd / config.minimum_effect) ** 2
    )
    normal_initial_n = max(MIN_POWER_N, raw_initial)
    rounded_initial = _round_up(normal_initial_n, config.round_to)
    centered = values - float(np.mean(values))
    rng = np.random.default_rng(config.seed)
    raw_design_floor = (
        MIN_FINAL_IID_SCENARIOS if config.unit == "scenario" else MIN_NESTED_REPLICATES
    )
    design_floor = _round_up(raw_design_floor, config.round_to)
    candidate_start = min(
        _round_up(max(design_floor, rounded_initial), config.round_to),
        config.max_n,
    )
    candidate_ns = list(range(candidate_start, config.max_n + 1, config.round_to))
    if config.max_n not in candidate_ns:
        candidate_ns.append(config.max_n)
    achieved_by_n: dict[int, float] = {}
    recommended_n: int | None = None
    simulation_draw_digest = hashlib.sha256()
    for candidate_n in candidate_ns:
        samples = (
            rng.choice(
                centered,
                size=(config.simulations, candidate_n),
                replace=True,
            )
            + config.minimum_effect
        )
        simulation_draw_digest.update(samples.tobytes())
        means = samples.mean(axis=1)
        standard_deviations = samples.std(axis=1, ddof=1)
        standard_errors = standard_deviations / math.sqrt(candidate_n)
        statistics = np.full(config.simulations, -np.inf, dtype=np.float64)
        nonzero = standard_errors > 0
        statistics[nonzero] = means[nonzero] / standard_errors[nonzero]
        statistics[~nonzero & (means > 0)] = np.inf
        achieved = float(np.mean(statistics > critical))
        achieved_by_n[candidate_n] = achieved
        if achieved >= config.target_power:
            recommended_n = candidate_n
            break
    if recommended_n is None:
        recommended_n = config.max_n
    achieved_power = achieved_by_n.get(recommended_n)
    if achieved_power is None:
        raise StatisticsError("power simulation did not evaluate recommended N")
    underpowered = achieved_power < config.target_power
    limitations = [
        "Normal approximation is an initial value; N is accepted only after empirical-residual simulation.",
        "N is fixed before the corresponding validation/test outcomes are observed.",
    ]
    if config.unit == "model_replicate" and values.size < MIN_NESTED_REPLICATES:
        limitations.append(
            "Pilot has fewer than five model replicates; variance and power are highly uncertain."
        )
    if underpowered:
        limitations.append(
            "Target power was not reached at max_n; the resulting design is underpowered."
        )
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "protocol": POWER_PROTOCOL,
        "design": "fixed-N",
        "sequential_stopping": False,
        "independent_unit": config.unit,
        "pilot_unit_count": int(values.size),
        "pilot_mean_difference": float(np.mean(values)),
        "paired_sample_sd": sample_sd,
        "pilot_values_sha256": sha256_canonical_json(values.tolist()),
        "config": config.to_dict(),
        "adjusted_one_sided_alpha": adjusted_alpha,
        "normal_initial_n": normal_initial_n,
        "normal_initial_n_rounded_up": rounded_initial,
        "design_minimum_n": design_floor,
        "recommended_n": recommended_n,
        "simulated_power_at_recommended_n": achieved_power,
        "simulated_power_by_n": {
            str(sample_size): power for sample_size, power in achieved_by_n.items()
        },
        "simulation_draw_sha256": simulation_draw_digest.hexdigest(),
        "underpowered": underpowered,
        "resampling_model": "centered paired empirical residual + minimum_effect",
        "limitations": limitations,
    }


def deployment_scenario_power_analysis(
    differences: Sequence[PairedDifference],
    config: PowerAnalysisConfig,
) -> dict[str, object]:
    """Power every deployment endpoint using scenarios—not seats—as units."""
    if config.unit != "scenario":
        raise StatisticsError("deployment power requires unit='scenario'")
    return _family_power_analysis(differences, config)


def method_replicate_power_analysis(
    differences: Sequence[PairedDifference],
    config: PowerAnalysisConfig,
) -> dict[str, object]:
    """Power every method endpoint using independent model replicates."""
    if config.unit != "model_replicate":
        raise StatisticsError("method power requires unit='model_replicate'")
    return _family_power_analysis(differences, config)


def _family_power_analysis(
    differences: Sequence[PairedDifference],
    config: PowerAnalysisConfig,
) -> dict[str, object]:
    if not differences or len({row.contrast_id for row in differences}) != 1:
        raise StatisticsError("family power requires exactly one non-empty contrast")
    endpoint_names: dict[str, str] = {}
    by_endpoint: dict[str, list[PairedDifference]] = defaultdict(list)
    for row in differences:
        previous = endpoint_names.setdefault(row.opponent_sha256, row.opponent_id)
        if previous != row.opponent_id:
            raise StatisticsError("one power endpoint has conflicting names")
        by_endpoint[row.opponent_sha256].append(row)
    if config.family_size < len(by_endpoint):
        raise StatisticsError(
            "power family_size cannot be smaller than the evaluated endpoint count"
        )
    coordinate_sets = {
        endpoint: {(row.replicate_id, row.scenario_id) for row in rows}
        for endpoint, rows in by_endpoint.items()
    }
    reference_coordinates = next(iter(coordinate_sets.values()))
    if any(
        coordinates != reference_coordinates for coordinates in coordinate_sets.values()
    ):
        raise StatisticsError("power endpoints do not share the same paired units")
    endpoint_results: dict[str, dict[str, object]] = {}
    for opponent_sha256 in sorted(by_endpoint):
        endpoint_rows = by_endpoint[opponent_sha256]
        effects = (
            scenario_effects(endpoint_rows)
            if config.unit == "scenario"
            else replicate_effects(endpoint_rows)
        )
        endpoint_results[
            hypothesis_endpoint_id(
                differences[0].contrast_id,
                endpoint_names[opponent_sha256],
                opponent_sha256,
            )
        ] = paired_power_analysis(effects, config)
    recommended_n = max(
        cast(int, result["recommended_n"]) for result in endpoint_results.values()
    )
    underpowered = any(
        result["underpowered"] is True for result in endpoint_results.values()
    )
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "protocol": POWER_PROTOCOL,
        "design": "fixed-N",
        "sequential_stopping": False,
        "independent_unit": config.unit,
        "contrast_id": differences[0].contrast_id,
        "endpoint_count": len(endpoint_results),
        "family_size": config.family_size,
        "config": config.to_dict(),
        "input_sha256": _difference_input_hash(differences),
        "endpoint_results": endpoint_results,
        "recommended_n": recommended_n,
        "underpowered": underpowered,
        "selection_rule": "maximum endpoint-specific N in the pre-registered family",
        "limitations": [
            "Endpoint-specific paired variance is preserved; endpoint averaging cannot reduce N.",
            "N remains fixed before validation/test outcomes are observed.",
        ],
    }


def _differences_for_contrast(
    records: Sequence[Mapping[str, object]], contrast: AnalysisContrast
) -> tuple[PairedDifference, ...]:
    coefficients = dict(contrast.components)
    if contrast.kind == "deployment":
        return checkpoint_contrast_differences(
            records,
            coefficients=coefficients,
            contrast_id=contrast.contrast_id,
        )
    return method_contrast_differences(
        records,
        coefficients=coefficients,
        contrast_id=contrast.contrast_id,
    )


def _bootstrap_config_for_rows(
    rows: Sequence[PairedDifference],
    base: BootstrapConfig,
    protocol: StatisticalProtocolSpec,
) -> BootstrapConfig:
    selection_kinds = {row.selection_kind for row in rows}
    if selection_kinds <= {"iid", "ci-fixture"}:
        return base
    if selection_kinds == {"stress-balanced"}:
        return replace(base, weighting=protocol.stress_weighting)
    raise StatisticsError("one analysis family mixes IID and stress scenarios")


def _validate_fixed_design_binding(
    evaluation_binding: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    protocol: StatisticalProtocolSpec,
) -> None:
    """Ensure an approved final schedule already has the pilot-selected N."""
    source = protocol.power_source
    if source.data_role != "approved-pilot-fixed-n":
        return
    if evaluation_binding.get("scenario_count") != source.fixed_scenario_n:
        raise StatisticsError(
            "paired evaluation scenario count differs from pre-registered fixed N"
        )
    method_contrast = next(
        contrast
        for contrast in protocol.analysis_contrasts
        if contrast.power_unit == "model_replicate"
    )
    replicate_maps: list[dict[int, int]] = []
    for treatment_id, _coefficient in method_contrast.components:
        policies = [
            policy
            for policy in candidates
            if policy.get("treatment_id") == treatment_id
        ]
        replicate_map: dict[int, int] = {}
        for policy in policies:
            replicate = policy.get("replicate_id")
            model_seed = policy.get("model_seed")
            if type(replicate) is not int or type(model_seed) is not int:
                raise StatisticsError(
                    "fixed method design has incomplete replicate provenance"
                )
            if replicate in replicate_map:
                raise StatisticsError(
                    "fixed method design repeats a treatment replicate"
                )
            replicate_map[replicate] = model_seed
        replicate_maps.append(replicate_map)
    if not replicate_maps or any(
        replicate_map != replicate_maps[0] for replicate_map in replicate_maps[1:]
    ):
        raise StatisticsError(
            "fixed method arms do not share replicate IDs and model seeds"
        )
    if len(replicate_maps[0]) != source.fixed_model_replicates:
        raise StatisticsError(
            "paired evaluation replicate count differs from pre-registered fixed N"
        )


def _validate_power_contrast_orientation(
    contrast: AnalysisContrast,
    candidate_roles: Mapping[str, str],
    treatment_roles: Mapping[str, set[str]],
) -> None:
    """Require a preregistered candidate-minus-control direction for power."""
    if len(contrast.components) != MIN_POWER_N or sorted(
        coefficient for _identity, coefficient in contrast.components
    ) != [-1.0, 1.0]:
        raise StatisticsError(
            "power-selected contrast must be an oriented +1/-1 difference"
        )
    coefficients = dict(contrast.components)
    positive = next(
        identity for identity, value in coefficients.items() if value == 1.0
    )
    negative = next(
        identity for identity, value in coefficients.items() if value == -1.0
    )
    if contrast.kind == "deployment":
        if candidate_roles[positive] not in {"official", "treatment"} or (
            candidate_roles[negative] not in {"frozen-baseline", "control"}
        ):
            raise StatisticsError(
                "deployment power contrast must orient official/treatment "
                "minus frozen-baseline/control"
            )
        return
    if treatment_roles[positive] != {"treatment"} or treatment_roles[negative] != {
        "control"
    }:
        raise StatisticsError(
            "method power contrast must orient treatment minus control"
        )


def _candidate_treatment_roles(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = defaultdict(set)
    for policy in candidates:
        treatment_id = policy["treatment_id"]
        if treatment_id is not None:
            roles[str(treatment_id)].add(str(policy["role"]))
    return roles


def validate_analysis_plan_binding(
    evaluation_binding: Mapping[str, object],
    protocol: StatisticalProtocolSpec,
) -> None:
    """Cross-check contrasts/endpoints against a paired-evaluation declaration."""
    raw_candidates = evaluation_binding.get("candidates")
    raw_opponents = evaluation_binding.get("opponents")
    if not isinstance(raw_candidates, list) or not isinstance(raw_opponents, list):
        raise StatisticsError("paired evaluation policy bindings are missing")
    candidates = [
        cast(Mapping[str, object], row)
        for row in raw_candidates
        if isinstance(row, Mapping)
    ]
    opponents = [
        cast(Mapping[str, object], row)
        for row in raw_opponents
        if isinstance(row, Mapping)
    ]
    if len(candidates) != len(raw_candidates) or len(opponents) != len(raw_opponents):
        raise StatisticsError("paired evaluation policy bindings are invalid")
    _validate_fixed_design_binding(evaluation_binding, candidates, protocol)
    candidate_hashes = {str(policy["policy_sha256"]) for policy in candidates}
    treatment_ids = {
        str(policy["treatment_id"])
        for policy in candidates
        if policy["treatment_id"] is not None
    }
    candidate_roles = {
        str(policy["policy_sha256"]): str(policy["role"]) for policy in candidates
    }
    treatment_roles = _candidate_treatment_roles(candidates)
    opponents.sort(key=lambda policy: str(policy["policy_sha256"]))
    contrasts_by_family: dict[str, list[AnalysisContrast]] = defaultdict(list)
    for contrast in protocol.analysis_contrasts:
        identities = {identity for identity, _coefficient in contrast.components}
        available = candidate_hashes if contrast.kind == "deployment" else treatment_ids
        if not identities <= available:
            raise StatisticsError(
                f"analysis contrast {contrast.contrast_id!r} references absent policies"
            )
        if contrast.power_unit is not None:
            _validate_power_contrast_orientation(
                contrast, candidate_roles, treatment_roles
            )
        contrasts_by_family[contrast.family_id].append(contrast)
    for family in protocol.hypothesis_families:
        contrasts = contrasts_by_family[family.family_id]
        if len({contrast.kind for contrast in contrasts}) != 1:
            raise StatisticsError(
                "one hypothesis family cannot mix deployment and method units"
            )
        expected_endpoints = tuple(
            sorted(
                hypothesis_endpoint_id(
                    contrast.contrast_id,
                    str(opponent["policy_id"]),
                    str(opponent["policy_sha256"]),
                )
                for contrast in contrasts
                for opponent in opponents
            )
        )
        if family.endpoints != expected_endpoints:
            raise StatisticsError(
                f"hypothesis family {family.family_id!r} endpoints do not match "
                "the evaluation identities"
            )


def _power_contrasts_by_unit(
    protocol: StatisticalProtocolSpec,
) -> dict[PowerUnit, AnalysisContrast]:
    return {
        cast(PowerUnit, contrast.power_unit): contrast
        for contrast in protocol.analysis_contrasts
        if contrast.power_unit is not None
    }


def pilot_power_design_sha256(pilot_document: Mapping[str, object]) -> str:
    """Validate and hash the pilot's exact protocol and evaluation matrix."""
    manifest_sha256 = pilot_document.get("manifest_declaration_sha256")
    manifest_declaration = pilot_document.get("manifest_declaration")
    evaluation_binding = pilot_document.get("evaluation_manifest_binding")
    statistical_binding = pilot_document.get("statistical_protocol")
    if not _is_sha256(manifest_sha256):
        raise StatisticsError("pilot manifest declaration SHA-256 is invalid")
    if (
        not isinstance(manifest_declaration, Mapping)
        or sha256_canonical_json(manifest_declaration) != manifest_sha256
    ):
        raise StatisticsError("pilot embedded manifest declaration SHA-256 mismatch")
    if not isinstance(evaluation_binding, Mapping):
        raise StatisticsError("pilot evaluation manifest binding is missing")
    if not isinstance(statistical_binding, Mapping):
        raise StatisticsError("pilot statistical protocol is missing")
    if (
        manifest_declaration.get("paired_evaluation") != evaluation_binding
        or manifest_declaration.get("statistical_protocol") != statistical_binding
    ):
        raise StatisticsError(
            "pilot embedded manifest does not bind its evaluation/statistics design"
        )
    try:
        validate_paired_evaluation_binding(evaluation_binding)
    except PairedEvaluationError as exc:
        raise StatisticsError("pilot evaluation manifest binding is invalid") from exc
    parsed_protocol = validate_statistical_protocol_binding(statistical_binding)
    if parsed_protocol.power_source.data_role != "current-batch-pilot":
        raise StatisticsError("power-design identity requires a pilot-only protocol")
    validate_analysis_plan_binding(evaluation_binding, parsed_protocol)
    return sha256_canonical_json(
        {
            "protocol": "task1-pilot-power-design-v1",
            "manifest_declaration_sha256": manifest_sha256,
            "manifest_declaration": dict(manifest_declaration),
            "evaluation_manifest_binding": dict(evaluation_binding),
            "statistical_protocol": parsed_protocol.manifest_binding(),
            "power_selected_contrasts": [
                contrast.to_dict()
                for contrast in sorted(
                    _power_contrasts_by_unit(parsed_protocol).values(),
                    key=lambda item: cast(str, item.power_unit),
                )
            ],
        }
    )


def _validate_pilot_power_table(
    unit: PowerUnit,
    result: Mapping[str, object],
    pilot_protocol: StatisticalProtocolSpec,
    evaluation_binding: Mapping[str, object],
) -> tuple[str, ...]:
    """Cross-check one stored power table against the pilot declaration."""
    required = {
        "schema_version",
        "protocol",
        "design",
        "sequential_stopping",
        "independent_unit",
        "contrast_id",
        "endpoint_count",
        "family_size",
        "config",
        "input_sha256",
        "endpoint_results",
        "recommended_n",
        "underpowered",
        "selection_rule",
        "limitations",
    }
    if set(result) != required:
        raise StatisticsError(f"pilot {unit} power table schema is invalid")
    contrast = _power_contrasts_by_unit(pilot_protocol)[unit]
    config = (
        pilot_protocol.scenario_power
        if unit == "scenario"
        else pilot_protocol.replicate_power
    )
    raw_opponents = evaluation_binding.get("opponents")
    if not isinstance(raw_opponents, list):  # validated above; retain local typing gate
        raise StatisticsError("pilot evaluation opponents are missing")
    expected_endpoints = tuple(
        sorted(
            hypothesis_endpoint_id(
                contrast.contrast_id,
                str(opponent["policy_id"]),
                str(opponent["policy_sha256"]),
            )
            for opponent in raw_opponents
            if isinstance(opponent, Mapping)
        )
    )
    endpoint_results = result.get("endpoint_results")
    if (
        result.get("schema_version") != STATISTICS_SCHEMA_VERSION
        or result.get("protocol") != POWER_PROTOCOL
        or result.get("design") != "fixed-N"
        or result.get("sequential_stopping") is not False
        or result.get("independent_unit") != unit
        or result.get("contrast_id") != contrast.contrast_id
        or result.get("config") != config.to_dict()
        or result.get("family_size") != config.family_size
        or result.get("endpoint_count") != len(expected_endpoints)
        or not _is_sha256(result.get("input_sha256"))
        or not isinstance(endpoint_results, Mapping)
        or tuple(sorted(endpoint_results)) != expected_endpoints
        or type(result.get("recommended_n")) is not int
        or type(result.get("underpowered")) is not bool
        or not isinstance(result.get("limitations"), list)
    ):
        raise StatisticsError(f"pilot {unit} power table disagrees with its protocol")
    endpoint_recommendations: list[int] = []
    endpoint_underpowered: list[bool] = []
    endpoint_required = {
        "schema_version",
        "protocol",
        "design",
        "sequential_stopping",
        "independent_unit",
        "pilot_unit_count",
        "pilot_mean_difference",
        "paired_sample_sd",
        "pilot_values_sha256",
        "config",
        "adjusted_one_sided_alpha",
        "normal_initial_n",
        "normal_initial_n_rounded_up",
        "design_minimum_n",
        "recommended_n",
        "simulated_power_at_recommended_n",
        "simulated_power_by_n",
        "simulation_draw_sha256",
        "underpowered",
        "resampling_model",
        "limitations",
    }
    for endpoint in expected_endpoints:
        endpoint_result = endpoint_results[endpoint]
        if not isinstance(endpoint_result, Mapping):
            raise StatisticsError(f"pilot {unit} endpoint power result is invalid")
        endpoint_n = endpoint_result.get("recommended_n")
        endpoint_flag = endpoint_result.get("underpowered")
        if (
            set(endpoint_result) != endpoint_required
            or endpoint_result.get("schema_version") != STATISTICS_SCHEMA_VERSION
            or endpoint_result.get("protocol") != POWER_PROTOCOL
            or endpoint_result.get("design") != "fixed-N"
            or endpoint_result.get("sequential_stopping") is not False
            or endpoint_result.get("independent_unit") != unit
            or endpoint_result.get("config") != config.to_dict()
            or type(endpoint_n) is not int
            or type(endpoint_flag) is not bool
            or not _is_sha256(endpoint_result.get("pilot_values_sha256"))
            or not _is_sha256(endpoint_result.get("simulation_draw_sha256"))
        ):
            raise StatisticsError(f"pilot {unit} endpoint power result is invalid")
        endpoint_recommendations.append(endpoint_n)
        endpoint_underpowered.append(endpoint_flag)
    if result["recommended_n"] != max(endpoint_recommendations) or result[
        "underpowered"
    ] != any(endpoint_underpowered):
        raise StatisticsError(f"pilot {unit} family selection rule was not preserved")
    return expected_endpoints


def _validate_pilot_analysis_section(
    pilot_document: Mapping[str, object],
    pilot_protocol: StatisticalProtocolSpec,
) -> None:
    analyses = pilot_document.get("analyses")
    if not isinstance(analyses, Mapping):
        raise StatisticsError("pilot analysis section is missing")
    family_by_id = {
        family.family_id: family for family in pilot_protocol.hypothesis_families
    }
    contrasts_by_family: dict[str, list[AnalysisContrast]] = defaultdict(list)
    for contrast in pilot_protocol.analysis_contrasts:
        contrasts_by_family[contrast.family_id].append(contrast)
    if set(analyses) != set(family_by_id):
        raise StatisticsError("pilot analysis families disagree with its protocol")
    required = {
        "hypothesis",
        "contrast_kind",
        "contrast_ids",
        "difference_count",
        "difference_input_sha256",
        "bootstrap",
        "method_summaries",
    }
    for family_id, family in family_by_id.items():
        raw = analyses[family_id]
        contrasts = contrasts_by_family[family_id]
        if (
            not isinstance(raw, Mapping)
            or set(raw) != required
            or raw.get("hypothesis") != family.to_dict()
            or raw.get("contrast_kind") != contrasts[0].kind
            or raw.get("contrast_ids")
            != [contrast.contrast_id for contrast in contrasts]
            or type(raw.get("difference_count")) is not int
            or cast(int, raw.get("difference_count")) <= 0
            or not _is_sha256(raw.get("difference_input_sha256"))
            or not isinstance(raw.get("bootstrap"), Mapping)
            or not isinstance(raw.get("method_summaries"), Mapping)
        ):
            raise StatisticsError(
                f"pilot analysis family {family_id!r} disagrees with its protocol"
            )


def _contrast_transfer_signature(contrast: AnalysisContrast) -> dict[str, object]:
    return {
        "kind": contrast.kind,
        "power_unit": contrast.power_unit,
        "coefficient_scale": sorted(
            coefficient for _identity, coefficient in contrast.components
        ),
    }


def validate_pilot_power_source(  # noqa: C901,PLR0912,PLR0915 - artifact gate
    source: PowerSourceSpec,
    pilot_document: Mapping[str, object] | None,
    protocol: StatisticalProtocolSpec,
) -> dict[str, object]:
    """Verify the exact pilot artifact selected before a fixed-N evaluation."""
    if source.data_role != "approved-pilot-fixed-n":
        raise StatisticsError("only a fixed-N source can consume a pilot artifact")
    if pilot_document is None:
        raise StatisticsError("fixed-N statistics require the bound pilot document")
    document = dict(pilot_document)
    content_sha256 = document.pop("content_sha256", None)
    if (
        content_sha256 != source.pilot_statistics_content_sha256
        or content_sha256 != sha256_canonical_json(document)
    ):
        raise StatisticsError("pilot statistics content SHA-256 mismatch")
    if (
        pilot_document.get("schema_version") != STATISTICS_SCHEMA_VERSION
        or pilot_document.get("manifest_declaration_sha256")
        != source.pilot_manifest_declaration_sha256
    ):
        raise StatisticsError("pilot statistics provenance mismatch")
    design_sha256 = pilot_power_design_sha256(pilot_document)
    if design_sha256 != source.pilot_power_design_sha256:
        raise StatisticsError("pilot power-design SHA-256 mismatch")
    evaluation = pilot_document.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("status") != "valid":
        raise StatisticsError("fixed-N source is not a valid pilot batch")
    pilot_protocol = pilot_document.get("statistical_protocol")
    if not isinstance(pilot_protocol, Mapping):
        raise StatisticsError("pilot statistical protocol is missing")
    parsed_pilot_protocol = validate_statistical_protocol_binding(pilot_protocol)
    pilot_evaluation_binding = pilot_document.get("evaluation_manifest_binding")
    if not isinstance(pilot_evaluation_binding, Mapping):
        raise StatisticsError("pilot evaluation manifest binding is missing")
    if (
        evaluation.get("protocol") != pilot_evaluation_binding.get("protocol")
        or evaluation.get("batch_id") != pilot_evaluation_binding.get("batch_id")
        or evaluation.get("schedule_sha256")
        != pilot_evaluation_binding.get("schedule_sha256")
        or evaluation.get("scheduled_games")
        != pilot_evaluation_binding.get("scheduled_games")
        or evaluation.get("completed_games") != evaluation.get("scheduled_games")
        or evaluation.get("failed_games") != 0
        or evaluation.get("missing_games") != 0
    ):
        raise StatisticsError("pilot evaluation audit disagrees with its matrix")
    _validate_pilot_analysis_section(pilot_document, parsed_pilot_protocol)
    pilot_power_declaration = pilot_protocol.get("power")
    if not isinstance(pilot_power_declaration, Mapping):
        raise StatisticsError("pilot power declaration is missing")
    pilot_source = pilot_power_declaration.get("source")
    if (
        not isinstance(pilot_source, Mapping)
        or pilot_source.get("data_role") != "current-batch-pilot"
    ):
        raise StatisticsError("fixed-N source must point to a pilot-only artifact")
    power = pilot_document.get("power")
    if (
        not isinstance(power, Mapping)
        or power.get("data_role") != "current-batch-pilot"
        or power.get("source_manifest_declaration_sha256")
        != source.pilot_manifest_declaration_sha256
        or power.get("prospective_use_only") is not True
    ):
        raise StatisticsError("pilot document did not label outcome reuse as pilot")
    scenario = power.get("scenario")
    replicate = power.get("model_replicate")
    if not isinstance(scenario, Mapping) or not isinstance(replicate, Mapping):
        raise StatisticsError("pilot document power tables are missing")
    pilot_endpoints_by_unit = {
        unit: _validate_pilot_power_table(
            unit,
            cast(Mapping[str, object], result),
            parsed_pilot_protocol,
            pilot_evaluation_binding,
        )
        for unit, result in (
            (cast(PowerUnit, "scenario"), scenario),
            (cast(PowerUnit, "model_replicate"), replicate),
        )
    }
    if (
        scenario.get("config") != protocol.scenario_power.to_dict()
        or replicate.get("config") != protocol.replicate_power.to_dict()
    ):
        raise StatisticsError("final power configuration differs from its pilot")
    family_by_id = {family.family_id: family for family in protocol.hypothesis_families}
    contrast_by_unit = _power_contrasts_by_unit(protocol)
    pilot_contrast_by_unit = _power_contrasts_by_unit(parsed_pilot_protocol)
    pilot_family_by_id = {
        family.family_id: family for family in parsed_pilot_protocol.hypothesis_families
    }

    def endpoint_opponents(endpoints: Sequence[str]) -> set[str]:
        opponents = {
            endpoint.split("|", 1)[-1] if "|" in endpoint else ""
            for endpoint in endpoints
        }
        hashes = {opponent.rsplit(":", 1)[-1] for opponent in opponents}
        if "" in opponents or any(not _is_sha256(digest) for digest in hashes):
            raise StatisticsError("power endpoint has an invalid opponent identity")
        return opponents

    for unit in ("scenario", "model_replicate"):
        contrast = contrast_by_unit[cast(PowerUnit, unit)]
        pilot_contrast = pilot_contrast_by_unit[cast(PowerUnit, unit)]
        if _contrast_transfer_signature(contrast) != _contrast_transfer_signature(
            pilot_contrast
        ):
            raise StatisticsError(
                f"final {unit} contrast scale differs from the bound pilot"
            )
        final_family = family_by_id[contrast.family_id]
        pilot_family = pilot_family_by_id[pilot_contrast.family_id]
        if (
            final_family.kind,
            final_family.alternative,
            final_family.minimum_effect,
            final_family.noninferiority_margin,
        ) != (
            pilot_family.kind,
            pilot_family.alternative,
            pilot_family.minimum_effect,
            pilot_family.noninferiority_margin,
        ):
            raise StatisticsError(
                f"final {unit} decision family differs from the bound pilot"
            )
        final_endpoints = family_by_id[contrast.family_id].endpoints
        pilot_endpoints = pilot_endpoints_by_unit[cast(PowerUnit, unit)]
        if endpoint_opponents(final_endpoints) != endpoint_opponents(pilot_endpoints):
            raise StatisticsError(
                f"final {unit} opponent endpoints differ from the bound pilot"
            )
    if (
        scenario.get("recommended_n") != source.fixed_scenario_n
        or replicate.get("recommended_n") != source.fixed_model_replicates
    ):
        raise StatisticsError("fixed N does not match the bound pilot recommendation")
    return {
        "pilot_manifest_declaration_sha256": (source.pilot_manifest_declaration_sha256),
        "pilot_statistics_content_sha256": source.pilot_statistics_content_sha256,
        "pilot_power_design_sha256": design_sha256,
        "variance_transfer_protocol": source.variance_transfer_protocol,
        "pilot_to_final_contrasts": {
            unit: {
                "pilot": pilot_contrast_by_unit[cast(PowerUnit, unit)].contrast_id,
                "final": contrast_by_unit[cast(PowerUnit, unit)].contrast_id,
            }
            for unit in ("scenario", "model_replicate")
        },
        "pilot_scenario_underpowered": scenario.get("underpowered"),
        "pilot_model_replicate_underpowered": replicate.get("underpowered"),
    }


def _build_power_section(
    evaluation_spec: PairedEvaluationSpec,
    differences_by_contrast: Mapping[str, Sequence[PairedDifference]],
    protocol: StatisticalProtocolSpec,
    pilot_document: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build prospective pilot power or a pre-registered fixed-N record."""
    by_unit = {
        cast(PowerUnit, contrast.power_unit): contrast
        for contrast in protocol.analysis_contrasts
        if contrast.power_unit is not None
    }
    scenario_contrast = by_unit["scenario"]
    replicate_contrast = by_unit["model_replicate"]
    source = protocol.power_source
    if source.data_role == "current-batch-pilot":
        if pilot_document is not None:
            raise StatisticsError("pilot statistics cannot consume another pilot")
        if evaluation_spec.scenario_bank.sealed or (
            evaluation_spec.scenario_bank.logical_split
            in {"validation-B", "sealed-test-iid", "ci-sealed-fixture"}
        ):
            raise StatisticsError(
                "confirmatory/sealed outcomes cannot be reused as a power pilot"
            )
        method_rows = differences_by_contrast[replicate_contrast.contrast_id]
        pilot_replicates = {
            row.replicate_id for row in method_rows if row.replicate_id is not None
        }
        if len(pilot_replicates) < MIN_PILOT_REPLICATES:
            raise StatisticsError(
                "formal crossed pilot requires at least 3 model replicates"
            )
        return {
            "data_role": "current-batch-pilot",
            "prospective_use_only": True,
            "source_manifest_declaration_sha256": (
                evaluation_spec.manifest_declaration_sha256
            ),
            "scenario": deployment_scenario_power_analysis(
                differences_by_contrast[scenario_contrast.contrast_id],
                protocol.scenario_power,
            ),
            "model_replicate": method_replicate_power_analysis(
                differences_by_contrast[replicate_contrast.contrast_id],
                protocol.replicate_power,
            ),
        }

    pilot_verification = validate_pilot_power_source(
        source,
        pilot_document,
        protocol,
    )
    fixed_scenario_n = cast(int, source.fixed_scenario_n)
    fixed_model_replicates = cast(int, source.fixed_model_replicates)
    method_rows = differences_by_contrast[replicate_contrast.contrast_id]
    observed_replicates = len(
        {row.replicate_id for row in method_rows if row.replicate_id is not None}
    )
    if evaluation_spec.scenario_bank.scenario_count != fixed_scenario_n:
        raise StatisticsError(
            "evaluation scenario count differs from pre-registered fixed N"
        )
    if observed_replicates != fixed_model_replicates:
        raise StatisticsError(
            "evaluation replicate count differs from pre-registered fixed N"
        )
    return {
        "data_role": "approved-pilot-fixed-n",
        "posthoc_power_reestimation": False,
        "pilot_verification": pilot_verification,
        "fixed_scenario_n": fixed_scenario_n,
        "fixed_model_replicates": fixed_model_replicates,
        "observed_scenario_n": evaluation_spec.scenario_bank.scenario_count,
        "observed_model_replicates": observed_replicates,
    }


def build_statistics_document(
    *,
    evaluation_spec: PairedEvaluationSpec,
    episode_records: Sequence[Mapping[str, object]],
    statistical_protocol: StatisticalProtocolSpec,
    source_manifest: Path,
    pilot_statistics_document: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Recompute outcomes and enforce the declared pilot/fixed-N power role."""
    require_paired_evaluation_manifest(source_manifest, evaluation_spec)
    from .manifest import load_manifest  # noqa: PLC0415 - avoid import cycle

    manifest = load_manifest(source_manifest)
    declaration = manifest.get("declaration")
    if (
        not isinstance(declaration, Mapping)
        or declaration.get("statistical_protocol")
        != statistical_protocol.manifest_binding()
    ):
        raise StatisticsError("statistics protocol is not bound by the manifest")
    artifact_contract = declaration.get("artifact_contract")
    statistics_filename = (
        artifact_contract.get("statistics")
        if isinstance(artifact_contract, Mapping)
        else None
    )
    if (
        type(statistics_filename) is not str
        or Path(statistics_filename).name != statistics_filename
        or not statistics_filename.endswith(".json")
    ):
        raise StatisticsError("statistics filename is not bound by the manifest")
    audit = audit_episode_batch(evaluation_spec, episode_records)
    if audit["status"] != "valid":
        raise StatisticsError("statistics.json cannot be built from an invalid batch")
    validate_analysis_plan_binding(
        evaluation_spec.manifest_binding(), statistical_protocol
    )
    differences_by_contrast = {
        contrast.contrast_id: _differences_for_contrast(episode_records, contrast)
        for contrast in statistical_protocol.analysis_contrasts
    }
    contrasts_by_family: dict[str, list[AnalysisContrast]] = defaultdict(list)
    for contrast in statistical_protocol.analysis_contrasts:
        contrasts_by_family[contrast.family_id].append(contrast)
    family_analyses: dict[str, object] = {}
    family_by_id = {
        family.family_id: family for family in statistical_protocol.hypothesis_families
    }
    for family_id in sorted(contrasts_by_family):
        family = family_by_id[family_id]
        contrasts = contrasts_by_family[family_id]
        rows = tuple(
            row
            for contrast in contrasts
            for row in differences_by_contrast[contrast.contrast_id]
        )
        kind = contrasts[0].kind
        if kind == "deployment":
            config = _bootstrap_config_for_rows(
                rows,
                statistical_protocol.scenario_bootstrap,
                statistical_protocol,
            )
            bootstrap: object = joint_scenario_cluster_bootstrap(rows, config)
            summaries: dict[str, object] = {}
        else:
            summaries = {
                contrast.contrast_id: summarize_method_differences(
                    differences_by_contrast[contrast.contrast_id]
                )
                for contrast in contrasts
            }
            replicate_count = len(
                {row.replicate_id for row in rows if row.replicate_id is not None}
            )
            if replicate_count >= MIN_NESTED_REPLICATES:
                config = _bootstrap_config_for_rows(
                    rows,
                    statistical_protocol.nested_bootstrap,
                    statistical_protocol,
                )
                bootstrap = nested_replicate_scenario_bootstrap(rows, config)
            else:
                bootstrap = {
                    "status": "not-reported",
                    "reason": (
                        "nested replicate-to-scenario bootstrap requires at least "
                        f"{MIN_NESTED_REPLICATES} independent model replicates"
                    ),
                    "replicate_count": replicate_count,
                }
        bootstrap_intervals = (
            bootstrap.get("intervals") if isinstance(bootstrap, Mapping) else None
        )
        if isinstance(bootstrap_intervals, Mapping) and (
            set(bootstrap_intervals) != set(family.endpoints)
        ):
            raise StatisticsError(
                f"computed endpoints for family {family_id!r} are incomplete"
            )
        family_analyses[family_id] = {
            "hypothesis": family.to_dict(),
            "contrast_kind": kind,
            "contrast_ids": [contrast.contrast_id for contrast in contrasts],
            "difference_count": len(rows),
            "difference_input_sha256": _difference_input_hash(rows),
            "bootstrap": bootstrap,
            "method_summaries": summaries,
        }

    power = _build_power_section(
        evaluation_spec,
        differences_by_contrast,
        statistical_protocol,
        pilot_statistics_document,
    )
    ordered_by_key = {
        validate_episode_record(record): dict(record) for record in episode_records
    }
    ordered_records = [ordered_by_key[row.key] for row in evaluation_spec.schedule()]
    document: dict[str, object] = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "statistics_filename": statistics_filename,
        "manifest_declaration_sha256": evaluation_spec.manifest_declaration_sha256,
        "manifest_declaration": dict(declaration),
        "evaluation": audit,
        "evaluation_manifest_binding": evaluation_spec.manifest_binding(),
        "episode_records_sha256": sha256_canonical_json(ordered_records),
        "statistical_protocol": statistical_protocol.manifest_binding(),
        "analyses": family_analyses,
        "power": power,
        "failure_handling": "batch-invalid; never drop or convert failed games",
        "wilson_usage": "descriptive-only; not used for paired decisions",
        "table_recalculation_source": (
            "all outcome analyses were recomputed inside this builder from the "
            "bound canonical episode rows; power is either explicitly pilot-only "
            "or inherited from the pre-bound pilot artifact without re-estimation"
        ),
    }
    document["content_sha256"] = sha256_canonical_json(document)
    # A strict JSON round-trip catches NaN/Inf before the artifact is written.
    json.dumps(document, allow_nan=False)
    return document


def write_statistics_document(path: Path, document: Mapping[str, object]) -> None:
    """Write one immutable statistics.json artifact without overwriting."""
    payload = dict(document)
    if Path(path).name != payload.get("statistics_filename"):
        raise StatisticsError("statistics output does not match the declared filename")
    content_hash = payload.pop("content_sha256", None)
    if content_hash != sha256_canonical_json(payload):
        raise StatisticsError("statistics document content SHA-256 mismatch")
    payload["content_sha256"] = content_hash
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    artifact = Path(path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    artifact.chmod(0o444)


__all__ = [
    "CLUSTER_BOOTSTRAP_PROTOCOL",
    "MIN_NESTED_REPLICATES",
    "MIN_PILOT_REPLICATES",
    "NESTED_BOOTSTRAP_PROTOCOL",
    "POWER_PROTOCOL",
    "POWER_VARIANCE_TRANSFER_PROTOCOL",
    "STATISTICS_SCHEMA_VERSION",
    "AnalysisContrast",
    "BootstrapConfig",
    "HypothesisFamily",
    "PairedDifference",
    "PowerAnalysisConfig",
    "PowerSourceSpec",
    "ScenarioScore",
    "StatisticalProtocolSpec",
    "StatisticsError",
    "aggregate_scenario_scores",
    "build_statistics_document",
    "checkpoint_contrast_differences",
    "checkpoint_differences",
    "deployment_scenario_power_analysis",
    "factorial_interaction_differences",
    "hypothesis_endpoint_id",
    "interquartile_mean",
    "joint_scenario_cluster_bootstrap",
    "method_contrast_differences",
    "method_differences",
    "method_replicate_power_analysis",
    "nested_replicate_scenario_bootstrap",
    "paired_power_analysis",
    "pilot_power_design_sha256",
    "replicate_effects",
    "scenario_effects",
    "summarize_method_differences",
    "validate_analysis_plan_binding",
    "validate_pilot_power_source",
    "validate_statistical_protocol_binding",
    "write_statistics_document",
]
