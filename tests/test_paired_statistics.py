"""Clustered contrasts, nested uncertainty, and fixed-N power contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace

import pytest

from splendor.agents.our_agents.policy_imitation.paired_evaluation import (
    EPISODE_SCHEMA_VERSION,
    PAIRED_EVALUATION_PROTOCOL,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    derive_seed,
    sha256_canonical_json,
)
from splendor.agents.our_agents.policy_imitation.statistics import (
    POWER_VARIANCE_TRANSFER_PROTOCOL,
    BootstrapConfig,
    PairedDifference,
    PowerAnalysisConfig,
    PowerSourceSpec,
    StatisticsError,
    aggregate_scenario_scores,
    deployment_scenario_power_analysis,
    factorial_interaction_differences,
    joint_scenario_cluster_bootstrap,
    method_contrast_differences,
    method_differences,
    method_replicate_power_analysis,
    nested_replicate_scenario_bootstrap,
    paired_power_analysis,
    scenario_effects,
    summarize_method_differences,
)

ScoreFunction = Callable[[str, int, int, str, int], float]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _outcome_fields(score_rate: float) -> tuple[int, float, float]:
    if score_rate == 1.0:
        return 1, 1.0, 0.0
    if score_rate == 0.5:
        return 0, 0.0, 0.0
    if score_rate == 0.0:
        return -1, 0.0, 1.0
    raise ValueError("synthetic seat score rates must be 0, 0.5, or 1")


def _episode_row(  # noqa: PLR0913 - explicit synthetic row coordinates
    *,
    treatment: str,
    replicate: int,
    model_seed: int,
    opponent: str,
    scenario_index: int,
    seat: int,
    score_rate: float,
    candidate_sha256: str | None = None,
    selection_kind: str = "iid",
    inclusion_probability: float = 1.0,
) -> dict[str, object]:
    candidate_hash = candidate_sha256 or _sha(f"candidate:{treatment}:{replicate}")
    opponent_hash = _sha(f"opponent:{opponent}")
    scenario_id = _sha(f"scenario:{scenario_index}")
    outcome, candidate_score, opponent_score = _outcome_fields(score_rate)
    key: dict[str, object] = {
        "candidate_sha256": candidate_hash,
        "opponent_sha256": opponent_hash,
        "scenario_id": scenario_id,
        "seat": seat,
        "replicate_id": replicate,
    }
    init_candidate = derive_seed(
        {"stream_name": "candidate_init", "scenario_id": scenario_id, "seat": seat}
    ).as_dict()
    init_opponent = derive_seed(
        {"stream_name": "opponent_init", "scenario_id": scenario_id, "seat": seat}
    ).as_dict()
    stress = selection_kind == "stress-balanced"
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "protocol": PAIRED_EVALUATION_PROTOCOL,
        "experiment_id": "synthetic-statistics",
        "phase": "T1.3",
        "batch_id": "synthetic-batch",
        "manifest_declaration_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "scenario_bank_sha256": "3" * 64,
        "scenario_bank_split": "stress" if stress else "validation-A",
        "schedule_sha256": "4" * 64,
        "episode_key": key,
        "episode_key_sha256": sha256_canonical_json(key),
        "candidate": {
            "policy_id": f"{treatment}-{replicate}",
            "policy_sha256": candidate_hash,
            "source_sha256": _sha(f"candidate-source:{treatment}:{replicate}"),
            "config_sha256": _sha(f"candidate-config:{treatment}:{replicate}"),
            "checkpoint_sha256": candidate_hash,
            "checkpoint_state_sha256": _sha(f"candidate-state:{treatment}:{replicate}"),
            "snapshot": f"/synthetic/{candidate_hash}.pth",
            "role": "control" if treatment == "O" else "treatment",
            "treatment_id": treatment,
            "replicate_id": replicate,
            "model_seed": model_seed,
            "feature_version": "public-v2",
        },
        "opponent": {
            "policy_id": opponent,
            "policy_sha256": opponent_hash,
            "source_sha256": _sha(f"opponent-source:{opponent}"),
            "config_sha256": _sha(f"opponent-config:{opponent}"),
            "checkpoint_sha256": None,
            "checkpoint_state_sha256": None,
            "snapshot": None,
            "role": "opponent",
            "treatment_id": None,
            "replicate_id": None,
            "model_seed": None,
            "feature_version": "v1",
        },
        "scenario_id": scenario_id,
        "canonical_state_sha256": scenario_id,
        "scenario_source_segment": "synthetic",
        "scenario_source_seed": scenario_index,
        "scenario_selection_kind": selection_kind,
        "scenario_inclusion_probability": inclusion_probability,
        "scenario_selection_stratum": "bin-0" if stress else None,
        "scenario_selection_design_sha256": "5" * 64 if stress else None,
        "seat": seat,
        "status": "completed",
        "failure": None,
        "outcome": outcome,
        "score_rate": score_rate,
        "candidate_raw_score": candidate_score,
        "opponent_raw_score": opponent_score,
        "candidate_cal_score": candidate_score,
        "opponent_cal_score": opponent_score,
        "plies": 0,
        "completed_rounds": 0,
        "candidate_queries": 0,
        "opponent_queries": 0,
        "candidate_illegal_actions": 0,
        "opponent_illegal_actions": 0,
        "candidate_search_nodes": 0,
        "opponent_search_nodes": 0,
        "candidate_action_counts": {},
        "opponent_action_counts": {},
        "candidate_inventory": {
            "bought_card_codes": [],
            "reserved_card_codes": [],
            "noble_codes": [],
            "gem_counts": {},
        },
        "opponent_inventory": {
            "bought_card_codes": [],
            "reserved_card_codes": [],
            "noble_codes": [],
            "gem_counts": {},
        },
        "candidate_latency": {
            "mean_seconds": 0.0,
            "p95_seconds": 0.0,
            "max_seconds": 0.0,
        },
        "opponent_latency": {
            "mean_seconds": 0.0,
            "p95_seconds": 0.0,
            "max_seconds": 0.0,
        },
        "elapsed_seconds": 0.0,
        "rng_lineage": {
            "candidate_init": init_candidate,
            "opponent_init": init_opponent,
            "candidate_actions": [],
            "opponent_actions": [],
        },
        "action_trace": [],
        "action_trace_sha256": sha256_canonical_json([]),
        "final_state_sha256": _sha("synthetic-final-state"),
    }


def _episode_matrix(  # noqa: PLR0913 - explicit crossed fixture axes
    treatments: Sequence[str],
    replicates: Sequence[int],
    scenarios: Sequence[int],
    opponents: Sequence[str],
    score: ScoreFunction,
    *,
    model_seed: Callable[[int], int] = lambda replicate: 100 + replicate,
) -> list[dict[str, object]]:
    return [
        _episode_row(
            treatment=treatment,
            replicate=replicate,
            model_seed=model_seed(replicate),
            opponent=opponent,
            scenario_index=scenario,
            seat=seat,
            score_rate=score(treatment, replicate, scenario, opponent, seat),
        )
        for treatment in treatments
        for replicate in replicates
        for scenario in scenarios
        for opponent in opponents
        for seat in (0, 1)
    ]


def _differences(
    *,
    replicates: Sequence[int | None],
    contrasts: Sequence[str] = ("d",),
    scenarios: int = 6,
    opponents: Sequence[str] = ("heuristic", "rush"),
) -> tuple[PairedDifference, ...]:
    rows = []
    for replicate in replicates:
        for scenario in range(scenarios):
            for contrast_index, contrast in enumerate(contrasts):
                for opponent_index, opponent in enumerate(opponents):
                    value = (
                        (scenario - (scenarios - 1) / 2) / (2 * scenarios)
                        + contrast_index * 0.1
                        + opponent_index * 0.02
                        + (0.01 * replicate if replicate is not None else 0.0)
                    )
                    candidate_score = 0.5 + value / 2
                    control_score = 0.5 - value / 2
                    rows.append(
                        PairedDifference(
                            contrast_id=contrast,
                            replicate_id=replicate,
                            opponent_id=opponent,
                            opponent_sha256=_sha(f"opponent:{opponent}"),
                            scenario_id=_sha(f"scenario:{scenario}"),
                            value=value,
                            component_scores=(
                                ("candidate", candidate_score),
                                ("control", control_score),
                            ),
                            component_policies=(
                                (
                                    "candidate",
                                    _sha(f"candidate:{replicate}"),
                                ),
                                ("control", _sha(f"control:{replicate}")),
                            ),
                            component_coefficients=(
                                ("candidate", 1.0),
                                ("control", -1.0),
                            ),
                            contrast_kind=(
                                "deployment" if replicate is None else "method"
                            ),
                            model_seed=(None if replicate is None else 100 + replicate),
                        )
                    )
    return tuple(rows)


def test_seats_are_aggregated_before_ties_and_method_contrasts() -> None:
    records = _episode_matrix(
        ("O", "A"),
        (0, 1),
        (0, 1),
        ("heuristic", "rush"),
        lambda treatment, _rep, _scenario, _opponent, seat: (
            0.5 if treatment == "O" else (1.0 if seat == 0 else 0.5)
        ),
    )
    scores = aggregate_scenario_scores(records)
    assert {row.score_rate for row in scores if row.treatment_id == "O"} == {0.5}
    assert {row.score_rate for row in scores if row.treatment_id == "A"} == {0.75}
    differences = method_differences(records, treatment_id="A")
    assert {row.value for row in differences} == {0.25}
    summary = summarize_method_differences(differences)
    assert summary["replicate_count"] == 2
    assert summary["mean_difference"] == pytest.approx(0.25)


def test_failed_missing_and_conflicting_metadata_are_rejected() -> None:
    records = _episode_matrix(
        ("O", "A"),
        (0,),
        (0,),
        ("heuristic",),
        lambda *_args: 0.5,
    )
    with pytest.raises(StatisticsError, match="both seats"):
        aggregate_scenario_scores(records[:-1])

    failed = deepcopy(records)
    failed[0]["status"] = "failed"
    failed[0]["failure"] = {"side": "candidate", "type": "X", "message": "x"}
    failed[0]["outcome"] = None
    failed[0]["score_rate"] = None
    with pytest.raises(StatisticsError, match="failed episode"):
        aggregate_scenario_scores(failed)

    opponent_relabel = deepcopy(records)
    opponent = opponent_relabel[-1]["opponent"]
    assert isinstance(opponent, dict)
    opponent["policy_id"] = "forged"
    with pytest.raises(StatisticsError, match="opponent digest has conflicting"):
        aggregate_scenario_scores(opponent_relabel)


def test_method_pairing_rejects_reused_checkpoint_or_model_seed() -> None:
    records = _episode_matrix(
        ("O", "A"),
        (0, 1),
        (0,),
        ("heuristic",),
        lambda *_args: 0.5,
    )
    reused_checkpoint = deepcopy(records)
    a_rows = [
        row
        for row in reused_checkpoint
        if isinstance(row["candidate"], Mapping)
        and row["candidate"]["treatment_id"] == "A"  # type: ignore[index]
    ]
    first_hash = a_rows[0]["candidate"]["policy_sha256"]  # type: ignore[index]
    for row in a_rows:
        candidate = row["candidate"]
        key = row["episode_key"]
        assert isinstance(candidate, dict) and isinstance(key, dict)
        if candidate["replicate_id"] == 1:
            candidate["policy_sha256"] = first_hash
            candidate["checkpoint_sha256"] = first_hash
            key["candidate_sha256"] = first_hash
            row["episode_key_sha256"] = sha256_canonical_json(key)
    with pytest.raises(StatisticsError, match=r"conflicting|reused"):
        method_differences(reused_checkpoint, treatment_id="A")

    reused_seed = _episode_matrix(
        ("O", "A"),
        (0, 1),
        (0,),
        ("heuristic",),
        lambda *_args: 0.5,
        model_seed=lambda _replicate: 777,
    )
    with pytest.raises(StatisticsError, match="model seed is reused"):
        method_differences(reused_seed, treatment_id="A")


def test_factorial_interaction_uses_c_minus_a_minus_b_plus_o() -> None:
    arm_scores = {"O": 0.0, "A": 0.5, "B": 0.5, "C": 1.0}
    records = _episode_matrix(
        ("O", "A", "B", "C"),
        (0,),
        (0, 1),
        ("heuristic",),
        lambda treatment, *_args: arm_scores[treatment],
    )
    interaction = factorial_interaction_differences(records)
    assert {row.value for row in interaction} == {0.0}

    arm_scores["C"] = 0.5
    negative = factorial_interaction_differences(
        _episode_matrix(
            ("O", "A", "B", "C"),
            (0,),
            (0,),
            ("heuristic",),
            lambda treatment, *_args: arm_scores[treatment],
        )
    )
    assert negative[0].value == -0.5


def test_method_contrast_math_is_canonical_and_flat_bootstrap_is_forbidden() -> None:
    records = _episode_matrix(
        ("O", "A", "B", "C"),
        (0,),
        (0, 1),
        ("heuristic",),
        lambda treatment, _rep, scenario, _opponent, seat: (
            1.0
            if (ord(treatment) + scenario + seat) % 3 == 0
            else (0.5 if (ord(treatment) + scenario + seat) % 3 == 1 else 0.0)
        ),
    )
    left = method_contrast_differences(
        records,
        coefficients={"C": 1.0, "A": -1.0, "B": -1.0, "O": 1.0},
        contrast_id="interaction",
    )
    right = method_contrast_differences(
        records,
        coefficients={"O": 1.0, "B": -1.0, "A": -1.0, "C": 1.0},
        contrast_id="interaction",
    )
    assert left == right
    with pytest.raises(StatisticsError, match="reserved for deployment"):
        joint_scenario_cluster_bootstrap(
            left,
            BootstrapConfig(resamples=1_000),
        )


def test_joint_bootstrap_is_order_invariant_and_duplicate_clusters_do_not_tighten() -> (
    None
):
    differences = _differences(replicates=(None,), contrasts=("deploy",))
    config = BootstrapConfig(resamples=1_000, seed=123)
    original = joint_scenario_cluster_bootstrap(differences, config)
    reordered = joint_scenario_cluster_bootstrap(tuple(reversed(differences)), config)
    duplicated = joint_scenario_cluster_bootstrap(
        tuple(row for row in differences for _ in range(7)), config
    )
    assert original == reordered
    assert original["intervals"] == duplicated["intervals"]
    assert original["resample_index_sha256"] == duplicated["resample_index_sha256"]
    assert original["scenario_count"] == duplicated["scenario_count"] == 6

    multiple = joint_scenario_cluster_bootstrap(
        _differences(replicates=(None,), contrasts=("deploy", "safety")), config
    )
    intervals = multiple["intervals"]
    assert isinstance(intervals, dict) and len(intervals) == 4
    assert any(key.startswith("deploy|") for key in intervals)
    assert any(key.startswith("safety|") for key in intervals)


def test_stress_rows_cannot_be_reported_as_unweighted_iid() -> None:
    stress = tuple(
        PairedDifference(
            **{
                **row.__dict__,
                "selection_kind": "stress-balanced",
                "inclusion_probability": 0.25,
                "selection_stratum": "bin-0",
            }
        )
        for row in _differences(replicates=(None,))
    )
    with pytest.raises(StatisticsError, match="IID estimands reject"):
        joint_scenario_cluster_bootstrap(stress, BootstrapConfig(resamples=1_000))
    weighted = joint_scenario_cluster_bootstrap(
        stress,
        BootstrapConfig(
            resamples=1_000,
            seed=1,
            weighting="inverse-inclusion",
        ),
    )
    assert weighted["config"]["weighting"] == "inverse-inclusion"  # type: ignore[index]

    with pytest.raises(StatisticsError, match="selection kind is invalid"):
        replace(
            _differences(replicates=(None,))[0],
            selection_kind="natural-deal-srswor",
            inclusion_probability=0.16,
            selection_stratum="replicate-0",
        )


def test_nested_bootstrap_keeps_replicate_then_scenario_hierarchy() -> None:
    with pytest.raises(StatisticsError, match="at least 5 replicates"):
        nested_replicate_scenario_bootstrap(
            _differences(replicates=(0, 1, 2, 3)),
            BootstrapConfig(resamples=1_000),
        )
    differences = _differences(
        replicates=(0, 1, 2, 3, 4),
        contrasts=("method", "interaction"),
    )
    config = BootstrapConfig(resamples=1_000, seed=98)
    result = nested_replicate_scenario_bootstrap(differences, config)
    reordered = nested_replicate_scenario_bootstrap(
        tuple(reversed(differences)), config
    )
    assert result == reordered
    assert result["replicate_count"] == 5
    assert result["scenario_count_per_replicate"] == 6
    assert len(result["intervals"]) == 4  # type: ignore[arg-type]

    reused_seed = tuple(
        replace(row, model_seed=100) if row.replicate_id == 4 else row
        for row in differences
    )
    with pytest.raises(StatisticsError, match="model seed is reused"):
        nested_replicate_scenario_bootstrap(reused_seed, config)


def test_power_keeps_scenario_and_model_replicate_units_separate() -> None:
    deployment = _differences(replicates=(None,), scenarios=8)
    scenario_config = PowerAnalysisConfig(
        "scenario",
        0.05,
        family_size=2,
        max_n=250,
        round_to=50,
        simulations=1_000,
        seed=11,
    )
    scenario_result = deployment_scenario_power_analysis(deployment, scenario_config)
    assert scenario_result["independent_unit"] == "scenario"
    assert scenario_result["recommended_n"] in {200, 250}
    assert scenario_result["recommended_n"] % 50 == 0  # type: ignore[operator]

    method = _differences(replicates=(0, 1, 2, 3, 4), scenarios=4)
    replicate_config = PowerAnalysisConfig(
        "model_replicate",
        0.05,
        family_size=2,
        max_n=10,
        simulations=1_000,
        seed=12,
    )
    replicate_result = method_replicate_power_analysis(method, replicate_config)
    assert (
        method_replicate_power_analysis(tuple(reversed(method)), replicate_config)
        == replicate_result
    )
    assert replicate_result["independent_unit"] == "model_replicate"
    endpoint_results = replicate_result["endpoint_results"]
    assert isinstance(endpoint_results, dict)
    assert {result["pilot_unit_count"] for result in endpoint_results.values()} == {5}
    with pytest.raises(StatisticsError, match="unit='scenario'"):
        deployment_scenario_power_analysis(deployment, replicate_config)
    with pytest.raises(StatisticsError, match="unit='model_replicate'"):
        method_replicate_power_analysis(method, scenario_config)
    with pytest.raises(StatisticsError, match="family_size"):
        deployment_scenario_power_analysis(
            deployment,
            PowerAnalysisConfig(
                "scenario",
                0.05,
                family_size=1,
                max_n=200,
                round_to=50,
                simulations=1_000,
            ),
        )

    underpowered = paired_power_analysis(
        (-0.5, -0.25, 0.0, 0.25, 0.5),
        PowerAnalysisConfig(
            "scenario",
            0.0001,
            max_n=200,
            round_to=50,
            simulations=1_000,
            seed=13,
        ),
    )
    assert underpowered["recommended_n"] == 200
    assert underpowered["underpowered"] is True
    with pytest.raises(StatisticsError, match="zero paired variance"):
        paired_power_analysis((0.1, 0.1, 0.1), replicate_config)

    rounded_replicates = paired_power_analysis(
        (-0.2, -0.1, 0.0, 0.1, 0.2),
        PowerAnalysisConfig(
            "model_replicate",
            0.05,
            max_n=12,
            round_to=2,
            simulations=1_000,
            seed=14,
        ),
    )
    assert rounded_replicates["design_minimum_n"] == 6
    assert rounded_replicates["recommended_n"] % 2 == 0  # type: ignore[operator]


def test_mechanical_scenario_duplicates_do_not_change_power_units() -> None:
    deployment = _differences(replicates=(None,), scenarios=5)
    duplicated = tuple(row for row in deployment for _ in range(5))
    endpoint = deployment[0].opponent_sha256
    deployment_endpoint = tuple(
        row for row in deployment if row.opponent_sha256 == endpoint
    )
    duplicated_endpoint = tuple(
        row for row in duplicated if row.opponent_sha256 == endpoint
    )
    assert scenario_effects(deployment_endpoint) == scenario_effects(
        duplicated_endpoint
    )

    conflicting = list(deployment_endpoint)
    first = conflicting[0]
    altered_scores = (("candidate", 0.75), ("control", 0.25))
    conflicting.append(
        replace(
            first,
            value=0.5,
            component_scores=altered_scores,
        )
    )
    with pytest.raises(StatisticsError, match="conflicting duplicate"):
        scenario_effects(conflicting)


def test_power_protocol_rejects_lax_thresholds_and_percentage_points() -> None:
    with pytest.raises(StatisticsError, match="cannot carry fixed-N"):
        PowerSourceSpec("current-batch-pilot", fixed_scenario_n=200)
    with pytest.raises(StatisticsError, match="pilot manifest/statistics hashes"):
        PowerSourceSpec(
            "approved-pilot-fixed-n",
            fixed_scenario_n=200,
            fixed_model_replicates=5,
        )
    with pytest.raises(StatisticsError, match="power-design SHA-256"):
        PowerSourceSpec(
            "approved-pilot-fixed-n",
            pilot_manifest_declaration_sha256="1" * 64,
            pilot_statistics_content_sha256="2" * 64,
            fixed_scenario_n=200,
            fixed_model_replicates=5,
        )
    with pytest.raises(StatisticsError, match="variance-transfer protocol"):
        PowerSourceSpec(
            "approved-pilot-fixed-n",
            pilot_manifest_declaration_sha256="1" * 64,
            pilot_statistics_content_sha256="2" * 64,
            fixed_scenario_n=200,
            fixed_model_replicates=5,
            pilot_power_design_sha256="3" * 64,
        )
    with pytest.raises(StatisticsError, match="50-scenario blocks"):
        PowerSourceSpec(
            "approved-pilot-fixed-n",
            pilot_manifest_declaration_sha256="1" * 64,
            pilot_statistics_content_sha256="2" * 64,
            fixed_scenario_n=225,
            fixed_model_replicates=5,
            pilot_power_design_sha256="3" * 64,
            variance_transfer_protocol=POWER_VARIANCE_TRANSFER_PROTOCOL,
        )
    with pytest.raises(StatisticsError, match="familywise_alpha"):
        PowerAnalysisConfig(
            "model_replicate",
            0.03,
            familywise_alpha=0.49,
        )
    with pytest.raises(StatisticsError, match="target_power"):
        PowerAnalysisConfig("model_replicate", 0.03, target_power=0.51)
    with pytest.raises(StatisticsError, match="200-scenario floor"):
        PowerAnalysisConfig("scenario", 0.03, max_n=100, round_to=50)
    with pytest.raises(StatisticsError, match="five model replicates"):
        PowerAnalysisConfig("model_replicate", 0.03, max_n=4)
    with pytest.raises(StatisticsError, match=r"\[-1, 1\]"):
        paired_power_analysis(
            (0.0, 5.0),
            PowerAnalysisConfig("model_replicate", 0.03),
        )
