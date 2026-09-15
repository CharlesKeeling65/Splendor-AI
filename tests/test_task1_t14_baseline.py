"""CI fixture for the T1.4 two-bank frozen-baseline driver."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from splendor.agents.generic.first_move import FirstActionAgent
from splendor.agents.generic.random import RandomAgent
from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest_v2,
    transition_manifest,
)
from splendor.agents.our_agents.policy_imitation.paired_evaluation import (
    EvaluationPolicy,
    PairedEvaluationSpec,
    play_paired_evaluation_game,
    snapshotless_policy_sha256,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.protocol import (
    capture_code_revision,
    sha256_canonical_json,
)
from splendor.agents.our_agents.policy_imitation.scenario_bank import (
    generate_iid_scenarios,
    load_scenario_bank,
    select_balanced_stress_scenarios,
    write_scenario_bank,
)
from splendor.agents.our_agents.policy_imitation.t14_baseline import (
    T14_BASELINE_OPPONENTS,
    T14_PPO_BEST_CONFIG_SHA256,
    T14_PPO_BEST_SOURCE_SHA256,
    T14_PPO_BEST_STATE_SHA256,
    BaselineCell,
    T14BaselineError,
    T14BaselineSpec,
    baseline_descriptive_statistical_binding,
    build_t14_baseline_document,
    finite_population_variance_decomposition,
    run_t14_baseline_evaluation,
    validate_t14_bank_pair,
    validate_t14_baseline_statistical_binding,
    write_t14_baseline_document,
)
from splendor.seed_registry import TASK1_SCENARIO_SPLITS

REPO = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _policy(
    policy_id: str,
    factory: type[FirstActionAgent] | type[RandomAgent],
    *,
    role: str,
) -> EvaluationPolicy:
    candidate = CandidateSpec(policy_id, "fixed_baseline", factory)
    digest = snapshotless_policy_sha256(candidate)
    return EvaluationPolicy(
        policy_id=policy_id,
        policy_sha256=digest,
        candidate=candidate,
        role=role,  # type: ignore[arg-type]
    )


def _stress_fixture(tmp_path: Path) -> Path:
    candidates = generate_iid_scenarios("ci-fixture", 12, offset=10)
    descriptors = sorted(dict(candidates[0].strata_v1))
    descriptor = ""
    edge = 0.0
    for name in descriptors:
        values = sorted({float(dict(row.strata_v1)[name]) for row in candidates})
        if len(values) >= 2:
            descriptor = name
            edge = (values[0] + values[-1]) / 2.0
            if any(value < edge for value in values) and any(
                value >= edge for value in values
            ):
                break
    assert descriptor
    selected = select_balanced_stress_scenarios(
        candidates,
        descriptor=descriptor,
        bin_edges=(edge,),
        per_bin=1,
        selection_key=_sha("ci-stress-selection"),
    )
    artifact = write_scenario_bank(
        tmp_path / "banks",
        "ci-fixture",
        selected,
        compression="none",
    )
    return Path(str(artifact["artifact_path"]))


@pytest.fixture
def baseline_spec(tmp_path: Path) -> T14BaselineSpec:
    iid_artifact = write_scenario_bank(
        tmp_path / "banks",
        "ci-fixture",
        generate_iid_scenarios("ci-fixture", 2),
        compression="none",
    )
    iid_bank = load_scenario_bank(Path(str(iid_artifact["artifact_path"])))
    stress_bank = load_scenario_bank(_stress_fixture(tmp_path))
    candidate = _policy("ci-baseline", FirstActionAgent, role="ci-fixture")
    opponents = (
        _policy("first", FirstActionAgent, role="opponent"),
        _policy("random", RandomAgent, role="opponent"),
    )

    def evaluation(
        bank: object,
        batch_id: str,
        episodes_filename: str,
    ) -> PairedEvaluationSpec:
        return PairedEvaluationSpec(
            experiment_id="task1-t14-ci",
            phase="T1.4-baseline-ci",
            batch_id=batch_id,
            code_sha256=sha256_canonical_json(capture_code_revision(REPO)),
            manifest_declaration_sha256=_sha(f"{batch_id}:provisional"),
            candidates=(candidate,),
            opponents=opponents,
            scenario_bank=bank,  # type: ignore[arg-type]
            episodes_filename=episodes_filename,
        )

    return T14BaselineSpec(
        iid_evaluation=evaluation(
            iid_bank, "iid-ci", "iid-ci.jsonl"
        ),
        stress_evaluation=evaluation(
            stress_bank, "stress-ci", "stress-ci.jsonl"
        ),
        expected_checkpoint_sha256=None,
        profile="ci-fixture",
    )


def _records(spec: PairedEvaluationSpec) -> list[dict[str, object]]:
    candidates = {policy.policy_sha256: policy for policy in spec.candidates}
    opponents = {policy.policy_sha256: policy for policy in spec.opponents}
    scenarios = {
        scenario.scenario_id: scenario for scenario in spec.scenario_bank.scenarios
    }
    return [
        play_paired_evaluation_game(
            spec,
            candidates[row.candidate_sha256],
            opponents[row.opponent_sha256],
            scenarios[row.scenario_id],
            row.seat,
        )
        for row in spec.schedule()
    ]


def _running_manifest(
    tmp_path: Path,
    evaluation: PairedEvaluationSpec,
) -> tuple[Path, PairedEvaluationSpec]:
    path = tmp_path / f"{evaluation.batch_id}-manifest.json"
    manifest = create_manifest_v2(
        path,
        experiment_id=evaluation.experiment_id,
        phase=evaluation.phase,
        purpose="CI-only T1.4 descriptive baseline fixture",
        seed_segments={
            name: segment.name for name, segment in TASK1_SCENARIO_SPLITS.items()
        },
        budget={"scheduled_games": len(evaluation.schedule())},
        hypotheses={"kind": "descriptive-only"},
        estimands={"kind": "finite-population-functional-anova"},
        decision_rule={"inference": "forbidden", "selection": "forbidden"},
        artifact_contract={
            "scenario_banks": {
                evaluation.scenario_bank.logical_split: (
                    evaluation.scenario_bank.payload_sha256
                )
            },
            "episodes": evaluation.episodes_filename,
        },
        baselines={"ci_fixture": evaluation.candidates[0].policy_sha256},
        paired_evaluation=evaluation.manifest_binding(),
        statistical_protocol=baseline_descriptive_statistical_binding(
            profile="ci-fixture"
        ),
        repo=REPO,
    )
    bound = replace(
        evaluation,
        manifest_declaration_sha256=str(manifest["declaration_sha256"]),
    )
    approve_manifest(path, "ci-reviewer", "CI fixture declaration reviewed")
    transition_manifest(
        path,
        "running",
        actor="ci-runner",
        note="start CI baseline fixture",
        declaration_sha256=bound.manifest_declaration_sha256,
    )
    return path, bound


def test_functional_anova_has_exact_additive_sum() -> None:
    scenarios = (_sha("deal-0"), _sha("deal-1"))
    opponents = (_sha("opp-0"), _sha("opp-1"))
    deal_effect = (-0.2, 0.2)
    seat_effect = (-0.1, 0.1)
    opponent_effect = (-0.05, 0.05)
    cells = [
        BaselineCell(
            scenario_id=scenario,
            seat=seat,
            opponent_sha256=opponent,
            score_rate=(
                0.5
                + deal_effect[scenario_index]
                + seat_effect[seat]
                + opponent_effect[opponent_index]
            ),
        )
        for scenario_index, scenario in enumerate(scenarios)
        for seat in (0, 1)
        for opponent_index, opponent in enumerate(opponents)
    ]
    result = finite_population_variance_decomposition(cells)
    components = result["component_variances"]
    assert components == pytest.approx(  # type: ignore[arg-type]
        {"deal": 0.04, "seat": 0.01, "opponent": 0.0025, "interaction": 0.0}
    )
    assert result["total_variance"] == pytest.approx(0.0525)


def test_ci_driver_builds_hashed_two_bank_report(
    tmp_path: Path, baseline_spec: T14BaselineSpec
) -> None:
    iid_manifest, iid_evaluation = _running_manifest(
        tmp_path, baseline_spec.iid_evaluation
    )
    stress_manifest, stress_evaluation = _running_manifest(
        tmp_path, baseline_spec.stress_evaluation
    )
    bound_spec = replace(
        baseline_spec,
        iid_evaluation=iid_evaluation,
        stress_evaluation=stress_evaluation,
    )
    run_result = run_t14_baseline_evaluation(
        bound_spec,
        tmp_path / "run",
        iid_manifest=iid_manifest,
        stress_manifest=stress_manifest,
    )
    document = run_result["document"]
    assert document["overall_status"] == "valid"
    results = document["results"]
    for split in ("validation-A", "stress"):
        bank_result = results[split]  # type: ignore[index]
        assert bank_result["audit"]["failed_games"] == 0
        assert bank_result["wdl_and_score_rate"]["overall"]["scheduled"] == 8
        variance = bank_result["variance_decomposition"]
        assert variance["cell_count"] == 8
        assert variance["total_variance"] == pytest.approx(
            sum(variance["component_variances"].values())
        )

    first = run_result["artifact"]
    second = write_t14_baseline_document(tmp_path / "reports", document)
    assert first["content_sha256"] == second["content_sha256"]
    artifact = Path(str(first["path"]))
    assert artifact.name == f"t14-baseline-{first['content_sha256']}.json"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == first["content_sha256"]


def test_failures_are_retained_and_invalidate_variance(
    baseline_spec: T14BaselineSpec,
) -> None:
    iid_records = _records(baseline_spec.iid_evaluation)
    stress_records = _records(baseline_spec.stress_evaluation)
    failed = dict(iid_records[0])
    failed.update(
        {
            "status": "failed",
            "failure": {
                "side": "finalization",
                "type": "FixtureFailure",
                "message": "retained CI failure",
            },
            "outcome": None,
            "score_rate": None,
        }
    )
    iid_records[0] = failed
    document = build_t14_baseline_document(
        baseline_spec,
        iid_records=iid_records,
        stress_records=stress_records,
    )
    iid = document["results"]["validation-A"]  # type: ignore[index]
    assert document["overall_status"] == "invalid"
    assert iid["audit"]["failed_games"] == 1
    assert iid["wdl_and_score_rate"]["overall"]["score_rate"] is None
    assert iid["failures"][0]["failure"]["message"] == "retained CI failure"
    assert iid["variance_decomposition"] is None


def test_profile_rejects_validation_b_and_any_sealed_bank(
    tmp_path: Path, baseline_spec: T14BaselineSpec
) -> None:
    validation_b_artifact = write_scenario_bank(
        tmp_path / "validation-b",
        "validation-B",
        generate_iid_scenarios("validation-B", 2),
        compression="none",
    )
    validation_b = load_scenario_bank(
        Path(str(validation_b_artifact["artifact_path"]))
    )
    with pytest.raises(T14BaselineError, match="CI profile"):
        validate_t14_bank_pair(
            validation_b,
            baseline_spec.stress_evaluation.scenario_bank,
            profile="ci-fixture",
        )
    with pytest.raises(T14BaselineError, match="sealed"):
        validate_t14_bank_pair(
            replace(baseline_spec.iid_evaluation.scenario_bank, sealed=True),
            baseline_spec.stress_evaluation.scenario_bank,
            profile="ci-fixture",
        )


def test_config_hash_binds_checkpoint_bank_schedule_and_opponents(
    baseline_spec: T14BaselineSpec,
) -> None:
    changed = replace(
        baseline_spec.iid_evaluation,
        manifest_declaration_sha256=_sha("changed-manifest"),
    )
    changed_spec = replace(baseline_spec, iid_evaluation=changed)
    assert changed_spec.config_sha256 != baseline_spec.config_sha256


def test_descriptive_manifest_branch_rejects_inference_and_extra_candidates(
    baseline_spec: T14BaselineSpec,
) -> None:
    statistics = baseline_descriptive_statistical_binding(profile="ci-fixture")
    binding = baseline_spec.iid_evaluation.manifest_binding()
    assert (
        validate_t14_baseline_statistical_binding(
            statistics,
            binding,
            phase="T1.4-baseline-ci",
        )
        == "ci-fixture"
    )
    inferential = dict(statistics)
    inferential["confidence_intervals"] = True
    with pytest.raises(T14BaselineError, match="schema mismatch"):
        validate_t14_baseline_statistical_binding(
            inferential,
            binding,
            phase="T1.4-baseline-ci",
        )
    two_candidates = dict(binding)
    two_candidates["candidates"] = binding["candidates"] * 2  # type: ignore[operator]
    with pytest.raises(T14BaselineError, match="exactly one"):
        validate_t14_baseline_statistical_binding(
            statistics,
            two_candidates,
            phase="T1.4-baseline-ci",
        )


def test_production_descriptive_manifest_rejects_validation_b(
    baseline_spec: T14BaselineSpec,
) -> None:
    binding = baseline_spec.iid_evaluation.manifest_binding()
    candidate = dict(binding["candidates"][0])  # type: ignore[index]
    candidate.update(
        {
            "policy_id": "ppo-best",
            "policy_sha256": "e225464c17a783bd91b51251336f917e9f867af4f475d8414372885fbb758102",
            "checkpoint_sha256": "e225464c17a783bd91b51251336f917e9f867af4f475d8414372885fbb758102",
            "source_sha256": T14_PPO_BEST_SOURCE_SHA256,
            "config_sha256": T14_PPO_BEST_CONFIG_SHA256,
            "checkpoint_state_sha256": T14_PPO_BEST_STATE_SHA256,
            "role": "frozen-baseline",
            "feature_version": "public-v2",
        }
    )
    forbidden = dict(binding)
    forbidden["candidates"] = [candidate]
    forbidden["opponents"] = [dict(item) for item in T14_BASELINE_OPPONENTS]
    forbidden["scenario_bank_split"] = "validation-B"
    with pytest.raises(T14BaselineError, match="only validation-A or stress"):
        validate_t14_baseline_statistical_binding(
            baseline_descriptive_statistical_binding(profile="production"),
            forbidden,
            phase="T1.4",
        )

    arbitrary_opponents = dict(forbidden)
    arbitrary_opponents["scenario_bank_split"] = "validation-A"
    arbitrary_opponents["opponents"] = binding["opponents"]
    with pytest.raises(T14BaselineError, match="frozen ga/heuristic/minimax"):
        validate_t14_baseline_statistical_binding(
            baseline_descriptive_statistical_binding(profile="production"),
            arbitrary_opponents,
            phase="T1.4",
        )
