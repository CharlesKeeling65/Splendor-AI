"""ScenarioV1 conservation, content addressing, split, and sealed gates."""

from __future__ import annotations

import hashlib
import json
import random
import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest_v2,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    sha256_canonical_json,
)
from splendor.agents.our_agents.policy_imitation.scenario import (
    SCENARIO_DECK_COUNTS,
    ScenarioV1,
    ScenarioValidationError,
    generate_scenario,
    scenario_from_state,
    state_from_scenario,
    validate_scenario,
)
from splendor.agents.our_agents.policy_imitation.scenario_bank import (
    FINAL_REPORT_PURPOSE,
    ScenarioBankError,
    consume_sealed_scenario_bank,
    freeze_candidate_set,
    generate_iid_scenarios,
    load_scenario_bank,
    load_scenario_bank_subset,
    select_balanced_stress_scenarios,
    validate_split_disjointness,
    validate_task1_split_registry,
    write_scenario_bank,
)
from splendor.seed_registry import TASK1_SCENARIO_SPLITS
from splendor.splendor.splendor_utils import CARDS

REPO = Path(__file__).resolve().parents[1]


def _scenario(seed: int = 825_000) -> ScenarioV1:
    return generate_scenario("ci_smoke", seed, selection_kind="ci-fixture")


def _manifest(path: Path, bank_payload_sha256: str) -> dict[str, object]:
    create_manifest_v2(
        path,
        experiment_id="task1-sealed-gate-test",
        phase="T1.2",
        purpose="sealed gate protocol fixture",
        seed_segments={
            "training": "training",
            "validation": "validation",
            "final_test": "independent_test",
        },
        budget={"games": 1},
        hypotheses={"H": "gate only"},
        estimands={"d": "none"},
        decision_rule={"selection": "fixed fixture"},
        artifact_contract={
            "bank": "fixture",
            "scenario_banks": {
                "ci-sealed-fixture": bank_payload_sha256,
            },
        },
        baselines={"ppo-best": {"sha256": "a" * 64}},
        repo=REPO,
    )
    return approve_manifest(path, "independent-test", "fixture declaration reviewed")


def test_scenario_roundtrip_conservation_and_known_vector() -> None:
    scenario = _scenario()
    assert scenario.scenario_id == (
        "4e36c685f742796ba79a02dce1b832e9084feb3f812c760b71937c1c06655e65"
    )
    assert scenario.scenario_id == scenario.canonical_state_sha256
    assert tuple(len(tier) for tier in scenario.decks) == SCENARIO_DECK_COUNTS
    assert [len(tier) for tier in scenario.dealt] == [4, 4, 4]
    assert len(scenario.nobles_in_order) == 3
    codes = [code for tier in (*scenario.dealt, *scenario.decks) for code in tier]
    assert len(codes) == len(set(codes)) == len(CARDS) == 90
    assert set(codes) == set(CARDS)
    assert ScenarioV1.from_dict(scenario.to_dict()) == scenario
    assert scenario.deck_top == "list_end"


def test_generation_and_loading_do_not_mutate_any_global_rng() -> None:
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    scenario = _scenario(825_001)
    first = state_from_scenario(scenario)
    second = state_from_scenario(scenario)

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert first.board.dealt[0][0] is not second.board.dealt[0][0]
    assert first.board.dealt[0][0].cost is not second.board.dealt[0][0].cost


def test_state_identity_excludes_provenance_but_binds_order() -> None:
    original = _scenario()
    same_state = scenario_from_state(
        state_from_scenario(original),
        source_segment="ci_smoke",
        source_seed=825_099,
        selection_kind="ci-fixture",
    )
    assert same_state.scenario_id == original.scenario_id
    assert same_state.source_seed != original.source_seed

    first_deck = list(original.decks[0])
    first_deck[-1], first_deck[-2] = first_deck[-2], first_deck[-1]
    tampered = replace(original, decks=(tuple(first_deck), *original.decks[1:]))
    with pytest.raises(ScenarioValidationError, match="canonical state hash"):
        validate_scenario(tampered)


def test_freezing_rejects_modified_source_card_primitives() -> None:
    scenario = _scenario()
    state = state_from_scenario(scenario)
    state.board.dealt[0][0].points += 1
    with pytest.raises(ScenarioValidationError, match="primitive fields were modified"):
        scenario_from_state(
            state,
            source_segment="ci_smoke",
            source_seed=825_000,
            selection_kind="ci-fixture",
        )


@pytest.mark.parametrize("field", ["card_registry_hash", "action_registry_hash"])
def test_scenario_rejects_registry_tampering(field: str) -> None:
    scenario = _scenario()
    with pytest.raises(ScenarioValidationError, match="registry hash mismatch"):
        validate_scenario(replace(scenario, **{field: "0" * 64}))


def test_scenario_rejects_missing_duplicate_and_wrong_tier_cards() -> None:
    scenario = _scenario()
    duplicate_dealt = [list(tier) for tier in scenario.dealt]
    duplicate_dealt[0][0] = duplicate_dealt[0][1]
    with pytest.raises(ScenarioValidationError, match="missing or duplicated"):
        validate_scenario(
            replace(scenario, dealt=tuple(tuple(tier) for tier in duplicate_dealt))
        )

    wrong_tier = [list(tier) for tier in scenario.dealt]
    wrong_tier[0][0], wrong_tier[1][0] = wrong_tier[1][0], wrong_tier[0][0]
    with pytest.raises(ScenarioValidationError, match="wrong tier"):
        validate_scenario(replace(scenario, dealt=tuple(map(tuple, wrong_tier))))


def test_strata_are_opening_only_and_formulas_are_locked() -> None:
    strata = dict(_scenario().strata_v1)
    assert strata == {
        "cost_colour_concentration_mean": 0.663293650794,
        "cost_colour_entropy_mean": 0.409690460426,
        "low_cost_high_point_count": 5,
        "noble_pair_weighted_jaccard_mean": 0.222222222222,
        "token_deficit_min": 3,
        "token_deficit_q25": 4.75,
        "token_deficit_q50": 6.0,
        "token_deficit_q75": 8.25,
        "visible_noble_alignment_cosine": 0.867721831275,
        "visible_point_density": 1.916666666667,
    }
    assert "opening_buyability" not in strata
    assert "rush_index" not in strata


def test_registered_task1_splits_are_distinct_and_correctly_sealed() -> None:
    validate_task1_split_registry()
    assert list(TASK1_SCENARIO_SPLITS) == [
        "train-schedule",
        "dagger-rollout",
        "teacher-validation",
        "validation-A",
        "validation-B",
        "sealed-test-iid",
        "stress",
    ]
    assert TASK1_SCENARIO_SPLITS["sealed-test-iid"].sealed
    assert not any(
        segment.sealed
        for name, segment in TASK1_SCENARIO_SPLITS.items()
        if name != "sealed-test-iid"
    )


def test_split_audit_detects_same_state_even_with_different_provenance() -> None:
    original = _scenario()
    duplicate_state = scenario_from_state(
        state_from_scenario(original),
        source_segment="stabilization_test",
        source_seed=823_000,
        selection_kind="ci-fixture",
    )
    with pytest.raises(ScenarioBankError, match=r"scenario_id.*duplicated"):
        validate_split_disjointness(
            {
                "ci-fixture": [original],
                "ci-sealed-fixture": [duplicate_state],
            }
        )


def test_balanced_stress_selection_records_inclusion_probability() -> None:
    candidates = generate_iid_scenarios("ci-fixture", 20)
    values = sorted(
        {
            float(dict(scenario.strata_v1)["visible_point_density"])
            for scenario in candidates
        }
    )
    assert len(values) > 1
    edge = (values[0] + values[-1]) / 2
    selected = select_balanced_stress_scenarios(
        candidates,
        descriptor="visible_point_density",
        bin_edges=(edge,),
        per_bin=1,
        selection_key="f" * 64,
    )
    assert len(selected.scenarios) == 2
    assert all(
        scenario.selection_kind == "stress-balanced" for scenario in selected.scenarios
    )
    assert all(
        0 < scenario.inclusion_probability < 1 for scenario in selected.scenarios
    )
    assert {scenario.selection_stratum for scenario in selected.scenarios} == {
        "visible_point_density:bin-0",
        "visible_point_density:bin-1",
    }
    assert all(
        scenario.selection_design_sha256 == selected.design.sha256
        for scenario in selected.scenarios
    )


def test_stress_lottery_is_replayable_and_rejects_duplicate_candidates(
    tmp_path: Path,
) -> None:
    candidates = generate_iid_scenarios("ci-fixture", 20)
    values = sorted(
        {
            float(dict(scenario.strata_v1)["visible_point_density"])
            for scenario in candidates
        }
    )
    edge = (values[0] + values[-1]) / 2
    kwargs = {
        "descriptor": "visible_point_density",
        "bin_edges": (edge,),
        "per_bin": 1,
        "selection_key": "1" * 64,
    }
    first = select_balanced_stress_scenarios(candidates, **kwargs)
    second = select_balanced_stress_scenarios(tuple(reversed(candidates)), **kwargs)
    assert first == second

    manifest = write_scenario_bank(
        tmp_path,
        "ci-fixture",
        first,
        compression="none",
    )
    loaded = load_scenario_bank(Path(str(manifest["artifact_path"])))
    assert loaded.scenarios == first.scenarios
    assert loaded.selection_design == first.design

    with pytest.raises(ScenarioBankError, match="duplicate states"):
        select_balanced_stress_scenarios(
            (*candidates, candidates[0]),
            **kwargs,
        )


def test_content_addressed_zstd_bank_roundtrip_and_no_overwrite(
    tmp_path: Path,
) -> None:
    scenarios = generate_iid_scenarios("ci-fixture", 2)
    manifest = write_scenario_bank(tmp_path, "ci-fixture", scenarios)
    artifact = Path(str(manifest["artifact_path"]))
    loaded = load_scenario_bank(artifact)

    assert loaded.scenarios == scenarios
    assert loaded.payload_sha256 in artifact.name
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        write_scenario_bank(tmp_path, "ci-fixture", scenarios)


def test_subset_loader_verifies_whole_bank_and_materializes_only_requested(
    tmp_path: Path,
) -> None:
    scenarios = generate_iid_scenarios("ci-fixture", 3)
    manifest = write_scenario_bank(tmp_path, "ci-fixture", scenarios)
    artifact = Path(str(manifest["artifact_path"]))
    loaded = load_scenario_bank_subset(artifact, {scenarios[1].scenario_id})

    assert loaded.scenario_count == 3
    assert loaded.scenarios == (scenarios[1],)
    assert loaded.payload_sha256 == manifest["payload_sha256"]
    with pytest.raises(ScenarioBankError, match="missing requested states"):
        load_scenario_bank_subset(artifact, {"0" * 64})


def test_bank_detects_manifest_and_artifact_tampering(tmp_path: Path) -> None:
    scenarios = generate_iid_scenarios("ci-fixture", 1)
    manifest = write_scenario_bank(
        tmp_path / "manifest", "ci-fixture", scenarios, compression="none"
    )
    artifact = Path(str(manifest["artifact_path"]))
    sidecar = Path(str(manifest["manifest_path"]))
    sidecar.chmod(0o644)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["scenario_count"] = 2
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScenarioBankError, match="row count mismatch"):
        load_scenario_bank(artifact)

    manifest = write_scenario_bank(
        tmp_path / "artifact", "ci-fixture", scenarios, compression="none"
    )
    artifact = Path(str(manifest["artifact_path"]))
    artifact.chmod(0o644)
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ScenarioBankError, match="artifact SHA-256"):
        load_scenario_bank(artifact)


def test_bank_rejects_renaming_and_noncanonical_row_order(tmp_path: Path) -> None:
    scenarios = generate_iid_scenarios("ci-fixture", 2)
    manifest = write_scenario_bank(
        tmp_path / "renamed", "ci-fixture", scenarios, compression="none"
    )
    original_artifact = Path(str(manifest["artifact_path"]))
    original_sidecar = Path(str(manifest["manifest_path"]))
    renamed_artifact = original_artifact.with_name("ci-fixture-" + "0" * 64 + ".jsonl")
    renamed_sidecar = renamed_artifact.with_name(
        f"{renamed_artifact.name}.manifest.json"
    )
    original_artifact.rename(renamed_artifact)
    original_sidecar.rename(renamed_sidecar)
    renamed_sidecar.chmod(0o644)
    renamed_manifest = json.loads(renamed_sidecar.read_text(encoding="utf-8"))
    renamed_manifest["artifact"] = renamed_artifact.name
    renamed_sidecar.write_text(json.dumps(renamed_manifest), encoding="utf-8")
    with pytest.raises(ScenarioBankError, match="content address"):
        load_scenario_bank(renamed_artifact)

    manifest = write_scenario_bank(
        tmp_path / "reordered", "ci-fixture", scenarios, compression="none"
    )
    original_artifact = Path(str(manifest["artifact_path"]))
    original_sidecar = Path(str(manifest["manifest_path"]))
    reordered_payload = (
        b"\n".join(reversed(original_artifact.read_bytes().splitlines())) + b"\n"
    )
    payload_sha256 = hashlib.sha256(reordered_payload).hexdigest()
    reordered_artifact = original_artifact.with_name(
        f"ci-fixture-{payload_sha256}.jsonl"
    )
    reordered_artifact.write_bytes(reordered_payload)
    reordered_sidecar = reordered_artifact.with_name(
        f"{reordered_artifact.name}.manifest.json"
    )
    reordered_manifest = json.loads(original_sidecar.read_text(encoding="utf-8"))
    reordered_manifest.update(
        {
            "artifact": reordered_artifact.name,
            "payload_sha256": payload_sha256,
            "artifact_sha256": payload_sha256,
        }
    )
    reordered_sidecar.write_text(
        json.dumps(reordered_manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ScenarioBankError, match="not canonical JSONL"):
        load_scenario_bank(reordered_artifact)


def test_candidate_freeze_deduplicates_checkpoint_hashes(tmp_path: Path) -> None:
    with pytest.raises(ScenarioBankError, match="deduplicate checkpoints"):
        freeze_candidate_set(
            tmp_path / "duplicate-candidates.json",
            {"official": "a" * 64, "alias": "a" * 64},
            manifest_declaration_sha256="b" * 64,
            frozen_by="reviewer",
        )


def test_sealed_bank_requires_final_freeze_and_appends_one_consumption(
    tmp_path: Path,
) -> None:
    scenarios = generate_iid_scenarios("ci-sealed-fixture", 1)
    bank_manifest = write_scenario_bank(
        tmp_path / "bank",
        "ci-sealed-fixture",
        scenarios,
        compression="none",
    )
    artifact = Path(str(bank_manifest["artifact_path"]))
    with pytest.raises(ScenarioBankError, match="consumption gate"):
        load_scenario_bank(artifact)

    manifest_path = tmp_path / "experiment.json"
    manifest = _manifest(manifest_path, str(bank_manifest["payload_sha256"]))
    freeze_path = tmp_path / "candidate-freeze.json"
    freeze_candidate_set(
        freeze_path,
        {"official": "b" * 64, "ppo-best": "a" * 64},
        manifest_declaration_sha256=str(manifest["declaration_sha256"]),
        frozen_by="selection-reviewer",
    )
    ledger_path = tmp_path / "sealed_test_consumption.jsonl"
    with pytest.raises(ScenarioBankError, match="purpose must be final_report"):
        with consume_sealed_scenario_bank(
            artifact,
            ledger_path,
            purpose="model_selection",
            candidate_freeze_path=freeze_path,
            experiment_manifest_path=manifest_path,
            actor="tester",
        ):
            pass
    assert not ledger_path.exists()

    wrong_manifest_path = tmp_path / "wrong-experiment.json"
    wrong_manifest = _manifest(wrong_manifest_path, "0" * 64)
    wrong_freeze_path = tmp_path / "wrong-candidate-freeze.json"
    freeze_candidate_set(
        wrong_freeze_path,
        {"official": "c" * 64},
        manifest_declaration_sha256=str(wrong_manifest["declaration_sha256"]),
        frozen_by="selection-reviewer",
    )
    with pytest.raises(ScenarioBankError, match=r"not bound.*manifest"):
        with consume_sealed_scenario_bank(
            artifact,
            tmp_path / "wrong-ledger.jsonl",
            purpose=FINAL_REPORT_PURPOSE,
            candidate_freeze_path=wrong_freeze_path,
            experiment_manifest_path=wrong_manifest_path,
            actor="tester",
        ):
            pass

    with consume_sealed_scenario_bank(
        artifact,
        ledger_path,
        purpose=FINAL_REPORT_PURPOSE,
        candidate_freeze_path=freeze_path,
        experiment_manifest_path=manifest_path,
        actor="tester",
    ) as bank:
        assert bank.scenarios == scenarios
    events = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["status"] for event in events] == ["running", "completed"]
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]

    with pytest.raises(ScenarioBankError, match="already has a consumption attempt"):
        with consume_sealed_scenario_bank(
            artifact,
            ledger_path,
            purpose=FINAL_REPORT_PURPOSE,
            candidate_freeze_path=freeze_path,
            experiment_manifest_path=manifest_path,
            actor="tester",
        ):
            pass


def test_consumption_ledger_rejects_terminal_event_before_start(
    tmp_path: Path,
) -> None:
    scenarios = generate_iid_scenarios("ci-sealed-fixture", 1, offset=1)
    bank_manifest = write_scenario_bank(
        tmp_path / "bank",
        "ci-sealed-fixture",
        scenarios,
        compression="none",
    )
    artifact = Path(str(bank_manifest["artifact_path"]))
    manifest_path = tmp_path / "experiment.json"
    manifest = _manifest(manifest_path, str(bank_manifest["payload_sha256"]))
    freeze_path = tmp_path / "candidate-freeze.json"
    freeze = freeze_candidate_set(
        freeze_path,
        {"official": "d" * 64},
        manifest_declaration_sha256=str(manifest["declaration_sha256"]),
        frozen_by="selection-reviewer",
    )
    ledger_path = tmp_path / "sealed_test_consumption.jsonl"
    terminal: dict[str, object] = {
        "schema_version": "splendor-sealed-consumption/1",
        "sequence": 0,
        "previous_event_sha256": None,
        "bank_payload_sha256": bank_manifest["payload_sha256"],
        "bank_artifact_sha256": bank_manifest["artifact_sha256"],
        "candidate_set_sha256": freeze["candidate_set_sha256"],
        "candidate_freeze_sha256": "e" * 64,
        "manifest_declaration_sha256": manifest["declaration_sha256"],
        "purpose": FINAL_REPORT_PURPOSE,
        "actor": "forged",
        "event": "finished",
        "status": "failed",
        "at": "2026-09-15T00:00:00+00:00",
        "error_type": "Forged",
        "error": "terminal first",
    }
    terminal["event_sha256"] = sha256_canonical_json(terminal)
    ledger_path.write_text(
        json.dumps(terminal, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ScenarioBankError, match="finishes before it starts"):
        with consume_sealed_scenario_bank(
            artifact,
            ledger_path,
            purpose=FINAL_REPORT_PURPOSE,
            candidate_freeze_path=freeze_path,
            experiment_manifest_path=manifest_path,
            actor="tester",
        ):
            pass
