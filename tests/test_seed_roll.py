"""Adversarial tests for the Task-1 one-shot seed-roll protocol."""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.policy_imitation import ppo_selfplay as ppo_module
from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest_v2,
    transition_manifest,
    validate_manifest,
)
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    FormalOpponentSpec,
    FormalPPOTrainingJob,
    PPOConfig,
    make_formal_treatment_contract,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    FormalGameRng,
    FormalJobOutput,
    FormalTrainingSpec,
    FormalTreatmentContract,
    make_paired_training_schedule,
    paired_schedule_hash,
    sha256_canonical_json,
)
from splendor.agents.our_agents.policy_imitation.scenario_bank import (
    audit_seed_roll_scenario_bank,
    load_scenario_bank,
    write_seed_roll_scenario_bank,
)
from splendor.agents.our_agents.policy_imitation.seed_roll import (
    AUTHORITY_LEDGER_NAME,
    HISTORICAL_MODEL_SEED_ALIASES,
    SeedRollArtifact,
    SeedRollError,
    SeedRollSelectionDesign,
    build_seed_roll_plan,
    create_confirmatory_activation_artifact,
    create_seed_roll_artifact,
    load_confirmatory_activation_artifact,
    load_seed_roll_artifact,
    make_seed_rolled_training_schedule,
    materialize_seed_roll_scenarios,
    randomization_root_sha256,
    require_seed_roll_binding,
    require_task1_formal_seed_roll,
    validate_scenarios_against_seed_roll,
    validate_seed_roll_binding,
    validate_seed_rolled_training_schedule,
)
from splendor.seed_registry import TASK1_SCENARIO_SPLITS, TASK1_TRAIN_SCHEDULE

ROOT_ZERO = "00" * 32
ROOT_ONE = "01" * 32


def _concurrent_roll_worker(arguments: tuple[str, str, str]) -> dict[str, object]:
    authority, experiment_id, phase = arguments
    return create_seed_roll_artifact(
        Path(authority),
        experiment_id=experiment_id,
        phase=phase,
        replicate_ids=(0,),
        games_per_replicate=2,
    ).manifest_binding()


def _create_roll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_id: str = "task1-seed-roll-test",
    replicate_ids: tuple[int, ...] = (0, 1),
    games_per_replicate: int = 2,
) -> SeedRollArtifact:
    monkeypatch.setattr(
        "splendor.agents.our_agents.policy_imitation.seed_roll.secrets.token_bytes",
        lambda count: bytes.fromhex(ROOT_ZERO) if count == 32 else b"",
    )
    return create_seed_roll_artifact(
        tmp_path / "authority",
        experiment_id=experiment_id,
        phase="T1.4",
        replicate_ids=replicate_ids,
        games_per_replicate=games_per_replicate,
    )


def _task1_segments() -> dict[str, str]:
    return {
        logical_split: segment.name
        for logical_split, segment in TASK1_SCENARIO_SPLITS.items()
    }


def _activation_training_spec(  # noqa: PLR0913 - protocol axes explicit
    artifact: SeedRollArtifact,
    tmp_path: Path,
    *,
    replicate_ids: tuple[int, ...],
    stage: str,
    activation_path: Path | None = None,
    activation_sha256: str | None = None,
) -> FormalTrainingSpec:
    scenarios = {
        replicate_id: tuple(
            f"scenario-{replicate_id}-{game_index}" for game_index in range(2)
        )
        for replicate_id in replicate_ids
    }
    seats = {
        replicate_id: artifact.plan.replicate(replicate_id).seats
        for replicate_id in replicate_ids
    }
    schedule = make_paired_training_schedule(
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        treatment_ids=("O",),
        scenarios_by_replicate=scenarios,
        seats_by_replicate=seats,
        updates=1,
        games_per_update=2,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
    )
    return FormalTrainingSpec(
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        replicate_id=replicate_ids[0],
        treatment_id="O",
        expected_treatments=("O",),
        schedule=schedule,
        scenario_bank_sha256="b" * 64,
        seed_roll_payload_sha256=artifact.payload_sha256,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
        replicate_stage=stage,  # type: ignore[arg-type]
        activation_artifact_path=(
            str(activation_path) if activation_path is not None else None
        ),
        activation_artifact_sha256=activation_sha256,
        treatment_contracts=(
            FormalTreatmentContract(
                treatment_id="O",
                initial_checkpoint_sha256="c" * 64,
                trainer_config_sha256="d" * 64,
                opponent_pool_sha256="e" * 64,
            ),
        ),
        job_outputs=tuple(
            FormalJobOutput(
                replicate_id=replicate_id,
                treatment_id="O",
                output_dir=str(
                    (tmp_path / "outputs" / f"{stage}-{replicate_id}").resolve()
                ),
            )
            for replicate_id in replicate_ids
        ),
    )


def _create_formal_manifest(
    path: Path,
    artifact: SeedRollArtifact,
    spec: FormalTrainingSpec,
) -> dict[str, object]:
    artifact_contract: dict[str, object] = {
        "scenario_banks": {"train-schedule": "b" * 64},
        "schedule": "training_schedule.jsonl.zst",
    }
    if spec.activation_artifact_path is not None:
        artifact_contract["confirmatory_activation_path"] = (
            spec.activation_artifact_path
        )
        artifact_contract["confirmatory_activation_sha256"] = (
            spec.activation_artifact_sha256
        )
    return create_manifest_v2(
        path,
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        purpose="confirmatory activation protocol test",
        seed_segments=_task1_segments(),
        budget={"updates": 1, "games_per_update": 2},
        hypotheses={"H": "protocol-only"},
        estimands={"d": "paired contrast"},
        decision_rule={"selection": "pre-registered"},
        artifact_contract=artifact_contract,
        baselines={"ppo-best": {"sha256": "a" * 64}},
        seed_roll=artifact.manifest_binding(),
        formal_training=spec.manifest_binding(),
        repo=Path(__file__).resolve().parents[1],
    )


def test_seed_roll_known_vector_prefix_stability_and_domain_separation() -> None:
    three = build_seed_roll_plan(
        experiment_id="task1-seed-roll-known-vector",
        phase="T1.4",
        replicate_ids=(0, 1, 2),
        games_per_replicate=4,
        randomization_root_hex=ROOT_ZERO,
    )
    five = build_seed_roll_plan(
        experiment_id="task1-seed-roll-known-vector",
        phase="T1.4",
        replicate_ids=(0, 1, 2, 3, 4),
        games_per_replicate=4,
        randomization_root_hex=ROOT_ZERO,
    )
    changed_root = build_seed_roll_plan(
        experiment_id="task1-seed-roll-known-vector",
        phase="T1.4",
        replicate_ids=(0, 1, 2),
        games_per_replicate=4,
        randomization_root_hex=ROOT_ONE,
    )

    assert five.payload_sha256 == (
        "bc62b1b0171bf51354f8a5af6c1a4cc69ec82fcc7732cd3c716eb66b66cf7118"
    )
    assert five.replicate(0).source_seeds == (
        1_264_584,
        1_191_316,
        1_302_346,
        1_207_795,
    )
    assert five.replicate(0).seats == (0, 0, 1, 1)
    assert five.replicate(0).model_init_lineage.seed63 == 7_880_272_485_446_773_790
    assert five.design_profile == "ci-fixture"
    assert three.replicates == five.replicates[:3]
    assert changed_root.replicates != three.replicates
    assert five.replicate(0).source_seeds != tuple(
        range(TASK1_TRAIN_SCHEDULE.start, TASK1_TRAIN_SCHEDULE.start + 4)
    )

    all_seeds = [seed for item in five.replicates for seed in item.source_seeds]
    assert len(all_seeds) == len(set(all_seeds))
    assert all(
        item.seats.count(0) == item.seats.count(1) == 2 for item in five.replicates
    )
    model_seeds = {item.model_init_lineage.seed63 for item in five.replicates}
    assert len(model_seeds) == 5
    assert not model_seeds & HISTORICAL_MODEL_SEED_ALIASES


def test_formal_scale_reserves_five_disjoint_balanced_blocks() -> None:
    plan = build_seed_roll_plan(
        experiment_id="task1-t14-formal-design-test",
        phase="T1.4",
        replicate_ids=(0, 1, 2, 3, 4),
        games_per_replicate=32_000,
        randomization_root_hex=ROOT_ZERO,
    )
    all_seeds = [seed for item in plan.replicates for seed in item.source_seeds]
    assert len(all_seeds) == len(set(all_seeds)) == 160_000
    assert all(len(item.source_seeds) == 32_000 for item in plan.replicates)
    assert all(
        item.seats.count(0) == item.seats.count(1) == 16_000 for item in plan.replicates
    )
    assert plan.pilot_replicate_ids == (0, 1, 2)
    assert plan.confirmatory_reserve_replicate_ids == (3, 4)
    assert plan.design_profile == "task1-formal-5x32000"
    pilot_design = SeedRollSelectionDesign(
        seed_roll_payload_sha256=plan.payload_sha256,
        randomization_root_sha256=plan.randomization_root_sha256,
        active_replicate_ids=(0, 1, 2),
        games_per_replicate=32_000,
        source_population=200_000,
    )
    reserved_design = replace(
        pilot_design,
        active_replicate_ids=(0, 1, 2, 3, 4),
    )
    assert pilot_design.inclusion_probability == 0.48
    assert pilot_design.per_replicate_inclusion_probability == 0.16
    assert reserved_design.inclusion_probability == 0.8


def test_roll_derivation_does_not_mutate_global_rng_state() -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    build_seed_roll_plan(
        experiment_id="task1-global-rng-isolation",
        phase="T1.4",
        replicate_ids=(0,),
        games_per_replicate=2,
        randomization_root_hex=ROOT_ZERO,
    )

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_seed_rolled_formal_streams_have_fixed_domains_and_axes() -> None:
    context = FormalGameRng(
        experiment_id="task1-stream-gate",
        phase="T1.4",
        coupling_group="replicate-0",
        replicate_id=0,
        treatment_id="O",
        scenario_id="scenario-0",
        seat=0,
        update=1,
        game_index=0,
        randomization_root_sha256=randomization_root_sha256(ROOT_ZERO),
    )
    assert context.lineage("policy_action", focal_step=0).digest_hex
    assert context.lineage(
        "opponent_action",
        opponent_id="random",
        opponent_step=0,
    ).digest_hex
    with pytest.raises(ValueError, match="unsupported formal game RNG stream"):
        context.lineage("minibatch")
    with pytest.raises(ValueError, match="requires only focal_step"):
        context.lineage("policy_action")
    with pytest.raises(ValueError, match="irrelevant event axes"):
        context.lineage("pool_draw", focal_step=0)


def test_t14_formal_training_cannot_fall_back_to_protocol_v1() -> None:
    schedule = make_paired_training_schedule(
        experiment_id="task1-no-v1-fallback",
        phase="T1.4",
        treatment_ids=("O",),
        scenarios_by_replicate={0: ("scenario-0", "scenario-1")},
        seats_by_replicate={0: (0, 1)},
        updates=1,
        games_per_update=2,
    )
    legacy = FormalTrainingSpec(
        experiment_id="task1-no-v1-fallback",
        phase="T1.4",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O",),
        schedule=schedule,
    )
    with pytest.raises(RuntimeError, match="requires paired-training-v2"):
        ppo_module._require_formal_seed_roll_manifest({}, legacy, None)  # noqa: SLF001

    aliased_schedule = make_paired_training_schedule(
        experiment_id="task1-no-v1-fallback",
        phase="T1.4 ",
        treatment_ids=("O",),
        scenarios_by_replicate={0: ("scenario-0", "scenario-1")},
        seats_by_replicate={0: (0, 1)},
        updates=1,
        games_per_update=2,
    )
    with pytest.raises(ValueError, match="canonical identifier"):
        FormalTrainingSpec(
            experiment_id="task1-no-v1-fallback",
            phase="T1.4 ",
            replicate_id=0,
            treatment_id="O",
            expected_treatments=("O",),
            schedule=aliased_schedule,
        )


def test_confirmatory_activation_binds_completed_pilot_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _create_roll(
        tmp_path,
        monkeypatch,
        experiment_id="task1-activation-test",
        replicate_ids=(0, 1, 2, 3, 4),
    )
    pilot = _activation_training_spec(
        artifact,
        tmp_path,
        replicate_ids=(0, 1, 2),
        stage="pilot",
    )
    pilot_manifest_path = (tmp_path / "pilot-manifest.json").resolve()
    _create_formal_manifest(pilot_manifest_path, artifact, pilot)
    approve_manifest(pilot_manifest_path, "pilot-reviewer", "pilot approved")
    transition_manifest(
        pilot_manifest_path,
        "running",
        actor="pilot-runner",
        note="pilot started",
    )
    transition_manifest(
        pilot_manifest_path,
        "completed",
        actor="pilot-reviewer",
        note="pilot evidence reviewed",
    )
    evidence_path = (tmp_path / "pilot-evidence.json").resolve()
    evidence_bytes = b'{"decision_inputs":"frozen-pilot-results","version":1}\n'
    evidence_path.write_bytes(evidence_bytes)
    activation_path = (tmp_path / "confirmatory-activation.json").resolve()
    activation = create_confirmatory_activation_artifact(
        activation_path,
        seed_roll=artifact,
        pilot_manifest_path=pilot_manifest_path,
        pilot_evidence_path=evidence_path,
        actor="independent-reviewer",
        note="pre-registered reserve activation criteria passed",
    )
    assert activation.pilot_replicate_ids == (0, 1, 2)
    assert activation.confirmatory_replicate_ids == (3, 4)
    assert activation.path == activation_path
    with pytest.raises(SeedRollError):
        create_confirmatory_activation_artifact(
            activation_path,
            seed_roll=artifact,
            pilot_manifest_path=pilot_manifest_path,
            pilot_evidence_path=evidence_path,
            actor="second-reviewer",
            note="must not overwrite the activation decision",
        )

    evidence_path.write_bytes(b"tampered pilot evidence\n")
    with pytest.raises(SeedRollError, match="evidence changed"):
        load_confirmatory_activation_artifact(
            activation_path,
            expected_experiment_id=artifact.plan.experiment_id,
            expected_phase=artifact.plan.phase,
            expected_seed_roll_payload_sha256=artifact.payload_sha256,
            expected_pilot_replicate_ids=(0, 1, 2),
            expected_confirmatory_replicate_ids=(3, 4),
        )
    evidence_path.write_bytes(evidence_bytes)

    reserve = _activation_training_spec(
        artifact,
        tmp_path,
        replicate_ids=(3, 4),
        stage="confirmatory-reserve",
        activation_path=activation.path,
        activation_sha256=activation.artifact_sha256,
    )
    reserve_manifest_path = (tmp_path / "reserve-manifest.json").resolve()
    reserve_manifest = _create_formal_manifest(
        reserve_manifest_path,
        artifact,
        reserve,
    )
    validate_manifest(reserve_manifest)
    ppo_module._require_formal_seed_roll_manifest(  # noqa: SLF001
        reserve_manifest["declaration"],  # type: ignore[arg-type]
        reserve,
        artifact,
    )

    evidence_path.write_bytes(b"post-approval mutation\n")
    with pytest.raises(SeedRollError, match="evidence changed"):
        validate_manifest(reserve_manifest)


def test_authority_draws_once_and_rejects_reroll_or_design_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = iter((bytes.fromhex(ROOT_ZERO), bytes.fromhex(ROOT_ONE)))
    calls: list[int] = []

    def fake_root(count: int) -> bytes:
        calls.append(count)
        return next(roots)

    monkeypatch.setattr(
        "splendor.agents.our_agents.policy_imitation.seed_roll.secrets.token_bytes",
        fake_root,
    )
    authority = tmp_path / "authority"
    first = create_seed_roll_artifact(
        authority,
        experiment_id="one-shot",
        phase="T1.4",
        replicate_ids=(0, 1),
        games_per_replicate=2,
    )
    repeated = create_seed_roll_artifact(
        authority,
        experiment_id="one-shot",
        phase="T1.4",
        replicate_ids=(0, 1),
        games_per_replicate=2,
    )
    assert calls == [32]
    assert repeated.manifest_binding() == first.manifest_binding()
    assert repeated.path == first.path

    with pytest.raises(SeedRollError, match="another design"):
        create_seed_roll_artifact(
            authority,
            experiment_id="one-shot",
            phase="T1.4",
            replicate_ids=(0,),
            games_per_replicate=2,
        )
    assert calls == [32]

    ledger_lines = (authority / AUTHORITY_LEDGER_NAME).read_bytes().splitlines()
    assert len(ledger_lines) == 1
    assert (
        load_seed_roll_artifact(first.path).manifest_binding()
        == first.manifest_binding()
    )


def test_crash_after_identity_reservation_blocks_a_second_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def crash(_count: int) -> bytes:
        calls.append("first")
        raise RuntimeError("simulated entropy-source interruption")

    monkeypatch.setattr(
        "splendor.agents.our_agents.policy_imitation.seed_roll.secrets.token_bytes",
        crash,
    )
    authority = tmp_path / "authority"
    with pytest.raises(RuntimeError, match="interruption"):
        create_seed_roll_artifact(
            authority,
            experiment_id="crash-safe-roll",
            phase="T1.4",
            replicate_ids=(0,),
            games_per_replicate=2,
        )

    def forbidden_redraw(_count: int) -> bytes:
        calls.append("second")
        return bytes.fromhex(ROOT_ONE)

    monkeypatch.setattr(
        "splendor.agents.our_agents.policy_imitation.seed_roll.secrets.token_bytes",
        forbidden_redraw,
    )
    with pytest.raises(SeedRollError, match="reservation"):
        create_seed_roll_artifact(
            authority,
            experiment_id="crash-safe-roll",
            phase="T1.4",
            replicate_ids=(0,),
            games_per_replicate=2,
        )
    assert calls == ["first"]


def test_concurrent_authority_calls_publish_one_identical_roll(tmp_path: Path) -> None:
    arguments = (str(tmp_path / "authority"), "concurrent-roll", "T1.4")
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        results = tuple(pool.map(_concurrent_roll_worker, (arguments, arguments)))
    assert results[0] == results[1]
    ledger = tmp_path / "authority" / AUTHORITY_LEDGER_NAME
    assert len(ledger.read_bytes().splitlines()) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("randomization_root_hex", ROOT_ONE),
        ("authority_dir", "/tmp/alternate-seed-roll-authority"),
        ("payload_sha256", "f" * 64),
        ("authority_event_sha256", "e" * 64),
        ("source_range", [TASK1_TRAIN_SCHEDULE.start + 1, TASK1_TRAIN_SCHEDULE.end]),
        ("replicate_ids", [0]),
        ("model_seed63_by_replicate", {"0": 42, "1": 1234}),
    ],
)
def test_manifest_binding_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    artifact = _create_roll(tmp_path, monkeypatch)
    forged = artifact.manifest_binding()
    forged[field] = replacement
    with pytest.raises(SeedRollError):
        validate_seed_roll_binding(forged)


def test_artifact_and_authority_tampering_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _create_roll(tmp_path, monkeypatch)
    artifact.path.chmod(0o644)
    raw = json.loads(artifact.path.read_text(encoding="utf-8"))
    raw["replicates"][0]["seats"] = [1, 1]
    artifact.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SeedRollError):
        load_seed_roll_artifact(artifact.path)

    second = _create_roll(
        tmp_path / "second",
        monkeypatch,
        experiment_id="authority-tamper",
    )
    ledger = second.authority_dir / AUTHORITY_LEDGER_NAME
    ledger.write_bytes(ledger.read_bytes().replace(b"roll-created", b"roll-forged!"))
    with pytest.raises(SeedRollError):
        load_seed_roll_artifact(second.path)

    third = _create_roll(
        tmp_path / "third",
        monkeypatch,
        experiment_id="reservation-tamper",
    )
    reservation = next((third.authority_dir / "reservations").glob("*.json"))
    reservation.chmod(0o600)
    reservation.write_bytes(
        reservation.read_bytes().replace(ROOT_ZERO.encode(), ROOT_ONE.encode())
    )
    with pytest.raises(SeedRollError, match="reservation"):
        load_seed_roll_artifact(third.path)


def test_authority_and_artifact_symlinks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _create_roll(tmp_path / "real", monkeypatch)

    authority_alias = tmp_path / "authority-alias"
    authority_alias.symlink_to(artifact.authority_dir, target_is_directory=True)
    with pytest.raises(SeedRollError, match="symlink"):
        create_seed_roll_artifact(
            authority_alias,
            experiment_id="task1-seed-roll-test",
            phase="T1.4",
            replicate_ids=(0, 1),
            games_per_replicate=2,
        )

    artifact_alias = tmp_path / "artifact-alias.json"
    artifact_alias.symlink_to(artifact.path)
    with pytest.raises(SeedRollError, match="symlink"):
        load_seed_roll_artifact(artifact_alias, authority_dir=artifact.authority_dir)

    poisoned = tmp_path / "poisoned-authority"
    poisoned.mkdir()
    (poisoned / "reservations").symlink_to(
        artifact.authority_dir / "reservations",
        target_is_directory=True,
    )
    with pytest.raises(SeedRollError, match="symlink"):
        create_seed_roll_artifact(
            poisoned,
            experiment_id="poisoned-roll",
            phase="T1.4",
            replicate_ids=(0,),
            games_per_replicate=2,
        )

    retry = _create_roll(
        tmp_path / "retry",
        monkeypatch,
        experiment_id="retry-symlink-roll",
    )
    artifact_directory = retry.authority_dir / "artifacts"
    external_artifacts = tmp_path / "external-artifacts"
    artifact_directory.rename(external_artifacts)
    artifact_directory.symlink_to(external_artifacts, target_is_directory=True)
    with pytest.raises(SeedRollError, match="symlink"):
        create_seed_roll_artifact(
            retry.authority_dir,
            experiment_id="retry-symlink-roll",
            phase="T1.4",
            replicate_ids=(0, 1),
            games_per_replicate=2,
        )


def test_rolled_bank_source_replay_schedule_and_manifest_binding(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _create_roll(
        tmp_path,
        monkeypatch,
        experiment_id="task1-roll-integration",
    )
    with pytest.raises(SeedRollError, match="5 x 32,000"):
        require_task1_formal_seed_roll(artifact)
    rolled = materialize_seed_roll_scenarios(artifact, (0, 1))
    bank_result = write_seed_roll_scenario_bank(
        tmp_path / "banks",
        artifact,
        (0, 1),
        compression="none",
    )
    bank = load_scenario_bank(Path(str(bank_result["artifact_path"])))
    audit = audit_seed_roll_scenario_bank(
        Path(str(bank_result["artifact_path"])), artifact, (0, 1)
    )
    assert audit["payload_sha256"] == bank.payload_sha256
    validate_scenarios_against_seed_roll(
        artifact,
        bank.scenarios,
        (0, 1),
        selection_design=bank.selection_design,  # type: ignore[arg-type]
    )

    rows = make_seed_rolled_training_schedule(
        artifact,
        bank.scenarios,
        active_replicate_ids=(0, 1),
        treatment_ids=("O", "O_bridge"),
        updates=1,
        games_per_update=2,
        selection_design=rolled.design,
    )
    assert len(rows) == 8
    assert {row.randomization_root_sha256 for row in rows} == {
        artifact.plan.randomization_root_sha256
    }
    assert paired_schedule_hash(rows, treatment_id="O") == paired_schedule_hash(
        rows,
        treatment_id="O_bridge",
    )
    validate_seed_rolled_training_schedule(
        artifact,
        bank.scenarios,
        rows,
        (0, 1),
        expected_treatments=("O", "O_bridge"),
        selection_design=rolled.design,
    )
    scenarios_by_replicate = {
        replicate_id: tuple(
            next(
                scenario.scenario_id
                for scenario in bank.scenarios
                if scenario.source_seed == source_seed
            )
            for source_seed in reversed(
                artifact.plan.replicate(replicate_id).source_seeds
            )
        )
        for replicate_id in (0, 1)
    }
    forged_rows = make_paired_training_schedule(
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        treatment_ids=("O", "O_bridge"),
        scenarios_by_replicate=scenarios_by_replicate,
        seats_by_replicate={
            replicate_id: artifact.plan.replicate(replicate_id).seats
            for replicate_id in (0, 1)
        },
        updates=1,
        games_per_update=2,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
    )
    # Generic CRN checks cannot know the roll's source-seed ordinal.  The
    # formal seed-roll gate must reject this fully self-consistent relabeling.
    forged_spec = FormalTrainingSpec(
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O", "O_bridge"),
        schedule=forged_rows,
        scenario_bank_sha256=bank.payload_sha256,
        seed_roll_payload_sha256=artifact.payload_sha256,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
        replicate_stage="pilot",
        treatment_contracts=tuple(
            FormalTreatmentContract(
                treatment_id=treatment_id,
                initial_checkpoint_sha256="a" * 64,
                trainer_config_sha256="b" * 64,
                opponent_pool_sha256="c" * 64,
            )
            for treatment_id in ("O", "O_bridge")
        ),
        job_outputs=tuple(
            FormalJobOutput(
                replicate_id=replicate_id,
                treatment_id=treatment_id,
                output_dir=str(
                    (
                        tmp_path / "forged-outputs" / f"{replicate_id}-{treatment_id}"
                    ).resolve()
                ),
            )
            for replicate_id in (0, 1)
            for treatment_id in ("O", "O_bridge")
        ),
    )
    assert forged_spec.schedule == forged_rows
    with pytest.raises(SeedRollError, match="scenario/seat order"):
        validate_seed_rolled_training_schedule(
            artifact,
            bank.scenarios,
            forged_rows,
            (0, 1),
            expected_treatments=("O", "O_bridge"),
            selection_design=rolled.design,
        )
    with pytest.raises(ValueError, match="schedule scenario/seat order"):
        audit_seed_roll_scenario_bank(
            Path(str(bank_result["artifact_path"])),
            artifact,
            (0, 1),
            schedule=forged_rows,
            expected_treatments=("O", "O_bridge"),
        )
    for replicate_id in (0, 1):
        selected = sorted(
            (
                row
                for row in rows
                if row.replicate_id == replicate_id and row.treatment_id == "O"
            ),
            key=lambda row: (row.update, row.game_index),
        )
        assert (
            tuple(row.seat for row in selected)
            == artifact.plan.replicate(replicate_id).seats
        )

    tampered = list(bank.scenarios)
    tampered[0] = replace(tampered[0], source_seed=tampered[0].source_seed + 1)
    with pytest.raises(SeedRollError, match="registered source seed"):
        validate_scenarios_against_seed_roll(
            artifact,
            tampered,
            (0, 1),
            selection_design=rolled.design,
        )
    with pytest.raises(SeedRollError, match="source population"):
        replace(
            rolled.design,
            source_population=rolled.design.source_population + 1,
        )
    invalid_metadata = list(bank.scenarios)
    invalid_metadata[0] = replace(
        invalid_metadata[0],
        source_segment="forged-source-segment",
    )
    with pytest.raises(SeedRollError, match=r"invalid|metadata"):
        validate_scenarios_against_seed_roll(
            artifact,
            invalid_metadata,
            (0, 1),
            selection_design=rolled.design,
        )

    opponent_pool = (FormalOpponentSpec("random", "random", 1.0),)
    initial_bc = tmp_path / "initial-bc.pth"
    initial_bc.write_bytes(b"test-only-checkpoint-bytes")
    with pytest.raises(ValueError, match="unused sentinel 0"):
        make_formal_treatment_contract(
            "O",
            initial_bc=initial_bc,
            config=PPOConfig(updates=1, games_per_update=2, hidden_layers=(8,)),
            opponent_pool=opponent_pool,
        )
    config = PPOConfig(
        updates=1,
        games_per_update=2,
        hidden_layers=(8,),
        seed=0,
        device_name="cpu",
    )
    base_spec = FormalTrainingSpec(
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O", "O_bridge"),
        schedule=rows,
        scenario_bank_sha256=bank.payload_sha256,
        seed_roll_payload_sha256=artifact.payload_sha256,
        randomization_root_sha256=artifact.plan.randomization_root_sha256,
        replicate_stage="pilot",
        treatment_contracts=tuple(
            make_formal_treatment_contract(
                treatment_id,
                initial_bc=initial_bc,
                config=config,
                opponent_pool=opponent_pool,
            )
            for treatment_id in ("O", "O_bridge")
        ),
        job_outputs=tuple(
            FormalJobOutput(
                replicate_id=replicate_id,
                treatment_id=treatment_id,
                output_dir=str(
                    (
                        tmp_path / "training" / f"r{replicate_id}-{treatment_id}"
                    ).resolve()
                ),
            )
            for replicate_id in (0, 1)
            for treatment_id in ("O", "O_bridge")
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = create_manifest_v2(
        manifest_path,
        experiment_id=artifact.plan.experiment_id,
        phase=artifact.plan.phase,
        purpose="seed-roll integration test",
        seed_segments=_task1_segments(),
        budget={"updates": 1, "games_per_update": 2, "replicates": 2},
        hypotheses={"H": "protocol-only integration"},
        estimands={"d": "paired O minus O_bridge"},
        decision_rule={"selection": "none"},
        artifact_contract={
            "scenario_banks": {"train-schedule": bank.payload_sha256},
            "schedule": "training_schedule.jsonl.zst",
        },
        baselines={"ppo-best": {"sha256": "a" * 64}},
        seed_roll=artifact.manifest_binding(),
        formal_training=base_spec.manifest_binding(),
        repo=Path(__file__).resolve().parents[1],
    )
    validate_manifest(manifest)
    approve_manifest(manifest_path, "test-reviewer", "seed roll reviewed")
    running = transition_manifest(
        manifest_path,
        "running",
        actor="test-runner",
        note="protocol test start",
    )
    assert running["status"] == "running"
    require_seed_roll_binding(
        artifact,
        running["declaration"]["seed_plan"]["seed_roll"],
    )
    ppo_module._require_formal_training_manifest(  # noqa: SLF001
        str(manifest_path),
        base_spec,
        artifact,
    )
    job = FormalPPOTrainingJob(
        initial_bc=initial_bc,
        output_dir=tmp_path / "training" / "r0-O",
        source_manifest=manifest_path,
        config=config,
        formal_spec=base_spec,
        opponent_pool=opponent_pool,
        scenario_bank_path=bank.artifact_path,
        seed_roll_path=artifact.path,
    )
    with pytest.raises(ValueError, match="config"):
        replace(job, config=replace(config, learning_rate=1e-2))
    with pytest.raises(ValueError, match="output directory"):
        replace(job, output_dir=tmp_path / "training" / "alternate")
    initial_bc.write_bytes(b"mutated-checkpoint")
    with pytest.raises(ValueError, match="initial checkpoint"):
        replace(job)
    initial_bc.write_bytes(b"test-only-checkpoint-bytes")
    ppo_module._reserve_formal_output_dirs((job,))  # noqa: SLF001
    ppo_module._require_formal_output_reservation(job)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="already attempted"):
        ppo_module._reserve_formal_output_dirs((job,))  # noqa: SLF001


def test_invalid_roots_and_noncanonical_replicates_are_rejected() -> None:
    for root in ("0" * 63, "A" * 64, "gg" * 32):
        with pytest.raises(SeedRollError):
            randomization_root_sha256(root)
    for replicate_ids in ((42,), (0, 2), ("0",)):  # type: ignore[list-item]
        with pytest.raises(SeedRollError):
            build_seed_roll_plan(
                experiment_id="invalid-roll",
                phase="T1.4",
                replicate_ids=replicate_ids,
                games_per_replicate=2,
                randomization_root_hex=ROOT_ZERO,
            )
    with pytest.raises(SeedRollError, match="canonical identifier"):
        build_seed_roll_plan(
            experiment_id="invalid-roll",
            phase="T1.4 ",
            replicate_ids=(0,),
            games_per_replicate=2,
            randomization_root_hex=ROOT_ZERO,
        )
    with pytest.raises(ValueError, match="required"):
        FormalTreatmentContract(
            treatment_id="O",
            initial_checkpoint_sha256=None,  # type: ignore[arg-type]
            trainer_config_sha256="a" * 64,
            opponent_pool_sha256="b" * 64,
        )


def test_roll_binding_hash_is_not_repairable_by_one_local_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _create_roll(tmp_path, monkeypatch)
    forged = artifact.manifest_binding()
    forged["source_seed_union_sha256"] = "d" * 64
    forged["payload_sha256"] = sha256_canonical_json(forged)
    with pytest.raises(SeedRollError, match="root-derived plan"):
        validate_seed_roll_binding(forged)


def test_manifest_binding_rejects_a_self_consistent_forged_authority_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _create_roll(tmp_path, monkeypatch)
    forged = artifact.manifest_binding()
    event = dict(forged["authority_event"])  # type: ignore[arg-type]
    request = dict(event["request"])  # type: ignore[arg-type]
    request["games_per_replicate"] = 4
    event["request"] = request
    event["request_sha256"] = sha256_canonical_json(request)
    event["event_sha256"] = sha256_canonical_json(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    forged["authority_event"] = event
    forged["authority_event_sha256"] = event["event_sha256"]

    with pytest.raises(SeedRollError, match="authority event"):
        validate_seed_roll_binding(forged)

    forged = artifact.manifest_binding()
    event = dict(forged["authority_event"])  # type: ignore[arg-type]
    event["at"] = "2030-01-01T00:00:00+00:00"
    event["event_sha256"] = sha256_canonical_json(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    forged["authority_event"] = event
    forged["authority_event_sha256"] = event["event_sha256"]
    with pytest.raises(SeedRollError, match="authority"):
        validate_seed_roll_binding(forged)
