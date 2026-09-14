"""Task-1 tests for paired treatment schedules and formal PPO streams."""

import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import numpy as np
import pytest
import torch
from torch import optim

from splendor.agents.generic.first_move import FirstActionAgent
from splendor.agents.generic.random import RandomAgent
from splendor.agents.our_agents.policy_imitation import manifest as manifest_module
from splendor.agents.our_agents.policy_imitation import ppo_selfplay as ppo_module
from splendor.agents.our_agents.policy_imitation import protocol as protocol_module
from splendor.agents.our_agents.policy_imitation.bc_network import (
    BehaviorCloningNetwork,
)
from splendor.agents.our_agents.policy_imitation.bc_training import (
    BCConfig,
    save_bc_checkpoint,
)
from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest_v2,
    transition_manifest,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    FormalOpponentSpec,
    FormalPPOTrainingJob,
    OpponentPoolEntry,
    PolicyValueNetwork,
    PPOConfig,
    PPOTransition,
    _policy_step,
    collect_ppo_game,
    ppo_update,
    run_formal_ppo_training_jobs,
    train_ppo_selfplay,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    FormalGameRng,
    FormalTrainingSpec,
    PairedTrainingRow,
    RngKey,
    derive_seed,
    make_paired_training_schedule,
    paired_schedule_hash,
    validate_paired_training_schedule,
)
from splendor.agents.our_agents.policy_imitation.runner import select_action
from splendor.agents.our_agents.policy_imitation.scenario import (
    ScenarioV1,
    generate_scenario,
)
from splendor.agents.our_agents.policy_imitation.scenario_bank import (
    ScenarioBank,
    load_scenario_bank,
    write_scenario_bank,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent


class _FailingAgent(Agent):
    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del actions, game_state, game_rule
        raise RuntimeError("intentional formal probe failure")


def _schedule() -> tuple[PairedTrainingRow, ...]:
    scenarios = {
        0: tuple(f"r0-s{index}" for index in range(4)),
        1: tuple(f"r1-s{index}" for index in range(4)),
    }
    seats = {0: (0, 1, 1, 0), 1: (1, 0, 0, 1)}
    return make_paired_training_schedule(
        experiment_id="task1",
        phase="T1.1",
        treatment_ids=("O", "A", "B", "C"),
        scenarios_by_replicate=scenarios,
        seats_by_replicate=seats,
        updates=2,
        games_per_update=2,
    )


def test_treatments_share_schedule_but_replicates_are_independent() -> None:
    rows = _schedule()
    hashes = {
        treatment: paired_schedule_hash(rows, treatment_id=treatment)
        for treatment in ("O", "A", "B", "C")
    }
    assert len(set(hashes.values())) == 1

    by_treatment = {
        treatment: [row for row in rows if row.treatment_id == treatment]
        for treatment in hashes
    }
    for ordinal in range(8):
        digests = {
            by_treatment[treatment][ordinal].pool_draw.digest_hex
            for treatment in by_treatment
        }
        assert len(digests) == 1
    assert (
        by_treatment["O"][0].pool_draw.digest_hex
        != by_treatment["O"][4].pool_draw.digest_hex
    )
    minibatch_digests = {
        derive_seed(
            RngKey(
                stream_name="minibatch",
                experiment_id="task1",
                phase="T1.1",
                coupling_group="replicate-0",
                replicate_id=0,
                treatment_id=treatment,
                update=1,
                epoch=0,
            )
        ).digest_hex
        for treatment in ("O", "A", "B", "C")
    }
    assert len(minibatch_digests) == 1


def test_schedule_hash_ignores_completion_order_and_worker_partition() -> None:
    rows = _schedule()
    original = paired_schedule_hash(rows, treatment_id="A")
    completion_order = tuple(reversed(rows[::2])) + tuple(reversed(rows[1::2]))
    assert paired_schedule_hash(completion_order, treatment_id="A") == original


def test_schedule_rejects_duplicates_cross_replicate_reuse_and_tampering() -> None:
    rows = _schedule()
    with pytest.raises(ValueError, match="duplicate row key"):
        validate_paired_training_schedule((*rows, rows[0]))
    reused = tuple(
        replace(row, scenario_id="r0-s0")
        if row.replicate_id == 1 and row.game_index == 0 and row.update == 1
        else row
        for row in rows
    )
    with pytest.raises(ValueError, match="crosses replicates"):
        validate_paired_training_schedule(reused)
    tampered = (replace(rows[0], seat=1 - rows[0].seat), *rows[1:])
    with pytest.raises(ValueError, match="CRN semantics"):
        validate_paired_training_schedule(tampered)

    repeated_scenario: list[PairedTrainingRow] = []
    for row in rows:
        if row.replicate_id == 0 and row.update == 1 and row.game_index == 1:
            context = replace(row.game_rng(), scenario_id="r0-s0")
            repeated_scenario.append(
                replace(
                    row,
                    scenario_id="r0-s0",
                    scenario_source=context.lineage("scenario_source"),
                    pool_draw=context.lineage("pool_draw"),
                )
            )
        else:
            repeated_scenario.append(row)
    with pytest.raises(ValueError, match="repeats a training scenario"):
        validate_paired_training_schedule(repeated_scenario)

    mixed_groups = tuple(
        replace(row, coupling_group="coordinate-specific")
        if row.replicate_id == 0 and row.update == 1 and row.game_index == 1
        else row
        for row in rows
    )
    with pytest.raises(ValueError, match="must use coupling group"):
        validate_paired_training_schedule(mixed_groups)


def test_formal_training_spec_binds_identity_budget_and_model_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _schedule()
    spec_o = FormalTrainingSpec(
        experiment_id="task1",
        phase="T1.1",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O", "A", "B", "C"),
        schedule=rows,
    )
    spec_a = replace(spec_o, treatment_id="A")
    assert len(spec_o.selected_rows(updates=2, games_per_update=2)) == 4
    assert (
        spec_o.model_init_lineage().digest_hex == spec_a.model_init_lineage().digest_hex
    )
    assert (
        derive_seed(spec_o.minibatch_key(1)).digest_hex
        == derive_seed(spec_a.minibatch_key(1)).digest_hex
    )
    with pytest.raises(ValueError, match="update budget"):
        spec_o.selected_rows(updates=3, games_per_update=2)

    multi_worker = replace(spec_o, worker_count=3)
    partitions = multi_worker.worker_partitions(updates=2, games_per_update=2)
    reordered = tuple(row for partition in reversed(partitions) for row in partition)
    assert paired_schedule_hash(reordered, treatment_id="O") == paired_schedule_hash(
        spec_o.selected_rows(updates=2, games_per_update=2),
        treatment_id="O",
    )
    assert len({multi_worker.worker_lineage(i).digest_hex for i in range(3)}) == 3
    assert spec_o.manifest_binding() == spec_a.manifest_binding()
    assert spec_o.manifest_binding()["replicate_ids"] == [0, 1]

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(protocol_module, "_started_with_zero_hash_seed", lambda: True)
    with pytest.raises(RuntimeError, match="spawned worker"):
        multi_worker.require_worker_runtime()


def test_game_coordinate_counterfactuals_have_expected_rng_boundaries() -> None:
    original = _formal_context()
    changed_treatment = replace(original, treatment_id="A")
    changed_scenario = replace(original, scenario_id="scenario-1")
    changed_seat = replace(original, seat=1)

    for stream in ("scenario_source", "pool_draw", "policy_action"):
        assert (
            original.lineage(stream).digest_hex
            == changed_treatment.lineage(stream).digest_hex
        )
    assert (
        original.lineage("scenario_source").digest_hex
        != changed_scenario.lineage("scenario_source").digest_hex
    )
    assert (
        original.lineage("scenario_source").digest_hex
        == changed_seat.lineage("scenario_source").digest_hex
    )
    assert (
        original.lineage("policy_action", focal_step=0).digest_hex
        != changed_seat.lineage("policy_action", focal_step=0).digest_hex
    )
    assert (
        original.lineage("policy_action", focal_step=0).digest_hex
        == original.lineage("policy_action", focal_step=0).digest_hex
    )
    assert (
        original.lineage(
            "opponent_action", opponent_id="random", opponent_step=0
        ).digest_hex
        != original.lineage(
            "opponent_action", opponent_id="minimax", opponent_step=0
        ).digest_hex
    )


def _formal_scenario(offset: int = 0) -> ScenarioV1:
    return generate_scenario("ci_smoke", 825_100 + offset, selection_kind="ci-fixture")


def _formal_context(scenario: ScenarioV1 | None = None) -> FormalGameRng:
    snapshot = scenario or _formal_scenario()
    return FormalGameRng(
        experiment_id="task1",
        phase="T1.1",
        coupling_group="replicate-0",
        replicate_id=0,
        treatment_id="O",
        scenario_id=snapshot.scenario_id,
        seat=0,
        update=1,
        game_index=0,
    )


def _scenario_bank(tmp_path: Path, count: int = 1) -> ScenarioBank:
    scenarios = tuple(_formal_scenario(index) for index in range(count))
    manifest = write_scenario_bank(
        tmp_path / "scenario-bank",
        "ci-fixture",
        scenarios,
        compression="none",
    )
    return load_scenario_bank(Path(str(manifest["artifact_path"])))


def test_formal_policy_sampling_ignores_global_torch_rng() -> None:
    torch.manual_seed(10)
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
    rule = LimitRoundsGameRule(2)
    state = rule.current_game_state
    actions = rule.getLegalActions(state, 0)
    lineage = _formal_context().lineage("policy_action", focal_step=0)

    torch.manual_seed(1)
    first = _policy_step(
        model,
        state,
        actions,
        0,
        device=torch.device("cpu"),
        rng_lineage=lineage,
    )
    torch.manual_seed(9999)
    second = _policy_step(
        model,
        state,
        actions,
        0,
        device=torch.device("cpu"),
        rng_lineage=lineage,
    )
    legal_indexes = set(np.flatnonzero(first[1]).tolist())
    assert first[2] == second[2]
    assert first[2] in legal_indexes


def test_legacy_opponent_adapter_is_per_decision_and_restores_globals() -> None:
    rule = LimitRoundsGameRule(2)
    state = rule.current_game_state
    actions = rule.getLegalActions(state, 1)
    lineage = _formal_context().lineage(
        "opponent_action", opponent_id="random", opponent_step=0
    )

    torch.manual_seed(101)
    torch_before = torch.get_rng_state().clone()
    first = select_action(RandomAgent(1), actions, state, rule, rng_lineage=lineage)
    assert torch.equal(torch.get_rng_state(), torch_before)
    torch.manual_seed(999)
    second = select_action(RandomAgent(1), actions, state, rule, rng_lineage=lineage)
    assert first.action_index == second.action_index


def _transitions(model: PolicyValueNetwork) -> list[PPOTransition]:
    result: list[PPOTransition] = []
    for index in range(4):
        observation = np.full(265, index / 10, dtype=np.float32)
        mask = np.zeros(3510, dtype=np.uint8)
        mask[:4] = 1
        with torch.no_grad():
            logits, value = model(
                torch.from_numpy(observation),
                torch.from_numpy(mask.astype(np.float32)),
            )
            log_probability = float(torch.log_softmax(logits, dim=-1)[0, index].item())
        result.append(
            PPOTransition(
                observation=observation,
                legal_mask=mask,
                action_index=index,
                old_log_probability=log_probability,
                old_value=float(value.item()),
                reward=float(index == 3),
                terminal=index == 3,
                seed=1,
                seat=0,
            )
        )
    return result


def test_formal_minibatches_ignore_global_rng_and_record_each_epoch() -> None:
    torch.manual_seed(5)
    source = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    left = deepcopy(source)
    right = deepcopy(source)
    transitions = _transitions(source)
    config = PPOConfig(
        hidden_layers=(8,),
        updates=1,
        games_per_update=1,
        update_epochs=2,
        minibatch_size=2,
        target_kl=None,
    )
    key = RngKey(
        stream_name="minibatch",
        experiment_id="task1",
        phase="T1.1",
        coupling_group="replicate-0",
        replicate_id=0,
        treatment_id="O",
        update=1,
    )
    torch.manual_seed(1)
    left_metrics = ppo_update(
        left,
        optim.Adam(left.parameters(), lr=config.learning_rate),
        transitions,
        config,
        minibatch_key=key,
    )
    torch.manual_seed(999)
    right_metrics = ppo_update(
        right,
        optim.Adam(right.parameters(), lr=config.learning_rate),
        transitions,
        config,
        minibatch_key=key,
    )

    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name])
    assert left_metrics["minibatch_lineage"] == right_metrics["minibatch_lineage"]
    assert len(left_metrics["minibatch_lineage"]) == 2


def test_formal_game_is_replayed_after_global_rng_perturbation() -> None:
    torch.manual_seed(31)
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    pool = [
        OpponentPoolEntry(
            "first",
            CandidateSpec("first", "fixed_baseline", FirstActionAgent),
        )
    ]
    config = PPOConfig(hidden_layers=(8,), updates=1, games_per_update=1)
    scenario = _formal_scenario()
    context = _formal_context(scenario)

    torch.manual_seed(1)
    first_record, first_transitions = collect_ppo_game(
        model,
        pool,
        seed=111,
        seat=0,
        config=config,
        update_index=1,
        game_index=0,
        formal_rng=context,
        scenario=scenario,
    )
    torch.manual_seed(999)
    second_record, second_transitions = collect_ppo_game(
        model,
        pool,
        seed=999_999,
        seat=0,
        config=config,
        update_index=1,
        game_index=0,
        formal_rng=context,
        scenario=scenario,
    )

    assert first_record["status"] == second_record["status"] == "completed"
    assert first_record["score"] == second_record["score"]
    assert first_record["rival_score"] == second_record["rival_score"]
    assert first_record["rng_lineage"] == second_record["rng_lineage"]
    assert [row.action_index for row in first_transitions] == [
        row.action_index for row in second_transitions
    ]


def test_formal_opponent_factory_has_its_own_restored_rng_stream() -> None:
    torch.manual_seed(31)
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    factory_draws: list[tuple[float, float, float]] = []

    def random_factory(agent_id: int) -> FirstActionAgent:
        factory_draws.append(
            (random.random(), float(np.random.random()), float(torch.rand(())))
        )
        return FirstActionAgent(agent_id)

    pool = [
        OpponentPoolEntry(
            "factory-rng",
            CandidateSpec("factory-rng", "fixed_baseline", random_factory),
        )
    ]
    config = PPOConfig(hidden_layers=(8,), updates=1, games_per_update=1)
    scenario = _formal_scenario()
    context = _formal_context(scenario)

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected_after = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    first, _ = collect_ppo_game(
        model,
        pool,
        seed=1,
        seat=0,
        config=config,
        update_index=1,
        game_index=0,
        formal_rng=context,
        scenario=scenario,
    )
    actual_after = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    second, _ = collect_ppo_game(
        model,
        pool,
        seed=999,
        seat=0,
        config=config,
        update_index=1,
        game_index=0,
        formal_rng=context,
        scenario=scenario,
    )

    assert expected_after == actual_after
    assert factory_draws[0] == factory_draws[1]
    assert (
        first["rng_lineage"]["opponent_initializations"]
        == second["rng_lineage"]["opponent_initializations"]
    )


def test_formal_two_scenario_seat_opponent_matrix_has_no_illegal_actions() -> None:
    torch.manual_seed(47)
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    config = PPOConfig(hidden_layers=(8,), updates=1, games_per_update=4)
    opponents = (
        OpponentPoolEntry(
            "first",
            CandidateSpec("first", "fixed_baseline", FirstActionAgent),
        ),
        OpponentPoolEntry(
            "random",
            CandidateSpec("random", "fixed_baseline", RandomAgent),
        ),
    )
    records: list[dict[str, Any]] = []
    scenarios = tuple(_formal_scenario(index) for index in range(2))
    for scenario_index, scenario in enumerate(scenarios):
        for seat in (0, 1):
            for opponent in opponents:
                context = FormalGameRng(
                    experiment_id="task1-matrix",
                    phase="T1.1",
                    coupling_group="replicate-0",
                    replicate_id=0,
                    treatment_id="O",
                    scenario_id=scenario.scenario_id,
                    seat=seat,
                    update=1,
                    game_index=scenario_index * 2 + seat,
                )
                record, transitions = collect_ppo_game(
                    model,
                    [opponent],
                    seed=0,
                    seat=seat,
                    config=config,
                    update_index=1,
                    game_index=context.game_index,
                    formal_rng=context,
                    scenario=scenario,
                )
                records.append(record)
                assert transitions

    assert len(records) == 8
    assert all(record["status"] == "completed" for record in records)
    assert all(record["opponent_illegal_actions"] == 0 for record in records)
    assert {record["scenario_id"] for record in records} == {
        scenario.scenario_id for scenario in scenarios
    }
    assert {record["seat"] for record in records} == {0, 1}
    assert {record["opponent"] for record in records} == {"first", "random"}
    for scenario_id in (scenario.scenario_id for scenario in scenarios):
        assert (
            len(
                {
                    record["scenario_source_digest"]
                    for record in records
                    if record["scenario_id"] == scenario_id
                }
            )
            == 1
        )


def test_failed_formal_opponent_attempt_keeps_event_lineage() -> None:
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    scenario = _formal_scenario()
    context = replace(_formal_context(scenario), seat=1)
    record, transitions = collect_ppo_game(
        model,
        [
            OpponentPoolEntry(
                "failing",
                CandidateSpec("failing", "fixed_baseline", _FailingAgent),
            )
        ],
        seed=4,
        seat=1,
        config=PPOConfig(hidden_layers=(8,), updates=1, games_per_update=1),
        update_index=1,
        game_index=0,
        formal_rng=context,
        scenario=scenario,
    )
    assert record["status"] == "failed"
    assert transitions == []
    assert len(record["rng_lineage"]["opponent_actions"]) == 1
    assert record["scenario_id"] == scenario.scenario_id
    assert record["action_trace_sha256"]


def test_formal_trainer_consumes_schedule_and_all_named_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bc = BehaviorCloningNetwork(265, feature_version="v1", hidden_layers=(8,))
    checkpoint = tmp_path / "bc.pth"
    save_bc_checkpoint(
        bc,
        checkpoint,
        epoch=1,
        config=BCConfig(feature_version="v1", hidden_layers=(8,), epochs=1),
        dataset_metadata={"feature_version": "v1"},
        metrics={},
    )
    bank = _scenario_bank(tmp_path)
    scenario = bank.scenarios[0]
    rows = make_paired_training_schedule(
        experiment_id="task1",
        phase="T1.1",
        treatment_ids=("O", "A"),
        scenarios_by_replicate={0: (scenario.scenario_id,)},
        seats_by_replicate={0: (1,)},
        updates=1,
        games_per_update=1,
    )
    spec = FormalTrainingSpec(
        experiment_id="task1",
        phase="T1.1",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O", "A"),
        schedule=rows,
        scenario_bank_sha256=bank.payload_sha256,
    )
    captured: list[FormalGameRng] = []

    def fake_collect(  # noqa: PLR0913 - mirrors the production collector
        model: PolicyValueNetwork,
        opponent_pool: list[OpponentPoolEntry] | tuple[OpponentPoolEntry, ...],
        *,
        seed: int,
        seat: int,
        config: PPOConfig,
        update_index: int,
        game_index: int,
        formal_rng: FormalGameRng,
        scenario: ScenarioV1,
    ) -> tuple[dict[str, Any], list[PPOTransition]]:
        del opponent_pool, config
        assert scenario.scenario_id == formal_rng.scenario_id
        captured.append(formal_rng)
        transition = _transitions(model)[-1]
        transition = replace(
            transition,
            seed=seed,
            seat=seat,
            scenario_id=formal_rng.scenario_id,
            scenario_source_digest=formal_rng.lineage("scenario_source").digest_hex,
        )
        return (
            {
                "status": "completed",
                "opponent": "first",
                "seed": seed,
                "seat": seat,
                "game_index": game_index,
                "update": update_index,
                "scenario_id": formal_rng.scenario_id,
            },
            [transition],
        )

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(protocol_module, "_started_with_zero_hash_seed", lambda: True)
    monkeypatch.setattr(
        ppo_module,
        "_require_formal_training_manifest",
        lambda _path, _spec: None,
    )
    monkeypatch.setattr(ppo_module, "collect_ppo_game", fake_collect)
    config = PPOConfig(
        hidden_layers=(8,),
        updates=1,
        games_per_update=1,
        update_epochs=1,
        minibatch_size=1,
        target_kl=None,
        device_name="cpu",
    )
    result = train_ppo_selfplay(
        checkpoint,
        tmp_path / "formal-run",
        (),
        [
            OpponentPoolEntry(
                "first",
                CandidateSpec("first", "fixed_baseline", FirstActionAgent),
            )
        ],
        config=config,
        source_manifest="authorized-manifest.json",
        formal_spec=spec,
        formal_scenario_bank=bank,
    )

    assert [(context.scenario_id, context.seat) for context in captured] == [
        (scenario.scenario_id, 1)
    ]
    assert result["training_seeds"] is None
    assert result["formal_protocol"]["treatment_id"] == "O"
    assert result["formal_runtime"]["resolved_device"] == "cpu"
    assert result["formal_runtime"]["deterministic_algorithms"] is True
    assert result["logs"][1]["update_metrics"]["minibatch_lineage"]
    payload = torch.load(
        tmp_path / "formal-run" / "final.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["formal_protocol"]["paired_schedule_sha256"]
    assert payload["opponent_pool_distribution"]["snapshot_sha256"]
    assert payload["opponent_pool_distribution"]["actual_counts"]["first"] == 1
    assert payload["opponent_pool_usage"]["actual_counts"] == {"first": 1}


def test_running_manifest_must_bind_the_exact_formal_schedule(
    tmp_path: Path,
) -> None:
    rows = make_paired_training_schedule(
        experiment_id="task1",
        phase="T1.1",
        treatment_ids=("O", "A"),
        scenarios_by_replicate={0: ("scenario-0",)},
        seats_by_replicate={0: (0,)},
        updates=1,
        games_per_update=1,
    )
    spec = FormalTrainingSpec(
        experiment_id="task1",
        phase="T1.1",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O", "A"),
        schedule=rows,
    )
    path = tmp_path / "manifest.json"
    create_manifest_v2(
        path,
        experiment_id="task1",
        phase="T1.1",
        purpose="formal training binding test",
        seed_segments={
            "training": "training",
            "validation": "validation",
            "final_test": "independent_test",
        },
        budget={"games": 1},
        hypotheses={"H": "protocol test"},
        estimands={"d": "paired difference"},
        decision_rule={"selection": "fixed"},
        artifact_contract={"schedule": "training_schedule.jsonl.zst"},
        baselines={"ppo-best": {"sha256": "a" * 64}},
        formal_training=spec.manifest_binding(),
        repo=Path(__file__).resolve().parents[1],
    )
    approve_manifest(path, "test-reviewer", "binding reviewed")
    transition_manifest(path, "running", actor="test-runner", note="start")

    ppo_module._require_formal_training_manifest(str(path), spec)  # noqa: SLF001
    mismatched = replace(spec, worker_count=2)
    with pytest.raises(RuntimeError, match="training binding"):
        ppo_module._require_formal_training_manifest(  # noqa: SLF001
            str(path), mismatched
        )


def test_formal_manifest_gate_requires_replayable_code_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = make_paired_training_schedule(
        experiment_id="task1",
        phase="T1.1",
        treatment_ids=("O",),
        scenarios_by_replicate={0: ("scenario-0",)},
        seats_by_replicate={0: (0,)},
        updates=1,
        games_per_update=1,
    )
    spec = FormalTrainingSpec(
        experiment_id="task1",
        phase="T1.1",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O",),
        schedule=rows,
    )
    forged_manifest = {
        "schema_version": manifest_module.MANIFEST_SCHEMA_V2,
        "status": "running",
        "declaration": {
            "experiment_id": spec.experiment_id,
            "phase": spec.phase,
            "formal_training": spec.manifest_binding(),
        },
        "provenance": {"code": {"commit": None, "dirty": False, "patch_sha256": ""}},
    }
    monkeypatch.setattr(
        manifest_module,
        "load_manifest",
        lambda _path: forged_manifest,
    )
    with pytest.raises(RuntimeError, match="requires a Git commit"):
        ppo_module._require_formal_training_manifest(  # noqa: SLF001
            "forged.json", spec
        )


def _running_manifest(
    path: Path,
    spec: FormalTrainingSpec,
) -> None:
    create_manifest_v2(
        path,
        experiment_id=spec.experiment_id,
        phase=spec.phase,
        purpose="formal spawn matrix test",
        seed_segments={
            "training": "training",
            "validation": "validation",
            "final_test": "independent_test",
        },
        budget={"games": 2},
        hypotheses={"H": "worker invariance"},
        estimands={"d": "semantic replay equality"},
        decision_rule={"selection": "none; protocol smoke only"},
        artifact_contract={"schedule": "in-memory test schedule"},
        baselines={"ppo-best": {"sha256": "a" * 64}},
        formal_training=spec.manifest_binding(),
        repo=Path(__file__).resolve().parents[1],
    )
    approve_manifest(path, "test-reviewer", "worker binding reviewed")
    transition_manifest(path, "running", actor="test-runner", note="start")


def _formal_result_semantics(result: dict[str, Any]) -> dict[str, Any]:
    record = result["logs"][1]["training_records"][0]
    return {
        "scenario_id": record["scenario_id"],
        "seat": record["seat"],
        "opponent": record["opponent"],
        "status": record["status"],
        "outcome": record["outcome"],
        "score": record["score"],
        "rival_score": record["rival_score"],
        "plies": record["plies"],
        "action_trace_sha256": record["action_trace_sha256"],
        "rng_lineage": record["rng_lineage"],
        "minibatch_lineage": result["logs"][1]["update_metrics"]["minibatch_lineage"],
    }


def test_one_and_two_spawn_workers_run_the_same_formal_training_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bc = BehaviorCloningNetwork(265, feature_version="v1", hidden_layers=(8,))
    checkpoint = tmp_path / "bc-spawn.pth"
    save_bc_checkpoint(
        bc,
        checkpoint,
        epoch=1,
        config=BCConfig(feature_version="v1", hidden_layers=(8,), epochs=1),
        dataset_metadata={"feature_version": "v1"},
        metrics={},
    )
    bank = _scenario_bank(tmp_path)
    scenario = bank.scenarios[0]
    rows = make_paired_training_schedule(
        experiment_id="task1-spawn",
        phase="T1.1",
        treatment_ids=("A", "O"),
        scenarios_by_replicate={0: (scenario.scenario_id,)},
        seats_by_replicate={0: (0,)},
        updates=1,
        games_per_update=1,
    )
    base_spec = FormalTrainingSpec(
        experiment_id="task1-spawn",
        phase="T1.1",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("A", "O"),
        schedule=rows,
        scenario_bank_sha256=bank.payload_sha256,
    )
    config = PPOConfig(
        hidden_layers=(8,),
        updates=1,
        games_per_update=1,
        update_epochs=1,
        minibatch_size=1024,
        target_kl=None,
        current_weight=0.0,
        history_weight=0.0,
        history_limit=0,
        device_name="cpu",
    )
    opponent_pool = (FormalOpponentSpec("random", "random", weight=1.0),)

    one_specs = [replace(base_spec, treatment_id=name) for name in ("O", "A")]
    one_manifest = tmp_path / "manifest-one.json"
    _running_manifest(one_manifest, one_specs[0])
    one_jobs = tuple(
        FormalPPOTrainingJob(
            initial_bc=checkpoint,
            output_dir=tmp_path / "one" / spec.treatment_id,
            source_manifest=one_manifest,
            config=config,
            formal_spec=spec,
            opponent_pool=opponent_pool,
            scenario_bank_path=bank.artifact_path,
        )
        for spec in one_specs
    )

    two_specs = [
        replace(base_spec, treatment_id=name, worker_count=2) for name in ("O", "A")
    ]
    two_manifest = tmp_path / "manifest-two.json"
    _running_manifest(two_manifest, two_specs[0])
    two_jobs = tuple(
        FormalPPOTrainingJob(
            initial_bc=checkpoint,
            output_dir=tmp_path / "two" / spec.treatment_id,
            source_manifest=two_manifest,
            config=config,
            formal_spec=spec,
            opponent_pool=opponent_pool,
            scenario_bank_path=bank.artifact_path,
        )
        for spec in two_specs
    )

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(protocol_module, "_started_with_zero_hash_seed", lambda: True)
    one_results = run_formal_ppo_training_jobs(one_jobs, worker_count=1)
    two_results = run_formal_ppo_training_jobs(two_jobs, worker_count=2)

    assert [_formal_result_semantics(result) for result in one_results] == [
        _formal_result_semantics(result) for result in two_results
    ]
    for one_job, two_job in zip(one_jobs, two_jobs, strict=True):
        one_state = torch.load(
            one_job.output_dir / "final.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        two_state = torch.load(
            two_job.output_dir / "final.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        assert all(
            torch.equal(value, two_state[name]) for name, value in one_state.items()
        )


def test_formal_trainer_fails_closed_without_running_manifest(tmp_path: Path) -> None:
    rows = make_paired_training_schedule(
        experiment_id="task1",
        phase="T1.1",
        treatment_ids=("O",),
        scenarios_by_replicate={0: ("scenario-0",)},
        seats_by_replicate={0: (0,)},
        updates=1,
        games_per_update=1,
    )
    spec = FormalTrainingSpec(
        experiment_id="task1",
        phase="T1.1",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=("O",),
        schedule=rows,
    )
    with pytest.raises(RuntimeError, match="running manifest v2"):
        train_ppo_selfplay(
            tmp_path / "missing-bc.pth",
            tmp_path / "never-created",
            (),
            [],
            config=PPOConfig(hidden_layers=(8,), updates=1, games_per_update=1),
            formal_spec=spec,
        )

    with pytest.raises(ValueError, match="requires n_seats=2"):
        train_ppo_selfplay(
            tmp_path / "missing-bc.pth",
            tmp_path / "never-created-3p",
            (),
            [],
            config=PPOConfig(
                feature_version="public-v2-multi",
                n_seats=3,
                hidden_layers=(8,),
                updates=1,
                games_per_update=1,
            ),
            formal_spec=spec,
        )
