"""Strict episode rows, CRN, manifest binding, append, and resume semantics."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from splendor.agents.generic.first_move import FirstActionAgent
from splendor.agents.generic.random import RandomAgent
from splendor.agents.our_agents.dqn.utils import DEFAULT_SAVED_DQN_PATH
from splendor.agents.our_agents.policy_imitation.manifest import (
    approve_manifest,
    create_manifest_v2,
    transition_manifest,
)
from splendor.agents.our_agents.policy_imitation.paired_evaluation import (
    EpisodeAppender,
    EvaluationPolicy,
    PairedEvaluationError,
    PairedEvaluationSpec,
    audit_episode_batch,
    load_episode_records,
    play_paired_evaluation_game,
    policy_config_sha256,
    require_paired_evaluation_manifest,
    run_paired_evaluation,
    snapshotless_policy_sha256,
    validate_episode_record,
    validate_paired_evaluation_binding,
)
from splendor.agents.our_agents.policy_imitation.policies import (
    CandidateSpec,
    build_builtin_candidate,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    capture_code_revision,
    sha256_canonical_json,
    sha256_file,
)
from splendor.agents.our_agents.policy_imitation.scenario import (
    scenario_from_state,
    state_from_scenario,
)
from splendor.agents.our_agents.policy_imitation.scenario_bank import (
    generate_iid_scenarios,
    load_scenario_bank,
    write_scenario_bank,
)
from splendor.agents.our_agents.policy_imitation.statistics import (
    POWER_VARIANCE_TRANSFER_PROTOCOL,
    AnalysisContrast,
    BootstrapConfig,
    HypothesisFamily,
    PowerAnalysisConfig,
    PowerSourceSpec,
    StatisticalProtocolSpec,
    StatisticsError,
    build_statistics_document,
    hypothesis_endpoint_id,
    pilot_power_design_sha256,
    validate_analysis_plan_binding,
    validate_pilot_power_source,
    validate_statistical_protocol_binding,
    write_statistics_document,
)
from splendor.agents.our_agents.ppo.ppo_agent import DEFAULT_SAVED_PPO_PATH
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

REPO = Path(__file__).resolve().parents[1]


class IllegalAgent(Agent):
    """CI-only policy that deliberately violates the legacy interface."""

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del actions, game_state, game_rule
        return cast(ActionType, None)


class LastActionAgent(Agent):
    """CI-only deterministic policy at the opposite end of the legal list."""

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        return actions[-1]


class CheckpointFixtureModule(nn.Module):
    """Tiny stateful policy used to prove checkpoint-to-agent attestation."""

    def __init__(self, strategy: int, nonce: int) -> None:
        super().__init__()
        self.input_dim = np.int64(265)
        self.register_buffer("strategy", torch.tensor(strategy, dtype=torch.int64))
        self.register_buffer("nonce", torch.tensor(nonce, dtype=torch.int64))


class CheckpointFixtureAgent(Agent):
    """CI policy whose selected action is controlled by its attached module."""

    def __init__(
        self,
        _id: int,
        *,
        module: CheckpointFixtureModule,
        reverse: bool = False,
    ) -> None:
        super().__init__(_id)
        self.net = deepcopy(module).eval()
        self.reverse = reverse
        self.fixture_mode: int | None = None

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_rule
        strategy = int(self.net.strategy.item())
        if self.reverse:
            strategy = 1 - strategy
        if strategy == 0:
            return actions[0]
        if strategy == 1:
            return actions[-1]
        if strategy == 3:
            if self.fixture_mode is None:
                initial_board_code_total = sum(
                    sum(ord(character) for character in card.code)
                    for tier in game_state.board.dealt
                    for card in tier
                    if card is not None
                )
                self.fixture_mode = initial_board_code_total % 2
            return actions[-1] if self.fixture_mode else actions[0]
        board_code_total = sum(
            sum(ord(character) for character in card.code)
            for tier in game_state.board.dealt
            for card in tier
            if card is not None
        )
        action_index = (int(self.net.nonce.item()) + board_code_total) % len(actions)
        return actions[action_index]


class DecoyCheckpointAgent(Agent):
    """Carries checkpoint state but deliberately ignores it when acting."""

    def __init__(self, _id: int, *, module: CheckpointFixtureModule) -> None:
        super().__init__(_id)
        self.net = deepcopy(module).eval()

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        return actions[0]


class DeadBranchDecoyCheckpointAgent(Agent):
    """Mentions checkpoint state only in compiler-eliminated decision code."""

    def __init__(self, _id: int, *, module: CheckpointFixtureModule) -> None:
        super().__init__(_id)
        self.net = deepcopy(module).eval()

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        if False:  # pragma: no cover - regression target for stale co_names
            return actions[int(self.net.strategy.item())]
        return actions[0]


class CallableFactory:
    """Deliberately unsupported stateful factory object."""

    def __init__(self, module: CheckpointFixtureModule) -> None:
        self.module = module

    def __call__(self, agent_id: int) -> Agent:
        return CheckpointFixtureAgent(agent_id, module=self.module)


_MUTABLE_FACTORY_GLOBAL = {"reverse": False}
_MUTABLE_HELPER_GLOBAL = {"reverse": False}


def _global_config_factory(agent_id: int, *, module: CheckpointFixtureModule) -> Agent:
    return CheckpointFixtureAgent(
        agent_id,
        module=module,
        reverse=_MUTABLE_FACTORY_GLOBAL["reverse"],
    )


def _global_factory_helper(agent_id: int, module: CheckpointFixtureModule) -> Agent:
    return CheckpointFixtureAgent(
        agent_id,
        module=module,
        reverse=_MUTABLE_HELPER_GLOBAL["reverse"],
    )


def _nested_global_config_factory(
    agent_id: int, *, module: CheckpointFixtureModule
) -> Agent:
    return _global_factory_helper(agent_id, module)


class GlobalCallableHelper:
    """Callable class whose live class config must enter factory identity."""

    reverse = False

    @classmethod
    def build(cls, agent_id: int, module: CheckpointFixtureModule) -> Agent:
        return CheckpointFixtureAgent(
            agent_id,
            module=module,
            reverse=cls.reverse,
        )


def _callable_class_config_factory(
    agent_id: int, *, module: CheckpointFixtureModule
) -> Agent:
    return GlobalCallableHelper.build(agent_id, module)


def _cyclic_global_factory(agent_id: int, *, module: CheckpointFixtureModule) -> Agent:
    if False:  # pragma: no cover - identity walker still sees the global edge
        return _cyclic_global_factory(agent_id, module=module)
    return CheckpointFixtureAgent(agent_id, module=module)


def _seat_dependent_factory(
    agent_id: int,
    *,
    modules: tuple[CheckpointFixtureModule, CheckpointFixtureModule],
) -> Agent:
    return CheckpointFixtureAgent(agent_id, module=modules[agent_id])


def _candidate(name: str, factory: Callable[[int], Agent]) -> CandidateSpec:
    return CandidateSpec(name, "fixed_baseline", factory)


def _checkpoint_candidate(
    path: Path,
    name: str,
    *,
    strategy: int,
    nonce: int,
    reverse: bool = False,
) -> CandidateSpec:
    module = CheckpointFixtureModule(strategy, nonce)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The legacy stream format is content-stable across temporary filenames;
    # zip checkpoints embed the archive basename and would perturb policy RNG.
    torch.save(
        {"model_state_dict": module.state_dict()},
        path,
        _use_new_zipfile_serialization=False,
    )
    return CandidateSpec(
        name,
        "teacher_candidate",
        partial(CheckpointFixtureAgent, module=module, reverse=reverse),
        snapshot=str(path),
    )


def _statistics_protocol(
    candidates: tuple[EvaluationPolicy, ...] | None = None,
    opponents: tuple[EvaluationPolicy, ...] | None = None,
) -> StatisticalProtocolSpec:
    candidate_hashes = (
        tuple(policy.policy_sha256 for policy in candidates)
        if candidates is not None
        else ("a" * 64, "b" * 64)
    )
    opponent_rows = (
        tuple((policy.policy_id, policy.policy_sha256) for policy in opponents)
        if opponents is not None
        else (("random", "c" * 64),)
    )
    deploy_endpoints = tuple(
        sorted(
            hypothesis_endpoint_id("d_deploy", opponent_id, opponent_sha256)
            for opponent_id, opponent_sha256 in opponent_rows
        )
    )
    method_endpoints = tuple(
        sorted(
            hypothesis_endpoint_id("d_method:A-O", opponent_id, opponent_sha256)
            for opponent_id, opponent_sha256 in opponent_rows
        )
    )
    return StatisticalProtocolSpec(
        hypothesis_families=(
            HypothesisFamily(
                "deployment",
                "superiority",
                deploy_endpoints,
                "greater",
                minimum_effect=0.03,
            ),
            HypothesisFamily(
                "method",
                "superiority",
                method_endpoints,
                "greater",
                minimum_effect=0.03,
            ),
        ),
        analysis_contrasts=(
            AnalysisContrast(
                "d_deploy",
                "deployment",
                "deployment",
                tuple(
                    sorted(
                        (
                            (candidate_hashes[0], 1.0),
                            (candidate_hashes[1], -1.0),
                        )
                    )
                ),
                "scenario",
            ),
            AnalysisContrast(
                "d_method:A-O",
                "method",
                "method",
                (("A", 1.0), ("O", -1.0)),
                "model_replicate",
            ),
        ),
        scenario_bootstrap=BootstrapConfig(resamples=1_000, seed=71),
        nested_bootstrap=BootstrapConfig(resamples=1_000, seed=72),
        scenario_power=PowerAnalysisConfig(
            "scenario",
            0.03,
            family_size=len(opponent_rows),
            max_n=200,
            round_to=50,
            simulations=1_000,
            seed=73,
        ),
        replicate_power=PowerAnalysisConfig(
            "model_replicate",
            0.03,
            family_size=len(opponent_rows),
            max_n=10,
            simulations=1_000,
            seed=74,
        ),
        power_source=PowerSourceSpec("current-batch-pilot"),
    )


@pytest.fixture(scope="module")
def formal_evaluation(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]]:
    root = tmp_path_factory.mktemp("paired-evaluation")
    bank_result = write_scenario_bank(
        root / "bank",
        "ci-fixture",
        generate_iid_scenarios("ci-fixture", 1),
        compression="none",
    )
    bank = load_scenario_bank(Path(str(bank_result["artifact_path"])))
    first_a_path = root / "first-a.pth"
    first_b_path = root / "first-b.pth"
    first_a = _checkpoint_candidate(
        first_a_path,
        "first-a",
        strategy=0,
        nonce=1,
    )
    first_b = _checkpoint_candidate(
        first_b_path,
        "first-b",
        strategy=0,
        nonce=2,
    )
    random_opponent = _candidate("random", RandomAgent)
    candidates = (
        EvaluationPolicy(
            "first-a",
            sha256_file(first_a_path),
            first_a,
            "treatment",
            "A",
            0,
            100,
            sha256_file(first_a_path),
        ),
        EvaluationPolicy(
            "first-b",
            sha256_file(first_b_path),
            first_b,
            "control",
            "O",
            0,
            100,
            sha256_file(first_b_path),
        ),
    )
    opponents = (
        EvaluationPolicy(
            "random",
            snapshotless_policy_sha256(random_opponent),
            random_opponent,
            "opponent",
        ),
    )
    code = capture_code_revision(REPO)
    provisional = PairedEvaluationSpec(
        experiment_id="task1-paired-ci",
        phase="T1.3",
        batch_id="ci-batch",
        code_sha256=sha256_canonical_json(code),
        manifest_declaration_sha256="0" * 64,
        candidates=candidates,
        opponents=opponents,
        scenario_bank=bank,
        episodes_filename="episodes.jsonl",
    )
    manifest_path = root / "manifest.json"
    manifest = create_manifest_v2(
        manifest_path,
        experiment_id=provisional.experiment_id,
        phase=provisional.phase,
        purpose="paired evaluation CI fixture",
        seed_segments={
            "training": "training",
            "validation": "validation",
            "final_test": "independent_test",
        },
        budget={"fixed_scenarios": 1, "scheduled_games": 4},
        hypotheses={"H": "protocol fixture only"},
        estimands={"primary": "paired score-rate difference"},
        decision_rule={"selection": "none; CI fixture"},
        artifact_contract={
            "episodes": "episodes.jsonl",
            "statistics": "statistics.json",
            "scenario_banks": {bank.logical_split: bank.payload_sha256},
        },
        baselines={"fixture": {"sha256": "f" * 64}},
        paired_evaluation=provisional.manifest_binding(),
        statistical_protocol=_statistics_protocol(
            candidates, opponents
        ).manifest_binding(),
        repo=REPO,
    )
    approve_manifest(manifest_path, "independent-ci", "protocol reviewed")
    transition_manifest(manifest_path, "running", actor="pytest", note="run fixture")
    spec = replace(
        provisional,
        manifest_declaration_sha256=str(manifest["declaration_sha256"]),
    )
    episodes_path = root / "episodes.jsonl"
    run_paired_evaluation(
        spec,
        episodes_path,
        source_manifest=manifest_path,
    )
    return spec, manifest_path, episodes_path, load_episode_records(episodes_path)


def test_formal_matrix_is_manifest_bound_complete_and_resumable(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
) -> None:
    spec, manifest_path, episodes_path, records = formal_evaluation
    audit = audit_episode_batch(spec, records)
    assert audit["status"] == "valid"
    assert audit["scheduled_games"] == audit["completed_games"] == 4
    before = episodes_path.read_bytes()
    resumed = run_paired_evaluation(
        spec,
        episodes_path,
        source_manifest=manifest_path,
    )
    assert resumed["status"] == "valid"
    assert episodes_path.read_bytes() == before
    require_paired_evaluation_manifest(manifest_path, spec)
    with pytest.raises(PairedEvaluationError, match=r"code SHA-256|does not bind"):
        require_paired_evaluation_manifest(
            manifest_path,
            replace(spec, code_sha256="9" * 64),
        )
    snapshot = spec.candidates[0].candidate.snapshot
    assert snapshot is not None
    checkpoint_path = Path(snapshot)
    original_checkpoint = checkpoint_path.read_bytes()
    try:
        checkpoint_path.write_bytes(b"tampered-after-manifest")
        with pytest.raises(PairedEvaluationError, match="checkpoint changed"):
            require_paired_evaluation_manifest(manifest_path, spec)
    finally:
        checkpoint_path.write_bytes(original_checkpoint)


def test_random_opponent_uses_candidate_independent_event_lineage(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
) -> None:
    _spec, _manifest, _path, records = formal_evaluation
    seat_zero = [record for record in records if record["seat"] == 0]
    assert len(seat_zero) == 2
    left_rng = seat_zero[0]["rng_lineage"]
    right_rng = seat_zero[1]["rng_lineage"]
    assert isinstance(left_rng, dict) and isinstance(right_rng, dict)
    assert left_rng["opponent_init"] == right_rng["opponent_init"]
    assert left_rng["opponent_actions"] == right_rng["opponent_actions"]
    assert left_rng["candidate_init"] != right_rng["candidate_init"]


def test_evaluator_respects_a_scenario_that_starts_with_seat_one(
    tmp_path: Path,
) -> None:
    base = generate_iid_scenarios("ci-fixture", 1)[0]
    state = state_from_scenario(base)
    state.agent_to_move = 1
    scenario = scenario_from_state(
        state,
        source_segment=base.source_segment,
        source_seed=base.source_seed,
        selection_kind="ci-fixture",
    )
    bank_result = write_scenario_bank(
        tmp_path / "seat-one-bank",
        "ci-fixture",
        (scenario,),
        compression="none",
    )
    bank = load_scenario_bank(Path(str(bank_result["artifact_path"])))
    candidate_spec = _candidate("first-seat-one", FirstActionAgent)
    opponent_spec = _candidate("random-seat-one", RandomAgent)
    candidate = EvaluationPolicy(
        "first-seat-one",
        snapshotless_policy_sha256(candidate_spec),
        candidate_spec,
        "ci-fixture",
    )
    opponent = EvaluationPolicy(
        "random-seat-one",
        snapshotless_policy_sha256(opponent_spec),
        opponent_spec,
        "opponent",
    )
    spec = PairedEvaluationSpec(
        experiment_id="seat-one-ci",
        phase="T1.3",
        batch_id="seat-one-batch",
        code_sha256="a" * 64,
        manifest_declaration_sha256="b" * 64,
        candidates=(candidate,),
        opponents=(opponent,),
        scenario_bank=bank,
        episodes_filename="episodes.jsonl",
    )
    with pytest.raises(PairedEvaluationError, match="training-only"):
        replace(
            spec,
            scenario_bank=replace(
                bank,
                selection_kind="natural-deal-srswor",
            ),
        )
    rows = tuple(
        play_paired_evaluation_game(spec, candidate, opponent, scenario, seat)
        for seat in (0, 1)
    )
    assert [row["action_trace"][0]["seat"] for row in rows] == [1, 1]  # type: ignore[index]
    assert audit_episode_batch(spec, rows)["status"] == "valid"


def test_episode_trace_and_spec_relabelling_are_detected(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
) -> None:
    spec, _manifest, _path, records = formal_evaluation
    trace_tamper = deepcopy(records[0])
    trace_tamper["action_trace_sha256"] = "0" * 64
    with pytest.raises(PairedEvaluationError, match="action_trace SHA-256"):
        validate_episode_record(trace_tamper)

    relabelled = deepcopy(records[0])
    candidate = relabelled["candidate"]
    assert isinstance(candidate, dict)
    candidate["policy_id"] = "forged-name"
    validate_episode_record(relabelled)
    with pytest.raises(PairedEvaluationError, match="metadata was relabelled"):
        audit_episode_batch(spec, [relabelled, *records[1:]])

    score_tamper = deepcopy(records[0])
    score_tamper["candidate_cal_score"] = 999.0
    score_tamper["outcome"] = 1
    score_tamper["score_rate"] = 1.0
    validate_episode_record(score_tamper)
    with pytest.raises(PairedEvaluationError, match="disagrees with replay"):
        audit_episode_batch(spec, [score_tamper, *records[1:]])

    action_tamper = deepcopy(records[0])
    trace = action_tamper["action_trace"]
    assert isinstance(trace, list) and trace
    first = trace[0]
    assert isinstance(first, dict)
    side = str(first["side"])
    old_type = str(first["action_type"])
    payload = first["action"]
    assert isinstance(payload, dict)
    payload["type"] = "forged-action"
    first["action_type"] = "forged-action"
    first["action_sha256"] = sha256_canonical_json(payload)
    counts = action_tamper[f"{side}_action_counts"]
    assert isinstance(counts, dict)
    counts[old_type] -= 1
    if counts[old_type] == 0:
        del counts[old_type]
    counts["forged-action"] = counts.get("forged-action", 0) + 1
    action_tamper["action_trace_sha256"] = sha256_canonical_json(trace)
    validate_episode_record(action_tamper)
    with pytest.raises(PairedEvaluationError, match="disagrees with ScenarioV1 replay"):
        audit_episode_batch(spec, [action_tamper, *records[1:]])


def test_policy_identity_is_computed_from_source_config_and_checkpoint_closure(
    tmp_path: Path,
) -> None:
    candidate = _candidate("first", FirstActionAgent)
    with pytest.raises(PairedEvaluationError, match="source/config SHA-256"):
        EvaluationPolicy("first", "0" * 64, candidate, "ci-fixture")
    accepted = EvaluationPolicy(
        "first",
        snapshotless_policy_sha256(candidate),
        candidate,
        "ci-fixture",
    )
    assert accepted.metadata()["source_sha256"] == accepted.source_sha256
    renamed = _candidate("renamed", FirstActionAgent)
    assert snapshotless_policy_sha256(renamed) != accepted.policy_sha256

    hidden_factory_config = {"choice": 1}

    def hidden_factory(agent_id: int) -> Agent:
        assert hidden_factory_config["choice"] == 1
        return FirstActionAgent(agent_id)

    hidden = _candidate("hidden", hidden_factory)
    with pytest.raises(PairedEvaluationError, match="cannot hide closure config"):
        snapshotless_policy_sha256(hidden)

    checkpoint = tmp_path / "closure.pth"
    module = CheckpointFixtureModule(0, 10)
    torch.save({"model_state_dict": module.state_dict()}, checkpoint)
    mutable_checkpoint_config = {"reverse": False}

    def checkpoint_factory(agent_id: int) -> Agent:
        return CheckpointFixtureAgent(
            agent_id,
            module=module,
            reverse=mutable_checkpoint_config["reverse"],
        )

    checkpoint_candidate = CandidateSpec(
        "closure-checkpoint",
        "teacher_candidate",
        checkpoint_factory,
        snapshot=str(checkpoint),
    )
    checkpoint_digest = sha256_file(checkpoint)
    checkpoint_policy = EvaluationPolicy(
        "closure-checkpoint",
        checkpoint_digest,
        checkpoint_candidate,
        "ci-fixture",
        checkpoint_sha256=checkpoint_digest,
    )
    mutable_checkpoint_config["reverse"] = True
    assert policy_config_sha256(checkpoint_candidate) != checkpoint_policy.config_sha256


def test_policy_identity_binds_mutable_defaults_globals_and_factory_kind(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mutable-config.pth"
    module = CheckpointFixtureModule(0, 11)
    torch.save({"model_state_dict": module.state_dict()}, checkpoint)

    default_config = {"reverse": False}

    def default_factory(
        agent_id: int,
        config: dict[str, bool] = default_config,
    ) -> Agent:
        return CheckpointFixtureAgent(
            agent_id,
            module=module,
            reverse=config["reverse"],
        )

    default_candidate = CandidateSpec(
        "default-config",
        "teacher_candidate",
        default_factory,
        snapshot=str(checkpoint),
    )
    before_default = policy_config_sha256(default_candidate)
    default_config["reverse"] = True
    assert policy_config_sha256(default_candidate) != before_default

    global_candidate = CandidateSpec(
        "global-config",
        "teacher_candidate",
        partial(_global_config_factory, module=module),
        snapshot=str(checkpoint),
    )
    before_global = policy_config_sha256(global_candidate)
    try:
        _MUTABLE_FACTORY_GLOBAL["reverse"] = True
        assert policy_config_sha256(global_candidate) != before_global
    finally:
        _MUTABLE_FACTORY_GLOBAL["reverse"] = False

    nested_global_candidate = CandidateSpec(
        "nested-global-config",
        "teacher_candidate",
        partial(_nested_global_config_factory, module=module),
        snapshot=str(checkpoint),
    )
    before_nested_global = policy_config_sha256(nested_global_candidate)
    try:
        _MUTABLE_HELPER_GLOBAL["reverse"] = True
        assert policy_config_sha256(nested_global_candidate) != before_nested_global
    finally:
        _MUTABLE_HELPER_GLOBAL["reverse"] = False

    callable_class_candidate = CandidateSpec(
        "callable-class-global",
        "teacher_candidate",
        partial(_callable_class_config_factory, module=module),
        snapshot=str(checkpoint),
    )
    before_callable_class = policy_config_sha256(callable_class_candidate)
    try:
        GlobalCallableHelper.reverse = True
        assert policy_config_sha256(callable_class_candidate) != before_callable_class
    finally:
        GlobalCallableHelper.reverse = False

    cyclic_candidate = CandidateSpec(
        "cyclic-global",
        "teacher_candidate",
        partial(_cyclic_global_factory, module=module),
        snapshot=str(checkpoint),
    )
    assert len(policy_config_sha256(cyclic_candidate)) == 64

    callable_candidate = CandidateSpec(
        "callable-object",
        "teacher_candidate",
        cast(Callable[[int], Agent], CallableFactory(module)),
        snapshot=str(checkpoint),
    )
    with pytest.raises(PairedEvaluationError, match="class, function, or partial"):
        policy_config_sha256(callable_candidate)


def test_checkpoint_identity_rejects_a_factory_that_does_not_expose_model_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "unused-checkpoint.pth"
    module = CheckpointFixtureModule(0, 12)
    torch.save({"model_state_dict": module.state_dict()}, checkpoint)
    candidate = CandidateSpec(
        "unused-checkpoint",
        "teacher_candidate",
        FirstActionAgent,
        snapshot=str(checkpoint),
    )
    digest = sha256_file(checkpoint)
    with pytest.raises(PairedEvaluationError, match="built agent must expose"):
        EvaluationPolicy(
            "unused-checkpoint",
            digest,
            candidate,
            "ci-fixture",
            checkpoint_sha256=digest,
        )

    decoy = CandidateSpec(
        "decoy-checkpoint",
        "teacher_candidate",
        partial(DecoyCheckpointAgent, module=module),
        snapshot=str(checkpoint),
    )
    with pytest.raises(PairedEvaluationError, match="decision-referenced"):
        EvaluationPolicy(
            "decoy-checkpoint",
            digest,
            decoy,
            "ci-fixture",
            checkpoint_sha256=digest,
        )

    dead_branch_decoy = CandidateSpec(
        "dead-branch-decoy-checkpoint",
        "teacher_candidate",
        partial(DeadBranchDecoyCheckpointAgent, module=module),
        snapshot=str(checkpoint),
    )
    with pytest.raises(PairedEvaluationError, match="decision-referenced"):
        EvaluationPolicy(
            "dead-branch-decoy-checkpoint",
            digest,
            dead_branch_decoy,
            "ci-fixture",
            checkpoint_sha256=digest,
        )


def test_checkpoint_attestation_rejects_seat_dependent_model_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "seat-dependent.pth"
    modules = (
        CheckpointFixtureModule(0, 21),
        CheckpointFixtureModule(1, 22),
    )
    torch.save(
        {
            "seat_zero_state_dict": modules[0].state_dict(),
            "seat_one_state_dict": modules[1].state_dict(),
        },
        checkpoint,
    )
    candidate = CandidateSpec(
        "seat-dependent-checkpoint",
        "teacher_candidate",
        partial(_seat_dependent_factory, modules=modules),
        snapshot=str(checkpoint),
    )
    digest = sha256_file(checkpoint)
    with pytest.raises(PairedEvaluationError, match="different model states by seat"):
        EvaluationPolicy(
            "seat-dependent-checkpoint",
            digest,
            candidate,
            "ci-fixture",
            checkpoint_sha256=digest,
        )


def test_checkpoint_attestation_accepts_the_installed_ppo_adapter() -> None:
    candidate = build_builtin_candidate(
        "ppo",
        checkpoint=DEFAULT_SAVED_PPO_PATH,
        device_name="cpu",
    )
    digest = sha256_file(DEFAULT_SAVED_PPO_PATH)
    policy = EvaluationPolicy(
        "ppo-installed",
        digest,
        candidate,
        "frozen-baseline",
        checkpoint_sha256=digest,
    )
    assert policy.checkpoint_state_sha256 is not None


def test_checkpoint_attestation_accepts_the_installed_dqn_adapter() -> None:
    candidate = build_builtin_candidate(
        "corrected-dqn",
        checkpoint=DEFAULT_SAVED_DQN_PATH,
        device_name="cpu",
    )
    digest = sha256_file(DEFAULT_SAVED_DQN_PATH)
    policy = EvaluationPolicy(
        "dqn-installed",
        digest,
        candidate,
        "frozen-baseline",
        checkpoint_sha256=digest,
    )
    assert policy.checkpoint_state_sha256 is not None


def test_append_rejects_duplicate_and_missing_rows_remain_in_denominator(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
    tmp_path: Path,
) -> None:
    spec, _manifest, _path, records = formal_evaluation
    append_path = tmp_path / "episodes.jsonl"
    with EpisodeAppender(append_path) as appender:
        appender.append(records[0])
        with pytest.raises(PairedEvaluationError, match="already recorded"):
            appender.append(records[0])
    audit = audit_episode_batch(spec, records[:-1])
    assert audit["status"] == "invalid"
    assert audit["scheduled_games"] == 4
    assert audit["missing_games"] == 1


def test_failed_policy_query_is_retained_as_an_invalid_scheduled_game(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
) -> None:
    spec, _manifest, _path, _records = formal_evaluation
    illegal_candidate = _candidate("illegal", IllegalAgent)
    bad = EvaluationPolicy(
        "illegal",
        snapshotless_policy_sha256(illegal_candidate),
        illegal_candidate,
        "ci-fixture",
    )
    failed_spec = replace(spec, candidates=(bad,))
    row = play_paired_evaluation_game(
        failed_spec,
        bad,
        failed_spec.opponents[0],
        failed_spec.scenario_bank.scenarios[0],
        0,
    )
    assert row["status"] == "failed"
    assert row["outcome"] is None and row["score_rate"] is None
    audit = audit_episode_batch(failed_spec, [row])
    assert audit["status"] == "invalid"
    assert audit["failed_games"] == 1
    assert audit["missing_games"] == 1


def test_finalization_exception_is_retained_as_a_failed_row(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _manifest, _path, _records = formal_evaluation

    def broken_cal_score(
        _rule: SplendorGameRule, _state: SplendorState, _seat: int
    ) -> float:
        raise RuntimeError("synthetic finalization failure")

    monkeypatch.setattr(SplendorGameRule, "calScore", broken_cal_score)
    row = play_paired_evaluation_game(
        spec,
        spec.candidates[0],
        spec.opponents[0],
        spec.scenario_bank.scenarios[0],
        0,
    )
    assert row["status"] == "failed"
    assert row["failure"] == {
        "side": "finalization",
        "type": "RuntimeError",
        "message": "synthetic finalization failure",
    }
    assert row["outcome"] is None and row["score_rate"] is None


def test_statistics_document_is_manifest_bound_and_never_overwritten(  # noqa: PLR0915
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
    tmp_path: Path,
) -> None:
    spec, manifest_path, _episodes_path, records = formal_evaluation
    # A one-scenario/one-replicate batch cannot be padded with unrelated
    # synthetic power results: the builder recomputes and rejects it.
    with pytest.raises(StatisticsError, match="at least 3 model replicates"):
        build_statistics_document(
            evaluation_spec=spec,
            episode_records=records,
            statistical_protocol=_statistics_protocol(spec.candidates, spec.opponents),
            source_manifest=manifest_path,
        )

    statistics_root = tmp_path / "statistics-fixture"
    bank_result = write_scenario_bank(
        statistics_root / "bank",
        "ci-fixture",
        generate_iid_scenarios("ci-fixture", 3),
        compression="none",
    )
    bank = load_scenario_bank(Path(str(bank_result["artifact_path"])))

    def checkpoint_policy(treatment: str, replicate: int) -> EvaluationPolicy:
        checkpoint = statistics_root / f"{treatment}-{replicate}.pth"
        # Keep the A checkpoints state-dependent so the deployment contrast has
        # scenario variance without depending on ambient/global RNG. Alternate
        # deterministic control behavior so the
        # replicate-level A-O effects cannot collapse merely because all
        # policies happened to draw the same score rate.
        if treatment == "A":
            strategy = 3
        else:
            strategy = replicate % 2
        candidate = _checkpoint_candidate(
            checkpoint,
            f"{treatment}-{replicate}",
            strategy=strategy,
            nonce=(102 + replicate if treatment == "A" else 200 + replicate),
        )
        digest = sha256_file(checkpoint)
        return EvaluationPolicy(
            f"{treatment}-{replicate}",
            digest,
            candidate,
            "control" if treatment == "O" else "treatment",
            treatment,
            replicate,
            100 + replicate,
            digest,
        )

    candidates = tuple(
        checkpoint_policy(treatment, replicate)
        for replicate in (0, 1, 2)
        for treatment in ("A", "O")
    )
    opponent_candidate = _candidate("first-statistics", FirstActionAgent)
    opponents = (
        EvaluationPolicy(
            "first-statistics",
            snapshotless_policy_sha256(opponent_candidate),
            opponent_candidate,
            "opponent",
        ),
    )
    code = capture_code_revision(REPO)
    provisional = PairedEvaluationSpec(
        experiment_id="task1-statistics-ci",
        phase="T1.3",
        batch_id="statistics-ci-batch",
        code_sha256=sha256_canonical_json(code),
        manifest_declaration_sha256="0" * 64,
        candidates=candidates,
        opponents=opponents,
        scenario_bank=bank,
        episodes_filename="episodes.jsonl",
    )
    protocol = _statistics_protocol(candidates, opponents)
    statistics_manifest_path = statistics_root / "manifest.json"
    manifest = create_manifest_v2(
        statistics_manifest_path,
        experiment_id=provisional.experiment_id,
        phase=provisional.phase,
        purpose="statistics evidence-chain CI fixture",
        seed_segments={
            "training": "training",
            "validation": "validation",
            "final_test": "independent_test",
        },
        budget={"fixed_scenarios": 3, "scheduled_games": 36},
        hypotheses={"H": "protocol fixture only"},
        estimands={"primary": "paired score-rate difference"},
        decision_rule={"selection": "none; CI fixture"},
        artifact_contract={
            "episodes": "episodes.jsonl",
            "statistics": "statistics.json",
            "scenario_banks": {bank.logical_split: bank.payload_sha256},
        },
        baselines={"fixture": {"sha256": "f" * 64}},
        paired_evaluation=provisional.manifest_binding(),
        statistical_protocol=protocol.manifest_binding(),
        repo=REPO,
    )
    approve_manifest(statistics_manifest_path, "independent-ci", "protocol reviewed")
    transition_manifest(
        statistics_manifest_path, "running", actor="pytest", note="run fixture"
    )
    statistics_spec = replace(
        provisional,
        manifest_declaration_sha256=str(manifest["declaration_sha256"]),
    )
    statistics_episodes = statistics_root / "episodes.jsonl"
    run_paired_evaluation(
        statistics_spec,
        statistics_episodes,
        source_manifest=statistics_manifest_path,
    )
    statistics_records = load_episode_records(statistics_episodes)
    document = build_statistics_document(
        evaluation_spec=statistics_spec,
        episode_records=tuple(reversed(statistics_records)),
        statistical_protocol=protocol,
        source_manifest=statistics_manifest_path,
    )
    power = document["power"]
    assert isinstance(power, dict)
    scenario_power = power["scenario"]
    replicate_power = power["model_replicate"]
    assert isinstance(scenario_power, dict) and isinstance(replicate_power, dict)
    fixed_source = PowerSourceSpec(
        "approved-pilot-fixed-n",
        pilot_manifest_declaration_sha256=str(document["manifest_declaration_sha256"]),
        pilot_statistics_content_sha256=str(document["content_sha256"]),
        fixed_scenario_n=cast(int, scenario_power["recommended_n"]),
        fixed_model_replicates=cast(int, replicate_power["recommended_n"]),
        pilot_power_design_sha256=pilot_power_design_sha256(document),
        variance_transfer_protocol=POWER_VARIANCE_TRANSFER_PROTOCOL,
    )
    fixed_protocol = replace(protocol, power_source=fixed_source)
    assert (
        validate_statistical_protocol_binding(fixed_protocol.manifest_binding())
        == fixed_protocol
    )
    verification = validate_pilot_power_source(fixed_source, document, fixed_protocol)
    assert verification["pilot_statistics_content_sha256"] == document["content_sha256"]
    with pytest.raises(StatisticsError, match="scenario count"):
        validate_analysis_plan_binding(
            statistics_spec.manifest_binding(), fixed_protocol
        )
    tampered_pilot = deepcopy(document)
    tampered_power = tampered_pilot["power"]
    assert isinstance(tampered_power, dict)
    tampered_power["prospective_use_only"] = False
    with pytest.raises(StatisticsError, match="content SHA-256"):
        validate_pilot_power_source(fixed_source, tampered_pilot, fixed_protocol)

    renamed_pilot = deepcopy(document)
    renamed_statistical = renamed_pilot["statistical_protocol"]
    renamed_power = renamed_pilot["power"]
    assert isinstance(renamed_statistical, dict) and isinstance(renamed_power, dict)
    contrasts = renamed_statistical["analysis_contrasts"]
    families = renamed_statistical["hypothesis_families"]
    scenario_table = renamed_power["scenario"]
    assert isinstance(contrasts, list) and isinstance(families, list)
    assert isinstance(scenario_table, dict)
    scenario_contrast = next(
        contrast
        for contrast in contrasts
        if isinstance(contrast, dict) and contrast["power_unit"] == "scenario"
    )
    old_contrast_id = str(scenario_contrast["contrast_id"])
    new_contrast_id = f"{old_contrast_id}-renamed"
    scenario_contrast["contrast_id"] = new_contrast_id
    family_id = scenario_contrast["family_id"]
    family = next(
        row
        for row in families
        if isinstance(row, dict) and row["family_id"] == family_id
    )
    family["endpoints"] = [
        str(endpoint).replace(f"{old_contrast_id}|", f"{new_contrast_id}|")
        for endpoint in family["endpoints"]
    ]
    scenario_table["contrast_id"] = new_contrast_id
    endpoint_results = scenario_table["endpoint_results"]
    assert isinstance(endpoint_results, dict)
    scenario_table["endpoint_results"] = {
        str(endpoint).replace(f"{old_contrast_id}|", f"{new_contrast_id}|"): value
        for endpoint, value in endpoint_results.items()
    }
    embedded_declaration = renamed_pilot["manifest_declaration"]
    assert isinstance(embedded_declaration, dict)
    embedded_declaration["statistical_protocol"] = deepcopy(renamed_statistical)
    renamed_manifest_sha256 = sha256_canonical_json(embedded_declaration)
    renamed_pilot["manifest_declaration_sha256"] = renamed_manifest_sha256
    renamed_power["source_manifest_declaration_sha256"] = renamed_manifest_sha256
    renamed_pilot.pop("content_sha256")
    renamed_pilot["content_sha256"] = sha256_canonical_json(renamed_pilot)
    renamed_source = replace(
        fixed_source,
        pilot_manifest_declaration_sha256=renamed_manifest_sha256,
        pilot_statistics_content_sha256=str(renamed_pilot["content_sha256"]),
    )
    renamed_protocol = replace(protocol, power_source=renamed_source)
    with pytest.raises(StatisticsError, match="power-design SHA-256"):
        validate_pilot_power_source(
            renamed_source,
            renamed_pilot,
            renamed_protocol,
        )

    forged_matrix = deepcopy(document)
    forged_binding = forged_matrix["evaluation_manifest_binding"]
    assert isinstance(forged_binding, dict)
    forged_binding["batch_id"] = "forged-pilot-batch"
    forged_matrix.pop("content_sha256")
    forged_matrix["content_sha256"] = sha256_canonical_json(forged_matrix)
    forged_source = replace(
        fixed_source,
        pilot_statistics_content_sha256=str(forged_matrix["content_sha256"]),
    )
    with pytest.raises(StatisticsError, match="embedded manifest"):
        validate_pilot_power_source(
            forged_source,
            forged_matrix,
            replace(protocol, power_source=forged_source),
        )

    table_tamper = deepcopy(document)
    table_power = table_tamper["power"]
    assert isinstance(table_power, dict)
    table_scenario = table_power["scenario"]
    assert isinstance(table_scenario, dict)
    table_scenario["contrast_id"] = "forged-table-contrast"
    table_tamper.pop("content_sha256")
    table_tamper["content_sha256"] = sha256_canonical_json(table_tamper)
    table_source = replace(
        fixed_source,
        pilot_statistics_content_sha256=str(table_tamper["content_sha256"]),
    )
    with pytest.raises(StatisticsError, match="power table disagrees"):
        validate_pilot_power_source(
            table_source,
            table_tamper,
            replace(protocol, power_source=table_source),
        )
    output = tmp_path / "statistics.json"
    write_statistics_document(output, document)
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        write_statistics_document(output, document)
    with pytest.raises(StatisticsError, match="statistics protocol is not bound"):
        build_statistics_document(
            evaluation_spec=statistics_spec,
            episode_records=statistics_records,
            statistical_protocol=replace(
                protocol,
                scenario_bootstrap=BootstrapConfig(resamples=1_000, seed=999),
            ),
            source_manifest=statistics_manifest_path,
        )


def test_manifest_bindings_reject_denominator_and_tie_rule_tampering(
    formal_evaluation: tuple[
        PairedEvaluationSpec, Path, Path, tuple[dict[str, object], ...]
    ],
) -> None:
    spec, _manifest_path, _episodes_path, _records = formal_evaluation
    evaluation_binding = spec.manifest_binding()
    validate_paired_evaluation_binding(evaluation_binding)
    evaluation_binding["scheduled_games"] = 3
    with pytest.raises(PairedEvaluationError, match="scheduled denominator"):
        validate_paired_evaluation_binding(evaluation_binding)

    statistics_binding = _statistics_protocol().manifest_binding()
    validate_statistical_protocol_binding(statistics_binding)
    statistics_binding["score_rate"] = {"win": 1.0, "draw": 0.0, "loss": 0.0}
    with pytest.raises(StatisticsError, match="tie rule"):
        validate_statistical_protocol_binding(statistics_binding)

    protocol = _statistics_protocol(spec.candidates, spec.opponents)
    validate_analysis_plan_binding(spec.manifest_binding(), protocol)

    swapped_deployment = replace(
        protocol,
        analysis_contrasts=tuple(
            replace(
                contrast,
                components=tuple(
                    (identity, -coefficient)
                    for identity, coefficient in contrast.components
                ),
            )
            if contrast.power_unit == "scenario"
            else contrast
            for contrast in protocol.analysis_contrasts
        ),
    )
    with pytest.raises(StatisticsError, match="official/treatment minus"):
        validate_analysis_plan_binding(spec.manifest_binding(), swapped_deployment)

    swapped_method = replace(
        protocol,
        analysis_contrasts=tuple(
            replace(
                contrast,
                components=tuple(
                    (identity, -coefficient)
                    for identity, coefficient in contrast.components
                ),
            )
            if contrast.power_unit == "model_replicate"
            else contrast
            for contrast in protocol.analysis_contrasts
        ),
    )
    with pytest.raises(StatisticsError, match="treatment minus control"):
        validate_analysis_plan_binding(spec.manifest_binding(), swapped_method)

    mismatched = protocol.manifest_binding()
    families = mismatched["hypothesis_families"]
    assert isinstance(families, list) and isinstance(families[0], dict)
    families[0]["endpoints"] = ["forged-endpoint"]
    parsed = validate_statistical_protocol_binding(mismatched)
    with pytest.raises(StatisticsError, match="endpoints do not match"):
        validate_analysis_plan_binding(spec.manifest_binding(), parsed)
