"""Formal ScenarioV1 validation and checkpoint-selection protocol tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from splendor.agents.our_agents.policy_imitation import manifest as manifest_module
from splendor.agents.our_agents.policy_imitation import (
    paired_evaluation as paired_module,
)
from splendor.agents.our_agents.policy_imitation.bc_network import (
    BehaviorCloningNetwork,
)
from splendor.agents.our_agents.policy_imitation.paired_evaluation import (
    EvaluationPolicy,
    PairedEvaluationSpec,
)
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    FormalOpponentSpec,
    PolicyValueNetwork,
    PPOConfig,
    evaluate_formal_validation_checkpoint,
    make_formal_validation_contract,
    save_ppo_checkpoint,
    select_formal_validation_record,
    validate_formal_validation_evidence,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    FORMAL_VALIDATION_UPDATES,
    sha256_canonical_json,
)
from splendor.agents.our_agents.policy_imitation.scenario import ScenarioV1
from splendor.agents.our_agents.policy_imitation.scenario_bank import (
    ScenarioBank,
    generate_iid_scenarios,
    load_scenario_bank,
    write_scenario_bank,
)


def _validation_opponents() -> tuple[FormalOpponentSpec, ...]:
    return tuple(
        FormalOpponentSpec(name, name)
        for name in ("ga", "heuristic", "minimax")
    )


def _validation_bank(tmp_path: Path) -> ScenarioBank:
    written = write_scenario_bank(
        tmp_path,
        "validation-A",
        generate_iid_scenarios("validation-A", 10),
        compression="none",
    )
    return load_scenario_bank(cast(Any, written["artifact_path"]))


def test_formal_validation_contract_freezes_matrix_and_selector(
    tmp_path: Path,
) -> None:
    bank = _validation_bank(tmp_path)
    contract = make_formal_validation_contract(
        bank,
        _validation_opponents(),
        device_name="cpu",
    )
    assert contract.eval_updates == FORMAL_VALIDATION_UPDATES
    assert contract.seats == (0, 1)
    assert tuple(item.opponent_id for item in contract.opponents) == (
        "ga",
        "heuristic",
        "minimax",
    )
    assert contract.scenario_ids == tuple(
        scenario.scenario_id for scenario in bank.scenarios
    )

    selected = select_formal_validation_record(
        (
            {"update": 50, "total_integer_wins": 31},
            {"update": 0, "total_integer_wins": 31},
            {"update": 100, "total_integer_wins": 30},
        )
    )
    assert selected["update"] == 0

    with pytest.raises(ValueError, match="exactly"):
        make_formal_validation_contract(
            bank,
            _validation_opponents()[:-1],
            device_name="cpu",
        )

    with pytest.raises(ValueError, match="fixed seats"):
        replace(contract, seats=(False, True))
    with pytest.raises(ValueError, match="updates must be exactly"):
        replace(
            contract,
            eval_updates=tuple(float(update) for update in FORMAL_VALIDATION_UPDATES),
        )


def test_formal_validation_uses_named_paired_runner_and_real_manifest_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = _validation_bank(tmp_path / "bank")
    opponents = _validation_opponents()
    contract = make_formal_validation_contract(bank, opponents, device_name="cpu")
    fake_spec = SimpleNamespace(
        validation=contract,
        experiment_id="task1-formal-validation",
        phase="T1.4",
        treatment_id="O",
        replicate_id=0,
        model_init_lineage=lambda: SimpleNamespace(seed63=123),
        manifest_binding=lambda: {"protocol": "paired-training-v2"},
        as_dict=lambda: {"formal": "test-spec"},
    )
    config = PPOConfig(hidden_layers=(8,), updates=2000, eval_every=50, seed=0)
    bc = BehaviorCloningNetwork(265, feature_version="v1", hidden_layers=(8,))
    model = PolicyValueNetwork.from_bc(bc)
    checkpoint = tmp_path / "initial.pth"
    save_ppo_checkpoint(
        model,
        checkpoint,
        update=0,
        config=config,
        source_bc="test-bc.pth",
        opponent_pool=(),
        metrics={},
        formal_protocol=fake_spec.as_dict(),
    )
    code = {"commit": "a" * 40, "repository": "/test/repository"}
    declaration_sha256 = "b" * 64
    monkeypatch.setattr(
        manifest_module,
        "load_manifest",
        lambda _path: {
            "status": "running",
            "declaration_sha256": declaration_sha256,
            "declaration": {
                "experiment_id": fake_spec.experiment_id,
                "phase": fake_spec.phase,
                "formal_training": fake_spec.manifest_binding(),
            },
            "provenance": {"code": code},
        },
    )
    seen_specs: list[PairedEvaluationSpec] = []

    def fake_game(
        spec: PairedEvaluationSpec,
        _candidate: EvaluationPolicy,
        _opponent: EvaluationPolicy,
        _scenario: ScenarioV1,
        _seat: int,
    ) -> dict[str, object]:
        seen_specs.append(spec)
        return {
            "status": "completed",
            "candidate_illegal_actions": 0,
            "opponent_illegal_actions": 0,
            "outcome": 1,
        }

    monkeypatch.setattr(paired_module, "play_paired_evaluation_game", fake_game)
    monkeypatch.setattr(
        paired_module,
        "audit_episode_batch",
        lambda _spec, records: {
            "status": "valid",
            "scheduled_games": len(records),
        },
    )
    result = evaluate_formal_validation_checkpoint(
        checkpoint,
        bank,
        opponents,
        cast(Any, fake_spec),
        tmp_path / "manifest.json",
        update=0,
        device_name="cpu",
    )
    assert len(seen_specs) == 10 * 2 * 3
    assert {spec.batch_id for spec in seen_specs} == {contract.batch_id}
    assert {spec.code_sha256 for spec in seen_specs} == {
        sha256_canonical_json(code)
    }
    assert {spec.manifest_declaration_sha256 for spec in seen_specs} == {
        declaration_sha256
    }
    assert result["total_integer_wins"] == 60
    assert result["scheduled_games"] == 60

    monkeypatch.setattr(
        paired_module,
        "play_paired_evaluation_game",
        lambda *_args: {
            "status": "completed",
            "candidate_illegal_actions": 0,
            "opponent_illegal_actions": 0,
            "outcome": -1,
        },
    )
    with pytest.raises(ValueError, match="independent policy rerun"):
        validate_formal_validation_evidence(
            result,
            checkpoint,
            bank,
            opponents,
            cast(Any, fake_spec),
            tmp_path / "manifest.json",
            update=0,
            device_name="cpu",
        )

    monkeypatch.setattr(
        paired_module,
        "audit_episode_batch",
        lambda _spec, _records: {"status": "invalid"},
    )
    monkeypatch.setattr(paired_module, "play_paired_evaluation_game", fake_game)
    with pytest.raises(ValueError, match="failed or missing"):
        evaluate_formal_validation_checkpoint(
            checkpoint,
            bank,
            opponents,
            cast(Any, fake_spec),
            tmp_path / "manifest.json",
            update=0,
            device_name="cpu",
        )
