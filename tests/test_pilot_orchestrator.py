"""Resource and lifecycle gates for the T1.4 production pilot orchestrator."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from splendor.agents.our_agents.policy_imitation import pilot_orchestrator as module
from splendor.agents.our_agents.policy_imitation.pilot_orchestrator import (
    LAUNCH_DECLARATION_FILE_NAME,
    LEGACY_ORCHESTRATION_SCHEMA,
    LEGACY_WALL_LIMIT_SECONDS,
    MIN_FREE_BYTES,
    ORCHESTRATION_SCHEMA,
    OUTPUT_LIMIT_BYTES,
    PILOT_JOB_COUNT,
    PILOT_REPLICATE_IDS,
    PILOT_TREATMENT_IDS,
    PILOT_WORKER_COUNT,
    WALL_LIMIT_SECONDS,
    PilotOrchestrationError,
    ProductionRuntime,
    ResourceSnapshot,
    load_pilot_declaration,
    require_production_runtime,
    require_resource_limits,
    write_pilot_declaration,
)
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import PPOConfig
from splendor.agents.our_agents.policy_imitation.protocol import FormalJobOutput
from splendor.agents.our_agents.policy_imitation.seed_roll import (
    SeedRollSelectionDesign,
)
from splendor.seed_registry import TASK1_VALIDATION_A


def _config(shaping_kind: str) -> dict[str, object]:
    return asdict(
        PPOConfig(
            feature_version="public-v2",
            hidden_layers=(128, 128, 128, 128),
            learning_rate=1e-4,
            value_coefficient=1.0,
            updates=2_000,
            games_per_update=16,
            critic_learning_rate=5e-4,
            critic_warmup_epochs=2,
            shaping_kind=shaping_kind,
            eval_every=50,
            device_name="cuda",
            seed=0,
        )
    )


def _payload(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repository = (tmp_path / "repo").resolve()
    control = repository / "runs" / "pilot" / "control"
    control.mkdir(parents=True)
    declaration_path = control / "orchestration.json"
    initial_bc = repository / module.PILOT_INITIAL_BC_RELATIVE_PATH
    opponent_pool = [
        {
            "entry_name": name,
            "candidate_name": name,
            "weight": 1.0,
            "checkpoint": None,
        }
        for name in module.PILOT_TRAINING_OPPONENT_IDS
    ]
    treatments = [
        {
            "treatment_id": "O",
            "initial_bc": str(initial_bc),
            "config": _config("safe-potential"),
            "opponent_pool": opponent_pool,
        },
        {
            "treatment_id": "O_bridge",
            "initial_bc": str(initial_bc),
            "config": _config("potential"),
            "opponent_pool": opponent_pool,
        },
    ]
    payload: dict[str, object] = {
        "schema_version": ORCHESTRATION_SCHEMA,
        "experiment_id": "task1-t14-crossed-pilot-20260915",
        "phase": "T1.4",
        "repository": str(repository),
        "python_executable": str(repository / ".venv-p5000" / "bin" / "python"),
        "manifest_path": str(repository / "runs" / "pilot" / "manifest.json"),
        "seed_roll_path": str(repository / "runs" / "pilot" / "roll.json"),
        "scenario_bank_path": str(
            repository / "runs" / "pilot" / "train-schedule.jsonl"
        ),
        "validation_scenario_bank_path": str(
            repository / "runs" / "pilot" / "validation-a-first10.jsonl"
        ),
        "output_root": str(repository / "runs" / "pilot" / "training"),
        "control_dir": str(control),
        "tmux_session": "task1-t14-pilot",
        "replicate_ids": list(PILOT_REPLICATE_IDS),
        "worker_count": PILOT_WORKER_COUNT,
        "limits": {
            "wall_seconds": WALL_LIMIT_SECONDS,
            "max_output_bytes": OUTPUT_LIMIT_BYTES,
            "min_free_bytes": MIN_FREE_BYTES,
        },
        "treatments": treatments,
    }
    declaration_path.write_text(json.dumps(payload), encoding="utf-8")
    return declaration_path, payload


def _runtime(declaration_path: Path) -> ProductionRuntime:
    declaration = load_pilot_declaration(declaration_path)
    return ProductionRuntime(
        python_executable=str(declaration.python_executable),
        python_prefix=str(declaration.repository / ".venv-p5000"),
        python_hash_seed="0",
        hash_randomization=0,
        cublas_workspace_config=":4096:8",
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_name="Quadro P5000",
        cuda_capability=(6, 1),
        cuda_arch_list=("sm_60", "sm_70"),
        torch_cuda_version="12.6",
    )


def _supervisor_binding(declaration: module.PilotDeclaration) -> dict[str, object]:
    output_root_device, output_root_inode = module._reserve_output_root(  # noqa: SLF001
        declaration.output_root
    )
    return {
        "expected_manifest_path": declaration.manifest_path,
        "expected_control_dir": declaration.control_dir,
        "expected_output_root": declaration.output_root,
        "expected_experiment_id": declaration.experiment_id,
        "expected_tmux_session": declaration.tmux_session,
        "expected_wall_limit_seconds": declaration.wall_limit_seconds,
        "expected_output_root_device": output_root_device,
        "expected_output_root_inode": output_root_inode,
    }


def _rewrite(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_declaration_freezes_exact_pilot_matrix_and_limits(tmp_path: Path) -> None:
    path, payload = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    assert tuple(item.treatment_id for item in declaration.treatments) == (
        "O",
        "O_bridge",
    )
    assert declaration.python_executable == (
        declaration.repository / ".venv-p5000" / "bin" / "python"
    )
    assert len(PILOT_REPLICATE_IDS) * len(declaration.treatments) == PILOT_JOB_COUNT
    assert declaration.wall_limit_seconds == 36 * 60 * 60

    for field, invalid in (
        ("replicate_ids", [0, 1]),
        ("worker_count", 2),
        ("phase", "T1.4 "),
    ):
        changed = dict(payload)
        changed[field] = invalid
        _rewrite(path, changed)
        with pytest.raises(PilotOrchestrationError):
            load_pilot_declaration(path)

    changed = dict(payload)
    changed["limits"] = {
        "wall_seconds": WALL_LIMIT_SECONDS + 1,
        "max_output_bytes": OUTPUT_LIMIT_BYTES,
        "min_free_bytes": MIN_FREE_BYTES,
    }
    _rewrite(path, changed)
    with pytest.raises(PilotOrchestrationError, match="orchestration schema"):
        load_pilot_declaration(path)


def test_legacy_eighteen_hour_declaration_remains_auditable(tmp_path: Path) -> None:
    path, payload = _payload(tmp_path)
    payload["schema_version"] = LEGACY_ORCHESTRATION_SCHEMA
    payload["limits"] = {
        "wall_seconds": LEGACY_WALL_LIMIT_SECONDS,
        "max_output_bytes": OUTPUT_LIMIT_BYTES,
        "min_free_bytes": MIN_FREE_BYTES,
    }
    _rewrite(path, payload)

    declaration = load_pilot_declaration(path)

    assert declaration.wall_limit_seconds == 18 * 60 * 60


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("replicate_ids", [False, True, 2]),
        ("worker_count", 3.0),
    ],
)
def test_declaration_rejects_bool_or_float_integer_axes(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    path, payload = _payload(tmp_path)
    payload[field] = invalid
    _rewrite(path, payload)
    with pytest.raises(PilotOrchestrationError):
        load_pilot_declaration(path)


@pytest.mark.parametrize(
    "limit_name", ["wall_seconds", "max_output_bytes", "min_free_bytes"]
)
def test_declaration_rejects_float_resource_limits(
    tmp_path: Path,
    limit_name: str,
) -> None:
    path, payload = _payload(tmp_path)
    limits = cast(dict[str, object], payload["limits"])
    limits[limit_name] = float(cast(int, limits[limit_name]))
    _rewrite(path, payload)
    with pytest.raises(PilotOrchestrationError, match="pilot limits"):
        load_pilot_declaration(path)


def test_declaration_writer_is_canonical_validated_and_one_shot(
    tmp_path: Path,
) -> None:
    existing_path, payload = _payload(tmp_path)
    target = existing_path.with_name("one-shot.json")
    declaration = write_pilot_declaration(target, payload)
    assert declaration.path == target
    assert target.read_bytes() == (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    original = target.read_bytes()
    with pytest.raises(PilotOrchestrationError, match="already exists"):
        write_pilot_declaration(target, payload)
    assert target.read_bytes() == original

    invalid_target = existing_path.with_name("invalid.json")
    with pytest.raises(PilotOrchestrationError, match="keys mismatch"):
        write_pilot_declaration(
            invalid_target,
            {key: value for key, value in payload.items() if key != "phase"},
        )
    assert not invalid_target.exists()


def test_declaration_rejects_cpu_and_treatment_relabeling(tmp_path: Path) -> None:
    path, payload = _payload(tmp_path)
    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[0]["config"]["device_name"] = "cpu"
    changed = dict(payload)
    changed["treatments"] = treatments
    _rewrite(path, changed)
    with pytest.raises(PilotOrchestrationError, match="request cuda"):
        load_pilot_declaration(path)

    treatments[0]["config"]["device_name"] = "cuda"
    treatments.reverse()
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="ordered exactly"):
        load_pilot_declaration(path)

    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[0]["config"]["eval_every"] = 100
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="every 50"):
        load_pilot_declaration(path)

    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[0]["config"]["shaping_kind"] = "potential"
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="frozen C2-R2"):
        load_pilot_declaration(path)

    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[1]["config"]["learning_rate"] = 2e-4
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="frozen C2-R2"):
        load_pilot_declaration(path)

    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[0]["config"]["minibatch_size"] = 256.0
    treatments[0]["config"]["seed"] = False
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="exact integers"):
        load_pilot_declaration(path)

    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[0]["opponent_pool"].reverse()
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="training pool"):
        load_pilot_declaration(path)

    treatments = json.loads(json.dumps(payload["treatments"]))
    treatments[0]["opponent_pool"][0]["weight"] = True
    _rewrite(path, {**payload, "treatments": treatments})
    with pytest.raises(PilotOrchestrationError, match="invalid opponent_pool"):
        load_pilot_declaration(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("python_executable", "/usr/bin/python3", ".venv-p5000"),
        ("python_hash_seed", None, "PYTHONHASHSEED"),
        ("hash_randomization", 1, "PYTHONHASHSEED"),
        ("cublas_workspace_config", None, "CUBLAS_WORKSPACE_CONFIG"),
        ("cuda_available", False, "never falls back"),
        ("cuda_device_name", "Tesla T4", "Quadro P5000"),
        ("cuda_capability", (7, 5), "sm61"),
        ("cuda_arch_list", ("sm_50", "sm_70"), "sm_61-compatible"),
    ],
)
def test_runtime_gate_has_no_cpu_or_non_p5000_fallback(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    runtime = replace(_runtime(path), **{field: value})
    with pytest.raises(PilotOrchestrationError, match=message):
        require_production_runtime(declaration, runtime)


@pytest.mark.parametrize("architecture", ["sm_60", "sm_61"])
def test_runtime_gate_accepts_binary_compatible_pascal_targets(
    tmp_path: Path,
    architecture: str,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    runtime = replace(_runtime(path), cuda_arch_list=(architecture, "sm_70"))

    assert require_production_runtime(declaration, runtime) == runtime


def test_disk_guards_are_exact_and_output_tree_refuses_symlinks(
    tmp_path: Path,
) -> None:
    require_resource_limits(
        ResourceSnapshot(
            output_bytes=OUTPUT_LIMIT_BYTES,
            free_bytes=MIN_FREE_BYTES,
        )
    )
    with pytest.raises(PilotOrchestrationError, match="exceeded 32GiB"):
        require_resource_limits(
            ResourceSnapshot(
                output_bytes=OUTPUT_LIMIT_BYTES + 1,
                free_bytes=MIN_FREE_BYTES,
            )
        )
    with pytest.raises(PilotOrchestrationError, match="below 40GiB"):
        require_resource_limits(
            ResourceSnapshot(
                output_bytes=0,
                free_bytes=MIN_FREE_BYTES - 1,
            )
        )
    output = tmp_path / "output"
    output.mkdir()
    (output / "outside-link").symlink_to(tmp_path)
    with pytest.raises(PilotOrchestrationError, match="contains a symlink"):
        module._tree_size_bytes(output)  # noqa: SLF001
    with pytest.raises(PilotOrchestrationError, match="output root disappeared"):
        module._tree_size_bytes(tmp_path / "missing-output")  # noqa: SLF001
    prelaunch = module.capture_resources(
        tmp_path / "missing-output",
        allow_missing_output_root=True,
    )
    assert prelaunch.output_bytes == 0


def test_launch_reserves_one_empty_real_output_root(tmp_path: Path) -> None:
    output = tmp_path / "output"
    module._reserve_output_root(output)  # noqa: SLF001
    assert output.is_dir()
    module._reserve_output_root(output)  # noqa: SLF001

    (output / "unexpected").write_bytes(b"occupied")
    with pytest.raises(PilotOrchestrationError, match="must be empty"):
        module._reserve_output_root(output)  # noqa: SLF001


def test_output_root_reservation_rejects_symlink_parent_and_replacement(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    redirected = linked_parent / "training"
    with pytest.raises(PilotOrchestrationError, match="parent must be a real"):
        module._reserve_output_root(redirected)  # noqa: SLF001
    assert not (real_parent / "training").exists()

    output = tmp_path / "training"
    device, inode = module._reserve_output_root(output)  # noqa: SLF001
    output.rename(tmp_path / "displaced-training")
    output.mkdir()
    with pytest.raises(PilotOrchestrationError, match="identity changed"):
        module.capture_resources(
            output,
            expected_device=device,
            expected_inode=inode,
        )


def test_output_scan_tolerates_atomic_temp_disappearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    stable = output / "checkpoint.pth"
    transient = output / ".result.123.tmp"
    stable.write_bytes(b"stable")
    transient.write_bytes(b"temporary")
    real_lstat = module.os.lstat

    def racing_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        if Path(path) == transient:
            transient.unlink(missing_ok=True)
            raise FileNotFoundError(path)
        return real_lstat(path)

    monkeypatch.setattr(module.os, "lstat", racing_lstat)
    assert module._tree_size_bytes(output) == len(b"stable")  # noqa: SLF001


def test_manifest_outputs_must_be_canonical_six_direct_children(
    tmp_path: Path,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    outputs = [
        FormalJobOutput(
            replicate_id=replicate_id,
            treatment_id=treatment_id,
            output_dir=str(
                declaration.output_root / f"r{replicate_id}-{treatment_id}"
            ),
        ).as_dict()
        for replicate_id in PILOT_REPLICATE_IDS
        for treatment_id in PILOT_TREATMENT_IDS
    ]
    manifest = {"declaration": {"formal_training": {"job_outputs": outputs}}}
    parsed = module._manifest_job_outputs(manifest, declaration)  # noqa: SLF001
    assert len(parsed) == PILOT_JOB_COUNT
    reordered = outputs.copy()
    reordered.reverse()
    with pytest.raises(PilotOrchestrationError, match="canonical six-job"):
        module._manifest_job_outputs(  # noqa: SLF001
            {"declaration": {"formal_training": {"job_outputs": reordered}}},
            declaration,
        )


def test_pilot_bank_refuses_sealed_or_reserved_rows(tmp_path: Path) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    selection = SeedRollSelectionDesign(
        seed_roll_payload_sha256="a" * 64,
        randomization_root_sha256="b" * 64,
        active_replicate_ids=PILOT_REPLICATE_IDS,
        games_per_replicate=32_000,
        source_population=200_000,
    )
    base = SimpleNamespace(
        sealed=False,
        logical_split="train-schedule",
        selection_design=selection,
        scenario_count=96_000,
        artifact_path=declaration.scenario_bank_path,
    )
    assert module._require_pilot_bank(base) == selection  # noqa: SLF001
    with pytest.raises(PilotOrchestrationError, match="sealed"):
        module._require_pilot_bank(  # noqa: SLF001
            SimpleNamespace(**{**vars(base), "sealed": True})
        )
    reserve = replace(selection, active_replicate_ids=(3, 4))
    with pytest.raises(PilotOrchestrationError, match="reserve access refused"):
        module._require_pilot_bank(  # noqa: SLF001
            SimpleNamespace(**{**vars(base), "selection_design": reserve})
        )


def test_validation_bank_is_exactly_first_ten_validation_a_rows() -> None:
    scenarios = tuple(
        SimpleNamespace(source_seed=TASK1_VALIDATION_A.start + index)
        for index in reversed(range(10))
    )
    base = SimpleNamespace(
        sealed=False,
        logical_split="validation-A",
        selection_kind="iid",
        scenario_count=10,
        scenarios=scenarios,
    )
    module._require_pilot_validation_bank(base)  # noqa: SLF001
    with pytest.raises(PilotOrchestrationError, match="first 10"):
        module._require_pilot_validation_bank(  # noqa: SLF001
            SimpleNamespace(
                **{
                    **vars(base),
                    "scenarios": (
                        *scenarios[:-1],
                        SimpleNamespace(source_seed=TASK1_VALIDATION_A.start + 10),
                    ),
                }
            )
        )
    with pytest.raises(PilotOrchestrationError, match="non-sealed"):
        module._require_pilot_validation_bank(  # noqa: SLF001
            SimpleNamespace(**{**vars(base), "sealed": True})
        )


def test_completed_selection_requires_all_41_validated_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "job"
    output.mkdir()
    best_path = output / "best.pth"
    best_path.write_bytes(b"selected checkpoint")
    best_sha256 = module.sha256_file(best_path)
    contract_dict = {
        "protocol": "scenario-validation-a-v1",
        "scenario_bank_sha256": "b" * 64,
    }
    contract = SimpleNamespace(
        scenario_bank_sha256="b" * 64,
        as_dict=lambda: contract_dict,
    )
    formal_spec = SimpleNamespace(
        validation=contract,
        experiment_id="experiment",
        phase="T1.4",
        replicate_id=0,
        treatment_id="O",
    )
    validation_bank_path = tmp_path / "validation-bank.jsonl"
    validation_bank_path.write_text("fixture\n", encoding="utf-8")
    job = SimpleNamespace(
        formal_spec=formal_spec,
        output_dir=output,
        validation_scenario_bank_path=validation_bank_path,
        validation_opponents=(SimpleNamespace(entry_name="fixture"),),
        source_manifest=tmp_path / "manifest.json",
        config=SimpleNamespace(device_name="cpu"),
    )
    monkeypatch.setattr(module, "load_scenario_bank", lambda _path: object())
    validated_updates: list[int] = []

    def validate_evidence(
        evidence: Mapping[str, object],
        *_args: object,
        update: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        validated_updates.append(update)
        return dict(evidence)

    monkeypatch.setattr(
        module.ppo_selfplay_module,
        "validate_formal_validation_evidence",
        validate_evidence,
    )
    evaluations: list[dict[str, object]] = []
    for update in module.FORMAL_VALIDATION_UPDATES:
        wins = 9 if update in {50, 100} else 8
        checkpoint_sha256 = best_sha256 if update == 50 else "a" * 64
        checkpoint_path = (
            output / "initial.pth"
            if update == 0
            else output / f"update-{update}.pth"
        )
        checkpoint_path.write_bytes(
            b"selected checkpoint" if update == 50 else bytes.fromhex("aa")
        )
        checkpoint_sha256 = module.sha256_file(checkpoint_path)
        if update == 50:
            assert checkpoint_sha256 == best_sha256
        validation: dict[str, object] = {
            "status": "valid",
            "update": update,
            "checkpoint_sha256": checkpoint_sha256,
            "total_integer_wins": wins,
            "scheduled_games": module.PILOT_VALIDATION_GAMES,
            "scenario_bank_sha256": "b" * 64,
        }
        validation["evidence_sha256"] = module.sha256_canonical_json(validation)
        _rewrite(output / f"validation-update-{update}.json", validation)
        evaluations.append(
            {
                "update": update,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "validation_evidence_sha256": validation["evidence_sha256"],
                "total_integer_wins": wins,
                "scheduled_games": module.PILOT_VALIDATION_GAMES,
            }
        )
    selection: dict[str, object] = {
        "protocol": "formal-checkpoint-selection-v1",
        "status": "completed",
        "experiment_id": "experiment",
        "phase": "T1.4",
        "replicate_id": 0,
        "treatment_id": "O",
        "validation_contract": contract_dict,
        "selection_rule": module.FORMAL_VALIDATION_SELECTION_RULE,
        "evaluations": evaluations,
        "selected_update": 50,
        "selected_checkpoint_sha256": best_sha256,
        "best_path": str(best_path),
        "best_sha256": best_sha256,
    }
    selection["selection_evidence_sha256"] = module.sha256_canonical_json(selection)
    selection_path = output / "checkpoint-selection.json"
    _rewrite(selection_path, selection)
    result = {
        "best_update": 50,
        "formal_checkpoint_selection": {
            "path": str(selection_path),
            "sha256": module.sha256_file(selection_path),
        },
    }
    module._require_completed_selection(job, result)  # noqa: SLF001
    assert validated_updates == list(module.FORMAL_VALIDATION_UPDATES)

    (output / "update-100.pth").write_bytes(b"tampered checkpoint")
    with pytest.raises(PilotOrchestrationError, match="extant checkpoint"):
        module._require_completed_selection(job, result)  # noqa: SLF001
    (output / "update-100.pth").write_bytes(bytes.fromhex("aa"))

    tampered = json.loads((output / "validation-update-2000.json").read_text())
    tampered["total_integer_wins"] = module.PILOT_VALIDATION_GAMES
    _rewrite(output / "validation-update-2000.json", tampered)
    with pytest.raises(PilotOrchestrationError, match="validation artifact"):
        module._require_completed_selection(job, result)  # noqa: SLF001


def test_launch_creates_detached_tmux_supervisor_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(
        module,
        "preflight",
        lambda _path: {
            "resources": {"output_bytes": 0, "free_bytes": MIN_FREE_BYTES}
        },
    )
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(module, "__name__", "__main__")
    handshakes: list[Path] = []
    monkeypatch.setattr(
        module,
        "_await_supervisor_handshake",
        lambda value: handshakes.append(value.path),
    )
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.launch(path)
    assert len(calls) == 1
    command = calls[0]
    assert command[:5] == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        declaration.tmux_session,
    ]
    assert "PYTHONHASHSEED=0" in command[-1]
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in command[-1]
    assert str(declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME) in command[-1]
    shell_tokens = shlex.split(command[-1])
    python = str(declaration.python_executable)
    assert shell_tokens.count(python) == 1
    python_index = shell_tokens.index(python)
    assert shell_tokens[python_index : python_index + 4] == [
        python,
        "-m",
        module.MODULE_NAME,
        "_supervise",
    ]
    assert str(declaration.control_dir / module.SUPERVISOR_LOG_FILE_NAME) in (
        shell_tokens
    )
    output_metadata = declaration.output_root.stat()
    assert shell_tokens[
        shell_tokens.index("--expected-output-root-device") + 1
    ] == str(output_metadata.st_dev)
    assert shell_tokens[
        shell_tokens.index("--expected-output-root-inode") + 1
    ] == str(output_metadata.st_ino)
    assert handshakes == [declaration.path]
    persisted = json.loads(declaration.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "launching"
    with pytest.raises(PilotOrchestrationError, match="already launched"):
        module.launch(path)


def test_preflight_requires_formal_journal_compressor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    with pytest.raises(PilotOrchestrationError, match="zstd executable"):
        module.preflight(path)


def test_launch_cli_handoff_uses_frozen_python_module_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(module, "launch", lambda _path: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["task1-pilot", "launch", str(declaration.path)],
    )

    module.main()

    attach_command = shlex.join(
        [
            str(declaration.python_executable),
            "-m",
            module.MODULE_NAME,
            "attach",
            str(declaration.path),
        ]
    )
    assert f"attach with: {attach_command}" in capsys.readouterr().out


def test_launch_blocks_when_tmux_dies_before_supervisor_handshake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(
        module,
        "preflight",
        lambda _path: {
            "resources": {"output_bytes": 0, "free_bytes": MIN_FREE_BYTES}
        },
    )
    liveness = iter((False, False))
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: next(liveness))
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    blocked: list[str] = []
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda _declaration, detail: blocked.append(detail),
    )
    monkeypatch.setattr(
        module,
        "capture_resources",
        lambda _root, **_kwargs: ResourceSnapshot(0, MIN_FREE_BYTES),
    )

    with pytest.raises(PilotOrchestrationError, match="before the running handshake"):
        module.launch(path)

    assert blocked == ["tmux supervisor exited before the running handshake"]
    persisted = json.loads(declaration.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"
    assert "before the running handshake" in cast(str, persisted["detail"])


def test_failed_supervisor_handshake_retries_manifest_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(
        module,
        "preflight",
        lambda _path: {
            "resources": {"output_bytes": 0, "free_bytes": MIN_FREE_BYTES}
        },
    )
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/tmux")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "_await_supervisor_handshake",
        lambda _declaration: (_ for _ in ()).throw(
            PilotOrchestrationError("child failed early")
        ),
    )
    blocked: list[str] = []
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda _declaration, detail: blocked.append(detail),
    )
    monkeypatch.setattr(
        module,
        "capture_resources",
        lambda _root, **_kwargs: ResourceSnapshot(0, MIN_FREE_BYTES),
    )

    with pytest.raises(PilotOrchestrationError, match="child failed early"):
        module.launch(path)

    assert blocked == ["child failed early"]
    assert calls[-1] == [
        "tmux",
        "kill-session",
        "-t",
        f"={declaration.tmux_session}",
    ]
    persisted = json.loads(declaration.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"


@pytest.mark.parametrize("entry_kind", ["regular", "symlink", "fifo"])
def test_preexisting_supervisor_log_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    log_path = declaration.control_dir / module.SUPERVISOR_LOG_FILE_NAME
    victim = declaration.control_dir / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    if entry_kind == "regular":
        log_path.write_text("occupied", encoding="utf-8")
    elif entry_kind == "symlink":
        log_path.symlink_to(victim)
    else:
        os.mkfifo(log_path)
    monkeypatch.setattr(
        module,
        "preflight",
        lambda _path: {
            "resources": {"output_bytes": 0, "free_bytes": MIN_FREE_BYTES}
        },
    )
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/tmux")
    blocked: list[str] = []
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda _declaration, detail: blocked.append(detail),
    )
    monkeypatch.setattr(
        module,
        "capture_resources",
        lambda _root, **_kwargs: ResourceSnapshot(0, MIN_FREE_BYTES),
    )

    with pytest.raises(PilotOrchestrationError, match="supervisor log already exists"):
        module.launch(path)

    assert len(blocked) == 1
    assert victim.read_text(encoding="utf-8") == "untouched"
    persisted = json.loads(declaration.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"


def test_status_rejects_self_hashed_truncated_running_record(
    tmp_path: Path,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    body: dict[str, object] = {
        "declaration_sha256": declaration.declaration_sha256,
        "state": "running",
    }
    body["status_sha256"] = module.sha256_canonical_json(body)
    _rewrite(declaration.status_path, body)

    with pytest.raises(PilotOrchestrationError, match="status keys"):
        module.status(path)


def test_status_accepts_exact_full_running_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    launched = write_pilot_declaration(
        declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME,
        payload,
    )
    record = module._status_payload(  # noqa: SLF001
        launched,
        "running",
        started_monotonic=0.0,
        resources=ResourceSnapshot(0, MIN_FREE_BYTES),
        child_pid=1235,
        supervisor_pid=1234,
    )
    _rewrite(declaration.status_path, record)
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: True)
    monkeypatch.setattr(module, "_tmux_pane_pid", lambda _session: 1234)

    snapshot = module.status(path)

    assert snapshot["persisted"] == record
    assert snapshot["tmux_alive"] is True
    assert snapshot["tmux_pane_pid"] == 1234


def test_status_rejects_unhashable_self_hashed_declaration_path(
    tmp_path: Path,
) -> None:
    path, payload = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    launched = write_pilot_declaration(
        declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME,
        payload,
    )
    record = module._status_payload(  # noqa: SLF001
        launched,
        "running",
        started_monotonic=0.0,
        resources=ResourceSnapshot(0, MIN_FREE_BYTES),
        child_pid=1235,
        supervisor_pid=1234,
    )
    record["declaration_path"] = []
    record.pop("status_sha256")
    record["status_sha256"] = module.sha256_canonical_json(record)
    _rewrite(declaration.status_path, record)

    with pytest.raises(PilotOrchestrationError, match="status identity"):
        module.status(path)


def test_running_handshake_binds_tmux_pane_to_supervisor_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(
        module,
        "status",
        lambda _path: {
            "persisted": {
                "state": "running",
                "supervisor_pid": 1234,
                "child_pid": 1235,
            },
            "tmux_alive": True,
            "tmux_pane_pid": 5678,
        },
    )

    with pytest.raises(PilotOrchestrationError, match="pane PID"):
        module._await_supervisor_handshake(declaration)  # noqa: SLF001


def test_completed_handshake_requires_manifest_and_completion_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    write_pilot_declaration(
        declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME,
        payload,
    )
    monkeypatch.setattr(
        module,
        "status",
        lambda _path: {
            "persisted": {
                "state": "completed",
                "detail": f"completion_sha256={'a' * 64}",
                "supervisor_pid": 1234,
                "child_pid": 1235,
            },
            "tmux_alive": False,
            "tmux_pane_pid": None,
        },
    )
    monkeypatch.setattr(
        module,
        "load_manifest",
        lambda _path: {"status": "running"},
    )

    with pytest.raises(PilotOrchestrationError, match="before the running handshake"):
        module._await_supervisor_handshake(declaration)  # noqa: SLF001


def test_tmux_pane_inspection_covers_the_entire_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "4321\n", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._tmux_pane_pid("pilot-session") == 4321  # noqa: SLF001
    assert calls == [
        [
            "tmux",
            "list-panes",
            "-s",
            "-t",
            "=pilot-session",
            "-F",
            "#{pane_pid}",
        ]
    ]


def test_launch_exec_failure_blocks_manifest_and_persists_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(
        module,
        "preflight",
        lambda _path: {
            "resources": {"output_bytes": 0, "free_bytes": MIN_FREE_BYTES}
        },
    )
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/tmux")
    blocked: list[str] = []
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda _declaration, detail: blocked.append(detail),
    )

    def fail_exec(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("tmux disappeared")

    monkeypatch.setattr(module.subprocess, "run", fail_exec)
    with pytest.raises(PilotOrchestrationError, match="tmux disappeared"):
        module.launch(path)
    assert len(blocked) == 1
    persisted = json.loads(declaration.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"
    assert "tmux disappeared" in cast(str, persisted["detail"])


def test_launch_snapshot_keeps_supervisor_bound_to_preflighted_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _payload(tmp_path)
    original = load_pilot_declaration(path)
    monkeypatch.setattr(
        module,
        "preflight",
        lambda _path: {
            "resources": {"output_bytes": 0, "free_bytes": MIN_FREE_BYTES}
        },
    )
    monkeypatch.setattr(module, "_tmux_has_session", lambda _session: False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(module, "_await_supervisor_handshake", lambda _value: None)
    module.launch(path)
    launch_path = original.control_dir / LAUNCH_DECLARATION_FILE_NAME
    launched = load_pilot_declaration(launch_path)
    assert launched.declaration_sha256 == original.declaration_sha256
    assert launched.manifest_path == original.manifest_path

    replacement_manifest = original.repository / "runs" / "other-manifest.json"
    payload["manifest_path"] = str(replacement_manifest)
    _rewrite(path, payload)
    blocked: list[Path] = []
    monkeypatch.setattr(
        module,
        "require_production_runtime",
        lambda _declaration: (_ for _ in ()).throw(
            PilotOrchestrationError("runtime admission failed")
        ),
    )
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda declaration, _detail: blocked.append(declaration.manifest_path),
    )
    with pytest.raises(PilotOrchestrationError, match="runtime admission failed"):
        module.supervise(
            launch_path,
            original.declaration_sha256,
            **_supervisor_binding(original),
        )
    assert blocked == [original.manifest_path]


def test_missing_launch_snapshot_still_closes_preflighted_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    original = load_pilot_declaration(path)
    launch_path = original.control_dir / LAUNCH_DECLARATION_FILE_NAME
    blocked: list[Path] = []
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda declaration, _detail: blocked.append(declaration.manifest_path),
    )
    monkeypatch.setattr(
        module,
        "capture_resources",
        lambda _root, **_kwargs: ResourceSnapshot(0, MIN_FREE_BYTES),
    )
    with pytest.raises(PilotOrchestrationError, match="cannot inspect declaration"):
        module.supervise(
            launch_path,
            original.declaration_sha256,
            **_supervisor_binding(original),
        )
    assert blocked == [original.manifest_path]
    persisted = json.loads(original.status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "failed"


def test_supervisor_marks_process_failure_and_blocks_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    monkeypatch.setattr(module, "__name__", "__main__")
    monkeypatch.setattr(module, "require_production_runtime", lambda _value: None)
    monkeypatch.setattr(
        module,
        "capture_resources",
        lambda _root, **_kwargs: ResourceSnapshot(0, MIN_FREE_BYTES),
    )
    blocked: list[str] = []
    monkeypatch.setattr(
        module,
        "_block_running_manifest",
        lambda _declaration, detail: blocked.append(detail),
    )

    class FailedProcess:
        pid = 4321

        @staticmethod
        def poll() -> int:
            return 17

    popen_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(
        command: list[str], **kwargs: object
    ) -> FailedProcess:
        popen_calls.append((command, kwargs))
        return FailedProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    with pytest.raises(PilotOrchestrationError, match="exited 17"):
        module.supervise(
            path,
            declaration.declaration_sha256,
            **_supervisor_binding(declaration),
        )
    command, kwargs = popen_calls[0]
    assert command[:3] == [
        str(declaration.python_executable),
        "-m",
        module.MODULE_NAME,
    ]
    assert "--supervisor-pid" in command
    assert "--supervisor-fd" in command
    assert "--output-root-device" in command
    assert "--output-root-inode" in command
    assert kwargs["start_new_session"] is True
    assert len(cast(tuple[int], kwargs["pass_fds"])) == 1
    assert blocked == ["formal matrix process exited 17"]
    status = json.loads(declaration.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "exited 17" in status["detail"]


def test_supervisor_failure_closes_only_reserved_job_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _payload(tmp_path)
    declaration = load_pilot_declaration(path)
    output_root = declaration.output_root
    output_root.mkdir(parents=True)
    outputs: list[dict[str, object]] = []
    output_paths: list[Path] = []
    for replicate_id in PILOT_REPLICATE_IDS:
        for treatment_id in PILOT_TREATMENT_IDS:
            output_dir = output_root / f"r{replicate_id}-{treatment_id}"
            output_dir.mkdir()
            (output_dir / module.ppo_selfplay_module.FORMAL_OUTPUT_RESERVATION_NAME).write_text(
                "reserved\n", encoding="utf-8"
            )
            outputs.append(
                {
                    "replicate_id": replicate_id,
                    "treatment_id": treatment_id,
                    "output_dir": str(output_dir),
                }
            )
            output_paths.append(output_dir)

    running_status = {
        "schema_version": module.ppo_selfplay_module.FORMAL_STATUS_SCHEMA,
        "status": "running",
        "update": 1432,
        "updates": 2000,
        "best_update": 1300,
        "best_validation_score": [40, 60],
        "update_seconds": 1.5,
        "elapsed_seconds": 10.0,
        "error": None,
        "phase_seconds": {"rollout": 1.0},
        "journal_entries": 1433,
    }
    (output_paths[0] / "status.json").write_text(
        json.dumps(running_status), encoding="utf-8"
    )
    queued_status = {"status": "queued", "update": 0, "sentinel": "keep"}
    (output_paths[2] / "status.json").write_text(
        json.dumps(queued_status), encoding="utf-8"
    )
    completed_bytes = b'{"status":"completed","sentinel":1}\n'
    failed_bytes = b'{"status":"failed","sentinel":2}\n'
    (output_paths[3] / "status.json").write_bytes(completed_bytes)
    (output_paths[4] / "status.json").write_bytes(failed_bytes)

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink_output = output_paths[5]
    target_status = symlink_target / "status.json"
    target_status.write_bytes(b'{"status":"running","sentinel":"target"}\n')
    (symlink_output / "status.json").symlink_to(target_status)
    stray = output_root / "unreserved-stray"
    stray.mkdir()
    (stray / "status.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        module,
        "load_manifest",
        lambda _path: {"declaration": {"formal_training": {"job_outputs": outputs}}},
    )
    issue = module._stamp_reserved_job_statuses(  # noqa: SLF001
        declaration, "18-hour wall-clock limit exceeded"
    )

    assert issue is not None
    interrupted = json.loads((output_paths[0] / "status.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["update"] == 1432
    assert interrupted["best_update"] == 1300
    assert interrupted["error"] == "18-hour wall-clock limit exceeded"
    not_started = json.loads((output_paths[1] / "status.json").read_text())
    assert not_started["status"] == "not-started"
    assert not_started["error"] == "18-hour wall-clock limit exceeded"
    queued = json.loads((output_paths[2] / "status.json").read_text())
    assert queued["status"] == "not-started"
    assert queued["sentinel"] == "keep"
    assert (output_paths[3] / "status.json").read_bytes() == completed_bytes
    assert (output_paths[4] / "status.json").read_bytes() == failed_bytes
    assert target_status.read_bytes() == b'{"status":"running","sentinel":"target"}\n'
    assert json.loads((stray / "status.json").read_text())["status"] == "failed"


def test_formal_result_rejects_missing_or_tampered_update_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "job"
    output.mkdir()
    config = PPOConfig(updates=2, games_per_update=16)
    formal_spec = SimpleNamespace(
        experiment_id="experiment",
        phase="T1.4",
        replicate_id=0,
        treatment_id="O",
    )
    job = SimpleNamespace(output_dir=output, config=config, formal_spec=formal_spec)
    metadata = {
        "schema_version": "journal/1",
        "path": str(output / module.ppo_selfplay_module.FORMAL_UPDATE_JOURNAL_NAME),
        "entries": 3,
        "first_update": 0,
        "last_update": 2,
        "last_entry_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "size_bytes": 3,
    }
    result = {
        "schema_version": module.ppo_selfplay_module.FORMAL_RESULT_SCHEMA,
        "formal_update_journal": metadata,
        "aggregate_opponent_pool_usage": {
            "actual_counts": {"ga": 32},
            "draws": 32,
        },
    }
    journal_path = output / module.ppo_selfplay_module.FORMAL_UPDATE_JOURNAL_NAME
    journal_path.write_text("good\n", encoding="utf-8")

    def validate(path: Path, *_args: object, **_kwargs: object) -> dict[str, object]:
        if not path.is_file() or path.read_text(encoding="utf-8") != "good\n":
            raise ValueError("tampered journal")
        return metadata

    monkeypatch.setattr(
        module.ppo_selfplay_module,
        "validate_formal_update_journal",
        validate,
    )
    assert module._require_formal_result_journal(job, result, result) == metadata  # noqa: SLF001

    journal_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(PilotOrchestrationError, match="update journal is invalid"):
        module._require_formal_result_journal(job, result, result)  # noqa: SLF001

    journal_path.write_text("good\n", encoding="utf-8")
    tampered_result = {
        **result,
        "formal_update_journal": {**metadata, "entries": 99},
    }
    with pytest.raises(PilotOrchestrationError, match="does not bind"):
        module._require_formal_result_journal(  # noqa: SLF001
            job, tampered_result, tampered_result
        )


def test_completed_formal_status_requires_exact_v2_schema() -> None:
    # The spawn worker may still hold a tuple while its JSON status is a list;
    # completion must compare their canonical JSON value, not Python container
    # identity.
    result = {"best_update": 1300, "best_validation_score": (40, 60)}
    journal = {"entries": 2001}
    status: dict[str, object] = {
        "schema_version": module.ppo_selfplay_module.FORMAL_STATUS_SCHEMA,
        "status": "completed",
        "update": module.PILOT_UPDATES,
        "updates": module.PILOT_UPDATES,
        "best_update": 1300,
        "best_validation_score": [40, 60],
        "update_seconds": 1.0,
        "elapsed_seconds": 2.0,
        "error": None,
        "phase_seconds": {
            "rollout": 1.0,
            "optimizer": 0.1,
            "checkpoint": 0.1,
            "validation": 0.1,
            "validation_evidence": 0.1,
            "selection": 0.1,
            "pruning": 0.1,
            "journal": 0.1,
        },
        "journal_entries": 2001,
    }
    module._require_formal_completed_status(status, result, journal)  # noqa: SLF001

    status.pop("schema_version")
    with pytest.raises(PilotOrchestrationError, match="schema"):
        module._require_formal_completed_status(status, result, journal)  # noqa: SLF001


def test_matrix_cli_requires_and_consumes_inherited_supervisor_pipe(
    tmp_path: Path,
) -> None:
    read_fd, write_fd = os.pipe()
    missing = tmp_path / "missing-declaration.json"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                module.MODULE_NAME,
                "_run-matrix",
                str(missing),
                "--expected-sha256",
                "a" * 64,
                "--supervisor-pid",
                str(os.getpid()),
                "--supervisor-fd",
                str(read_fd),
                "--output-root-device",
                "1",
                "--output-root-inode",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(read_fd,),
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert result.returncode == 2
    assert "cannot inspect declaration" in result.stderr
    assert "supervisor lease" not in result.stderr
