"""Read-only T1.4 pilot monitor tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from splendor.agents.our_agents.policy_imitation import pilot_monitor as module
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    FORMAL_UPDATE_JOURNAL_NAME,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_monitor_journal_name_matches_formal_trainer() -> None:
    assert module.UPDATE_JOURNAL_NAME == FORMAL_UPDATE_JOURNAL_NAME


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    control = root / "control"
    output_root = root / "training"
    control.mkdir(parents=True)
    experiment_id = "task1-monitor-fixture"
    treatments = [
        {"treatment_id": "O", "config": {"updates": 10}},
        {"treatment_id": "O_bridge", "config": {"updates": 10}},
    ]
    declaration = {
        "schema_version": "splendor-t14-pilot-orchestration/2",
        "experiment_id": experiment_id,
        "phase": "T1.4",
        "manifest_path": str(root / "manifest.json"),
        "control_dir": str(control),
        "output_root": str(output_root),
        "tmux_session": "task1-monitor-fixture",
        "treatments": treatments,
    }
    declaration_path = control / "orchestration.json"
    _write_json(declaration_path, declaration)
    jobs = [
        {
            "replicate_id": replicate,
            "treatment_id": treatment,
            "output_dir": str(output_root / f"r{replicate}-{treatment}"),
        }
        for replicate in (0, 1, 2)
        for treatment in ("O", "O_bridge")
    ]
    _write_json(
        root / "manifest.json",
        {
            "status": "running",
            "declaration": {
                "experiment_id": experiment_id,
                "formal_training": {"job_outputs": jobs},
            },
        },
    )
    return declaration_path


def _status_payload(
    declaration_path: Path,
    *,
    state: str = "running",
    manifest_status: str | None = None,
    resources: dict[str, int] | None = None,
) -> dict[str, Any]:
    declaration = module._load_declaration(declaration_path)  # noqa: SLF001
    body: dict[str, Any] = {
        "schema_version": "splendor-t14-pilot-status/1",
        "declaration_path": str(declaration.path),
        "declaration_sha256": declaration.digest,
        "experiment_id": declaration.experiment_id,
        "state": state,
        "tmux_session": declaration.tmux_session,
        "updated_unix_seconds": 1_700_000_000.0,
        "detail": None,
    }
    if resources is not None:
        body["resources"] = resources
    if state in {"running", "completed"}:
        body["elapsed_seconds"] = 12.5
    body["status_sha256"] = module._canonical_digest(body)  # noqa: SLF001
    del manifest_status
    return body


def _write_job(  # noqa: PLR0913
    declaration_path: Path,
    replicate: int,
    treatment: str,
    *,
    status: str,
    update: int,
    updates: int = 10,
    journal: str | None = None,
) -> Path:
    declaration = module._load_declaration(declaration_path)  # noqa: SLF001
    output = declaration.output_root / f"r{replicate}-{treatment}"
    output.mkdir(parents=True, exist_ok=True)
    status_payload: dict[str, object] = {
        "status": status,
        "update": update,
        "updates": updates,
        "best_update": update,
        "best_validation_score": None,
        "update_seconds": 1.0,
        "elapsed_seconds": 12.5,
        "error": None,
    }
    if journal == "zstd":
        status_payload.update(
            {
                "schema_version": "splendor-formal-ppo-status/2",
                "journal_entries": update + 1,
            }
        )
        (output / module.UPDATE_JOURNAL_NAME).write_bytes(b"zstd-frame-fixture")
    elif journal == "plain":
        (output / module.LEGACY_UPDATE_JOURNAL_NAME).write_text(
            "", encoding="utf-8"
        )
    _write_json(output / "status.json", status_payload)
    _write_json(
        output / "result.json",
        {
            "status": status,
            "best_update": update,
            "config": {"updates": updates},
        },
    )
    return output


def test_snapshot_counts_progress_resources_and_legacy_journal(
    tmp_path: Path,
) -> None:
    declaration_path = _fixture(tmp_path)
    _write_job(declaration_path, 0, "O", status="running", update=3)
    _write_job(
        declaration_path,
        0,
        "O_bridge",
        status="completed",
        update=10,
        journal="zstd",
    )
    _write_job(
        declaration_path,
        1,
        "O",
        status="failed",
        update=4,
        journal="plain",
    )
    declaration = module._load_declaration(declaration_path)  # noqa: SLF001
    status = _status_payload(
        declaration_path,
        resources={"output_bytes": 123, "free_bytes": 456},
    )
    _write_json(declaration.status_path, status)

    before = declaration.status_path.read_bytes()
    observed = module.snapshot(declaration_path)
    assert declaration.status_path.read_bytes() == before
    assert observed["state"] == "running"
    assert observed["complete"] is False
    assert observed["counts"] == {
        "total": 6,
        "active": 1,
        "queued": 3,
        "terminal": 2,
        "unknown": 0,
    }
    assert observed["progress"] == {
        "current_updates": 17,
        "target_updates": 60,
        "fraction": 17 / 60,
        "percent": 100 * 17 / 60,
        "complete": False,
    }
    assert observed["output_bytes"] == 123
    assert observed["free_bytes"] == 456
    observed_jobs = cast(list[dict[str, Any]], observed["jobs"])
    jobs = {job["key"]: job for job in observed_jobs}
    assert jobs["r0-O"]["journal"]["compatibility"] == "legacy-no-journal"
    assert jobs["r0-O_bridge"]["journal"]["compatibility"] == "formal-journal-zstd"
    assert jobs["r0-O_bridge"]["journal"]["entries"] == 11
    assert jobs["r1-O"]["journal"]["compatibility"] == "formal-journal"


def test_large_result_is_bounded_and_old_r3_without_journal_is_supported(
    tmp_path: Path,
) -> None:
    declaration_path = _fixture(tmp_path)
    output = _write_job(
        declaration_path,
        0,
        "O",
        status="running",
        update=2,
    )
    _write_job(declaration_path, 0, "O_bridge", status="running", update=0)
    large = {"status": "running", "blob": "x" * (module._MAX_FULL_RESULT_BYTES + 1)}  # noqa: SLF001
    _write_json(output / "result.json", large)
    observed = module.snapshot(declaration_path)
    observed_jobs = cast(list[dict[str, Any]], observed["jobs"])
    job = next(job for job in observed_jobs if job["key"] == "r0-O")
    assert job["result"]["read_mode"] == "bounded-edges"
    assert job["result"]["status"] == "running"
    assert job["journal"]["compatibility"] == "legacy-no-journal"


def test_not_started_is_terminal_but_never_counts_as_completion(
    tmp_path: Path,
) -> None:
    declaration_path = _fixture(tmp_path)
    _write_job(
        declaration_path,
        0,
        "O",
        status="not-started",
        update=0,
    )
    observed = module.snapshot(declaration_path)
    jobs = cast(list[dict[str, Any]], observed["jobs"])
    not_started = next(job for job in jobs if job["key"] == "r0-O")
    assert not_started["classification"] == "terminal"
    assert cast(dict[str, int], observed["counts"])["terminal"] == 1
    assert observed["complete"] is False


def test_invalid_status_hash_and_symlink_are_explicit_errors(tmp_path: Path) -> None:
    declaration_path = _fixture(tmp_path)
    declaration = module._load_declaration(declaration_path)  # noqa: SLF001
    _write_json(
        declaration.status_path,
        {"state": "running", "status_sha256": "not-the-body-hash"},
    )
    with pytest.raises(module.PilotMonitorError, match="status hash"):
        module.snapshot(declaration_path)

    declaration.status_path.unlink()
    target = tmp_path / "status-target.json"
    target.write_text("{}", encoding="utf-8")
    declaration.status_path.symlink_to(target)
    with pytest.raises(module.PilotMonitorError, match="regular non-symlink"):
        module.snapshot(declaration_path)


def test_completed_requires_explicit_orchestrator_and_manifest_evidence(
    tmp_path: Path,
) -> None:
    declaration_path = _fixture(tmp_path)
    declaration = module._load_declaration(declaration_path)  # noqa: SLF001
    for replicate in (0, 1, 2):
        for treatment in ("O", "O_bridge"):
            _write_job(
                declaration_path,
                replicate,
                treatment,
                status="completed",
                update=10,
                journal="zstd",
            )
    status = _status_payload(declaration_path, state="completed")
    _write_json(declaration.status_path, status)
    observed = module.snapshot(declaration_path)
    assert observed["state"] == "completed"
    assert observed["complete"] is False
    progress = cast(dict[str, Any], observed["progress"])
    assert progress["complete"] is False

    manifest = json.loads(declaration.manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    _write_json(declaration.manifest_path, manifest)
    jobs: list[dict[str, object]] = []
    for replicate in (0, 1, 2):
        for treatment in ("O", "O_bridge"):
            output = declaration.output_root / f"r{replicate}-{treatment}"
            files: dict[str, str] = {}
            for name in module._completion_file_names():  # noqa: SLF001
                artifact = output / name
                if not artifact.exists():
                    artifact.write_bytes(f"{replicate}/{treatment}/{name}".encode())
                files[name] = module._sha256_regular_file(artifact)  # noqa: SLF001
            jobs.append(
                {
                    "replicate_id": replicate,
                    "treatment_id": treatment,
                    "output_dir": str(output),
                    "files": files,
                }
            )
    completion: dict[str, object] = {
        "schema_version": "splendor-t14-pilot-completion/2",
        "declaration_path": str(declaration.path),
        "declaration_sha256": declaration.digest,
        "experiment_id": declaration.experiment_id,
        "manifest_path": str(declaration.manifest_path),
        "seed_roll_payload_sha256": "1" * 64,
        "scenario_bank_payload_sha256": "2" * 64,
        "validation_bank_payload_sha256": "3" * 64,
        "formal_training_sha256": "4" * 64,
        "jobs": jobs,
    }
    completion["completion_sha256"] = module._canonical_digest(completion)  # noqa: SLF001
    _write_json(declaration.completion_path, completion)
    observed = module.snapshot(declaration_path)
    assert observed["complete"] is True
    completion_summary = cast(dict[str, object], observed["completion"])
    assert completion_summary["files_verified"] == 6 * len(
        module._completion_file_names()  # noqa: SLF001
    )

    (declaration.output_root / "r0-O" / "result.json").write_text(
        '{"status":"tampered"}\n',
        encoding="utf-8",
    )
    tampered = module.snapshot(declaration_path)
    assert tampered["complete"] is False
    tampered_completion = cast(dict[str, object], tampered["completion"])
    assert tampered_completion["valid"] is False
    assert "hash changed" in cast(str, tampered_completion["error"])


def test_completed_rejects_update_beyond_target(tmp_path: Path) -> None:
    declaration_path = _fixture(tmp_path)
    _write_job(
        declaration_path,
        0,
        "O",
        status="completed",
        update=11,
        updates=10,
    )
    observed = module.snapshot(declaration_path)
    jobs = cast(list[dict[str, Any]], observed["jobs"])
    job = next(item for item in jobs if item["key"] == "r0-O")
    assert job["issues"] == ["update exceeds target"]
    assert observed["complete"] is False


def test_cli_emits_one_compact_json_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    declaration_path = _fixture(tmp_path)
    module.main([str(declaration_path)])
    line = capsys.readouterr().out.strip()
    parsed = json.loads(line)
    assert parsed["schema_version"] == module.MONITOR_SCHEMA
    assert "jobs" in parsed
    assert "risk_budget" in parsed
