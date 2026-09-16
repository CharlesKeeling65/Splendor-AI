"""Adversarial tests for the formal PPO update journal boundary.

These tests deliberately exercise the journal and checkpoint-audit seam with
small temporary artifacts.  They do not invoke the trainer or touch ``runs/``.
The journal is the durable source of per-update evidence; ``result.json`` and
checkpoint metrics are expected to remain compact summaries of that evidence.
"""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from splendor.agents.our_agents.policy_imitation import ppo_selfplay as ppo_module
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    FORMAL_UPDATE_JOURNAL_NAME,
    OpponentPoolEntry,
    _checkpoint_pool_distribution,
    _compact_json_bytes,
    _FormalUpdateJournal,
    _prune_formal_update_checkpoints,
    validate_formal_update_journal,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class _FirstActionAgent(Agent):
    """Minimal factory shape needed by weighted-pool provenance tests."""

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        return actions[0]


def _formal_identity() -> SimpleNamespace:
    """Return the identity fields the journal validator binds per game."""
    protocol = {
        "protocol": "paired-training-v2",
        "experiment_id": "journal-adversarial",
        "phase": "T1.4",
        "replicate_id": 3,
        "treatment_id": "fixed",
    }

    def as_dict() -> dict[str, Any]:
        return protocol

    return SimpleNamespace(
        experiment_id="journal-adversarial",
        phase="T1.4",
        replicate_id=3,
        treatment_id="fixed",
        as_dict=as_dict,
    )


def _training_game(
    formal: SimpleNamespace,
    *,
    update: int,
    game_index: int,
    opponent: str = "ga",
) -> dict[str, Any]:
    """Build a completed terminal record with the public game accounting fields."""
    return {
        "status": "completed",
        "update": update,
        "game_index": game_index,
        "experiment_id": formal.experiment_id,
        "phase": formal.phase,
        "replicate_id": formal.replicate_id,
        "treatment_id": formal.treatment_id,
        "opponent": opponent,
        "failure": None,
        "outcome": 1 if game_index % 2 == 0 else -1,
        "score": 15.0,
        "rival_score": 10.0,
        "plies": 42,
        "student_queries": 9,
        "action_trace_sha256": "a" * 64,
    }


def _initial_record(formal: SimpleNamespace) -> dict[str, Any]:
    return {
        "update": 0,
        "status": "initial",
        "formal_protocol": formal.as_dict(),
        "training_records": [],
        "training_games": 0,
        "training_failed_games": 0,
        "teacher_queries": 0,
        "opponent_names": [],
        "opponent_pool": None,
        "update_metrics": None,
        "critic_warmup": None,
        "validation": None,
        "elapsed_seconds": 0.0,
        "phase_seconds": {
            "checkpoint": 0.0,
            "validation": 0.0,
            "validation_evidence": 0.0,
            "selection": 0.0,
        },
    }


def _completed_record(
    formal: SimpleNamespace,
    *,
    update: int,
    games_per_update: int,
) -> dict[str, Any]:
    games = [
        _training_game(formal, update=update, game_index=index)
        for index in range(games_per_update)
    ]
    return {
        "update": update,
        "status": "completed",
        "training_records": games,
        "training_games": games_per_update,
        "training_failed_games": 0,
        "teacher_queries": 0,
        "opponent_names": ["ga"],
        "opponent_pool": None,
        "update_metrics": {},
        "critic_warmup": None,
        "validation": None,
        "rollout_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "phase_seconds": {
            "rollout": 0.0,
            "optimizer": 0.0,
            "checkpoint": 0.0,
            "validation": 0.0,
            "validation_evidence": 0.0,
            "selection": 0.0,
            "pruning": 0.0,
        },
    }


def _make_valid_journal(
    path: Path,
    *,
    updates: int = 2,
    games_per_update: int = 2,
) -> SimpleNamespace:
    formal = _formal_identity()
    journal = _FormalUpdateJournal.create(path)
    journal.append(_initial_record(formal))
    for update in range(1, updates + 1):
        journal.append(
            _completed_record(
                formal,
                update=update,
                games_per_update=games_per_update,
            )
        )
    return formal


def _read_lines(path: Path) -> list[bytes]:
    result = subprocess.run(
        ["zstd", "--quiet", "--decompress", "--stdout"],
        input=path.read_bytes(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout.splitlines(keepends=True)


def _write_lines(path: Path, lines: list[bytes]) -> None:
    frames: list[bytes] = []
    for line in lines:
        result = subprocess.run(
            ["zstd", "--quiet", "--compress", "--stdout", "--threads=1", "-1"],
            input=line,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        frames.append(result.stdout)
    path.write_bytes(b"".join(frames))


def _rewrite_tail_record(
    path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """Make a self-consistent tail entry after changing its semantic payload."""
    lines = _read_lines(path)
    entry = json.loads(lines[-1])
    mutate(entry["record"]["training_records"][0])
    body = dict(entry)
    body.pop("entry_sha256")
    entry["entry_sha256"] = hashlib.sha256(_compact_json_bytes(body)).hexdigest()
    lines[-1] = _compact_json_bytes(entry) + b"\n"
    _write_lines(path, lines)


def test_valid_initial_and_contiguous_completed_chain_round_trips(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path, updates=2, games_per_update=2)

    validated = validate_formal_update_journal(
        path,
        formal,
        expected_updates=2,
        games_per_update=2,
    )
    journal = _FormalUpdateJournal(
        path=path,
        device=path.stat().st_dev,
        inode=path.stat().st_ino,
        zstd_executable=shutil.which("zstd") or "zstd",
        entries=3,
        last_entry_sha256=validated["last_entry_sha256"],
    )

    assert validated["entries"] == 3
    assert validated["first_update"] == 0
    assert validated["last_update"] == 2
    assert validated["artifact_sha256"] == journal.metadata(completed=True)[
        "artifact_sha256"
    ]
    assert validated["size_bytes"] == path.stat().st_size
    assert all(line.endswith(b"\n") for line in _read_lines(path))


def test_journal_binds_result_opponent_usage(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path, updates=2, games_per_update=2)
    expected = {"actual_counts": {"ga": 4}, "draws": 4}

    validate_formal_update_journal(
        path,
        formal,
        expected_updates=2,
        games_per_update=2,
        expected_opponent_usage=expected,
    )

    with pytest.raises(ValueError, match="opponent usage"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
            expected_opponent_usage={"actual_counts": {"ga": 3}, "draws": 3},
        )


@pytest.mark.parametrize("mutation", ["truncate", "reorder", "duplicate", "malformed"])
def test_journal_rejects_structural_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)
    lines = _read_lines(path)
    if mutation == "truncate":
        lines[-1] = lines[-1].rstrip(b"\n")[:-1]
    elif mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "duplicate":
        lines.append(lines[-1])
    else:
        lines[1] = b"not-json\n"
    _write_lines(path, lines)

    with pytest.raises(ValueError):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )


def test_journal_rejects_truncated_zstd_frame(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)
    compressed = path.read_bytes()
    path.write_bytes(compressed[:-1])

    with pytest.raises(ValueError, match=r"decompress|invalid|truncated"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )


def test_journal_rejects_self_hash_and_record_tamper(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)
    lines = _read_lines(path)
    entry = json.loads(lines[-1])
    entry["entry_sha256"] = "0" * 64
    lines[-1] = _compact_json_bytes(entry) + b"\n"
    _write_lines(path, lines)

    with pytest.raises(ValueError, match="chain"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )

    path.unlink()
    _make_valid_journal(path)
    lines = _read_lines(path)
    entry = json.loads(lines[-1])
    entry["record"]["training_records"][0]["score"] = 999.0
    lines[-1] = _compact_json_bytes(entry) + b"\n"
    _write_lines(path, lines)
    with pytest.raises(ValueError, match="chain"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )


@pytest.mark.parametrize("noncanonical", ["whitespace", "duplicate-key"])
def test_journal_rejects_noncanonical_json(
    tmp_path: Path,
    noncanonical: str,
) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)
    lines = _read_lines(path)
    if noncanonical == "whitespace":
        parsed = json.loads(lines[1])
        lines[1] = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(", ", ": "),
        ).encode("utf-8") + b"\n"
    else:
        lines[1] = lines[1].replace(
            b'"index":1,', b'"index":1,"index":1,', 1
        )
    _write_lines(path, lines)

    with pytest.raises(ValueError, match=r"(schema|chain|canonical)"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )


def test_journal_rejects_numeric_coordinate_coercion(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)
    lines = _read_lines(path)
    entry = json.loads(lines[-1])
    # JSON's 2.0 compares equal to Python's 2, but is not the exact integer
    # coordinate emitted by the journal contract.  Recompute the tail hash so
    # this probes semantic/type validation rather than hash-chain detection.
    entry["index"] = 2.0
    entry["update"] = 2.0
    entry["record"]["update"] = 2.0
    body = dict(entry)
    body.pop("entry_sha256")
    entry["entry_sha256"] = hashlib.sha256(_compact_json_bytes(body)).hexdigest()
    lines[-1] = _compact_json_bytes(entry) + b"\n"
    _write_lines(path, lines)

    with pytest.raises(ValueError, match=r"(chain|coordinate|record)"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )


@pytest.mark.parametrize("field", ["opponent", "outcome", "score", "rival_score"])
def test_completed_terminal_records_are_required_for_aggregate_audit(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)

    def remove_field(game: dict[str, Any]) -> None:
        game.pop(field)

    _rewrite_tail_record(path, remove_field)

    with pytest.raises(ValueError, match="training record"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
        )


def test_journal_append_refuses_out_of_order_update(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    journal = _FormalUpdateJournal.create(path)

    with pytest.raises(RuntimeError, match=r"updates 0\.\.N"):
        journal.append({"update": 1, "status": "initial"})


def test_journal_append_detects_inode_replacement(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    journal = _FormalUpdateJournal.create(path)
    old_path = tmp_path / "updates-old.jsonl"
    path.rename(old_path)
    path.write_bytes(old_path.read_bytes())

    with pytest.raises(RuntimeError, match="identity changed"):
        journal.append(_initial_record(_formal_identity()))


def test_journal_validator_detects_bound_inode_replacement(tmp_path: Path) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    formal = _make_valid_journal(path)
    metadata = path.stat()
    replacement = tmp_path / "replacement.zst"
    path.rename(replacement)
    path.write_bytes(replacement.read_bytes())

    with pytest.raises(ValueError, match="identity changed"):
        validate_formal_update_journal(
            path,
            formal,
            expected_updates=2,
            games_per_update=2,
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )


def test_journal_metadata_is_compact_and_defers_full_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / FORMAL_UPDATE_JOURNAL_NAME
    journal = _FormalUpdateJournal.create(path)
    journal.append(_initial_record(_formal_identity()))

    def unexpected_hash(_path: Path) -> str:
        raise AssertionError("partial journal metadata must not hash the artifact")

    monkeypatch.setattr(ppo_module, "sha256_file", unexpected_hash)
    partial = journal.metadata(completed=False)
    assert partial["artifact_sha256"] is None
    assert partial["entries"] == 1
    assert partial["size_bytes"] == path.stat().st_size
    assert set(partial) == {
        "schema_version",
        "path",
        "entries",
        "first_update",
        "last_update",
        "last_entry_sha256",
        "artifact_sha256",
        "size_bytes",
    }


def _pool_entry(name: str) -> OpponentPoolEntry:
    return OpponentPoolEntry(
        name,
        CandidateSpec(
            name,
            "fixed_baseline",
            _FirstActionAgent,
        ),
    )


def test_formal_aggregate_opponent_usage_round_trips_into_checkpoint_audit() -> None:
    entries = [_pool_entry("ga"), _pool_entry("minimax")]
    metrics = {
        "aggregate_opponent_pool_usage": {
            "actual_counts": {"ga": 3, "minimax": 1},
            "draws": 4,
        }
    }

    distribution, usage = _checkpoint_pool_distribution(entries, metrics)

    # The distribution is the latest pool CDF; all-training provenance lives in
    # the separate aggregate usage record and must retain the full counts.
    assert distribution["actual_counts"] == {"ga": 0, "minimax": 0}
    assert distribution["draws"] == 0
    assert usage == {
        "count_scope": "all-training-records-in-checkpoint-metrics",
        "actual_counts": {"ga": 3, "minimax": 1},
        "draws": 4,
    }


def test_formal_aggregate_opponent_usage_rejects_unknown_name() -> None:
    entries = [_pool_entry("ga")]
    metrics = {
        "aggregate_opponent_pool_usage": {
            "actual_counts": {"ga": 1, "forged-opponent": 1},
            "draws": 2,
        }
    }

    with pytest.raises(ValueError, match=r"unknown|pool"):
        _checkpoint_pool_distribution(entries, metrics)


def test_formal_aggregate_usage_cannot_mix_raw_draws() -> None:
    entries = [_pool_entry("ga")]
    metrics = {
        "training_records": [{"opponent": "ga"}],
        "aggregate_opponent_pool_usage": {
            "actual_counts": {"ga": 1},
            "draws": 1,
        },
    }

    with pytest.raises(ValueError, match="mix raw"):
        _checkpoint_pool_distribution(entries, metrics)


def test_pruning_keeps_history_window_validation_and_terminal_artifacts(
    tmp_path: Path,
) -> None:
    for update in range(1, 7):
        (tmp_path / f"update-{update}.pth").write_bytes(f"update-{update}".encode())
    for name in (
        "initial.pth",
        "best.pth",
        "final.pth",
        "validation-update-2.json",
        "checkpoint-selection.json",
        FORMAL_UPDATE_JOURNAL_NAME,
    ):
        (tmp_path / name).write_bytes(name.encode())

    removed = _prune_formal_update_checkpoints(
        tmp_path,
        # update 2 is a validation milestone; 4..6 are the last-K history.
        keep_updates={2, 4, 5, 6},
    )

    assert removed == (1, 3)
    assert [
        update
        for update in range(1, 7)
        if (tmp_path / f"update-{update}.pth").exists()
    ] == [2, 4, 5, 6]
    for name in (
        "initial.pth",
        "best.pth",
        "final.pth",
        "validation-update-2.json",
        "checkpoint-selection.json",
        FORMAL_UPDATE_JOURNAL_NAME,
    ):
        assert (tmp_path / name).exists()


def test_pruning_fails_closed_on_special_checkpoint_file(tmp_path: Path) -> None:
    target = tmp_path / "target.pth"
    target.write_bytes(b"checkpoint")
    (tmp_path / "update-1.pth").symlink_to(target)

    with pytest.raises(RuntimeError, match="special file"):
        _prune_formal_update_checkpoints(tmp_path, keep_updates=set())


def test_pruning_binds_the_reserved_output_directory_inode(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    metadata = output.stat()
    displaced = tmp_path / "displaced"
    output.rename(displaced)
    output.mkdir()
    (output / "update-1.pth").write_bytes(b"replacement")

    with pytest.raises(RuntimeError, match="directory identity changed"):
        _prune_formal_update_checkpoints(
            output,
            keep_updates=set(),
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )

    assert (output / "update-1.pth").read_bytes() == b"replacement"
