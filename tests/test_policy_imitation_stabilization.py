"""Small, non-training tests for the PPO stabilisation driver."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from splendor.agents.our_agents.policy_imitation.stabilization import (
    StabilizationConfig,
    audit_evaluation_matrix,
    build_arg_parser,
    build_seed_groups,
    configure_worker_hash_seed,
    expected_validation_updates,
    make_training_jobs,
    parse_baseline_spec,
    ppo_config_kwargs,
    prepare_output_dir,
    summarize_test_records,
    trainer_api_status,
    verify_seed_groups,
)


def test_spawned_interpreters_share_hash_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    configure_worker_hash_seed()
    assert os.environ["PYTHONHASHSEED"] == "0"
    command = [sys.executable, "-c", "print(hash('splendor-action-order'))"]
    fingerprints = [subprocess.check_output(command, text=True) for _ in range(3)]
    assert len(set(fingerprints)) == 1


def _smoke_config(tmp_path: Path) -> StabilizationConfig:
    return StabilizationConfig(
        output=tmp_path / "run",
        initial_bc=tmp_path / "bc.pth",
        device="cpu",
        workers=1,
        updates=1,
        games_per_update=2,
        validation_deals=1,
        test_deals=1,
        seeds=(42, 1234),
        variants=("fixed", "scratch"),
        smoke=True,
        seed_base=824000,
    )


def test_baseline_parser_keeps_checkpoint_sources_serializable() -> None:
    spec = parse_baseline_spec("dqn42:dqn:/models/corrected-42/best.pth")

    assert spec.name == "dqn42"
    assert spec.kind == "dqn"
    assert spec.path.endswith("corrected-42/best.pth")
    assert parse_baseline_spec("builtin-ga:builtin:asga").path == "ga"

    with pytest.raises(ValueError, match="NAME:KIND:PATH"):
        parse_baseline_spec("not-a-triple")


def test_cpu_is_only_allowed_for_explicit_smoke_and_bounds_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires explicit --smoke"):
        StabilizationConfig(output=tmp_path / "run", initial_bc=tmp_path / "bc.pth", device="cpu")
    with pytest.raises(ValueError, match="between 1 and 200"):
        StabilizationConfig(
            output=tmp_path / "run",
            initial_bc=tmp_path / "bc.pth",
            device="cpu",
            smoke=True,
            validation_deals=201,
        )
    with pytest.raises(ValueError, match="between 1 and 3"):
        StabilizationConfig(
            output=tmp_path / "run",
            initial_bc=tmp_path / "bc.pth",
            device="cpu",
            smoke=True,
            workers=4,
        )


def test_seed_base_separates_smoke_from_formal_and_rejects_history() -> None:
    groups = build_seed_groups(seed_base=824000, validation_deals=2, test_deals=3)

    assert groups["training"] == list(range(824000, 826000))
    assert groups["validation"] == [826000, 826001]
    assert groups["final_test"] == [827000, 827001, 827002]
    assert expected_validation_updates(8) == (0, 2, 4, 6, 8)

    with pytest.raises(ValueError, match="reserved historical"):
        build_seed_groups(seed_base=810000, validation_deals=1, test_deals=1)
    with pytest.raises(ValueError, match="overlaps"):
        verify_seed_groups(
            {"training": [824000], "validation": [824000], "final_test": [827000]}
        )


def test_training_jobs_use_disjoint_deals_and_keep_nine_job_budget(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    jobs = make_training_jobs(config, root=tmp_path / "run")
    assigned = [seed for job in jobs for seed in job.training_seeds]

    assert [job.name for job in jobs] == [
        "fixed-seed42",
        "fixed-seed1234",
        "scratch-seed42",
        "scratch-seed1234",
    ]
    assert len(assigned) == config.training_games_total == 8
    assert len(assigned) == len(set(assigned))
    assert assigned == list(range(824000, 824008))


def test_variant_config_separation_is_explicit() -> None:
    fixed = ppo_config_kwargs(
        "fixed",
        feature_version="v1",
        model_seed=42,
        updates=8,
        games_per_update=16,
        device="cuda",
    )
    anchor = ppo_config_kwargs(
        "anchor",
        feature_version="v1",
        model_seed=42,
        updates=8,
        games_per_update=16,
        device="cuda",
    )
    scratch = ppo_config_kwargs(
        "scratch",
        feature_version="v1",
        model_seed=42,
        updates=8,
        games_per_update=16,
        device="cuda",
    )

    assert fixed["initialization"] == anchor["initialization"] == "bc"
    assert scratch["initialization"] == "scratch"
    assert fixed["reference_kl_coefficient"] == scratch["reference_kl_coefficient"] == 0.0
    assert anchor["reference_kl_coefficient"] == 0.02
    assert {fixed["learning_rate"], anchor["learning_rate"], scratch["learning_rate"]} == {
        1e-4
    }
    assert fixed["target_kl"] == 0.02
    assert fixed["critic_warmup_epochs"] == 2
    assert fixed["history_limit"] == 4


def test_test_audit_preserves_failures_and_reports_paired_seed_rates() -> None:
    records = [
        {
            "candidate": "fixed-seed42",
            "opponent": "ga",
            "seed": 823000,
            "seat": 0,
            "status": "completed",
            "score": 10,
            "rival_score": 5,
            "outcome": 1,
        },
        {
            "candidate": "fixed-seed42",
            "opponent": "ga",
            "seed": 823000,
            "seat": 1,
            "status": "completed",
            "score": 5,
            "rival_score": 10,
            "outcome": -1,
        },
        {
            "candidate": "fixed-seed42",
            "opponent": "ga",
            "seed": 823001,
            "seat": 0,
            "status": "completed",
            "score": 5,
            "rival_score": 5,
            "outcome": 0,
        },
        {
            "candidate": "fixed-seed42",
            "opponent": "ga",
            "seed": 823001,
            "seat": 1,
            "status": "failed",
            "score": 0,
            "rival_score": 0,
            "outcome": None,
        },
    ]
    matrix = {"fixed-seed42": {"ga": {"records": records}}}

    audit = audit_evaluation_matrix(
        matrix,
        seeds=(823000, 823001),
        opponent_names=("ga",),
    )
    summary = summarize_test_records(records, seeds=(823000, 823001))

    assert audit["status"] == "failed_records"
    assert summary["games"] == 4
    assert summary["failed_games"] == 1
    assert summary["wins"] == 1
    assert summary["draws"] == 1
    assert summary["losses"] == 1
    assert summary["scheduled_win_rate"] == 0.25
    assert summary["confidence_intervals"] is None
    assert len(summary["seed_rates"]) == 2

    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        audit_evaluation_matrix(
            matrix,
            seeds=(823000, 823001),
            opponent_names=("ga",),
        )


def test_output_directory_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_output_dir(target)


def test_cli_exposes_smoke_seed_base() -> None:
    args = build_arg_parser().parse_args(
        [
            "--output",
            "/tmp/stabilization-smoke",
            "--initial-bc",
            "/tmp/bc.pth",
            "--device",
            "cpu",
            "--smoke",
            "--seed-base",
            "824000",
        ]
    )

    assert args.seed_base == 824000
    assert args.device == "cpu"
    assert args.smoke is True


def test_coordinated_ppo_api_is_visible_to_driver() -> None:
    status = trainer_api_status()

    assert status["ready"] is True
    assert status["missing_config_fields"] == []
    assert status["missing_trainer_parameters"] == []
