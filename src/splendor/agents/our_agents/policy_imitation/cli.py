"""Command-line entrypoint for declared teacher, BC, and audit runs."""

import argparse
import json
from pathlib import Path
from typing import Any

from .bc_training import BCConfig, train_bc
from .evaluation import collect_teacher_dataset, evaluate_matrix
from .information_audit import audit_candidate_information
from .manifest import (
    approve_manifest,
    create_manifest,
    load_manifest,
    require_approved,
)
from .policies import CandidateSpec, build_builtin_candidate, build_fixed_baseline
from .trajectory import TrajectoryDataset, split_by_seed

DEFAULT_CANDIDATES = ("ga", "minimax", "ppo", "corrected-dqn", "heuristic")
DEFAULT_OPPONENTS = ("random", "heuristic", "minimax")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an experiment result with deterministic key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _seed_group(manifest: dict[str, Any], name: str) -> list[int]:
    """Read one declared group and fail rather than silently reusing a split."""
    groups = manifest["seed_plan"]["groups"]
    if name not in groups or not groups[name]:
        raise ValueError(f"manifest has no non-empty seed group {name!r}")
    return [int(seed) for seed in groups[name]]


def _candidate(name: str, args: argparse.Namespace) -> CandidateSpec:
    """Build a candidate with only the explicitly supplied snapshot."""
    checkpoint = None
    if name == "corrected-dqn":
        checkpoint = args.corrected_dqn_checkpoint
    elif name == "ppo":
        checkpoint = args.ppo_checkpoint
    return build_builtin_candidate(
        name,
        checkpoint=checkpoint,
        device_name=args.device,
    )


def _add_snapshot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--ppo-checkpoint", type=Path, default=None)
    parser.add_argument("--corrected-dqn-checkpoint", type=Path, default=None)


def _create_manifest(args: argparse.Namespace) -> None:
    """Create a proposed manifest; approval remains a separate command."""
    create_manifest(
        args.output,
        experiment_id=args.experiment_id,
        phase=args.phase,
        purpose=args.purpose,
        seed_groups={
            "training": args.training_seeds,
            "validation": args.validation_seeds,
            "final_test": args.final_test_seeds,
            "teacher_evaluation": args.teacher_evaluation_seeds,
        },
        budget={
            "teacher_candidates": list(DEFAULT_CANDIDATES),
            "teacher_opponents": list(DEFAULT_OPPONENTS),
            "teacher_games_per_pair": args.teacher_games_per_pair,
            "bc_epochs": args.bc_epochs,
            "bc_max_samples": args.bc_max_samples,
        },
        success_thresholds={
            "max_illegal_actions": 0,
            "max_failed_games": 0,
            "min_information_stable_rate": 1.0,
            "bc_validation_selection": "lowest masked validation loss; exact ties keep earliest epoch",
            "bc_min_validation_legal_rate": 1.0,
        },
        requested_device=args.device,
        repo=args.repo,
        notes=(
            "Proposal only; run approve after manually checking budget, seeds, and thresholds.",
            "DAgger and PPO self-play require new manifests after the preceding gates.",
        ),
    )
    print(f"Proposed manifest written to {args.output}")


def _approve(args: argparse.Namespace) -> None:
    """Record an explicit reviewer decision."""
    approve_manifest(args.manifest, args.reviewer, args.note)
    print(f"Approved manifest: {args.manifest}")


def _teacher_eval(args: argparse.Namespace) -> None:
    """Run a fixed candidate x opponent matrix after approval."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    names = args.candidates
    opponent_names = args.opponents
    candidates = [_candidate(name, args) for name in names]
    opponents = [build_fixed_baseline(name, device_name=args.device) for name in opponent_names]
    seeds = _seed_group(manifest, "teacher_evaluation")
    results = evaluate_matrix(candidates, opponents, seeds)
    _write_json(
        args.output,
        {
            "manifest": str(args.manifest),
            "experiment_id": manifest["experiment_id"],
            "phase": manifest["phase"],
            "seed_group": "teacher_evaluation",
            "seeds": seeds,
            "candidates": names,
            "opponents": opponent_names,
            "results": results,
        },
    )
    print(f"Teacher evaluation written to {args.output}")


def _information_audit(args: argparse.Namespace) -> None:
    """Run projection-preserving hidden-state probes after approval."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    seeds = _seed_group(manifest, "teacher_evaluation")
    opponent = build_fixed_baseline(args.opponent, device_name=args.device)
    results = {}
    for name in args.candidates:
        candidate = _candidate(name, args)
        results[name] = audit_candidate_information(
            candidate,
            opponent,
            seeds,
            max_states=args.max_states,
        )
    _write_json(
        args.output,
        {
            "manifest": str(args.manifest),
            "experiment_id": manifest["experiment_id"],
            "seed_group": "teacher_evaluation",
            "opponent_for_state_access": args.opponent,
            "results": results,
        },
    )
    print(f"Information audit written to {args.output}")


def _collect_bc(args: argparse.Namespace) -> None:
    """Collect one teacher's focal trajectories for a declared split."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    teacher = _candidate(args.teacher, args)
    opponent = build_fixed_baseline(args.opponent, device_name=args.device)
    seeds = _seed_group(manifest, args.seed_group)
    result = collect_teacher_dataset(
        teacher,
        opponent,
        seeds,
        args.output,
        feature_version=args.feature_version,
        source_manifest=str(args.manifest),
    )
    _write_json(args.output.with_suffix(".result.json"), result)
    print(f"Teacher dataset written to {args.output}")


def _train_bc(args: argparse.Namespace) -> None:
    """Train one BC feature variant with validation-only checkpoint selection."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    dataset = TrajectoryDataset.load(args.dataset)
    splits = split_by_seed(dataset, manifest["seed_plan"]["groups"])
    config = BCConfig(
        feature_version=args.feature_version or dataset.feature_version,
        hidden_layers=tuple(args.hidden_layers),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
    )
    result = train_bc(
        splits["training"],
        splits["validation"],
        args.output,
        config=config,
        test_data=splits["final_test"],
        source_manifest=str(args.manifest),
    )
    print(f"BC training written to {result['best']}")


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - subcommands are explicit
    parser = argparse.ArgumentParser(prog="policy-imitation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repo", type=Path, default=Path.cwd())
    create.add_argument("--experiment-id", required=True)
    create.add_argument("--phase", default="3.0-3.2")
    create.add_argument("--purpose", choices=("smoke", "formal"), default="formal")
    create.add_argument("--training-seeds", nargs="+", type=int, default=[42, 1234, 2024])
    create.add_argument("--validation-seeds", nargs="+", type=int, default=list(range(800101, 800111)))
    create.add_argument("--final-test-seeds", nargs="+", type=int, default=list(range(800201, 800221)))
    create.add_argument("--teacher-evaluation-seeds", nargs="+", type=int, default=list(range(800301, 800311)))
    create.add_argument("--teacher-games-per-pair", type=int, default=20)
    create.add_argument("--bc-epochs", type=int, default=10)
    create.add_argument("--bc-max-samples", type=int, default=50000)
    create.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    create.set_defaults(handler=_create_manifest)

    approve = subparsers.add_parser("approve")
    approve.add_argument("manifest", type=Path)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--note", required=True)
    approve.set_defaults(handler=_approve)

    teacher = subparsers.add_parser("teacher-eval")
    teacher.add_argument("manifest", type=Path)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--candidates", nargs="+", choices=DEFAULT_CANDIDATES, default=list(DEFAULT_CANDIDATES))
    teacher.add_argument("--opponents", nargs="+", choices=DEFAULT_OPPONENTS, default=list(DEFAULT_OPPONENTS))
    _add_snapshot_arguments(teacher)
    teacher.set_defaults(handler=_teacher_eval)

    audit = subparsers.add_parser("information-audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--candidates", nargs="+", choices=DEFAULT_CANDIDATES, default=list(DEFAULT_CANDIDATES))
    audit.add_argument("--opponent", choices=DEFAULT_OPPONENTS, default="random")
    audit.add_argument("--max-states", type=int, default=24)
    _add_snapshot_arguments(audit)
    audit.set_defaults(handler=_information_audit)

    collect = subparsers.add_parser("collect-bc")
    collect.add_argument("manifest", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--teacher", choices=DEFAULT_CANDIDATES, required=True)
    collect.add_argument("--opponent", choices=DEFAULT_OPPONENTS, default="random")
    collect.add_argument("--seed-group", default="training")
    collect.add_argument("--feature-version", choices=("v1", "public-v2"), default="v1")
    _add_snapshot_arguments(collect)
    collect.set_defaults(handler=_collect_bc)

    train = subparsers.add_parser("train-bc")
    train.add_argument("manifest", type=Path)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--feature-version", choices=("v1", "public-v2"), default=None)
    train.add_argument("--hidden-layers", nargs="+", type=int, default=[128, 128, 128, 128])
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    train.set_defaults(handler=_train_bc)
    return parser


def main() -> None:
    """Dispatch one declared subcommand."""
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
