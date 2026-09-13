"""Command-line entrypoint for declared teacher, BC, and audit runs."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bc_training import BCConfig, train_bc
from .dagger import aggregate_dagger_datasets, collect_dagger_dataset
from .distillation import DistillConfig, distill_dqn_teacher
from .evaluation import collect_teacher_dataset, evaluate_matrix
from .information_audit import audit_candidate_information
from .manifest import (
    approve_manifest,
    create_manifest,
    load_manifest,
    require_approved,
)
from .mcts_gate import (
    assess_search_gate,
    build_mcts_candidate,
    calibrate_outcome_value,
    run_search_gate,
)
from .policies import (
    CandidateSpec,
    build_bc_candidate,
    build_builtin_candidate,
    build_fixed_baseline,
)
from .ppo_selfplay import (
    OpponentPoolEntry,
    PPOConfig,
    build_policy_candidate,
    load_ppo_checkpoint,
    train_ppo_selfplay,
)
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
    opponents = [
        build_fixed_baseline(name, device_name=args.device) for name in opponent_names
    ]
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


def _distill_dqn(args: argparse.Namespace) -> None:
    """Distill a DQN value prior into a BC checkpoint (roadmap D3)."""
    dataset = TrajectoryDataset.load(args.dataset)
    if args.feature_version and dataset.feature_version != args.feature_version:
        raise ValueError(
            f"dataset schema {dataset.feature_version!r} != requested "
            f"{args.feature_version!r}"
        )
    # Deterministic split on deal seeds: every 5th seed (sorted) becomes
    # validation; recorded in the output metadata for audit.
    seeds = sorted(dataset.seed_set)
    validation_seeds = {seed for index, seed in enumerate(seeds) if index % 5 == 0}
    masks = np.array(
        [seed in validation_seeds for seed in dataset.deal_seeds.tolist()], dtype=bool
    )

    def _subset(keep: NDArray[np.bool_]) -> TrajectoryDataset:
        return TrajectoryDataset(
            dataset.observations[keep],
            dataset.legal_masks[keep],
            dataset.actions[keep],
            dataset.deal_seeds[keep],
            dataset.seats[keep],
            dataset.plies[keep],
            dataset.steps_in_episode[keep],
            dataset.rewards[keep],
            dataset.terminals[keep],
            dict(dataset.metadata),
        )

    training = _subset(~masks)
    validation = _subset(masks)
    config = DistillConfig(
        feature_version=dataset.feature_version,
        temperature=args.temperature,
        teacher_mode=args.teacher_mode,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    result = distill_dqn_teacher(
        args.teacher,
        training,
        validation,
        args.output,
        config=config,
        source_manifest=args.manifest if hasattr(args, "manifest") else None,
    )
    print(f"DQN distillation written to {result['best']}")


def _dagger_round(args: argparse.Namespace) -> None:
    """Collect one student-visited round and query the approved teacher."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    teacher = _candidate(args.teacher, args)
    student = build_bc_candidate(args.student_checkpoint, device_name=args.device)
    opponent = build_fixed_baseline(args.opponent, device_name=args.device)
    seeds = _seed_group(manifest, args.seed_group)
    result = collect_dagger_dataset(
        teacher,
        student,
        opponent,
        seeds,
        args.output,
        feature_version=args.feature_version or student.feature_version,
        round_index=args.round,
        source_manifest=str(args.manifest),
    )
    _write_json(args.output.with_suffix(".result.json"), result)
    print(f"DAgger round written to {args.output}")


def _dagger_aggregate(args: argparse.Namespace) -> None:
    """Aggregate base demonstrations and one or more DAgger rounds."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    result = aggregate_dagger_datasets(
        args.inputs,
        args.output,
        round_index=args.round,
        source_manifest=str(args.manifest),
    )
    _write_json(args.output.with_suffix(".result.json"), result)
    print(f"DAgger aggregate written to {args.output}")


def _pool_entries(
    args: argparse.Namespace, initial_bc: Path
) -> list[OpponentPoolEntry]:
    """Build fixed pool entries; ``current`` is supplied by the PPO trainer."""
    entries: list[OpponentPoolEntry] = []
    for raw in args.opponent_pool.split(","):
        name, separator, raw_weight = raw.strip().partition(":")
        if not name or name == "current":
            continue
        weight = float(raw_weight) if separator else 1.0
        if name == "bc":
            candidate = build_bc_candidate(initial_bc, device_name=args.device)
        else:
            candidate = build_fixed_baseline(name, device_name=args.device)
        entries.append(OpponentPoolEntry(name, candidate, weight))
    return entries


def _ppo_selfplay(args: argparse.Namespace) -> None:
    """Train imitation PPO with one complete-game policy-pool choice."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    initial_bc = args.initial_bc
    pool = _pool_entries(args, initial_bc)
    validation_opponents = [
        build_fixed_baseline(name.strip(), device_name=args.device)
        for name in args.validation_opponents.split(",")
        if name.strip()
    ]
    config = PPOConfig(
        feature_version=args.feature_version,
        hidden_layers=tuple(args.hidden_layers),
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        updates=args.updates,
        games_per_update=args.games_per_update,
        terminal_value=args.terminal_value,
        seed=args.seed,
        device_name=args.device,
        initialization=args.initialization,
        target_kl=args.target_kl,
        reference_kl_coefficient=args.reference_kl_coefficient,
        current_weight=_current_pool_weight(args.opponent_pool),
        history_weight=args.history_weight,
        history_limit=args.history_limit,
        critic_warmup_epochs=args.critic_warmup_epochs,
        critic_learning_rate=args.critic_learning_rate,
        critic_hidden_dim=args.critic_hidden_dim,
        eval_every=args.eval_every,
    )
    result = train_ppo_selfplay(
        initial_bc,
        args.output,
        _seed_group(manifest, "training"),
        pool,
        config=config,
        validation_seeds=_seed_group(manifest, "validation"),
        validation_opponents=validation_opponents,
        source_manifest=str(args.manifest),
    )
    print(f"PPO self-play written to {result['best']}")


def _current_pool_weight(pool: str) -> float:
    """Honor the declared current bucket instead of silently forcing weight one."""
    weights = []
    for raw in pool.split(","):
        name, separator, weight = raw.strip().partition(":")
        if name == "current":
            weights.append(float(weight) if separator else 1.0)
    if len(weights) > 1:
        raise ValueError("current opponent bucket must not appear twice")
    return weights[0] if weights else 0.0


def _ppo_eval(args: argparse.Namespace) -> None:
    """Evaluate an imitation-PPO checkpoint on one declared seed group."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    model = load_ppo_checkpoint(args.checkpoint, device_name=args.device)
    candidate = build_policy_candidate(
        model,
        name="ppo-selfplay",
        snapshot=str(args.checkpoint),
        device_name=args.device,
    )
    opponents = [
        build_fixed_baseline(name.strip(), device_name=args.device)
        for name in args.opponents.split(",")
        if name.strip()
    ]
    seeds = _seed_group(manifest, args.seed_group)
    results = evaluate_matrix([candidate], opponents, seeds)[candidate.name]
    _write_json(
        args.output,
        {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "phase": manifest["phase"],
            "seed_group": args.seed_group,
            "seeds": seeds,
            "results": results,
        },
    )
    print(f"PPO evaluation written to {args.output}")


def _bc_eval(args: argparse.Namespace) -> None:
    """Evaluate a BC or DAgger checkpoint without querying a teacher."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    candidate = build_bc_candidate(args.checkpoint, device_name=args.device)
    opponents = [
        build_fixed_baseline(name.strip(), device_name=args.device)
        for name in args.opponents.split(",")
        if name.strip()
    ]
    seeds = _seed_group(manifest, args.seed_group)
    results = evaluate_matrix([candidate], opponents, seeds)[candidate.name]
    _write_json(
        args.output,
        {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "phase": manifest["phase"],
            "seed_group": args.seed_group,
            "seeds": seeds,
            "teacher_queries": 0,
            "opponents": [opponent.name for opponent in opponents],
            "results": results,
        },
    )
    print(f"BC evaluation written to {args.output}")


def _mcts_gate(args: argparse.Namespace) -> None:
    """Run same-weight MCTS budgets, value calibration, and semantic audit."""
    manifest = load_manifest(args.manifest)
    require_approved(manifest)
    seeds = _seed_group(manifest, args.seed_group)
    opponent_names = [
        name.strip() for name in args.opponents.split(",") if name.strip()
    ]
    if not opponent_names:
        raise ValueError("MCTS gate requires at least one fixed opponent")
    opponents = [
        build_fixed_baseline(name, device_name=args.device) for name in opponent_names
    ]
    gate = run_search_gate(
        args.checkpoint,
        seeds,
        opponents,
        args.simulations,
        device_name=args.device,
        rng_seed=args.rng_seed,
        max_depth=args.max_depth,
    )
    calibration = calibrate_outcome_value(
        args.checkpoint,
        opponents[0],
        seeds,
        device_name=args.device,
    )
    audit_budget = max(args.simulations)
    audit_candidate = build_mcts_candidate(
        args.checkpoint,
        simulations=audit_budget,
        device_name=args.device,
        rng_seed=args.rng_seed,
        max_depth=args.max_depth,
    )
    information_audit = audit_candidate_information(
        audit_candidate,
        opponents[0],
        seeds,
        max_states=args.audit_max_states,
    )
    assessment = assess_search_gate(gate, calibration, information_audit)
    _write_json(
        args.output,
        {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "phase": manifest["phase"],
            "seed_group": args.seed_group,
            "seeds": seeds,
            "opponents": opponent_names,
            "simulations": sorted(set(args.simulations)),
            "gate": gate,
            "value_calibration": calibration,
            "information_audit": information_audit,
            "assessment": assessment,
        },
    )
    print(f"MCTS gate written to {args.output} ({assessment['status']})")


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - subcommands are explicit
    parser = argparse.ArgumentParser(prog="policy-imitation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repo", type=Path, default=Path.cwd())
    create.add_argument("--experiment-id", required=True)
    create.add_argument("--phase", default="3.0-3.2")
    create.add_argument("--purpose", choices=("smoke", "formal"), default="formal")
    create.add_argument(
        "--training-seeds", nargs="+", type=int, default=[42, 1234, 2024]
    )
    create.add_argument(
        "--validation-seeds", nargs="+", type=int, default=list(range(800101, 800111))
    )
    create.add_argument(
        "--final-test-seeds", nargs="+", type=int, default=list(range(800201, 800221))
    )
    create.add_argument(
        "--teacher-evaluation-seeds",
        nargs="+",
        type=int,
        default=list(range(800301, 800311)),
    )
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
    teacher.add_argument(
        "--candidates",
        nargs="+",
        choices=DEFAULT_CANDIDATES,
        default=list(DEFAULT_CANDIDATES),
    )
    teacher.add_argument(
        "--opponents",
        nargs="+",
        choices=DEFAULT_OPPONENTS,
        default=list(DEFAULT_OPPONENTS),
    )
    _add_snapshot_arguments(teacher)
    teacher.set_defaults(handler=_teacher_eval)

    audit = subparsers.add_parser("information-audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument(
        "--candidates",
        nargs="+",
        choices=DEFAULT_CANDIDATES,
        default=list(DEFAULT_CANDIDATES),
    )
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
    collect.add_argument(
        "--feature-version",
        choices=("v1", "public-v2", "public-v2-multi"),
        default="v1",
    )
    _add_snapshot_arguments(collect)
    collect.set_defaults(handler=_collect_bc)

    train = subparsers.add_parser("train-bc")
    train.add_argument("manifest", type=Path)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--feature-version",
        choices=("v1", "public-v2", "public-v2-multi"),
        default=None,
    )
    train.add_argument(
        "--hidden-layers", nargs="+", type=int, default=[128, 128, 128, 128]
    )
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    train.set_defaults(handler=_train_bc)

    distill = subparsers.add_parser(
        "distill-dqn",
        help="Distill a DQN value prior into a BC checkpoint (roadmap D3)",
    )
    distill.add_argument("--teacher", type=Path, required=True)
    distill.add_argument("--dataset", type=Path, required=True)
    distill.add_argument("--output", type=Path, required=True)
    distill.add_argument(
        "--teacher-mode",
        choices=("q-softmax", "policy-head"),
        default="q-softmax",
    )
    distill.add_argument("--temperature", type=float, default=1.0)
    distill.add_argument("--learning-rate", type=float, default=1e-3)
    distill.add_argument("--epochs", type=int, default=10)
    distill.add_argument("--batch-size", type=int, default=256)
    distill.add_argument("--seed", type=int, default=1234)
    distill.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    distill.set_defaults(handler=_distill_dqn)

    dagger = subparsers.add_parser("dagger-round")
    dagger.add_argument("manifest", type=Path)
    dagger.add_argument("--output", type=Path, required=True)
    dagger.add_argument("--teacher", choices=DEFAULT_CANDIDATES, required=True)
    dagger.add_argument("--student-checkpoint", type=Path, required=True)
    dagger.add_argument("--opponent", choices=DEFAULT_OPPONENTS, default="random")
    dagger.add_argument("--seed-group", default="training")
    dagger.add_argument(
        "--feature-version",
        choices=("v1", "public-v2", "public-v2-multi"),
        default=None,
    )
    dagger.add_argument("--round", type=int, required=True)
    _add_snapshot_arguments(dagger)
    dagger.set_defaults(handler=_dagger_round)

    aggregate = subparsers.add_parser("dagger-aggregate")
    aggregate.add_argument("manifest", type=Path)
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--round", type=int, required=True)
    aggregate.set_defaults(handler=_dagger_aggregate)

    ppo = subparsers.add_parser("ppo-selfplay")
    ppo.add_argument("manifest", type=Path)
    ppo.add_argument("--initial-bc", type=Path, required=True)
    ppo.add_argument("--output", type=Path, required=True)
    ppo.add_argument("--opponent-pool", default="ga:1,heuristic:1,current:1")
    ppo.add_argument("--validation-opponents", default="random,heuristic,minimax")
    ppo.add_argument(
        "--feature-version",
        choices=("v1", "public-v2", "public-v2-multi"),
        default="v1",
    )
    ppo.add_argument(
        "--hidden-layers", nargs="+", type=int, default=[128, 128, 128, 128]
    )
    ppo.add_argument("--learning-rate", type=float, default=3e-4)
    ppo.add_argument("--discount-factor", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-epsilon", type=float, default=0.2)
    ppo.add_argument("--entropy-coefficient", type=float, default=0.005)
    ppo.add_argument("--value-coefficient", type=float, default=0.5)
    ppo.add_argument("--minibatch-size", type=int, default=256)
    ppo.add_argument("--update-epochs", type=int, default=4)
    ppo.add_argument("--updates", type=int, default=10)
    ppo.add_argument("--games-per-update", type=int, default=4)
    ppo.add_argument("--terminal-value", type=float, default=10.0)
    ppo.add_argument("--initialization", choices=("bc", "scratch"), default="bc")
    ppo.add_argument("--target-kl", type=float, default=0.02)
    ppo.add_argument("--reference-kl-coefficient", type=float, default=0.0)
    ppo.add_argument("--history-weight", type=float, default=1.0)
    ppo.add_argument("--history-limit", type=int, default=4)
    ppo.add_argument("--critic-warmup-epochs", type=int, default=0)
    ppo.add_argument(
        "--critic-learning-rate",
        type=float,
        default=None,
        help="Roadmap C1: separate Adam lr for the value head",
    )
    ppo.add_argument(
        "--critic-hidden-dim",
        type=int,
        default=0,
        help="Roadmap C1: critic-private hidden layer width (0 = off)",
    )
    ppo.add_argument("--eval-every", type=int, default=1)
    ppo.add_argument("--seed", type=int, default=1234)
    ppo.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ppo.set_defaults(handler=_ppo_selfplay)

    ppo_eval = subparsers.add_parser("ppo-eval")
    ppo_eval.add_argument("manifest", type=Path)
    ppo_eval.add_argument("--checkpoint", type=Path, required=True)
    ppo_eval.add_argument("--output", type=Path, required=True)
    ppo_eval.add_argument("--seed-group", default="final_test")
    ppo_eval.add_argument("--opponents", default="random,heuristic,minimax")
    ppo_eval.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ppo_eval.set_defaults(handler=_ppo_eval)

    bc_eval = subparsers.add_parser("bc-eval")
    bc_eval.add_argument("manifest", type=Path)
    bc_eval.add_argument("--checkpoint", type=Path, required=True)
    bc_eval.add_argument("--output", type=Path, required=True)
    bc_eval.add_argument("--seed-group", default="final_test")
    bc_eval.add_argument("--opponents", default="random,heuristic,minimax")
    bc_eval.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    bc_eval.set_defaults(handler=_bc_eval)

    mcts = subparsers.add_parser("mcts-gate")
    mcts.add_argument("manifest", type=Path)
    mcts.add_argument("--checkpoint", type=Path, required=True)
    mcts.add_argument("--output", type=Path, required=True)
    mcts.add_argument("--seed-group", default="teacher_evaluation")
    mcts.add_argument("--opponents", default="random,heuristic,minimax")
    mcts.add_argument("--simulations", nargs="+", type=int, default=[0, 1, 4])
    mcts.add_argument("--max-depth", type=int, default=24)
    mcts.add_argument("--audit-max-states", type=int, default=12)
    mcts.add_argument("--rng-seed", type=int, default=1234)
    mcts.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    mcts.set_defaults(handler=_mcts_gate)
    return parser


def main() -> None:
    """Dispatch one declared subcommand."""
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
