"""Task-1 baseline and historical-denominator audit.

This module is deliberately read-only: it validates the frozen C2-R2 model,
its BC/DAgger parent, built-in opponent implementations, and the C4-R2 raw
league rows without starting games or training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from splendor.agents.our_agents.minmax import DEPTH

from .policies import RUSH_WEIGHTS
from .protocol import sha256_canonical_json, sha256_file
from .trajectory import TrajectoryDataset

OFFICIAL_CHECKPOINT = Path("runs/c2r2-selfplay-2000/training/fixed-seed1234/best.pth")
PARENT_BC_CHECKPOINT = Path(
    "runs/policy-imitation/formal-3.3-20260907/bc-dagger-2/best.pth"
)
PARENT_DATASET = Path("runs/policy-imitation/formal-3.3-20260907/aggregate-round-2.npz")
PARENT_MANIFEST = Path("runs/policy-imitation/formal-3.3-20260907/manifest.json")
C4R2_RESULTS = Path("runs/c4r2-league-20260914/league_results.json")
OFFICIAL_UPDATE = 1700
OFFICIAL_INPUT_DIM = 312

EXPECTED_OFFICIAL_SHA256 = (
    "e225464c17a783bd91b51251336f917e9f867af4f475d8414372885fbb758102"
)
EXPECTED_PARENT_BC_SHA256 = (
    "e64b722f34a8eed390ae03bdd5adf2f03be9ec6238db556b86180adff1e6f867"
)
EXPECTED_PARENT_DATA_HASH = (
    "31bd073b25d84c8d95e3d6d459244e32be4fb449ae72467c07526677e696c3d4"
)
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "69e96bfb9f25c540e79a956cf47eeea28be09376bbe5a3b4989fa1f0150f8fae"
)

EXPECTED_CONFIG: dict[str, object] = {
    "feature_version": "public-v2",
    "input_dim": 312,
    "output_dim": 3510,
    "hidden_layers": [128, 128, 128, 128],
    "learning_rate": 1e-4,
    "discount_factor": 0.99,
    "terminal_value": 10.0,
    "updates": 2000,
    "games_per_update": 16,
    "value_coefficient": 1.0,
    "critic_learning_rate": 5e-4,
    "shaping_kind": "potential",
    "shaping_kappa": 0.05,
    "seed": 1234,
    "normalizer_fitted": True,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _expect_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"artifact SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("official checkpoint is missing config")
    for field, expected in EXPECTED_CONFIG.items():
        if config.get(field) != expected:
            raise ValueError(
                f"official checkpoint config.{field} mismatch: "
                f"expected {expected!r}, got {config.get(field)!r}"
            )
    if checkpoint.get("model_type") != "imitation_ppo_policy_value":
        raise ValueError("official checkpoint model_type mismatch")
    if checkpoint.get("update") != OFFICIAL_UPDATE:
        raise ValueError("official checkpoint must be the update-1700 selection")
    source_bc = Path(str(checkpoint.get("source_bc")))
    if source_bc.as_posix() != PARENT_BC_CHECKPOINT.as_posix():
        raise ValueError("official checkpoint source_bc pointer mismatch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("official checkpoint is missing model_state_dict")
    for key in ("normalizer.mean", "normalizer.variance"):
        tensor = state.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != OFFICIAL_INPUT_DIM:
            raise ValueError(f"official checkpoint has invalid {key}")
    return config


def audit_official_checkpoint(repo: Path) -> dict[str, Any]:
    """Validate the deployed C2-R2 checkpoint and every direct parent hash."""
    checkpoint_path = repo / OFFICIAL_CHECKPOINT
    parent_path = repo / PARENT_BC_CHECKPOINT
    dataset_path = repo / PARENT_DATASET
    manifest_path = repo / PARENT_MANIFEST
    checkpoint_hash = _expect_hash(checkpoint_path, EXPECTED_OFFICIAL_SHA256)
    parent_hash = _expect_hash(parent_path, EXPECTED_PARENT_BC_SHA256)
    manifest_hash = _expect_hash(manifest_path, EXPECTED_PARENT_MANIFEST_SHA256)

    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(raw_checkpoint, Mapping):
        raise ValueError("official checkpoint root must be a mapping")
    checkpoint = cast(Mapping[str, Any], raw_checkpoint)
    config = _validate_checkpoint_contract(checkpoint)

    dataset = TrajectoryDataset.load(dataset_path)
    dataset_hash = dataset.content_hash()
    if dataset_hash != EXPECTED_PARENT_DATA_HASH:
        raise ValueError(
            "parent aggregate dataset content hash mismatch: "
            f"expected {EXPECTED_PARENT_DATA_HASH}, got {dataset_hash}"
        )
    return {
        "id": "ppo-best",
        "role": "frozen-deployment-baseline",
        "checkpoint": OFFICIAL_CHECKPOINT.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_update": OFFICIAL_UPDATE,
        "config": dict(config),
        "config_sha256": sha256_canonical_json(dict(config)),
        "contract": {
            "feature_version": "public-v2",
            "input_dim": 312,
            "action_dim": 3510,
            "hidden_layers": [128, 128, 128, 128],
            "inference": "eval + masked greedy argmax",
            "training_action": "masked categorical (legacy global torch RNG)",
            "reward": "terminal +/-10/0 + nonzero-terminal potential kappa=0.05",
        },
        "normalizer": {
            "fitted": True,
            "source": PARENT_BC_CHECKPOINT.as_posix(),
            "state_fields": ["normalizer.mean", "normalizer.variance"],
        },
        "parent_bc": {
            "path": PARENT_BC_CHECKPOINT.as_posix(),
            "sha256": parent_hash,
        },
        "parent_dataset": {
            "path": PARENT_DATASET.as_posix(),
            "content_sha256": dataset_hash,
            "file_sha256": sha256_file(dataset_path),
            "samples": dataset.size,
        },
        "parent_manifest": {
            "path": PARENT_MANIFEST.as_posix(),
            "sha256": manifest_hash,
        },
    }


def _opponent_files(repo: Path, name: str) -> list[Path]:
    roots: dict[str, tuple[str, ...]] = {
        "random": ("src/splendor/agents/generic/random.py",),
        "heuristic": ("src/splendor/agents/our_agents/dqn/population.py",),
        "heuristic-rush": (
            "src/splendor/agents/our_agents/dqn/population.py",
            "src/splendor/agents/our_agents/policy_imitation/policies.py",
        ),
        "minimax": ("src/splendor/agents/our_agents/minmax.py",),
        "ga": (
            "src/splendor/agents/our_agents/genetic_algorithm/genes.py",
            "src/splendor/agents/our_agents/genetic_algorithm/genetic_algorithm_agent.py",
            "src/splendor/agents/our_agents/genetic_algorithm/manager.npy",
            "src/splendor/agents/our_agents/genetic_algorithm/strategy1.npy",
            "src/splendor/agents/our_agents/genetic_algorithm/strategy2.npy",
            "src/splendor/agents/our_agents/genetic_algorithm/strategy3.npy",
        ),
    }
    try:
        return [repo / relative for relative in roots[name]]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen opponent {name!r}") from exc


def freeze_builtin_opponents(
    repo: Path,
    names: Sequence[str] = ("random", "heuristic", "heuristic-rush", "ga", "minimax"),
) -> dict[str, Any]:
    """Hash source/config for built-in opponents with no checkpoint."""
    configs: dict[str, Mapping[str, object]] = {
        "random": {
            "selection": "random.choice over runner-provided legal actions",
            "rng": "legacy global Python random",
        },
        "heuristic": {"class": "dqn.population.HeuristicAgent"},
        "heuristic-rush": asdict(RUSH_WEIGHTS),
        "ga": {"genes": "four checked-in NumPy arrays"},
        "minimax": {
            "depth": DEPTH,
            "tie_break": "global random.shuffle then stable sort by action type",
        },
    }
    frozen: dict[str, Any] = {}
    for name in names:
        files = {
            path.relative_to(repo).as_posix(): sha256_file(path)
            for path in _opponent_files(repo, name)
        }
        config = dict(configs[name])
        frozen[name] = {
            "kind": "builtin",
            "source_files": files,
            "source_sha256": sha256_canonical_json(files),
            "config": config,
            "config_sha256": sha256_canonical_json(config),
        }
    return frozen


def audit_c4r2_denominators(
    repo: Path,
    *,
    focal: str = "ppo-best",
) -> dict[str, Any]:
    """Rebuild scheduled, unique, and wrapped C4-R2 denominators."""
    path = repo / C4R2_RESULTS
    payload = _read_json(path)
    raw_games = payload.get("games")
    if not isinstance(raw_games, list):
        raise ValueError("C4-R2 results are missing games")
    games = [cast(Mapping[str, Any], row) for row in raw_games]
    roster = payload.get("roster")
    if not isinstance(roster, list):
        raise ValueError("C4-R2 results are missing roster")
    opponent_names = sorted(
        str(cast(Mapping[str, Any], row)["name"])
        for row in roster
        if cast(Mapping[str, Any], row).get("name") != focal
    )
    matchups: dict[str, Any] = {}
    for opponent in opponent_names:
        rows = [
            row
            for row in games
            if focal in row.get("names", []) and opponent in row.get("names", [])
        ]
        cell_counts: Counter[tuple[int, int]] = Counter()
        wins = draws = losses = errors = 0
        for row in rows:
            names = [str(name) for name in row["names"]]
            focal_seat = names.index(focal)
            cell_counts[(int(row["seed"]), focal_seat)] += 1
            if row.get("error") is not None:
                errors += 1
                continue
            winner_seats = [int(seat) for seat in row["winner_seats"]]
            if len(winner_seats) > 1:
                draws += 1
            elif focal_seat in winner_seats:
                wins += 1
            else:
                losses += 1
        scheduled = len(rows)
        matchups[opponent] = {
            "scheduled_rows": scheduled,
            "completed_rows": scheduled - errors,
            "error_rows": errors,
            "unique_source_seeds": len({int(row["seed"]) for row in rows}),
            "unique_seed_seat_cells": len(cell_counts),
            "duplicated_cell_count": sum(count > 1 for count in cell_counts.values()),
            "rows_in_duplicated_cells": sum(
                count for count in cell_counts.values() if count > 1
            ),
            "duplicate_excess_rows": sum(count - 1 for count in cell_counts.values()),
            "wdl": {"wins": wins, "draws": draws, "losses": losses},
            "score_rate_scheduled": (wins + 0.5 * draws) / scheduled,
            "inference_note": (
                "historical description only; rows are wrapped and clustered by seed"
            ),
        }
    return {
        "source": C4R2_RESULTS.as_posix(),
        "source_sha256": sha256_file(path),
        "declared_seed_count": int(payload.get("seed_count", -1)),
        "scheduled_rows_total": len(games),
        "unique_source_seeds_total": len({int(row["seed"]) for row in games}),
        "error_rows_total": sum(row.get("error") is not None for row in games),
        "focal": focal,
        "matchups": matchups,
        "legacy_wilson_use": "descriptive-only; not valid for task-1 decisions",
    }


def build_baseline_audit(repo: Path) -> dict[str, Any]:
    """Run all read-only T1.0 baseline checks."""
    root = repo.resolve()
    return {
        "schema": "splendor-task1-baseline-audit/1",
        "official": audit_official_checkpoint(root),
        "opponents": freeze_builtin_opponents(root),
        "historical_c4r2": audit_c4r2_denominators(root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the baseline and optionally persist the JSON audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    audit = build_baseline_audit(args.repo)
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module invocation
    raise SystemExit(main())
