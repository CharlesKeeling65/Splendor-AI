"""Reproducible, process-isolated PPO stabilisation experiments.

This module is deliberately a driver rather than another training
implementation.  It freezes the experiment protocol, starts every training
job in a ``spawn`` worker, and keeps the training, validation, and final-test
artifacts auditable.  The formal run is intentionally gated on the revised
``ppo_selfplay`` API; an older trainer is never used with silently ignored
stabilisation options.

The command-line entry point is available without changing the package
console scripts::

    python -m splendor.agents.our_agents.policy_imitation.stabilization \
        --output /path/to/new-run --device cuda --workers 3 \
        --updates 8 --games-per-update 16 --validation-deals 10 \
        --test-deals 25 --seeds 42 1234 2024 \
        --variants fixed anchor scratch --initial-bc /path/to/bc.pth

``--device cpu`` is accepted only with ``--smoke``.  The driver never turns a
requested CUDA run into a CPU run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import math
import multiprocessing as multiprocessing_module
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import torch

from .evaluation import evaluate_matrix
from .manifest import create_manifest, validate_manifest
from .policies import CandidateSpec, build_bc_candidate, build_builtin_candidate
from .ppo_selfplay import (
    OpponentPoolEntry,
    PPOConfig,
    build_policy_candidate,
    load_ppo_checkpoint,
    train_ppo_selfplay,
)
from .protocol import RuntimeSnapshot, capture_code_revision, seed_everything

VariantName = Literal["fixed", "anchor", "scratch"]
BaselineKind = Literal["bc", "ppo", "dqn", "builtin"]

VARIANTS: tuple[VariantName, ...] = ("fixed", "anchor", "scratch")
BASELINE_KINDS = frozenset({"bc", "ppo", "dqn", "builtin"})
TRAINING_POOL_NAMES: tuple[str, ...] = ("ga", "heuristic")
VALIDATION_OPPONENT_NAMES: tuple[str, ...] = ("ga", "heuristic", "minimax")
FINAL_OPPONENT_NAMES: tuple[str, ...] = (
    "random",
    "ga",
    "heuristic",
    "minimax",
)

DEFAULT_MODEL_SEEDS: tuple[int, ...] = (42, 1234, 2024)
TRAINING_SEED_START = 820000
TRAINING_SEED_COUNT = 2000
VALIDATION_SEED_START = 822000
FINAL_TEST_SEED_START = 823000
VALIDATION_SEED_OFFSET = VALIDATION_SEED_START - TRAINING_SEED_START
FINAL_TEST_SEED_OFFSET = FINAL_TEST_SEED_START - TRAINING_SEED_START
MAX_DEALS = 200
MAX_WORKERS = 3
DEFAULT_EVAL_EVERY = 2
USER_AUTHORITY_DATE = "2026-09-08"

# Keep the new protocol disjoint from every historical experimental family
# mentioned in the repository.  The broad half-open ranges intentionally
# reserve the old 800xxx/810xxx and 900xxx-series namespaces as well as the
# explicitly documented 910xxx/920xxx/930xxx sets.
HISTORICAL_FORBIDDEN_SEED_RANGES: tuple[tuple[int, int], ...] = (
    (800000, 820000),
    (900000, 940000),
)

REQUIRED_PPO_CONFIG_FIELDS = frozenset(
    {
        "initialization",
        "reference_kl_coefficient",
        "target_kl",
        "current_weight",
        "history_weight",
        "history_limit",
        "critic_warmup_epochs",
        "eval_every",
    }
)
REQUIRED_TRAINER_PARAMETERS = frozenset(
    {
        "initial_bc",
        "output_dir",
        "training_seeds",
        "opponent_pool",
        "config",
        "validation_seeds",
        "validation_opponents",
        "source_manifest",
    }
)


@dataclass(frozen=True)
class BaselineSpec:
    """A serialisable candidate source used by an evaluation worker.

    ``path`` is a checkpoint path for ``bc``, ``ppo``, and ``dqn``.  For a
    ``builtin`` source it is the built-in candidate name (for example
    ``ga``).  Keeping this descriptor free of factories is important: spawn
    workers rebuild every ``CandidateSpec`` locally, after the process starts.
    """

    name: str
    kind: BaselineKind
    path: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("baseline name must not be empty")
        if self.kind not in BASELINE_KINDS:
            raise ValueError(
                f"unsupported baseline kind {self.kind!r}; "
                f"expected one of {sorted(BASELINE_KINDS)}"
            )
        if not self.path.strip():
            raise ValueError("baseline path or built-in name must not be empty")

    @property
    def source(self) -> str:
        """Alias used in manifests where a source is not always a file."""
        return self.path

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-friendly descriptor."""
        return {"name": self.name, "kind": self.kind, "path": self.path}


@dataclass(frozen=True)
class StabilizationConfig:
    """Validated top-level budget and execution configuration."""

    output: Path
    initial_bc: Path
    device: str = "cuda"
    workers: int = MAX_WORKERS
    updates: int = 8
    games_per_update: int = 16
    validation_deals: int = 10
    test_deals: int = 25
    seeds: tuple[int, ...] = DEFAULT_MODEL_SEEDS
    variants: tuple[VariantName, ...] = VARIANTS
    baselines: tuple[BaselineSpec, ...] = ()
    smoke: bool = False
    repo: Path | None = None
    eval_every: int = DEFAULT_EVAL_EVERY
    seed_base: int = TRAINING_SEED_START
    # Roadmap C1 critic-repair ablation knobs (defaults keep the frozen run).
    critic_learning_rate: float | None = None
    critic_hidden_dim: int = 0
    critic_warmup_epochs: int = 2
    value_coefficient: float = 0.5
    # Roadmap C3: stylized pool names; defaults keep the frozen pair.
    pool_names: tuple[str, ...] = TRAINING_POOL_NAMES
    # Roadmap A2: scalable training-seed group (registry-declared segment).
    training_seed_count: int = TRAINING_SEED_COUNT
    # Roadmap E1: self-play seat count for the training jobs (2 = frozen run).
    seats: int = 2
    # Roadmap E1: override the BC-derived schema (needed for >2 seats).
    feature_version_override: str | None = None
    # Roadmap B2->C2-R2: training-side reward shaping.
    shaping_kind: str = "none"
    shaping_kappa: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "initial_bc", Path(self.initial_bc))
        if self.repo is not None:
            object.__setattr__(self, "repo", Path(self.repo))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "variants", tuple(self.variants))
        object.__setattr__(self, "baselines", tuple(self.baselines))
        object.__setattr__(self, "pool_names", tuple(self.pool_names))
        validate_configuration(self, check_paths=False)

    @property
    def training_games_per_model(self) -> int:
        """Number of complete training games assigned to one model."""
        return self.updates * self.games_per_update

    @property
    def training_job_count(self) -> int:
        """Number of independent variant-by-model-seed jobs."""
        return len(self.variants) * len(self.seeds)

    @property
    def training_games_total(self) -> int:
        """Scheduled training-game denominator across all jobs."""
        return self.training_job_count * self.training_games_per_model


@dataclass(frozen=True)
class TrainingJob:
    """Pickle-safe input to one spawned training worker."""

    variant: VariantName
    model_seed: int
    job_index: int
    initial_bc: Path
    output_dir: Path
    manifest_path: Path
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    device: str
    updates: int
    games_per_update: int
    eval_every: int
    critic_learning_rate: float | None = None
    critic_hidden_dim: int = 0
    critic_warmup_epochs: int = 2
    value_coefficient: float = 0.5
    pool_names: tuple[str, ...] = TRAINING_POOL_NAMES
    seats: int = 2
    feature_version_override: str | None = None
    shaping_kind: str = "none"
    shaping_kappa: float = 0.05

    @property
    def name(self) -> str:
        """Stable candidate/job label used in every artifact."""
        return f"{self.variant}-seed{self.model_seed}"


@dataclass(frozen=True)
class EvaluationJob:
    """Pickle-safe input to one spawned evaluation worker."""

    benchmark: str
    candidate: BaselineSpec
    opponents: tuple[BaselineSpec, ...]
    seeds: tuple[int, ...]
    output_dir: Path
    device: str


def parse_baseline_spec(raw: str) -> BaselineSpec:
    """Parse ``NAME:KIND:PATH`` without constructing a policy in the parent."""
    parts = raw.split(":", 2)
    if len(parts) != 3:  # noqa: PLR2004 - NAME:KIND:PATH has exactly three fields
        raise ValueError(
            f"invalid baseline {raw!r}; expected NAME:KIND:PATH "
            "(repeat --baseline for multiple sources)"
        )
    name, raw_kind, path = (part.strip() for part in parts)
    kind = raw_kind.lower()
    if kind not in BASELINE_KINDS:
        raise ValueError(
            f"invalid baseline kind {raw_kind!r}; "
            f"expected one of {sorted(BASELINE_KINDS)}"
        )
    if kind == "builtin":
        # ``as ga`` occasionally gets written as ``asga`` in shell notes.  It
        # is harmless to accept that spelling while retaining the canonical
        # built-in name in all saved artifacts.
        builtin_names = set(FINAL_OPPONENT_NAMES) | {"corrected-dqn", "ppo"}
        if path.startswith("as") and path[2:] in builtin_names:
            path = path[2:]
    return BaselineSpec(name=name, kind=kind, path=path)  # type: ignore[arg-type]


def parse_baseline_specs(raw_values: Sequence[str]) -> tuple[BaselineSpec, ...]:
    """Parse and reject duplicate baseline labels."""
    specs = tuple(parse_baseline_spec(raw) for raw in raw_values)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("baseline names must be unique")
    return specs


def validate_configuration(  # noqa: C901,PLR0912 - independent experiment safety gates
    config: StabilizationConfig,
    *,
    check_paths: bool,
) -> None:
    """Validate argument bounds before any output directory is created."""
    if config.device not in {"cuda", "cpu"}:
        raise ValueError("device must be exactly 'cuda' or 'cpu'")
    if config.device == "cpu" and not config.smoke:
        raise ValueError("--device cpu requires explicit --smoke")
    if not 1 <= config.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if config.updates < 1 or config.games_per_update < 1:
        raise ValueError("updates and games_per_update must be positive")
    if not 1 <= config.validation_deals <= MAX_DEALS:
        raise ValueError(f"validation_deals must be between 1 and {MAX_DEALS}")
    if not 1 <= config.test_deals <= MAX_DEALS:
        raise ValueError(f"test_deals must be between 1 and {MAX_DEALS}")
    if not 1 <= config.eval_every:
        raise ValueError("eval_every must be positive")
    if config.seed_base < 0:
        raise ValueError("seed_base must be non-negative")
    if not config.seeds:
        raise ValueError("at least one model seed is required")
    if any(seed < 0 for seed in config.seeds):
        raise ValueError("model seeds must be non-negative")
    if len(config.seeds) != len(set(config.seeds)):
        raise ValueError("model seeds must be unique")
    if not config.variants:
        raise ValueError("at least one variant is required")
    if any(variant not in VARIANTS for variant in config.variants):
        raise ValueError(f"variants must be drawn from {VARIANTS}")
    if len(config.variants) != len(set(config.variants)):
        raise ValueError("variants must be unique")
    baseline_names = [spec.name for spec in config.baselines]
    if len(baseline_names) != len(set(baseline_names)):
        raise ValueError("baseline names must be unique")
    if config.training_games_total > config.training_seed_count:
        raise ValueError(
            "training budget exceeds the declared seed group "
            f"[{config.seed_base}, {config.seed_base + config.training_seed_count}): "
            f"{config.training_games_total} games > {config.training_seed_count} seeds"
        )
    if check_paths:
        if not config.initial_bc.is_file():
            raise FileNotFoundError(f"initial BC checkpoint does not exist: {config.initial_bc}")
        for spec in config.baselines:
            if spec.kind != "builtin" and not Path(spec.path).is_file():
                raise FileNotFoundError(
                    f"baseline checkpoint does not exist for {spec.name!r}: {spec.path}"
                )


def verify_seed_groups(
    seed_groups: Mapping[str, Sequence[int]],
    *,
    forbidden_ranges: Sequence[tuple[int, int]] = HISTORICAL_FORBIDDEN_SEED_RANGES,
) -> dict[str, list[int]]:
    """Verify group disjointness and historical-seed exclusion."""
    required = {"training", "validation", "final_test"}
    missing = sorted(required - set(seed_groups))
    if missing:
        raise ValueError(f"missing seed groups: {', '.join(missing)}")
    normalized: dict[str, list[int]] = {}
    owners: dict[int, str] = {}
    for name, values in seed_groups.items():
        seeds = [int(value) for value in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{name} seed group contains duplicates")
        for seed in seeds:
            if seed < 0:
                raise ValueError(f"{name} contains a negative seed")
            if any(start <= seed < end for start, end in forbidden_ranges):
                raise ValueError(f"{name} contains reserved historical seed {seed}")
            previous = owners.setdefault(seed, name)
            if previous != name:
                raise ValueError(f"seed {seed} overlaps groups {previous} and {name}")
        normalized[name] = seeds
    return normalized


def build_seed_groups(
    *,
    seed_base: int = TRAINING_SEED_START,
    validation_deals: int = 10,
    test_deals: int = 25,
    training_seed_count: int = TRAINING_SEED_COUNT,
) -> dict[str, list[int]]:
    """Construct the declared fresh deal-seed groups.

    Validation and final-test groups sit *behind* the training group so any
    scalable (roadmap C2) budget keeps the groups disjoint; the frozen
    stabilization offsets (2000/3000) are the training_count=2000 special case.
    """
    if seed_base < 0:
        raise ValueError("seed_base must be non-negative")
    if not 1 <= validation_deals <= MAX_DEALS:
        raise ValueError(f"validation_deals must be between 1 and {MAX_DEALS}")
    if not 1 <= test_deals <= MAX_DEALS:
        raise ValueError(f"test_deals must be between 1 and {MAX_DEALS}")
    if training_seed_count < 1:
        raise ValueError("training_seed_count must be positive")
    validation_offset = training_seed_count
    final_test_offset = training_seed_count + 1000
    groups = {
        "training": list(
            range(seed_base, seed_base + training_seed_count)
        ),
        "validation": list(
            range(
                seed_base + validation_offset,
                seed_base + validation_offset + validation_deals,
            )
        ),
        "final_test": list(
            range(
                seed_base + final_test_offset,
                seed_base + final_test_offset + test_deals,
            )
        ),
    }
    return verify_seed_groups(groups)


def expected_validation_updates(
    updates: int,
    *,
    eval_every: int = DEFAULT_EVAL_EVERY,
) -> tuple[int, ...]:
    """Return update zero plus every declared evaluation interval."""
    if updates < 1 or eval_every < 1:
        raise ValueError("updates and eval_every must be positive")
    return (0, *range(eval_every, updates + 1, eval_every))


def prepare_output_dir(path: Path) -> Path:
    """Create one fresh run directory and refuse every existing target."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _default_repo() -> Path:
    """Resolve the repository containing this package."""
    return Path(__file__).resolve().parents[5]


def _safe_component(value: str) -> str:
    """Make a candidate label safe as one output-directory component."""
    safe = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    )
    return safe or "unnamed"


def make_training_jobs(
    config: StabilizationConfig,
    *,
    root: Path | None = None,
    manifest_path: Path | None = None,
) -> list[TrainingJob]:
    """Allocate non-overlapping training deal seeds to every model job."""
    groups = build_seed_groups(
        seed_base=config.seed_base,
        validation_deals=config.validation_deals,
        test_deals=config.test_deals,
        training_seed_count=config.training_seed_count,
    )
    games_per_model = config.training_games_per_model
    output_root = Path(root) if root is not None else config.output
    declared_manifest = (
        Path(manifest_path) if manifest_path is not None else output_root / "manifest.json"
    )
    jobs: list[TrainingJob] = []
    job_index = 0
    training_group = groups["training"]
    for variant in config.variants:
        for model_seed in config.seeds:
            start = job_index * games_per_model
            deal_seeds = tuple(training_group[start : start + games_per_model])
            if len(deal_seeds) != games_per_model:
                raise ValueError("training seed allocation exceeded the declared group")
            jobs.append(
                TrainingJob(
                    variant=variant,
                    model_seed=model_seed,
                    job_index=job_index,
                    initial_bc=config.initial_bc,
                    output_dir=output_root
                    / "training"
                    / _safe_component(f"{variant}-seed{model_seed}"),
                    manifest_path=declared_manifest,
                    training_seeds=deal_seeds,
                    validation_seeds=tuple(groups["validation"]),
                    device=config.device,
                    updates=config.updates,
                    games_per_update=config.games_per_update,
                    eval_every=config.eval_every,
                    critic_learning_rate=config.critic_learning_rate,
                    critic_hidden_dim=config.critic_hidden_dim,
                    critic_warmup_epochs=config.critic_warmup_epochs,
                    value_coefficient=config.value_coefficient,
                    pool_names=config.pool_names,
                    seats=config.seats,
                    feature_version_override=config.feature_version_override,
                    shaping_kind=config.shaping_kind,
                    shaping_kappa=config.shaping_kappa,
                )
            )
            job_index += 1
    assigned = [seed for job in jobs for seed in job.training_seeds]
    if len(assigned) != len(set(assigned)):
        raise AssertionError("internal training seed allocation overlap")
    return jobs


def ppo_config_kwargs(  # noqa: PLR0913 - explicit per-run experiment parameters
    variant: VariantName,
    *,
    feature_version: str,
    model_seed: int,
    updates: int,
    games_per_update: int,
    device: str,
    eval_every: int = DEFAULT_EVAL_EVERY,
    critic_learning_rate: float | None = None,
    critic_hidden_dim: int = 0,
    critic_warmup_epochs: int = 2,
    value_coefficient: float = 0.5,
    seats: int = 2,
    feature_version_override: str | None = None,
    shaping_kind: str = "none",
    shaping_kappa: float = 0.05,
) -> dict[str, Any]:
    """Return the frozen PPO configuration for one stabilisation branch."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown PPO stabilisation variant {variant!r}")
    if model_seed < 0:
        raise ValueError("model_seed must be non-negative")
    if updates < 1 or games_per_update < 1:
        raise ValueError("updates and games_per_update must be positive")
    if eval_every < 1:
        raise ValueError("eval_every must be positive")
    return {
        "feature_version": feature_version,
        "learning_rate": 1e-4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coefficient": 0.005,
        "value_coefficient": value_coefficient,
        "max_grad_norm": 1.0,
        "minibatch_size": 256,
        "update_epochs": 4,
        "updates": updates,
        "games_per_update": games_per_update,
        "terminal_value": 10.0,
        "seed": model_seed,
        "device_name": device,
        "initialization": "bc" if variant in {"fixed", "anchor"} else "scratch",
        "reference_kl_coefficient": 0.02 if variant == "anchor" else 0.0,
        "target_kl": 0.02,
        "current_weight": 1.0,
        "history_weight": 1.0,
        "history_limit": 4,
        "critic_warmup_epochs": critic_warmup_epochs,
        "eval_every": eval_every,
        "critic_learning_rate": critic_learning_rate,
        "critic_hidden_dim": critic_hidden_dim,
        "n_seats": seats,
        "shaping_kind": shaping_kind,
        "shaping_kappa": shaping_kappa,
    }


def trainer_api_status() -> dict[str, Any]:
    """Describe whether the coordinated PPO changes are import-ready."""
    config_parameters = set(inspect.signature(PPOConfig).parameters)
    trainer_parameters = set(inspect.signature(train_ppo_selfplay).parameters)
    return {
        "missing_config_fields": sorted(REQUIRED_PPO_CONFIG_FIELDS - config_parameters),
        "missing_trainer_parameters": sorted(
            REQUIRED_TRAINER_PARAMETERS - trainer_parameters
        ),
        "ready": not (
            REQUIRED_PPO_CONFIG_FIELDS - config_parameters
            or REQUIRED_TRAINER_PARAMETERS - trainer_parameters
        ),
    }


def require_trainer_api_ready() -> None:
    """Reject an old trainer rather than dropping requested safeguards."""
    status = trainer_api_status()
    if not status["ready"]:
        raise RuntimeError(
            "ppo_selfplay is not agent-ready for stabilization: "
            + json.dumps(status, sort_keys=True)
        )


def _read_bc_feature_version(path: Path) -> str:
    """Read the BC schema without moving a model into the parent's GPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("BC checkpoint root must be a mapping")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping) or not config.get("feature_version"):
        raise ValueError("BC checkpoint is missing config.feature_version")
    return str(config["feature_version"])


def _make_ppo_config(  # noqa: PLR0913 - mirror per-run configuration inputs
    *,
    variant: VariantName,
    feature_version: str,
    model_seed: int,
    updates: int,
    games_per_update: int,
    device: str,
    eval_every: int,
    critic_learning_rate: float | None = None,
    critic_hidden_dim: int = 0,
    critic_warmup_epochs: int = 2,
    value_coefficient: float = 0.5,
    seats: int = 2,
    feature_version_override: str | None = None,
    shaping_kind: str = "none",
    shaping_kappa: float = 0.05,
) -> PPOConfig:
    """Instantiate the promised PPOConfig surface with no silent fallback."""
    schema = feature_version_override or feature_version
    kwargs = ppo_config_kwargs(
        variant,
        feature_version=schema,
        model_seed=model_seed,
        updates=updates,
        games_per_update=games_per_update,
        device=device,
        eval_every=eval_every,
        critic_learning_rate=critic_learning_rate,
        critic_hidden_dim=critic_hidden_dim,
        critic_warmup_epochs=critic_warmup_epochs,
        value_coefficient=value_coefficient,
        seats=seats,
        feature_version_override=feature_version_override,
        shaping_kind=shaping_kind,
        shaping_kappa=shaping_kappa,
    )
    try:
        return PPOConfig(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "PPOConfig rejected the declared stabilization configuration; "
            "review the coordinated ppo_selfplay API"
        ) from exc


def _fixed_role(candidate: CandidateSpec) -> CandidateSpec:
    """Mark a frozen candidate as an opponent without changing its factory."""
    return replace(candidate, role="fixed_baseline")


def build_candidate(spec: BaselineSpec, *, device: str) -> CandidateSpec:
    """Build one candidate inside the calling worker process.

    Checkpoint loading is intentionally here, rather than in the parent, so
    every evaluation job rereads its source and no CUDA module/factory is
    pickled across the spawn boundary.
    """
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unsupported device: {device}")
    device_name: Literal["cpu", "cuda", "mps"] = (
        "cuda" if device == "cuda" else "mps" if device == "mps" else "cpu"
    )
    if spec.kind == "bc":
        candidate = build_bc_candidate(Path(spec.path), device_name=device_name)
        return replace(candidate, name=spec.name)
    if spec.kind == "ppo":
        model = load_ppo_checkpoint(Path(spec.path), device_name=device_name)
        return build_policy_candidate(
            model,
            name=spec.name,
            snapshot=spec.path,
            device_name=device_name,
        )
    if spec.kind == "dqn":
        candidate = build_builtin_candidate(
            "corrected-dqn",
            checkpoint=Path(spec.path),
            device_name=device,
        )
        return replace(candidate, name=spec.name)
    candidate = build_builtin_candidate(spec.path, device_name=device)
    return replace(candidate, name=spec.name)


def _build_builtin_fixed(name: str, *, device: str) -> CandidateSpec:
    """Build a fixed built-in opponent in a worker."""
    return _fixed_role(build_builtin_candidate(name, device_name=device))


def _build_training_pool(
    *,
    device: str,
    names: Sequence[str] = TRAINING_POOL_NAMES,
) -> tuple[OpponentPoolEntry, ...]:
    """Build the fixed pool buckets; current/history are trainer-owned."""
    return tuple(
        OpponentPoolEntry(
            name,
            _build_builtin_fixed(name, device=device),
            weight=1.0,
        )
        for name in names
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one flushed JSON artifact, retaining partial results on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _job_as_dict(job: TrainingJob | EvaluationJob) -> dict[str, Any]:
    """Serialize a worker job without relying on JSON's Path handling."""
    return json.loads(json.dumps(asdict(job), default=str))


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _record_status_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count W/D/L and failures using the scheduled-game denominator."""
    counts = {
        "games": len(records),
        "completed_games": 0,
        "failed_games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
    }
    for record in records:
        if record.get("status") == "completed":
            counts["completed_games"] += 1
            outcome = record.get("outcome")
            if outcome == 1:
                counts["wins"] += 1
            elif outcome == 0:
                counts["draws"] += 1
            elif outcome == -1:
                counts["losses"] += 1
            else:
                raise ValueError("completed record has an invalid outcome")
        elif record.get("status") == "failed":
            counts["failed_games"] += 1
        else:
            raise ValueError("game record has an unknown status")
    return counts


def summarize_test_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Summarize raw records and paired-seed WDL rates without fake CIs."""
    expected_seeds = [int(seed) for seed in seeds]
    counts: dict[str, Any] = dict(_record_status_counts(records))
    per_seed: list[dict[str, Any]] = []
    for seed in expected_seeds:
        seed_records = [record for record in records if int(record["seed"]) == seed]
        seed_counts: dict[str, Any] = dict(_record_status_counts(seed_records))
        seed_counts.update(
            {
                "seed": seed,
                "scheduled_win_rate": seed_counts["wins"] / seed_counts["games"]
                if seed_counts["games"]
                else None,
                "completed_win_rate": (
                    seed_counts["wins"] / seed_counts["completed_games"]
                    if seed_counts["completed_games"]
                    else None
                ),
            }
        )
        per_seed.append(seed_counts)
    counts.update(
        {
            "scheduled_win_rate": counts["wins"] / counts["games"]
            if counts["games"]
            else None,
            "completed_win_rate": (
                counts["wins"] / counts["completed_games"]
                if counts["completed_games"]
                else None
            ),
            "seed_rates": per_seed,
            "rate_unit": "paired_seed_with_both_seats",
            "confidence_intervals": None,
            "confidence_interval_note": (
                "No independent CI is reported; both seats share one deal seed."
            ),
        }
    )
    return counts


def audit_game_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    candidate_name: str | None = None,
    opponent_name: str | None = None,
) -> dict[str, Any]:
    """Audit every expected ``(seed, seat)`` row before accepting a result."""
    expected = {(int(seed), seat) for seed in seeds for seat in (0, 1)}
    actual: set[tuple[int, int]] = set()
    for record in records:
        key = (int(record["seed"]), int(record["seat"]))
        if key in actual:
            raise ValueError(f"duplicate test record for seed/seat {key}")
        actual.add(key)
        if candidate_name is not None and record.get("candidate") != candidate_name:
            raise ValueError("test record candidate name disagrees with job")
        if opponent_name is not None and record.get("opponent") != opponent_name:
            raise ValueError("test record opponent name disagrees with job")
        if record["status"] == "completed":
            score = float(record["score"])
            rival_score = float(record["rival_score"])
            expected_outcome = int(score > rival_score) - int(score < rival_score)
            if record.get("outcome") != expected_outcome:
                raise ValueError("completed test record outcome disagrees with scores")
        elif record["status"] == "failed":
            # Failed games remain in the denominator; a failed outcome is not
            # converted into a loss or silently removed.
            if record.get("outcome") is not None:
                raise ValueError("failed test record must have outcome=null")
        else:
            raise ValueError("test record status must be completed or failed")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"test record coverage mismatch; missing={missing}, extra={extra}")
    return {
        "status": "pass",
        "games": len(records),
        "failed_games": sum(record["status"] == "failed" for record in records),
        "expected_seed_count": len(expected),
        "seat_set": [0, 1],
    }


def audit_evaluation_matrix(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    opponent_names: Sequence[str],
) -> dict[str, Any]:
    """Audit a complete candidate-by-opponent evaluation matrix."""
    if not matrix:
        raise ValueError("evaluation matrix is empty")
    audits: dict[str, Any] = {}
    failed_games = 0
    for candidate_name, opponent_reports in matrix.items():
        candidate_audits: dict[str, Any] = {}
        for opponent_name in opponent_names:
            if opponent_name not in opponent_reports:
                raise ValueError(
                    f"matrix is missing opponent {opponent_name!r} for {candidate_name!r}"
                )
            report = opponent_reports[opponent_name]
            records = report.get("records")
            if not isinstance(records, list):
                raise ValueError("evaluation report is missing raw records")
            game_audit = audit_game_records(
                records,
                seeds=seeds,
                candidate_name=candidate_name,
                opponent_name=opponent_name,
            )
            failed_games += int(game_audit["failed_games"])
            candidate_audits[opponent_name] = game_audit
        audits[candidate_name] = candidate_audits
    return {
        "status": "pass" if failed_games == 0 else "failed_records",
        "failed_games": failed_games,
        "candidates": audits,
    }


def summarize_evaluation_matrix(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Produce per-candidate/per-opponent WDL and paired-seed summaries."""
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate_name, opponent_reports in matrix.items():
        summaries[candidate_name] = {}
        for opponent_name, report in opponent_reports.items():
            records = report.get("records")
            if not isinstance(records, list):
                raise ValueError("evaluation report is missing raw records")
            summaries[candidate_name][opponent_name] = summarize_test_records(
                records,
                seeds=seeds,
            )
    return summaries


def _matrix_from_validation(
    value: Mapping[str, Any],
    *,
    opponent_names: Sequence[str],
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    """Accept the direct or wrapped matrix shapes used by PPO validation."""
    for key in ("results", "matrix", "reports"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return _matrix_from_validation(nested, opponent_names=opponent_names)
    if all(name in value for name in opponent_names):
        candidate_name = "validation-candidate"
        for report in value.values():
            if isinstance(report, Mapping) and isinstance(report.get("records"), list):
                if report["records"] and "candidate" in report["records"][0]:
                    candidate_name = str(report["records"][0]["candidate"])
                    break
        return {candidate_name: value}  # type: ignore[return-value]
    for candidate_name, nested in value.items():
        if isinstance(nested, Mapping) and all(
            name in nested for name in opponent_names
        ):
            return {str(candidate_name): nested}  # type: ignore[return-value]
    raise ValueError("validation entry does not contain all expected opponents")


def _validation_rows(train_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Find update rows while preserving the trainer's raw structure."""
    for key in ("validation_logs", "validation_history", "logs"):
        value = train_result.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    value = train_result.get("validation")
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for update, row in value.items():
            if isinstance(row, Mapping):
                rows.append({"update": int(update), **row})
        return rows
    return []


def audit_training_validation(  # noqa: C901,PLR0912,PLR0913,PLR0915 - explicit experiment accounting
    train_result: Mapping[str, Any],
    *,
    updates: int,
    eval_every: int,
    validation_seeds: Sequence[int],
    opponent_names: Sequence[str] = VALIDATION_OPPONENT_NAMES,
    games_per_update: int | None = None,
) -> dict[str, Any]:
    """Require update-zero/every-N validation and audit all raw records."""
    rows = _validation_rows(train_result)
    by_update: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if "update" not in row:
            continue
        update = int(row["update"])
        if update in by_update:
            raise ValueError(f"duplicate validation update {update}")
        by_update[update] = row
    expected_all_updates = set(range(updates + 1))
    if set(by_update) != expected_all_updates:
        missing_updates = sorted(expected_all_updates - set(by_update))
        extra_updates = sorted(set(by_update) - expected_all_updates)
        raise ValueError(
            "training logs must contain every update 0..N; "
            f"missing={missing_updates}, extra={extra_updates}"
        )
    for update, row in by_update.items():
        _assert_finite(row, path=f"logs[{update}]")
    if games_per_update is not None:
        total_training_games = sum(
            int(by_update[update].get("training_games", 0))
            for update in range(1, updates + 1)
        )
        expected_training_games = updates * games_per_update
        if total_training_games != expected_training_games:
            raise ValueError(
                "training game budget mismatch; "
                f"expected={expected_training_games}, actual={total_training_games}"
            )
    expected = expected_validation_updates(updates, eval_every=eval_every)
    missing = [update for update in expected if update not in by_update]
    if missing:
        raise ValueError(f"validation is missing required updates {missing}")
    normalized: list[dict[str, Any]] = []
    validation_scores: list[tuple[int, int, int]] = []
    validation_failed_games = 0
    for update in expected:
        row = by_update[update]
        if "status" not in row:
            raise ValueError(f"validation update {update} is missing status")
        raw_validation = row.get("validation")
        if not isinstance(raw_validation, Mapping):
            raw_validation = row.get("validation_results")
        if not isinstance(raw_validation, Mapping):
            raise ValueError(f"validation update {update} is missing its matrix")
        matrix = _matrix_from_validation(
            raw_validation,
            opponent_names=opponent_names,
        )
        audit = audit_evaluation_matrix(
            matrix,
            seeds=validation_seeds,
            opponent_names=opponent_names,
        )
        summary_counts = summarize_evaluation_matrix(
            matrix,
            seeds=validation_seeds,
        )
        wins = sum(
            int(opponent_summary["wins"])
            for candidate_summary in summary_counts.values()
            for opponent_summary in candidate_summary.values()
        )
        games = sum(
            int(opponent_summary["games"])
            for candidate_summary in summary_counts.values()
            for opponent_summary in candidate_summary.values()
        )
        validation_scores.append((wins, -games, update))
        validation_failed_games += int(audit.get("failed_games", 0))
        normalized.append(
            {
                "update": update,
                "status": row["status"],
                "audit": audit,
                "summary_counts": summary_counts,
                "raw": dict(row),
            }
        )
    best_wins = max(score[0] for score in validation_scores)
    best_update = min(score[2] for score in validation_scores if score[0] == best_wins)
    declared_best_update = train_result.get("best_update")
    if declared_best_update is None or int(declared_best_update) != best_update:
        raise ValueError(
            "best_update disagrees with integer validation wins; "
            f"declared={declared_best_update}, recomputed={best_update}"
        )
    declared_score = train_result.get("best_validation_score")
    if not isinstance(declared_score, (list, tuple)) or not declared_score:
        raise ValueError("best_validation_score is missing")
    if int(declared_score[0]) != best_wins:
        raise ValueError("best_validation_score wins disagree with validation records")
    best_path = Path(str(train_result.get("best", "")))
    if not best_path.is_file():
        raise FileNotFoundError("best checkpoint is missing for validation selection audit")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or int(checkpoint.get("update", -1)) != best_update:
        raise ValueError("best checkpoint update disagrees with validation selection")
    return {
        "status": "pass",
        "expected_updates": list(expected),
        "updates": normalized,
        "best_update": best_update,
        "best_wins": best_wins,
        "validation_failed_games": validation_failed_games,
    }


def _assert_finite(value: object, *, path: str) -> None:
    """Reject NaN/Inf in trainer metrics before a job can complete."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite metric at {path}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite(nested, path=f"{path}[{index}]")


def _training_game_failures(train_result: Mapping[str, Any]) -> int:
    failures = 0
    for row in train_result.get("logs", []):
        if not isinstance(row, Mapping):
            continue
        records = row.get("training_records", [])
        if isinstance(records, list):
            failures += sum(record.get("status") == "failed" for record in records)
    return failures


def _run_training_job(job: TrainingJob) -> dict[str, Any]:
    """Run and persist one independent PPO training job."""
    torch.set_num_threads(1)
    started = time.perf_counter()
    job.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    try:
        seed_everything(job.model_seed)
        feature_version = _read_bc_feature_version(job.initial_bc)
        config = _make_ppo_config(
            variant=job.variant,
            feature_version=feature_version,
            model_seed=job.model_seed,
            updates=job.updates,
            games_per_update=job.games_per_update,
            device=job.device,
            eval_every=job.eval_every,
            critic_learning_rate=job.critic_learning_rate,
            critic_hidden_dim=job.critic_hidden_dim,
            critic_warmup_epochs=job.critic_warmup_epochs,
            value_coefficient=job.value_coefficient,
            seats=job.seats,
            feature_version_override=job.feature_version_override,
            shaping_kind=job.shaping_kind,
            shaping_kappa=job.shaping_kappa,
        )
        pool = _build_training_pool(device=job.device, names=job.pool_names)
        validation_opponents = tuple(
            _build_builtin_fixed(name, device=job.device)
            for name in VALIDATION_OPPONENT_NAMES
        )
        train_result = train_ppo_selfplay(
            job.initial_bc,
            job.output_dir,
            job.training_seeds,
            pool,
            config=config,
            validation_seeds=job.validation_seeds,
            validation_opponents=validation_opponents,
            source_manifest=str(job.manifest_path),
        )
        validation_audit = audit_training_validation(
            train_result,
            updates=job.updates,
            eval_every=job.eval_every,
            validation_seeds=job.validation_seeds,
            games_per_update=job.games_per_update,
        )
        best_path = Path(str(train_result.get("best", "")))
        final_path = Path(str(train_result.get("final", "")))
        if not best_path.is_file() or not final_path.is_file():
            raise FileNotFoundError("trainer did not produce both best and final checkpoints")
        training_failures = _training_game_failures(train_result)
        validation_failures = int(validation_audit["validation_failed_games"])
        status = (
            "completed"
            if training_failures == 0 and validation_failures == 0
            else "completed_with_game_failures"
        )
        payload = {
            "status": status,
            "job": _job_as_dict(job),
            "candidate": job.name,
            "config": asdict(config),
            "training_seed_schedule": list(job.training_seeds),
            "validation_seed_schedule": list(job.validation_seeds),
            "training_game_failures": training_failures,
            "validation_game_failures": validation_failures,
            "validation_audit": validation_audit,
            "result": train_result,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # preserve a job-level failure as an artifact
        payload = {
            "status": "failed",
            "job": _job_as_dict(job),
            "candidate": job.name,
            "error": _error_payload(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
    _write_json(job.output_dir / "runresult.json", payload)
    return payload


def _run_evaluation_job(job: EvaluationJob) -> dict[str, Any]:
    """Rebuild candidates inside a spawn worker and evaluate raw records."""
    torch.set_num_threads(1)
    started = time.perf_counter()
    job.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Evaluation itself isolates every deal seed.  This initial seed keeps
        # any model construction or library-level RNG deterministic as well.
        seed_everything(job.seeds[0] if job.seeds else 0)
        candidate = build_candidate(job.candidate, device=job.device)
        opponents = [
            build_candidate(spec, device=job.device) for spec in job.opponents
        ]
        matrix = evaluate_matrix([candidate], opponents, job.seeds)
        audit = audit_evaluation_matrix(
            matrix,
            seeds=job.seeds,
            opponent_names=[opponent.name for opponent in opponents],
        )
        summaries = summarize_evaluation_matrix(matrix, seeds=job.seeds)
        failed_games = sum(
            summary[opponent]["failed_games"]
            for summary in summaries.values()
            for opponent in summary
        )
        status = "completed_with_game_failures" if failed_games else "completed"
        payload: dict[str, Any] = {
            "status": status,
            "benchmark": job.benchmark,
            "job": _job_as_dict(job),
            "candidate": job.candidate.as_dict(),
            "opponents": [spec.as_dict() for spec in job.opponents],
            "seeds": list(job.seeds),
            "audit": audit,
            "summary_counts": summaries,
            "matrix": matrix,
            "failed_games": failed_games,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "benchmark": job.benchmark,
            "job": _job_as_dict(job),
            "candidate": job.candidate.as_dict(),
            "opponents": [spec.as_dict() for spec in job.opponents],
            "seeds": list(job.seeds),
            "error": _error_payload(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
    _write_json(job.output_dir / "runresult.json", payload)
    return payload


def _future_failure(job: TrainingJob | EvaluationJob, exc: BaseException) -> dict[str, Any]:
    """Turn a broken worker process into a durable partial-result record."""
    output_dir = job.output_dir
    payload = {
        "status": "failed",
        "job": _job_as_dict(job),
        "error": _error_payload(exc),
    }
    _write_json(output_dir / "runresult.json", payload)
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks for manifest provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(repo: Path) -> list[Path]:
    """List policy-imitation and directly coupled agent source files."""
    roots = [
        repo / "src/splendor/agents/our_agents/policy_imitation",
        repo / "src/splendor/agents/our_agents/dqn",
        repo / "src/splendor/agents/our_agents/ppo",
        repo / "src/splendor/agents/our_agents/genetic_algorithm",
        repo / "src/splendor/agents/our_agents/minmax.py",
    ]
    paths: set[Path] = set()
    for root in roots:
        if root.is_dir():
            paths.update(path for path in root.glob("*.py") if path.is_file())
        elif root.is_file() and root.suffix == ".py":
            paths.add(root)
    test_path = repo / "tests/test_policy_imitation_stabilization.py"
    if test_path.is_file():
        paths.add(test_path)
    return sorted(paths)


def source_hashes(
    *,
    repo: Path,
    initial_bc: Path,
    baselines: Sequence[BaselineSpec],
) -> dict[str, Any]:
    """Hash model sources plus every directly coupled Python source file."""
    files: dict[str, str] = {}
    for path in _source_paths(repo):
        try:
            label = path.relative_to(repo).as_posix()
        except ValueError:
            label = str(path)
        files[label] = sha256_file(path)
    baseline_hashes: dict[str, Any] = {}
    for spec in baselines:
        entry: dict[str, Any] = {"kind": spec.kind, "source": spec.path}
        if spec.kind != "builtin":
            path = Path(spec.path).resolve()
            entry.update({"path": str(path), "sha256": sha256_file(path)})
        baseline_hashes[spec.name] = entry
    return {
        "source_bc": {
            "path": str(initial_bc.resolve()),
            "sha256": sha256_file(initial_bc),
        },
        "baselines": baseline_hashes,
        "source_files": files,
        "source_files_sha256": _sha256_bytes(
            json.dumps(files, sort_keys=True).encode("utf-8")
        ),
    }


def _git_bytes(repo: Path, arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return b""
    return result.stdout


def _dirty_patch_snapshot(repo: Path) -> bytes:
    """Create a deterministic tracked+untracked patch snapshot."""
    tracked = _git_bytes(repo, ("diff", "HEAD", "--binary"))
    untracked_raw = _git_bytes(
        repo,
        ("ls-files", "--others", "--exclude-standard"),
    )
    untracked = [
        line.decode("utf-8")
        for line in untracked_raw.splitlines()
        if line
    ]
    chunks = [b"# splendor uncommitted patch snapshot v1\n", tracked]
    for relative in sorted(untracked):
        path = repo / relative
        if not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes())
        chunks.extend(
            [
                b"\n# UNTRACKED_FILE_BEGIN ",
                relative.encode("utf-8"),
                b"\n",
                encoded,
                b"\n# UNTRACKED_FILE_END\n",
            ]
        )
    return b"".join(chunks)


def code_provenance(repo: Path, *, output_dir: Path) -> dict[str, Any]:
    """Capture commit, dirty paths, and a reproducible uncommitted snapshot."""
    revision = capture_code_revision(repo)
    patch = _dirty_patch_snapshot(repo)
    patch_path = output_dir / "code-uncommitted.patch"
    patch_path.write_bytes(patch)
    revision.update(
        {
            "dirty_diff_hash": _sha256_bytes(patch),
            "dirty_diff_path": str(patch_path.relative_to(output_dir)),
            "dirty_diff_format": "tracked git diff plus base64 untracked files",
        }
    )
    return revision


def fixed_config_manifest(config: StabilizationConfig, *, feature_version: str) -> dict[str, Any]:
    """Return the explicit numerical choices recorded in the manifest."""
    common = {
        "feature_version": feature_version,
        "learning_rate": 1e-4,
        "update_epochs": 4,
        "minibatch_size": 256,
        "target_kl": 0.02,
        "current_weight": 1.0,
        "history_weight": 1.0,
        "history_limit": 4,
        "eval_every": config.eval_every,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coefficient": 0.005,
        "max_grad_norm": 1.0,
        "terminal_value": 10.0,
        "critic_learning_rate": config.critic_learning_rate,
        "critic_hidden_dim": config.critic_hidden_dim,
        "critic_warmup_epochs": config.critic_warmup_epochs,
        "value_coefficient": config.value_coefficient,
        "pool_names": list(config.pool_names),
        "seats": config.seats,
        "feature_version_override": config.feature_version_override,
        "shaping_kind": config.shaping_kind,
        "shaping_kappa": config.shaping_kappa,
    }
    return {
        "common": common,
        "worker_pythonhashseed": "0",
        "variants": {
            variant: {
                "initialization": "bc"
                if variant in {"fixed", "anchor"}
                else "scratch",
                "reference_kl_coefficient": 0.02 if variant == "anchor" else 0.0,
            }
            for variant in config.variants
        },
        "training_pool": [
            {"name": name, "weight": 1.0} for name in TRAINING_POOL_NAMES
        ]
        + [
            {"name": "current", "weight": 1.0},
            {"name": "history", "weight": 1.0, "limit": 4},
        ],
        "validation_opponents": list(VALIDATION_OPPONENT_NAMES),
        "final_opponents": list(FINAL_OPPONENT_NAMES),
    }


def _update_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    manifest["updated_at"] = _now()
    validate_manifest(manifest)
    _write_json(manifest_path, manifest)


def _persist_state(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Flush all three monitoring artifacts after every completed job."""
    status["updated_at"] = _now()
    _write_json(output_dir / "suite-status.json", status)
    _write_json(output_dir / "results.json", results)
    _update_manifest(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "phase": status.get("phase"),
                "status": status.get("status"),
                "completed_training": status.get("completed_training"),
                "completed_evaluations": status.get("completed_evaluations"),
                "updated_at": status["updated_at"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _training_barrier_ok(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(payloads) and all(
        payload.get("status") == "completed"
        and int(payload.get("training_game_failures", 0)) == 0
        and int(payload.get("validation_game_failures", 0)) == 0
        for payload in payloads.values()
    )


def _make_evaluation_jobs(
    config: StabilizationConfig,
    *,
    output_dir: Path,
    training_payloads: Mapping[str, Mapping[str, Any]],
    test_seeds: Sequence[int],
) -> list[EvaluationJob]:
    """Create final-test jobs only from completed fixed checkpoints."""
    candidates: list[BaselineSpec] = []
    for variant in config.variants:
        for model_seed in config.seeds:
            name = f"{variant}-seed{model_seed}"
            payload = training_payloads[name]
            result = payload.get("result")
            if not isinstance(result, Mapping) or not result.get("best"):
                raise ValueError(f"training result has no best checkpoint for {name}")
            checkpoint = Path(str(result["best"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"best checkpoint does not exist for {name}: {checkpoint}")
            candidates.append(BaselineSpec(name=name, kind="ppo", path=str(checkpoint)))
    candidates.extend(config.baselines)
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("trained and baseline candidate names must be unique")
    opponents = tuple(
        BaselineSpec(name=name, kind="builtin", path=name)
        for name in FINAL_OPPONENT_NAMES
    )
    return [
        EvaluationJob(
            benchmark="final_test",
            candidate=candidate,
            opponents=opponents,
            seeds=tuple(int(seed) for seed in test_seeds),
            output_dir=output_dir / "final-test" / _safe_component(candidate.name),
            device=config.device,
        )
        for candidate in candidates
    ]


def _audit_all_final_records(
    evaluation_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check cross-job uniqueness after every individual matrix was audited."""
    seen: set[tuple[str, str, int, int]] = set()
    records = 0
    for payload in evaluation_payloads.values():
        matrix = payload.get("matrix")
        if not isinstance(matrix, Mapping):
            continue
        for candidate_name, opponent_reports in matrix.items():
            if not isinstance(opponent_reports, Mapping):
                continue
            for opponent_name, report in opponent_reports.items():
                if not isinstance(report, Mapping) or not isinstance(
                    report.get("records"), list
                ):
                    continue
                for record in report["records"]:
                    key = (
                        str(candidate_name),
                        str(opponent_name),
                        int(record["seed"]),
                        int(record["seat"]),
                    )
                    if key in seen:
                        raise ValueError(f"duplicate final-test record {key}")
                    seen.add(key)
                    records += 1
    return {"status": "pass", "unique_records": records}


Job = TypeVar("Job", TrainingJob, EvaluationJob)


def configure_worker_hash_seed() -> None:
    """Pin hash ordering before spawn starts fresh Python interpreters.

    Seeding random/NumPy/Torch cannot change Python's startup hash secret.
    Legacy legal-action enumeration uses sets; every training/evaluation
    worker must therefore inherit the same explicit startup hash seed.
    This does not claim to change the already-running parent's hash secret.
    """
    os.environ["PYTHONHASHSEED"] = "0"


def _run_pool_jobs(
    jobs: Sequence[Job],
    *,
    worker: Callable[[Job], dict[str, Any]],
    workers: int,
    on_result: Callable[[Job, dict[str, Any]], None],
) -> None:
    """Run top-level workers through a spawn-only process pool."""
    configure_worker_hash_seed()
    context = multiprocessing_module.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        future_map: dict[Future[dict[str, Any]], Job] = {
            pool.submit(worker, job): job for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                payload = future.result()
            except BaseException as exc:
                payload = _future_failure(job, exc)
            on_result(job, payload)


def run_stabilization(config: StabilizationConfig) -> dict[str, Any]:  # noqa: C901,PLR0915 - persist every lifecycle phase explicitly
    """Run the declared experiment, preserving every partial artifact."""
    validate_configuration(config, check_paths=True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback")
    if config.device == "cpu" and not config.smoke:
        raise ValueError("CPU execution is only allowed for an explicit smoke run")
    require_trainer_api_ready()

    output_dir = prepare_output_dir(config.output)
    repo = (config.repo or _default_repo()).resolve()
    groups = build_seed_groups(
        seed_base=config.seed_base,
        validation_deals=config.validation_deals,
        test_deals=config.test_deals,
        training_seed_count=config.training_seed_count,
    )
    feature_version = _read_bc_feature_version(config.initial_bc)
    hashes = source_hashes(
        repo=repo,
        initial_bc=config.initial_bc,
        baselines=config.baselines,
    )
    code = code_provenance(repo, output_dir=output_dir)
    runtime = RuntimeSnapshot.collect(config.device).as_dict()
    if runtime["resolved_device"] != config.device:
        raise RuntimeError(
            f"requested {config.device} but runtime resolved {runtime['resolved_device']}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest = create_manifest(
        manifest_path,
        experiment_id="ppo-stabilization-20260908",
        phase="3.4-ppo-stabilization",
        purpose="user-authorized local PPO stabilization implementation/training run",
        seed_groups=groups,
        budget={
            "model_seeds": list(config.seeds),
            "variants": list(config.variants),
            "training_jobs": config.training_job_count,
            "updates": config.updates,
            "games_per_update": config.games_per_update,
            "training_games_per_model": config.training_games_per_model,
            "training_games_total": config.training_games_total,
            "validation_deals": config.validation_deals,
            "validation_games_per_checkpoint": config.validation_deals
            * len(VALIDATION_OPPONENT_NAMES)
            * 2,
            "validation_updates": list(
                expected_validation_updates(config.updates, eval_every=config.eval_every)
            ),
            "test_deals": config.test_deals,
            "test_games_per_model": config.test_deals * len(FINAL_OPPONENT_NAMES) * 2,
            "workers": config.workers,
        },
        success_thresholds={
            "training_barrier": "all scheduled training jobs return a checkpoint",
            "validation_schedule": "update 0 plus every eval_every update",
            "final_test_barrier": "never start before all training jobs complete",
        },
        seats=(0, 1),
        forbidden_ranges=HISTORICAL_FORBIDDEN_SEED_RANGES,
        requested_device=config.device,
        repo=repo,
        status="running",
        notes=(
            "Authority: explicit user request dated 2026-09-08 for implementation/training.",
            "Numerical pilot budget was agent-chosen and is recorded, not represented as human-reviewed approval.",
            "Machine preflight passed before execution; approve_manifest was intentionally not invoked.",
            "This driver does not run official training during implementation or tests.",
        ),
    )
    manifest.update(
        {
            "authority": {
                "kind": "explicit_user_request",
                "date": USER_AUTHORITY_DATE,
                "scope": "implementation/training",
                "numerical_pilot_budget": "agent-chosen",
                "human_reviewed": False,
                "approve_manifest_invoked": False,
            },
            "machine_preflight": {
                "passed": True,
                "requested_device": config.device,
                "resolved_device": runtime["resolved_device"],
                "strict_cuda": config.device == "cuda",
                "workers": config.workers,
                "worker_torch_num_threads": 1,
            },
            "runtime": runtime,
            "code": code,
            "model_hashes": hashes,
            "fixed_configs": fixed_config_manifest(
                config,
                feature_version=feature_version,
            ),
            "candidate_names": {
                "trained": [
                    f"{variant}-seed{seed}"
                    for variant in config.variants
                    for seed in config.seeds
                ],
                "source_baselines": [spec.name for spec in config.baselines],
                "validation_opponents": list(VALIDATION_OPPONENT_NAMES),
                "final_opponents": list(FINAL_OPPONENT_NAMES),
            },
            "execution": {"status": "initializing", "started_at": _now()},
        }
    )
    validate_manifest(manifest)

    jobs = make_training_jobs(
        config,
        root=output_dir,
        manifest_path=manifest_path,
    )
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=False)
    status: dict[str, Any] = {
        "status": "training",
        "phase": "training",
        "expected_training": len(jobs),
        "completed_training": 0,
        "expected_evaluations": 0,
        "completed_evaluations": 0,
        "training_jobs": {job.name: {"status": "pending"} for job in jobs},
        "evaluation_jobs": {},
    }
    results: dict[str, Any] = {
        "status": "running",
        "training": {},
        "validation": {},
        "final_test": {},
        "summary_counts": {},
        "failures": [],
    }
    _persist_state(
        output_dir=output_dir,
        manifest=manifest,
        status=status,
        results=results,
    )

    training_payloads: dict[str, dict[str, Any]] = {}

    def on_training_result(job: TrainingJob | EvaluationJob, payload: dict[str, Any]) -> None:
        assert isinstance(job, TrainingJob)
        training_payloads[job.name] = payload
        results["training"][job.name] = payload
        if payload.get("status") == "failed":
            results["failures"].append({"phase": "training", "job": job.name})
        result = payload.get("result")
        if isinstance(result, Mapping):
            validation_audit = payload.get("validation_audit")
            if validation_audit is not None:
                results["validation"][job.name] = validation_audit
            trained_hashes = manifest["model_hashes"].setdefault("trained", {})
            if result.get("best") and Path(str(result["best"])).is_file():
                trained_hashes[job.name] = {
                    "best": {
                        "path": str(Path(str(result["best"])).resolve()),
                        "sha256": sha256_file(Path(str(result["best"]))),
                    },
                    "final": {
                        "path": str(Path(str(result["final"])).resolve()),
                        "sha256": sha256_file(Path(str(result["final"]))),
                    },
                }
        status["training_jobs"][job.name] = {
            "status": payload.get("status"),
            "runresult": str(job.output_dir / "runresult.json"),
            "training_game_failures": payload.get("training_game_failures", 0),
        }
        status["completed_training"] = len(training_payloads)
        manifest["execution"]["training_jobs"] = status["training_jobs"]
        _persist_state(
            output_dir=output_dir,
            manifest=manifest,
            status=status,
            results=results,
        )

    _run_pool_jobs(
        jobs,
        worker=_run_training_job,
        workers=config.workers,
        on_result=on_training_result,
    )

    if not _training_barrier_ok(training_payloads) or len(training_payloads) != len(jobs):
        status.update({"status": "training_failed", "phase": "training"})
        results.update(
            {
                "status": "partial",
                "final_test_status": "not_started_due_training_barrier",
            }
        )
        manifest["status"] = "blocked"
        manifest["execution"].update(
            {
                "status": "blocked",
                "reason": "training barrier failed; final test was not started",
                "finished_at": _now(),
            }
        )
        _persist_state(
            output_dir=output_dir,
            manifest=manifest,
            status=status,
            results=results,
        )
        return {
            "manifest": manifest,
            "status": status,
            "results": results,
        }

    # This is the explicit barrier: no EvaluationJob is submitted before all
    # training futures have returned and all best checkpoints are hashed.
    status.update({"status": "evaluating", "phase": "final_test"})
    test_jobs = _make_evaluation_jobs(
        config,
        output_dir=output_dir,
        training_payloads=training_payloads,
        test_seeds=groups["final_test"],
    )
    status["expected_evaluations"] = len(test_jobs)
    status["evaluation_jobs"] = {
        job.candidate.name: {"status": "pending"} for job in test_jobs
    }
    manifest["execution"]["status"] = "evaluating"
    manifest["execution"]["final_test_started_after_training_barrier"] = True
    _persist_state(
        output_dir=output_dir,
        manifest=manifest,
        status=status,
        results=results,
    )

    evaluation_payloads: dict[str, dict[str, Any]] = {}

    def on_evaluation_result(
        job: TrainingJob | EvaluationJob,
        payload: dict[str, Any],
    ) -> None:
        assert isinstance(job, EvaluationJob)
        name = job.candidate.name
        evaluation_payloads[name] = payload
        results["final_test"][name] = payload
        if payload.get("summary_counts"):
            results["summary_counts"][name] = payload["summary_counts"]
        if payload.get("status") == "failed":
            results["failures"].append({"phase": "final_test", "job": name})
        status["evaluation_jobs"][name] = {
            "status": payload.get("status"),
            "runresult": str(job.output_dir / "runresult.json"),
        }
        status["completed_evaluations"] = len(evaluation_payloads)
        manifest["execution"]["evaluation_jobs"] = status["evaluation_jobs"]
        _persist_state(
            output_dir=output_dir,
            manifest=manifest,
            status=status,
            results=results,
        )

    _run_pool_jobs(
        test_jobs,
        worker=_run_evaluation_job,
        workers=config.workers,
        on_result=on_evaluation_result,
    )

    evaluation_ok = len(evaluation_payloads) == len(test_jobs) and all(
        payload.get("status") == "completed"
        and int(payload.get("failed_games", 0)) == 0
        for payload in evaluation_payloads.values()
    )
    if evaluation_ok:
        try:
            final_audit = _audit_all_final_records(evaluation_payloads)
        except Exception as exc:
            evaluation_ok = False
            results["failures"].append(
                {"phase": "final_test_audit", "error": _error_payload(exc)}
            )
        else:
            results["final_test_audit"] = final_audit
    status["status"] = "completed" if evaluation_ok else "partial"
    status["phase"] = "final_test"
    results["status"] = "completed" if evaluation_ok else "partial"
    manifest["status"] = "completed" if evaluation_ok else "blocked"
    manifest["execution"].update(
        {
            "status": "completed" if evaluation_ok else "blocked",
            "finished_at": _now(),
            "final_test_audit": results.get("final_test_audit"),
        }
    )
    _persist_state(
        output_dir=output_dir,
        manifest=manifest,
        status=status,
        results=results,
    )
    return {"manifest": manifest, "status": status, "results": results}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone stabilisation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
        help="validation-evaluation interval in updates (every N updates)",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--games-per-update", type=int, default=16)
    parser.add_argument("--validation-deals", type=int, default=10)
    parser.add_argument("--test-deals", type=int, default=25)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=TRAINING_SEED_START,
        help="training seed base; validation/test follow the training group",
    )
    parser.add_argument(
        "--training-seed-count",
        type=int,
        default=TRAINING_SEED_COUNT,
        help=(
            "size of the declared training seed group; must cover "
            "updates x games-per-update x len(seeds) (roadmap A2 registry)"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_MODEL_SEEDS))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--initial-bc", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="NAME:KIND:PATH",
        help="repeat for bc, ppo, dqn, or builtin sources",
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=None,
        help="Roadmap C1 ablation: separate Adam lr for the value head",
    )
    parser.add_argument(
        "--critic-hidden-dim",
        type=int,
        default=0,
        help="Roadmap C1 ablation: critic-private hidden layer width (0 = off)",
    )
    parser.add_argument(
        "--critic-warmup-epochs",
        type=int,
        default=2,
        help="Roadmap C1 ablation: value-head warmup epochs before PPO updates",
    )
    parser.add_argument(
        "--value-coefficient",
        type=float,
        default=0.5,
        help="Roadmap C1 ablation: value-loss weight in the PPO total loss",
    )
    parser.add_argument(
        "--shaping",
        choices=("none", "potential", "event"),
        default="none",
        help=(
            "Roadmap B2: training-side reward shaping; 'potential' is the "
            "policy-invariant default (kappa via --shaping-kappa)"
        ),
    )
    parser.add_argument(
        "--shaping-kappa",
        type=float,
        default=0.05,
        help="noble-coverage weight in the potential function",
    )
    parser.add_argument(
        "--feature-version",
        choices=("v1", "public-v2", "public-v2-multi"),
        default=None,
        help=(
            "Roadmap E1: override the BC-derived observation schema; required "
            "with --seats 3/4 (public-v2-multi) unless the BC already matches"
        ),
    )
    parser.add_argument(
        "--seats",
        type=int,
        default=2,
        choices=(2, 3, 4),
        help=(
            "Roadmap E1: self-play seat count for training jobs; >2 requires "
            "the public-v2-multi feature schema and scratch initialization"
        ),
    )
    parser.add_argument(
        "--pool-names",
        type=str,
        default=",".join(TRAINING_POOL_NAMES),
        help=(
            "Roadmap C3: comma list of fixed training-pool opponents "
            "(ga, heuristic, heuristic-rush, heuristic-hoard, minimax, random)"
        ),
    )
    parser.add_argument("--repo", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> StabilizationConfig:
    """Convert parsed arguments into the validated driver configuration."""
    return StabilizationConfig(
        output=args.output,
        initial_bc=args.initial_bc,
        device=args.device,
        workers=args.workers,
        updates=args.updates,
        games_per_update=args.games_per_update,
        validation_deals=args.validation_deals,
        test_deals=args.test_deals,
        seeds=tuple(args.seeds),
        variants=tuple(args.variants),
        baselines=parse_baseline_specs(args.baseline),
        smoke=bool(args.smoke),
        repo=args.repo,
        seed_base=args.seed_base,
        training_seed_count=args.training_seed_count,
        eval_every=args.eval_every,
        critic_learning_rate=args.critic_learning_rate,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_warmup_epochs=args.critic_warmup_epochs,
        value_coefficient=args.value_coefficient,
        pool_names=tuple(
            name.strip() for name in args.pool_names.split(",") if name.strip()
        ),
        seats=args.seats,
        feature_version_override=args.feature_version,
        shaping_kind=args.shaping,
        shaping_kappa=args.shaping_kappa,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the driver CLI and return a shell status code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        run_stabilization(config)
    except Exception as exc:
        print(f"stabilization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
