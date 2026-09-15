"""Fail-closed tmux orchestration for the Task-1 T1.4 crossed pilot.

The module deliberately does not create a seed roll, materialize a scenario
bank, or transition a manifest into ``running``.  It consumes only an already
approved/running ``paired-training-v2`` declaration and delegates the formal
bank, schedule, manifest, checkpoint, and one-shot output gates to the existing
Task-1 implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import FrameType
from typing import Any, Final, cast

import torch

from splendor.seed_registry import TASK1_VALIDATION_A

from . import ppo_selfplay as ppo_selfplay_module
from .manifest import load_manifest, transition_manifest
from .ppo_selfplay import (
    FormalOpponentSpec,
    FormalPPOTrainingJob,
    PPOConfig,
    formal_ppo_config_sha256,
    make_formal_treatment_contract,
    make_formal_validation_contract,
    run_formal_ppo_training_jobs,
)
from .protocol import (
    FORMAL_VALIDATION_SELECTION_RULE,
    FORMAL_VALIDATION_UPDATES,
    FormalJobOutput,
    FormalTrainingSpec,
    sha256_canonical_json,
    sha256_file,
)
from .scenario_bank import (
    ScenarioBank,
    inspect_scenario_bank,
    load_scenario_bank,
)
from .seed_roll import (
    SeedRollArtifact,
    SeedRollSelectionDesign,
    load_seed_roll_artifact,
    make_seed_rolled_training_schedule,
    require_seed_roll_binding,
    require_task1_formal_seed_roll,
)

ORCHESTRATION_SCHEMA: Final = "splendor-t14-pilot-orchestration/2"
COMPLETION_SCHEMA: Final = "splendor-t14-pilot-completion/2"
STATUS_SCHEMA: Final = "splendor-t14-pilot-status/2"
PILOT_REPLICATE_IDS: Final = (0, 1, 2)
PILOT_TREATMENT_IDS: Final = ("O", "O_bridge")
PILOT_WORKER_COUNT: Final = 3
PILOT_JOB_COUNT: Final = 6
PILOT_UPDATES: Final = 2_000
PILOT_GAMES_PER_UPDATE: Final = 16
PILOT_GAMES_PER_REPLICATE: Final = 32_000
PILOT_SEAT_COUNT: Final = 2
PILOT_EVAL_EVERY: Final = 50
PILOT_VALIDATION_SCENARIO_COUNT: Final = 10
PILOT_VALIDATION_OPPONENT_IDS: Final = ("ga", "heuristic", "minimax")
PILOT_VALIDATION_GAMES: Final = 60
PILOT_TRAINING_OPPONENT_IDS: Final = (
    "ga",
    "heuristic",
    "heuristic-rush",
    "heuristic-hoard",
    "minimax",
)
PILOT_INITIAL_BC_RELATIVE_PATH: Final = Path(
    "runs/policy-imitation/formal-3.3-20260907/bc-dagger-2/best.pth"
)
PILOT_INITIAL_BC_SHA256: Final = (
    "e64b722f34a8eed390ae03bdd5adf2f03be9ec6238db556b86180adff1e6f867"
)
WALL_LIMIT_SECONDS: Final = 18 * 60 * 60
OUTPUT_LIMIT_BYTES: Final = 32 * 1024**3
MIN_FREE_BYTES: Final = 40 * 1024**3
WATCHDOG_POLL_SECONDS: Final = 5.0
STATUS_REFRESH_SECONDS: Final = 30.0
LAUNCH_HANDSHAKE_SECONDS: Final = 30.0
LAUNCH_HANDSHAKE_POLL_SECONDS: Final = 0.25
TERMINATE_GRACE_SECONDS: Final = 15.0
EXPECTED_GPU_CAPABILITY: Final = (6, 1)
EXPECTED_GPU_COMPATIBLE_ARCHES: Final = ("sm_60", "sm_61")
EXPECTED_GPU_NAME_SUFFIX: Final = "Quadro P5000"
STATUS_FILE_NAME: Final = "orchestrator-status.json"
COMPLETION_FILE_NAME: Final = "completion.json"
MATRIX_FAILURE_FILE_NAME: Final = "matrix-watchdog-failure.json"
LAUNCH_DECLARATION_FILE_NAME: Final = "launch-declaration.json"
SUPERVISOR_LOG_FILE_NAME: Final = "supervisor.log"
_SESSION_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


class PilotOrchestrationError(RuntimeError):
    """Raised when a production declaration or runtime fails closed."""


@dataclass(frozen=True)
class ProductionRuntime:
    """Small runtime snapshot used by the P5000 admission gate."""

    python_executable: str
    python_prefix: str
    python_hash_seed: str | None
    hash_randomization: int
    cublas_workspace_config: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_capability: tuple[int, int] | None
    cuda_arch_list: tuple[str, ...]
    torch_cuda_version: str | None


@dataclass(frozen=True)
class ResourceSnapshot:
    """Current output footprint and free-space measurement."""

    output_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class TreatmentLaunch:
    """One manifest-bound treatment input bundle."""

    treatment_id: str
    initial_bc: Path
    config: PPOConfig
    opponent_pool: tuple[FormalOpponentSpec, ...]


@dataclass(frozen=True)
class PilotDeclaration:
    """Canonical, immutable inputs for one T1.4 pilot attempt."""

    path: Path
    declaration_sha256: str
    experiment_id: str
    repository: Path
    python_executable: Path
    manifest_path: Path
    seed_roll_path: Path
    scenario_bank_path: Path
    validation_scenario_bank_path: Path
    output_root: Path
    control_dir: Path
    tmux_session: str
    treatments: tuple[TreatmentLaunch, ...]

    @property
    def status_path(self) -> Path:
        return self.control_dir / STATUS_FILE_NAME

    @property
    def completion_path(self) -> Path:
        return self.control_dir / COMPLETION_FILE_NAME


@dataclass(frozen=True)
class PreparedPilot:
    """Fully audited six-job matrix ready for the existing formal runner."""

    declaration: PilotDeclaration
    seed_roll: SeedRollArtifact
    bank: ScenarioBank
    validation_bank: ScenarioBank
    jobs: tuple[FormalPPOTrainingJob, ...]
    formal_training_sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PilotOrchestrationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _exact_keys(
    raw: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise PilotOrchestrationError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _canonical_path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise PilotOrchestrationError(f"{label} must be an absolute path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or os.path.normpath(value) != value
        or Path(os.path.abspath(value)) != path  # noqa: PTH100
    ):
        raise PilotOrchestrationError(f"{label} must be normalized and absolute")
    return path


def _read_regular_json(path: Path) -> tuple[dict[str, object], str]:
    lexical = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    if lexical != path or lexical.resolve(strict=False) != lexical:
        raise PilotOrchestrationError(
            "orchestration declaration path must not traverse a symlink"
        )
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise PilotOrchestrationError(f"cannot inspect declaration: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PilotOrchestrationError("orchestration declaration is not a regular file")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotOrchestrationError(f"cannot load declaration JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PilotOrchestrationError("orchestration declaration root must be an object")
    typed = cast(dict[str, object], raw)
    return typed, sha256_canonical_json(typed)


def _parse_config(raw: object) -> PPOConfig:
    if not isinstance(raw, Mapping):
        raise PilotOrchestrationError("treatment config must be an object")
    values = dict(raw)
    allowed = {field.name for field in fields(PPOConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise PilotOrchestrationError(f"unknown PPO config fields: {unknown}")
    if isinstance(values.get("hidden_layers"), list):
        values["hidden_layers"] = tuple(values["hidden_layers"])
    try:
        config = PPOConfig(**values)
    except (TypeError, ValueError) as exc:
        raise PilotOrchestrationError(f"invalid PPO config: {exc}") from exc
    if config.device_name != "cuda":
        raise PilotOrchestrationError("pilot PPO configs must request cuda")
    if config.seed != 0:
        raise PilotOrchestrationError(
            "pilot PPO config.seed must be the formal unused sentinel 0"
        )
    if config.n_seats != PILOT_SEAT_COUNT:
        raise PilotOrchestrationError("T1.4 pilot PPO configs must use two seats")
    if (
        config.updates != PILOT_UPDATES
        or config.games_per_update != PILOT_GAMES_PER_UPDATE
    ):
        raise PilotOrchestrationError(
            "each pilot PPO config must use exactly 2,000 updates x 16 games"
        )
    if config.eval_every != PILOT_EVAL_EVERY:
        raise PilotOrchestrationError(
            "each pilot PPO config must validate every 50 updates"
        )
    return config


def _expected_treatment_config(treatment_id: str) -> PPOConfig:
    """Return the frozen C2-R2 recipe with only the reward contract changed."""
    shaping_kind = {
        "O": "safe-potential",
        "O_bridge": "potential",
    }.get(treatment_id)
    if shaping_kind is None:
        raise PilotOrchestrationError(f"unknown pilot treatment {treatment_id!r}")
    return PPOConfig(
        feature_version="public-v2",
        hidden_layers=(128, 128, 128, 128),
        value_mode="return",
        learning_rate=1e-4,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.005,
        value_coefficient=1.0,
        max_grad_norm=1.0,
        minibatch_size=256,
        update_epochs=4,
        updates=PILOT_UPDATES,
        games_per_update=PILOT_GAMES_PER_UPDATE,
        terminal_value=10.0,
        target_kl=0.02,
        reference_kl_coefficient=0.0,
        current_weight=1.0,
        history_weight=1.0,
        history_limit=4,
        critic_warmup_epochs=2,
        critic_learning_rate=5e-4,
        critic_hidden_dim=0,
        n_seats=PILOT_SEAT_COUNT,
        shaping_kind=shaping_kind,
        shaping_kappa=0.05,
        initialization="bc",
        eval_every=PILOT_EVAL_EVERY,
        seed=0,
        device_name="cuda",
    )


def _require_exact_treatment_recipe(treatments: Sequence[TreatmentLaunch]) -> None:
    """Prevent a correctly hashed manifest from approving a mislabeled arm."""
    for treatment in treatments:
        expected = _expected_treatment_config(treatment.treatment_id)
        if formal_ppo_config_sha256(treatment.config) != formal_ppo_config_sha256(
            expected
        ):
            raise PilotOrchestrationError(
                f"treatment {treatment.treatment_id!r} does not match the frozen "
                "C2-R2/safe-PBRS pilot recipe"
            )
        observed_pool = tuple(
            (
                entry.entry_name,
                entry.candidate_name,
                entry.weight,
                entry.checkpoint,
            )
            for entry in treatment.opponent_pool
        )
        expected_pool = tuple(
            (name, name, 1.0, None) for name in PILOT_TRAINING_OPPONENT_IDS
        )
        if observed_pool != expected_pool or any(
            type(entry.weight) is not float for entry in treatment.opponent_pool
        ):
            raise PilotOrchestrationError(
                "pilot training pool must be the ordered, unit-weight source-only "
                "ga/heuristic/rush/hoard/minimax pool"
            )


def _parse_opponent_pool(raw: object) -> tuple[FormalOpponentSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise PilotOrchestrationError("treatment opponent_pool must be non-empty")
    result: list[FormalOpponentSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise PilotOrchestrationError(f"opponent_pool[{index}] must be an object")
        _exact_keys(
            item,
            {"entry_name", "candidate_name", "weight", "checkpoint"},
            label=f"opponent_pool[{index}]",
        )
        checkpoint_value = item["checkpoint"]
        checkpoint = (
            None
            if checkpoint_value is None
            else _canonical_path(
                checkpoint_value,
                label=f"opponent_pool[{index}].checkpoint",
            )
        )
        try:
            result.append(
                FormalOpponentSpec(
                    entry_name=cast(str, item["entry_name"]),
                    candidate_name=cast(str, item["candidate_name"]),
                    weight=cast(float, item["weight"]),
                    checkpoint=checkpoint,
                )
            )
        except (TypeError, ValueError) as exc:
            raise PilotOrchestrationError(
                f"invalid opponent_pool[{index}]: {exc}"
            ) from exc
    names = [entry.entry_name for entry in result]
    if len(names) != len(set(names)):
        raise PilotOrchestrationError("opponent_pool entry names must be unique")
    return tuple(result)


def load_pilot_declaration(  # noqa: C901,PLR0912,PLR0915 - strict schema gate
    path: Path,
) -> PilotDeclaration:
    """Load a strict, semantically content-addressed pilot declaration."""
    source = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    raw, digest = _read_regular_json(source)
    _exact_keys(
        raw,
        {
            "schema_version",
            "experiment_id",
            "phase",
            "repository",
            "python_executable",
            "manifest_path",
            "seed_roll_path",
            "scenario_bank_path",
            "validation_scenario_bank_path",
            "output_root",
            "control_dir",
            "tmux_session",
            "replicate_ids",
            "worker_count",
            "limits",
            "treatments",
        },
        label="orchestration declaration",
    )
    if raw["schema_version"] != ORCHESTRATION_SCHEMA:
        raise PilotOrchestrationError("orchestration schema mismatch")
    if raw["phase"] != "T1.4":
        raise PilotOrchestrationError("pilot orchestration phase must be exactly T1.4")
    experiment_id = raw["experiment_id"]
    if type(experiment_id) is not str or not experiment_id.strip():
        raise PilotOrchestrationError("experiment_id must be a non-empty string")
    replicate_ids = raw["replicate_ids"]
    if (
        type(replicate_ids) is not list
        or len(replicate_ids) != len(PILOT_REPLICATE_IDS)
        or any(type(item) is not int for item in replicate_ids)
        or tuple(replicate_ids) != PILOT_REPLICATE_IDS
    ):
        raise PilotOrchestrationError("pilot replicate_ids must be exactly [0, 1, 2]")
    if type(raw["worker_count"]) is not int or raw["worker_count"] != (
        PILOT_WORKER_COUNT
    ):
        raise PilotOrchestrationError("pilot worker_count must be exactly 3")
    limits = raw["limits"]
    if not isinstance(limits, Mapping):
        raise PilotOrchestrationError("limits must be an object")
    _exact_keys(
        limits,
        {"wall_seconds", "max_output_bytes", "min_free_bytes"},
        label="limits",
    )
    expected_limits = {
        "wall_seconds": WALL_LIMIT_SECONDS,
        "max_output_bytes": OUTPUT_LIMIT_BYTES,
        "min_free_bytes": MIN_FREE_BYTES,
    }
    if (
        any(type(limits[name]) is not int for name in expected_limits)
        or dict(limits) != expected_limits
    ):
        raise PilotOrchestrationError(
            "pilot limits must be exactly 18h wall, 32GiB output, and 40GiB free"
        )
    treatments_raw = raw["treatments"]
    if not isinstance(treatments_raw, list) or len(treatments_raw) != len(
        PILOT_TREATMENT_IDS
    ):
        raise PilotOrchestrationError("pilot must declare exactly two treatments")
    treatments: list[TreatmentLaunch] = []
    for index, item in enumerate(treatments_raw):
        if not isinstance(item, Mapping):
            raise PilotOrchestrationError(f"treatments[{index}] must be an object")
        _exact_keys(
            item,
            {"treatment_id", "initial_bc", "config", "opponent_pool"},
            label=f"treatments[{index}]",
        )
        treatment_id = item["treatment_id"]
        if type(treatment_id) is not str:
            raise PilotOrchestrationError("treatment_id must be a string")
        treatments.append(
            TreatmentLaunch(
                treatment_id=treatment_id,
                initial_bc=_canonical_path(
                    item["initial_bc"], label=f"treatments[{index}].initial_bc"
                ),
                config=_parse_config(item["config"]),
                opponent_pool=_parse_opponent_pool(item["opponent_pool"]),
            )
        )
    if tuple(item.treatment_id for item in treatments) != PILOT_TREATMENT_IDS:
        raise PilotOrchestrationError(
            "pilot treatments must be ordered exactly ['O', 'O_bridge']"
        )
    _require_exact_treatment_recipe(treatments)
    if len(
        {
            (item.config.updates, item.config.games_per_update)
            for item in treatments
        }
    ) != 1:
        raise PilotOrchestrationError(
            "paired pilot treatments must share one update/game schedule shape"
        )
    repository = _canonical_path(raw["repository"], label="repository")
    python_executable = _canonical_path(
        raw["python_executable"], label="python_executable"
    )
    expected_python = repository / ".venv-p5000" / "bin" / "python"
    if python_executable != expected_python:
        raise PilotOrchestrationError(
            "python_executable must be <repository>/.venv-p5000/bin/python"
        )
    expected_initial_bc = repository / PILOT_INITIAL_BC_RELATIVE_PATH
    if any(treatment.initial_bc != expected_initial_bc for treatment in treatments):
        raise PilotOrchestrationError(
            "both pilot treatments must use the frozen DAgger-2 BC initializer"
        )
    tmux_session = raw["tmux_session"]
    if type(tmux_session) is not str or _SESSION_PATTERN.fullmatch(tmux_session) is None:
        raise PilotOrchestrationError("tmux_session is not a canonical session name")
    declaration = PilotDeclaration(
        path=source,
        declaration_sha256=digest,
        experiment_id=experiment_id,
        repository=repository,
        python_executable=python_executable,
        manifest_path=_canonical_path(raw["manifest_path"], label="manifest_path"),
        seed_roll_path=_canonical_path(
            raw["seed_roll_path"], label="seed_roll_path"
        ),
        scenario_bank_path=_canonical_path(
            raw["scenario_bank_path"], label="scenario_bank_path"
        ),
        validation_scenario_bank_path=_canonical_path(
            raw["validation_scenario_bank_path"],
            label="validation_scenario_bank_path",
        ),
        output_root=_canonical_path(raw["output_root"], label="output_root"),
        control_dir=_canonical_path(raw["control_dir"], label="control_dir"),
        tmux_session=tmux_session,
        treatments=tuple(treatments),
    )
    if declaration.control_dir != declaration.path.parent:
        raise PilotOrchestrationError(
            "control_dir must be the real parent of the declaration"
        )
    if declaration.output_root == declaration.repository or (
        declaration.repository not in declaration.output_root.parents
    ):
        raise PilotOrchestrationError(
            "output_root must be a dedicated directory inside the repository"
        )
    if declaration.control_dir == declaration.output_root or (
        declaration.output_root in declaration.control_dir.parents
    ):
        raise PilotOrchestrationError(
            "control_dir must not lie inside the formal output root"
        )
    return declaration


def write_pilot_declaration(
    path: Path,
    payload: Mapping[str, object],
) -> PilotDeclaration:
    """Validate and atomically publish one immutable production declaration.

    A same-directory temporary inode is parsed through the complete declaration
    gate before an exclusive hard-link publishes it.  Therefore an invalid
    payload never occupies the requested one-shot path, and an existing path is
    never replaced.
    """
    target = _canonical_path(str(path), label="declaration output path")
    parent = target.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise PilotOrchestrationError(
            "declaration output parent must be an existing real directory"
        )
    try:
        data = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotOrchestrationError(
            f"declaration payload is not canonical JSON: {exc}"
        ) from exc
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        load_pilot_declaration(temporary)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except OSError as exc:
            raise PilotOrchestrationError(
                f"one-shot orchestration declaration already exists: {target}"
            ) from exc
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return load_pilot_declaration(target)


def capture_production_runtime() -> ProductionRuntime:
    """Capture the exact interpreter and visible CUDA device."""
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    name = torch.cuda.get_device_name(0) if available and count else None
    capability = (
        tuple(torch.cuda.get_device_capability(0)) if available and count else None
    )
    architectures = tuple(torch.cuda.get_arch_list()) if available and count else ()
    return ProductionRuntime(
        python_executable=sys.executable,
        python_prefix=sys.prefix,
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        hash_randomization=sys.flags.hash_randomization,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cuda_available=available,
        cuda_device_count=count,
        cuda_device_name=name,
        cuda_capability=cast(tuple[int, int] | None, capability),
        cuda_arch_list=architectures,
        torch_cuda_version=torch.version.cuda,
    )


def require_production_runtime(
    declaration: PilotDeclaration,
    runtime: ProductionRuntime | None = None,
) -> ProductionRuntime:
    """Fail closed unless this is the declared P5000/sm61 CUDA environment."""
    snapshot = runtime or capture_production_runtime()
    expected_prefix = declaration.repository / ".venv-p5000"
    if Path(snapshot.python_executable) != declaration.python_executable:
        raise PilotOrchestrationError(
            "production pilot must run via .venv-p5000/bin/python"
        )
    if Path(snapshot.python_prefix) != expected_prefix:
        raise PilotOrchestrationError("production pilot has the wrong virtualenv prefix")
    if snapshot.python_hash_seed != "0" or snapshot.hash_randomization != 0:
        raise PilotOrchestrationError(
            "production pilot requires PYTHONHASHSEED=0 before Python starts"
        )
    if snapshot.cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise PilotOrchestrationError(
            "production pilot requires CUBLAS_WORKSPACE_CONFIG before CUDA starts"
        )
    if (
        not snapshot.cuda_available
        or snapshot.cuda_device_count < 1
        or snapshot.torch_cuda_version is None
    ):
        raise PilotOrchestrationError(
            "production pilot requires CUDA and never falls back to CPU"
        )
    if (
        snapshot.cuda_device_name is None
        or not snapshot.cuda_device_name.endswith(EXPECTED_GPU_NAME_SUFFIX)
    ):
        raise PilotOrchestrationError(
            f"production pilot requires {EXPECTED_GPU_NAME_SUFFIX}, got "
            f"{snapshot.cuda_device_name!r}"
        )
    if snapshot.cuda_capability != EXPECTED_GPU_CAPABILITY:
        raise PilotOrchestrationError(
            "production pilot requires compute capability sm61 (6.1)"
        )
    # CUDA cubins are binary-compatible within one compute-capability major
    # version from a lower minor target to a higher minor device.  The official
    # The PyTorch wheel installed in .venv-p5000 advertises sm_60 and executes
    # it on this exact sm_61 desktop GPU; an sm_61-native build is valid too.
    if not any(
        architecture in snapshot.cuda_arch_list
        for architecture in EXPECTED_GPU_COMPATIBLE_ARCHES
    ):
        raise PilotOrchestrationError(
            "the installed torch build has no sm_61-compatible Pascal target"
        )
    return snapshot


def _manifest_job_outputs(
    manifest: Mapping[str, object], declaration: PilotDeclaration
) -> tuple[FormalJobOutput, ...]:
    raw_declaration = manifest.get("declaration")
    if not isinstance(raw_declaration, Mapping):
        raise PilotOrchestrationError("manifest declaration is missing")
    formal = raw_declaration.get("formal_training")
    if not isinstance(formal, Mapping):
        raise PilotOrchestrationError("manifest formal_training binding is missing")
    raw_outputs = formal.get("job_outputs")
    if not isinstance(raw_outputs, list):
        raise PilotOrchestrationError("manifest formal job_outputs are missing")
    outputs: list[FormalJobOutput] = []
    for raw in raw_outputs:
        if not isinstance(raw, Mapping):
            raise PilotOrchestrationError("manifest formal job output is invalid")
        try:
            output = FormalJobOutput(
                replicate_id=cast(int, raw["replicate_id"]),
                treatment_id=cast(str, raw["treatment_id"]),
                output_dir=cast(str, raw["output_dir"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PilotOrchestrationError(
                f"manifest formal job output is invalid: {exc}"
            ) from exc
        output_path = Path(output.output_dir)
        if output_path.parent != declaration.output_root:
            raise PilotOrchestrationError(
                "every formal job output must be a direct child of output_root"
            )
        outputs.append(output)
    expected = tuple(
        (replicate_id, treatment_id)
        for replicate_id in PILOT_REPLICATE_IDS
        for treatment_id in PILOT_TREATMENT_IDS
    )
    actual = tuple((item.replicate_id, item.treatment_id) for item in outputs)
    if actual != expected:
        raise PilotOrchestrationError(
            "manifest must contain the canonical six-job pilot output matrix"
        )
    return tuple(outputs)


def _require_unattempted_outputs(outputs: Sequence[FormalJobOutput]) -> None:
    for output in outputs:
        if os.path.lexists(output.output_dir):
            raise PilotOrchestrationError(
                f"formal output was already attempted: {output.output_dir}"
            )


def _require_pilot_bank(bank: ScenarioBank) -> SeedRollSelectionDesign:
    if bank.sealed:
        raise PilotOrchestrationError("pilot orchestration refuses sealed banks")
    if bank.logical_split != "train-schedule":
        raise PilotOrchestrationError(
            "pilot orchestration accepts only the train-schedule bank"
        )
    design = bank.selection_design
    if not isinstance(design, SeedRollSelectionDesign):
        raise PilotOrchestrationError(
            "pilot requires a seed-rolled natural-deal scenario bank"
        )
    if design.active_replicate_ids != PILOT_REPLICATE_IDS:
        raise PilotOrchestrationError(
            "pilot bank must contain exactly replicates 0, 1, 2; reserve access refused"
        )
    if bank.scenario_count != PILOT_GAMES_PER_REPLICATE * len(
        PILOT_REPLICATE_IDS
    ):
        raise PilotOrchestrationError("pilot bank must contain exactly 96,000 rows")
    return design


def _pilot_validation_opponents() -> tuple[FormalOpponentSpec, ...]:
    """Return the fixed, source-only checkpoint-selection opponents."""
    return tuple(
        FormalOpponentSpec(entry_name=name, candidate_name=name)
        for name in PILOT_VALIDATION_OPPONENT_IDS
    )


def _require_pilot_validation_bank(bank: ScenarioBank) -> None:
    """Require precisely the first ten registered validation-A scenarios."""
    if bank.sealed:
        raise PilotOrchestrationError("pilot validation bank must be non-sealed")
    if bank.logical_split != "validation-A" or bank.selection_kind != "iid":
        raise PilotOrchestrationError(
            "pilot validation requires an IID validation-A ScenarioV1 bank"
        )
    if (
        bank.scenario_count != PILOT_VALIDATION_SCENARIO_COUNT
        or len(bank.scenarios) != PILOT_VALIDATION_SCENARIO_COUNT
    ):
        raise PilotOrchestrationError(
            "pilot validation bank must contain exactly 10 materialized scenarios"
        )
    expected_seeds = tuple(
        range(
            TASK1_VALIDATION_A.start,
            TASK1_VALIDATION_A.start + PILOT_VALIDATION_SCENARIO_COUNT,
        )
    )
    actual_seeds = tuple(
        scenario.source_seed
        for scenario in sorted(bank.scenarios, key=lambda item: item.source_seed)
    )
    if actual_seeds != expected_seeds:
        raise PilotOrchestrationError(
            "pilot validation bank must be validation-A's first 10 source scenarios"
        )


def prepare_pilot(  # noqa: C901,PLR0912,PLR0915 - full admission sequence
    declaration: PilotDeclaration,
    *,
    runtime: ProductionRuntime | None = None,
) -> PreparedPilot:
    """Run every non-mutating admission gate and construct exactly six jobs."""
    require_production_runtime(declaration, runtime)
    initial_bc = declaration.repository / PILOT_INITIAL_BC_RELATIVE_PATH
    if (
        not initial_bc.is_file()
        or initial_bc.is_symlink()
        or initial_bc.resolve(strict=True) != initial_bc
        or sha256_file(initial_bc) != PILOT_INITIAL_BC_SHA256
    ):
        raise PilotOrchestrationError(
            "pilot BC initializer is missing, symlinked, or has the wrong SHA-256"
        )
    roll = require_task1_formal_seed_roll(
        load_seed_roll_artifact(declaration.seed_roll_path)
    )
    if (
        roll.plan.experiment_id != declaration.experiment_id
        or roll.plan.phase != "T1.4"
        or roll.plan.pilot_replicate_ids != PILOT_REPLICATE_IDS
    ):
        raise PilotOrchestrationError("seed roll does not match this T1.4 pilot")
    manifest = load_manifest(declaration.manifest_path)
    if manifest.get("status") != "running":
        raise PilotOrchestrationError(
            "pilot manifest must already be explicitly approved and running"
        )
    raw_manifest_declaration = manifest.get("declaration")
    if not isinstance(raw_manifest_declaration, Mapping):
        raise PilotOrchestrationError("manifest declaration is missing")
    if (
        raw_manifest_declaration.get("experiment_id") != declaration.experiment_id
        or raw_manifest_declaration.get("phase") != "T1.4"
    ):
        raise PilotOrchestrationError("manifest identity does not match the pilot")
    seed_plan = raw_manifest_declaration.get("seed_plan")
    if not isinstance(seed_plan, Mapping):
        raise PilotOrchestrationError("manifest seed_plan is missing")
    require_seed_roll_binding(roll, seed_plan.get("seed_roll"))
    public_bank = inspect_scenario_bank(declaration.scenario_bank_path)
    if public_bank.get("sealed") is not False:
        raise PilotOrchestrationError("pilot bank must be explicitly non-sealed")
    if public_bank.get("logical_split") != "train-schedule":
        raise PilotOrchestrationError("pilot refuses non-training scenario banks")
    bank = load_scenario_bank(declaration.scenario_bank_path)
    selection_design = _require_pilot_bank(bank)
    public_validation_bank = inspect_scenario_bank(
        declaration.validation_scenario_bank_path
    )
    if public_validation_bank.get("sealed") is not False:
        raise PilotOrchestrationError(
            "pilot validation bank must be explicitly non-sealed"
        )
    if public_validation_bank.get("logical_split") != "validation-A":
        raise PilotOrchestrationError("pilot refuses non-validation-A selector banks")
    validation_bank = load_scenario_bank(declaration.validation_scenario_bank_path)
    _require_pilot_validation_bank(validation_bank)
    validation_opponents = _pilot_validation_opponents()
    validation_contract = make_formal_validation_contract(
        validation_bank,
        validation_opponents,
        device_name="cuda",
    )
    updates, games_per_update = (
        declaration.treatments[0].config.updates,
        declaration.treatments[0].config.games_per_update,
    )
    schedule = make_seed_rolled_training_schedule(
        roll,
        bank.scenarios,
        active_replicate_ids=PILOT_REPLICATE_IDS,
        treatment_ids=PILOT_TREATMENT_IDS,
        updates=updates,
        games_per_update=games_per_update,
        selection_design=selection_design,
    )
    contracts = tuple(
        sorted(
            (
                make_formal_treatment_contract(
                    treatment.treatment_id,
                    initial_bc=treatment.initial_bc,
                    config=treatment.config,
                    opponent_pool=treatment.opponent_pool,
                )
                for treatment in declaration.treatments
            ),
            key=lambda contract: contract.treatment_id,
        )
    )
    outputs = _manifest_job_outputs(manifest, declaration)
    _require_unattempted_outputs(outputs)
    base_spec = FormalTrainingSpec(
        experiment_id=declaration.experiment_id,
        phase="T1.4",
        replicate_id=0,
        treatment_id="O",
        expected_treatments=PILOT_TREATMENT_IDS,
        schedule=schedule,
        scenario_bank_sha256=bank.payload_sha256,
        worker_count=PILOT_WORKER_COUNT,
        seed_roll_payload_sha256=roll.payload_sha256,
        randomization_root_sha256=roll.plan.randomization_root_sha256,
        treatment_contracts=contracts,
        job_outputs=outputs,
        replicate_stage="pilot",
        validation=validation_contract,
    )
    formal_binding = base_spec.manifest_binding()
    if raw_manifest_declaration.get("formal_training") != formal_binding:
        raise PilotOrchestrationError(
            "constructed pilot does not match the manifest formal_training binding"
        )
    treatment_by_id = {
        treatment.treatment_id: treatment for treatment in declaration.treatments
    }
    output_by_coordinate = {
        (output.replicate_id, output.treatment_id): Path(output.output_dir)
        for output in outputs
    }
    jobs = tuple(
        FormalPPOTrainingJob(
            initial_bc=treatment_by_id[treatment_id].initial_bc,
            output_dir=output_by_coordinate[(replicate_id, treatment_id)],
            source_manifest=declaration.manifest_path,
            config=treatment_by_id[treatment_id].config,
            formal_spec=replace(
                base_spec,
                replicate_id=replicate_id,
                treatment_id=treatment_id,
            ),
            opponent_pool=treatment_by_id[treatment_id].opponent_pool,
            scenario_bank_path=declaration.scenario_bank_path,
            seed_roll_path=declaration.seed_roll_path,
            validation_scenario_bank_path=(
                declaration.validation_scenario_bank_path
            ),
            validation_opponents=validation_opponents,
        )
        for replicate_id in PILOT_REPLICATE_IDS
        for treatment_id in PILOT_TREATMENT_IDS
    )
    if len(jobs) != PILOT_JOB_COUNT:
        raise PilotOrchestrationError("pilot job construction was incomplete")
    for job in jobs:
        # Reuse the existing non-mutating manifest/runtime/provenance gate now;
        # the formal runner repeats it immediately before one-shot reservation.
        ppo_selfplay_module._require_formal_training_manifest(  # noqa: SLF001
            str(job.source_manifest),
            job.formal_spec,
            roll,
        )
    return PreparedPilot(
        declaration=declaration,
        seed_roll=roll,
        bank=bank,
        validation_bank=validation_bank,
        jobs=jobs,
        formal_training_sha256=sha256_canonical_json(formal_binding),
    )


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise PilotOrchestrationError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _tree_size_bytes(path: Path) -> int:  # noqa: C901 - concurrent tree audit
    if not os.path.lexists(path):
        return 0
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            if current != path:
                # Atomic writers unlink or rename temporary descendants between
                # scandir() and lstat().  Their vanished bytes cannot make this
                # upper-bound gate unsafe, whereas disappearance of the declared
                # output root remains a hard failure.
                continue
            raise PilotOrchestrationError(
                f"cannot inspect formal output path {current}: {exc}"
            ) from exc
        except OSError as exc:
            raise PilotOrchestrationError(
                f"cannot inspect formal output path {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PilotOrchestrationError(
                f"formal output tree contains a symlink: {current}"
            )
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise PilotOrchestrationError(
                f"formal output tree contains a special file: {current}"
            )
        try:
            stack.extend(Path(entry.path) for entry in os.scandir(current))
        except FileNotFoundError as exc:
            if current != path:
                continue
            raise PilotOrchestrationError(
                f"cannot scan formal output directory {current}: {exc}"
            ) from exc
        except OSError as exc:
            raise PilotOrchestrationError(
                f"cannot scan formal output directory {current}: {exc}"
            ) from exc
    return total


def capture_resources(output_root: Path) -> ResourceSnapshot:
    """Measure logical output bytes and available bytes on its filesystem."""
    ancestor = _existing_ancestor(output_root)
    usage = shutil.disk_usage(ancestor)
    return ResourceSnapshot(
        output_bytes=_tree_size_bytes(output_root),
        free_bytes=usage.free,
    )


def require_resource_limits(snapshot: ResourceSnapshot) -> None:
    """Apply the fixed 32GiB/40GiB disk gates."""
    if snapshot.output_bytes > OUTPUT_LIMIT_BYTES:
        raise PilotOrchestrationError(
            f"formal output exceeded 32GiB: {snapshot.output_bytes} bytes"
        )
    if snapshot.free_bytes < MIN_FREE_BYTES:
        raise PilotOrchestrationError(
            f"filesystem free space fell below 40GiB: {snapshot.free_bytes} bytes"
        )


def _status_payload(  # noqa: PLR0913 - all optional audit fields are explicit
    declaration: PilotDeclaration,
    state: str,
    *,
    started_monotonic: float | None = None,
    resources: ResourceSnapshot | None = None,
    detail: str | None = None,
    child_pid: int | None = None,
    supervisor_pid: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": STATUS_SCHEMA,
        "declaration_path": str(declaration.path),
        "declaration_sha256": declaration.declaration_sha256,
        "experiment_id": declaration.experiment_id,
        "state": state,
        "tmux_session": declaration.tmux_session,
        "worker_count": PILOT_WORKER_COUNT,
        "job_count": PILOT_JOB_COUNT,
        "limits": {
            "wall_seconds": WALL_LIMIT_SECONDS,
            "max_output_bytes": OUTPUT_LIMIT_BYTES,
            "min_free_bytes": MIN_FREE_BYTES,
        },
        "child_pid": child_pid,
        "supervisor_pid": supervisor_pid,
        "detail": detail,
        "updated_unix_seconds": time.time(),
    }
    if started_monotonic is not None:
        payload["elapsed_seconds"] = max(0.0, time.monotonic() - started_monotonic)
    if resources is not None:
        payload["resources"] = asdict(resources)
    payload["status_sha256"] = sha256_canonical_json(payload)
    return payload


def _validate_status_payload(  # noqa: C901,PLR0912 - exact fail-closed schema
    declaration: PilotDeclaration,
    raw: object,
) -> dict[str, object]:
    """Validate the exact status-v2 identity and state-dependent fields."""
    if not isinstance(raw, Mapping):
        raise PilotOrchestrationError("orchestrator status root is invalid")
    payload = dict(raw)
    required = {
        "schema_version",
        "declaration_path",
        "declaration_sha256",
        "experiment_id",
        "state",
        "tmux_session",
        "worker_count",
        "job_count",
        "limits",
        "child_pid",
        "supervisor_pid",
        "detail",
        "updated_unix_seconds",
        "status_sha256",
    }
    allowed = required | {"elapsed_seconds", "resources"}
    if not required <= set(payload) or not set(payload) <= allowed:
        raise PilotOrchestrationError("orchestrator status keys are invalid")
    status_body = dict(payload)
    status_sha256 = status_body.pop("status_sha256")
    if (
        type(status_sha256) is not str
        or status_sha256 != sha256_canonical_json(status_body)
    ):
        raise PilotOrchestrationError("orchestrator status hash is invalid")
    state = payload["state"]
    if type(state) is not str or state not in {
        "launching",
        "running",
        "completed",
        "failed",
    }:
        raise PilotOrchestrationError("orchestrator status state is invalid")
    original_path = str(declaration.path)
    snapshot_path = str(declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME)
    expected_paths = (
        {original_path}
        if state == "launching"
        else {snapshot_path}
        if state in {"running", "completed"}
        else {original_path, snapshot_path}
    )
    expected_limits = {
        "wall_seconds": WALL_LIMIT_SECONDS,
        "max_output_bytes": OUTPUT_LIMIT_BYTES,
        "min_free_bytes": MIN_FREE_BYTES,
    }
    limits = payload["limits"]
    if (
        type(payload["schema_version"]) is not str
        or payload["schema_version"] != STATUS_SCHEMA
        or type(payload["declaration_path"]) is not str
        or payload["declaration_path"] not in expected_paths
        or type(payload["declaration_sha256"]) is not str
        or payload["declaration_sha256"] != declaration.declaration_sha256
        or type(payload["experiment_id"]) is not str
        or payload["experiment_id"] != declaration.experiment_id
        or type(payload["tmux_session"]) is not str
        or payload["tmux_session"] != declaration.tmux_session
        or type(payload["worker_count"]) is not int
        or payload["worker_count"] != PILOT_WORKER_COUNT
        or type(payload["job_count"]) is not int
        or payload["job_count"] != PILOT_JOB_COUNT
        or not isinstance(limits, Mapping)
        or set(limits) != set(expected_limits)
        or any(
            type(limits[field]) is not int
            or limits[field] != expected_limits[field]
            for field in expected_limits
        )
    ):
        raise PilotOrchestrationError("orchestrator status identity is invalid")
    updated = payload["updated_unix_seconds"]
    if (
        type(updated) not in {int, float}
        or not math.isfinite(updated)
        or updated <= 0
    ):
        raise PilotOrchestrationError("orchestrator status timestamp is invalid")
    for field in ("child_pid", "supervisor_pid"):
        value = payload[field]
        if value is not None and (type(value) is not int or value <= 1):
            raise PilotOrchestrationError(
                f"orchestrator status {field} is invalid"
            )
    elapsed = payload.get("elapsed_seconds")
    if elapsed is not None and (
        type(elapsed) not in {int, float}
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise PilotOrchestrationError("orchestrator status elapsed time is invalid")
    resources = payload.get("resources")
    if resources is not None and (
        not isinstance(resources, Mapping)
        or set(resources) != {"output_bytes", "free_bytes"}
        or any(
            type(resources[field]) is not int or resources[field] < 0
            for field in ("output_bytes", "free_bytes")
        )
    ):
        raise PilotOrchestrationError("orchestrator status resources are invalid")
    detail = payload["detail"]
    if state in {"launching", "running"} and detail is not None:
        raise PilotOrchestrationError("nonterminal orchestrator detail must be null")
    if state in {"completed", "failed"} and (
        type(detail) is not str or not detail
    ):
        raise PilotOrchestrationError("terminal orchestrator detail is invalid")
    if state == "launching" and (
        payload["child_pid"] is not None
        or payload["supervisor_pid"] is not None
        or "elapsed_seconds" in payload
        or resources is None
    ):
        raise PilotOrchestrationError("launching orchestrator status is invalid")
    if state in {"running", "completed"} and (
        type(payload["child_pid"]) is not int
        or type(payload["supervisor_pid"]) is not int
        or elapsed is None
        or resources is None
    ):
        raise PilotOrchestrationError(
            f"{state} orchestrator status lacks live process evidence"
        )
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PilotOrchestrationError(
            f"one-shot orchestration artifact already exists: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tmux_has_session(session: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise PilotOrchestrationError(
            f"cannot inspect tmux session {session!r}: {exc}"
        ) from exc
    return result.returncode == 0


def _tmux_pane_pid(session: str) -> int:
    """Return the sole supervisor pane PID for an exact tmux session."""
    try:
        result = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-s",
                "-t",
                f"={session}",
                "-F",
                "#{pane_pid}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PilotOrchestrationError(f"cannot inspect tmux supervisor PID: {exc}") from exc
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 1:
        raise PilotOrchestrationError(
            "tmux supervisor session does not contain exactly one pane"
        )
    try:
        pane_pid = int(lines[0])
    except ValueError as exc:
        raise PilotOrchestrationError("tmux supervisor pane PID is invalid") from exc
    if pane_pid <= 1:
        raise PilotOrchestrationError("tmux supervisor pane PID is invalid")
    return pane_pid


def _kill_owned_tmux_session(session: str) -> str | None:
    """Kill only the exact session whose successful creation this launch owns."""
    try:
        result = subprocess.run(
            ["tmux", "kill-session", "-t", f"={session}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"cannot kill failed tmux supervisor session: {exc}"
    if result.returncode == 0:
        return None
    try:
        still_alive = _tmux_has_session(session)
    except PilotOrchestrationError as exc:
        return f"cannot verify failed tmux supervisor cleanup: {exc}"
    if still_alive:
        return (
            "failed to kill owned tmux supervisor session: "
            f"{result.stderr.strip()}"
        )
    return None


def _process_parent_pid(pid: int) -> int:
    """Read one Linux process parent without invoking a mutable shell tool."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = raw.rsplit(")", 1)[1].split()
        parent_pid = int(suffix[1])
    except (OSError, UnicodeError, IndexError, ValueError) as exc:
        raise PilotOrchestrationError(
            f"cannot verify formal matrix process {pid}: {exc}"
        ) from exc
    if parent_pid <= 1:
        raise PilotOrchestrationError("formal matrix parent PID is invalid")
    return parent_pid


def _selection_artifact_names() -> tuple[str, ...]:
    return (
        "checkpoint-selection.json",
        *(f"validation-update-{update}.json" for update in FORMAL_VALIDATION_UPDATES),
    )


def _selection_checkpoint_names() -> tuple[str, ...]:
    """Return the nonzero-update checkpoints referenced by selector evidence."""
    return tuple(
        f"update-{update}.pth" for update in FORMAL_VALIDATION_UPDATES if update
    )


def _selection_checkpoint_path(job: FormalPPOTrainingJob, update: int) -> Path:
    return job.output_dir / ("initial.pth" if update == 0 else f"update-{update}.pth")


def _require_completed_selection(  # noqa: C901,PLR0912,PLR0915 - full selector proof
    job: FormalPPOTrainingJob,
    result: Mapping[str, Any],
) -> None:
    """Reject a nominally completed job without the full formal selector."""
    contract = job.formal_spec.validation
    if contract is None:
        raise PilotOrchestrationError("completed pilot job has no validation contract")
    selection_path = job.output_dir / "checkpoint-selection.json"
    try:
        raw = json.loads(
            selection_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotOrchestrationError(
            f"cannot read formal checkpoint selection: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise PilotOrchestrationError("formal checkpoint selection root is invalid")
    selection = dict(raw)
    claimed_sha256 = selection.pop("selection_evidence_sha256", None)
    if (
        type(claimed_sha256) is not str
        or claimed_sha256 != sha256_canonical_json(selection)
    ):
        raise PilotOrchestrationError("formal checkpoint selection hash is invalid")
    _exact_keys(
        selection,
        {
            "protocol",
            "status",
            "experiment_id",
            "phase",
            "replicate_id",
            "treatment_id",
            "validation_contract",
            "selection_rule",
            "evaluations",
            "selected_update",
            "selected_checkpoint_sha256",
            "best_path",
            "best_sha256",
        },
        label="formal checkpoint selection",
    )
    if (
        selection.get("status") != "completed"
        or selection.get("experiment_id") != job.formal_spec.experiment_id
        or selection.get("phase") != job.formal_spec.phase
        or selection.get("replicate_id") != job.formal_spec.replicate_id
        or selection.get("treatment_id") != job.formal_spec.treatment_id
        or selection.get("validation_contract") != contract.as_dict()
        or selection.get("selection_rule") != FORMAL_VALIDATION_SELECTION_RULE
    ):
        raise PilotOrchestrationError(
            "formal checkpoint selection identity or contract is invalid"
        )
    evaluations = selection.get("evaluations")
    if not isinstance(evaluations, list) or tuple(
        evaluation.get("update")
        for evaluation in evaluations
        if isinstance(evaluation, Mapping)
    ) != FORMAL_VALIDATION_UPDATES:
        raise PilotOrchestrationError(
            "formal checkpoint selection does not cover 0,50,...,2000"
        )
    if not all(
        isinstance(evaluation, Mapping)
        and set(evaluation)
        == {
            "update",
            "checkpoint_path",
            "checkpoint_sha256",
            "validation_evidence_sha256",
            "total_integer_wins",
            "scheduled_games",
        }
        and type(evaluation.get("total_integer_wins")) is int
        and 0
        <= cast(int, evaluation.get("total_integer_wins"))
        <= PILOT_VALIDATION_GAMES
        and evaluation.get("scheduled_games") == PILOT_VALIDATION_GAMES
        and type(evaluation.get("checkpoint_sha256")) is str
        and _SHA256_PATTERN.fullmatch(
            cast(str, evaluation.get("checkpoint_sha256"))
        )
        is not None
        and type(evaluation.get("validation_evidence_sha256")) is str
        and _SHA256_PATTERN.fullmatch(
            cast(str, evaluation.get("validation_evidence_sha256"))
        )
        is not None
        for evaluation in evaluations
    ):
        raise PilotOrchestrationError("formal validation evaluation record is invalid")
    if job.validation_scenario_bank_path is None or not job.validation_opponents:
        raise PilotOrchestrationError(
            "completed pilot job is missing its validation reconstruction inputs"
        )
    try:
        validation_bank = load_scenario_bank(job.validation_scenario_bank_path)
    except Exception as exc:
        raise PilotOrchestrationError(
            f"cannot load formal validation bank: {type(exc).__name__}: {exc}"
        ) from exc
    for evaluation in evaluations:
        update = cast(int, evaluation["update"])
        checkpoint_path = _selection_checkpoint_path(job, update)
        if (
            evaluation.get("checkpoint_path") != str(checkpoint_path)
            or not checkpoint_path.is_file()
            or checkpoint_path.is_symlink()
            or sha256_file(checkpoint_path) != evaluation["checkpoint_sha256"]
        ):
            raise PilotOrchestrationError(
                "formal validation index does not bind an extant checkpoint"
            )
        validation_path = job.output_dir / f"validation-update-{update}.json"
        try:
            raw_validation = json.loads(
                validation_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PilotOrchestrationError(
                f"cannot read formal validation evidence: {exc}"
            ) from exc
        if not isinstance(raw_validation, Mapping):
            raise PilotOrchestrationError("formal validation evidence root is invalid")
        try:
            validation = ppo_selfplay_module.validate_formal_validation_evidence(
                raw_validation,
                checkpoint_path,
                validation_bank,
                job.validation_opponents,
                job.formal_spec,
                job.source_manifest,
                update=update,
                device_name=job.config.device_name,
            )
        except Exception as exc:
            raise PilotOrchestrationError(
                "formal validation episode evidence failed reconstruction: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if (
            validation["evidence_sha256"]
            != evaluation["validation_evidence_sha256"]
            or validation.get("status") != "valid"
            or validation.get("update") != update
            or validation.get("checkpoint_sha256")
            != evaluation["checkpoint_sha256"]
            or validation.get("total_integer_wins")
            != evaluation["total_integer_wins"]
            or validation.get("scheduled_games") != PILOT_VALIDATION_GAMES
            or validation.get("scenario_bank_sha256")
            != contract.scenario_bank_sha256
        ):
            raise PilotOrchestrationError(
                "formal validation artifact does not match selection evidence"
            )
    best_wins = max(cast(int, evaluation["total_integer_wins"]) for evaluation in evaluations)
    expected_update = next(
        cast(int, evaluation["update"])
        for evaluation in evaluations
        if evaluation["total_integer_wins"] == best_wins
    )
    selected = next(
        evaluation
        for evaluation in evaluations
        if evaluation["update"] == expected_update
    )
    best_path = job.output_dir / "best.pth"
    if (
        selection.get("selected_update") != expected_update
        or result.get("best_update") != expected_update
        or selection.get("selected_checkpoint_sha256")
        != selected["checkpoint_sha256"]
        or selection.get("best_path") != str(best_path)
        or selection.get("best_sha256") != sha256_file(best_path)
        or selection.get("best_sha256") != selected["checkpoint_sha256"]
    ):
        raise PilotOrchestrationError(
            "formal checkpoint selection did not apply integer wins/earliest tie"
        )
    result_selector = result.get("formal_checkpoint_selection")
    if not isinstance(result_selector, Mapping) or (
        result_selector.get("path") != str(selection_path)
        or result_selector.get("sha256") != sha256_file(selection_path)
    ):
        raise PilotOrchestrationError(
            "completed result does not bind formal checkpoint selection"
        )


def _completion_evidence(  # noqa: C901 - full completion attestation
    prepared: PreparedPilot,
    results: Sequence[dict[str, Any]],
) -> dict[str, object]:
    if len(results) != PILOT_JOB_COUNT or any(
        result.get("status") != "completed" for result in results
    ):
        raise PilotOrchestrationError(
            "formal runner did not return six completed pilot results"
        )
    jobs: list[dict[str, object]] = []
    for job, result in zip(prepared.jobs, results, strict=True):
        _require_completed_selection(job, result)
        coordinate = (
            job.formal_spec.replicate_id,
            job.formal_spec.treatment_id,
        )
        result_path = job.output_dir / "result.json"
        status_path = job.output_dir / "status.json"
        try:
            disk_result = json.loads(
                result_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
            disk_status = json.loads(
                status_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PilotOrchestrationError(
                f"completed job {coordinate} result/status is unreadable: {exc}"
            ) from exc
        if (
            not isinstance(disk_result, Mapping)
            or sha256_canonical_json(disk_result)
            != sha256_canonical_json(result)
            or disk_result.get("status") != "completed"
            or disk_result.get("formal_protocol") != job.formal_spec.as_dict()
            or sha256_canonical_json(disk_result.get("config"))
            != sha256_canonical_json(asdict(job.config))
            or disk_result.get("source_bc") != str(job.initial_bc)
        ):
            raise PilotOrchestrationError(
                f"completed job {coordinate} on-disk result was relabelled"
            )
        if not isinstance(disk_status, Mapping) or set(disk_status) != {
            "status",
            "update",
            "updates",
            "best_update",
            "best_validation_score",
            "update_seconds",
            "elapsed_seconds",
            "error",
        }:
            raise PilotOrchestrationError(
                f"completed job {coordinate} status schema is invalid"
            )
        if (
            disk_status.get("status") != "completed"
            or disk_status.get("update") != PILOT_UPDATES
            or disk_status.get("updates") != PILOT_UPDATES
            or disk_status.get("best_update") != result.get("best_update")
            or disk_status.get("best_validation_score")
            != result.get("best_validation_score")
            or disk_status.get("error") is not None
        ):
            raise PilotOrchestrationError(
                f"completed job {coordinate} status is inconsistent"
            )
        try:
            ppo_selfplay_module.validate_formal_ppo_checkpoint_binding(
                job.output_dir / "final.pth",
                job.formal_spec,
                expected_update=PILOT_UPDATES,
            )
            final_state_sha256 = (
                ppo_selfplay_module.ppo_checkpoint_model_state_sha256(
                    job.output_dir / "final.pth"
                )
            )
            update_state_sha256 = (
                ppo_selfplay_module.ppo_checkpoint_model_state_sha256(
                    job.output_dir / f"update-{PILOT_UPDATES}.pth"
                )
            )
        except Exception as exc:
            raise PilotOrchestrationError(
                f"completed job {coordinate} final checkpoint is invalid: {exc}"
            ) from exc
        if final_state_sha256 != update_state_sha256:
            raise PilotOrchestrationError(
                f"completed job {coordinate} final/update-2000 model states differ"
            )
        files: dict[str, str] = {}
        try:
            ppo_selfplay_module._require_formal_output_reservation(job)  # noqa: SLF001
        except Exception as exc:
            raise PilotOrchestrationError(
                f"completed job {coordinate} lost its one-shot reservation: {exc}"
            ) from exc
        for name in (
            ppo_selfplay_module.FORMAL_OUTPUT_RESERVATION_NAME,
            "initial.pth",
            "best.pth",
            "final.pth",
            "result.json",
            "status.json",
            *_selection_artifact_names(),
            *_selection_checkpoint_names(),
        ):
            path = job.output_dir / name
            if not path.is_file():
                raise PilotOrchestrationError(
                    f"completed job {coordinate} is missing {name}"
                )
            files[name] = sha256_file(path)
        jobs.append(
            {
                "replicate_id": coordinate[0],
                "treatment_id": coordinate[1],
                "output_dir": str(job.output_dir),
                "files": files,
            }
        )
    body: dict[str, object] = {
        "schema_version": COMPLETION_SCHEMA,
        "declaration_path": str(prepared.declaration.path),
        "declaration_sha256": prepared.declaration.declaration_sha256,
        "experiment_id": prepared.declaration.experiment_id,
        "manifest_path": str(prepared.declaration.manifest_path),
        "seed_roll_payload_sha256": prepared.seed_roll.payload_sha256,
        "scenario_bank_payload_sha256": prepared.bank.payload_sha256,
        "validation_bank_payload_sha256": (
            prepared.validation_bank.payload_sha256
        ),
        "formal_training_sha256": prepared.formal_training_sha256,
        "jobs": jobs,
    }
    body["completion_sha256"] = sha256_canonical_json(body)
    return body


def _verify_completion_evidence(  # noqa: C901,PLR0912,PLR0915 - attestation gate
    declaration: PilotDeclaration,
) -> str:
    """Re-hash the exact six declared outputs before completing the manifest."""
    try:
        completion = json.loads(
            declaration.completion_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotOrchestrationError(f"cannot read completion evidence: {exc}") from exc
    if not isinstance(completion, Mapping):
        raise PilotOrchestrationError("completion evidence root is invalid")
    completion_body = dict(completion)
    completion_sha256 = completion_body.pop("completion_sha256", None)
    if (
        completion_body.get("schema_version") != COMPLETION_SCHEMA
        or completion_body.get("declaration_sha256")
        != declaration.declaration_sha256
        or type(completion_sha256) is not str
        or completion_sha256 != sha256_canonical_json(completion_body)
    ):
        raise PilotOrchestrationError("completion evidence hash or identity is invalid")
    _exact_keys(
        completion_body,
        {
            "schema_version",
            "declaration_path",
            "declaration_sha256",
            "experiment_id",
            "manifest_path",
            "seed_roll_payload_sha256",
            "scenario_bank_payload_sha256",
            "validation_bank_payload_sha256",
            "formal_training_sha256",
            "jobs",
        },
        label="pilot completion evidence",
    )
    if completion_body.get("declaration_path") != str(declaration.path):
        raise PilotOrchestrationError(
            "completion evidence declaration path is invalid"
        )
    manifest = load_manifest(declaration.manifest_path)
    manifest_declaration = manifest.get("declaration")
    if not isinstance(manifest_declaration, Mapping):
        raise PilotOrchestrationError("manifest declaration is missing")
    formal_binding = manifest_declaration.get("formal_training")
    seed_plan = manifest_declaration.get("seed_plan")
    if not isinstance(formal_binding, Mapping) or not isinstance(seed_plan, Mapping):
        raise PilotOrchestrationError(
            "manifest formal_training or seed_plan binding is missing"
        )
    validation_binding = formal_binding.get("validation")
    seed_roll_binding = seed_plan.get("seed_roll")
    if not isinstance(validation_binding, Mapping) or not isinstance(
        seed_roll_binding, Mapping
    ):
        raise PilotOrchestrationError(
            "manifest validation or seed-roll binding is missing"
        )
    expected_metadata = {
        "experiment_id": declaration.experiment_id,
        "manifest_path": str(declaration.manifest_path),
        "seed_roll_payload_sha256": seed_roll_binding.get("payload_sha256"),
        "scenario_bank_payload_sha256": formal_binding.get(
            "scenario_bank_sha256"
        ),
        "validation_bank_payload_sha256": validation_binding.get(
            "scenario_bank_sha256"
        ),
        "formal_training_sha256": sha256_canonical_json(formal_binding),
    }
    if any(completion_body.get(key) != value for key, value in expected_metadata.items()):
        raise PilotOrchestrationError(
            "completion evidence does not match manifest bank/training bindings"
        )
    outputs = _manifest_job_outputs(manifest, declaration)
    raw_jobs = completion_body.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != PILOT_JOB_COUNT:
        raise PilotOrchestrationError("completion evidence must contain six jobs")
    required_files = {
        ppo_selfplay_module.FORMAL_OUTPUT_RESERVATION_NAME,
        "initial.pth",
        "best.pth",
        "final.pth",
        "result.json",
        "status.json",
        *_selection_artifact_names(),
        *_selection_checkpoint_names(),
    }
    treatment_contracts_raw = formal_binding.get("treatment_contracts")
    if not isinstance(treatment_contracts_raw, list):
        raise PilotOrchestrationError(
            "manifest formal training treatment contracts are invalid"
        )
    treatment_contracts = {
        contract.get("treatment_id"): contract
        for contract in treatment_contracts_raw
        if isinstance(contract, Mapping)
    }
    if set(treatment_contracts) != set(PILOT_TREATMENT_IDS):
        raise PilotOrchestrationError(
            "manifest formal training treatment contracts are incomplete"
        )
    for output, raw_job in zip(outputs, raw_jobs, strict=True):
        if not isinstance(raw_job, Mapping):
            raise PilotOrchestrationError("completion job record is invalid")
        _exact_keys(
            raw_job,
            {"replicate_id", "treatment_id", "output_dir", "files"},
            label="completion job record",
        )
        if (
            raw_job.get("replicate_id") != output.replicate_id
            or raw_job.get("treatment_id") != output.treatment_id
            or raw_job.get("output_dir") != output.output_dir
        ):
            raise PilotOrchestrationError(
                "completion job matrix differs from manifest outputs"
            )
        raw_files = raw_job.get("files")
        if not isinstance(raw_files, Mapping) or set(raw_files) != required_files:
            raise PilotOrchestrationError("completion job file hashes are incomplete")
        for name in required_files:
            expected_hash = raw_files[name]
            file_path = Path(output.output_dir) / name
            if (
                type(expected_hash) is not str
                or _SHA256_PATTERN.fullmatch(expected_hash) is None
                or not file_path.is_file()
                or file_path.is_symlink()
                or sha256_file(file_path) != expected_hash
            ):
                raise PilotOrchestrationError(
                    f"completion output hash mismatch: {file_path}"
                )
        marker_path = (
            Path(output.output_dir)
            / ppo_selfplay_module.FORMAL_OUTPUT_RESERVATION_NAME
        )
        try:
            marker_raw = json.loads(
                marker_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PilotOrchestrationError(
                f"cannot read formal output reservation: {exc}"
            ) from exc
        treatment_contract = treatment_contracts[output.treatment_id]
        expected_marker: dict[str, object] = {
            "protocol": "formal-ppo-output-reservation-v1",
            "manifest_path": str(declaration.manifest_path.resolve()),
            "manifest_declaration_sha256": manifest.get("declaration_sha256"),
            "manifest_provenance_sha256": manifest.get("provenance_sha256"),
            "formal_training_sha256": sha256_canonical_json(formal_binding),
            "replicate_id": output.replicate_id,
            "treatment_id": output.treatment_id,
            "treatment_contract": treatment_contract,
            "output": output.as_dict(),
            "scenario_bank_sha256": formal_binding.get("scenario_bank_sha256"),
            "seed_roll_payload_sha256": seed_roll_binding.get("payload_sha256"),
        }
        expected_marker["reservation_sha256"] = sha256_canonical_json(
            expected_marker
        )
        if not isinstance(marker_raw, Mapping) or dict(marker_raw) != expected_marker:
            raise PilotOrchestrationError(
                "formal output reservation does not match the manifest-bound job"
            )
    return completion_sha256


def _require_supervisor_lease(supervisor_pid: int, supervisor_fd: int) -> None:
    """Require an inherited pipe from this process's live direct parent."""
    if supervisor_pid <= 1 or os.getppid() != supervisor_pid:
        raise PilotOrchestrationError(
            "formal matrix must be a direct child of its tmux supervisor"
        )
    try:
        mode = os.fstat(supervisor_fd).st_mode
    except OSError as exc:
        raise PilotOrchestrationError(
            f"formal matrix supervisor lease is unavailable: {exc}"
        ) from exc
    if not stat.S_ISFIFO(mode):
        raise PilotOrchestrationError(
            "formal matrix supervisor lease must be an inherited pipe"
        )


def _abort_matrix_from_watchdog(
    declaration: PilotDeclaration,
    detail: str,
) -> None:
    """Persist a child-side watchdog failure and terminate its process group."""
    try:
        _write_json_atomic(
            declaration.control_dir / MATRIX_FAILURE_FILE_NAME,
            {
                "schema_version": STATUS_SCHEMA,
                "declaration_sha256": declaration.declaration_sha256,
                "state": "failed",
                "detail": detail,
            },
        )
    finally:
        os.killpg(os.getpgrp(), signal.SIGTERM)


def _matrix_watchdog(
    declaration: PilotDeclaration,
    supervisor_pid: int,
    supervisor_fd: int,
    stop: threading.Event,
    started: float,
) -> None:
    """Enforce a parent lease and duplicate wall/disk gates inside the child."""
    while not stop.is_set():
        try:
            readable, _, _ = select.select(
                [supervisor_fd], [], [], WATCHDOG_POLL_SECONDS
            )
            if stop.is_set():
                return
            if os.getppid() != supervisor_pid:
                _abort_matrix_from_watchdog(
                    declaration, "tmux supervisor parent identity changed"
                )
                return
            if readable and os.read(supervisor_fd, 1) == b"":
                _abort_matrix_from_watchdog(
                    declaration, "tmux supervisor lease closed"
                )
                return
            if time.monotonic() - started > WALL_LIMIT_SECONDS:
                _abort_matrix_from_watchdog(
                    declaration, "18-hour child wall-clock limit exceeded"
                )
                return
            require_resource_limits(capture_resources(declaration.output_root))
        except BaseException as exc:
            _abort_matrix_from_watchdog(
                declaration,
                f"matrix watchdog failed with {type(exc).__name__}: {exc}",
            )
            return


def run_matrix(
    declaration_path: Path,
    expected_sha256: str,
    *,
    supervisor_pid: int,
    supervisor_fd: int,
) -> None:
    """Internal supervised child: reserve, train, and attest outputs."""
    _require_supervisor_lease(supervisor_pid, supervisor_fd)
    declaration = load_pilot_declaration(declaration_path)
    if declaration.declaration_sha256 != expected_sha256:
        raise PilotOrchestrationError("declaration changed after tmux launch")
    started = time.monotonic()
    stop = threading.Event()
    watchdog = threading.Thread(
        target=_matrix_watchdog,
        args=(declaration, supervisor_pid, supervisor_fd, stop, started),
        name="task1-matrix-watchdog",
        daemon=True,
    )
    watchdog.start()
    try:
        prepared = prepare_pilot(declaration)
        resources = capture_resources(declaration.output_root)
        require_resource_limits(resources)
        results = run_formal_ppo_training_jobs(
            prepared.jobs,
            worker_count=PILOT_WORKER_COUNT,
        )
        evidence = _completion_evidence(prepared, results)
        _write_json_exclusive(declaration.completion_path, evidence)
    finally:
        stop.set()
        watchdog.join(timeout=WATCHDOG_POLL_SECONDS + 1.0)
        os.close(supervisor_fd)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _failed_job_status(declaration: PilotDeclaration) -> str | None:
    for status_path in sorted(declaration.output_root.glob("*/status.json")):
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(raw, Mapping) and raw.get("status") == "failed":
            return str(status_path)
    return None


def _raise_on_supervisor_signal(signum: int, _frame: FrameType | None) -> None:
    """Turn tmux/operator termination into an auditable fail-closed path."""
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = str(signum)
    raise PilotOrchestrationError(f"supervisor received {signal_name}")


def _block_running_manifest(declaration: PilotDeclaration, detail: str) -> None:
    try:
        manifest = load_manifest(declaration.manifest_path)
        if manifest.get("status") == "running":
            transition_manifest(
                declaration.manifest_path,
                "blocked",
                actor="task1-pilot-orchestrator",
                note=detail[:500],
            )
    except Exception as exc:  # preserve the primary failure in status
        raise PilotOrchestrationError(
            f"{detail}; additionally failed to block the running manifest: {exc}"
        ) from exc


def _matrix_failure_detail(declaration: PilotDeclaration) -> str | None:
    """Read a child-side watchdog reason when one was durably recorded."""
    path = declaration.control_dir / MATRIX_FAILURE_FILE_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return f"matrix watchdog failure marker is unreadable: {path}"
    if (
        not isinstance(raw, Mapping)
        or raw.get("declaration_sha256") != declaration.declaration_sha256
        or type(raw.get("detail")) is not str
    ):
        return f"matrix watchdog failure marker is invalid: {path}"
    return cast(str, raw["detail"])


def supervise(  # noqa: C901,PLR0912,PLR0913,PLR0915 - lifecycle is explicit
    declaration_path: Path,
    expected_sha256: str,
    *,
    expected_manifest_path: Path,
    expected_control_dir: Path,
    expected_output_root: Path,
    expected_experiment_id: str,
    expected_tmux_session: str,
) -> None:
    """Internal tmux parent enforcing wall/disk/process fail-closed gates."""
    source = _canonical_path(str(declaration_path), label="launched declaration path")
    manifest_path = _canonical_path(
        str(expected_manifest_path), label="expected manifest path"
    )
    control_dir = _canonical_path(
        str(expected_control_dir), label="expected control directory"
    )
    output_root = _canonical_path(
        str(expected_output_root), label="expected output root"
    )
    if type(expected_experiment_id) is not str or not expected_experiment_id.strip():
        raise PilotOrchestrationError("expected experiment ID is invalid")
    if (
        type(expected_tmux_session) is not str
        or _SESSION_PATTERN.fullmatch(expected_tmux_session) is None
    ):
        raise PilotOrchestrationError("expected tmux session is invalid")
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise PilotOrchestrationError("expected declaration SHA-256 is invalid")
    # This minimal identity is constructed entirely from launch-bound argv, so
    # even a missing or malformed snapshot can still close the original
    # manifest and status lifecycle.
    lifecycle_declaration = PilotDeclaration(
        path=source,
        declaration_sha256=expected_sha256,
        experiment_id=expected_experiment_id,
        repository=output_root.parent,
        python_executable=source,
        manifest_path=manifest_path,
        seed_roll_path=source,
        scenario_bank_path=source,
        validation_scenario_bank_path=source,
        output_root=output_root,
        control_dir=control_dir,
        tmux_session=expected_tmux_session,
        treatments=(),
    )
    declaration: PilotDeclaration | None = None
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    lease_read_fd: int | None = None
    lease_write_fd: int | None = None
    watched_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous_handlers = {
        watched: signal.getsignal(watched) for watched in watched_signals
    }
    for watched in watched_signals:
        signal.signal(watched, _raise_on_supervisor_signal)
    try:
        declaration = load_pilot_declaration(source)
        if declaration.declaration_sha256 != expected_sha256 or (
            declaration.manifest_path != manifest_path
            or declaration.control_dir != control_dir
            or declaration.output_root != output_root
            or declaration.experiment_id != expected_experiment_id
            or declaration.tmux_session != expected_tmux_session
        ):
            raise PilotOrchestrationError("declaration changed after tmux launch")
        require_production_runtime(declaration)
        initial_resources = capture_resources(declaration.output_root)
        require_resource_limits(initial_resources)
        lease_read_fd, lease_write_fd = os.pipe()
        command = [
            str(declaration.python_executable),
            "-m",
            __name__,
            "_run-matrix",
            str(declaration.path),
            "--expected-sha256",
            expected_sha256,
            "--supervisor-pid",
            str(os.getpid()),
            "--supervisor-fd",
            str(lease_read_fd),
        ]
        process = subprocess.Popen(
            command,
            start_new_session=True,
            pass_fds=(lease_read_fd,),
        )
        os.close(lease_read_fd)
        lease_read_fd = None
        _write_json_atomic(
            declaration.status_path,
            _status_payload(
                declaration,
                "running",
                started_monotonic=started,
                resources=initial_resources,
                child_pid=process.pid,
                supervisor_pid=os.getpid(),
            ),
        )
        last_status = started
        failure: str | None = None
        while True:
            return_code = process.poll()
            resources = capture_resources(declaration.output_root)
            elapsed = time.monotonic() - started
            if elapsed > WALL_LIMIT_SECONDS:
                failure = "18-hour wall-clock limit exceeded"
                break
            try:
                require_resource_limits(resources)
            except PilotOrchestrationError as exc:
                failure = str(exc)
                break
            failed_status = _failed_job_status(declaration)
            if failed_status is not None:
                failure = f"formal worker reported failure in {failed_status}"
                break
            if return_code is not None:
                if return_code != 0:
                    failure = _matrix_failure_detail(declaration) or (
                        f"formal matrix process exited {return_code}"
                    )
                break
            if time.monotonic() - last_status >= STATUS_REFRESH_SECONDS:
                _write_json_atomic(
                    declaration.status_path,
                    _status_payload(
                        declaration,
                        "running",
                        started_monotonic=started,
                        resources=resources,
                        child_pid=process.pid,
                        supervisor_pid=os.getpid(),
                    ),
                )
                last_status = time.monotonic()
            time.sleep(WATCHDOG_POLL_SECONDS)
        if failure is not None:
            raise PilotOrchestrationError(failure)
        if not declaration.completion_path.is_file():
            raise PilotOrchestrationError(
                "matrix exited zero without immutable completion evidence"
            )
        completion_sha256 = _verify_completion_evidence(declaration)
        final_resources = capture_resources(declaration.output_root)
        require_resource_limits(final_resources)
        _write_json_atomic(
            declaration.status_path,
            _status_payload(
                declaration,
                "completed",
                started_monotonic=started,
                resources=final_resources,
                detail=f"completion_sha256={completion_sha256}",
                child_pid=process.pid,
                supervisor_pid=os.getpid(),
            ),
        )
        # This is deliberately the last fallible mutation: all evidence,
        # resource checks, and the terminal status are durable before the
        # scientific manifest can become completed.
        transition_manifest(
            declaration.manifest_path,
            "completed",
            actor="task1-pilot-orchestrator",
            note=f"six-job pilot completed; completion_sha256={completion_sha256}",
        )
    except BaseException as exc:
        failure = (
            str(exc)
            if isinstance(exc, PilotOrchestrationError)
            else f"supervisor failed with {type(exc).__name__}: {exc}"
        )
        if process is not None:
            _terminate_process_group(process)
        try:
            _block_running_manifest(lifecycle_declaration, failure)
        except PilotOrchestrationError as block_exc:
            failure = str(block_exc)
        try:
            final_resources = capture_resources(lifecycle_declaration.output_root)
        except Exception as resource_exc:
            final_resources = None
            failure = (
                f"{failure}; additionally failed to capture final resources: "
                f"{resource_exc}"
            )
        _write_json_atomic(
            lifecycle_declaration.status_path,
            _status_payload(
                lifecycle_declaration,
                "failed",
                started_monotonic=started,
                resources=final_resources,
                detail=failure,
                child_pid=process.pid if process is not None else None,
                supervisor_pid=os.getpid(),
            ),
        )
        raise PilotOrchestrationError(failure) from exc
    finally:
        for watched, handler in previous_handlers.items():
            signal.signal(watched, handler)
        for descriptor in (lease_read_fd, lease_write_fd):
            if descriptor is not None:
                os.close(descriptor)


def preflight(declaration_path: Path) -> dict[str, object]:
    """Run read-only production admission checks and return an audit summary."""
    declaration = load_pilot_declaration(declaration_path)
    prepared = prepare_pilot(declaration)
    resources = capture_resources(declaration.output_root)
    require_resource_limits(resources)
    return {
        "schema_version": ORCHESTRATION_SCHEMA,
        "status": "ready",
        "declaration_sha256": declaration.declaration_sha256,
        "experiment_id": declaration.experiment_id,
        "seed_roll_payload_sha256": prepared.seed_roll.payload_sha256,
        "scenario_bank_payload_sha256": prepared.bank.payload_sha256,
        "validation_bank_payload_sha256": (
            prepared.validation_bank.payload_sha256
        ),
        "formal_training_sha256": prepared.formal_training_sha256,
        "job_coordinates": [
            [job.formal_spec.replicate_id, job.formal_spec.treatment_id]
            for job in prepared.jobs
        ],
        "worker_count": PILOT_WORKER_COUNT,
        "resources": asdict(resources),
        "mutations": [],
    }


def _reserve_supervisor_log(declaration: PilotDeclaration) -> Path:
    """Create the one-shot regular file that captures pre-handshake failures."""
    path = declaration.control_dir / SUPERVISOR_LOG_FILE_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PilotOrchestrationError(
            f"one-shot supervisor log already exists: {path}"
        ) from exc
    os.close(descriptor)
    _fsync_directory(path.parent)
    return path


def _record_launch_handshake_failure(
    declaration: PilotDeclaration,
    detail: str,
) -> None:
    """Close the lifecycle when tmux never establishes its supervisor."""
    try:
        _block_running_manifest(declaration, detail)
    except PilotOrchestrationError as exc:
        detail = str(exc)
    try:
        resources = capture_resources(declaration.output_root)
    except Exception as exc:
        resources = None
        detail = f"{detail}; additionally failed to capture resources: {exc}"
    _write_json_atomic(
        declaration.status_path,
        _status_payload(
            declaration,
            "failed",
            resources=resources,
            detail=detail,
        ),
    )


def _await_supervisor_handshake(  # noqa: C901,PLR0912,PLR0915 - fail closed
    declaration: PilotDeclaration,
) -> None:
    """Require durable supervisor liveness instead of trusting tmux creation."""
    deadline = time.monotonic() + LAUNCH_HANDSHAKE_SECONDS
    while True:
        try:
            snapshot = status(declaration.path)
        except PilotOrchestrationError as exc:
            detail = f"cannot verify the tmux supervisor handshake: {exc}"
            raise PilotOrchestrationError(detail) from exc
        persisted = snapshot["persisted"]
        persisted_mapping = (
            cast(Mapping[str, object], persisted)
            if isinstance(persisted, Mapping)
            else {}
        )
        state = persisted_mapping.get("state")
        if state == "completed":
            try:
                launched = load_pilot_declaration(
                    declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME
                )
                if (
                    launched.declaration_sha256
                    != declaration.declaration_sha256
                ):
                    raise PilotOrchestrationError(
                        "launch declaration changed before completed handshake"
                    )
                manifest = load_manifest(launched.manifest_path)
            except Exception as exc:
                raise PilotOrchestrationError(
                    f"cannot verify completed launch handshake: {exc}"
                ) from exc
            manifest_state = manifest.get("status")
            if manifest_state == "completed":
                completion_sha256 = _verify_completion_evidence(launched)
                if persisted_mapping.get("detail") != (
                    f"completion_sha256={completion_sha256}"
                ):
                    raise PilotOrchestrationError(
                        "completed launch handshake does not match completion evidence"
                    )
                return
            if manifest_state != "running":
                raise PilotOrchestrationError(
                    "completed launch handshake has a non-running manifest"
                )
        if state == "failed":
            failed_detail = persisted_mapping.get("detail")
            detail = (
                "tmux supervisor failed during launch handshake: "
                f"{failed_detail}"
            )
            raise PilotOrchestrationError(detail)
        tmux_alive = snapshot["tmux_alive"] is True
        if state == "running" and tmux_alive:
            supervisor_pid = cast(int, persisted_mapping["supervisor_pid"])
            child_pid = cast(int, persisted_mapping["child_pid"])
            if snapshot.get("tmux_pane_pid") != supervisor_pid:
                raise PilotOrchestrationError(
                    "tmux pane PID does not match the durable supervisor identity"
                )
            if _process_parent_pid(child_pid) != supervisor_pid:
                raise PilotOrchestrationError(
                    "formal matrix is not a direct child of the tmux supervisor"
                )
            try:
                process_group = os.getpgid(child_pid)
            except OSError as exc:
                raise PilotOrchestrationError(
                    f"cannot inspect formal matrix process group: {exc}"
                ) from exc
            if process_group != child_pid:
                raise PilotOrchestrationError(
                    "formal matrix is not its own supervised process group"
                )
            return
        if not tmux_alive:
            detail = "tmux supervisor exited before the running handshake"
            raise PilotOrchestrationError(detail)
        if time.monotonic() >= deadline:
            detail = (
                "tmux supervisor did not reach the running state within "
                f"{LAUNCH_HANDSHAKE_SECONDS:g} seconds"
            )
            raise PilotOrchestrationError(detail)
        time.sleep(LAUNCH_HANDSHAKE_POLL_SECONDS)


def launch(declaration_path: Path) -> None:
    """Preflight and launch exactly one detached tmux supervisor."""
    declaration = load_pilot_declaration(declaration_path)
    if declaration.status_path.exists() or declaration.completion_path.exists():
        raise PilotOrchestrationError(
            "this orchestration declaration was already launched"
        )
    tmux_started = False
    try:
        summary = preflight(declaration.path)
        if _tmux_has_session(declaration.tmux_session):
            raise PilotOrchestrationError(
                f"tmux session {declaration.tmux_session!r} already exists"
            )
        if shutil.which("tmux") is None:
            raise PilotOrchestrationError("tmux executable is unavailable")
        # Every mutation below is inside this fail-closed region.  Pin the
        # exact preflighted bytes before tmux starts so replacing the
        # operator-facing declaration cannot redirect lifecycle handling.
        raw, current_sha256 = _read_regular_json(declaration.path)
        if current_sha256 != declaration.declaration_sha256:
            raise PilotOrchestrationError(
                "declaration changed during launch preflight"
            )
        supervisor_log = _reserve_supervisor_log(declaration)
        launched_declaration = write_pilot_declaration(
            declaration.control_dir / LAUNCH_DECLARATION_FILE_NAME,
            raw,
        )
        _write_json_exclusive(
            declaration.status_path,
            _status_payload(
                declaration,
                "launching",
                resources=ResourceSnapshot(
                    **cast(dict[str, int], summary["resources"])
                ),
            ),
        )
        child = [
            str(declaration.python_executable),
            "-m",
            __name__,
            "_supervise",
            str(launched_declaration.path),
            "--expected-sha256",
            declaration.declaration_sha256,
            "--expected-manifest-path",
            str(declaration.manifest_path),
            "--expected-control-dir",
            str(declaration.control_dir),
            "--expected-output-root",
            str(declaration.output_root),
            "--expected-experiment-id",
            declaration.experiment_id,
            "--expected-tmux-session",
            declaration.tmux_session,
        ]
        shell_command = (
            "exec env PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            + shlex.join(child)
            + f" >> {shlex.quote(str(supervisor_log))} 2>&1"
        )
        result = subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                declaration.tmux_session,
                shell_command,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PilotOrchestrationError(
                f"tmux launch failed: {result.stderr.strip()}"
            )
        tmux_started = True
        _await_supervisor_handshake(declaration)
    except BaseException as exc:
        detail = (
            str(exc)
            if isinstance(exc, PilotOrchestrationError)
            else f"pilot launch failed with {type(exc).__name__}: {exc}"
        )
        if tmux_started:
            cleanup_failure = _kill_owned_tmux_session(
                declaration.tmux_session
            )
            if cleanup_failure is not None:
                detail = f"{detail}; {cleanup_failure}"
        try:
            _record_launch_handshake_failure(declaration, detail)
        except BaseException as close_exc:
            detail = (
                f"{detail}; additionally failed to close launch lifecycle: "
                f"{type(close_exc).__name__}: {close_exc}"
            )
        raise PilotOrchestrationError(detail) from exc


def status(declaration_path: Path) -> dict[str, object]:
    """Return persisted supervisor state plus current tmux liveness."""
    declaration = load_pilot_declaration(declaration_path)
    persisted: object = None
    if declaration.status_path.is_file():
        try:
            persisted = json.loads(
                declaration.status_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PilotOrchestrationError(f"cannot read orchestrator status: {exc}") from exc
        persisted = _validate_status_payload(declaration, persisted)
    tmux_alive = _tmux_has_session(declaration.tmux_session)
    tmux_pane_pid = (
        _tmux_pane_pid(declaration.tmux_session) if tmux_alive else None
    )
    return {
        "declaration_sha256": declaration.declaration_sha256,
        "tmux_session": declaration.tmux_session,
        "tmux_alive": tmux_alive,
        "tmux_pane_pid": tmux_pane_pid,
        "persisted": persisted,
    }


def attach(declaration_path: Path) -> int:
    """Attach the operator terminal to the declared tmux session."""
    declaration = load_pilot_declaration(declaration_path)
    if not _tmux_has_session(declaration.tmux_session):
        raise PilotOrchestrationError(
            f"tmux session {declaration.tmux_session!r} is not running"
        )
    return subprocess.run(
        ["tmux", "attach-session", "-t", f"={declaration.tmux_session}"],
        check=False,
    ).returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task1-pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "launch", "status", "attach"):
        command = subparsers.add_parser(name)
        command.add_argument("declaration", type=Path)
    for name in ("_supervise", "_run-matrix"):
        command = subparsers.add_parser(name, help=argparse.SUPPRESS)
        command.add_argument("declaration", type=Path)
        command.add_argument("--expected-sha256", required=True)
        if name == "_supervise":
            command.add_argument("--expected-manifest-path", required=True, type=Path)
            command.add_argument("--expected-control-dir", required=True, type=Path)
            command.add_argument("--expected-output-root", required=True, type=Path)
            command.add_argument("--expected-experiment-id", required=True)
            command.add_argument("--expected-tmux-session", required=True)
        else:
            command.add_argument("--supervisor-pid", required=True, type=int)
            command.add_argument("--supervisor-fd", required=True, type=int)
    return parser


def main() -> None:
    """Dispatch operator and internal tmux commands."""
    args = _parser().parse_args()
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(args.declaration), indent=2, sort_keys=True))
        elif args.command == "launch":
            launch(args.declaration)
            declaration = load_pilot_declaration(args.declaration)
            print(
                f"launched tmux session {declaration.tmux_session}; "
                f"attach with: task1-pilot attach {declaration.path}"
            )
        elif args.command == "status":
            print(json.dumps(status(args.declaration), indent=2, sort_keys=True))
        elif args.command == "attach":
            raise SystemExit(attach(args.declaration))
        elif args.command == "_supervise":
            if _SHA256_PATTERN.fullmatch(args.expected_sha256) is None:
                raise PilotOrchestrationError("expected declaration SHA-256 is invalid")
            supervise(
                args.declaration,
                args.expected_sha256,
                expected_manifest_path=args.expected_manifest_path,
                expected_control_dir=args.expected_control_dir,
                expected_output_root=args.expected_output_root,
                expected_experiment_id=args.expected_experiment_id,
                expected_tmux_session=args.expected_tmux_session,
            )
        elif args.command == "_run-matrix":
            if _SHA256_PATTERN.fullmatch(args.expected_sha256) is None:
                raise PilotOrchestrationError("expected declaration SHA-256 is invalid")
            run_matrix(
                args.declaration,
                args.expected_sha256,
                supervisor_pid=args.supervisor_pid,
                supervisor_fd=args.supervisor_fd,
            )
    except PilotOrchestrationError as exc:
        parser = argparse.ArgumentParser(prog="task1-pilot")
        parser.exit(2, f"task1-pilot: error: {exc}\n")


if __name__ == "__main__":
    main()
