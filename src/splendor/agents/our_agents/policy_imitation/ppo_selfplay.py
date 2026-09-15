"""PPO self-play initialized from BC with a declared fixed opponent pool."""

import json
import os
import random
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions, nn, optim

from splendor.agents.our_agents.dqn.constants import HIDDEN_DIMS, HUGE_NEG
from splendor.agents.our_agents.dqn.features import extract_observation, observation_dim
from splendor.agents.our_agents.policy_imitation.shaping import (
    EventRewardShaper,
    PotentialRewardShaper,
)
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent

from .bc_network import ACTION_DIM, BehaviorCloningNetwork, FixedNormalizer
from .bc_training import DeviceName, load_bc_checkpoint, resolve_device
from .policies import CandidateSpec, build_builtin_candidate
from .protocol import (
    FormalGameRng,
    FormalTrainingSpec,
    FormalTreatmentContract,
    RngKey,
    RuntimeSnapshot,
    SeedLineage,
    capture_code_revision,
    capture_dependency_provenance,
    configure_formal_torch_determinism,
    derive_seed,
    inverse_cdf_index,
    isolated_seed,
    require_formal_spawn_context,
    require_reproducible_code,
    run_formal_spawn_jobs,
    seed_everything,
    sha256_canonical_json,
    sha256_file,
)
from .runner import TeacherDecisionError, select_action
from .scenario import ScenarioV1, rule_from_scenario
from .scenario_bank import (
    ScenarioBank,
    audit_seed_roll_scenario_bank,
    inspect_scenario_bank,
    load_scenario_bank_subset,
    scenario_lookup,
)
from .seed_roll import (
    SeedRollArtifact,
    SeedRollSelectionDesign,
    load_confirmatory_activation_artifact,
    load_seed_roll_artifact,
    require_seed_roll_binding,
    require_task1_formal_seed_roll,
    validate_seed_rolled_training_schedule,
)

ValueMode = Literal["return", "outcome"]
InitializationMode = Literal["bc", "scratch"]
ADVANTAGE_STD_EPSILON = 1e-8
EXPLAINED_VARIANCE_EPSILON = 1e-12
FORMAL_OUTPUT_RESERVATION_NAME = "formal-output-reservation.json"


def _validate_value_mode(value_mode: str) -> ValueMode:
    """Validate and narrow the two checkpoint-compatible critic semantics."""
    if value_mode not in {"return", "outcome"}:
        raise ValueError("value_mode must be 'return' or 'outcome'")
    return cast(ValueMode, value_mode)


def _validate_initialization(initialization: str) -> InitializationMode:
    """Validate the policy initialization mode."""
    if initialization not in {"bc", "scratch"}:
        raise ValueError("initialization must be 'bc' or 'scratch'")
    return cast(InitializationMode, initialization)


MIN_SEATS = 2
MAX_SEATS = 4


class PolicyValueNetwork(nn.Module):
    """Masked policy and configurable value heads for imitation PPO.

    New models use an unbounded return critic.  ``outcome`` is retained for
    loading the historical tanh-bounded checkpoints whose value semantics were
    trained around terminal outcome targets.
    """

    def __init__(  # noqa: PLR0913 - mirrors the checkpoint config surface
        self,
        input_dim: int,
        *,
        feature_version: str,
        hidden_layers: tuple[int, ...] = HIDDEN_DIMS,
        output_dim: int = ACTION_DIM,
        value_mode: ValueMode = "return",
        critic_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        expected_dim = observation_dim(feature_version)
        if input_dim != expected_dim:
            raise ValueError(
                f"feature schema {feature_version!r} requires {expected_dim} inputs, got {input_dim}"
            )
        if output_dim != ACTION_DIM:
            raise ValueError(f"PPO action head must have {ACTION_DIM} outputs")
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        if critic_hidden_dim < 0:
            raise ValueError("critic_hidden_dim must be non-negative")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_version = feature_version
        self.hidden_layers = hidden_layers
        self.critic_hidden_dim = critic_hidden_dim
        self.value_mode = _validate_value_mode(value_mode)
        self.normalizer = FixedNormalizer(input_dim)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()])
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(previous, output_dim)
        if critic_hidden_dim:
            # Roadmap C1 ablation: a critic-private hidden layer on top of the
            # shared trunk, so value fitting cannot fight the policy trunk.
            self.value_head: nn.Module = nn.Sequential(
                nn.Linear(previous, critic_hidden_dim),
                nn.LayerNorm(critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim, 1),
            )
        else:
            self.value_head = nn.Linear(previous, 1)
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        value_final = (
            self.value_head[-1]
            if isinstance(self.value_head, nn.Sequential)
            else self.value_head
        )
        assert isinstance(value_final, nn.Linear)
        nn.init.orthogonal_(value_final.weight, gain=1.0)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use the same orthogonal MLP initialization as the repository PPO."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()

    @classmethod
    def from_bc(
        cls,
        bc_model: BehaviorCloningNetwork,
        *,
        initialization: InitializationMode = "bc",
        value_mode: ValueMode = "return",
        critic_hidden_dim: int = 0,
    ) -> "PolicyValueNetwork":
        """Build PPO from BC while keeping normalizer/schema provenance.

        ``scratch`` intentionally keeps the BC normalizer and input schema but
        does not copy the BC trunk or policy head, so a comparison changes only
        policy/trunk initialization while using the same source statistics.
        """
        _validate_initialization(initialization)
        model = cls(
            bc_model.input_dim,
            feature_version=bc_model.feature_version,
            hidden_layers=bc_model.hidden_layers,
            output_dim=bc_model.output_dim,
            value_mode=value_mode,
            critic_hidden_dim=critic_hidden_dim,
        )
        model.normalizer.mean.copy_(bc_model.normalizer.mean)
        model.normalizer.variance.copy_(bc_model.normalizer.variance)
        model.normalizer.fitted = bc_model.normalizer.fitted
        if initialization == "bc":
            model.trunk.load_state_dict(deepcopy(bc_model.trunk.state_dict()))
            model.policy_head.load_state_dict(
                deepcopy(bc_model.policy_head.state_dict())
            )
        return model

    def _hidden(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        return self.trunk(self.normalizer(observations))

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked action logits and values in the configured mode."""
        if legal_masks.dim() == 1:
            legal_masks = legal_masks.unsqueeze(0)
        hidden = self._hidden(observations)
        logits = self.policy_head(hidden)
        if logits.shape != legal_masks.shape:
            raise ValueError(
                f"logits and masks must have equal shape, got {logits.shape} and {legal_masks.shape}"
            )
        if not torch.all(legal_masks.sum(dim=1) > 0):
            raise ValueError("each PPO row must contain at least one legal action")
        values = self.value_head(hidden).squeeze(-1)
        if self.value_mode == "outcome":
            values = torch.tanh(values)
        return logits.masked_fill(legal_masks <= 0, HUGE_NEG), values


class PPOPolicyAgent(Agent):
    """Greedy frozen-policy wrapper used by evaluation and opponent pools."""

    def __init__(
        self,
        _id: int,
        *,
        model: PolicyValueNetwork,
        device_name: DeviceName = "cpu",
        stochastic: bool = False,
    ) -> None:
        super().__init__(_id)
        self.device = resolve_device(device_name)
        self.model = model.to(self.device).eval()
        self.stochastic = stochastic

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Select a legal action from the frozen policy/value network."""
        del game_rule
        observation = extract_observation(
            game_state, self.id, self.model.feature_version
        )
        mask = create_legal_actions_mask(actions, game_state, self.id).astype(
            np.float32
        )
        with torch.no_grad():
            logits, _ = self.model(
                torch.from_numpy(observation).to(self.device),
                torch.from_numpy(mask).to(self.device),
            )
            if self.stochastic:
                action_index = int(
                    distributions.Categorical(logits=logits).sample().item()
                )
            else:
                action_index = int(logits.argmax(dim=-1).item())
        return create_action_mapping(actions, game_state, self.id)[action_index]


def build_policy_candidate(
    model: PolicyValueNetwork,
    *,
    name: str = "ppo-policy",
    snapshot: str | None = None,
    device_name: DeviceName = "cpu",
) -> CandidateSpec:
    """Freeze a policy/value model as an isolated fixed candidate."""
    device = resolve_device(device_name)
    template = deepcopy(model).to(device).eval()
    template.requires_grad_(False)

    def factory(agent_id: int) -> PPOPolicyAgent:
        return PPOPolicyAgent(
            agent_id,
            model=deepcopy(template).to(device).eval(),
            device_name=device_name,
        )

    return CandidateSpec(
        name=name,
        role="fixed_baseline",
        factory=factory,
        feature_version=model.feature_version,
        snapshot=snapshot,
    )


@dataclass(frozen=True)
class PPOConfig:
    """Declared PPO update budget and terminal-value semantics."""

    feature_version: str = "v1"
    hidden_layers: tuple[int, ...] = HIDDEN_DIMS
    value_mode: ValueMode = "return"
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    minibatch_size: int = 256
    update_epochs: int = 4
    updates: int = 10
    games_per_update: int = 4
    terminal_value: float = 10.0
    target_kl: float | None = 0.02
    reference_kl_coefficient: float = 0.0
    current_weight: float = 1.0
    history_weight: float = 1.0
    history_limit: int = 4
    critic_warmup_epochs: int = 0
    # Roadmap C1 critic-repair knobs; defaults preserve the shared-lr model.
    critic_learning_rate: float | None = None
    critic_hidden_dim: int = 0
    # Roadmap E1: seat count for self-play games (2 by default, frozen run).
    n_seats: int = 2
    # Roadmap B2->C: training-side reward shaping ("none" = frozen run).
    shaping_kind: str = "none"
    shaping_kappa: float = 0.05
    initialization: InitializationMode = "bc"
    eval_every: int = 1
    seed: int = 1234
    device_name: DeviceName = "cpu"

    def __post_init__(self) -> None:  # noqa: C901, PLR0912 - validate each bound
        _validate_value_mode(self.value_mode)
        _validate_initialization(self.initialization)
        if (
            not np.isfinite(self.learning_rate)
            or not np.isfinite(self.discount_factor)
            or self.learning_rate <= 0
            or self.discount_factor <= 0
        ):
            raise ValueError("PPO learning rate and discount must be positive")
        if (
            not np.isfinite(self.gae_lambda)
            or not np.isfinite(self.clip_epsilon)
            or not 0 < self.gae_lambda <= 1
            or not 0 < self.clip_epsilon < 1
        ):
            raise ValueError("invalid GAE or PPO clip value")
        if (
            not np.isfinite(self.entropy_coefficient)
            or not np.isfinite(self.value_coefficient)
            or self.entropy_coefficient < 0
            or self.value_coefficient <= 0
        ):
            raise ValueError("invalid PPO loss coefficients")
        if (
            not np.isfinite(self.max_grad_norm)
            or not np.isfinite(self.terminal_value)
            or self.max_grad_norm <= 0
            or self.terminal_value <= 0
        ):
            raise ValueError("gradient limit and terminal value must be positive")
        if (
            min(
                self.minibatch_size,
                self.update_epochs,
                self.updates,
                self.games_per_update,
                self.eval_every,
            )
            < 1
        ):
            raise ValueError("PPO budgets must be positive")
        if self.target_kl is not None and (
            not np.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be positive or None")
        for name, value in (
            ("reference_kl_coefficient", self.reference_kl_coefficient),
            ("current_weight", self.current_weight),
            ("history_weight", self.history_weight),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.history_limit < 0 or self.critic_warmup_epochs < 0:
            raise ValueError(
                "history_limit and critic_warmup_epochs must be non-negative"
            )
        if self.critic_learning_rate is not None and (
            not np.isfinite(self.critic_learning_rate) or self.critic_learning_rate <= 0
        ):
            raise ValueError("critic_learning_rate must be positive when set")
        if self.critic_hidden_dim < 0:
            raise ValueError("critic_hidden_dim must be non-negative")
        if not MIN_SEATS <= self.n_seats <= MAX_SEATS:
            raise ValueError("n_seats must lie in [2, 4]")
        if self.shaping_kind not in ("none", "potential", "event"):
            raise ValueError(f"unknown shaping kind {self.shaping_kind!r}")
        if self.value_mode == "outcome" and self.terminal_value > 1.0:
            raise ValueError(
                "value_mode='outcome' bounds the critic to [-1, 1] via tanh, "
                f"so terminal_value={self.terminal_value} is unrepresentable "
                "(the 2026-09-07 era ran exactly this mismatch; use "
                "value_mode='return' for score/terminal-scale targets)"
            )
        if not np.isfinite(self.shaping_kappa) or self.shaping_kappa < 0:
            raise ValueError("shaping_kappa must be finite and non-negative")
        if self.n_seats > MIN_SEATS and self.feature_version != "public-v2-multi":
            raise ValueError(
                "training with more than two seats requires the "
                "public-v2-multi feature schema (roadmap B1/E1)"
            )
        if self.seed < 0:
            raise ValueError("PPO seed must be non-negative")


@dataclass(frozen=True)
class OpponentPoolEntry:
    """One complete-game opponent choice with an explicit sampling weight."""

    name: str
    candidate: CandidateSpec
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("opponent pool entry needs a name")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("opponent pool weights must be finite and non-negative")


@dataclass(frozen=True)
class FormalOpponentSpec:
    """Pickle-safe declaration rebuilt inside a formal training worker."""

    entry_name: str
    candidate_name: str
    weight: float = 1.0
    checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if not self.entry_name or not self.candidate_name:
            raise ValueError("formal opponent names must not be empty")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("formal opponent weight must be finite and non-negative")
        checkpoint_required = self.candidate_name in {"ppo", "corrected-dqn"}
        if checkpoint_required and self.checkpoint is None:
            raise ValueError(
                f"formal opponent {self.candidate_name!r} requires a checkpoint"
            )
        if not checkpoint_required and self.checkpoint is not None:
            raise ValueError(
                f"formal built-in opponent {self.candidate_name!r} "
                "must not declare a checkpoint"
            )

    def build(self, *, device_name: DeviceName) -> OpponentPoolEntry:
        """Rebuild the candidate only after the spawn boundary."""
        candidate = build_builtin_candidate(
            self.candidate_name,
            checkpoint=self.checkpoint,
            device_name=device_name,
        )
        return OpponentPoolEntry(self.entry_name, candidate, self.weight)

    def manifest_dict(self) -> dict[str, object]:
        """Return the immutable fixed-pool declaration used before spawning."""
        return {
            "entry_name": self.entry_name,
            "candidate_name": self.candidate_name,
            "weight": float(self.weight),
            "checkpoint_sha256": (
                sha256_file(self.checkpoint) if self.checkpoint is not None else None
            ),
        }


def formal_opponent_pool_sha256(entries: Sequence[FormalOpponentSpec]) -> str:
    """Hash a fixed opponent pool independently of caller/worker ordering."""
    ordered = sorted(entries, key=lambda entry: entry.entry_name)
    if not ordered or len({entry.entry_name for entry in ordered}) != len(ordered):
        raise ValueError("formal opponent pool entries must be non-empty and unique")
    if not any(entry.weight > 0 for entry in ordered):
        raise ValueError("formal opponent pool must contain positive probability mass")
    return sha256_canonical_json(
        {
            "protocol": "formal-opponent-pool-v1",
            "items": [entry.manifest_dict() for entry in ordered],
        }
    )


def formal_ppo_config_sha256(config: PPOConfig) -> str:
    """Hash every trainer/reward/pool-dynamics setting used by one arm."""
    return sha256_canonical_json(
        {
            "protocol": "formal-ppo-config-v1",
            "config": asdict(config),
        }
    )


def make_formal_treatment_contract(
    treatment_id: str,
    *,
    initial_bc: Path,
    config: PPOConfig,
    opponent_pool: Sequence[FormalOpponentSpec],
) -> FormalTreatmentContract:
    """Bind one treatment's checkpoint, full PPO config and fixed pool."""
    if config.seed != 0:
        raise ValueError(
            "formal PPO config.seed must be the unused sentinel 0; all random "
            "streams come from the seed-roll protocol"
        )
    return FormalTreatmentContract(
        treatment_id=treatment_id,
        initial_checkpoint_sha256=sha256_file(initial_bc),
        trainer_config_sha256=formal_ppo_config_sha256(config),
        opponent_pool_sha256=formal_opponent_pool_sha256(opponent_pool),
    )


@dataclass(frozen=True)
class FormalPPOTrainingJob:
    """Pickle-safe input for one treatment/replicate optimizer job."""

    initial_bc: Path
    output_dir: Path
    source_manifest: Path
    config: PPOConfig
    formal_spec: FormalTrainingSpec
    opponent_pool: tuple[FormalOpponentSpec, ...]
    scenario_bank_path: Path
    seed_roll_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.opponent_pool:
            raise ValueError("formal PPO job needs a non-empty opponent pool")
        names = [entry.entry_name for entry in self.opponent_pool]
        if len(names) != len(set(names)):
            raise ValueError("formal PPO opponent entry names must be unique")
        if not self.scenario_bank_path.name:
            raise ValueError("formal PPO job needs a scenario bank path")
        if (self.formal_spec.seed_roll_payload_sha256 is None) != (
            self.seed_roll_path is None
        ):
            raise ValueError(
                "paired-training-v2 jobs require exactly one seed-roll artifact"
            )
        if self.formal_spec.seed_roll_payload_sha256 is not None:
            contract = self.formal_spec.treatment_contract(
                self.formal_spec.treatment_id
            )
            if sha256_file(self.initial_bc) != contract.initial_checkpoint_sha256:
                raise ValueError(
                    "formal PPO initial checkpoint does not match its spec"
                )
            if formal_ppo_config_sha256(self.config) != contract.trainer_config_sha256:
                raise ValueError("formal PPO config does not match its treatment spec")
            if (
                formal_opponent_pool_sha256(self.opponent_pool)
                != contract.opponent_pool_sha256
            ):
                raise ValueError("formal PPO opponent pool does not match its spec")
            expected_output = self.formal_spec.output_for(
                self.formal_spec.replicate_id,
                self.formal_spec.treatment_id,
            )
            if str(self.output_dir.resolve()) != expected_output.output_dir:
                raise ValueError("formal PPO output directory does not match its spec")


def _formal_output_reservation_payload_for(
    *,
    source_manifest: Path,
    formal_spec: FormalTrainingSpec,
    output_dir: Path,
) -> dict[str, object]:
    """Build the manifest-bound one-shot marker for a v2 optimizer job."""
    from .manifest import load_manifest  # noqa: PLC0415 - avoid import cycle

    manifest = load_manifest(source_manifest)
    if manifest.get("status") != "running":
        raise RuntimeError("formal output reservation requires a running manifest")
    contract = formal_spec.treatment_contract(formal_spec.treatment_id)
    output = formal_spec.output_for(
        formal_spec.replicate_id,
        formal_spec.treatment_id,
    )
    if str(output_dir.resolve()) != output.output_dir:
        raise RuntimeError("formal PPO output directory does not match its declaration")
    body: dict[str, object] = {
        "protocol": "formal-ppo-output-reservation-v1",
        "manifest_path": str(source_manifest.resolve()),
        "manifest_declaration_sha256": manifest.get("declaration_sha256"),
        "manifest_provenance_sha256": manifest.get("provenance_sha256"),
        "formal_training_sha256": sha256_canonical_json(formal_spec.manifest_binding()),
        "replicate_id": formal_spec.replicate_id,
        "treatment_id": formal_spec.treatment_id,
        "treatment_contract": contract.as_dict(),
        "output": output.as_dict(),
        "scenario_bank_sha256": formal_spec.scenario_bank_sha256,
        "seed_roll_payload_sha256": formal_spec.seed_roll_payload_sha256,
    }
    body["reservation_sha256"] = sha256_canonical_json(body)
    return body


def _formal_output_reservation_payload(
    job: FormalPPOTrainingJob,
) -> dict[str, object]:
    return _formal_output_reservation_payload_for(
        source_manifest=job.source_manifest,
        formal_spec=job.formal_spec,
        output_dir=job.output_dir,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_real_output_directory(path: Path) -> None:
    """Reject replacement/symlink output directories at mutation time."""
    lexical = Path(os.path.abspath(os.fspath(path)))  # noqa: PTH100
    try:
        metadata = os.lstat(lexical)
    except OSError as exc:
        raise RuntimeError(f"cannot access formal PPO output directory: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or lexical.resolve(strict=True) != lexical
    ):
        raise RuntimeError("formal PPO output path is not a real directory")


def _reserve_formal_output_dirs(jobs: Sequence[FormalPPOTrainingJob]) -> None:
    """One-shot claim every manifest-declared v2 output before spawning."""
    ordered = sorted(
        jobs,
        key=lambda job: (
            job.formal_spec.replicate_id,
            job.formal_spec.treatment_id,
        ),
    )
    for job in ordered:
        if os.path.lexists(job.output_dir):
            raise RuntimeError(
                f"formal PPO output was already attempted: {job.output_dir}"
            )
    for job in ordered:
        job.output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            job.output_dir.mkdir()
        except OSError as exc:
            raise RuntimeError(
                f"cannot reserve formal PPO output {job.output_dir}: {exc}"
            ) from exc
        _require_real_output_directory(job.output_dir)
        marker = job.output_dir / FORMAL_OUTPUT_RESERVATION_NAME
        payload = _formal_output_reservation_payload(job)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(marker, flags, 0o444)
        except OSError as exc:
            raise RuntimeError(
                f"cannot write formal PPO output reservation {marker}: {exc}"
            ) from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(job.output_dir)
        _fsync_directory(job.output_dir.parent)


def _require_formal_output_reservation(job: FormalPPOTrainingJob) -> None:
    """Require the exact parent-created marker before a v2 worker mutates state."""
    _require_formal_output_reservation_for(
        source_manifest=job.source_manifest,
        formal_spec=job.formal_spec,
        output_dir=job.output_dir,
    )


def _require_formal_output_reservation_for(
    *,
    source_manifest: Path,
    formal_spec: FormalTrainingSpec,
    output_dir: Path,
) -> None:
    """Require the exact parent-created marker at the low-level trainer gate."""
    _require_real_output_directory(output_dir)
    marker = output_dir / FORMAL_OUTPUT_RESERVATION_NAME
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker, flags)
    except OSError as exc:
        raise RuntimeError(f"formal PPO output is not reserved: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("formal PPO output reservation is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            actual = stream.read()
    finally:
        os.close(descriptor)
    expected = (
        json.dumps(
            _formal_output_reservation_payload_for(
                source_manifest=source_manifest,
                formal_spec=formal_spec,
                output_dir=output_dir,
            ),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if actual != expected:
        raise RuntimeError("formal PPO output reservation does not match its manifest")


@dataclass(frozen=True)
class WeightedPoolItem:
    """One stable entry in a normalized formal opponent-pool CDF."""

    name: str
    raw_weight: float
    probability: float
    cumulative_probability: float
    candidate_role: str
    feature_version: str
    snapshot: str | None


@dataclass(frozen=True)
class WeightedPoolSnapshot:
    """Canonical, auditable opponent pool used only by the formal path."""

    version: str
    items: tuple[WeightedPoolItem, ...]
    snapshot_sha256: str

    @classmethod
    def from_entries(
        cls, entries: Sequence[OpponentPoolEntry]
    ) -> "WeightedPoolSnapshot":
        """Normalize unique entries after sorting by their stable names."""
        ordered = sorted(entries, key=lambda entry: entry.name)
        if not ordered:
            raise ValueError("PPO opponent pool must not be empty")
        if len({entry.name for entry in ordered}) != len(ordered):
            raise ValueError("opponent pool entry names must be unique")
        total = float(sum(entry.weight for entry in ordered))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("PPO opponent pool must contain positive mass")
        cumulative = 0.0
        items: list[WeightedPoolItem] = []
        for index, entry in enumerate(ordered):
            probability = float(entry.weight / total)
            cumulative += probability
            items.append(
                WeightedPoolItem(
                    name=entry.name,
                    raw_weight=float(entry.weight),
                    probability=probability,
                    cumulative_probability=1.0
                    if index == len(ordered) - 1
                    else cumulative,
                    candidate_role=entry.candidate.role,
                    feature_version=entry.candidate.feature_version,
                    snapshot=entry.candidate.snapshot,
                )
            )
        payload = [asdict(item) for item in items]
        return cls(
            version="weighted-pool-v1/name-ascending",
            items=tuple(items),
            snapshot_sha256=sha256_canonical_json(
                {"version": "weighted-pool-v1/name-ascending", "items": payload}
            ),
        )

    def select(
        self,
        entries: Sequence[OpponentPoolEntry],
        u53: float,
    ) -> OpponentPoolEntry:
        """Select from the exact entries used to build this snapshot."""
        by_name = {entry.name: entry for entry in entries}
        if WeightedPoolSnapshot.from_entries(entries) != self:
            raise ValueError("opponent entries do not match the pool snapshot")
        index = inverse_cdf_index(
            [item.probability for item in self.items],
            u53,
        )
        return by_name[self.items[index].name]

    def as_dict(self, draws: Sequence[str] = ()) -> dict[str, Any]:
        """Return probabilities, CDF, snapshot hash, and reproducible counts."""
        names = {item.name for item in self.items}
        unknown = sorted(set(draws) - names)
        if unknown:
            raise ValueError(f"pool counts contain unknown entries: {unknown}")
        return {
            "version": self.version,
            "snapshot_sha256": self.snapshot_sha256,
            "items": [asdict(item) for item in self.items],
            "actual_counts": {
                item.name: sum(draw == item.name for draw in draws)
                for item in self.items
            },
            "draws": len(draws),
        }


@dataclass(frozen=True)
class PPOTransition:
    """One focal-policy transition with old-policy statistics for PPO."""

    observation: np.ndarray
    legal_mask: np.ndarray
    action_index: int
    old_log_probability: float
    old_value: float
    reward: float
    terminal: bool
    seed: int
    seat: int
    scenario_id: str | None = None
    scenario_source_digest: str | None = None


def _sample_pool_entry(
    entries: Sequence[OpponentPoolEntry], rng: random.Random
) -> OpponentPoolEntry:
    """Choose exactly one pool policy for an entire game."""
    eligible = [entry for entry in entries if entry.weight > 0]
    if not eligible:
        raise ValueError("PPO opponent pool must not be empty")
    return rng.choices(eligible, weights=[entry.weight for entry in eligible], k=1)[0]


def build_opponent_pool(  # noqa: PLR0913 - explicit opponent bucket controls
    fixed_entries: Sequence[OpponentPoolEntry],
    history_entries: Sequence[OpponentPoolEntry] = (),
    current_entry: OpponentPoolEntry | None = None,
    *,
    current_weight: float = 1.0,
    history_weight: float = 1.0,
    history_limit: int = 4,
) -> list[OpponentPoolEntry]:
    """Build a non-drifting fixed/current/history sampling pool.

    Fixed entries retain their declared weights.  The current policy receives
    one current-bucket mass, while the retained history receives one shared
    history-bucket mass divided equally across the most recent snapshots.  If
    there is no retained history, the history mass is explicitly transferred
    to the current entry; callers can record that decision in their run log.
    Zero-weight entries are retained for auditable configuration but are never
    sampled.
    """
    if history_limit < 0:
        raise ValueError("history_limit must be non-negative")
    for name, weight in (
        ("current_weight", current_weight),
        ("history_weight", history_weight),
    ):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    retained_history = list(history_entries[-history_limit:]) if history_limit else []
    if not retained_history and history_weight > 0 and current_entry is None:
        raise ValueError("history mass cannot be redistributed without current_entry")

    result: list[OpponentPoolEntry] = list(fixed_entries)
    names = {entry.name for entry in result}
    if len(names) != len(result):
        raise ValueError("opponent pool entry names must be unique")

    redistributed_history = history_weight if not retained_history else 0.0
    if current_entry is not None:
        if current_entry.name in names:
            raise ValueError("opponent pool entry names must be unique")
        names.add(current_entry.name)
        result.append(
            replace(
                current_entry,
                weight=current_weight + redistributed_history,
            )
        )

    history_share = history_weight / len(retained_history) if retained_history else 0.0
    for entry in retained_history:
        if entry.name in names:
            raise ValueError("opponent pool entry names must be unique")
        names.add(entry.name)
        result.append(replace(entry, weight=history_share))
    return result


def _policy_step(  # noqa: PLR0913 - formal RNG is an explicit opt-in coordinate
    model: PolicyValueNetwork,
    state: SplendorState,
    actions: list[ActionType],
    seat: int,
    *,
    device: torch.device,
    rng_lineage: SeedLineage | None = None,
) -> tuple[np.ndarray, np.ndarray, int, ActionType, float, float]:
    """Sample one focal action and return its observation/log-prob/value.

    Legacy calls retain ``Categorical.sample``.  Formal calls consume the
    event-keyed ``u53`` against a CPU float64 CDF over ascending legal action
    indexes, making the choice independent of global Torch RNG state.
    """
    observation = extract_observation(state, seat, model.feature_version)
    legal_mask = create_legal_actions_mask(actions, state, seat).astype(np.uint8)
    observation_tensor = torch.from_numpy(observation).to(device)
    mask_tensor = torch.from_numpy(legal_mask.astype(np.float32)).to(device)
    with torch.no_grad():
        logits, value = model(observation_tensor, mask_tensor)
        distribution = distributions.Categorical(logits=logits)
        if rng_lineage is None:
            sampled = distribution.sample()
        else:
            legal_indexes = np.flatnonzero(legal_mask)
            legal_index_tensor = torch.from_numpy(legal_indexes).to(logits.device)
            legal_logits = logits.squeeze(0)[legal_index_tensor].detach().cpu().double()
            shifted = legal_logits - legal_logits.max()
            probabilities = torch.exp(shifted).numpy()
            selected_offset = inverse_cdf_index(probabilities.tolist(), rng_lineage.u53)
            sampled = torch.tensor(
                int(legal_indexes[selected_offset]),
                dtype=torch.long,
                device=logits.device,
            )
        log_probability = float(distribution.log_prob(sampled).item())
        state_value = float(value.item())
    action_index = int(sampled.item())
    action = create_action_mapping(actions, state, seat)[action_index]
    return (
        observation,
        legal_mask,
        action_index,
        action,
        log_probability,
        state_value,
    )


def _outcome(rule: SplendorGameRule, seat: int) -> int:
    """Use calScore's card-count tie-break for terminal utility.

    Seat-count agnostic: the outcome is judged against the *best* rival, so
    two-player games keep the original semantics exactly.
    """
    state = rule.current_game_state
    own = float(rule.calScore(state, seat))
    best_rival = max(
        (
            float(rule.calScore(state, agent.id))
            for agent in state.agents
            if agent.id != seat
        ),
        default=own,
    )
    return int(own > best_rival) - int(own < best_rival)


def collect_ppo_game(  # noqa: C901,PLR0912,PLR0913,PLR0915 - game accounting is explicit
    model: PolicyValueNetwork,
    opponent_pool: Sequence[OpponentPoolEntry],
    *,
    seed: int,
    seat: int,
    config: PPOConfig,
    update_index: int,
    game_index: int,
    formal_rng: FormalGameRng | None = None,
    scenario: ScenarioV1 | None = None,
) -> tuple[dict[str, Any], list[PPOTransition]]:
    """Collect one on-policy game without teacher labels or policy switching.

    Supplying ``formal_rng`` opts into named event streams.  Omitting it keeps
    the historical whole-game seed behavior byte-for-byte compatible.
    """
    n_seats = config.n_seats
    if seat not in range(n_seats):
        raise ValueError(f"seat must lie in [0, {n_seats}), got {seat}")
    if formal_rng is not None and (
        formal_rng.seat != seat
        or formal_rng.update != update_index
        or formal_rng.game_index != game_index
    ):
        raise ValueError("formal RNG coordinates do not match the PPO game")
    if formal_rng is not None and scenario is None:
        raise ValueError("formal PPO games require an immutable ScenarioV1 snapshot")
    if scenario is not None and scenario.n_seats != n_seats:
        raise ValueError("PPO seat count does not match the ScenarioV1 snapshot")
    if (
        formal_rng is not None
        and scenario is not None
        and formal_rng.scenario_id != scenario.scenario_id
    ):
        raise ValueError("formal RNG scenario_id does not match the snapshot")
    device = next(model.parameters()).device
    pool_snapshot: WeightedPoolSnapshot | None = None
    pool_lineage: SeedLineage | None = None
    scenario_lineage: SeedLineage | None = None
    if formal_rng is None:
        pool_rng = random.Random(seed + 1_000_003 * (update_index + 1) + game_index)
        selected_entry = _sample_pool_entry(opponent_pool, pool_rng)
    else:
        pool_snapshot = WeightedPoolSnapshot.from_entries(opponent_pool)
        pool_lineage = formal_rng.lineage("pool_draw")
        selected_entry = pool_snapshot.select(opponent_pool, pool_lineage.u53)
        scenario_lineage = formal_rng.lineage("scenario_source")
    rivals: dict[int, Any] = {}
    opponent_init_lineages: list[dict[str, object]] = []

    def rival_for(turn: int) -> Agent:
        """Build each rival seat once per game from the sampled entry."""
        if turn not in rivals:
            if formal_rng is None:
                rivals[turn] = selected_entry.candidate.build(turn)
            else:
                init_lineage = formal_rng.lineage(
                    "opponent_init",
                    opponent_id=selected_entry.name,
                    opponent_step=turn,
                )
                opponent_init_lineages.append(init_lineage.as_dict())
                # Candidate factories are legacy callables and may initialize
                # from global Python/NumPy/Torch RNGs.  Isolate that one-time
                # construction just like each subsequent opponent decision.
                with isolated_seed(init_lineage.seed63):
                    rivals[turn] = selected_entry.candidate.build(turn)
        return rivals[turn]

    transitions: list[PPOTransition] = []
    failure: dict[str, str] | None = None
    opponent_queries = 0
    opponent_illegal = 0
    opponent_search_nodes = 0
    opponent_latencies: list[float] = []
    policy_lineages: list[dict[str, object]] = []
    opponent_lineages: list[dict[str, object]] = []
    action_trace: list[dict[str, object]] = []
    started = time.perf_counter()

    game_scope = isolated_seed(seed) if formal_rng is None else nullcontext()
    with game_scope:
        if scenario is not None:
            rule = rule_from_scenario(scenario)
        elif scenario_lineage is None:
            rule = LimitRoundsGameRule(n_seats)
        else:  # pragma: no cover - rejected by the formal snapshot gate above
            raise AssertionError("formal ScenarioV1 gate was bypassed")
        # Roadmap B2->C: potential shaping is anchored at the first focal
        # decision and credits each transition when the next focal state (or
        # the terminal state) arrives - gamma*phi(s') - phi(s) telescopes.
        potential_shaper: PotentialRewardShaper | None = None
        event_shaper: EventRewardShaper | None = None
        if config.shaping_kind == "potential":
            potential_shaper = PotentialRewardShaper(
                kappa=config.shaping_kappa,
                discount_factor=config.discount_factor,
            )
        elif config.shaping_kind == "event":
            event_shaper = EventRewardShaper()
        shaper_anchored = False
        pending_bonus_index: int | None = None
        while not rule.gameEnds():
            state = rule.current_game_state
            turn = rule.current_agent_index
            legal_actions = rule.getLegalActions(state, turn)
            if turn == seat:
                if potential_shaper is not None:
                    if not shaper_anchored:
                        potential_shaper.reset(state, seat, rule)
                        shaper_anchored = True
                    elif pending_bonus_index is not None:
                        shaping_bonus = potential_shaper.advance(state, seat, rule)
                        transitions[pending_bonus_index] = replace(
                            transitions[pending_bonus_index],
                            reward=transitions[pending_bonus_index].reward
                            + shaping_bonus,
                        )
                        pending_bonus_index = None
                score_before = state.agents[seat].score
                try:
                    policy_lineage = (
                        formal_rng.lineage(
                            "policy_action",
                            focal_step=len(transitions),
                        )
                        if formal_rng is not None
                        else None
                    )
                    if policy_lineage is not None:
                        policy_lineages.append(policy_lineage.as_dict())
                    (
                        observation,
                        legal_mask,
                        action_index,
                        action,
                        old_log_probability,
                        old_value,
                    ) = _policy_step(
                        model,
                        state,
                        legal_actions,
                        seat,
                        device=device,
                        rng_lineage=policy_lineage,
                    )
                except Exception as exc:
                    failure = {
                        "side": "student",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                rule.update(action)
                action_trace.append(
                    {
                        "ply": rule.action_counter - 1,
                        "seat": turn,
                        "action_index": action_index,
                        "action_type": str(action["type"]),
                        "rng_digest": policy_lineage.digest_hex
                        if policy_lineage is not None
                        else None,
                    }
                )
                base_reward = float(
                    rule.current_game_state.agents[seat].score - score_before
                )
                if event_shaper is not None:
                    base_reward += event_shaper.bonus(
                        action, rule.current_game_state, rule, seat
                    )
                transitions.append(
                    PPOTransition(
                        observation=observation,
                        legal_mask=legal_mask,
                        action_index=action_index,
                        old_log_probability=old_log_probability,
                        old_value=old_value,
                        reward=base_reward,
                        terminal=rule.gameEnds(),
                        seed=scenario.source_seed if scenario is not None else seed,
                        seat=seat,
                        scenario_id=formal_rng.scenario_id
                        if formal_rng is not None
                        else None,
                        scenario_source_digest=scenario_lineage.digest_hex
                        if scenario_lineage is not None
                        else None,
                    )
                )
                pending_bonus_index = len(transitions) - 1
            else:
                opponent_queries += 1
                try:
                    opponent_lineage = (
                        formal_rng.lineage(
                            "opponent_action",
                            opponent_id=selected_entry.name,
                            opponent_step=opponent_queries - 1,
                        )
                        if formal_rng is not None
                        else None
                    )
                    if opponent_lineage is not None:
                        opponent_lineages.append(opponent_lineage.as_dict())
                    decision = select_action(
                        rival_for(turn),
                        legal_actions,
                        state,
                        rule,
                        rng_lineage=opponent_lineage,
                    )
                except TeacherDecisionError as exc:
                    opponent_illegal += int(exc.illegal)
                    opponent_search_nodes += exc.search_nodes
                    opponent_latencies.append(exc.elapsed_seconds)
                    failure = {
                        "side": "opponent",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                opponent_latencies.append(decision.elapsed_seconds)
                opponent_search_nodes += decision.search_nodes
                rule.update(decision.action)
                action_trace.append(
                    {
                        "ply": rule.action_counter - 1,
                        "seat": turn,
                        "action_index": decision.action_index,
                        "action_type": str(decision.action["type"]),
                        "rng_digest": opponent_lineage.digest_hex
                        if opponent_lineage is not None
                        else None,
                    }
                )

    completed = failure is None and rule.gameEnds()
    terminal_outcome = _outcome(rule, seat) if completed else None
    if completed and potential_shaper is not None and pending_bonus_index is not None:
        shaping_bonus = potential_shaper.advance(rule.current_game_state, seat, rule)
        transitions[pending_bonus_index] = replace(
            transitions[pending_bonus_index],
            reward=transitions[pending_bonus_index].reward + shaping_bonus,
        )
        pending_bonus_index = None
    if completed and transitions:
        assert terminal_outcome is not None
        last = transitions[-1]
        transitions[-1] = replace(
            last,
            reward=last.reward + config.terminal_value * int(terminal_outcome),
            terminal=True,
        )
    state = rule.current_game_state
    score = float(rule.calScore(state, seat))
    rival_score = max(
        (
            float(rule.calScore(state, agent.id))
            for agent in state.agents
            if agent.id != seat
        ),
        default=0.0,
    )
    record: dict[str, Any] = {
        "update": update_index,
        "game_index": game_index,
        "seed": seed,
        "seat": seat,
        "opponent": selected_entry.name,
        "opponent_snapshot": selected_entry.candidate.snapshot,
        "status": "completed" if completed else "failed",
        "failure": failure,
        "outcome": terminal_outcome,
        "score": score,
        "rival_score": rival_score,
        "plies": rule.action_counter,
        "student_queries": len(transitions),
        "teacher_queries": 0,
        "opponent_queries": opponent_queries,
        "opponent_illegal_actions": opponent_illegal,
        "opponent_search_nodes": opponent_search_nodes,
        "opponent_action_seconds_mean": float(np.mean(opponent_latencies))
        if opponent_latencies
        else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "terminal_value_mapping": {
            -1: -config.terminal_value,
            0: 0.0,
            1: config.terminal_value,
        },
        "action_trace_sha256": sha256_canonical_json(action_trace),
    }
    if scenario is not None:
        record.update(
            {
                "scenario_id": scenario.scenario_id,
                "canonical_state_sha256": scenario.canonical_state_sha256,
                "scenario_source_segment": scenario.source_segment,
                "scenario_source_seed": scenario.source_seed,
                "legacy_seed_argument": seed,
            }
        )
    if formal_rng is not None:
        assert pool_snapshot is not None
        assert pool_lineage is not None
        assert scenario_lineage is not None
        record["rng_protocol"] = formal_rng.key("pool_draw").protocol_version
        record.update(
            {
                "experiment_id": formal_rng.experiment_id,
                "phase": formal_rng.phase,
                "replicate_id": formal_rng.replicate_id,
                "treatment_id": formal_rng.treatment_id,
                "scenario_id": formal_rng.scenario_id,
                "coupling_group": formal_rng.coupling_group,
                "legacy_seed_argument": seed,
                "scenario_source_digest": scenario_lineage.digest_hex,
                "scenario_source_seed63": scenario_lineage.seed63,
            }
        )
        assert scenario is not None
        record["seed"] = scenario.source_seed
        record["rng_lineage"] = {
            "scenario_source": scenario_lineage.as_dict(),
            "pool_draw": pool_lineage.as_dict(),
            "opponent_initializations": opponent_init_lineages,
            "policy_actions": policy_lineages,
            "opponent_actions": opponent_lineages,
        }
        record["opponent_pool_distribution"] = pool_snapshot.as_dict(
            [selected_entry.name]
        )
    # A failed game can contain useful diagnostics but it is not an on-policy
    # trajectory.  Never let a caller accidentally optimize on its prefix.
    return record, transitions if completed else []


def compute_gae(
    transitions: Sequence[PPOTransition],
    *,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute terminal-aware GAE across a batch of complete games."""
    if not transitions:
        raise ValueError("cannot compute GAE for an empty rollout")
    values = np.asarray(
        [transition.old_value for transition in transitions], dtype=np.float32
    )
    rewards = np.asarray(
        [transition.reward for transition in transitions], dtype=np.float32
    )
    terminals = np.asarray(
        [transition.terminal for transition in transitions], dtype=np.bool_
    )
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise FloatingPointError("non-finite PPO value or reward in GAE inputs")
    advantages = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        is_terminal = bool(terminals[index])
        if is_terminal:
            next_value = 0.0
            continuation = 0.0
        else:
            has_next_transition = index + 1 < len(values)
            next_value = float(values[index + 1]) if has_next_transition else 0.0
            continuation = float(has_next_transition)
        delta = (
            float(rewards[index]) + discount_factor * next_value - float(values[index])
        )
        running = delta + discount_factor * gae_lambda * continuation * running
        advantages[index] = running
    returns = advantages + values
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise FloatingPointError("non-finite PPO advantages or return targets")
    return advantages, returns


def _policy_logits(
    policy: BehaviorCloningNetwork | PolicyValueNetwork,
    observations: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Get masked logits from either a BC reference or PPO policy."""
    if isinstance(policy, BehaviorCloningNetwork):
        return policy(observations, legal_masks)
    logits, _ = policy(observations, legal_masks)
    return logits


def _critic_values_from_hidden(
    model: PolicyValueNetwork, hidden: torch.Tensor
) -> torch.Tensor:
    """Apply the configured value semantics without running the policy head."""
    values = model.value_head(hidden).squeeze(-1)
    if model.value_mode == "outcome":
        values = torch.tanh(values)
    return values


def warmup_critic(  # noqa: C901 - explicit finite-value and warmup safety checks
    model: PolicyValueNetwork,
    transitions: Sequence[PPOTransition],
    config: PPOConfig,
) -> dict[str, float | int]:
    """Fit only ``value_head`` to first-rollout return targets.

    The hidden representation is detached and a separate optimizer owns only
    the value head.  Thus BC's pretrained trunk and policy head cannot change
    during critic warmup.
    """
    if not transitions:
        raise ValueError("critic warmup needs at least one transition")
    if config.critic_warmup_epochs < 1:
        raise ValueError("critic warmup requires critic_warmup_epochs >= 1")
    _, returns = compute_gae(
        transitions,
        discount_factor=config.discount_factor,
        gae_lambda=config.gae_lambda,
    )
    if not np.isfinite(returns).all():
        raise FloatingPointError("non-finite critic warmup return targets")
    device = next(model.parameters()).device
    observations = torch.from_numpy(
        np.stack([transition.observation for transition in transitions]).astype(
            np.float32
        )
    ).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    model.eval()
    with torch.no_grad():
        hidden = model.trunk(model.normalizer(observations)).detach()
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("non-finite critic warmup hidden states")
    warmup_optimizer = optim.Adam(
        model.value_head.parameters(),
        lr=config.critic_learning_rate or config.learning_rate,
    )
    losses: list[float] = []
    for _ in range(config.critic_warmup_epochs):
        predicted = _critic_values_from_hidden(model, hidden)
        loss = F.smooth_l1_loss(predicted, return_tensor)
        if not torch.isfinite(predicted).all() or not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite critic warmup value or loss")
        warmup_optimizer.zero_grad()
        loss.backward()
        for parameter in model.value_head.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite critic warmup gradient")
        nn.utils.clip_grad_norm_(model.value_head.parameters(), config.max_grad_norm)
        warmup_optimizer.step()
        for parameter in model.value_head.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("non-finite critic warmup parameter")
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return {
        "epochs": config.critic_warmup_epochs,
        "optimizer_steps": config.critic_warmup_epochs,
        "loss": float(np.mean(losses)),
        "target_min": float(np.min(returns)),
        "target_max": float(np.max(returns)),
    }


def ppo_update(  # noqa: C901, PLR0912, PLR0913, PLR0915 - PPO accounting is explicit
    model: PolicyValueNetwork,
    optimizer: optim.Optimizer,
    transitions: Sequence[PPOTransition],
    config: PPOConfig,
    *,
    update_seed: int | None = None,
    minibatch_key: RngKey | None = None,
    reference_model: BehaviorCloningNetwork | PolicyValueNetwork | None = None,
) -> dict[str, Any]:
    """Apply clipped PPO updates to one frozen-policy rollout batch.

    The KL diagnostic uses the non-negative Schulman approximation
    ``((ratio - 1) - log_ratio)`` before each optimizer step.  If it exceeds
    ``target_kl``, the current minibatch is not applied and the update stops.
    """
    if not transitions:
        raise ValueError("PPO update needs at least one transition")
    if reference_model is model:
        raise ValueError("reference_model must be independent of the on-policy model")
    if (update_seed is None) == (minibatch_key is None):
        raise ValueError("provide exactly one of update_seed or minibatch_key")
    if minibatch_key is not None and minibatch_key.stream_name != "minibatch":
        raise ValueError("formal PPO update requires the minibatch RNG stream")
    advantages, returns = compute_gae(
        transitions,
        discount_factor=config.discount_factor,
        gae_lambda=config.gae_lambda,
    )
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + ADVANTAGE_STD_EPSILON
        )
    device = next(model.parameters()).device
    observations = torch.from_numpy(
        np.stack([transition.observation for transition in transitions]).astype(
            np.float32
        )
    ).to(device)
    masks = torch.from_numpy(
        np.stack([transition.legal_mask for transition in transitions]).astype(
            np.float32
        )
    ).to(device)
    actions = torch.tensor(
        [transition.action_index for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    old_log_probabilities = torch.tensor(
        [transition.old_log_probability for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    advantage_tensor = torch.from_numpy(advantages).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    if (
        not torch.isfinite(advantage_tensor).all()
        or not torch.isfinite(return_tensor).all()
    ):
        raise FloatingPointError("non-finite PPO advantage or return target")
    legacy_generator: torch.Generator | None = None
    if update_seed is not None:
        legacy_generator = torch.Generator(device="cpu")
        legacy_generator.manual_seed(update_seed)
    minibatch_lineages: list[dict[str, object]] = []
    model.train()
    if reference_model is not None:
        reference_model.eval()
    metrics: list[dict[str, float]] = []
    optimizer_steps = 0
    early_stopped = False
    epochs_completed = 0
    for epoch_index in range(config.update_epochs):
        epochs_completed += 1
        if minibatch_key is None:
            assert legacy_generator is not None
            generator = legacy_generator
        else:
            epoch_key = replace(minibatch_key, epoch=epoch_index)
            lineage = derive_seed(epoch_key)
            minibatch_lineages.append(lineage.as_dict())
            generator = torch.Generator(device="cpu")
            generator.manual_seed(lineage.seed63)
        permutation = torch.randperm(len(transitions), generator=generator)
        for start in range(0, len(transitions), config.minibatch_size):
            batch = permutation[start : start + config.minibatch_size].to(device)
            logits, values = model(observations[batch], masks[batch])
            if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                raise FloatingPointError("non-finite PPO policy logits or values")
            distribution = distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions[batch])
            log_ratio = log_probabilities - old_log_probabilities[batch]
            ratio = log_ratio.exp()
            if (
                not torch.isfinite(log_probabilities).all()
                or not torch.isfinite(ratio).all()
            ):
                raise FloatingPointError("non-finite PPO log-probability or ratio")
            approx_kl_tensor = (ratio - 1.0) - log_ratio
            approx_kl = float(
                torch.clamp(approx_kl_tensor.mean(), min=0.0).detach().cpu().item()
            )
            clip_fraction = float(
                (torch.abs(ratio - 1.0) > config.clip_epsilon)
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            )
            surrogate_one = ratio * advantage_tensor[batch]
            surrogate_two = (
                torch.clamp(
                    ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon,
                )
                * advantage_tensor[batch]
            )
            policy_loss = -torch.minimum(surrogate_one, surrogate_two).mean()
            value_loss = F.smooth_l1_loss(values, return_tensor[batch])
            entropy = distribution.entropy().mean()
            reference_kl = torch.zeros((), device=device)
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits = _policy_logits(
                        reference_model,
                        observations[batch],
                        masks[batch],
                    )
                if not torch.isfinite(reference_logits).all():
                    raise FloatingPointError("non-finite reference policy logits")
                reference_distribution = distributions.Categorical(
                    logits=reference_logits
                )
                reference_kl = distributions.kl_divergence(
                    reference_distribution, distribution
                ).mean()
            reference_kl_value = float(
                torch.clamp(reference_kl.detach(), min=0.0).cpu().item()
            )
            loss_values = (
                approx_kl,
                clip_fraction,
                float(policy_loss.detach().cpu().item()),
                float(value_loss.detach().cpu().item()),
                float(entropy.detach().cpu().item()),
                reference_kl_value,
            )
            if not all(np.isfinite(value) for value in loss_values):
                raise FloatingPointError("non-finite PPO loss or diagnostic")
            metrics.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "reference_kl": reference_kl_value,
                }
            )
            if config.target_kl is not None and approx_kl > config.target_kl:
                early_stopped = True
                break
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + config.reference_kl_coefficient * reference_kl
            )
            optimizer.zero_grad()
            loss.backward()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("non-finite PPO loss")
            for parameter in model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("non-finite PPO loss gradient")
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError("non-finite PPO parameter after update")
            optimizer_steps += 1
        if early_stopped:
            break
    if not metrics:
        raise RuntimeError("PPO update produced no minibatches")
    model.eval()
    with torch.no_grad():
        _, final_values = model(observations, masks)
    model.train()
    if not torch.isfinite(final_values).all():
        raise FloatingPointError("non-finite PPO values after update")
    predicted_values = final_values.detach().cpu().numpy()
    return_variance = float(np.var(returns))
    explained_variance = (
        0.0
        if return_variance <= EXPLAINED_VARIANCE_EPSILON
        else 1.0 - float(np.var(returns - predicted_values)) / return_variance
    )
    if not np.isfinite(explained_variance):
        raise FloatingPointError("non-finite PPO explained variance")
    aggregate_metrics = {
        key: float(np.mean([metric[key] for metric in metrics])) for key in metrics[0]
    }
    result: dict[str, Any] = {
        "samples": len(transitions),
        **aggregate_metrics,
        "advantage_mean": float(np.mean(advantages)),
        "return_mean": float(np.mean(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "value_mean": float(np.mean(predicted_values)),
        "value_min": float(np.min(predicted_values)),
        "value_max": float(np.max(predicted_values)),
        "explained_variance": explained_variance,
        "optimizer_steps": optimizer_steps,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
    }
    if minibatch_key is not None:
        result["rng_protocol"] = minibatch_key.protocol_version
        result["minibatch_lineage"] = minibatch_lineages
    return result


def _pool_draws_from_metrics(metrics: Mapping[str, Any]) -> list[str]:
    """Collect declared opponent draws from one update or a full run log."""
    records = metrics.get("training_records")
    if isinstance(records, list):
        return [
            str(record["opponent"])
            for record in records
            if isinstance(record, Mapping) and record.get("opponent") is not None
        ]
    logs = metrics.get("logs")
    if isinstance(logs, list):
        draws: list[str] = []
        for log in logs:
            if isinstance(log, Mapping):
                draws.extend(_pool_draws_from_metrics(log))
        return draws
    return []


def _checkpoint_pool_distribution(  # noqa: C901 - provenance branches stay explicit
    opponent_pool: Sequence[OpponentPoolEntry],
    metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact latest CDF plus aggregate draw provenance."""
    draws = _pool_draws_from_metrics(metrics)
    counts = Counter(draws)
    if not opponent_pool:
        if draws:
            raise ValueError("checkpoint metrics contain draws from an empty pool")
        empty_distribution = {
            "version": "weighted-pool-v1/name-ascending",
            "snapshot_sha256": sha256_canonical_json(
                {"version": "weighted-pool-v1/name-ascending", "items": []}
            ),
            "items": [],
            "actual_counts": {},
            "draws": 0,
            "count_scope": "empty-legacy-checkpoint",
        }
        return empty_distribution, {
            "count_scope": "all-training-records-in-checkpoint-metrics",
            "actual_counts": {},
            "draws": 0,
        }
    snapshot = WeightedPoolSnapshot.from_entries(opponent_pool)
    latest_distribution: Mapping[str, Any] | None = None
    direct_pool = metrics.get("opponent_pool")
    if isinstance(direct_pool, Mapping):
        candidate = direct_pool.get("normalized_distribution")
        if isinstance(candidate, Mapping):
            latest_distribution = candidate
    logs = metrics.get("logs")
    if latest_distribution is None and isinstance(logs, list):
        for log in reversed(logs):
            if not isinstance(log, Mapping):
                continue
            pool = log.get("opponent_pool")
            if not isinstance(pool, Mapping):
                continue
            candidate = pool.get("normalized_distribution")
            if isinstance(candidate, Mapping):
                latest_distribution = candidate
                break
    if latest_distribution is not None:
        if latest_distribution.get("snapshot_sha256") != snapshot.snapshot_sha256:
            raise ValueError(
                "checkpoint opponent pool does not match its latest rollout CDF"
            )
        distribution = dict(latest_distribution)
        distribution["count_scope"] = "latest-checkpoint-update"
    else:
        matching_draws = [
            draw for draw in draws if draw in {item.name for item in snapshot.items}
        ]
        distribution = snapshot.as_dict(matching_draws)
        distribution["count_scope"] = "matching-metrics-records"
    usage = {
        "count_scope": "all-training-records-in-checkpoint-metrics",
        "actual_counts": dict(sorted(counts.items())),
        "draws": len(draws),
    }
    return distribution, usage


def save_ppo_checkpoint(  # noqa: PLR0913 - provenance fields are explicit
    model: PolicyValueNetwork,
    path: Path,
    *,
    update: int,
    config: PPOConfig,
    source_bc: str,
    opponent_pool: Sequence[OpponentPoolEntry],
    metrics: dict[str, Any],
    formal_protocol: Mapping[str, Any] | None = None,
) -> None:
    """Save policy, critic, normalizer, source BC and opponent provenance."""
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    pool_distribution, pool_usage = _checkpoint_pool_distribution(
        opponent_pool,
        metrics,
    )
    payload: dict[str, Any] = {
        "model_type": "imitation_ppo_policy_value",
        "model_state_dict": state_dict,
        "update": update,
        "value_mode": model.value_mode,
        "config": {
            **asdict(config),
            "hidden_layers": list(config.hidden_layers),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "value_mode": model.value_mode,
            "normalizer_fitted": model.normalizer.fitted,
        },
        "source_bc": source_bc,
        "normalizer_source": source_bc,
        "initialization": config.initialization,
        "opponent_pool": [
            {
                "name": entry.name,
                "weight": entry.weight,
                "snapshot": entry.candidate.snapshot,
            }
            for entry in opponent_pool
        ],
        "opponent_pool_distribution": pool_distribution,
        "opponent_pool_usage": pool_usage,
        "metrics": metrics,
        "runtime": RuntimeSnapshot.collect(config.device_name).as_dict(),
    }
    if formal_protocol is not None:
        payload["formal_protocol"] = dict(formal_protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_ppo_checkpoint(
    path: Path,
    *,
    device_name: DeviceName = "cpu",
) -> PolicyValueNetwork:
    """Load and validate an imitation-PPO policy/value checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "imitation_ppo_policy_value":
        raise ValueError("checkpoint is not an imitation PPO policy/value model")
    config = checkpoint.get("config") or {}
    # Checkpoints written before value_mode existed used the tanh outcome
    # critic.  Preserve that behavior explicitly instead of silently changing
    # the meaning of their value head on load.
    value_mode = _validate_value_mode(
        str(config.get("value_mode", checkpoint.get("value_mode", "outcome")))
    )
    model = PolicyValueNetwork(
        int(config["input_dim"]),
        feature_version=str(config["feature_version"]),
        hidden_layers=tuple(config["hidden_layers"]),
        output_dim=int(config["output_dim"]),
        value_mode=value_mode,
        critic_hidden_dim=int(config.get("critic_hidden_dim", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.normalizer.fitted = bool(config.get("normalizer_fitted", False))
    return model.to(resolve_device(device_name)).eval()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an incremental run artifact with JSON-safe fallback formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validation_score(validation: dict[str, Any]) -> tuple[int, int]:
    """Return exact integer W/D/L coordinates after fail-closed auditing."""
    if not validation:
        raise ValueError("validation matrix is empty")
    reports = list(validation.values())
    scheduled_counts: list[int] = []
    for report in reports:
        try:
            scheduled = int(report["games"])
            completed = int(report["completed_games"])
            failed = int(report["failed_games"])
            candidate_illegal = int(report["candidate_illegal_actions"])
            opponent_illegal = int(report["opponent_illegal_actions"])
            wins = int(report["wins"])
            draws = int(report["draws"])
            losses = int(report["losses"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("validation report is missing audit fields") from exc
        if scheduled <= 0:
            raise ValueError("validation report has zero scheduled games")
        if completed != scheduled or failed != 0:
            raise ValueError("validation report contains failed or incomplete games")
        if candidate_illegal != 0 or opponent_illegal != 0:
            raise ValueError("validation report contains illegal actions")
        if wins < 0 or draws < 0 or losses < 0 or wins + draws + losses != scheduled:
            raise ValueError("validation W/D/L counts do not match scheduled games")
        scheduled_counts.append(scheduled)
    if len(set(scheduled_counts)) != 1:
        raise ValueError("validation reports have unequal scheduled game counts")
    wins = sum(int(report["wins"]) for report in reports)
    games = sum(int(report["games"]) for report in reports)
    return wins, -games


def _pool_metadata(
    fixed_entries: Sequence[OpponentPoolEntry],
    history_entries: Sequence[OpponentPoolEntry],
    current_entry: OpponentPoolEntry,
    config: PPOConfig,
    training_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Describe pool bucket mass so redistribution is auditable."""
    retained_history = (
        list(history_entries[-config.history_limit :]) if config.history_limit else []
    )
    redistributed = bool(config.history_weight > 0 and not retained_history)
    current_mass = config.current_weight + (
        config.history_weight if redistributed else 0.0
    )
    history_mass = config.history_weight if retained_history else 0.0
    effective_entries = build_opponent_pool(
        fixed_entries,
        history_entries,
        current_entry,
        current_weight=config.current_weight,
        history_weight=config.history_weight,
        history_limit=config.history_limit,
    )
    draws = [
        str(record["opponent"])
        for record in training_records
        if record.get("opponent") is not None
    ]
    return {
        "fixed": [
            {"name": entry.name, "weight": entry.weight} for entry in fixed_entries
        ],
        "current": {"name": current_entry.name, "weight": current_mass},
        "history": [
            {"name": entry.name, "weight": history_mass / len(retained_history)}
            for entry in retained_history
        ],
        "current_weight_requested": config.current_weight,
        "history_weight_requested": config.history_weight,
        "history_weight_allocated": history_mass,
        "history_weight_redistributed_to_current": redistributed,
        "history_limit": config.history_limit,
        "total_weight": float(
            sum(entry.weight for entry in fixed_entries) + current_mass + history_mass
        ),
        "normalized_distribution": WeightedPoolSnapshot.from_entries(
            effective_entries
        ).as_dict(draws),
    }


def _build_initial_model(
    bc_model: BehaviorCloningNetwork, config: PPOConfig
) -> PolicyValueNetwork:
    """Build the PPO model from the BC initializer (roadmap E1).

    Same schema: the established from_bc path (including normalizer transfer).
    Cross-schema with ``scratch`` initialization: build directly at the
    target schema with an unfitted (identity) normalizer - the only way to
    cold-start 3/4-seat training before a multi-seat BC exists.  Any other
    cross-schema combination stays a hard error.
    """
    if bc_model.feature_version == config.feature_version:
        return PolicyValueNetwork.from_bc(
            bc_model,
            initialization=config.initialization,
            value_mode=config.value_mode,
            critic_hidden_dim=config.critic_hidden_dim,
        )
    if config.initialization != "scratch":
        raise ValueError(
            f"BC checkpoint schema {bc_model.feature_version!r} differs from "
            f"config schema {config.feature_version!r}; cross-schema starts "
            "require initialization='scratch'"
        )
    return PolicyValueNetwork(
        observation_dim(config.feature_version),
        feature_version=config.feature_version,
        hidden_layers=config.hidden_layers,
        value_mode=config.value_mode,
        critic_hidden_dim=config.critic_hidden_dim,
    )


def _require_formal_training_manifest(  # noqa: C901,PLR0912 - lifecycle gate
    source_manifest: str | None,
    formal_spec: FormalTrainingSpec,
    formal_seed_roll: SeedRollArtifact | None = None,
) -> None:
    """Fail closed unless a matching v2 manifest passed approval and is running."""
    if source_manifest is None:
        raise RuntimeError("formal PPO training requires a running manifest v2")
    from .manifest import MANIFEST_SCHEMA_V2, load_manifest  # noqa: PLC0415

    manifest = load_manifest(Path(source_manifest))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_V2:
        raise RuntimeError("formal PPO training requires manifest schema v2")
    if manifest.get("status") != "running":
        raise RuntimeError(
            "formal PPO training requires approved->running manifest state"
        )
    declaration = manifest.get("declaration")
    if not isinstance(declaration, dict):
        raise RuntimeError("formal manifest declaration is missing")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("formal manifest provenance is missing")
    code = provenance.get("code")
    if not isinstance(code, Mapping):
        raise RuntimeError("formal manifest code provenance is missing")
    require_reproducible_code(code)
    repository = code.get("repository")
    if type(repository) is not str or capture_code_revision(Path(repository)) != dict(
        code
    ):
        raise RuntimeError(
            "formal PPO code changed after manifest provenance was captured"
        )
    if formal_spec.seed_roll_payload_sha256 is not None:
        dependencies = provenance.get("dependencies")
        if not isinstance(dependencies, Mapping) or capture_dependency_provenance(
            Path(repository)
        ) != dict(dependencies):
            raise RuntimeError(
                "formal PPO dependency declarations changed after manifest capture"
            )
        declared_runtime = provenance.get("runtime")
        if not isinstance(declared_runtime, Mapping):
            raise RuntimeError("formal PPO runtime provenance is missing")
        requested_device = declared_runtime.get("requested_device")
        if requested_device not in {"cpu", "cuda"}:
            raise RuntimeError("formal PPO runtime requested_device is invalid")
        current_runtime = RuntimeSnapshot.collect(cast(str, requested_device)).as_dict()
        static_runtime_fields = (
            "python_version",
            "python_executable",
            "platform",
            "numpy_version",
            "torch_version",
            "torch_cuda_version",
            "cuda_available",
            "cuda_device_count",
            "cuda_devices",
            "requested_device",
            "resolved_device",
            "cpu_count",
            "torch_num_threads",
            "torch_num_interop_threads",
            "cublas_workspace_config",
            "python_hash_seed",
            "python_hash_randomization",
            "cuda_driver_version",
        )
        declared_static = {
            field: declared_runtime.get(field) for field in static_runtime_fields
        }
        current_static = {
            field: current_runtime.get(field) for field in static_runtime_fields
        }
        if sha256_canonical_json(declared_static) != sha256_canonical_json(
            current_static
        ):
            raise RuntimeError(
                "formal PPO runtime changed after manifest provenance was captured"
            )
    if declaration.get("experiment_id") != formal_spec.experiment_id:
        raise RuntimeError("formal manifest experiment_id does not match schedule")
    if declaration.get("phase") != formal_spec.phase:
        raise RuntimeError("formal manifest phase does not match schedule")
    if declaration.get("formal_training") != formal_spec.manifest_binding():
        raise RuntimeError("formal manifest training binding does not match schedule")
    _require_formal_seed_roll_manifest(
        declaration,
        formal_spec,
        formal_seed_roll,
    )


def _require_formal_seed_roll_manifest(
    declaration: Mapping[str, Any],
    formal_spec: FormalTrainingSpec,
    formal_seed_roll: SeedRollArtifact | None,
) -> None:
    """Cross-check the optional v2 roll after the ordinary manifest gate."""
    raw_seed_roll = declaration.get("seed_plan", {}).get("seed_roll")
    if formal_spec.seed_roll_payload_sha256 is None:
        if formal_spec.phase == "T1.4":
            raise RuntimeError(
                "T1.4 formal training requires paired-training-v2 and an "
                "approved seed roll"
            )
        if formal_seed_roll is not None or raw_seed_roll is not None:
            raise RuntimeError("paired-training-v1 cannot consume a seed roll")
        return
    if formal_seed_roll is None:
        raise RuntimeError("paired-training-v2 requires a seed-roll artifact")
    require_seed_roll_binding(formal_seed_roll, raw_seed_roll)
    schedule_replicates = tuple(
        sorted({row.replicate_id for row in formal_spec.schedule})
    )
    expected_replicates = (
        formal_seed_roll.plan.pilot_replicate_ids
        if formal_spec.replicate_stage == "pilot"
        else formal_seed_roll.plan.confirmatory_reserve_replicate_ids
    )
    if schedule_replicates != expected_replicates:
        raise RuntimeError(
            "formal PPO schedule does not match its pre-registered replicate stage"
        )
    if formal_spec.replicate_stage == "confirmatory-reserve":
        assert formal_spec.activation_artifact_path is not None
        assert formal_spec.activation_artifact_sha256 is not None
        activation = load_confirmatory_activation_artifact(
            Path(formal_spec.activation_artifact_path),
            expected_experiment_id=formal_seed_roll.plan.experiment_id,
            expected_phase=formal_seed_roll.plan.phase,
            expected_seed_roll_payload_sha256=formal_seed_roll.payload_sha256,
            expected_pilot_replicate_ids=formal_seed_roll.plan.pilot_replicate_ids,
            expected_confirmatory_replicate_ids=(
                formal_seed_roll.plan.confirmatory_reserve_replicate_ids
            ),
        )
        if activation.artifact_sha256 != formal_spec.activation_artifact_sha256:
            raise RuntimeError(
                "formal PPO confirmatory activation artifact hash changed"
            )
    reserved = formal_seed_roll.plan.replicate(formal_spec.replicate_id)
    effective = formal_spec.model_init_lineage()
    if (
        effective.seed63 != reserved.model_init_lineage.seed63
        or effective.digest_hex != reserved.model_init_lineage.digest_hex
    ):
        raise RuntimeError("formal PPO model-init lineage does not match the seed roll")


def train_ppo_selfplay(  # noqa: C901, PLR0912, PLR0913, PLR0915 - lifecycle is explicit
    initial_bc: Path,
    output_dir: Path,
    training_seeds: Sequence[int],
    opponent_pool: Sequence[OpponentPoolEntry],
    *,
    config: PPOConfig,
    validation_seeds: Sequence[int] | None = None,
    validation_opponents: Sequence[CandidateSpec] = (),
    source_manifest: str | None = None,
    formal_spec: FormalTrainingSpec | None = None,
    formal_scenario_bank: ScenarioBank | None = None,
    formal_seed_roll: SeedRollArtifact | None = None,
    formal_opponent_specs: Sequence[FormalOpponentSpec] | None = None,
) -> dict[str, Any]:
    """Train PPO from BC and select only by fixed validation opponents.

    ``status.json`` and a partial ``result.json`` are refreshed after every
    completed update.  The final result keeps the complete update-0-through-N
    log, including validation and pool provenance.
    """
    formal_rows = None
    formal_runtime: dict[str, Any] | None = None
    formal_protocol: dict[str, Any] | None = None
    formal_scenarios: dict[str, ScenarioV1] = {}
    if formal_spec is None:
        if not training_seeds or len(set(training_seeds)) != len(training_seeds):
            raise ValueError("PPO training seeds must be nonempty and unique")
        if formal_scenario_bank is not None:
            raise ValueError(
                "legacy PPO training cannot receive a formal scenario bank"
            )
        if formal_seed_roll is not None:
            raise ValueError("legacy PPO training cannot receive a seed roll")
        if formal_opponent_specs is not None:
            raise ValueError("legacy PPO training cannot receive formal opponent specs")
    else:
        if training_seeds:
            raise ValueError("formal PPO training forbids legacy training_seeds")
        if config.n_seats != MIN_SEATS:
            raise ValueError("formal Task-1 PPO training requires n_seats=2")
        if validation_seeds:
            raise ValueError(
                "formal PPO validation requires ScenarioV1; integer seeds are forbidden"
            )
        formal_rows = formal_spec.selected_rows(
            updates=config.updates,
            games_per_update=config.games_per_update,
        )
        if formal_seed_roll is None:
            # Preserve the established two-argument seam used by legacy formal
            # tests and callers; rolled v2 runs enter the explicit third axis.
            _require_formal_training_manifest(source_manifest, formal_spec)
        else:
            _require_formal_training_manifest(
                source_manifest,
                formal_spec,
                formal_seed_roll,
            )
        if formal_scenario_bank is None:
            raise ValueError("formal PPO training requires a ScenarioV1 bank")
        if formal_scenario_bank.logical_split not in {"train-schedule", "ci-fixture"}:
            raise ValueError("formal PPO training requires the train-schedule bank")
        if (
            formal_spec.seed_roll_payload_sha256 is None
            and formal_scenario_bank.selection_kind == "natural-deal-srswor"
        ):
            raise ValueError(
                "paired-training-v1 cannot consume a seed-rolled scenario bank"
            )
        if formal_spec.scenario_bank_sha256 is None:
            raise ValueError("formal training spec must bind a scenario-bank SHA-256")
        if formal_scenario_bank.payload_sha256 != formal_spec.scenario_bank_sha256:
            raise ValueError("formal training scenario-bank SHA-256 mismatch")
        formal_scenarios = scenario_lookup(formal_scenario_bank.scenarios)
        scheduled_ids = {row.scenario_id for row in formal_rows}
        missing_scenarios = sorted(scheduled_ids - set(formal_scenarios))
        if missing_scenarios:
            raise ValueError(
                "formal scenario bank is missing scheduled states: "
                f"{missing_scenarios[:3]}"
            )
        if formal_spec.seed_roll_payload_sha256 is not None:
            if formal_seed_roll is None or not isinstance(
                formal_scenario_bank.selection_design,
                SeedRollSelectionDesign,
            ):
                raise ValueError(
                    "paired-training-v2 requires a natural-deal SRSWOR scenario bank"
                )
            require_task1_formal_seed_roll(formal_seed_roll)
            schedule_replicates = tuple(
                sorted({row.replicate_id for row in formal_spec.schedule})
            )
            if (
                formal_scenario_bank.selection_design.active_replicate_ids
                != schedule_replicates
            ):
                raise ValueError(
                    "paired-training-v2 bank must bind the schedule's exact "
                    "active replicate set"
                )
            validate_seed_rolled_training_schedule(
                formal_seed_roll,
                tuple(formal_scenarios[scenario_id] for scenario_id in scheduled_ids),
                formal_rows,
                (formal_spec.replicate_id,),
                expected_treatments=(formal_spec.treatment_id,),
                selection_design=formal_scenario_bank.selection_design,
                allow_active_replicate_subset=True,
            )
            if source_manifest is None:
                raise RuntimeError("paired-training-v2 requires a source manifest")
            contract = formal_spec.treatment_contract(formal_spec.treatment_id)
            if config.seed != 0:
                raise ValueError(
                    "paired-training-v2 requires config.seed=0 because its "
                    "randomness is root-derived"
                )
            if sha256_file(initial_bc) != contract.initial_checkpoint_sha256:
                raise ValueError(
                    "formal PPO initial checkpoint does not match its treatment contract"
                )
            if formal_ppo_config_sha256(config) != contract.trainer_config_sha256:
                raise ValueError(
                    "formal PPO config does not match its treatment contract"
                )
            if formal_opponent_specs is None or not formal_opponent_specs:
                raise ValueError(
                    "paired-training-v2 requires immutable formal opponent specs"
                )
            if opponent_pool:
                raise ValueError(
                    "paired-training-v2 rebuilds its opponent pool from frozen specs"
                )
            if (
                formal_opponent_pool_sha256(formal_opponent_specs)
                != contract.opponent_pool_sha256
            ):
                raise ValueError(
                    "formal PPO opponent specs do not match their treatment contract"
                )
            _require_formal_output_reservation_for(
                source_manifest=Path(source_manifest),
                formal_spec=formal_spec,
                output_dir=output_dir,
            )
        elif formal_opponent_specs is not None:
            raise ValueError(
                "paired-training-v1 cannot receive v2 formal opponent specs"
            )
        formal_spec.require_worker_runtime()
        formal_runtime = configure_formal_torch_determinism(
            config.device_name
        ).as_dict()
        formal_protocol = formal_spec.as_dict()
        if formal_spec.seed_roll_payload_sha256 is not None:
            assert formal_opponent_specs is not None
            opponent_pool = tuple(
                opponent.build(device_name=config.device_name)
                for opponent in formal_opponent_specs
            )
            if (
                formal_opponent_pool_sha256(formal_opponent_specs)
                != formal_spec.treatment_contract(
                    formal_spec.treatment_id
                ).opponent_pool_sha256
            ):
                raise ValueError("formal PPO opponent checkpoint changed while loading")
    if not opponent_pool:
        raise ValueError("PPO self-play needs a non-empty opponent pool")
    if validation_seeds and not validation_opponents:
        raise ValueError("validation opponents are required with validation seeds")
    if formal_spec is None:
        seed_everything(config.seed)
        device = resolve_device(config.device_name)
        bc_model = load_bc_checkpoint(initial_bc, device_name=config.device_name)
        model = _build_initial_model(bc_model, config).to(device)
    else:
        device = torch.device(config.device_name)
        model_init_lineage = formal_spec.model_init_lineage()
        with isolated_seed(model_init_lineage.seed63):
            bc_model = load_bc_checkpoint(initial_bc, device_name=config.device_name)
            model = _build_initial_model(bc_model, config).to(device)
        if (
            formal_spec.seed_roll_payload_sha256 is not None
            and sha256_file(initial_bc)
            != formal_spec.treatment_contract(
                formal_spec.treatment_id
            ).initial_checkpoint_sha256
        ):
            raise ValueError("formal PPO initial checkpoint changed while loading")
    reference_model: BehaviorCloningNetwork | None = None
    if config.reference_kl_coefficient > 0:
        reference_model = bc_model.to(device).eval()
        reference_model.requires_grad_(False)
    if config.critic_learning_rate is not None:
        critic_parameters = list(model.value_head.parameters())
        critic_ids = {id(parameter) for parameter in critic_parameters}
        shared_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in critic_ids
        ]
        optimizer = optim.Adam(
            [
                {"params": shared_parameters, "lr": config.learning_rate},
                {"params": critic_parameters, "lr": config.critic_learning_rate},
            ]
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    result_path = output_dir / "result.json"
    logs: list[dict[str, Any]] = []
    history: list[OpponentPoolEntry] = []
    last_effective_entries: list[OpponentPoolEntry] = list(opponent_pool)
    best_score: tuple[int, int] | None = None
    best_update: int | None = 0
    best_path = output_dir / "best.pth"
    run_started = time.perf_counter()
    validation_enabled = bool(validation_seeds and validation_opponents)
    formal_row_by_coordinate = (
        {(row.update, row.game_index): row for row in formal_rows}
        if formal_rows is not None
        else {}
    )

    def result_payload(
        status: str,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build both partial and final result artifacts from one source."""
        payload: dict[str, Any] = {
            "status": status,
            "best": str(best_path),
            "final": str(output_dir / "final.pth"),
            "best_update": best_update,
            "best_validation_score": best_score,
            "config": asdict(config),
            "source_bc": str(initial_bc),
            "normalizer_source": str(initial_bc),
            "source_manifest": source_manifest,
            "training_seeds": [int(seed) for seed in training_seeds]
            if formal_spec is None
            else None,
            "formal_protocol": formal_protocol,
            "formal_runtime": formal_runtime,
            "formal_scenario_bank": {
                "logical_split": formal_scenario_bank.logical_split,
                "payload_sha256": formal_scenario_bank.payload_sha256,
                "state_set_sha256": formal_scenario_bank.state_set_sha256,
                "artifact_sha256": formal_scenario_bank.artifact_sha256,
                "artifact_path": str(formal_scenario_bank.artifact_path),
                "scenario_count": formal_scenario_bank.scenario_count,
                "materialized_scenario_count": len(formal_scenario_bank.scenarios),
            }
            if formal_scenario_bank is not None
            else None,
            "validation_seeds": [int(seed) for seed in validation_seeds]
            if validation_seeds
            else None,
            "logs": logs,
            "elapsed_seconds": time.perf_counter() - run_started,
        }
        if error is not None:
            payload["error"] = error
        return payload

    def write_progress(
        status: str,
        update: int,
        *,
        update_seconds: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Persist current status and the complete log accumulated so far."""
        _write_json(
            status_path,
            {
                "status": status,
                "update": update,
                "updates": config.updates,
                "best_update": best_update,
                "best_validation_score": best_score,
                "update_seconds": update_seconds,
                "elapsed_seconds": time.perf_counter() - run_started,
                "error": error,
            },
        )
        _write_json(result_path, result_payload(status, error=error))

    initial_started = time.perf_counter()
    initial_validation: dict[str, Any] | None = None
    if validation_enabled:
        from .evaluation import evaluate_matrix  # noqa: PLC0415

        initial_candidate = build_policy_candidate(
            model,
            name="ppo-initial",
            snapshot=str(output_dir / "initial.pth"),
            device_name=config.device_name,
        )
        initial_validation = evaluate_matrix(
            [initial_candidate], validation_opponents, validation_seeds or ()
        )[initial_candidate.name]
        best_score = _validation_score(initial_validation)
    initial_record: dict[str, Any] = {
        "update": 0,
        "status": "initial",
        "training_records": [],
        "training_games": 0,
        "training_failed_games": 0,
        "teacher_queries": 0,
        "opponent_names": [],
        "opponent_pool": None,
        "update_metrics": None,
        "critic_warmup": None,
        "validation": initial_validation,
        "formal_protocol": formal_protocol,
        "elapsed_seconds": time.perf_counter() - initial_started,
    }
    logs.append(initial_record)
    initial_path = output_dir / "initial.pth"
    save_ppo_checkpoint(
        model,
        initial_path,
        update=0,
        config=config,
        source_bc=str(initial_bc),
        opponent_pool=opponent_pool,
        metrics=initial_record,
        formal_protocol=formal_protocol,
    )
    save_ppo_checkpoint(
        model,
        best_path,
        update=0,
        config=config,
        source_bc=str(initial_bc),
        opponent_pool=opponent_pool,
        metrics=initial_record,
        formal_protocol=formal_protocol,
    )
    write_progress("running", 0, update_seconds=initial_record["elapsed_seconds"])

    for update in range(1, config.updates + 1):
        update_started = time.perf_counter()
        previous_snapshot = (
            output_dir / "initial.pth"
            if update == 1
            else output_dir / f"update-{update - 1}.pth"
        )
        current = build_policy_candidate(
            model,
            name=f"current-update-{update}",
            snapshot=str(previous_snapshot),
            device_name=config.device_name,
        )
        current_entry = OpponentPoolEntry(current.name, current)
        entries = build_opponent_pool(
            opponent_pool,
            history,
            current_entry,
            current_weight=config.current_weight,
            history_weight=config.history_weight,
            history_limit=config.history_limit,
        )
        last_effective_entries = list(entries)
        transitions: list[PPOTransition] = []
        records: list[dict[str, Any]] = []
        rollout_started = time.perf_counter()
        for game_index in range(config.games_per_update):
            formal_game_rng: FormalGameRng | None = None
            game_scenario: ScenarioV1 | None = None
            if formal_spec is None:
                seed = int(
                    training_seeds[
                        ((update - 1) * config.games_per_update + game_index)
                        % len(training_seeds)
                    ]
                )
                seat = (update + game_index) % 2
            else:
                schedule_row = formal_row_by_coordinate[(update, game_index)]
                seat = schedule_row.seat
                formal_game_rng = schedule_row.game_rng()
                game_scenario = formal_scenarios[formal_game_rng.scenario_id]
                seed = game_scenario.source_seed
            try:
                if formal_game_rng is None:
                    record, game_transitions = collect_ppo_game(
                        model,
                        entries,
                        seed=seed,
                        seat=seat,
                        config=config,
                        update_index=update,
                        game_index=game_index,
                    )
                else:
                    assert game_scenario is not None
                    record, game_transitions = collect_ppo_game(
                        model,
                        entries,
                        seed=seed,
                        seat=seat,
                        config=config,
                        update_index=update,
                        game_index=game_index,
                        formal_rng=formal_game_rng,
                        scenario=game_scenario,
                    )
            except Exception as exc:  # preserve a failed rollout in the log
                record = {
                    "update": update,
                    "game_index": game_index,
                    "seed": seed,
                    "seat": seat,
                    "status": "failed",
                    "failure": {
                        "side": "rollout",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "elapsed_seconds": time.perf_counter() - rollout_started,
                }
                if formal_game_rng is not None:
                    assert game_scenario is not None
                    record.update(
                        {
                            "experiment_id": formal_game_rng.experiment_id,
                            "phase": formal_game_rng.phase,
                            "replicate_id": formal_game_rng.replicate_id,
                            "treatment_id": formal_game_rng.treatment_id,
                            "scenario_id": formal_game_rng.scenario_id,
                            "canonical_state_sha256": (
                                game_scenario.canonical_state_sha256
                            ),
                            "scenario_source_segment": game_scenario.source_segment,
                            "scenario_source_seed": game_scenario.source_seed,
                            "coupling_group": formal_game_rng.coupling_group,
                            "rng_lineage": {
                                "scenario_source": schedule_row.scenario_source.as_dict(),
                                "pool_draw": schedule_row.pool_draw.as_dict(),
                            },
                        }
                    )
                game_transitions = []
            records.append(record)
            transitions.extend(game_transitions)
        rollout_seconds = time.perf_counter() - rollout_started
        failed_games = [
            record for record in records if record.get("status") != "completed"
        ]
        if failed_games or not transitions:
            failure_message = (
                f"PPO update {update} collected an incomplete rollout; "
                f"failed_games={len(failed_games)}, transitions={len(transitions)}"
            )
            failed_record: dict[str, Any] = {
                "update": update,
                "status": "failed",
                "training_records": records,
                "training_games": len(records),
                "training_failed_games": len(failed_games),
                "teacher_queries": 0,
                "opponent_names": sorted(
                    {
                        str(record["opponent"])
                        for record in records
                        if "opponent" in record
                    }
                ),
                "opponent_pool": _pool_metadata(
                    opponent_pool, history, current_entry, config, records
                ),
                "update_metrics": None,
                "critic_warmup": None,
                "validation": None,
                "rollout_seconds": rollout_seconds,
                "elapsed_seconds": time.perf_counter() - update_started,
                "error": failure_message,
            }
            logs.append(failed_record)
            write_progress(
                "failed",
                update,
                update_seconds=failed_record["elapsed_seconds"],
                error=failure_message,
            )
            raise RuntimeError(failure_message)

        warmup_metrics: dict[str, float | int] | None = None
        if update == 1 and config.critic_warmup_epochs:
            warmup_metrics = warmup_critic(model, transitions, config)
        if formal_spec is None:
            update_metrics = ppo_update(
                model,
                optimizer,
                transitions,
                config,
                update_seed=config.seed + update,
                reference_model=reference_model,
            )
        else:
            update_metrics = ppo_update(
                model,
                optimizer,
                transitions,
                config,
                minibatch_key=formal_spec.minibatch_key(update),
                reference_model=reference_model,
            )
        validation: dict[str, Any] | None = None
        should_evaluate = validation_enabled and (
            update % config.eval_every == 0 or update == config.updates
        )
        if should_evaluate:
            from .evaluation import evaluate_matrix  # noqa: PLC0415

            candidate = build_policy_candidate(
                model,
                name="ppo-current",
                snapshot=str(output_dir / f"update-{update}.pth"),
                device_name=config.device_name,
            )
            validation = evaluate_matrix(
                [candidate], validation_opponents, validation_seeds or ()
            )[candidate.name]
        update_record: dict[str, Any] = {
            "update": update,
            "status": "completed",
            "training_records": records,
            "training_games": len(records),
            "training_failed_games": 0,
            "teacher_queries": 0,
            "opponent_names": sorted({record["opponent"] for record in records}),
            "opponent_pool": _pool_metadata(
                opponent_pool, history, current_entry, config, records
            ),
            "update_metrics": update_metrics,
            "critic_warmup": warmup_metrics,
            "validation": validation,
            "rollout_seconds": rollout_seconds,
            "elapsed_seconds": time.perf_counter() - update_started,
        }
        logs.append(update_record)
        update_path = output_dir / f"update-{update}.pth"
        save_ppo_checkpoint(
            model,
            update_path,
            update=update,
            config=config,
            source_bc=str(initial_bc),
            opponent_pool=entries,
            metrics=update_record,
            formal_protocol=formal_protocol,
        )
        if validation is not None:
            score = _validation_score(validation)
            # Validation games are fixed by the manifest.  Compare exact wins
            # only, keeping the earliest checkpoint on a tie.
            if best_score is None or score[0] > best_score[0]:
                best_score = score
                best_update = update
                save_ppo_checkpoint(
                    model,
                    best_path,
                    update=update,
                    config=config,
                    source_bc=str(initial_bc),
                    opponent_pool=entries,
                    metrics=update_record,
                    formal_protocol=formal_protocol,
                )
        history.append(
            OpponentPoolEntry(
                name=f"history-update-{update}",
                candidate=build_policy_candidate(
                    model,
                    name=f"history-update-{update}",
                    snapshot=str(update_path),
                    device_name=config.device_name,
                ),
            )
        )
        if config.history_limit:
            history = history[-config.history_limit :]
        else:
            history = []
        write_progress(
            "running",
            update,
            update_seconds=update_record["elapsed_seconds"],
        )

    final_path = output_dir / "final.pth"
    save_ppo_checkpoint(
        model,
        final_path,
        update=config.updates,
        config=config,
        source_bc=str(initial_bc),
        opponent_pool=last_effective_entries,
        metrics={"logs": logs},
        formal_protocol=formal_protocol,
    )
    result = result_payload("completed")
    _write_json(
        status_path,
        {
            "status": "completed",
            "update": config.updates,
            "updates": config.updates,
            "best_update": best_update,
            "best_validation_score": best_score,
            "update_seconds": logs[-1].get("elapsed_seconds", 0.0),
            "elapsed_seconds": time.perf_counter() - run_started,
            "error": None,
        },
    )
    _write_json(result_path, result)
    return result


def _run_formal_ppo_training_job(job: FormalPPOTrainingJob) -> dict[str, Any]:
    """Rebuild one pool and run one optimizer entirely inside a spawn worker."""
    if job.formal_spec.seed_roll_payload_sha256 is not None:
        _require_formal_output_reservation(job)
        contract = job.formal_spec.treatment_contract(job.formal_spec.treatment_id)
        if sha256_file(job.initial_bc) != contract.initial_checkpoint_sha256:
            raise ValueError("formal PPO initial checkpoint changed before worker")
        if formal_ppo_config_sha256(job.config) != contract.trainer_config_sha256:
            raise ValueError("formal PPO config changed before worker")
        if (
            formal_opponent_pool_sha256(job.opponent_pool)
            != contract.opponent_pool_sha256
        ):
            raise ValueError("formal PPO opponent pool changed before worker execution")
    is_v2 = job.formal_spec.seed_roll_payload_sha256 is not None
    opponent_pool = (
        []
        if is_v2
        else [
            opponent.build(device_name=job.config.device_name)
            for opponent in job.opponent_pool
        ]
    )
    selected_rows = job.formal_spec.selected_rows(
        updates=job.config.updates,
        games_per_update=job.config.games_per_update,
    )
    scenario_bank = load_scenario_bank_subset(
        job.scenario_bank_path,
        {row.scenario_id for row in selected_rows},
    )
    seed_roll = (
        load_seed_roll_artifact(job.seed_roll_path)
        if job.seed_roll_path is not None
        else None
    )
    return train_ppo_selfplay(
        job.initial_bc,
        job.output_dir,
        (),
        opponent_pool,
        config=job.config,
        source_manifest=str(job.source_manifest),
        formal_spec=job.formal_spec,
        formal_scenario_bank=scenario_bank,
        formal_seed_roll=seed_roll,
        formal_opponent_specs=job.opponent_pool if is_v2 else None,
    )


def run_formal_ppo_training_jobs(  # noqa: C901,PLR0912 - complete matrix gate
    jobs: Sequence[FormalPPOTrainingJob],
    *,
    worker_count: int,
) -> tuple[dict[str, Any], ...]:
    """Run a paired treatment/replicate matrix through spawn workers.

    Every job freezes the same treatment-neutral schedule binding while
    selecting its own treatment and replicate. Candidate factories are rebuilt
    in the child, so no closure or live neural module crosses the process
    boundary.
    """
    if not jobs:
        raise ValueError("formal PPO training job matrix must not be empty")
    if any(job.formal_spec.worker_count != worker_count for job in jobs):
        raise ValueError("formal PPO jobs do not match executor worker_count")
    if any(
        job.formal_spec.seed_roll_payload_sha256 is not None
        and (
            sha256_file(job.initial_bc)
            != job.formal_spec.treatment_contract(
                job.formal_spec.treatment_id
            ).initial_checkpoint_sha256
            or formal_ppo_config_sha256(job.config)
            != job.formal_spec.treatment_contract(
                job.formal_spec.treatment_id
            ).trainer_config_sha256
            or formal_opponent_pool_sha256(job.opponent_pool)
            != job.formal_spec.treatment_contract(
                job.formal_spec.treatment_id
            ).opponent_pool_sha256
        )
        for job in jobs
    ):
        raise ValueError("formal PPO treatment inputs changed before spawn")
    coordinates = [
        (job.formal_spec.replicate_id, job.formal_spec.treatment_id) for job in jobs
    ]
    first_spec = jobs[0].formal_spec
    expected_coordinates = {
        (replicate_id, treatment_id)
        for replicate_id in sorted({row.replicate_id for row in first_spec.schedule})
        for treatment_id in first_spec.expected_treatments
    }
    if (
        len(coordinates) != len(set(coordinates))
        or set(coordinates) != expected_coordinates
    ):
        raise ValueError(
            "formal PPO jobs must cover the exact treatment x replicate matrix"
        )
    bank_manifests = [inspect_scenario_bank(job.scenario_bank_path) for job in jobs]
    if any(
        bank["payload_sha256"] != job.formal_spec.scenario_bank_sha256
        for job, bank in zip(jobs, bank_manifests, strict=True)
    ):
        raise ValueError("formal PPO job scenario bank does not match its spec")
    bindings = {
        sha256_canonical_json(job.formal_spec.manifest_binding()) for job in jobs
    }
    if len(bindings) != 1:
        raise ValueError("formal PPO jobs do not share one manifest binding")
    if first_spec.seed_roll_payload_sha256 is not None:
        require_formal_spawn_context()
        roll_paths = {job.seed_roll_path for job in jobs}
        if None in roll_paths or len(roll_paths) != 1:
            raise ValueError("formal PPO jobs must share one seed-roll artifact")
        roll = load_seed_roll_artifact(cast(Path, next(iter(roll_paths))))
        require_task1_formal_seed_roll(roll)
        if (
            roll.payload_sha256 != first_spec.seed_roll_payload_sha256
            or roll.plan.randomization_root_sha256
            != first_spec.randomization_root_sha256
        ):
            raise ValueError("formal PPO jobs do not match their seed roll")
        bank_paths = {job.scenario_bank_path.resolve() for job in jobs}
        if len(bank_paths) != 1:
            raise ValueError(
                "paired-training-v2 jobs must share one physical scenario bank"
            )
        audit_seed_roll_scenario_bank(
            next(iter(bank_paths)),
            roll,
            tuple(sorted({replicate_id for replicate_id, _ in coordinates})),
            schedule=first_spec.schedule,
            expected_treatments=first_spec.expected_treatments,
        )
        manifest_paths = {job.source_manifest.resolve() for job in jobs}
        if len(manifest_paths) != 1:
            raise ValueError("paired-training-v2 jobs must share one manifest file")
        for job in jobs:
            _require_formal_training_manifest(
                str(job.source_manifest),
                job.formal_spec,
                roll,
            )
        _reserve_formal_output_dirs(jobs)
    output_dirs = [job.output_dir.resolve() for job in jobs]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("formal PPO jobs must use distinct output directories")
    return run_formal_spawn_jobs(
        jobs,
        worker=_run_formal_ppo_training_job,
        worker_count=worker_count,
    )
