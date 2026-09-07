"""Policy imitation experiments kept separate from the established agents."""

from .bc_network import BehaviorCloningNetwork
from .bc_training import BCConfig, evaluate_bc_model, load_bc_checkpoint, train_bc
from .dagger import (
    aggregate_dagger_datasets,
    collect_dagger_dataset,
    play_dagger_game,
    summarize_dagger_records,
)
from .manifest import (
    DEFAULT_FORBIDDEN_SEED_RANGES,
    approve_manifest,
    create_manifest,
    load_manifest,
    require_approved,
    validate_manifest,
)
from .policies import (
    CandidateSpec,
    build_bc_candidate,
    build_builtin_candidate,
    build_fixed_baseline,
)
from .ppo_selfplay import (
    OpponentPoolEntry,
    PolicyValueNetwork,
    PPOConfig,
    PPOPolicyAgent,
    PPOTransition,
    build_policy_candidate,
    collect_ppo_game,
    compute_gae,
    load_ppo_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
    train_ppo_selfplay,
)
from .protocol import (
    RuntimeSnapshot,
    capture_code_revision,
    isolated_seed,
    seed_everything,
)
from .trajectory import (
    TrajectoryDataset,
    TrajectoryStep,
    concatenate_datasets,
    split_by_seed,
)

__all__ = [
    "DEFAULT_FORBIDDEN_SEED_RANGES",
    "BCConfig",
    "BehaviorCloningNetwork",
    "CandidateSpec",
    "OpponentPoolEntry",
    "PPOConfig",
    "PPOPolicyAgent",
    "PPOTransition",
    "PolicyValueNetwork",
    "RuntimeSnapshot",
    "TrajectoryDataset",
    "TrajectoryStep",
    "aggregate_dagger_datasets",
    "approve_manifest",
    "build_bc_candidate",
    "build_builtin_candidate",
    "build_fixed_baseline",
    "build_policy_candidate",
    "capture_code_revision",
    "collect_dagger_dataset",
    "collect_ppo_game",
    "compute_gae",
    "concatenate_datasets",
    "create_manifest",
    "evaluate_bc_model",
    "isolated_seed",
    "load_bc_checkpoint",
    "load_manifest",
    "load_ppo_checkpoint",
    "play_dagger_game",
    "ppo_update",
    "require_approved",
    "save_ppo_checkpoint",
    "seed_everything",
    "split_by_seed",
    "summarize_dagger_records",
    "train_bc",
    "train_ppo_selfplay",
    "validate_manifest",
]
