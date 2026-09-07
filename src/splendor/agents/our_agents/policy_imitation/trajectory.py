"""Auditable teacher trajectories and seed-grouped dataset persistence."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.features import observation_dim
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

TRAJECTORY_SCHEMA_VERSION = 1
ACTION_DIM = len(ALL_ACTIONS)


@dataclass(frozen=True)
class TrajectoryStep:
    """One focal-agent decision with all context needed to replay its label."""

    observation: NDArray[np.float32]
    legal_mask: NDArray[np.uint8]
    action_index: int
    deal_seed: int
    seat: int
    ply: int
    step_in_episode: int
    reward_delta: float
    terminal: bool


@dataclass
class TrajectoryDataset:
    """Columnar trajectory data with strict shape and mask validation."""

    observations: NDArray[np.float32]
    legal_masks: NDArray[np.uint8]
    actions: NDArray[np.int64]
    deal_seeds: NDArray[np.int64]
    seats: NDArray[np.int8]
    plies: NDArray[np.int32]
    steps_in_episode: NDArray[np.int32]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.bool_]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_steps(
        cls, steps: list[TrajectoryStep], *, metadata: dict[str, Any]
    ) -> "TrajectoryDataset":
        """Build a dataset without dropping terminal or game-boundary rows."""
        feature_version = str(metadata.get("feature_version", "v1"))
        dimensions = observation_dim(feature_version)
        if not steps:
            empty = np.empty((0, dimensions), dtype=np.float32)
            empty_masks = np.empty((0, ACTION_DIM), dtype=np.uint8)
            return cls(
                empty,
                empty_masks,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.bool_),
                dict(metadata),
            )
        dataset = cls(
            np.stack([step.observation for step in steps]).astype(np.float32),
            np.stack([step.legal_mask for step in steps]).astype(np.uint8),
            np.asarray([step.action_index for step in steps], dtype=np.int64),
            np.asarray([step.deal_seed for step in steps], dtype=np.int64),
            np.asarray([step.seat for step in steps], dtype=np.int8),
            np.asarray([step.ply for step in steps], dtype=np.int32),
            np.asarray([step.step_in_episode for step in steps], dtype=np.int32),
            np.asarray([step.reward_delta for step in steps], dtype=np.float32),
            np.asarray([step.terminal for step in steps], dtype=np.bool_),
            dict(metadata),
        )
        return dataset

    @property
    def size(self) -> int:
        """Number of focal-agent decisions."""
        return int(self.actions.shape[0])

    @property
    def feature_version(self) -> str:
        """Versioned observation schema carried by the dataset."""
        return str(self.metadata.get("feature_version", "v1"))

    @property
    def seed_set(self) -> set[int]:
        """Return unique deal seeds represented by this dataset."""
        return {int(seed) for seed in np.unique(self.deal_seeds)}

    def validate(self) -> None:
        """Reject malformed rows before training can consume them."""
        dimensions = observation_dim(self.feature_version)
        arrays = (
            self.legal_masks,
            self.actions,
            self.deal_seeds,
            self.seats,
            self.plies,
            self.steps_in_episode,
            self.rewards,
            self.terminals,
        )
        if self.observations.ndim != 2 or self.observations.shape[1] != dimensions:  # noqa: PLR2004
            raise ValueError(
                f"observations must have shape (N, {dimensions}), got {self.observations.shape}"
            )
        if self.legal_masks.shape != (self.size, ACTION_DIM):
            raise ValueError(
                f"legal_masks must have shape ({self.size}, {ACTION_DIM}), got {self.legal_masks.shape}"
            )
        if any(array.shape[0] != self.size for array in arrays):
            raise ValueError("trajectory columns have inconsistent lengths")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.rewards).all():
            raise ValueError("trajectory contains non-finite values")
        if not np.isin(self.legal_masks, (0, 1)).all():
            raise ValueError("legal masks must contain only 0/1 values")
        if not np.isin(self.seats, (0, 1)).all():
            raise ValueError("trajectory seats must be 0 or 1")
        if np.any(self.actions < 0) or np.any(self.actions >= ACTION_DIM):
            raise ValueError("trajectory contains an out-of-range action")
        if self.size and not self.legal_masks[np.arange(self.size), self.actions].all():
            raise ValueError("trajectory contains a label outside its legal mask")
        if np.any(self.steps_in_episode < 0) or np.any(self.plies < 0):
            raise ValueError("trajectory positions must be non-negative")

    def select(self, indices: NDArray[np.bool_], split_name: str) -> "TrajectoryDataset":
        """Select rows while retaining all columns and provenance metadata."""
        if indices.shape != (self.size,):
            raise ValueError("selection mask has the wrong shape")
        metadata = {**self.metadata, "split": split_name}
        return TrajectoryDataset(
            self.observations[indices],
            self.legal_masks[indices],
            self.actions[indices],
            self.deal_seeds[indices],
            self.seats[indices],
            self.plies[indices],
            self.steps_in_episode[indices],
            self.rewards[indices],
            self.terminals[indices],
            metadata,
        )

    def content_hash(self) -> str:
        """Hash data columns for manifest references and label replay."""
        digest = hashlib.sha256()
        for array in (
            self.observations,
            self.legal_masks,
            self.actions,
            self.deal_seeds,
            self.seats,
            self.plies,
            self.steps_in_episode,
            self.rewards,
            self.terminals,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        """Save compressed arrays and a JSON sidecar with provenance."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            **self.metadata,
            "samples": self.size,
            "observation_dim": int(self.observations.shape[1]),
            "action_dim": ACTION_DIM,
            "deal_seeds": sorted(self.seed_set),
            "content_hash": self.content_hash(),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        np.savez_compressed(
            path,
            observations=self.observations,
            legal_masks=self.legal_masks,
            actions=self.actions,
            deal_seeds=self.deal_seeds,
            seats=self.seats,
            plies=self.plies,
            steps_in_episode=self.steps_in_episode,
            rewards=self.rewards,
            terminals=self.terminals,
        )
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "TrajectoryDataset":
        """Load arrays and verify the saved content hash."""
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("unsupported trajectory schema version")
        with np.load(path, allow_pickle=False) as arrays:
            dataset = cls(
                arrays["observations"].astype(np.float32),
                arrays["legal_masks"].astype(np.uint8),
                arrays["actions"].astype(np.int64),
                arrays["deal_seeds"].astype(np.int64),
                arrays["seats"].astype(np.int8),
                arrays["plies"].astype(np.int32),
                arrays["steps_in_episode"].astype(np.int32),
                arrays["rewards"].astype(np.float32),
                arrays["terminals"].astype(np.bool_),
                metadata,
            )
        expected_hash = metadata.get("content_hash")
        if expected_hash is not None and dataset.content_hash() != expected_hash:
            raise ValueError("trajectory content hash does not match metadata")
        return dataset


def split_by_seed(
    dataset: TrajectoryDataset,
    seed_groups: dict[str, list[int]],
) -> dict[str, TrajectoryDataset]:
    """Split by whole deal seeds so paired seats and complete games stay together."""
    groups = {name: {int(seed) for seed in seeds} for name, seeds in seed_groups.items()}
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            overlap = groups[name] & groups[other]
            if overlap:
                raise ValueError(f"seed groups overlap: {name} and {other}: {sorted(overlap)}")
    unknown = dataset.seed_set - set().union(*groups.values()) if groups else dataset.seed_set
    if unknown:
        raise ValueError(f"dataset contains seeds absent from split: {sorted(unknown)}")
    return {
        name: dataset.select(np.isin(dataset.deal_seeds, sorted(seeds)), name)
        for name, seeds in groups.items()
    }


def concatenate_datasets(
    datasets: list[TrajectoryDataset],
    *,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryDataset:
    """Concatenate trajectory versions while retaining every provenance record.

    DAgger deliberately revisits the same training seeds across rounds.  Seed
    overlap is therefore allowed here; leakage is still rejected later by
    :func:`split_by_seed`, which keeps validation and final-test groups
    separate from the aggregate.
    """
    if not datasets:
        raise ValueError("cannot concatenate an empty dataset list")
    versions = {dataset.feature_version for dataset in datasets}
    if len(versions) != 1:
        raise ValueError("cannot concatenate different feature schemas")
    base = datasets[0]
    merged_metadata: dict[str, Any] = {
        **base.metadata,
        "feature_version": base.feature_version,
        "source_hashes": [dataset.content_hash() for dataset in datasets],
    }
    if metadata:
        merged_metadata.update(metadata)
    records = [
        record
        for dataset in datasets
        for record in dataset.metadata.get("game_records", [])
    ]
    if records:
        merged_metadata["game_records"] = records
    return TrajectoryDataset(
        np.concatenate([dataset.observations for dataset in datasets]),
        np.concatenate([dataset.legal_masks for dataset in datasets]),
        np.concatenate([dataset.actions for dataset in datasets]),
        np.concatenate([dataset.deal_seeds for dataset in datasets]),
        np.concatenate([dataset.seats for dataset in datasets]),
        np.concatenate([dataset.plies for dataset in datasets]),
        np.concatenate([dataset.steps_in_episode for dataset in datasets]),
        np.concatenate([dataset.rewards for dataset in datasets]),
        np.concatenate([dataset.terminals for dataset in datasets]),
        merged_metadata,
    )
