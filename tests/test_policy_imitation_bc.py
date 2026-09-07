"""Tests for masked behavior-cloning models and training provenance."""

from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.policy_imitation.bc_network import (
    ACTION_DIM,
    BehaviorCloningNetwork,
)
from splendor.agents.our_agents.policy_imitation.bc_training import (
    BCConfig,
    evaluate_bc_model,
    load_bc_checkpoint,
    masked_cross_entropy,
    train_bc,
    validate_bc_splits,
)
from splendor.agents.our_agents.policy_imitation.evaluation import (
    collect_teacher_dataset,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.trajectory import (
    TrajectoryDataset,
    split_by_seed,
)


def _synthetic_dataset(feature_version: str, seed: int) -> TrajectoryDataset:
    dimension = 265 if feature_version == "v1" else 312
    observations = np.zeros((2, dimension), dtype=np.float32)
    masks = np.zeros((2, ACTION_DIM), dtype=np.uint8)
    masks[0, 0] = 1
    masks[0, 1] = 1
    masks[1, 2] = 1
    return TrajectoryDataset(
        observations,
        masks,
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([seed, seed], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int8),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.zeros(2, dtype=np.float32),
        np.asarray([False, True], dtype=np.bool_),
        {"feature_version": feature_version},
    )


def test_masked_cross_entropy_rejects_illegal_label() -> None:
    logits = torch.zeros((1, ACTION_DIM))
    masks = torch.zeros((1, ACTION_DIM))
    masks[0, 0] = 1

    with pytest.raises(ValueError, match="outside its legal"):
        masked_cross_entropy(logits, torch.tensor([1]), masks)


def test_public_feature_schema_has_same_masked_action_head() -> None:
    dataset = _synthetic_dataset("public-v2", 800401)
    model = BehaviorCloningNetwork(
        dataset.observations.shape[1],
        feature_version="public-v2",
        hidden_layers=(8,),
    )
    model.fit_normalizer(torch.from_numpy(dataset.observations))

    logits = model(
        torch.from_numpy(dataset.observations),
        torch.from_numpy(dataset.legal_masks.astype(np.float32)),
    )

    assert logits.shape == (2, ACTION_DIM)
    assert torch.isneginf(logits).sum().item() == 0
    assert torch.all(logits[0, 2:] < -1e8)


def test_bc_training_round_trip_and_test_isolation(tmp_path: Path) -> None:
    candidate = CandidateSpec("random", "teacher_candidate", RandomAgent)
    opponent = CandidateSpec("rival", "fixed_baseline", RandomAgent)
    collected = []
    for seed in (800402, 800403, 800404):
        path = tmp_path / f"teacher-{seed}.npz"
        collect_teacher_dataset(
            candidate,
            opponent,
            [seed],
            path,
            feature_version="v1",
        )
        collected.append(TrajectoryDataset.load(path))
    dataset = TrajectoryDataset(
        np.concatenate([item.observations for item in collected]),
        np.concatenate([item.legal_masks for item in collected]),
        np.concatenate([item.actions for item in collected]),
        np.concatenate([item.deal_seeds for item in collected]),
        np.concatenate([item.seats for item in collected]),
        np.concatenate([item.plies for item in collected]),
        np.concatenate([item.steps_in_episode for item in collected]),
        np.concatenate([item.rewards for item in collected]),
        np.concatenate([item.terminals for item in collected]),
        {"feature_version": "v1"},
    )
    splits = split_by_seed(
        dataset,
        {"train": [800402], "validation": [800403], "test": [800404]},
    )
    config = BCConfig(
        feature_version="v1",
        hidden_layers=(8,),
        batch_size=32,
        epochs=1,
        seed=17,
    )
    result = train_bc(
        splits["train"],
        splits["validation"],
        tmp_path / "bc-run",
        config=config,
        test_data=splits["test"],
        source_manifest="proposal.json",
    )
    assert len(collected[0].metadata["game_records"]) == 2
    model = load_bc_checkpoint(Path(result["best"]))
    metrics = evaluate_bc_model(model, splits["validation"])

    assert result["best_epoch"] == 1
    assert metrics["predicted_legal_rate"] == 1.0
    assert result["test_metrics"]["samples"] == splits["test"].size
    assert model.normalizer.fitted


def test_bc_splits_reject_seed_overlap() -> None:
    dataset = _synthetic_dataset("v1", 800405)

    with pytest.raises(ValueError, match="seed leakage"):
        validate_bc_splits(dataset, dataset)
