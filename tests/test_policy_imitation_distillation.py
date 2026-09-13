"""Roadmap D3 tests: DQN value-prior distillation into a BC initializer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.policy_imitation.bc_network import (
    BehaviorCloningNetwork,
)
from splendor.agents.our_agents.policy_imitation.bc_training import (
    load_bc_checkpoint,
)
from splendor.agents.our_agents.policy_imitation.distillation import (
    DistillConfig,
    distill_dqn_teacher,
    teacher_log_probs,
)
from splendor.agents.our_agents.policy_imitation.trajectory import TrajectoryDataset

OBS_DIM = 265
ACTION_DIM = 3510


def _synthetic_dataset(
    n: int, seed: int, feature_version: str = "v1"
) -> TrajectoryDataset:
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(n, OBS_DIM)).astype(np.float32)
    masks = np.zeros((n, ACTION_DIM), dtype=np.uint8)
    for row in masks:
        legal = rng.choice(ACTION_DIM, size=24, replace=False)
        row[legal] = 1
    actions = np.array([int(np.flatnonzero(row)[0]) for row in masks], dtype=np.int64)
    return TrajectoryDataset(
        observations,
        masks,
        actions,
        (np.arange(n, dtype=np.int64) + seed * 1_000_000),
        np.zeros(n, dtype=np.int8),
        np.arange(n, dtype=np.int32),
        np.zeros(n, dtype=np.int32),
        rng.random(n).astype(np.float32),
        np.zeros(n, dtype=np.bool_),
        {"feature_version": feature_version},
    )


def _tiny_teacher(auxiliary: bool, seed: int = 7) -> QNetwork:
    torch.manual_seed(seed)
    teacher = QNetwork(auxiliary_heads=auxiliary)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher.eval()


@pytest.mark.parametrize("mode", ["q-softmax", "policy-head"])
def test_distillation_reduces_kl_and_saves(tmp_path: Path, mode: str) -> None:
    train = _synthetic_dataset(64, seed=1)
    validation = _synthetic_dataset(32, seed=2)
    teacher = _tiny_teacher(auxiliary=(mode == "policy-head"))
    teacher_path = tmp_path / "teacher.pth"
    torch.save(
        {
            "model_state_dict": teacher.state_dict(),
            "config": {"auxiliary_heads": mode != "q-softmax"},
        },
        teacher_path,
    )
    # load_saved_dqn expects the repository checkpoint layout; simpler here to
    # pass the in-memory teacher through a monkeypatched loader.
    import splendor.agents.our_agents.policy_imitation.distillation as dist

    original = dist.load_dqn_template
    dist.load_dqn_template = lambda _path: teacher  # type: ignore[assignment]
    try:
        config = DistillConfig(
            teacher_mode=mode,
            epochs=2,
            batch_size=16,
            seed=825_410,
        )
        result = distill_dqn_teacher(
            teacher_path,
            train,
            validation,
            tmp_path / "out",
            config=config,
        )
    finally:
        dist.load_dqn_template = original  # type: ignore[assignment]
    assert result["best_validation_kl"] is not None
    assert result["logs"][-1]["validation_kl"] >= 0.0
    assert (tmp_path / "out" / "best.pth").is_file()
    assert (tmp_path / "out" / "final.pth").is_file()
    loaded = load_bc_checkpoint(tmp_path / "out" / "best.pth")
    assert isinstance(loaded, BehaviorCloningNetwork)
    assert loaded.input_dim == OBS_DIM


def test_distillation_validates_mode_and_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="teacher mode"):
        DistillConfig(teacher_mode="magic")
    with pytest.raises(ValueError, match="temperature"):
        DistillConfig(temperature=0.0)
    teacher = _tiny_teacher(auxiliary=False)
    teacher_path = tmp_path / "teacher.pth"
    torch.save({"model_state_dict": teacher.state_dict()}, teacher_path)
    import splendor.agents.our_agents.policy_imitation.distillation as dist

    original = dist.load_dqn_template
    dist.load_dqn_template = lambda _path: teacher  # type: ignore[assignment]
    try:
        config = DistillConfig(teacher_mode="policy-head", epochs=1)
        with pytest.raises(ValueError, match="auxiliary heads"):
            distill_dqn_teacher(
                teacher_path,
                _synthetic_dataset(16, seed=3),
                _synthetic_dataset(8, seed=4),
                tmp_path / "out",
                config=config,
            )
    finally:
        dist.load_dqn_template = original  # type: ignore[assignment]


def test_teacher_log_probs_are_masked() -> None:
    teacher = _tiny_teacher(auxiliary=False)
    obs = torch.zeros(2, OBS_DIM)
    mask = torch.zeros(2, ACTION_DIM)
    mask[0, [3, 5, 7]] = 1.0
    mask[1, [11]] = 1.0
    log_probs = teacher_log_probs(teacher, obs, mask, mode="q-softmax", temperature=1.0)
    probabilities = log_probs.exp()
    assert torch.isfinite(probabilities[mask == 0]).all()
    assert probabilities.sum(-1) == pytest.approx(torch.ones(2), abs=1e-5)
    assert probabilities[1, 11] == pytest.approx(1.0, abs=1e-5)
