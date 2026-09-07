"""Masked behavior-cloning training and checkpoint audit helpers."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from splendor.agents.our_agents.dqn.constants import (
    HIDDEN_DIMS,
    HUGE_NEG,
    MAX_GRADIENT_NORM,
)

from .bc_network import ACTION_DIM, BehaviorCloningNetwork
from .protocol import RuntimeSnapshot, seed_everything
from .trajectory import TrajectoryDataset

DeviceName = Literal["cpu", "cuda", "mps"]
MATRIX_RANK = 2


@dataclass(frozen=True)
class BCConfig:
    """One-factor BC settings; feature version is the only planned variant."""

    feature_version: str = "v1"
    hidden_layers: tuple[int, ...] = HIDDEN_DIMS
    learning_rate: float = 1e-4
    batch_size: int = 256
    epochs: int = 10
    seed: int = 1234
    device_name: DeviceName = "cpu"
    max_grad_norm: float = MAX_GRADIENT_NORM

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def resolve_device(device_name: DeviceName) -> torch.device:
    """Resolve an unavailable accelerator to CPU and keep the choice auditable."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute CE after masking, rejecting an illegal teacher label."""
    if (
        logits.ndim != MATRIX_RANK
        or actions.ndim != 1
        or legal_masks.ndim != MATRIX_RANK
    ):
        raise ValueError("BC logits, actions, and masks have unexpected ranks")
    if logits.shape != legal_masks.shape or logits.shape[0] != actions.shape[0]:
        raise ValueError("BC logits, actions, and masks have incompatible shapes")
    actions = actions.long()
    if torch.any(actions < 0) or torch.any(actions >= logits.shape[1]):
        raise ValueError("BC label is outside the action head")
    if torch.any(legal_masks.sum(dim=1) <= 0):
        raise ValueError("BC row contains no legal actions")
    if not torch.all(legal_masks.gather(1, actions.unsqueeze(1)).squeeze(1) > 0):
        raise ValueError("BC label is outside its legal action mask")
    masked_logits = logits.masked_fill(legal_masks <= 0, HUGE_NEG)
    return F.cross_entropy(masked_logits, actions)


@torch.no_grad()
def evaluate_bc_model(
    model: BehaviorCloningNetwork,
    dataset: TrajectoryDataset,
    *,
    batch_size: int = 512,
) -> dict[str, float | int]:
    """Measure masked loss, action agreement, and state/action coverage."""
    if dataset.feature_version != model.feature_version:
        raise ValueError("dataset and BC model feature schemas differ")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    device = next(model.parameters()).device
    loader = _make_loader(dataset, batch_size, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total = 0
    predicted_legal = 0
    for batch_observations, batch_masks, batch_actions in loader:
        observations = batch_observations.to(device)
        masks = batch_masks.to(device)
        actions = batch_actions.to(device)
        logits = model(observations, masks)
        loss = masked_cross_entropy(logits, actions, masks)
        predictions = logits.argmax(dim=1)
        count = int(actions.shape[0])
        total_loss += float(loss.item()) * count
        total_correct += int((predictions == actions).sum().item())
        predicted_legal += int(masks.gather(1, predictions.unsqueeze(1)).sum().item())
        total += count
    if total == 0:
        return {
            "samples": 0,
            "masked_loss": 0.0,
            "action_agreement": 0.0,
            "predicted_legal_rate": 0.0,
            **coverage_stats(dataset),
        }
    return {
        "samples": total,
        "masked_loss": total_loss / total,
        "action_agreement": total_correct / total,
        "predicted_legal_rate": predicted_legal / total,
        **coverage_stats(dataset),
    }


def coverage_stats(dataset: TrajectoryDataset) -> dict[str, float | int]:
    """Report uncovered action and state groups without using test data."""
    if dataset.size == 0:
        return {
            "unique_teacher_actions": 0,
            "uncovered_actions": ACTION_DIM,
            "teacher_action_coverage": 0.0,
            "unique_deal_seeds": 0,
            "mean_legal_actions": 0.0,
            "min_legal_actions": 0,
            "max_legal_actions": 0,
        }
    unique_actions = int(np.unique(dataset.actions).size)
    legal_counts = dataset.legal_masks.sum(axis=1)
    return {
        "unique_teacher_actions": unique_actions,
        "uncovered_actions": ACTION_DIM - unique_actions,
        "teacher_action_coverage": unique_actions / ACTION_DIM,
        "unique_deal_seeds": len(dataset.seed_set),
        "mean_legal_actions": float(np.mean(legal_counts)),
        "min_legal_actions": int(np.min(legal_counts)),
        "max_legal_actions": int(np.max(legal_counts)),
    }


def validate_bc_splits(
    train_data: TrajectoryDataset,
    validation_data: TrajectoryDataset,
    test_data: TrajectoryDataset | None = None,
) -> None:
    """Require feature agreement and whole-seed separation before training."""
    datasets = [train_data, validation_data]
    if test_data is not None:
        datasets.append(test_data)
    versions = {dataset.feature_version for dataset in datasets}
    if len(versions) != 1:
        raise ValueError("BC splits use different feature schemas")
    for index, first in enumerate(datasets):
        for second in datasets[index + 1 :]:
            overlap = first.seed_set & second.seed_set
            if overlap:
                raise ValueError(f"BC seed leakage detected: {sorted(overlap)}")
    if train_data.size == 0 or validation_data.size == 0:
        raise ValueError("BC training and validation splits must be non-empty")


def _make_loader(
    dataset: TrajectoryDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """Create a loader over only the requested dataset columns."""
    tensors = TensorDataset(
        torch.from_numpy(dataset.observations),
        torch.from_numpy(dataset.legal_masks.astype(np.float32)),
        torch.from_numpy(dataset.actions),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def save_bc_checkpoint(  # noqa: PLR0913 - checkpoint provenance is explicit
    model: BehaviorCloningNetwork,
    path: Path,
    *,
    epoch: int,
    config: BCConfig,
    dataset_metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Save model, preprocessing, exact config, and dataset provenance."""
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload: dict[str, Any] = {
        "model_type": "behavior_cloning",
        "model_state_dict": state_dict,
        "epoch": epoch,
        "config": {
            **asdict(config),
            "hidden_layers": list(config.hidden_layers),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "normalizer_fitted": model.normalizer.fitted,
        },
        "dataset": dataset_metadata,
        "metrics": metrics,
        "runtime": RuntimeSnapshot.collect(config.device_name).as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_bc_checkpoint(
    path: Path,
    *,
    device_name: DeviceName = "cpu",
) -> BehaviorCloningNetwork:
    """Load a BC checkpoint and verify its versioned model metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "behavior_cloning":
        raise ValueError("checkpoint is not a behavior-cloning model")
    config = checkpoint.get("config") or {}
    model = BehaviorCloningNetwork(
        int(config["input_dim"]),
        feature_version=str(config["feature_version"]),
        hidden_layers=tuple(config["hidden_layers"]),
        output_dim=int(config["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.normalizer.fitted = bool(config.get("normalizer_fitted", False))
    return model.to(resolve_device(device_name)).eval()


def train_bc(  # noqa: PLR0913,PLR0915 - training lifecycle is intentionally explicit
    train_data: TrajectoryDataset,
    validation_data: TrajectoryDataset,
    output_dir: Path,
    *,
    config: BCConfig,
    test_data: TrajectoryDataset | None = None,
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Train BC and select the earliest exact minimum validation loss."""
    validate_bc_splits(train_data, validation_data, test_data)
    if train_data.feature_version != config.feature_version:
        raise ValueError("BC config and training dataset feature schemas differ")
    seed_everything(config.seed)
    device = resolve_device(config.device_name)
    model = BehaviorCloningNetwork(
        train_data.observations.shape[1],
        feature_version=config.feature_version,
        hidden_layers=config.hidden_layers,
    ).to(device)
    model.fit_normalizer(torch.from_numpy(train_data.observations))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = _make_loader(
        train_data,
        config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_metadata = {
        "source_manifest": source_manifest,
        "feature_version": config.feature_version,
        "train": train_data.metadata,
        "validation": validation_data.metadata,
        "test": test_data.metadata if test_data is not None else None,
        "train_hash": train_data.content_hash(),
        "validation_hash": validation_data.content_hash(),
        "test_hash": test_data.content_hash() if test_data is not None else None,
    }
    logs: list[dict[str, Any]] = []
    best_loss: float | None = None
    best_epoch: int | None = None
    best_path = output_dir / "best.pth"
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch_observations, batch_masks, batch_actions in loader:
            observations = batch_observations.to(device)
            masks = batch_masks.to(device)
            actions = batch_actions.to(device)
            logits = model(observations, masks)
            loss = masked_cross_entropy(logits, actions, masks)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            count = int(actions.shape[0])
            total_loss += float(loss.item()) * count
            samples += count
        train_metrics = evaluate_bc_model(model, train_data, batch_size=config.batch_size)
        validation_metrics = evaluate_bc_model(
            model, validation_data, batch_size=config.batch_size
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total_loss / samples if samples else 0.0,
            "train_samples": samples,
            "validation_loss": validation_metrics["masked_loss"],
            "train_action_agreement": train_metrics["action_agreement"],
            "validation_action_agreement": validation_metrics["action_agreement"],
            "validation_legal_rate": validation_metrics["predicted_legal_rate"],
        }
        logs.append(row)
        epoch_path = output_dir / f"epoch-{epoch}.pth"
        save_bc_checkpoint(
            model,
            epoch_path,
            epoch=epoch,
            config=config,
            dataset_metadata=dataset_metadata,
            metrics=row,
        )
        validation_loss = float(validation_metrics["masked_loss"])
        if best_loss is None or validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            save_bc_checkpoint(
                model,
                best_path,
                epoch=epoch,
                config=config,
                dataset_metadata=dataset_metadata,
                metrics=row,
            )
    if best_epoch is None or best_loss is None:
        raise RuntimeError("BC produced no validation checkpoint")
    final_path = output_dir / "final.pth"
    save_bc_checkpoint(
        model,
        final_path,
        epoch=config.epochs,
        config=config,
        dataset_metadata=dataset_metadata,
        metrics=logs[-1],
    )
    result: dict[str, Any] = {
        "best": str(best_path),
        "final": str(final_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "feature_version": config.feature_version,
        "config": asdict(config),
        "train_coverage": coverage_stats(train_data),
        "validation_coverage": coverage_stats(validation_data),
        "logs": logs,
    }
    if test_data is not None:
        best_model = load_bc_checkpoint(best_path, device_name=config.device_name)
        result["test_metrics"] = evaluate_bc_model(
            best_model, test_data, batch_size=config.batch_size
        )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
