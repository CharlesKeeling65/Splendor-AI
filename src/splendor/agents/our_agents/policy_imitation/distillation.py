"""DQN value-prior distillation into a BC-style PPO initializer (roadmap D3).

The DAgger-2 initializer gives PPO a behaviour-cloned starting policy; the
DQN's off-policy value function encodes a *stronger* prior.  This module
distills the teacher's masked preference into the BC network as soft targets:

* ``q-softmax`` teacher mode - masked Boltzmann over the teacher's Q values,
  ``pi_teacher(a|s) proportional to exp(Q(s,a) / T)`` restricted to legal
  actions.  Works with every existing DQN checkpoint (no auxiliary heads
  required).
* ``policy-head`` teacher mode - the auxiliary ``policy_value`` head
  (:meth:`QNetwork.policy_value`) when the checkpoint was trained with
  auxiliary heads.

The public-information audit of the teacher (results doc §4.4) must run
before distillation; the CLI wires ``information-audit`` upstream of this
module so a clean report is a precondition, not an afterthought.
"""

from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.policy_imitation.bc_network import (
    BehaviorCloningNetwork,
)
from splendor.agents.our_agents.policy_imitation.bc_training import (
    BCConfig,
    DeviceName,
    evaluate_bc_model,
    resolve_device,
    save_bc_checkpoint,
    validate_bc_splits,
)
from splendor.agents.our_agents.policy_imitation.protocol import seed_everything
from splendor.agents.our_agents.policy_imitation.trajectory import TrajectoryDataset

from .dqn_utils import load_dqn_template

TEACHER_MODES = ("q-softmax", "policy-head")


@dataclass(frozen=True)
class DistillConfig:
    """Declared distillation hyper-parameters."""

    feature_version: str = "v1"
    temperature: float = 1.0
    teacher_mode: str = "q-softmax"
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 256
    max_grad_norm: float = 1.0
    seed: int = 42
    device_name: DeviceName = "cpu"

    def __post_init__(self) -> None:
        if self.teacher_mode not in TEACHER_MODES:
            raise ValueError(f"unknown teacher mode {self.teacher_mode!r}")
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")


def _batch_iterator(
    data: TrajectoryDataset, batch_size: int, generator: torch.Generator
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yield (observation, mask) minibatches over one epoch."""
    indices = torch.randperm(data.size, generator=generator)
    for start in range(0, data.size, batch_size):
        batch = indices[start : start + batch_size]
        observations = torch.from_numpy(data.observations[batch]).float()
        masks = torch.from_numpy(data.legal_masks[batch]).float()
        yield observations, masks


def teacher_log_probs(
    teacher: QNetwork,
    observations: torch.Tensor,
    masks: torch.Tensor,
    *,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    """Masked teacher log-probabilities over the global action space."""
    with torch.no_grad():
        if mode == "policy-head":
            logits, _ = teacher.policy_value(observations, masks)
        else:
            q_values = teacher(observations, masks)
            logits = q_values.masked_fill(masks == 0, -torch.inf)
        return F.log_softmax(logits / temperature, dim=-1)


def distill_loss(
    student_logits: torch.Tensor,
    teacher_log_probability: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) restricted to legal actions.

    Illegal-action entries are neutralized *before* the reduction: the
    teacher may carry ``-inf`` log-probabilities there, and
    ``exp(-inf) * (-inf - -inf)`` would poison F.kl_div with NaNs.
    """
    legal = masks > 0
    student_log_probability = F.log_softmax(
        student_logits.masked_fill(~legal, -torch.inf), dim=-1
    )
    teacher = teacher_log_probability.masked_fill(~legal, 0.0)
    kl_terms = torch.exp(teacher) * (teacher - student_log_probability)
    kl_terms = torch.where(legal, kl_terms, torch.zeros_like(kl_terms))
    return kl_terms.sum(-1).mean()


def distill_dqn_teacher(  # noqa: PLR0913 - lifecycle mirrors train_bc
    teacher_path: Path,
    train_data: TrajectoryDataset,
    validation_data: TrajectoryDataset,
    output_dir: Path,
    *,
    config: DistillConfig,
    test_data: TrajectoryDataset | None = None,
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Distill the DQN value prior into a BC checkpoint usable by PPO."""
    validate_bc_splits(train_data, validation_data, test_data)
    if train_data.feature_version != config.feature_version:
        raise ValueError("distill config and dataset feature schemas differ")
    seed_everything(config.seed)
    device = resolve_device(config.device_name)
    teacher = load_dqn_template(teacher_path).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if config.teacher_mode == "policy-head" and not teacher.auxiliary_heads:
        raise ValueError("teacher checkpoint has no policy/value auxiliary heads")

    model = BehaviorCloningNetwork(
        train_data.observations.shape[1],
        feature_version=config.feature_version,
    ).to(device)
    model.fit_normalizer(torch.from_numpy(train_data.observations))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pth"
    final_path = output_dir / "final.pth"
    logs: list[dict[str, Any]] = []
    best_loss: float | None = None
    best_epoch: int | None = None
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch_observations, batch_masks in _batch_iterator(
            train_data, config.batch_size, generator
        ):
            observations = batch_observations.to(device)
            masks = batch_masks.to(device)
            targets = teacher_log_probs(
                teacher,
                observations,
                masks,
                mode=config.teacher_mode,
                temperature=config.temperature,
            )
            logits = model(observations, masks)
            loss = distill_loss(logits, targets, masks)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            count = int(observations.shape[0])
            total_loss += float(loss.item()) * count
            samples += count
        train_kl = total_loss / samples if samples else 0.0
        validation_kl = evaluate_distillation(
            model, teacher, validation_data, config=config
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_kl": train_kl,
            "validation_kl": validation_kl,
            **{
                f"validation_{key}": value
                for key, value in evaluate_bc_model(
                    model, validation_data, batch_size=config.batch_size
                ).items()
            },
        }
        logs.append(row)
        if best_loss is None or validation_kl < best_loss:
            best_loss = validation_kl
            best_epoch = epoch
            save_bc_checkpoint(
                model,
                best_path,
                epoch=epoch,
                config=BCConfig(
                    feature_version=config.feature_version,
                    learning_rate=config.learning_rate,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    max_grad_norm=config.max_grad_norm,
                    seed=config.seed,
                    device_name=config.device_name,
                ),
                dataset_metadata={
                    "source_manifest": source_manifest,
                    "teacher_path": str(teacher_path),
                    "teacher_mode": config.teacher_mode,
                    "temperature": config.temperature,
                    "train_hash": train_data.content_hash(),
                    "validation_hash": validation_data.content_hash(),
                },
                metrics={"validation_kl": validation_kl, "train_kl": train_kl},
            )
    save_bc_checkpoint(
        model,
        final_path,
        epoch=config.epochs,
        config=BCConfig(
            feature_version=config.feature_version,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            device_name=config.device_name,
        ),
        dataset_metadata={
            "source_manifest": source_manifest,
            "teacher_path": str(teacher_path),
            "teacher_mode": config.teacher_mode,
            "temperature": config.temperature,
        },
        metrics={"final_validation_kl": logs[-1]["validation_kl"]},
    )
    return {
        "epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_validation_kl": best_loss,
        "logs": logs,
        "best": str(best_path),
        "final": str(final_path),
        "config": asdict(config),
    }


def evaluate_distillation(
    model: BehaviorCloningNetwork,
    teacher: QNetwork,
    data: TrajectoryDataset,
    *,
    config: DistillConfig,
) -> float:
    """Mean masked KL between student and teacher over the dataset."""
    model.eval()
    observations = torch.from_numpy(data.observations).float()
    masks = torch.from_numpy(data.legal_masks).float()
    total = 0.0
    with torch.no_grad():
        for start in range(0, data.size, config.batch_size):
            batch = slice(start, min(start + config.batch_size, data.size))
            obs = observations[batch]
            mask = masks[batch]
            targets = teacher_log_probs(
                teacher,
                obs,
                mask,
                mode=config.teacher_mode,
                temperature=config.temperature,
            )
            logits = model(obs, mask)
            total += float(distill_loss(logits, targets, mask).item()) * obs.shape[0]
    return total / data.size


def freeze_module(module: nn.Module) -> None:
    """Utility for tests and audits: disable gradients module-wide."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
