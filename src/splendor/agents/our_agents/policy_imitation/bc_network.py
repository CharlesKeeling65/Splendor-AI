"""Masked 3510-way behavior-cloning network."""

from typing import Any, override

import torch
from torch import nn

from splendor.agents.our_agents.dqn.constants import HIDDEN_DIMS, HUGE_NEG
from splendor.agents.our_agents.dqn.features import observation_dim
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

ACTION_DIM = len(ALL_ACTIONS)
MATRIX_RANK = 2


class FixedNormalizer(nn.Module):
    """A train-only-fitted normalizer that never reads validation/test rows."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, input_dim))
        self.register_buffer("variance", torch.ones(1, input_dim))
        self.mean: torch.Tensor
        self.variance: torch.Tensor
        self.fitted = False

    @torch.no_grad()
    def fit(self, observations: torch.Tensor) -> None:
        """Fit mean and variance from the training observations only."""
        if (
            observations.ndim != MATRIX_RANK
            or observations.shape[1] != self.mean.shape[1]
        ):
            raise ValueError("normalizer observations have the wrong shape")
        if observations.shape[0] == 0:
            raise ValueError("cannot fit a normalizer on an empty dataset")
        self.mean.copy_(observations.float().mean(dim=0, keepdim=True))
        variance = observations.float().var(dim=0, unbiased=False, keepdim=True)
        self.variance.copy_(variance.clamp_min(1e-6))
        self.fitted = True

    @override
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.mean) / torch.sqrt(self.variance)


class BehaviorCloningNetwork(nn.Module):
    """MLP policy with the same action enumeration and hidden trunk as DQN."""

    def __init__(
        self,
        input_dim: int,
        *,
        feature_version: str,
        hidden_layers: tuple[int, ...] = HIDDEN_DIMS,
        output_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        expected_dim = observation_dim(feature_version)
        if input_dim != expected_dim:
            raise ValueError(
                f"feature schema {feature_version!r} requires {expected_dim} inputs, got {input_dim}"
            )
        if output_dim != ACTION_DIM:
            raise ValueError(f"BC action head must have {ACTION_DIM} outputs")
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_version = feature_version
        self.hidden_layers = hidden_layers
        self.normalizer = FixedNormalizer(input_dim)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()]
            )
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(previous, output_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Match the validated orthogonal initialization used by DQN/PPO."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()

    def fit_normalizer(self, observations: torch.Tensor) -> None:
        """Fit the preprocessing statistics from train rows."""
        self.normalizer.fit(observations)

    def raw_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """Return logits before legal-action masking."""
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        return self.policy_head(self.trunk(self.normalizer(observations)))

    @override
    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return logits where every illegal action is excluded."""
        del args, kwargs
        if legal_masks.dim() == 1:
            legal_masks = legal_masks.unsqueeze(0)
        logits = self.raw_logits(observations)
        if logits.shape != legal_masks.shape:
            raise ValueError(
                f"logits and masks must have equal shape, got {logits.shape} and {legal_masks.shape}"
            )
        if not torch.all(legal_masks.sum(dim=1) > 0):
            raise ValueError("each BC row must contain at least one legal action")
        return logits.masked_fill(legal_masks <= 0, HUGE_NEG)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, legal_mask: torch.Tensor) -> int:
        """Select the highest-logit legal action for one state."""
        return int(self.forward(observations, legal_mask).argmax(dim=-1).item())
