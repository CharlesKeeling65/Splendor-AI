"""
Implementation of the Dueling DQN neural network.
"""

from typing import Any, override

import numpy as np
import torch
from torch import nn

from splendor.agents.our_agents.ppo.input_norm import InputNormalization
from splendor.splendor.features import METRICS_WITH_CARDS_SIZE
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

from .constants import (
    HIDDEN_DIMS,
    HUGE_NEG,
    RUNNING_STATS_DECAY,
)

OBS_DIM = METRICS_WITH_CARDS_SIZE  # 265
ACTION_DIM = len(ALL_ACTIONS)  # 3510


class QNetwork(nn.Module):
    """
    Dueling Q-network.

    The trunk reuses the PPO recipe ([Linear + LayerNorm + ReLU] x N, no
    Dropout) and splits into a value head V(s) and an advantage head A(s, a),
    aggregated as q = V + A - mean(A). Dueling pays off in this action space:
    most of the 3510 actions are illegal in any given state, so learning the
    state value once and only the relative action differences is far more
    sample efficient than a single head learning every action independently.

    :note: action masking happens *inside* forward - any caller (greedy act,
           training gather, bootstrap argmax) gets illegal actions replaced by
           HUGE_NEG, so a forgotten mask at some call site cannot corrupt the
           learning. HUGE_NEG never enters the loss: the training loop only
           gathers already executed (hence legal) actions, and the bootstrap
           max lands on a legal action by construction.

    :note: the network is set to eval mode at construction and should stay
           there. InputNormalization updates its running statistics from
           *batch* statistics while in training mode, which degenerates on the
           single-observation forwards used for greedy action selection
           (1-sample variance is 0, so both the normalized output and the
           running variance collapse). The training loop therefore feeds
           observations through :meth:`observe` to keep the statistics fresh
           while action selection, gradient updates and deployment all
           normalize with identical running statistics.
    """

    def __init__(
        self,
        input_dim: int = OBS_DIM,
        output_dim: int = ACTION_DIM,
        hidden_layers: tuple[int, ...] = HIDDEN_DIMS,
        use_input_norm: bool = True,
        dueling: bool = True,
    ) -> None:
        """
        Create a new Q-network.

        :param input_dim: how many features the observations have.
        :param output_dim: how many actions the (global) action space has.
        :param hidden_layers: widths of the hidden layers of the trunk.
        :param use_input_norm: whether to normalize the inputs with a running
                               mean & variance (kept in the checkpoint).
        :param dueling: whether to use the Dueling (V + A) heads or a single
                        Q head.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dueling = dueling

        self.input_norm: InputNormalization | None = (
            InputNormalization(input_dim) if use_input_norm else None
        )

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for next_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, next_dim),
                    nn.LayerNorm(next_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = next_dim
        self.net = nn.Sequential(*layers)

        if dueling:
            self.value_head = nn.Linear(prev_dim, 1)
            self.advantage_head = nn.Linear(prev_dim, output_dim)
        else:
            self.q_head = nn.Linear(prev_dim, output_dim)

        # Initialize weights (recursively), mirroring the PPO network.
        self.apply(self._init_weights)

        # See the class docstring: training-mode forwards through
        # InputNormalization are degenerate for this network, so it lives in
        # eval mode for its whole life (gradient descent works identically).
        self.eval()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Orthogonal initialization of the weights as suggested by (bullet #2):
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def _unmasked_q(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the raw (unmasked) Q values for the given observations.

        :param obs: observations of shape (features,) or (batch, features).
        :return: Q values of shape (batch, output_dim).
        """
        if obs.dim() == 1:
            # assumes that the batch dimension is missing.
            obs = obs.unsqueeze(0)

        x = self.input_norm(obs) if self.input_norm is not None else obs
        hidden = self.net(x)

        if not self.dueling:
            return self.q_head(hidden)

        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        # Subtracting the mean advantage keeps Q identifiable (V + A is
        # under-determined otherwise).
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    @override
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Pass input through the network to gain masked Q values.

        :param obs: the input to the network.
                    expected shape: (features,) or (batch_size, features).
        :param action_mask: a binary masking tensor, 1's signal a valid action
                            and 0's signal an invalid action.
                            expected shape: (actions,) or (batch_size, actions).
        :return: the masked Q values, of shape (batch_size, actions). Illegal
                 actions hold HUGE_NEG.
        """
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        q_values = self._unmasked_q(obs)
        return q_values.masked_fill(action_mask == 0, HUGE_NEG)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        """
        Select the greedy action for a single state (evaluation / deployment).

        :param obs: a single observation, of shape (features,).
        :param action_mask: a binary masking tensor, of shape (actions,).
        :return: the index of the greedy *legal* action.
        """
        q_values = self.forward(obs, action_mask)
        return int(q_values.argmax(dim=-1).item())

    @torch.no_grad()
    def observe(self, obs: torch.Tensor) -> None:
        """
        Fold a single observation into the input normalization statistics.

        InputNormalization only updates its running statistics in training
        mode, from batch statistics - which degenerate on single observations
        (see the class docstring). The training loop calls this method once per
        collected step instead, using the same EMA as the original layer, with
        the variance estimated from the squared deviation of the observation
        from the *previous* mean (a stable online proxy for 1-sample updates).

        :param obs: a single observation, of shape (features,).
        """
        if self.input_norm is None:
            return

        x = obs.unsqueeze(0) if obs.dim() == 1 else obs
        mean = x.mean(dim=0)
        variance = (x - self.input_norm.running_mean).pow(2).mean(dim=0)
        self.input_norm.running_mean = (
            self.input_norm.running_mean * RUNNING_STATS_DECAY
            + mean * (1 - RUNNING_STATS_DECAY)
        )
        self.input_norm.running_var = (
            self.input_norm.running_var * RUNNING_STATS_DECAY
            + variance * (1 - RUNNING_STATS_DECAY)
        )

    def raw_q(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return the unmasked Q values (debugging interface).

        Also serves phase 4 (payment dimension): the trunk features can be
        reused for a payment head without re-deriving the forward pass.

        :param obs: observations of shape (features,) or (batch, features).
        :return: unmasked Q values, of shape (batch, output_dim).
        """
        return self._unmasked_q(obs)
