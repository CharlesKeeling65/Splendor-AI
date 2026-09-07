"""Training-only heuristic guidance with legal margins and an explicit exit schedule.

No reward or environment changes. Labels are from the current public position,
never from a future state. The learned Q network alone acts at evaluation time.
"""

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState

from .network import QNetwork
from .population import HeuristicAgent


def teacher_action_index(
    state: SplendorState, rule: SplendorGameRule, seat: int
) -> int:
    """Reuse engine action mapping, including gold payment and noble choices."""
    actions = rule.getLegalActions(state, seat)
    chosen = HeuristicAgent(seat).SelectAction(actions, state, rule)
    return next(iter(create_action_mapping([chosen], state, seat)))


def guidance_fraction(step: int, decay_steps: int) -> float:
    """Linear guidance annealing, exactly zero after the declared duration."""
    if decay_steps <= 0:
        raise ValueError("guidance decay must be positive")
    return max(0.0, 1.0 - max(0, step) / decay_steps)


def legal_margin_loss(
    q_values: torch.Tensor,
    masks: torch.Tensor,
    teachers: torch.Tensor,
    margin: float = 0.8,
) -> torch.Tensor:
    """max_legal[Q(s,a)+margin(a!=teacher)] - Q(s,teacher), averaged."""
    if margin < 0 or not torch.isfinite(torch.tensor(margin)):
        raise ValueError("margin must be finite and nonnegative")
    valid = masks.bool()
    if not bool(valid.gather(1, teachers[:, None]).all()):
        raise ValueError("teacher action must be legal")
    penalty = torch.full_like(q_values, margin)
    penalty.scatter_(1, teachers[:, None], 0.0)
    best = (q_values + penalty).masked_fill(~valid, -torch.inf).max(1).values
    chosen = q_values.gather(1, teachers[:, None]).squeeze(1)
    return (best - chosen).mean()


class GuidanceBuffer:
    """Independent ring of current-state labels; boolean masks bound memory use."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int) -> None:
        if min(capacity, obs_dim, action_dim) <= 0:
            raise ValueError("buffer dimensions must be positive")
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, action_dim), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self, obs: NDArray[np.float32], mask: NDArray[np.float32], teacher: int
    ) -> None:
        """Store a legal public label before taking any environment action."""
        if teacher not in range(self.masks.shape[1]) or not mask[teacher]:
            raise ValueError("illegal teacher action")
        self.obs[self.position] = obs
        self.masks[self.position] = mask
        self.actions[self.position] = teacher
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def loss(self, net: QNetwork, batch_size: int) -> torch.Tensor:
        """Differentiable extra loss, combined with TD in ONE optimizer step."""
        if self.size == 0 or batch_size < 1:
            raise ValueError("need labels and a positive batch size")
        indices = self.rng.integers(self.size, size=batch_size)
        device = next(net.parameters()).device
        obs = torch.from_numpy(self.obs[indices]).to(device)
        masks = torch.from_numpy(self.masks[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        return legal_margin_loss(net.raw_q(obs), masks, actions)
