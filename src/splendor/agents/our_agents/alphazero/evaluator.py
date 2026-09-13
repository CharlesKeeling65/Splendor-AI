"""Pluggable leaf evaluators for the AlphaZero search.

The search tree is evaluator-agnostic: Z1 measures pure search strength with
uniform priors and zero values, Z2 plugs in the trained policy/value network.
Both evaluators return the prior over the *provided* legal-action indices
(same order as ``legal_action_table``) and a scalar value from the
perspective of the seat to act.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import ACTION_DIM, QNetwork
from splendor.splendor.splendor_model import SplendorState
from splendor.splendor.types import ActionType


class Evaluator(Protocol):
    """Prior + value for one position, from the acting seat's perspective.

    ``key_kind`` selects the search transposition key: ``"obs"`` (feature
    vector, needed when the evaluator consumes features anyway) or
    ``"fingerprint"`` (full canonical state, cheapest for feature-free
    evaluators).
    """

    key_kind: str

    def evaluate(
        self,
        state: SplendorState,
        seat: int,
        indices: Sequence[int],
        actions: Sequence[ActionType],
    ) -> tuple[NDArray[np.float64], float]:
        """Return (prior over ``indices`` summing to 1, value in [-1, 1]).

        ``actions[i]`` is the engine action for ``indices[i]``; evaluators
        that only need the index set may ignore it (the audit's positive
        control is the intended consumer).
        """
        ...


class UniformEvaluator:
    """Search-only baseline: uniform prior, zero leaf value (roadmap Z1).

    With zero leaf values only playouts that reach a terminal position inside
    the depth bound contribute Q evidence - the honest "what does the search
    alone see" anchor the Z1 league run measures.
    """

    key_kind = "fingerprint"

    def evaluate(
        self,
        state: SplendorState,
        seat: int,
        indices: Sequence[int],
        actions: Sequence[ActionType],
    ) -> tuple[NDArray[np.float64], float]:
        del state, seat, actions
        prior = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        return prior, 0.0


class NetEvaluator:
    """Masked policy prior + tanh outcome value from a QNetwork checkpoint."""

    key_kind = "obs"

    def __init__(self, net: QNetwork, device: torch.device | str = "cpu") -> None:
        if not net.auxiliary_heads:
            raise ValueError("AZ search needs a checkpoint with policy/value heads")
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        state: SplendorState,
        seat: int,
        indices: Sequence[int],
        actions: Sequence[ActionType],
    ) -> tuple[NDArray[np.float64], float]:
        del actions
        obs = extract_observation(state, seat, self.net.feature_version)
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        mask[list(indices)] = 1
        logits, value = self.net.policy_value(
            torch.from_numpy(obs).to(self.device),
            torch.from_numpy(mask).to(self.device),
        )
        prior = (
            logits.softmax(dim=-1)[0, list(indices)].cpu().numpy().astype(np.float64)
        )
        total = prior.sum()
        if not np.isfinite(total) or total <= 0:
            # HUGE_NEG masking can underflow every entry when no legal action
            # was set; fall back to uniform rather than produce NaNs.
            prior = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        else:
            prior /= total
        return prior, float(value.item())
