"""Legacy Agent wrapper for a behavior-cloning checkpoint."""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .bc_network import BehaviorCloningNetwork
from .bc_training import DeviceName, load_bc_checkpoint, resolve_device


class BehaviorCloningAgent(Agent):
    """Greedy masked policy used for fixed-opponent BC evaluation."""

    def __init__(
        self,
        _id: int,
        checkpoint: Path | None = None,
        *,
        model: BehaviorCloningNetwork | None = None,
        device_name: DeviceName = "cpu",
    ) -> None:
        super().__init__(_id)
        if (checkpoint is None) == (model is None):
            raise ValueError("provide exactly one of checkpoint or model")
        self.device = resolve_device(device_name)
        if model is None:
            assert checkpoint is not None
            self.net = load_bc_checkpoint(checkpoint, device_name=device_name)
        else:
            self.net = model
        self.net = self.net.to(self.device).eval()

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Select an action after applying the engine-derived legal mask."""
        del game_rule
        with torch.no_grad():
            observation: NDArray[np.float32] = extract_observation(
                game_state, self.id, self.net.feature_version
            )
            mask = create_legal_actions_mask(actions, game_state, self.id).astype(
                np.float32
            )
            action_index = self.net.act(
                torch.from_numpy(observation).to(self.device),
                torch.from_numpy(mask).to(self.device),
            )
        return create_action_mapping(actions, game_state, self.id)[action_index]


def build_bc_agent_factory(
    checkpoint: Path,
    *,
    device_name: DeviceName = "cpu",
) -> Callable[[int], BehaviorCloningAgent]:
    """Return an isolated factory that copies one BC model per game."""
    template = load_bc_checkpoint(checkpoint, device_name=device_name)
    template.requires_grad_(False)

    def factory(agent_id: int) -> BehaviorCloningAgent:
        return BehaviorCloningAgent(
            agent_id,
            model=deepcopy(template),
            device_name=device_name,
        )

    return factory
