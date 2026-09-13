"""Shared greedy checkpoint-agent machinery for the league entries."""

import os
from pathlib import Path
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.policy_imitation.bc_training import DeviceName
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class GreedyCheckpointAgent(Agent):
    """Greedy masked-argmax agent over one environment-selected checkpoint."""

    checkpoint_env: str = ""

    def __init__(self, _id: int) -> None:
        super().__init__(_id)
        self.device_name: DeviceName = "cpu"
        self.net = self._load_net().to(self.device_name).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    def _load_net(self):  # noqa: ANN202 - subclass-specific network type
        raise NotImplementedError

    def _checkpoint_path(self) -> Path:
        raw = os.environ.get(self.checkpoint_env)
        if not raw:
            raise ValueError(
                f"set {self.checkpoint_env} to the checkpoint to evaluate "
                "(no installed default is used, avoiding silent weight swaps)"
            )
        return Path(raw)

    def _feature_version(self) -> str:
        return str(getattr(self.net, "feature_version", "v1"))

    def _extract_observation(self, state: SplendorState) -> NDArray[np.float32]:
        raise NotImplementedError

    def _action_index(
        self, observation: NDArray[np.float32], mask: NDArray[np.float32]
    ) -> int:
        obs = torch.from_numpy(observation).to(self.device_name)
        mask_tensor = torch.from_numpy(mask).to(self.device_name)
        with torch.no_grad():
            selected = self.net.act(obs, mask_tensor)
        # QNetwork.act already returns an int; other nets may return tensors.
        return selected if isinstance(selected, int) else int(selected.item())

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_rule
        observation = self._extract_observation(game_state)
        mask = create_legal_actions_mask(actions, game_state, self.id).astype(
            np.float32
        )
        action_index = self._action_index(observation, mask)
        return create_action_mapping(actions, game_state, self.id)[action_index]
