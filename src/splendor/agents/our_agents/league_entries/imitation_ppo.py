"""League entry: greedy imitation-PPO checkpoint (``SPLENDOR_PPO_CHECKPOINT``)."""

from typing import override

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from splendor.agents.our_agents.policy_imitation.bc_training import DeviceName
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    PolicyValueNetwork,
    load_ppo_checkpoint,
)
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.features_v2 import extract_observation_v2
from splendor.splendor.splendor_model import SplendorState

from .base import GreedyCheckpointAgent


class ImitationPPOCheckpointAgent(GreedyCheckpointAgent):
    """Greedy policy/value-head agent for the league evaluator."""

    checkpoint_env = "SPLENDOR_PPO_CHECKPOINT"

    def _load_net(self) -> nn.Module:
        device: DeviceName = "cpu"
        return load_ppo_checkpoint(self._checkpoint_path(), device_name=device)

    def _extract_observation(self, state: SplendorState) -> NDArray[np.float32]:
        version = self._feature_version()
        if version == "v1":
            return extract_metrics_with_cards(state, self.id).astype(np.float32)
        return extract_observation_v2(state, self.id, version)

    @override
    def _action_index(
        self, observation: NDArray[np.float32], mask: NDArray[np.float32]
    ) -> int:
        # PolicyValueNetwork has no .act; its forward returns masked logits.
        net = self.net
        assert isinstance(net, PolicyValueNetwork)
        with torch.no_grad():
            logits, _value = net(
                torch.from_numpy(observation).to(self.device_name),
                torch.from_numpy(mask).to(self.device_name),
            )
        return int(logits.argmax(dim=-1).item())


myAgent = ImitationPPOCheckpointAgent
