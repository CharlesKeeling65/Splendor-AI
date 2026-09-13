"""League entry: greedy DQN checkpoint (``SPLENDOR_DQN_CHECKPOINT``)."""

from typing import override

import numpy as np
from numpy.typing import NDArray
from torch import nn

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.splendor.splendor_model import SplendorState

from .base import GreedyCheckpointAgent


class DQNCheckpointAgent(GreedyCheckpointAgent):
    """Greedy Q-network agent for the league evaluator."""

    checkpoint_env = "SPLENDOR_DQN_CHECKPOINT"

    def _load_net(self) -> nn.Module:
        return load_saved_dqn(self._checkpoint_path())

    def _extract_observation(self, state: SplendorState) -> NDArray[np.float32]:
        return extract_observation(state, self.id, self._feature_version())

    @override
    def _action_index(
        self, observation: NDArray[np.float32], mask: NDArray[np.float32]
    ) -> int:
        net = self.net
        assert isinstance(net, QNetwork)
        return super()._action_index(observation, mask)


myAgent = DQNCheckpointAgent
