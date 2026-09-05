"""
Implementation of a DQN agent with a Dueling MLP neural network.
"""

from typing import override

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .network import QNetwork
from .utils import load_saved_dqn


class DQNAgent(Agent):
    """
    DQN agent with a Dueling MLP neural network.
    """

    def __init__(self, _id: int, load_net: bool = True) -> None:
        """
        Create a new DQN agent.

        :param _id: the id (turn) of the agent.
        :param load_net: whether to load the installed weights into the agent.
                         Pass ``False`` when the network is assigned later via
                         ``load_policy`` (e.g. self-play training).
        """
        super().__init__(_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: QNetwork | None = None
        if load_net:
            self.load_policy(self.load())

    def load(self) -> QNetwork:
        """
        Load and return the weights of the network.
        """
        return load_saved_dqn()

    def load_policy(self, policy: QNetwork) -> None:
        """
        Use a given policy as the agent's network policy.
        """
        self.net = policy.to(self.device)
        self.net.eval()

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """
        select an action to play from the given actions.
        """
        with torch.no_grad():
            state: NDArray = extract_metrics_with_cards(game_state, self.id).astype(
                np.float32
            )
            state_tensor: torch.Tensor = torch.from_numpy(state).to(self.device)

            action_mask = torch.from_numpy(
                create_legal_actions_mask(actions, game_state, self.id).astype(
                    np.float32
                )
            ).to(self.device)

            # this assertion is only for mypy.
            assert self.net is not None

            chosen_action = self.net.act(state_tensor, action_mask)
            mapping = create_action_mapping(actions, game_state, self.id)

        return mapping[chosen_action]


myAgent = DQNAgent  # pylint: disable=invalid-name
