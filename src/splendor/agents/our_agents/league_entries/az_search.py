"""League entry: AlphaZero search agent (roadmap Z1/Z2).

Configured entirely through environment variables so the league evaluator can
instantiate it per seat without a custom roster format:

* ``AZ_SEARCH_SIMS`` - simulations per decision (default 100, split across
  trees);
* ``AZ_SEARCH_TREES`` - determinization ensemble size (default 4);
* ``AZ_SEARCH_MAX_DEPTH`` - playout depth bound (default 24);
* ``AZ_SEARCH_CHECKPOINT`` - optional QNetwork checkpoint path; absent means
  the Z1 uniform-prior / zero-value pure-search configuration;
* ``AZ_SEARCH_DEVICE`` - torch device for the checkpoint evaluator (default
  ``cpu``);
* ``AZ_SEARCH_SEED`` - rng stream base (default 20260913). Streams are
  per-seat (``seed + id``) and independent of the engine's global random, so
  league runs are reproducible run-to-run.

Root Dirichlet noise is disabled here: noise is a *self-play exploration*
device, not an evaluation-time one.
"""

import os
from pathlib import Path
from typing import override

import numpy as np

from splendor.agents.our_agents.alphazero.evaluator import (
    Evaluator,
    NetEvaluator,
    UniformEvaluator,
)
from splendor.agents.our_agents.alphazero.mcts import az_search, select_action
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class AzSearchAgent(Agent):
    """MCTS decision maker for league evaluation (two-player games)."""

    def __init__(self, _id: int) -> None:
        super().__init__(_id)
        simulations = int(os.environ.get("AZ_SEARCH_SIMS", "100"))
        n_trees = int(os.environ.get("AZ_SEARCH_TREES", "4"))
        if simulations < n_trees or n_trees < 1:
            raise ValueError("AZ_SEARCH_SIMS must be >= AZ_SEARCH_TREES >= 1")
        self._simulations = simulations
        self._n_trees = n_trees
        self._max_depth = int(os.environ.get("AZ_SEARCH_MAX_DEPTH", "24"))
        checkpoint = os.environ.get("AZ_SEARCH_CHECKPOINT")
        if checkpoint:
            net = load_saved_dqn(Path(checkpoint))
            self._evaluator: Evaluator = NetEvaluator(
                net, os.environ.get("AZ_SEARCH_DEVICE", "cpu")
            )
        else:
            self._evaluator = UniformEvaluator()
        self._rng = np.random.default_rng(
            int(os.environ.get("AZ_SEARCH_SEED", "20260913")) + _id
        )

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state
        if game_rule.gameEnds():
            return actions[0]
        result = az_search(
            game_rule,
            self._evaluator,
            self._rng,
            simulations=self._simulations,
            n_trees=self._n_trees,
            max_depth=self._max_depth,
        )
        index = select_action(result, temperature=0.0, rng=self._rng)
        return create_action_mapping(
            actions, game_rule.current_game_state, self.id
        )[index]


myAgent = AzSearchAgent  # pylint: disable=invalid-name
