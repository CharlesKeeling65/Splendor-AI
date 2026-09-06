"""DQN-only public heuristic and bounded, frozen historical opponents."""

import random
from collections import Counter
from copy import deepcopy
from typing import override

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.minmax import MiniMaxAgent
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .dqn_agent import DQNAgent
from .network import QNetwork


class HeuristicAgent(Agent):
    """Fast public baseline: immediate points, discounts, then useful resources."""

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Rank actions without simulation, hidden decks or mutation."""
        agent = game_state.agents[self.id]

        def score(action: ActionType) -> float:
            value = 3.0 if action.get("noble") else 0.0
            if action["type"] in ("buy_available", "buy_reserve"):
                card = action["card"]
                return (
                    value
                    + 2 * card.points
                    + 1
                    + 1 / (1 + len(agent.cards[card.colour]))
                )
            if action["type"] in ("collect_diff", "collect_same"):
                return (
                    value
                    + sum(
                        count / (1 + agent.gems[c] + len(agent.cards[c]))
                        for c, count in action["collected_gems"].items()
                    )
                    * 0.2
                )
            return value - 0.1

        return max(actions, key=score)


class PopulationAgent(Agent):
    """Choose once per game; snapshots never share parameters with the learner."""

    def __init__(
        self, seat: int, seed: int, history: bool = True, limit: int = 4
    ) -> None:
        super().__init__(seat)
        if limit < 1:
            raise ValueError("history limit must be positive")
        self.rng = random.Random(seed)
        self.history_enabled = history
        self.limit = limit
        self.snapshots: list[DQNAgent] = []
        self.base: list[tuple[str, Agent]] = [
            ("random", RandomAgent(seat)),
            ("minimax", MiniMaxAgent(seat)),
        ]
        if history:
            self.base.append(("heuristic", HeuristicAgent(seat)))
        self.active: Agent | None = None
        self.active_name = ""
        self.counts: Counter[str] = Counter()

    def add_snapshot(self, net: QNetwork) -> None:
        """Keep at most limit CPU policies; replacing history does not alter active."""
        if not self.history_enabled:
            return
        rival = DQNAgent(self.id, load_net=False)
        rival.device = next(net.parameters()).device
        rival.load_policy(deepcopy(net))
        assert rival.net is not None
        rival.net.requires_grad_(False)
        rival.net.normalization_frozen = True
        self.snapshots.append(rival)
        self.snapshots = self.snapshots[-self.limit :]

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Isolate minimax's in-place search from the live training state."""
        if (
            self.active is None
            or not game_state.agents[self.id].agent_trace.action_reward
        ):
            choices = list(self.base)
            # A single history bucket avoids increasing its weight as it grows.
            if self.snapshots:
                choices.append(("history", self.rng.choice(self.snapshots)))
            self.active_name, self.active = self.rng.choice(choices)
            self.counts[self.active_name] += 1
        self.active.id = self.id
        if self.active_name == "minimax":
            return self.active.SelectAction(
                actions, deepcopy(game_state), deepcopy(game_rule)
            )
        return self.active.SelectAction(actions, game_state, game_rule)


myAgent = HeuristicAgent
