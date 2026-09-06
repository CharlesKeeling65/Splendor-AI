"""Weighted opponent pools for DQN curriculum training."""

import math
import random
from collections.abc import Callable, Mapping
from typing import NamedTuple

from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

OpponentFactory = Callable[[int], list[Agent]]


class OpponentPoolEntry(NamedTuple):
    """One named opponent policy and its sampling weight."""

    name: str
    weight: float


class OpponentPoolAgent(Agent):
    """Delegate one game to a policy sampled from a weighted opponent pool.

    The gym environment contains one opponent object, so the pool is exposed
    as one ``Agent`` and chooses a policy at the first opponent turn of each
    game.  The selected policy is then kept for the rest of that game; this
    makes the training distribution a mixture of complete opponents instead
    of an artificial policy switch after every move.
    """

    def __init__(
        self,
        entries: list[OpponentPoolEntry],
        agents: list[Agent],
        _id: int,
    ) -> None:
        """Create a pool with already-constructed opponent agents."""
        if len(entries) != len(agents) or not entries:
            raise ValueError("opponent pool entries and agents must be non-empty and aligned")

        super().__init__(_id)
        self.entries = tuple(entries)
        self._agents = tuple(agents)
        self._weights = tuple(entry.weight for entry in entries)
        self._active_agent: Agent | None = None

    @property
    def description(self) -> str:
        """Return a stable CLI/config description of the pool."""
        return ",".join(
            f"{entry.name}:{entry.weight:g}" for entry in self.entries
        )

    def _choose_for_new_game(self, game_state: SplendorState) -> Agent:
        """Sample a policy when the opponent's trace is empty."""
        del game_state
        self._active_agent = random.choices(self._agents, weights=self._weights, k=1)[0]
        return self._active_agent

    @staticmethod
    def _trace_length(game_state: SplendorState, agent_id: int) -> int:
        """Read the opponent action count used as the episode boundary marker."""
        return len(game_state.agents[agent_id].agent_trace.action_reward)

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Select an action using one policy sampled for this game."""
        if self._active_agent is None or self._trace_length(game_state, self.id) == 0:
            agent = self._choose_for_new_game(game_state)
        else:
            # The guard above ensures this is initialized; keeping the local
            # assignment explicit helps both readers and static type checkers.
            assert self._active_agent is not None
            agent = self._active_agent

        # SplendorEnv assigns the pool object's seat after reset.  Propagate
        # that runtime id to the delegated policy (important for minimax).
        agent.id = self.id
        return agent.SelectAction(actions, game_state, game_rule)


def parse_opponent_pool(spec: str) -> list[OpponentPoolEntry]:
    """Parse ``name[:weight],name[:weight]`` into validated entries."""
    entries: list[OpponentPoolEntry] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        name, separator, raw_weight = item.partition(":")
        name = name.strip()
        if not name:
            raise ValueError(f"invalid opponent pool item: {raw_item!r}")
        try:
            weight = float(raw_weight) if separator else 1.0
        except ValueError as error:
            raise ValueError(f"invalid opponent weight in {raw_item!r}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"opponent weights must be finite and positive: {raw_item!r}")
        entries.append(OpponentPoolEntry(name, weight))

    if not entries:
        raise ValueError("opponent pool must contain at least one opponent")
    return entries


def build_opponent_pool(
    spec: str,
    factories: Mapping[str, OpponentFactory],
    agent_id: int = 0,
) -> OpponentPoolAgent:
    """Construct a pool from a CLI spec and the repository opponent registry."""
    entries = parse_opponent_pool(spec)
    agents: list[Agent] = []
    for entry in entries:
        factory = factories.get(entry.name)
        if factory is None:
            available = ", ".join(sorted(factories))
            raise ValueError(
                f"unknown pool opponent {entry.name!r}; available: {available}"
            )
        produced = factory(agent_id)
        if len(produced) != 1:
            raise ValueError(
                f"pool opponent {entry.name!r} must produce exactly one agent, "
                f"got {len(produced)}"
            )
        agents.append(produced[0])
    return OpponentPoolAgent(entries, agents, agent_id)
