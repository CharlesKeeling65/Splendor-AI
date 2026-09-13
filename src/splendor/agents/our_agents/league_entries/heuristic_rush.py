"""League entry: rush-style weighted heuristic (roadmap C3/C4)."""

from functools import partial

from splendor.agents.our_agents.policy_imitation.policies import (
    RUSH_WEIGHTS,
    WeightedHeuristicAgent,
)

myAgent = partial(WeightedHeuristicAgent, weights=RUSH_WEIGHTS)  # noqa: PLC0414
