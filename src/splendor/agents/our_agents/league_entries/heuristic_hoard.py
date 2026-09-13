"""League entry: hoard-style weighted heuristic (roadmap C3/C4)."""

from functools import partial

from splendor.agents.our_agents.policy_imitation.policies import (
    HOARD_WEIGHTS,
    WeightedHeuristicAgent,
)

myAgent = partial(WeightedHeuristicAgent, weights=HOARD_WEIGHTS)
