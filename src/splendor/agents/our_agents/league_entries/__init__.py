"""League entry modules for trained checkpoints (roadmap C4).

The league evaluator loads agents by module path and instantiates
``myAgent(seat)``; this package bridges arbitrary checkpoint files into that
convention via environment variables, so fixed modules serve every evaluation
run without copying weights over the installed defaults:

* :mod:`splendor.agents.our_agents.league_entries.imitation_ppo` - an
  imitation-PPO (policy/value) checkpoint from ``SPLENDOR_PPO_CHECKPOINT``.
* :mod:`splendor.agents.our_agents.league_entries.dqn` - a DQN checkpoint from
  ``SPLENDOR_DQN_CHECKPOINT``.

Both act greedily (argmax over the engine-derived legal mask) with torch in
eval mode; no installed default is ever substituted silently.
"""

from splendor.agents.our_agents.league_entries.base import (
    GreedyCheckpointAgent,
)

__all__ = ["GreedyCheckpointAgent"]
