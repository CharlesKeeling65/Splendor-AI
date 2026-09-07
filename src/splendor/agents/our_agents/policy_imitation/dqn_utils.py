"""Small DQN loading adapter used by the policy-imitation package."""

from pathlib import Path

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn


def load_dqn_template(path: Path) -> QNetwork:
    """Load a frozen DQN snapshot without modifying the installed DQN weight."""
    return load_saved_dqn(path).eval()
