"""AlphaZero-style search & self-play for Splendor (roadmap Z0-Z3).

Monte Carlo tree search with learned priors/values is the policy improvement
operator; the training signal is the root visit distribution plus the true
terminal outcome z (no TD bootstrapping, no reward shaping). Hidden
information (unseen deck order + rival face-down reservations) is handled by
determinization ensembles reusing :mod:`splendor.agents.our_agents.dqn.search`.
"""

from .state_utils import Transactor, state_fingerprint

__all__ = ["Transactor", "state_fingerprint"]
