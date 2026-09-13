"""Training-side reward shaping (roadmap 2026-09-12 §B2).

Two variants, kept strictly apart because only one of them is safe:

**Potential shaping** (default, policy-invariant).  Following Ng, Harada &
Russell (1999), the reward is augmented with

    F(s, s') = gamma * phi(s') - phi(s),
    phi(s)   = calScore(s, i) - max_{j != i} calScore(s, j)
               + kappa * noble_progress(s, i)

which provably leaves the optimal policy set unchanged for any phi that
depends only on the state.  The potential reuses ``calScore`` - the exact
function the evaluation path uses - so the shaped signal can never disagree
with scoring, plus a small noble-coverage term to break the flat plateau of
zero score deltas before the first purchase.

**Event shaping** (experimental, NOT policy-invariant).  A few hand-tuned
event bonuses (purchase, noble visit, denial reserve) in the spirit of
event-value functions.  Because event bonuses depend on the *action*, they
can change which policy is optimal; this variant exists only as an ablation
control group and must be labelled as such in every manifest that uses it.

Both variants mirror :mod:`splendor.agents.our_agents.dqn.reward_wrapper`
in interface: they wrap a trainer-side reward without touching the base
environment, and they work on top of any ``SplendorEnvBase`` implementation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, override

import gymnasium as gym
from numpy.typing import NDArray

from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.splendor_env import SplendorEnv
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import Card, SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType

DEFAULT_KAPPA = 0.05
DEFAULT_DISCOUNT = 0.99


@dataclass(frozen=True)
class EventWeights:
    """Hand-tuned event bonuses (experimental group only)."""

    buy: float = 0.1
    noble: float = 1.0
    denial_reserve: float = 0.05


def noble_progress(state: SplendorState, seat: int) -> float:
    """Best coverage of any board noble by the agent's purchased cards.

    Coverage of one noble is ``sum(min(owned_c, cost_c)) / sum(cost_c)`` over
    the required colours; the value is in [0, 1] and 0 when no nobles remain.
    """
    agent = state.agents[seat]
    best = 0.0
    for _, cost in state.board.nobles:
        total = sum(cost.values())
        if total <= 0:
            continue
        covered = sum(
            min(len(agent.cards[colour]), need) for colour, need in cost.items()
        )
        best = max(best, covered / total)
    return best


def potential(
    state: SplendorState,
    rule: SplendorGameRule,
    seat: int,
    *,
    kappa: float = DEFAULT_KAPPA,
) -> float:
    """State-only potential: score lead over the best rival plus noble term."""
    own = float(rule.calScore(state, seat))
    best_rival = max(
        (
            float(rule.calScore(state, agent.id))
            for agent in state.agents
            if agent.id != seat
        ),
        default=own,
    )
    return own - best_rival + kappa * noble_progress(state, seat)


class PotentialRewardShaper:
    """Stateful telescoping shaper for raw-engine training loops.

    The loop owns the MDP's step granularity (for PPO self-play one step is
    one focal-agent decision, rivals folded in).  Call :meth:`reset` at the
    game's first decision state, then :meth:`advance` once per subsequent
    state the focal agent observes - the returned bonus is credited to the
    transition that *led* to that state.  The sum of all bonuses over an
    episode telescopes to ``gamma^T * phi(s_T) - phi(s_0)``.
    """

    def __init__(
        self,
        kappa: float = DEFAULT_KAPPA,
        discount_factor: float = DEFAULT_DISCOUNT,
    ) -> None:
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError("discount_factor must lie in (0, 1]")
        self.kappa = kappa
        self.discount_factor = discount_factor
        self._previous: dict[int, float] = {}

    def phi(self, state: SplendorState, seat: int, rule: SplendorGameRule) -> float:
        """Expose the potential so tests and logs can audit telescoping."""
        return potential(state, rule, seat, kappa=self.kappa)

    def reset(self, state: SplendorState, seat: int, rule: SplendorGameRule) -> float:
        """Anchor the shaper at the first decision state; returns phi(s_0)."""
        value = self.phi(state, seat, rule)
        self._previous[seat] = value
        return value

    def advance(self, state: SplendorState, seat: int, rule: SplendorGameRule) -> float:
        """Credit the transition leading to ``state`` and re-anchor there."""
        value = self.phi(state, seat, rule)
        bonus = self.discount_factor * value - self._previous[seat]
        self._previous[seat] = value
        return bonus


def card_affordable(agent: SplendorState.AgentState, card_cost: dict[str, int]) -> bool:
    """Whether the agent's gems (+ yellow wildcards) can pay ``card_cost``.

    Yellow wildcards are shared across colours, so shortfalls are accumulated
    first and covered by the single yellow pool at the end.
    """
    yellow_needed = 0
    for colour, need in card_cost.items():
        if colour == "yellow":  # cards never cost yellow; stay defensive
            continue
        available = agent.gems.get(colour, 0)
        if available < need:
            yellow_needed += need - available
    return agent.gems.get("yellow", 0) >= yellow_needed


class EventRewardShaper:
    """Action-dependent event bonuses - the non-invariant control group."""

    def __init__(self, weights: EventWeights | None = None) -> None:
        self.weights = weights or EventWeights()

    def reset(self) -> None:
        """Stateless per-step bonuses; nothing to anchor."""

    def bonus(
        self,
        action: ActionType,
        state_after: SplendorState,
        rule: SplendorGameRule,
        seat: int,
    ) -> float:
        """Bonus owed for the focal agent taking ``action``."""
        del rule  # kept in the signature so both shapers share one call site
        weights = self.weights
        runtime_action = cast(dict[str, Any], action)
        action_type = action["type"]
        total = 0.0
        if "buy" in action_type:
            total += weights.buy
        if runtime_action.get("noble"):
            total += weights.noble
        # ReserveAction carries "card" at runtime although the TypedDict
        # schema in splendor.types does not declare it.
        card = runtime_action.get("card")
        if action_type == "reserve" and card is not None:
            cost = cast("Card", card).cost
            rivals = [a for a in state_after.agents if a.id != seat]
            if any(card_affordable(rival, dict(cost)) for rival in rivals):
                total += weights.denial_reserve
        return total


@dataclass(frozen=True)
class ShapingConfig:
    """Declarative shaping selection for manifests and CLI flags."""

    kind: str = "none"  # "none" | "potential" | "event"
    kappa: float = DEFAULT_KAPPA
    discount_factor: float = DEFAULT_DISCOUNT

    def __post_init__(self) -> None:
        if self.kind not in {"none", "potential", "event"}:
            raise ValueError(f"unknown shaping kind: {self.kind!r}")


class PotentialShapingWrapper(gym.Wrapper):
    """Gym-side potential shaping, composing with any inner reward stack.

    Wrap *outside* :class:`~splendor.agents.our_agents.dqn.reward_wrapper.
    TerminalRewardWrapper`: the shaped bonus is additive to the base reward,
    so ``reward + gamma*phi(s') - phi(s)`` applies to the base signal however
    it is composed.  ``gamma`` must match the trainer's discount factor.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        kappa: float = DEFAULT_KAPPA,
        discount_factor: float = DEFAULT_DISCOUNT,
    ) -> None:
        super().__init__(env)
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError("discount_factor must lie in (0, 1]")
        self.kappa = kappa
        self.discount_factor = discount_factor
        self.my_id: int = -1
        self._previous = 0.0
        self.potentials: list[float] = []

    def _phi(self, state: SplendorState, rule: SplendorGameRule) -> float:
        return potential(state, rule, self.my_id, kappa=self.kappa)

    @override
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.my_id = int(info["my_id"])
        splendor_env = cast(SplendorEnv, self.env.unwrapped)
        self._previous = self._phi(splendor_env.state, splendor_env.game_rule)
        self.potentials = [self._previous]
        return obs, info

    @override
    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[NDArray, float, bool, bool, dict]:
        # gymnasium 0.29's stock wrappers do not forward keyword arguments, so
        # payment is only passed through when it carries information.
        if payment is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            env = cast(SplendorEnvBase, self.env)
            obs, reward, terminated, truncated, info = env.step(action, payment)
        splendor_env = cast(SplendorEnv, self.env.unwrapped)
        value = self._phi(splendor_env.state, splendor_env.game_rule)
        self.potentials.append(value)
        reward = float(reward) + self.discount_factor * value - self._previous
        self._previous = value
        return obs, reward, terminated, truncated, info


class EventShapingWrapper(gym.Wrapper):
    """Gym-side event bonuses - experimental group, not policy-invariant."""

    def __init__(self, env: gym.Env, weights: EventWeights | None = None) -> None:
        super().__init__(env)
        self.my_id: int = -1
        self.shaper = EventRewardShaper(weights)

    @override
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.my_id = int(info["my_id"])
        self.shaper.reset()
        return obs, info

    @override
    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[NDArray, float, bool, bool, dict]:
        # The engine action dict is recovered the same way SplendorEnv.step
        # resolves it internally (create_action_mapping on the pre-step state)
        # so event classification matches the raw-engine path exactly.
        splendor_env = cast(SplendorEnv, self.env.unwrapped)
        legal_actions = splendor_env.game_rule.getLegalActions(
            splendor_env.state, splendor_env.my_turn
        )
        engine_action = create_action_mapping(
            legal_actions, splendor_env.state, splendor_env.my_turn
        )[int(action)]
        if payment is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            env = cast(SplendorEnvBase, self.env)
            obs, reward, terminated, truncated, info = env.step(action, payment)
        state_after = cast(SplendorEnv, self.env.unwrapped).state
        reward = float(reward) + self.shaper.bonus(
            engine_action, state_after, splendor_env.game_rule, self.my_id
        )
        return obs, reward, terminated, truncated, info


# ----- Roadmap E2: ranking utilities for 3/4-player training -----------------

#: Suphx-style utility per final rank (1 = best).  In a 4-player game the
#: third-vs-fourth gap matters as much as first-vs-second, which a binary
#: win/loss terminal signal cannot express.
RANK_UTILITIES: dict[int, float] = {1: 1.0, 2: 0.0, 3: -0.5, 4: -1.0}


def rank_of(scores: Sequence[float], seat: int) -> int:
    """1-based competition rank of ``seat`` within ``scores`` (ties share)."""
    if not 0 <= seat < len(scores):
        raise ValueError(f"seat {seat} outside 0..{len(scores) - 1}")
    own = float(scores[seat])
    return 1 + sum(1 for s in scores if float(s) > own)


def rank_utility(scores: Sequence[float], seat: int) -> float:
    """Utility of ``seat``'s final rank; tied seats average their utilities.

    Ties are averaged so a shared first place scores (1 + 0) / 2 = 0.5 rather
    than handing both seats the full first-place utility.
    """
    own = float(scores[seat])
    better = sum(1 for s in scores if float(s) > own)
    tied = sum(1 for s in scores if float(s) == own)
    utilities = [RANK_UTILITIES[better + offset + 1] for offset in range(tied)]
    return sum(utilities) / len(utilities)


class RankUtilityWrapper(gym.Wrapper):
    """Terminal ranking utility for n-seat self-play (roadmap E2).

    Composes exactly like :class:`TerminalRewardWrapper`: the per-step score
    deltas pass through untouched and the terminal step receives the seat's
    rank utility scaled by ``terminal_scale`` (default 10.0 keeps the reward
    magnitude comparable to the 2p ±10 win bonus).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        terminal_scale: float = 10.0,
    ) -> None:
        super().__init__(env)
        if terminal_scale <= 0:
            raise ValueError("terminal_scale must be positive")
        self.terminal_scale = terminal_scale
        self.my_id: int = -1

    @override
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.my_id = int(info["my_id"])
        return obs, info

    @override
    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[NDArray, float, bool, bool, dict]:
        if payment is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            env = cast(SplendorEnvBase, self.env)
            obs, reward, terminated, truncated, info = env.step(action, payment)
        reward = float(reward)
        if terminated:
            splendor_env = cast(SplendorEnv, self.env.unwrapped)
            state = splendor_env.state
            rule = splendor_env.game_rule
            scores = [float(rule.calScore(state, agent.id)) for agent in state.agents]
            reward += self.terminal_scale * rank_utility(scores, self.my_id)
        return obs, reward, terminated, truncated, info
