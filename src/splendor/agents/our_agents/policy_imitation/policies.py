"""Candidate policy factories for the teacher audit and imitation pipeline."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, override

import torch
from torch import nn

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.dqn.dqn_agent import DQNAgent
from splendor.agents.our_agents.dqn.population import HeuristicAgent
from splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent import (
    GeneAlgoAgent,
)
from splendor.agents.our_agents.minmax import MiniMaxAgent
from splendor.agents.our_agents.ppo.ppo_agent import (
    DEFAULT_SAVED_PPO_PATH,
    PPOAgent,
)
from splendor.agents.our_agents.ppo.utils import load_saved_ppo
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .bc_agent import build_bc_agent_factory
from .bc_training import DeviceName, load_bc_checkpoint
from .dqn_utils import load_dqn_template

AgentFactory = Callable[[int], Agent]
CandidateRole = Literal["teacher_candidate", "fixed_baseline"]


@dataclass(frozen=True)
class HeuristicWeights:
    """Ranking weights for :class:`WeightedHeuristicAgent` style variants.

    Defaults reproduce the frozen ``HeuristicAgent`` scoring exactly; the
    roadmap C3 style variants push the same ranking towards buying fast
    (``rush``) or hoarding gems (``hoard``).
    """

    noble: float = 3.0
    points: float = 2.0
    buy: float = 1.0
    colour_affinity: float = 1.0
    collect: float = 0.2
    reserve: float = -0.1


RUSH_WEIGHTS = HeuristicWeights(
    noble=6.0,
    points=4.0,
    buy=2.0,
    colour_affinity=0.5,
    collect=0.05,
    reserve=-0.3,
)
HOARD_WEIGHTS = HeuristicWeights(
    noble=1.0,
    points=1.0,
    buy=0.5,
    colour_affinity=1.0,
    collect=0.8,
    reserve=0.3,
)


class WeightedHeuristicAgent(HeuristicAgent):
    """Heuristic ranking with configurable style weights (roadmap C3)."""

    def __init__(self, _id: int, weights: HeuristicWeights | None = None) -> None:
        super().__init__(_id)
        self.weights = weights or HeuristicWeights()

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Rank actions with the configured style weights."""
        weights = self.weights
        agent = game_state.agents[self.id]

        def score(action: ActionType) -> float:
            value = weights.noble if action.get("noble") else 0.0
            if action["type"] in ("buy_available", "buy_reserve"):
                card = action["card"]
                return (
                    value
                    + weights.points * card.points
                    + weights.buy
                    + weights.colour_affinity
                    / (1 + len(agent.cards[card.colour]))
                )
            if action["type"] in ("collect_diff", "collect_same"):
                return (
                    value
                    + weights.collect
                    * sum(
                        count / (1 + agent.gems[c] + len(agent.cards[c]))
                        for c, count in action["collected_gems"].items()
                    )
                )
            return value + weights.reserve

        return max(actions, key=score)


@dataclass(frozen=True)
class CandidateSpec:
    """A named, reproducible policy snapshot used by an experiment."""

    name: str
    role: CandidateRole
    factory: AgentFactory
    feature_version: str = "v1"
    snapshot: str | None = None

    def build(self, agent_id: int) -> Agent:
        """Build an isolated agent instance for one game."""
        agent = self.factory(agent_id)
        if agent.id != agent_id:
            agent.id = agent_id
        return agent


def _resolve_device(device_name: str) -> torch.device:
    """Resolve an accelerator with the same safe fallback as DQN."""
    if device_name not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device {device_name!r}")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _frozen_module_factory(
    module: nn.Module,
    build_agent: Callable[[int, nn.Module], Agent],
    device: torch.device,
) -> AgentFactory:
    """Copy one loaded network per game so policies never share mutable state."""
    template = deepcopy(module).to(device).eval()
    template.requires_grad_(False)

    def factory(agent_id: int) -> Agent:
        return build_agent(agent_id, deepcopy(template).to(device).eval())

    return factory


def _build_dqn_agent(agent_id: int, module: nn.Module) -> Agent:
    """Attach a frozen DQN module to the existing DQN agent wrapper."""
    if not hasattr(module, "feature_version"):
        raise TypeError("DQN snapshot does not expose feature_version")
    agent = DQNAgent(agent_id, load_net=False)
    agent.device = next(module.parameters()).device
    agent.load_policy(module)  # type: ignore[arg-type]
    return agent


def _build_ppo_agent(agent_id: int, module: nn.Module) -> Agent:
    """Attach a frozen PPO module to the existing PPO agent wrapper."""
    agent = PPOAgent(agent_id, load_net=False)
    agent.device = next(module.parameters()).device
    agent.load_policy(module)
    return agent


def _static_candidate(name: str, factory: AgentFactory) -> CandidateSpec:
    return CandidateSpec(name=name, role="teacher_candidate", factory=factory)


def build_builtin_candidate(  # noqa: C901, PLR0911 - candidate declarations are explicit
    name: str,
    *,
    checkpoint: Path | None = None,
    device_name: str = "cpu",
) -> CandidateSpec:
    """Build one supported candidate without changing its established code path.

    ``corrected-dqn`` and ``ppo`` load an explicit snapshot when provided. The
    default PPO snapshot is the repository's installed weight; DQN requires a
    path because no deployed DQN weight is checked into the package.
    """
    device = _resolve_device(device_name)
    if name == "random":
        return _static_candidate(name, RandomAgent)
    if name == "heuristic":
        return _static_candidate(name, HeuristicAgent)
    if name == "heuristic-rush":
        return _static_candidate(name, partial(WeightedHeuristicAgent, weights=RUSH_WEIGHTS))
    if name == "heuristic-hoard":
        return _static_candidate(name, partial(WeightedHeuristicAgent, weights=HOARD_WEIGHTS))
    if name == "minimax":
        return _static_candidate(name, MiniMaxAgent)
    if name == "ga":
        return _static_candidate(name, GeneAlgoAgent)
    if name == "ppo":
        ppo_path = checkpoint
        if ppo_path is None:
            ppo_path = DEFAULT_SAVED_PPO_PATH
        if not ppo_path.is_file():
            raise FileNotFoundError(f"PPO snapshot does not exist: {ppo_path}")
        ppo_module = load_saved_ppo(ppo_path)
        return CandidateSpec(
            name=name,
            role="teacher_candidate",
            factory=_frozen_module_factory(ppo_module, _build_ppo_agent, device),
            snapshot=str(ppo_path),
        )
    if name == "corrected-dqn":
        if checkpoint is None:
            raise ValueError("corrected-dqn requires an explicit checkpoint path")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DQN snapshot does not exist: {checkpoint}")
        dqn_module = load_dqn_template(checkpoint)
        resolved_device = _resolve_device(device_name)
        return CandidateSpec(
            name=name,
            role="teacher_candidate",
            factory=_frozen_module_factory(
                dqn_module, _build_dqn_agent, resolved_device
            ),
            feature_version=str(dqn_module.feature_version),
            snapshot=str(checkpoint),
        )
    raise ValueError(
        f"unknown candidate {name!r}; expected random, heuristic, heuristic-rush, "
        "heuristic-hoard, minimax, ga, ppo, or corrected-dqn"
    )


def build_fixed_baseline(name: str, *, device_name: str = "cpu") -> CandidateSpec:
    """Build a fixed opponent role, kept separate from teacher selection."""
    candidate = build_builtin_candidate(name, device_name=device_name)
    return CandidateSpec(
        name=candidate.name,
        role="fixed_baseline",
        factory=candidate.factory,
        feature_version=candidate.feature_version,
        snapshot=candidate.snapshot,
    )


def build_bc_candidate(
    checkpoint: Path,
    *,
    device_name: DeviceName = "cpu",
) -> CandidateSpec:
    """Expose a trained BC checkpoint through the same evaluation interface."""
    factory = build_bc_agent_factory(checkpoint, device_name=device_name)
    model = load_bc_checkpoint(checkpoint, device_name=device_name)
    return CandidateSpec(
        name="bc",
        role="teacher_candidate",
        factory=factory,
        feature_version=model.feature_version,
        snapshot=str(checkpoint),
    )
