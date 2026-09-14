"""AZ self-play collection and greedy evaluation (roadmap Z2).

Every move runs :func:`~.mcts.az_search` with root Dirichlet noise; the root
visit distribution is the policy training target and the game's terminal
outcome (engine tiebreak sign, ties = 0) is the value target ``z`` for every
position of the game. Actions are sampled by visit temperature for the first
``temperature_moves`` plies and greedily afterwards.

Deal determinism follows the repo seed discipline: ``random.seed`` +
``np.random.seed`` (the engine deal consumes global ``random``), while the
search stream is an independent ``default_rng`` derived from the game seed.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.alphazero.evaluator import NetEvaluator
from splendor.agents.our_agents.alphazero.mcts import az_search, select_action
from splendor.agents.our_agents.alphazero.state_utils import legal_action_table
from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent

FEATURE_VERSION = "public-v2"
OBS_DIM = 312


@dataclass(frozen=True)
class SelfPlayConfig:
    """Search and sampling knobs for one self-play batch."""

    simulations: int = 100
    n_trees: int = 4
    max_depth: int = 24
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 12


@dataclass(frozen=True)
class TrainingSample:
    """One (position, search target, outcome) training triple."""

    seat: int
    observation: NDArray[np.float32]
    indices: list[int]
    target: NDArray[np.float32]
    z: float


def build_az_network() -> QNetwork:
    """The AZ policy/value network (BC-warm-startable architecture)."""
    return QNetwork(
        input_dim=OBS_DIM,
        feature_version=FEATURE_VERSION,
        auxiliary_heads=True,
        use_input_norm=True,
        dueling=False,
        hidden_layers=(128, 128, 128, 128),
    )


def warm_start_from_bc(net: QNetwork, bc_checkpoint: str) -> None:
    """Copy BC trunk/policy/normalizer weights into the AZ network.

    Mapping: ``trunk.N`` -> ``net.N`` (identical [Linear+LN+ReLU] x4 layout),
    ``normalizer.mean/variance`` -> ``input_norm.running_*`` (frozen - the
    statistics are BC-fitted and must not drift during AZ training),
    ``policy_head`` -> ``policy_head``. The outcome head stays randomly
    initialized: its target (terminal z) only exists once self-play starts.
    """
    checkpoint = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
    bc_state = checkpoint["model_state_dict"]
    missing = [
        key
        for key in ("normalizer.mean", "normalizer.variance", "policy_head.weight")
        if key not in bc_state
    ]
    if missing:
        raise ValueError(f"BC checkpoint lacks expected keys: {missing}")
    own = net.state_dict()
    for name, value in bc_state.items():
        if name.startswith("trunk."):
            target = "net." + name[len("trunk.") :]
            if own[target].shape != value.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape}")
            own[target] = value
        elif name.startswith("policy_head."):
            if own[name].shape != value.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape}")
            own[name] = value
        elif name == "normalizer.mean":
            own["input_norm.running_mean"] = value.reshape(1, -1)
        elif name == "normalizer.variance":
            own["input_norm.running_var"] = value.reshape(1, -1)
    net.load_state_dict(own)
    net.normalization_frozen = True


def _outcome(rule: SplendorGameRule, seat: int) -> float:
    """Terminal outcome from ``seat``'s perspective (+1 win / 0 tie / -1 loss)."""
    state = rule.current_game_state
    return float(np.sign(rule.calScore(state, seat) - rule.calScore(state, 1 - seat)))


def play_game(net: QNetwork, seed: int, config: SelfPlayConfig) -> list[TrainingSample]:
    """Play one seeded self-play game; return per-move training samples."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rule: SplendorGameRule = LimitRoundsGameRule(2)
    evaluator = NetEvaluator(net, "cpu")
    rng = np.random.default_rng(seed + 9_999_983)
    samples: list[TrainingSample] = []
    ply = 0
    while not rule.gameEnds():
        seat = rule.getCurrentAgentIndex()
        result = az_search(
            rule,
            evaluator,
            rng,
            simulations=config.simulations,
            n_trees=config.n_trees,
            c_puct=config.c_puct,
            max_depth=config.max_depth,
            root_noise=True,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
        )
        observation = extract_observation(
            rule.current_game_state, seat, FEATURE_VERSION
        )
        visits = result.visits
        target = (visits / visits.sum()).astype(np.float32)
        samples.append(
            TrainingSample(
                seat=seat,
                observation=observation,
                indices=list(result.indices),
                target=target,
                z=0.0,
            )
        )
        temperature = 1.0 if ply < config.temperature_moves else 0.0
        index = select_action(result, temperature, rng)
        live_indices, live_actions = legal_action_table(rule, seat)
        if live_indices != result.indices:  # pragma: no cover - engine invariant
            raise RuntimeError("root legal set diverged between search and play")
        rule.update(dict(zip(live_indices, live_actions, strict=True))[index])
        ply += 1
    return [
        TrainingSample(s.seat, s.observation, s.indices, s.target, _outcome(rule, s.seat))
        for s in samples
    ]


def pack_samples(samples: list[TrainingSample]) -> dict[str, Any]:
    """Compact ragged sample list into fixed-width arrays for transport."""
    pointer = np.zeros(len(samples) + 1, dtype=np.int64)
    for i, sample in enumerate(samples):
        pointer[i + 1] = pointer[i] + len(sample.indices)
    flat_indices = np.zeros(pointer[-1], dtype=np.int64)
    flat_target = np.zeros(pointer[-1], dtype=np.float32)
    for i, sample in enumerate(samples):
        flat_indices[pointer[i] : pointer[i + 1]] = sample.indices
        flat_target[pointer[i] : pointer[i + 1]] = sample.target
    return {
        "seat": np.asarray([s.seat for s in samples], dtype=np.int8),
        "observation": np.stack([s.observation for s in samples]).astype(np.float32),
        "ind_ptr": pointer,
        "ind_flat": flat_indices,
        "target_flat": flat_target,
        "z": np.asarray([s.z for s in samples], dtype=np.float32),
    }


def unpack_samples(payload: dict[str, Any]) -> list[TrainingSample]:
    """Inverse of :func:`pack_samples`."""
    samples: list[TrainingSample] = []
    seats = payload["seat"]
    observations = payload["observation"]
    pointer = payload["ind_ptr"]
    flat_indices = payload["ind_flat"]
    flat_target = payload["target_flat"]
    z_values = payload["z"]
    for i in range(len(seats)):
        lo, hi = int(pointer[i]), int(pointer[i + 1])
        samples.append(
            TrainingSample(
                seat=int(seats[i]),
                observation=observations[i],
                indices=[int(v) for v in flat_indices[lo:hi]],
                target=flat_target[lo:hi],
                z=float(z_values[i]),
            )
        )
    return samples


class GreedyAZAgent(Agent):
    """Greedy masked-policy-argmax agent for iteration-level evaluation."""

    def __init__(self, _id: int, net: QNetwork) -> None:
        super().__init__(_id)
        self.evaluator = NetEvaluator(net, "cpu")

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        indices, table_actions = legal_action_table(game_rule, self.id)
        prior, _value = self.evaluator.evaluate(
            game_state, self.id, indices, table_actions
        )
        index = indices[int(np.argmax(prior))]
        return create_action_mapping(actions, game_state, self.id)[index]


def play_matchup(
    net: QNetwork,
    opponent_factory: Callable[[int], Agent],
    seed: int,
    az_seat: int,
) -> float:
    """One greedy AZ game vs a constructed opponent; AZ-perspective outcome."""
    random.seed(seed)
    np.random.seed(seed)
    rule: SplendorGameRule = LimitRoundsGameRule(2)
    agents = {
        az_seat: GreedyAZAgent(az_seat, net),
        1 - az_seat: opponent_factory(1 - az_seat),
    }
    while not rule.gameEnds():
        seat = rule.getCurrentAgentIndex()
        state = rule.current_game_state
        legal = rule.getLegalActions(state, seat)
        rule.update(agents[seat].SelectAction(legal, state, rule))
    return _outcome(rule, az_seat)


def evaluate_greedy(
    net: QNetwork,
    opponent_factory: Callable[[int], Agent],
    seeds: list[int],
) -> dict[str, float]:
    """Greedy win rate vs one opponent over deals, both seat assignments.

    Repo convention: wins / all games, ties count as losses in the rate but
    are reported separately.
    """
    outcomes = [
        play_matchup(net, opponent_factory, seed, az_seat)
        for seed in seeds
        for az_seat in (0, 1)
    ]
    wins = sum(1 for o in outcomes if o > 0)
    ties = sum(1 for o in outcomes if o == 0)
    return {
        "win_rate": wins / len(outcomes),
        "ties": float(ties),
        "games": float(len(outcomes)),
    }
