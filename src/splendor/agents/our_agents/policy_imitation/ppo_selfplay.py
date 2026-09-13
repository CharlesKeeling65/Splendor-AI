"""PPO self-play initialized from BC with a declared fixed opponent pool."""

import json
import random
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions, nn, optim

from splendor.agents.our_agents.dqn.constants import HIDDEN_DIMS, HUGE_NEG
from splendor.agents.our_agents.dqn.features import extract_observation, observation_dim
from splendor.agents.our_agents.policy_imitation.shaping import (
    EventRewardShaper,
    PotentialRewardShaper,
)
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent

from .bc_network import ACTION_DIM, BehaviorCloningNetwork, FixedNormalizer
from .bc_training import DeviceName, load_bc_checkpoint, resolve_device
from .policies import CandidateSpec
from .protocol import RuntimeSnapshot, isolated_seed, seed_everything
from .runner import TeacherDecisionError, select_action

ValueMode = Literal["return", "outcome"]
InitializationMode = Literal["bc", "scratch"]
ADVANTAGE_STD_EPSILON = 1e-8
EXPLAINED_VARIANCE_EPSILON = 1e-12


def _validate_value_mode(value_mode: str) -> ValueMode:
    """Validate and narrow the two checkpoint-compatible critic semantics."""
    if value_mode not in {"return", "outcome"}:
        raise ValueError("value_mode must be 'return' or 'outcome'")
    return cast(ValueMode, value_mode)


def _validate_initialization(initialization: str) -> InitializationMode:
    """Validate the policy initialization mode."""
    if initialization not in {"bc", "scratch"}:
        raise ValueError("initialization must be 'bc' or 'scratch'")
    return cast(InitializationMode, initialization)


MIN_SEATS = 2
MAX_SEATS = 4


class PolicyValueNetwork(nn.Module):
    """Masked policy and configurable value heads for imitation PPO.

    New models use an unbounded return critic.  ``outcome`` is retained for
    loading the historical tanh-bounded checkpoints whose value semantics were
    trained around terminal outcome targets.
    """

    def __init__(  # noqa: PLR0913 - mirrors the checkpoint config surface
        self,
        input_dim: int,
        *,
        feature_version: str,
        hidden_layers: tuple[int, ...] = HIDDEN_DIMS,
        output_dim: int = ACTION_DIM,
        value_mode: ValueMode = "return",
        critic_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        expected_dim = observation_dim(feature_version)
        if input_dim != expected_dim:
            raise ValueError(
                f"feature schema {feature_version!r} requires {expected_dim} inputs, got {input_dim}"
            )
        if output_dim != ACTION_DIM:
            raise ValueError(f"PPO action head must have {ACTION_DIM} outputs")
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        if critic_hidden_dim < 0:
            raise ValueError("critic_hidden_dim must be non-negative")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_version = feature_version
        self.hidden_layers = hidden_layers
        self.critic_hidden_dim = critic_hidden_dim
        self.value_mode = _validate_value_mode(value_mode)
        self.normalizer = FixedNormalizer(input_dim)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()]
            )
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(previous, output_dim)
        if critic_hidden_dim:
            # Roadmap C1 ablation: a critic-private hidden layer on top of the
            # shared trunk, so value fitting cannot fight the policy trunk.
            self.value_head: nn.Module = nn.Sequential(
                nn.Linear(previous, critic_hidden_dim),
                nn.LayerNorm(critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim, 1),
            )
        else:
            self.value_head = nn.Linear(previous, 1)
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        value_final = (
            self.value_head[-1] if isinstance(self.value_head, nn.Sequential) else self.value_head
        )
        assert isinstance(value_final, nn.Linear)
        nn.init.orthogonal_(value_final.weight, gain=1.0)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use the same orthogonal MLP initialization as the repository PPO."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()

    @classmethod
    def from_bc(
        cls,
        bc_model: BehaviorCloningNetwork,
        *,
        initialization: InitializationMode = "bc",
        value_mode: ValueMode = "return",
        critic_hidden_dim: int = 0,
    ) -> "PolicyValueNetwork":
        """Build PPO from BC while keeping normalizer/schema provenance.

        ``scratch`` intentionally keeps the BC normalizer and input schema but
        does not copy the BC trunk or policy head, so a comparison changes only
        policy/trunk initialization while using the same source statistics.
        """
        _validate_initialization(initialization)
        model = cls(
            bc_model.input_dim,
            feature_version=bc_model.feature_version,
            hidden_layers=bc_model.hidden_layers,
            output_dim=bc_model.output_dim,
            value_mode=value_mode,
            critic_hidden_dim=critic_hidden_dim,
        )
        model.normalizer.mean.copy_(bc_model.normalizer.mean)
        model.normalizer.variance.copy_(bc_model.normalizer.variance)
        model.normalizer.fitted = bc_model.normalizer.fitted
        if initialization == "bc":
            model.trunk.load_state_dict(deepcopy(bc_model.trunk.state_dict()))
            model.policy_head.load_state_dict(deepcopy(bc_model.policy_head.state_dict()))
        return model

    def _hidden(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        return self.trunk(self.normalizer(observations))

    def forward(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked action logits and values in the configured mode."""
        if legal_masks.dim() == 1:
            legal_masks = legal_masks.unsqueeze(0)
        hidden = self._hidden(observations)
        logits = self.policy_head(hidden)
        if logits.shape != legal_masks.shape:
            raise ValueError(
                f"logits and masks must have equal shape, got {logits.shape} and {legal_masks.shape}"
            )
        if not torch.all(legal_masks.sum(dim=1) > 0):
            raise ValueError("each PPO row must contain at least one legal action")
        values = self.value_head(hidden).squeeze(-1)
        if self.value_mode == "outcome":
            values = torch.tanh(values)
        return logits.masked_fill(legal_masks <= 0, HUGE_NEG), values


class PPOPolicyAgent(Agent):
    """Greedy frozen-policy wrapper used by evaluation and opponent pools."""

    def __init__(
        self,
        _id: int,
        *,
        model: PolicyValueNetwork,
        device_name: DeviceName = "cpu",
        stochastic: bool = False,
    ) -> None:
        super().__init__(_id)
        self.device = resolve_device(device_name)
        self.model = model.to(self.device).eval()
        self.stochastic = stochastic

    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """Select a legal action from the frozen policy/value network."""
        del game_rule
        observation = extract_observation(
            game_state, self.id, self.model.feature_version
        )
        mask = create_legal_actions_mask(actions, game_state, self.id).astype(
            np.float32
        )
        with torch.no_grad():
            logits, _ = self.model(
                torch.from_numpy(observation).to(self.device),
                torch.from_numpy(mask).to(self.device),
            )
            if self.stochastic:
                action_index = int(distributions.Categorical(logits=logits).sample().item())
            else:
                action_index = int(logits.argmax(dim=-1).item())
        return create_action_mapping(actions, game_state, self.id)[action_index]


def build_policy_candidate(
    model: PolicyValueNetwork,
    *,
    name: str = "ppo-policy",
    snapshot: str | None = None,
    device_name: DeviceName = "cpu",
) -> CandidateSpec:
    """Freeze a policy/value model as an isolated fixed candidate."""
    device = resolve_device(device_name)
    template = deepcopy(model).to(device).eval()
    template.requires_grad_(False)

    def factory(agent_id: int) -> PPOPolicyAgent:
        return PPOPolicyAgent(
            agent_id,
            model=deepcopy(template).to(device).eval(),
            device_name=device_name,
        )

    return CandidateSpec(
        name=name,
        role="fixed_baseline",
        factory=factory,
        feature_version=model.feature_version,
        snapshot=snapshot,
    )


@dataclass(frozen=True)
class PPOConfig:
    """Declared PPO update budget and terminal-value semantics."""

    feature_version: str = "v1"
    hidden_layers: tuple[int, ...] = HIDDEN_DIMS
    value_mode: ValueMode = "return"
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    minibatch_size: int = 256
    update_epochs: int = 4
    updates: int = 10
    games_per_update: int = 4
    terminal_value: float = 10.0
    target_kl: float | None = 0.02
    reference_kl_coefficient: float = 0.0
    current_weight: float = 1.0
    history_weight: float = 1.0
    history_limit: int = 4
    critic_warmup_epochs: int = 0
    # Roadmap C1 critic-repair knobs; defaults preserve the shared-lr model.
    critic_learning_rate: float | None = None
    critic_hidden_dim: int = 0
    # Roadmap E1: seat count for self-play games (2 by default, frozen run).
    n_seats: int = 2
    # Roadmap B2->C: training-side reward shaping ("none" = frozen run).
    shaping_kind: str = "none"
    shaping_kappa: float = 0.05
    initialization: InitializationMode = "bc"
    eval_every: int = 1
    seed: int = 1234
    device_name: DeviceName = "cpu"

    def __post_init__(self) -> None:  # noqa: C901, PLR0912 - validate each bound
        _validate_value_mode(self.value_mode)
        _validate_initialization(self.initialization)
        if not np.isfinite(self.learning_rate) or not np.isfinite(
            self.discount_factor
        ) or self.learning_rate <= 0 or self.discount_factor <= 0:
            raise ValueError("PPO learning rate and discount must be positive")
        if not np.isfinite(self.gae_lambda) or not np.isfinite(
            self.clip_epsilon
        ) or not 0 < self.gae_lambda <= 1 or not 0 < self.clip_epsilon < 1:
            raise ValueError("invalid GAE or PPO clip value")
        if not np.isfinite(self.entropy_coefficient) or not np.isfinite(
            self.value_coefficient
        ) or self.entropy_coefficient < 0 or self.value_coefficient <= 0:
            raise ValueError("invalid PPO loss coefficients")
        if not np.isfinite(self.max_grad_norm) or not np.isfinite(
            self.terminal_value
        ) or self.max_grad_norm <= 0 or self.terminal_value <= 0:
            raise ValueError("gradient limit and terminal value must be positive")
        if min(
            self.minibatch_size,
            self.update_epochs,
            self.updates,
            self.games_per_update,
            self.eval_every,
        ) < 1:
            raise ValueError("PPO budgets must be positive")
        if self.target_kl is not None and (
            not np.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be positive or None")
        for name, value in (
            ("reference_kl_coefficient", self.reference_kl_coefficient),
            ("current_weight", self.current_weight),
            ("history_weight", self.history_weight),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.history_limit < 0 or self.critic_warmup_epochs < 0:
            raise ValueError("history_limit and critic_warmup_epochs must be non-negative")
        if self.critic_learning_rate is not None and (
            not np.isfinite(self.critic_learning_rate)
            or self.critic_learning_rate <= 0
        ):
            raise ValueError("critic_learning_rate must be positive when set")
        if self.critic_hidden_dim < 0:
            raise ValueError("critic_hidden_dim must be non-negative")
        if not MIN_SEATS <= self.n_seats <= MAX_SEATS:
            raise ValueError("n_seats must lie in [2, 4]")
        if self.shaping_kind not in ("none", "potential", "event"):
            raise ValueError(f"unknown shaping kind {self.shaping_kind!r}")
        if self.value_mode == "outcome" and self.terminal_value > 1.0:
            raise ValueError(
                "value_mode='outcome' bounds the critic to [-1, 1] via tanh, "
                f"so terminal_value={self.terminal_value} is unrepresentable "
                "(the 2026-09-07 era ran exactly this mismatch; use "
                "value_mode='return' for score/terminal-scale targets)"
            )
        if not np.isfinite(self.shaping_kappa) or self.shaping_kappa < 0:
            raise ValueError("shaping_kappa must be finite and non-negative")
        if self.n_seats > MIN_SEATS and self.feature_version != "public-v2-multi":
            raise ValueError(
                "training with more than two seats requires the "
                "public-v2-multi feature schema (roadmap B1/E1)"
            )
        if self.seed < 0:
            raise ValueError("PPO seed must be non-negative")


@dataclass(frozen=True)
class OpponentPoolEntry:
    """One complete-game opponent choice with an explicit sampling weight."""

    name: str
    candidate: CandidateSpec
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("opponent pool entry needs a name")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("opponent pool weights must be finite and non-negative")


@dataclass(frozen=True)
class PPOTransition:
    """One focal-policy transition with old-policy statistics for PPO."""

    observation: np.ndarray
    legal_mask: np.ndarray
    action_index: int
    old_log_probability: float
    old_value: float
    reward: float
    terminal: bool
    seed: int
    seat: int


def _sample_pool_entry(
    entries: Sequence[OpponentPoolEntry], rng: random.Random
) -> OpponentPoolEntry:
    """Choose exactly one pool policy for an entire game."""
    eligible = [entry for entry in entries if entry.weight > 0]
    if not eligible:
        raise ValueError("PPO opponent pool must not be empty")
    return rng.choices(
        eligible, weights=[entry.weight for entry in eligible], k=1
    )[0]


def build_opponent_pool(  # noqa: PLR0913 - explicit opponent bucket controls
    fixed_entries: Sequence[OpponentPoolEntry],
    history_entries: Sequence[OpponentPoolEntry] = (),
    current_entry: OpponentPoolEntry | None = None,
    *,
    current_weight: float = 1.0,
    history_weight: float = 1.0,
    history_limit: int = 4,
) -> list[OpponentPoolEntry]:
    """Build a non-drifting fixed/current/history sampling pool.

    Fixed entries retain their declared weights.  The current policy receives
    one current-bucket mass, while the retained history receives one shared
    history-bucket mass divided equally across the most recent snapshots.  If
    there is no retained history, the history mass is explicitly transferred
    to the current entry; callers can record that decision in their run log.
    Zero-weight entries are retained for auditable configuration but are never
    sampled.
    """
    if history_limit < 0:
        raise ValueError("history_limit must be non-negative")
    for name, weight in (
        ("current_weight", current_weight),
        ("history_weight", history_weight),
    ):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    retained_history = (
        list(history_entries[-history_limit:]) if history_limit else []
    )
    if not retained_history and history_weight > 0 and current_entry is None:
        raise ValueError("history mass cannot be redistributed without current_entry")

    result: list[OpponentPoolEntry] = list(fixed_entries)
    names = {entry.name for entry in result}
    if len(names) != len(result):
        raise ValueError("opponent pool entry names must be unique")

    redistributed_history = history_weight if not retained_history else 0.0
    if current_entry is not None:
        if current_entry.name in names:
            raise ValueError("opponent pool entry names must be unique")
        names.add(current_entry.name)
        result.append(
            replace(
                current_entry,
                weight=current_weight + redistributed_history,
            )
        )

    history_share = history_weight / len(retained_history) if retained_history else 0.0
    for entry in retained_history:
        if entry.name in names:
            raise ValueError("opponent pool entry names must be unique")
        names.add(entry.name)
        result.append(replace(entry, weight=history_share))
    return result


def _policy_step(
    model: PolicyValueNetwork,
    state: SplendorState,
    actions: list[ActionType],
    seat: int,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int, ActionType, float, float]:
    """Sample one focal action and return its observation/log-prob/value."""
    observation = extract_observation(state, seat, model.feature_version)
    legal_mask = create_legal_actions_mask(actions, state, seat).astype(np.uint8)
    observation_tensor = torch.from_numpy(observation).to(device)
    mask_tensor = torch.from_numpy(legal_mask.astype(np.float32)).to(device)
    with torch.no_grad():
        logits, value = model(observation_tensor, mask_tensor)
        distribution = distributions.Categorical(logits=logits)
        sampled = distribution.sample()
        log_probability = float(distribution.log_prob(sampled).item())
        state_value = float(value.item())
    action_index = int(sampled.item())
    action = create_action_mapping(actions, state, seat)[action_index]
    return (
        observation,
        legal_mask,
        action_index,
        action,
        log_probability,
        state_value,
    )


def _outcome(rule: SplendorGameRule, seat: int) -> int:
    """Use calScore's card-count tie-break for terminal utility.

    Seat-count agnostic: the outcome is judged against the *best* rival, so
    two-player games keep the original semantics exactly.
    """
    state = rule.current_game_state
    own = float(rule.calScore(state, seat))
    best_rival = max(
        (
            float(rule.calScore(state, agent.id))
            for agent in state.agents
            if agent.id != seat
        ),
        default=own,
    )
    return int(own > best_rival) - int(own < best_rival)


def collect_ppo_game(  # noqa: C901,PLR0912,PLR0913,PLR0915 - game accounting is explicit
    model: PolicyValueNetwork,
    opponent_pool: Sequence[OpponentPoolEntry],
    *,
    seed: int,
    seat: int,
    config: PPOConfig,
    update_index: int,
    game_index: int,
) -> tuple[dict[str, Any], list[PPOTransition]]:
    """Collect one on-policy game without teacher labels or policy switching."""
    n_seats = config.n_seats
    if seat not in range(n_seats):
        raise ValueError(f"seat must lie in [0, {n_seats}), got {seat}")
    device = next(model.parameters()).device
    pool_rng = random.Random(seed + 1_000_003 * (update_index + 1) + game_index)
    selected_entry = _sample_pool_entry(opponent_pool, pool_rng)
    rivals: dict[int, Any] = {}

    def rival_for(turn: int) -> Agent:
        """Build each rival seat once per game from the sampled entry."""
        if turn not in rivals:
            rivals[turn] = selected_entry.candidate.build(turn)
        return rivals[turn]
    transitions: list[PPOTransition] = []
    failure: dict[str, str] | None = None
    opponent_queries = 0
    opponent_illegal = 0
    opponent_search_nodes = 0
    opponent_latencies: list[float] = []
    started = time.perf_counter()

    with isolated_seed(seed):
        rule = LimitRoundsGameRule(n_seats)
        # Roadmap B2->C: potential shaping is anchored at the first focal
        # decision and credits each transition when the next focal state (or
        # the terminal state) arrives - gamma*phi(s') - phi(s) telescopes.
        potential_shaper: PotentialRewardShaper | None = None
        event_shaper: EventRewardShaper | None = None
        if config.shaping_kind == "potential":
            potential_shaper = PotentialRewardShaper(
                kappa=config.shaping_kappa,
                discount_factor=config.discount_factor,
            )
        elif config.shaping_kind == "event":
            event_shaper = EventRewardShaper()
        shaper_anchored = False
        pending_bonus_index: int | None = None
        while not rule.gameEnds():
            state = rule.current_game_state
            turn = rule.current_agent_index
            legal_actions = rule.getLegalActions(state, turn)
            if turn == seat:
                if potential_shaper is not None:
                    if not shaper_anchored:
                        potential_shaper.reset(state, seat, rule)
                        shaper_anchored = True
                    elif pending_bonus_index is not None:
                        shaping_bonus = potential_shaper.advance(state, seat, rule)
                        transitions[pending_bonus_index] = replace(
                            transitions[pending_bonus_index],
                            reward=transitions[pending_bonus_index].reward + shaping_bonus,
                        )
                        pending_bonus_index = None
                score_before = state.agents[seat].score
                try:
                    (
                        observation,
                        legal_mask,
                        action_index,
                        action,
                        old_log_probability,
                        old_value,
                    ) = _policy_step(model, state, legal_actions, seat, device=device)
                except Exception as exc:
                    failure = {
                        "side": "student",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                rule.update(action)
                base_reward = float(
                    rule.current_game_state.agents[seat].score - score_before
                )
                if event_shaper is not None:
                    base_reward += event_shaper.bonus(
                        action, rule.current_game_state, rule, seat
                    )
                transitions.append(
                    PPOTransition(
                        observation=observation,
                        legal_mask=legal_mask,
                        action_index=action_index,
                        old_log_probability=old_log_probability,
                        old_value=old_value,
                        reward=base_reward,
                        terminal=rule.gameEnds(),
                        seed=seed,
                        seat=seat,
                    )
                )
                pending_bonus_index = len(transitions) - 1
            else:
                opponent_queries += 1
                try:
                    decision = select_action(rival_for(turn), legal_actions, state, rule)
                except TeacherDecisionError as exc:
                    opponent_illegal += int(exc.illegal)
                    opponent_search_nodes += exc.search_nodes
                    opponent_latencies.append(exc.elapsed_seconds)
                    failure = {
                        "side": "opponent",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    break
                opponent_latencies.append(decision.elapsed_seconds)
                opponent_search_nodes += decision.search_nodes
                rule.update(decision.action)

    completed = failure is None and rule.gameEnds()
    terminal_outcome = _outcome(rule, seat) if completed else None
    if completed and potential_shaper is not None and pending_bonus_index is not None:
        shaping_bonus = potential_shaper.advance(
            rule.current_game_state, seat, rule
        )
        transitions[pending_bonus_index] = replace(
            transitions[pending_bonus_index],
            reward=transitions[pending_bonus_index].reward + shaping_bonus,
        )
        pending_bonus_index = None
    if completed and transitions:
        assert terminal_outcome is not None
        last = transitions[-1]
        transitions[-1] = replace(
            last,
            reward=last.reward + config.terminal_value * int(terminal_outcome),
            terminal=True,
        )
    state = rule.current_game_state
    score = float(rule.calScore(state, seat))
    rival_score = max(
        (
            float(rule.calScore(state, agent.id))
            for agent in state.agents
            if agent.id != seat
        ),
        default=0.0,
    )
    record: dict[str, Any] = {
        "update": update_index,
        "game_index": game_index,
        "seed": seed,
        "seat": seat,
        "opponent": selected_entry.name,
        "opponent_snapshot": selected_entry.candidate.snapshot,
        "status": "completed" if completed else "failed",
        "failure": failure,
        "outcome": terminal_outcome,
        "score": score,
        "rival_score": rival_score,
        "plies": rule.action_counter,
        "student_queries": len(transitions),
        "teacher_queries": 0,
        "opponent_queries": opponent_queries,
        "opponent_illegal_actions": opponent_illegal,
        "opponent_search_nodes": opponent_search_nodes,
        "opponent_action_seconds_mean": float(np.mean(opponent_latencies))
        if opponent_latencies
        else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "terminal_value_mapping": {-1: -config.terminal_value, 0: 0.0, 1: config.terminal_value},
    }
    # A failed game can contain useful diagnostics but it is not an on-policy
    # trajectory.  Never let a caller accidentally optimize on its prefix.
    return record, transitions if completed else []


def compute_gae(
    transitions: Sequence[PPOTransition],
    *,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute terminal-aware GAE across a batch of complete games."""
    if not transitions:
        raise ValueError("cannot compute GAE for an empty rollout")
    values = np.asarray(
        [transition.old_value for transition in transitions], dtype=np.float32
    )
    rewards = np.asarray(
        [transition.reward for transition in transitions], dtype=np.float32
    )
    terminals = np.asarray(
        [transition.terminal for transition in transitions], dtype=np.bool_
    )
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise FloatingPointError("non-finite PPO value or reward in GAE inputs")
    advantages = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        is_terminal = bool(terminals[index])
        if is_terminal:
            next_value = 0.0
            continuation = 0.0
        else:
            has_next_transition = index + 1 < len(values)
            next_value = float(values[index + 1]) if has_next_transition else 0.0
            continuation = float(has_next_transition)
        delta = float(rewards[index]) + discount_factor * next_value - float(values[index])
        running = delta + discount_factor * gae_lambda * continuation * running
        advantages[index] = running
    returns = advantages + values
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise FloatingPointError("non-finite PPO advantages or return targets")
    return advantages, returns


def _policy_logits(
    policy: BehaviorCloningNetwork | PolicyValueNetwork,
    observations: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Get masked logits from either a BC reference or PPO policy."""
    if isinstance(policy, BehaviorCloningNetwork):
        return policy(observations, legal_masks)
    logits, _ = policy(observations, legal_masks)
    return logits


def _critic_values_from_hidden(
    model: PolicyValueNetwork, hidden: torch.Tensor
) -> torch.Tensor:
    """Apply the configured value semantics without running the policy head."""
    values = model.value_head(hidden).squeeze(-1)
    if model.value_mode == "outcome":
        values = torch.tanh(values)
    return values


def warmup_critic(  # noqa: C901 - explicit finite-value and warmup safety checks
    model: PolicyValueNetwork,
    transitions: Sequence[PPOTransition],
    config: PPOConfig,
) -> dict[str, float | int]:
    """Fit only ``value_head`` to first-rollout return targets.

    The hidden representation is detached and a separate optimizer owns only
    the value head.  Thus BC's pretrained trunk and policy head cannot change
    during critic warmup.
    """
    if not transitions:
        raise ValueError("critic warmup needs at least one transition")
    if config.critic_warmup_epochs < 1:
        raise ValueError("critic warmup requires critic_warmup_epochs >= 1")
    _, returns = compute_gae(
        transitions,
        discount_factor=config.discount_factor,
        gae_lambda=config.gae_lambda,
    )
    if not np.isfinite(returns).all():
        raise FloatingPointError("non-finite critic warmup return targets")
    device = next(model.parameters()).device
    observations = torch.from_numpy(
        np.stack([transition.observation for transition in transitions]).astype(
            np.float32
        )
    ).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    model.eval()
    with torch.no_grad():
        hidden = model.trunk(model.normalizer(observations)).detach()
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("non-finite critic warmup hidden states")
    warmup_optimizer = optim.Adam(
        model.value_head.parameters(),
        lr=config.critic_learning_rate or config.learning_rate,
    )
    losses: list[float] = []
    for _ in range(config.critic_warmup_epochs):
        predicted = _critic_values_from_hidden(model, hidden)
        loss = F.smooth_l1_loss(predicted, return_tensor)
        if not torch.isfinite(predicted).all() or not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite critic warmup value or loss")
        warmup_optimizer.zero_grad()
        loss.backward()
        for parameter in model.value_head.parameters():
            if parameter.grad is not None and not torch.isfinite(
                parameter.grad
            ).all():
                raise FloatingPointError("non-finite critic warmup gradient")
        nn.utils.clip_grad_norm_(model.value_head.parameters(), config.max_grad_norm)
        warmup_optimizer.step()
        for parameter in model.value_head.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("non-finite critic warmup parameter")
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return {
        "epochs": config.critic_warmup_epochs,
        "optimizer_steps": config.critic_warmup_epochs,
        "loss": float(np.mean(losses)),
        "target_min": float(np.min(returns)),
        "target_max": float(np.max(returns)),
    }


def ppo_update(  # noqa: C901, PLR0912, PLR0913, PLR0915 - PPO accounting is explicit
    model: PolicyValueNetwork,
    optimizer: optim.Optimizer,
    transitions: Sequence[PPOTransition],
    config: PPOConfig,
    *,
    update_seed: int,
    reference_model: BehaviorCloningNetwork | PolicyValueNetwork | None = None,
) -> dict[str, float | int | bool]:
    """Apply clipped PPO updates to one frozen-policy rollout batch.

    The KL diagnostic uses the non-negative Schulman approximation
    ``((ratio - 1) - log_ratio)`` before each optimizer step.  If it exceeds
    ``target_kl``, the current minibatch is not applied and the update stops.
    """
    if not transitions:
        raise ValueError("PPO update needs at least one transition")
    if reference_model is model:
        raise ValueError("reference_model must be independent of the on-policy model")
    advantages, returns = compute_gae(
        transitions,
        discount_factor=config.discount_factor,
        gae_lambda=config.gae_lambda,
    )
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + ADVANTAGE_STD_EPSILON
        )
    device = next(model.parameters()).device
    observations = torch.from_numpy(
        np.stack([transition.observation for transition in transitions]).astype(np.float32)
    ).to(device)
    masks = torch.from_numpy(
        np.stack([transition.legal_mask for transition in transitions]).astype(np.float32)
    ).to(device)
    actions = torch.tensor(
        [transition.action_index for transition in transitions], dtype=torch.long, device=device
    )
    old_log_probabilities = torch.tensor(
        [transition.old_log_probability for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    advantage_tensor = torch.from_numpy(advantages).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    if not torch.isfinite(advantage_tensor).all() or not torch.isfinite(
        return_tensor
    ).all():
        raise FloatingPointError("non-finite PPO advantage or return target")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(update_seed)
    model.train()
    if reference_model is not None:
        reference_model.eval()
    metrics: list[dict[str, float]] = []
    optimizer_steps = 0
    early_stopped = False
    epochs_completed = 0
    for _ in range(config.update_epochs):
        epochs_completed += 1
        permutation = torch.randperm(len(transitions), generator=generator)
        for start in range(0, len(transitions), config.minibatch_size):
            batch = permutation[start : start + config.minibatch_size].to(device)
            logits, values = model(observations[batch], masks[batch])
            if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                raise FloatingPointError("non-finite PPO policy logits or values")
            distribution = distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions[batch])
            log_ratio = log_probabilities - old_log_probabilities[batch]
            ratio = log_ratio.exp()
            if not torch.isfinite(log_probabilities).all() or not torch.isfinite(
                ratio
            ).all():
                raise FloatingPointError("non-finite PPO log-probability or ratio")
            approx_kl_tensor = (ratio - 1.0) - log_ratio
            approx_kl = float(
                torch.clamp(approx_kl_tensor.mean(), min=0.0)
                .detach()
                .cpu()
                .item()
            )
            clip_fraction = float(
                (torch.abs(ratio - 1.0) > config.clip_epsilon)
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            )
            surrogate_one = ratio * advantage_tensor[batch]
            surrogate_two = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * advantage_tensor[batch]
            policy_loss = -torch.minimum(surrogate_one, surrogate_two).mean()
            value_loss = F.smooth_l1_loss(values, return_tensor[batch])
            entropy = distribution.entropy().mean()
            reference_kl = torch.zeros((), device=device)
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits = _policy_logits(
                        reference_model,
                        observations[batch],
                        masks[batch],
                    )
                if not torch.isfinite(reference_logits).all():
                    raise FloatingPointError("non-finite reference policy logits")
                reference_distribution = distributions.Categorical(
                    logits=reference_logits
                )
                reference_kl = distributions.kl_divergence(
                    reference_distribution, distribution
                ).mean()
            reference_kl_value = float(
                torch.clamp(reference_kl.detach(), min=0.0).cpu().item()
            )
            loss_values = (
                approx_kl,
                clip_fraction,
                float(policy_loss.detach().cpu().item()),
                float(value_loss.detach().cpu().item()),
                float(entropy.detach().cpu().item()),
                reference_kl_value,
            )
            if not all(np.isfinite(value) for value in loss_values):
                raise FloatingPointError("non-finite PPO loss or diagnostic")
            metrics.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "reference_kl": reference_kl_value,
                }
            )
            if config.target_kl is not None and approx_kl > config.target_kl:
                early_stopped = True
                break
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + config.reference_kl_coefficient * reference_kl
            )
            optimizer.zero_grad()
            loss.backward()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("non-finite PPO loss")
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(
                    parameter.grad
                ).all():
                    raise FloatingPointError("non-finite PPO loss gradient")
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError("non-finite PPO parameter after update")
            optimizer_steps += 1
        if early_stopped:
            break
    if not metrics:
        raise RuntimeError("PPO update produced no minibatches")
    model.eval()
    with torch.no_grad():
        _, final_values = model(observations, masks)
    model.train()
    if not torch.isfinite(final_values).all():
        raise FloatingPointError("non-finite PPO values after update")
    predicted_values = final_values.detach().cpu().numpy()
    return_variance = float(np.var(returns))
    explained_variance = (
        0.0
        if return_variance <= EXPLAINED_VARIANCE_EPSILON
        else 1.0 - float(np.var(returns - predicted_values)) / return_variance
    )
    if not np.isfinite(explained_variance):
        raise FloatingPointError("non-finite PPO explained variance")
    aggregate_metrics = {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }
    return {
        "samples": len(transitions),
        **aggregate_metrics,
        "advantage_mean": float(np.mean(advantages)),
        "return_mean": float(np.mean(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "value_mean": float(np.mean(predicted_values)),
        "value_min": float(np.min(predicted_values)),
        "value_max": float(np.max(predicted_values)),
        "explained_variance": explained_variance,
        "optimizer_steps": optimizer_steps,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
    }


def save_ppo_checkpoint(  # noqa: PLR0913 - provenance fields are explicit
    model: PolicyValueNetwork,
    path: Path,
    *,
    update: int,
    config: PPOConfig,
    source_bc: str,
    opponent_pool: Sequence[OpponentPoolEntry],
    metrics: dict[str, Any],
) -> None:
    """Save policy, critic, normalizer, source BC and opponent provenance."""
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload: dict[str, Any] = {
        "model_type": "imitation_ppo_policy_value",
        "model_state_dict": state_dict,
        "update": update,
        "value_mode": model.value_mode,
        "config": {
            **asdict(config),
            "hidden_layers": list(config.hidden_layers),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "value_mode": model.value_mode,
            "normalizer_fitted": model.normalizer.fitted,
        },
        "source_bc": source_bc,
        "normalizer_source": source_bc,
        "initialization": config.initialization,
        "opponent_pool": [
            {
                "name": entry.name,
                "weight": entry.weight,
                "snapshot": entry.candidate.snapshot,
            }
            for entry in opponent_pool
        ],
        "metrics": metrics,
        "runtime": RuntimeSnapshot.collect(config.device_name).as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ppo_checkpoint(
    path: Path,
    *,
    device_name: DeviceName = "cpu",
) -> PolicyValueNetwork:
    """Load and validate an imitation-PPO policy/value checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "imitation_ppo_policy_value":
        raise ValueError("checkpoint is not an imitation PPO policy/value model")
    config = checkpoint.get("config") or {}
    # Checkpoints written before value_mode existed used the tanh outcome
    # critic.  Preserve that behavior explicitly instead of silently changing
    # the meaning of their value head on load.
    value_mode = _validate_value_mode(
        str(config.get("value_mode", checkpoint.get("value_mode", "outcome")))
    )
    model = PolicyValueNetwork(
        int(config["input_dim"]),
        feature_version=str(config["feature_version"]),
        hidden_layers=tuple(config["hidden_layers"]),
        output_dim=int(config["output_dim"]),
        value_mode=value_mode,
        critic_hidden_dim=int(config.get("critic_hidden_dim", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.normalizer.fitted = bool(config.get("normalizer_fitted", False))
    return model.to(resolve_device(device_name)).eval()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an incremental run artifact with JSON-safe fallback formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _validation_score(validation: dict[str, Any]) -> tuple[int, int]:
    """Return exact integer W/D/L coordinates after fail-closed auditing."""
    if not validation:
        raise ValueError("validation matrix is empty")
    reports = list(validation.values())
    scheduled_counts: list[int] = []
    for report in reports:
        try:
            scheduled = int(report["games"])
            completed = int(report["completed_games"])
            failed = int(report["failed_games"])
            candidate_illegal = int(report["candidate_illegal_actions"])
            opponent_illegal = int(report["opponent_illegal_actions"])
            wins = int(report["wins"])
            draws = int(report["draws"])
            losses = int(report["losses"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("validation report is missing audit fields") from exc
        if scheduled <= 0:
            raise ValueError("validation report has zero scheduled games")
        if completed != scheduled or failed != 0:
            raise ValueError("validation report contains failed or incomplete games")
        if candidate_illegal != 0 or opponent_illegal != 0:
            raise ValueError("validation report contains illegal actions")
        if wins < 0 or draws < 0 or losses < 0 or wins + draws + losses != scheduled:
            raise ValueError("validation W/D/L counts do not match scheduled games")
        scheduled_counts.append(scheduled)
    if len(set(scheduled_counts)) != 1:
        raise ValueError("validation reports have unequal scheduled game counts")
    wins = sum(int(report["wins"]) for report in reports)
    games = sum(int(report["games"]) for report in reports)
    return wins, -games


def _pool_metadata(
    fixed_entries: Sequence[OpponentPoolEntry],
    history_entries: Sequence[OpponentPoolEntry],
    current_entry: OpponentPoolEntry,
    config: PPOConfig,
) -> dict[str, Any]:
    """Describe pool bucket mass so redistribution is auditable."""
    retained_history = (
        list(history_entries[-config.history_limit:]) if config.history_limit else []
    )
    redistributed = bool(config.history_weight > 0 and not retained_history)
    current_mass = config.current_weight + (
        config.history_weight if redistributed else 0.0
    )
    history_mass = config.history_weight if retained_history else 0.0
    return {
        "fixed": [
            {"name": entry.name, "weight": entry.weight}
            for entry in fixed_entries
        ],
        "current": {"name": current_entry.name, "weight": current_mass},
        "history": [
            {"name": entry.name, "weight": history_mass / len(retained_history)}
            for entry in retained_history
        ],
        "current_weight_requested": config.current_weight,
        "history_weight_requested": config.history_weight,
        "history_weight_allocated": history_mass,
        "history_weight_redistributed_to_current": redistributed,
        "history_limit": config.history_limit,
        "total_weight": float(
            sum(entry.weight for entry in fixed_entries)
            + current_mass
            + history_mass
        ),
    }


def _build_initial_model(
    bc_model: BehaviorCloningNetwork, config: PPOConfig
) -> PolicyValueNetwork:
    """Build the PPO model from the BC initializer (roadmap E1).

    Same schema: the established from_bc path (including normalizer transfer).
    Cross-schema with ``scratch`` initialization: build directly at the
    target schema with an unfitted (identity) normalizer - the only way to
    cold-start 3/4-seat training before a multi-seat BC exists.  Any other
    cross-schema combination stays a hard error.
    """
    if bc_model.feature_version == config.feature_version:
        return PolicyValueNetwork.from_bc(
            bc_model,
            initialization=config.initialization,
            value_mode=config.value_mode,
            critic_hidden_dim=config.critic_hidden_dim,
        )
    if config.initialization != "scratch":
        raise ValueError(
            f"BC checkpoint schema {bc_model.feature_version!r} differs from "
            f"config schema {config.feature_version!r}; cross-schema starts "
            "require initialization='scratch'"
        )
    return PolicyValueNetwork(
        observation_dim(config.feature_version),
        feature_version=config.feature_version,
        hidden_layers=config.hidden_layers,
        value_mode=config.value_mode,
        critic_hidden_dim=config.critic_hidden_dim,
    )


def train_ppo_selfplay(  # noqa: C901, PLR0912, PLR0913, PLR0915 - lifecycle is explicit
    initial_bc: Path,
    output_dir: Path,
    training_seeds: Sequence[int],
    opponent_pool: Sequence[OpponentPoolEntry],
    *,
    config: PPOConfig,
    validation_seeds: Sequence[int] | None = None,
    validation_opponents: Sequence[CandidateSpec] = (),
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Train PPO from BC and select only by fixed validation opponents.

    ``status.json`` and a partial ``result.json`` are refreshed after every
    completed update.  The final result keeps the complete update-0-through-N
    log, including validation and pool provenance.
    """
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("PPO training seeds must be nonempty and unique")
    if not opponent_pool:
        raise ValueError("PPO self-play needs a non-empty opponent pool")
    if validation_seeds and not validation_opponents:
        raise ValueError("validation opponents are required with validation seeds")
    seed_everything(config.seed)
    device = resolve_device(config.device_name)
    bc_model = load_bc_checkpoint(initial_bc, device_name=config.device_name)
    model = _build_initial_model(bc_model, config).to(device)
    reference_model: BehaviorCloningNetwork | None = None
    if config.reference_kl_coefficient > 0:
        reference_model = bc_model.to(device).eval()
        reference_model.requires_grad_(False)
    if config.critic_learning_rate is not None:
        critic_parameters = list(model.value_head.parameters())
        critic_ids = {id(parameter) for parameter in critic_parameters}
        shared_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in critic_ids
        ]
        optimizer = optim.Adam(
            [
                {"params": shared_parameters, "lr": config.learning_rate},
                {"params": critic_parameters, "lr": config.critic_learning_rate},
            ]
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    result_path = output_dir / "result.json"
    logs: list[dict[str, Any]] = []
    history: list[OpponentPoolEntry] = []
    best_score: tuple[int, int] | None = None
    best_update: int | None = 0
    best_path = output_dir / "best.pth"
    run_started = time.perf_counter()
    validation_enabled = bool(validation_seeds and validation_opponents)

    def result_payload(
        status: str,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build both partial and final result artifacts from one source."""
        payload: dict[str, Any] = {
            "status": status,
            "best": str(best_path),
            "final": str(output_dir / "final.pth"),
            "best_update": best_update,
            "best_validation_score": best_score,
            "config": asdict(config),
            "source_bc": str(initial_bc),
            "normalizer_source": str(initial_bc),
            "source_manifest": source_manifest,
            "training_seeds": [int(seed) for seed in training_seeds],
            "validation_seeds": [int(seed) for seed in validation_seeds]
            if validation_seeds
            else None,
            "logs": logs,
            "elapsed_seconds": time.perf_counter() - run_started,
        }
        if error is not None:
            payload["error"] = error
        return payload

    def write_progress(
        status: str,
        update: int,
        *,
        update_seconds: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Persist current status and the complete log accumulated so far."""
        _write_json(
            status_path,
            {
                "status": status,
                "update": update,
                "updates": config.updates,
                "best_update": best_update,
                "best_validation_score": best_score,
                "update_seconds": update_seconds,
                "elapsed_seconds": time.perf_counter() - run_started,
                "error": error,
            },
        )
        _write_json(result_path, result_payload(status, error=error))

    initial_started = time.perf_counter()
    initial_validation: dict[str, Any] | None = None
    if validation_enabled:
        from .evaluation import evaluate_matrix  # noqa: PLC0415

        initial_candidate = build_policy_candidate(
            model,
            name="ppo-initial",
            snapshot=str(output_dir / "initial.pth"),
            device_name=config.device_name,
        )
        initial_validation = evaluate_matrix(
            [initial_candidate], validation_opponents, validation_seeds or ()
        )[initial_candidate.name]
        best_score = _validation_score(initial_validation)
    initial_record: dict[str, Any] = {
        "update": 0,
        "status": "initial",
        "training_records": [],
        "training_games": 0,
        "training_failed_games": 0,
        "teacher_queries": 0,
        "opponent_names": [],
        "opponent_pool": None,
        "update_metrics": None,
        "critic_warmup": None,
        "validation": initial_validation,
        "elapsed_seconds": time.perf_counter() - initial_started,
    }
    logs.append(initial_record)
    initial_path = output_dir / "initial.pth"
    save_ppo_checkpoint(
        model,
        initial_path,
        update=0,
        config=config,
        source_bc=str(initial_bc),
        opponent_pool=opponent_pool,
        metrics=initial_record,
    )
    save_ppo_checkpoint(
        model,
        best_path,
        update=0,
        config=config,
        source_bc=str(initial_bc),
        opponent_pool=opponent_pool,
        metrics=initial_record,
    )
    write_progress("running", 0, update_seconds=initial_record["elapsed_seconds"])

    for update in range(1, config.updates + 1):
        update_started = time.perf_counter()
        previous_snapshot = (
            output_dir / "initial.pth"
            if update == 1
            else output_dir / f"update-{update - 1}.pth"
        )
        current = build_policy_candidate(
            model,
            name=f"current-update-{update}",
            snapshot=str(previous_snapshot),
            device_name=config.device_name,
        )
        current_entry = OpponentPoolEntry(current.name, current)
        entries = build_opponent_pool(
            opponent_pool,
            history,
            current_entry,
            current_weight=config.current_weight,
            history_weight=config.history_weight,
            history_limit=config.history_limit,
        )
        transitions: list[PPOTransition] = []
        records: list[dict[str, Any]] = []
        rollout_started = time.perf_counter()
        for game_index in range(config.games_per_update):
            seed = int(
                training_seeds[
                    ((update - 1) * config.games_per_update + game_index)
                    % len(training_seeds)
                ]
            )
            seat = (update + game_index) % 2
            try:
                record, game_transitions = collect_ppo_game(
                    model,
                    entries,
                    seed=seed,
                    seat=seat,
                    config=config,
                    update_index=update,
                    game_index=game_index,
                )
            except Exception as exc:  # preserve a failed rollout in the log
                record = {
                    "update": update,
                    "game_index": game_index,
                    "seed": seed,
                    "seat": seat,
                    "status": "failed",
                    "failure": {
                        "side": "rollout",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "elapsed_seconds": time.perf_counter() - rollout_started,
                }
                game_transitions = []
            records.append(record)
            transitions.extend(game_transitions)
        rollout_seconds = time.perf_counter() - rollout_started
        failed_games = [
            record for record in records if record.get("status") != "completed"
        ]
        if failed_games or not transitions:
            failure_message = (
                f"PPO update {update} collected an incomplete rollout; "
                f"failed_games={len(failed_games)}, transitions={len(transitions)}"
            )
            failed_record: dict[str, Any] = {
                "update": update,
                "status": "failed",
                "training_records": records,
                "training_games": len(records),
                "training_failed_games": len(failed_games),
                "teacher_queries": 0,
                "opponent_names": sorted(
                    {
                        str(record["opponent"])
                        for record in records
                        if "opponent" in record
                    }
                ),
                "opponent_pool": _pool_metadata(
                    opponent_pool, history, current_entry, config
                ),
                "update_metrics": None,
                "critic_warmup": None,
                "validation": None,
                "rollout_seconds": rollout_seconds,
                "elapsed_seconds": time.perf_counter() - update_started,
                "error": failure_message,
            }
            logs.append(failed_record)
            write_progress(
                "failed",
                update,
                update_seconds=failed_record["elapsed_seconds"],
                error=failure_message,
            )
            raise RuntimeError(failure_message)

        warmup_metrics: dict[str, float | int] | None = None
        if update == 1 and config.critic_warmup_epochs:
            warmup_metrics = warmup_critic(model, transitions, config)
        update_metrics = ppo_update(
            model,
            optimizer,
            transitions,
            config,
            update_seed=config.seed + update,
            reference_model=reference_model,
        )
        validation: dict[str, Any] | None = None
        should_evaluate = validation_enabled and (
            update % config.eval_every == 0 or update == config.updates
        )
        if should_evaluate:
            from .evaluation import evaluate_matrix  # noqa: PLC0415

            candidate = build_policy_candidate(
                model,
                name="ppo-current",
                snapshot=str(output_dir / f"update-{update}.pth"),
                device_name=config.device_name,
            )
            validation = evaluate_matrix(
                [candidate], validation_opponents, validation_seeds or ()
            )[candidate.name]
        update_record: dict[str, Any] = {
            "update": update,
            "status": "completed",
            "training_records": records,
            "training_games": len(records),
            "training_failed_games": 0,
            "teacher_queries": 0,
            "opponent_names": sorted({record["opponent"] for record in records}),
            "opponent_pool": _pool_metadata(
                opponent_pool, history, current_entry, config
            ),
            "update_metrics": update_metrics,
            "critic_warmup": warmup_metrics,
            "validation": validation,
            "rollout_seconds": rollout_seconds,
            "elapsed_seconds": time.perf_counter() - update_started,
        }
        logs.append(update_record)
        update_path = output_dir / f"update-{update}.pth"
        save_ppo_checkpoint(
            model,
            update_path,
            update=update,
            config=config,
            source_bc=str(initial_bc),
            opponent_pool=entries,
            metrics=update_record,
        )
        if validation is not None:
            score = _validation_score(validation)
            # Validation games are fixed by the manifest.  Compare exact wins
            # only, keeping the earliest checkpoint on a tie.
            if best_score is None or score[0] > best_score[0]:
                best_score = score
                best_update = update
                save_ppo_checkpoint(
                    model,
                    best_path,
                    update=update,
                    config=config,
                    source_bc=str(initial_bc),
                    opponent_pool=entries,
                    metrics=update_record,
                )
        history.append(
            OpponentPoolEntry(
                name=f"history-update-{update}",
                candidate=build_policy_candidate(
                    model,
                    name=f"history-update-{update}",
                    snapshot=str(update_path),
                    device_name=config.device_name,
                ),
            )
        )
        if config.history_limit:
            history = history[-config.history_limit:]
        else:
            history = []
        write_progress(
            "running",
            update,
            update_seconds=update_record["elapsed_seconds"],
        )

    final_path = output_dir / "final.pth"
    save_ppo_checkpoint(
        model,
        final_path,
        update=config.updates,
        config=config,
        source_bc=str(initial_bc),
        opponent_pool=[*opponent_pool, *history],
        metrics={"logs": logs},
    )
    result = result_payload("completed")
    _write_json(
        status_path,
        {
            "status": "completed",
            "update": config.updates,
            "updates": config.updates,
            "best_update": best_update,
            "best_validation_score": best_score,
            "update_seconds": logs[-1].get("elapsed_seconds", 0.0),
            "elapsed_seconds": time.perf_counter() - run_started,
            "error": None,
        },
    )
    _write_json(result_path, result)
    return result
