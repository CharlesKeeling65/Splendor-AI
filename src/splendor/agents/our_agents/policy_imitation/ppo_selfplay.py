"""PPO self-play initialized from BC with a declared fixed opponent pool."""

import json
import random
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions, nn, optim

from splendor.agents.our_agents.dqn.constants import HIDDEN_DIMS, HUGE_NEG
from splendor.agents.our_agents.dqn.features import extract_observation, observation_dim
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


class PolicyValueNetwork(nn.Module):
    """Masked policy and bounded outcome-value heads for imitation PPO."""

    def __init__(
        self,
        input_dim: int,
        *,
        feature_version: str,
        hidden_layers: tuple[int, ...] = HIDDEN_DIMS,
        output_dim: int = ACTION_DIM,
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_version = feature_version
        self.hidden_layers = hidden_layers
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
        self.value_head = nn.Linear(previous, 1)
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use the same orthogonal MLP initialization as the repository PPO."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            module.bias.data.zero_()

    @classmethod
    def from_bc(cls, bc_model: BehaviorCloningNetwork) -> "PolicyValueNetwork":
        """Copy the BC trunk/policy and initialize a fresh outcome critic."""
        model = cls(
            bc_model.input_dim,
            feature_version=bc_model.feature_version,
            hidden_layers=bc_model.hidden_layers,
            output_dim=bc_model.output_dim,
        )
        model.normalizer.mean.copy_(bc_model.normalizer.mean)
        model.normalizer.variance.copy_(bc_model.normalizer.variance)
        model.normalizer.fitted = bc_model.normalizer.fitted
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
        """Return masked action logits and tanh-bounded outcome values."""
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
        return (
            logits.masked_fill(legal_masks <= 0, HUGE_NEG),
            torch.tanh(self.value_head(hidden)).squeeze(-1),
        )


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
    seed: int = 1234
    device_name: DeviceName = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.discount_factor <= 0:
            raise ValueError("PPO learning rate and discount must be positive")
        if not 0 < self.gae_lambda <= 1 or not 0 < self.clip_epsilon < 1:
            raise ValueError("invalid GAE or PPO clip value")
        if self.entropy_coefficient < 0 or self.value_coefficient <= 0:
            raise ValueError("invalid PPO loss coefficients")
        if self.max_grad_norm <= 0 or self.terminal_value <= 0:
            raise ValueError("gradient limit and terminal value must be positive")
        if min(
            self.minibatch_size,
            self.update_epochs,
            self.updates,
            self.games_per_update,
        ) < 1:
            raise ValueError("PPO budgets must be positive")
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
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("opponent pool weights must be finite and positive")


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
    if not entries:
        raise ValueError("PPO opponent pool must not be empty")
    return rng.choices(
        list(entries), weights=[entry.weight for entry in entries], k=1
    )[0]


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
    """Use calScore's card-count tie-break for terminal utility."""
    state = rule.current_game_state
    own = float(rule.calScore(state, seat))
    rival = float(rule.calScore(state, 1 - seat))
    return int(own > rival) - int(own < rival)


def collect_ppo_game(  # noqa: PLR0913,PLR0915 - game accounting is explicit
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
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    device = next(model.parameters()).device
    pool_rng = random.Random(seed + 1_000_003 * (update_index + 1) + game_index)
    selected_entry = _sample_pool_entry(opponent_pool, pool_rng)
    rival = selected_entry.candidate.build(1 - seat)
    transitions: list[PPOTransition] = []
    failure: dict[str, str] | None = None
    opponent_queries = 0
    opponent_illegal = 0
    opponent_search_nodes = 0
    opponent_latencies: list[float] = []
    started = time.perf_counter()

    with isolated_seed(seed):
        rule = LimitRoundsGameRule(2)
        while not rule.gameEnds():
            state = rule.current_game_state
            turn = rule.current_agent_index
            legal_actions = rule.getLegalActions(state, turn)
            if turn == seat:
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
                transitions.append(
                    PPOTransition(
                        observation=observation,
                        legal_mask=legal_mask,
                        action_index=action_index,
                        old_log_probability=old_log_probability,
                        old_value=old_value,
                        reward=float(
                            rule.current_game_state.agents[seat].score - score_before
                        ),
                        terminal=rule.gameEnds(),
                        seed=seed,
                        seat=seat,
                    )
                )
            else:
                opponent_queries += 1
                try:
                    decision = select_action(rival, legal_actions, state, rule)
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
    rival_score = float(rule.calScore(state, 1 - seat))
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
    return record, transitions


def compute_gae(
    transitions: Sequence[PPOTransition],
    *,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute terminal-aware GAE across a batch of complete games."""
    if not transitions:
        raise ValueError("cannot compute GAE for an empty rollout")
    values = np.asarray([transition.old_value for transition in transitions], dtype=np.float32)
    rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
    terminals = np.asarray([transition.terminal for transition in transitions], dtype=np.bool_)
    advantages = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if terminals[index]:
            next_value = 0.0
            continuation = 0.0
        else:
            next_value = float(values[index + 1]) if index + 1 < len(values) else 0.0
            continuation = 1.0
        delta = float(rewards[index]) + discount_factor * next_value - float(values[index])
        running = delta + discount_factor * gae_lambda * continuation * running
        advantages[index] = running
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: PolicyValueNetwork,
    optimizer: optim.Optimizer,
    transitions: Sequence[PPOTransition],
    config: PPOConfig,
    *,
    update_seed: int,
) -> dict[str, float | int]:
    """Apply clipped PPO updates to one frozen-policy rollout batch."""
    if not transitions:
        raise ValueError("PPO update needs at least one transition")
    advantages, returns = compute_gae(
        transitions,
        discount_factor=config.discount_factor,
        gae_lambda=config.gae_lambda,
    )
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
    generator = torch.Generator(device="cpu")
    generator.manual_seed(update_seed)
    model.train()
    metrics: list[dict[str, float]] = []
    for _ in range(config.update_epochs):
        permutation = torch.randperm(len(transitions), generator=generator)
        for start in range(0, len(transitions), config.minibatch_size):
            batch = permutation[start : start + config.minibatch_size].to(device)
            logits, values = model(observations[batch], masks[batch])
            distribution = distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(actions[batch])
            ratio = (log_probabilities - old_log_probabilities[batch]).exp()
            surrogate_one = ratio * advantage_tensor[batch]
            surrogate_two = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * advantage_tensor[batch]
            policy_loss = -torch.minimum(surrogate_one, surrogate_two).mean()
            value_loss = F.smooth_l1_loss(values, return_tensor[batch])
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "approx_kl": float(
                        (old_log_probabilities[batch] - log_probabilities)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                }
            )
    if not metrics:
        raise RuntimeError("PPO update produced no minibatches")
    return {
        "samples": len(transitions),
        **{
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in metrics[0]
        },
        "advantage_mean": float(np.mean(advantages)),
        "return_mean": float(np.mean(returns)),
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
        "config": {
            **asdict(config),
            "hidden_layers": list(config.hidden_layers),
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "normalizer_fitted": model.normalizer.fitted,
        },
        "source_bc": source_bc,
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
    model = PolicyValueNetwork(
        int(config["input_dim"]),
        feature_version=str(config["feature_version"]),
        hidden_layers=tuple(config["hidden_layers"]),
        output_dim=int(config["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.normalizer.fitted = bool(config.get("normalizer_fitted", False))
    return model.to(resolve_device(device_name)).eval()


def train_ppo_selfplay(  # noqa: C901,PLR0913,PLR0915 - lifecycle is explicit
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
    """Train PPO from BC and select only by fixed validation opponents."""
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("PPO training seeds must be nonempty and unique")
    if not opponent_pool:
        raise ValueError("PPO self-play needs a non-empty opponent pool")
    if validation_seeds and not validation_opponents:
        raise ValueError("validation opponents are required with validation seeds")
    seed_everything(config.seed)
    device = resolve_device(config.device_name)
    bc_model = load_bc_checkpoint(initial_bc, device_name=config.device_name)
    if bc_model.feature_version != config.feature_version:
        raise ValueError("BC checkpoint and PPO config feature schemas differ")
    model = PolicyValueNetwork.from_bc(bc_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    history: list[OpponentPoolEntry] = []
    best_score: tuple[int, int] | None = None
    best_update: int | None = None
    best_path = output_dir / "best.pth"
    for update in range(1, config.updates + 1):
        current = build_policy_candidate(
            model,
            name=f"current-update-{update}",
            snapshot=str(output_dir / f"update-{update}.pth"),
            device_name=config.device_name,
        )
        entries = [*opponent_pool, *history, OpponentPoolEntry(current.name, current)]
        transitions: list[PPOTransition] = []
        records: list[dict[str, Any]] = []
        for game_index in range(config.games_per_update):
            seed = int(
                training_seeds[
                    ((update - 1) * config.games_per_update + game_index)
                    % len(training_seeds)
                ]
            )
            seat = (update + game_index) % 2
            record, game_transitions = collect_ppo_game(
                model,
                entries,
                seed=seed,
                seat=seat,
                config=config,
                update_index=update,
                game_index=game_index,
            )
            records.append(record)
            transitions.extend(game_transitions)
        if not transitions:
            raise RuntimeError(f"PPO update {update} collected no transitions")
        update_metrics = ppo_update(
            model,
            optimizer,
            transitions,
            config,
            update_seed=config.seed + update,
        )
        validation: dict[str, Any] | None = None
        if validation_seeds and validation_opponents:
            from .evaluation import evaluate_matrix  # noqa: PLC0415

            candidate = build_policy_candidate(
                model,
                name="ppo-current",
                snapshot=str(output_dir / f"update-{update}.pth"),
                device_name=config.device_name,
            )
            validation = evaluate_matrix(
                [candidate], validation_opponents, validation_seeds
            )[candidate.name]
        update_record: dict[str, Any] = {
            "update": update,
            "training_records": records,
            "training_games": len(records),
            "training_failed_games": sum(
                record["status"] == "failed" for record in records
            ),
            "teacher_queries": 0,
            "opponent_names": sorted({record["opponent"] for record in records}),
            "update_metrics": update_metrics,
            "validation": validation,
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
            wins = sum(
                int(report["wins"]) for report in validation.values()
            )
            games = sum(
                int(report["games"]) for report in validation.values()
            )
            score = (wins, -games)
            if best_score is None or score > best_score:
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
        elif best_update is None:
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
    result: dict[str, Any] = {
        "best": str(best_path),
        "final": str(final_path),
        "best_update": best_update,
        "best_validation_score": best_score,
        "config": asdict(config),
        "source_bc": str(initial_bc),
        "source_manifest": source_manifest,
        "training_seeds": [int(seed) for seed in training_seeds],
        "validation_seeds": [int(seed) for seed in validation_seeds]
        if validation_seeds
        else None,
        "logs": logs,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result
