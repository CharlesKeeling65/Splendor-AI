"""
Implementation of the DQN training core: collection loop, Double DQN update
and greedy evaluation.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.optim.optimizer import Optimizer

import splendor.splendor.gym  # noqa: F401  # registers the splendor-v1 env
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.splendor_env import SplendorEnv
from splendor.template import Agent

from .constants import (
    DISCOUNT_FACTOR,
    EVAL_GAMES,
    LEARNING_RATE,
    MAX_GRADIENT_NORM,
    SEED,
    TARGET_UPDATE_FREQ,
    TARGET_UPDATE_TAU,
)
from .network import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DQNParams:
    """
    Placeholder for various learning parameters.

    gamma: by how much the reward decays over environment steps.
    lr: the learning rate of the gradient descent based learning.
    batch_size: how many transitions are sampled per gradient step.
    warmup: how many purely random steps are collected before the first update.
    eps_start / eps_end: the exploration rate bounds.
    eps_decay_steps: over how many steps epsilon decays linearly from
                     ``eps_start`` to ``eps_end``.
    tau: the soft update coefficient of the target network
         (theta' <- theta' + tau * (theta - theta')).
    target_update_freq: when positive, the target network is hard-updated
                        (full state_dict copy) every this many gradient steps
                        instead of being soft-updated.
    max_grad_norm: the global gradient norm clipping threshold.
    seed: the seed of the training session (set at the entry-point).
    device: on which device the computations run.
    """

    # pylint: disable=too-many-instance-attributes

    gamma: float = DISCOUNT_FACTOR
    lr: float = LEARNING_RATE
    batch_size: int = 512
    warmup: int = 5_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 40_000
    tau: float = TARGET_UPDATE_TAU
    target_update_freq: int = TARGET_UPDATE_FREQ
    max_grad_norm: float = MAX_GRADIENT_NORM
    seed: int = SEED
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def epsilon_at(step: int, params: DQNParams) -> float:
    """
    Compute the exploration rate at the given global step (linear decay).

    :param step: the current global training step (0-based).
    :param params: the learning parameters holding the epsilon schedule.
    :return: the epsilon value, linearly interpolated between ``eps_start``
             and ``eps_end`` across ``eps_decay_steps``.
    """
    if step >= params.eps_decay_steps:
        return params.eps_end
    if params.eps_decay_steps <= 0:
        return params.eps_end
    progress = step / params.eps_decay_steps
    return params.eps_start + (params.eps_end - params.eps_start) * progress


def _update_target_network(
    q_net: QNetwork, target_net: QNetwork, params: DQNParams, step: int
) -> None:
    """
    Nudge the target network towards the online one.

    Soft update (τ-EMA over the whole state_dict, running statistics included
    so both networks normalize inputs identically) by default; hard copy every
    ``target_update_freq`` steps when a positive frequency is configured.

    :param q_net: the online Q-network.
    :param target_net: the target Q-network (updated in place).
    :param params: the learning parameters holding tau & the update frequency.
    :param step: the current global step (only matters for hard updates).
    """
    with torch.no_grad():
        if params.target_update_freq > 0:
            if step % params.target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
            return

        online_state = q_net.state_dict()
        for name, target_value in target_net.state_dict().items():
            target_value.mul_(1.0 - params.tau).add_(
                online_state[name], alpha=params.tau
            )


def dqn_update(  # noqa: PLR0913, PLR0917 - mirrors the spec's function signature
    q_net: QNetwork,
    target_net: QNetwork,
    buffer: ReplayBuffer,
    optimizer: Optimizer,
    params: DQNParams,
    step: int = 0,
) -> dict[str, float]:
    """
    Perform one Double DQN gradient step.

    The action for the bootstrap term is selected by the *online* network and
    evaluated by the *target* network - decoupling selection from evaluation
    cuts the overestimation feedback loop that a 3510-way output head with
    sparse rewards would otherwise amplify through bootstrapping.

    :param q_net: the online Q-network (updated by the optimizer).
    :param target_net: the (slower) target Q-network.
    :param buffer: the replay buffer to sample the minibatch from.
    :param optimizer: the optimizer of the online network.
    :param params: the learning parameters.
    :param step: the current global step (only used to time hard target
                 updates when ``params.target_update_freq > 0``).
    :return: {"loss", "q_mean", "td_abs_mean"} for stats.csv / monitoring.
    """
    obs, actions, rewards, next_obs, next_masks, dones = buffer.sample(
        params.batch_size
    )
    obs = obs.to(params.device)
    actions = actions.to(params.device)
    rewards = rewards.to(params.device)
    next_obs = next_obs.to(params.device)
    next_masks = next_masks.to(params.device)
    dones = dones.to(params.device)

    with torch.no_grad():
        # Double DQN: the online network selects the bootstrap action, the
        # target network evaluates it. Masks inside forward make the argmax
        # land on a legal action by construction.
        best_next_actions = q_net(next_obs, next_masks).argmax(dim=1, keepdim=True)
        next_q = (
            target_net(next_obs, next_masks).gather(1, best_next_actions).squeeze(1)
        )
        targets = rewards + params.gamma * (1.0 - dones) * next_q

    # The executed action is legal by construction, so the current-step mask
    # is not needed - an all-ones mask leaves the gathered values untouched.
    current_masks = torch.ones_like(next_masks)
    q_pred = q_net(obs, current_masks).gather(1, actions.unsqueeze(1)).squeeze(1)

    loss = F.smooth_l1_loss(q_pred, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), params.max_grad_norm)
    optimizer.step()

    _update_target_network(q_net, target_net, params, step)

    td_errors = q_pred.detach() - targets
    return {
        "loss": float(loss.item()),
        "q_mean": float(q_pred.detach().mean().item()),
        "td_abs_mean": float(td_errors.abs().mean().item()),
    }


def collect_one_step(
    env: gym.Env,
    q_net: QNetwork,
    buffer: ReplayBuffer,
    params: DQNParams,
    step: int,
) -> dict[str, float | bool | None]:
    """
    Collect one ε-greedy environment step into the buffer, resetting the
    environment when the episode ends.

    The function is stateless across calls: the current observation & mask are
    re-derived from the environment itself, so the caller does not have to
    thread ``(obs, mask)`` through the training loop. Random actions are
    sampled uniformly among *legal* actions - a raw ``randint(3510)`` would be
    illegal with probability 99%+ and crash the env step (KeyError, no
    fallback).

    :param env: the (reward-wrapped) training environment; must already be
                reset.
    :param q_net: the online Q-network (kept in eval mode; its input
                  normalization statistics are refreshed via ``observe``).
    :param buffer: the replay buffer to store the transition into.
    :param params: the learning parameters (device & epsilon schedule).
    :param step: the current global training step (drives the epsilon decay).
    :return: {"step_reward", "episode_ended", "final_score"} - the step's
             reward, whether the episode ended (the environment is already
             reset when it did), and the final calScore of our agent when the
             episode ended (None otherwise).
    """
    splendor_env = cast(SplendorEnv, env.unwrapped)

    # The env vectorizes the same way it does internally, so the recomputed
    # observation always equals the one returned by the previous step call.
    obs: NDArray[np.float32] = extract_metrics_with_cards(
        splendor_env.state, splendor_env.my_turn
    ).astype(np.float32)
    mask: NDArray[np.float32] = splendor_env.get_legal_actions_mask().astype(np.float32)

    obs_tensor = torch.from_numpy(obs).to(params.device)
    q_net.observe(obs_tensor)

    epsilon = epsilon_at(step, params)
    if random.random() < epsilon:
        action = int(np.random.choice(np.flatnonzero(mask)))
    else:
        action = q_net.act(obs_tensor, torch.from_numpy(mask).to(params.device))

    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_obs = np.asarray(next_obs, dtype=np.float32)

    # A (never expected, defensive) time-limit truncation is treated as an
    # episode boundary for both the n-step fold and the reset, so that a
    # pending window can never span two episodes.
    episode_ended = bool(terminated or truncated)
    next_mask = (
        np.zeros_like(mask)
        if terminated
        else splendor_env.get_legal_actions_mask().astype(np.float32)
    )

    buffer.add(obs, action, float(reward), next_obs, next_mask, episode_ended)

    final_score: float | None = None
    if episode_ended:
        final_score = float(
            splendor_env.game_rule.calScore(splendor_env.state, splendor_env.my_turn)
        )
        env.reset()

    return {
        "step_reward": float(reward),
        "episode_ended": episode_ended,
        "final_score": final_score,
    }


@torch.no_grad()
def evaluate(
    q_net: QNetwork,
    make_opponents: Callable[[], list[Agent]],
    n_games: int = EVAL_GAMES,
) -> dict[str, float]:
    """
    Play ``n_games`` with the greedy policy and tally win/draw/loss by calScore.

    A fresh environment (and fresh opponents) is built for every game: the
    seat order and the dealing are random per reset, so evaluating on
    independent games averages over the seating dimension instead of
    overfitting the report to whoever happened to go first.

    :param q_net: the network whose greedy policy is evaluated (temporarily
                  forced to eval mode, then restored).
    :param make_opponents: zero-argument factory producing the opponent agents
                           of a single evaluation game.
    :param n_games: how many games to play.
    :return: {"win", "draw", "loss", "avg_score"} rates / average final
             calScore of our agent.
    """
    was_training = q_net.training
    q_net.eval()
    try:
        wins = 0
        draws = 0
        losses = 0
        total_score = 0.0

        for _ in range(n_games):
            env = gym.make("splendor-v1", agents=make_opponents())
            splendor_env = cast(SplendorEnv, env.unwrapped)
            env.reset()

            mask: NDArray[np.float32] = splendor_env.get_legal_actions_mask().astype(
                np.float32
            )
            terminated, truncated = False, False
            while not (terminated or truncated):
                obs: NDArray[np.float32] = extract_metrics_with_cards(
                    splendor_env.state, splendor_env.my_turn
                ).astype(np.float32)
                action = q_net.act(torch.from_numpy(obs), torch.from_numpy(mask))
                _, _, terminated, truncated, _ = env.step(action)
                if not (terminated or truncated):
                    mask = splendor_env.get_legal_actions_mask().astype(np.float32)

            state = splendor_env.state
            game_rule = splendor_env.game_rule
            my_id = splendor_env.my_turn
            my_score = float(game_rule.calScore(state, my_id))
            best_rival_score = max(
                (
                    float(game_rule.calScore(state, agent.id))
                    for agent in state.agents
                    if agent.id != my_id
                ),
                default=my_score,
            )
            if my_score > best_rival_score:
                wins += 1
            elif my_score < best_rival_score:
                losses += 1
            else:
                draws += 1
            total_score += my_score
    finally:
        if was_training:
            q_net.train()

    return {
        "win": wins / n_games,
        "draw": draws / n_games,
        "loss": losses / n_games,
        "avg_score": total_score / n_games,
    }


def collect_from_browser(
    browser_env: SplendorEnvBase,
    buffer: ReplayBuffer,
    n_games: int,
    q_net: QNetwork | None = None,
) -> dict[str, float]:
    """
    Fold completed web games into the local replay buffer (off-policy).

    Real opponent data corrects the distribution shift no local opponent pool
    covers - humans hoard gems, starve colours, and err non-linearly. Even a
    slow trickle of browser transitions (minutes per game vs seconds locally)
    is enough to nudge the value function, because the replay accepts data
    from *any* behaviour policy. This is the红利 that on-policy algorithms
    (PPO) cannot cash.

    :param browser_env: a reset-ready BrowserSplendorEnv (reward semantics
                        already match the local env, so no wrapper needed).
    :param buffer: the replay buffer to append transitions to.
    :param n_games: how many web games to harvest.
    :param q_net: when given, actions come from its greedy policy; otherwise
                  transitions are collected under the random-in-mask policy
                  (useful to seed the buffer before any checkpoint exists).
    :return: {"games", "steps", "avg_score"} - harvest statistics for logs.
    """
    total_steps = 0
    total_score = 0.0

    for _ in range(n_games):
        obs, _info = browser_env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        mask = np.asarray(browser_env.get_legal_actions_mask(), dtype=np.float32)
        terminated = False
        game_reward = 0.0
        while not terminated:
            if q_net is not None:
                action = q_net.act(torch.from_numpy(obs), torch.from_numpy(mask))
            else:
                action = int(np.random.choice(np.flatnonzero(mask)))
            next_obs, reward, terminated, _truncated, _info = browser_env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            next_mask = (
                np.zeros_like(mask)
                if terminated
                else np.asarray(
                    browser_env.get_legal_actions_mask(), dtype=np.float32
                )
            )
            buffer.add(obs, action, float(reward), next_obs, next_mask, terminated)
            obs, mask = next_obs, next_mask
            total_steps += 1
            # the browser env's rewards are panel score deltas, so their sum
            # telescopes to the final score (no terminal wrapper on the web)
            game_reward += float(reward)
        total_score += game_reward

    return {
        "games": float(n_games),
        "steps": float(total_steps),
        "avg_score": total_score / n_games,
    }
