"""
Micro smoke test for the DQN training loop (phase-1 acceptance A1.2).

A tiny (<= 30 steps) CPU training loop verifies that collect / update / reset
are wired together without crashing and that the losses stay finite. This is
deliberately NOT a training test - real training happens on the maintainer's
machine, not in CI.
"""

import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import optim

import splendor.splendor.gym  # noqa: F401  # registers the splendor-v1 env
from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer
from splendor.agents.our_agents.dqn.reward_wrapper import TerminalRewardWrapper
from splendor.agents.our_agents.dqn.training import (
    DQNParams,
    collect_one_step,
    dqn_update,
)
from splendor.splendor import features

SMOKE_STEPS = 30


def test_tiny_training_loop_smoke(monkeypatch: pytest.MonkeyPatch):
    # force games to end after 2 rounds so the terminal / reset wiring is
    # exercised within the tiny step budget
    monkeypatch.setattr(features, "ROUNDS_LIMIT", 2)

    # reproducibility trio
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    env = TerminalRewardWrapper(
        gym.make("splendor-v1", agents=[RandomAgent(0)]), win_bonus=10.0
    )
    env.reset(seed=7)

    q_net = QNetwork(hidden_layers=(32, 32))
    target_net = QNetwork(hidden_layers=(32, 32))
    target_net.load_state_dict(q_net.state_dict())
    buffer = ReplayBuffer(capacity=500, n_step=3)
    params = DQNParams(batch_size=8, warmup=10, eps_decay_steps=15)
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    episodes_ended = 0
    losses: list[float] = []
    for step in range(SMOKE_STEPS):
        result = collect_one_step(env, q_net, buffer, params, step)
        if len(buffer) >= params.warmup:
            stats = dqn_update(q_net, target_net, buffer, optimizer, params, step=step)
            losses.append(stats["loss"])
        if result["episode_ended"]:
            episodes_ended += 1
            assert result["final_score"] is not None

    assert episodes_ended >= 1, "the terminal -> reset wiring never ran"
    assert len(buffer) >= params.warmup, "the buffer never reached the warmup size"
    assert losses, "no gradient update ever ran"
    assert all(np.isfinite(loss) for loss in losses)
