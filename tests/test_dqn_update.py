"""
Hand-constructed assertions for the Double DQN update (phase-1 acceptance A1.1).

A constant stand-in Q-network makes every TD target computable by hand, which
lets the tests pin down the three properties the implementation must get right
at once: the bootstrap action is selected by the *online* network, evaluated by
the *target* network, and the selection respects the legal-action mask.
"""

import numpy as np
import pytest
import torch
from torch import nn, optim

from splendor.agents.our_agents.dqn.network import HUGE_NEG, QNetwork
from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer
from splendor.agents.our_agents.dqn.training import DQNParams, dqn_update

OBS_DIM = 4
ACTION_DIM = 4

ONLINE_TABLE = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
TARGET_TABLE = torch.tensor([[10.0, 40.0, 30.0, 20.0]])
NEXT_MASK = torch.tensor([[0.0, 1.0, 1.0, 0.0]])


class ConstantQNetwork(nn.Module):
    """
    A stand-in Q-network returning a fixed (masked) Q-table scaled by a single
    learnable gain, so gradients exist but the tables stay hand-checkable.
    """

    def __init__(self, q_table: torch.Tensor) -> None:
        super().__init__()
        self.q_table = q_table
        self.gain = nn.Parameter(torch.ones(1))

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.q_table.masked_fill(action_mask == 0, HUGE_NEG) * self.gain


def make_buffer(done: bool) -> ReplayBuffer:
    """
    A one-transition buffer: action 2 with reward 1, successor masked to
    actions {1, 2}, not terminal (unless ``done``).
    """
    buffer = ReplayBuffer(
        capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM, n_step=1, gamma=0.5
    )
    buffer.add(
        obs=np.zeros(OBS_DIM, dtype=np.float32),
        action=2,
        reward=1.0,
        next_obs=np.ones(OBS_DIM, dtype=np.float32),
        next_mask=NEXT_MASK.numpy().astype(np.float32),
        done=done,
    )
    return buffer


def make_params(
    lr: float = 0.0,
    tau: float = 0.005,
    target_update_freq: int = 0,
    batch_size: int = 1,
) -> DQNParams:
    return DQNParams(
        batch_size=batch_size,
        gamma=0.5,
        warmup=0,
        lr=lr,
        tau=tau,
        target_update_freq=target_update_freq,
        max_grad_norm=100.0,  # disable clipping for deterministic gradients
    )


def test_double_dqn_target_selects_online_evaluates_target():
    torch.manual_seed(0)
    q_net = ConstantQNetwork(ONLINE_TABLE)
    target_net = ConstantQNetwork(TARGET_TABLE)
    optimizer = optim.SGD(q_net.parameters(), lr=0.0)
    params = make_params(lr=0.0)
    buffer = make_buffer(done=False)

    np.random.seed(0)
    stats = dqn_update(q_net, target_net, buffer, optimizer, params)

    # online masked row: [-, 2, 3, -] -> argmax = action 2
    # target evaluates action 2 -> 30, so y = 1 + 0.5 * 30 = 16
    # q_pred = online[action 2] = 3
    assert stats["loss"] == pytest.approx(12.5)
    assert stats["q_mean"] == pytest.approx(3.0)
    assert stats["td_abs_mean"] == pytest.approx(13.0)
    assert set(stats.keys()) == {"loss", "q_mean", "td_abs_mean"}


def test_bootstrap_uses_masked_argmax():
    """If the mask were ignored, the online argmax would be action 3 (4.0)
    and the target evaluation 20 - a different hand-computable loss."""
    torch.manual_seed(0)
    q_net = ConstantQNetwork(ONLINE_TABLE)
    target_net = ConstantQNetwork(TARGET_TABLE)
    optimizer = optim.SGD(q_net.parameters(), lr=0.0)
    params = make_params()
    buffer = ReplayBuffer(
        capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM, n_step=1, gamma=0.5
    )
    buffer.add(
        obs=np.zeros(OBS_DIM, dtype=np.float32),
        action=2,
        reward=1.0,
        next_obs=np.ones(OBS_DIM, dtype=np.float32),
        next_mask=np.ones(ACTION_DIM, dtype=np.float32),  # all actions legal
        done=False,
    )

    np.random.seed(0)
    stats = dqn_update(q_net, target_net, buffer, optimizer, params)

    # unmasked online argmax = action 3 -> target evaluates 20 -> y = 1 + 0.5*20
    assert stats["loss"] == pytest.approx(7.5)  # |3 - 11| - 0.5


def test_terminal_step_drops_bootstrap():
    torch.manual_seed(0)
    q_net = ConstantQNetwork(ONLINE_TABLE)
    target_net = ConstantQNetwork(TARGET_TABLE)
    optimizer = optim.SGD(q_net.parameters(), lr=0.0)
    params = make_params()
    buffer = make_buffer(done=True)

    np.random.seed(0)
    stats = dqn_update(q_net, target_net, buffer, optimizer, params)

    # done=1 -> y = reward = 1, q_pred = 3
    assert stats["loss"] == pytest.approx(1.5)  # |3 - 1| - 0.5
    assert stats["q_mean"] == pytest.approx(3.0)
    assert stats["td_abs_mean"] == pytest.approx(2.0)


def test_soft_update_pulls_target_towards_online():
    torch.manual_seed(0)
    q_net = ConstantQNetwork(ONLINE_TABLE)
    target_net = ConstantQNetwork(TARGET_TABLE)
    optimizer = optim.SGD(q_net.parameters(), lr=0.1)
    params = make_params(lr=0.1, tau=0.5)
    buffer = make_buffer(done=False)

    assert q_net.gain.item() == pytest.approx(1.0)
    assert target_net.gain.item() == pytest.approx(1.0)

    np.random.seed(0)
    dqn_update(q_net, target_net, buffer, optimizer, params)

    # grad of smooth_l1(3*gain, 16) w.r.t. gain is -3 -> gain becomes 1.3
    assert q_net.gain.item() == pytest.approx(1.3)
    # target gain = 0.5 * 1.0 + 0.5 * 1.3
    assert target_net.gain.item() == pytest.approx(1.15)


def test_hard_update_copies_online_state():
    torch.manual_seed(0)
    q_net = ConstantQNetwork(ONLINE_TABLE)
    target_net = ConstantQNetwork(TARGET_TABLE)
    optimizer = optim.SGD(q_net.parameters(), lr=0.1)
    params = make_params(lr=0.1, target_update_freq=1)
    buffer = make_buffer(done=False)

    np.random.seed(0)
    dqn_update(q_net, target_net, buffer, optimizer, params, step=0)

    # full state_dict copy: the learnable gain matches exactly
    assert target_net.gain.item() == q_net.gain.item()


def test_update_works_with_real_network_shapes():
    """End-to-end shapes sanity: the real QNetwork consumes the real buffer
    layout (265-dim observations, 3510 actions) without shape errors."""
    torch.manual_seed(0)
    q_net = QNetwork(hidden_layers=(16, 16))
    target_net = QNetwork(hidden_layers=(16, 16))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.SGD(q_net.parameters(), lr=1e-3)
    params = make_params(batch_size=2)

    buffer = ReplayBuffer(capacity=8, n_step=1)
    for index in range(3):
        buffer.add(
            obs=np.full(265, index, dtype=np.float32),
            action=index * 10,
            reward=float(index),
            next_obs=np.full(265, index + 1, dtype=np.float32),
            next_mask=np.zeros(3510, dtype=np.float32),
            done=False,
        )
    # make the successor masks legal somewhere so the argmax is meaningful
    buffer.next_mask[0, 5] = 1.0
    buffer.next_mask[1, 6] = 1.0
    buffer.next_mask[2, 7] = 1.0

    np.random.seed(0)
    stats = dqn_update(q_net, target_net, buffer, optimizer, params)

    assert all(np.isfinite(value) for value in stats.values())
