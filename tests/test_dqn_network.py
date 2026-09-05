"""
Unit tests for the Dueling DQN network (phase-1 acceptance A1.1).
"""

import torch

from splendor.agents.our_agents.dqn.network import (
    ACTION_DIM,
    HUGE_NEG,
    OBS_DIM,
    QNetwork,
)


def test_forward_output_shape():
    net = QNetwork()
    obs = torch.zeros(3, OBS_DIM)
    mask = torch.ones(3, ACTION_DIM)

    q_values = net(obs, mask)

    assert q_values.shape == (3, ACTION_DIM)


def test_forward_masks_illegal_actions():
    net = QNetwork()
    obs = torch.zeros(2, OBS_DIM)
    mask = torch.zeros(2, ACTION_DIM)
    mask[0, 5] = 1.0
    mask[0, 100] = 1.0
    mask[1, 7] = 1.0

    q_values = net(obs, mask)

    assert torch.all(q_values[mask == 0] == HUGE_NEG)
    # legal positions hold real network outputs (finite, not the mask value)
    assert torch.isfinite(q_values[mask == 1]).all()


def test_act_returns_legal_action():
    net = QNetwork()
    obs = torch.randn(OBS_DIM)

    for _ in range(20):
        mask = torch.zeros(ACTION_DIM)
        legal_indices = torch.randperm(ACTION_DIM)[:10]
        mask[legal_indices] = 1.0

        action = net.act(obs, mask)

        assert mask[action] == 1.0
        assert isinstance(action, int)


def test_single_state_forward_matches_batch():
    net = QNetwork()
    obs = torch.randn(OBS_DIM)
    mask = torch.ones(ACTION_DIM)

    q_single = net(obs, mask)
    q_batch = net(obs.unsqueeze(0), mask.unsqueeze(0))

    # 1-D inputs get an implicit batch dimension (like the PPO network)
    assert q_single.shape == (1, ACTION_DIM)
    assert torch.allclose(q_single[0], q_batch[0])


def test_dueling_disabled_path():
    net = QNetwork(dueling=False)
    obs = torch.zeros(2, OBS_DIM)
    mask = torch.ones(2, ACTION_DIM)

    assert not hasattr(net, "value_head")
    assert not hasattr(net, "advantage_head")

    q_values = net(obs, mask)

    assert q_values.shape == (2, ACTION_DIM)


def test_input_norm_disabled_path():
    net = QNetwork(use_input_norm=False)

    assert net.input_norm is None

    q_values = net(torch.zeros(2, OBS_DIM), torch.ones(2, ACTION_DIM))

    assert q_values.shape == (2, ACTION_DIM)


def test_raw_q_is_unmasked():
    net = QNetwork()
    obs = torch.zeros(1, OBS_DIM)

    raw = net.raw_q(obs)
    # masking with an all-legal mask is a no-op, so it must match raw_q
    all_legal = net(obs, torch.ones(1, ACTION_DIM))

    assert torch.isfinite(raw).all()
    assert torch.allclose(raw, all_legal)


def test_observe_updates_running_statistics():
    net = QNetwork(use_input_norm=True)
    assert net.input_norm is not None

    obs = torch.full((OBS_DIM,), 2.0)
    net.observe(obs)

    assert torch.allclose(
        net.input_norm.running_mean.flatten(), torch.full((OBS_DIM,), 0.2)
    )
    # variance: EMA of the squared deviation from the previous mean (2 - 0)^2
    assert torch.allclose(
        net.input_norm.running_var.flatten(), torch.full((OBS_DIM,), 1.3)
    )
