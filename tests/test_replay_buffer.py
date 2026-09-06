"""
Unit tests for the DQN replay buffer (phase-1 acceptance A1.1).
"""

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer

OBS_DIM = 4
ACTION_DIM = 6


def make_transition(
    index: int,
) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray, bool]:
    """
    Build a fully deterministic transition where the first obs feature, the
    action and the reward all encode the step index - making it trivial to
    verify what a sampled row actually contains.
    """
    obs = np.full(OBS_DIM, index, dtype=np.float32)
    next_obs = np.full(OBS_DIM, index + 1, dtype=np.float32)
    next_mask = np.zeros(ACTION_DIM, dtype=np.float32)
    next_mask[index % ACTION_DIM] = 1.0
    return obs, index, float(index), next_obs, next_mask, False


def test_single_step_storage_and_sample():
    buffer = ReplayBuffer(capacity=8, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    buffer.add(*make_transition(0))
    assert len(buffer) == 1

    np.random.seed(0)
    obs, actions, rewards, next_obs, next_mask, dones = buffer.sample(1)

    assert obs.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert next_obs.dtype == torch.float32
    assert next_mask.dtype == torch.float32
    assert dones.dtype == torch.float32

    assert obs[0, 0].item() == 0
    assert actions[0].item() == 0
    assert rewards[0].item() == 0
    assert next_obs[0, 0].item() == 1
    assert next_mask[0, 0].item() == 1.0
    assert dones[0].item() == 0.0


def test_nstep_discount_sum():
    gamma = 0.5
    buffer = ReplayBuffer(
        capacity=8, obs_dim=OBS_DIM, action_dim=ACTION_DIM, n_step=3, gamma=gamma
    )
    for index in range(3):
        buffer.add(*make_transition(index))

    # the window folded into exactly one transition
    assert len(buffer) == 1

    np.random.seed(0)
    obs, actions, rewards, next_obs, next_mask, dones = buffer.sample(1)

    expected_reward = 0 + gamma * 1 + gamma**2 * 2
    assert rewards.item() == pytest.approx(expected_reward)
    # the folded transition carries the window's head action & observation...
    assert actions.item() == 0
    assert obs[0, 0].item() == 0
    # ...and the tail's successor state & mask
    assert next_obs[0, 0].item() == 3
    assert next_mask[0, 2 % ACTION_DIM].item() == 1.0
    assert dones.item() == 0.0


def test_nstep_done_truncation():
    buffer = ReplayBuffer(
        capacity=8, obs_dim=OBS_DIM, action_dim=ACTION_DIM, n_step=3, gamma=0.5
    )
    obs0, act0, reward0, next0, mask0, _ = make_transition(0)
    buffer.add(obs0, act0, reward0, next0, mask0, False)
    obs1, act1, reward1, next1, mask1, _ = make_transition(1)
    buffer.add(obs1, act1, reward1, next1, mask1, True)  # episode ends early

    # Both origins are retained, including the final action itself.
    assert len(buffer) == 2
    np.testing.assert_allclose(buffer.rewards[:2], [0.5, 1.0])
    np.testing.assert_array_equal(buffer.dones[:2], [1.0, 1.0])
    np.testing.assert_array_equal(buffer.next_obs[:2, 0], [2, 2])

    # the pending window was cleared - a new episode starts a fresh window,
    # so the next add is held back instead of being folded across episodes
    buffer.add(*make_transition(10))
    assert len(buffer) == 2


def test_wraparound():
    buffer = ReplayBuffer(capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM, n_step=1)
    for index in range(6):
        buffer.add(*make_transition(index))

    # steps 0 & 1 were overwritten by steps 4 & 5
    assert len(buffer) == 4

    np.random.seed(0)
    obs, actions, rewards, next_obs, next_mask, _ = buffer.sample(16)

    stored_values = {2.0, 3.0, 4.0, 5.0}
    for row in range(16):
        index = obs[row, 0].item()
        # every sampled row is internally consistent
        assert actions[row].item() == index
        assert rewards[row].item() == pytest.approx(index)
        assert next_obs[row, 0].item() == pytest.approx(index + 1)
        assert next_mask[row, int(index) % ACTION_DIM].item() == 1.0
        assert index in stored_values
    # the ring actually wrapped (old entries are gone)
    assert set(obs[:, 0].tolist()) <= stored_values
    assert len(set(obs[:, 0].tolist())) >= 2


def test_sample_from_empty_buffer_raises():
    buffer = ReplayBuffer(capacity=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    with pytest.raises(ValueError):
        buffer.sample(2)
