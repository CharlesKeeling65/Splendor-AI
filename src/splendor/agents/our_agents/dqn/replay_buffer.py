"""
Uniform-sampling replay buffer with optional n-step return folding.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .network import ACTION_DIM, OBS_DIM

# (obs, action, reward, next_obs, next_mask, done) as accumulated by add().
PendingTransition = tuple[
    NDArray[np.float32], int, float, NDArray[np.float32], NDArray[np.float32], bool
]


class ReplayBuffer:
    """
    Pre-allocated circular buffer of transitions, sampled uniformly.

    Stores ``(obs, action, reward, next_obs, next_mask, done)`` tuples in
    float32 numpy arrays. The current step's action mask is deliberately *not*
    stored: training only gathers the Q value of the already executed action
    (legal by construction), so keeping the (3510,) mask per transition would
    double the memory footprint for zero benefit.

    With ``n_step > 1`` transitions are folded by a pending queue: the queue is
    collapsed into a single n-step transition once it holds ``n_step`` entries
    or a terminal step arrives, with ``R = sum(gamma^k * r_k)``, the tail's
    ``next_obs`` / ``next_mask`` and the tail's ``done``. A terminal step
    truncates the fold (done=1, so the TD target drops the bootstrap term) and
    clears the queue so a window never spans two episodes. Note that a
    time-limit truncation would be folded as terminal too - the local engine
    never truncates, so this only matters for future environments.

    :note: sampling uses numpy's global RNG - the reproducibility "seed trio"
           (random / np.random / torch) set at the training entry-point covers
           it.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        capacity: int,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        """
        Create a new empty buffer.

        :param capacity: how many transitions the buffer can hold.
        :param obs_dim: how many features the observations have.
        :param action_dim: how many actions the (global) action space has.
        :param n_step: how many environment steps to fold into one transition.
        :param gamma: discount factor used by the n-step return accumulation.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, action_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self._pending: list[PendingTransition] = []

    def _store(  # noqa: PLR0913 - one argument per stored tuple field
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        next_mask: NDArray[np.float32],
        done: bool,
    ) -> None:
        """
        Write one folded transition at the cursor position (overwriting the
        oldest entry once the buffer is full).
        """
        index = self.pos
        self.obs[index] = obs
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_obs[index] = next_obs
        self.next_mask[index] = next_mask
        self.dones[index] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add(  # noqa: PLR0913 - one argument per transition field
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        next_mask: NDArray[np.float32],
        done: bool,
    ) -> None:
        """
        Add one environment step to the buffer.

        With ``n_step == 1`` the transition is stored as-is; otherwise it is
        appended to the pending queue which is folded once full or terminal
        (see the class docstring).

        :param obs: the observation *before* the action was taken.
        :param action: the executed action index (legal by construction).
        :param reward: the reward received for taking the action.
        :param next_obs: the observation *after* the action (and the folded
                         opponent turns of the gym environment).
        :param next_mask: the legal action mask of ``next_obs``.
        :param done: whether the episode ended with this step.
        """
        self._pending.append(
            (obs, int(action), float(reward), next_obs, next_mask, bool(done))
        )

        while self._pending and (done or len(self._pending) >= self.n_step):
            self._fold_oldest()

    def _fold_oldest(self) -> None:
        """Store one origin, including every shorter suffix at termination."""

        # Fold the pending window: R = sum over the window of gamma^k * r_k,
        # the successor state is the tail's, and a terminal step truncates.
        discounted_reward = 0.0
        discount = 1.0
        for _, _, step_reward, _, _, step_done in self._pending:
            discounted_reward += discount * step_reward
            discount *= self.gamma
            if step_done:
                break

        first_obs, first_action = self._pending[0][0], self._pending[0][1]
        last_next_obs = self._pending[-1][3]
        last_next_mask = self._pending[-1][4]
        last_done = self._pending[-1][5]

        self._store(
            first_obs,
            first_action,
            discounted_reward,
            last_next_obs,
            last_next_mask,
            last_done,
        )

        self._pending.pop(0)

    def sample(
        self, batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a minibatch uniformly (with replacement).

        :param batch_size: how many transitions to sample.
        :return: ``(obs, action, reward, next_obs, next_mask, done)`` tensors
                 ready to be fed into the network (obs float32, action int64,
                 reward / next_mask / done float32).
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_obs[indices]),
            torch.from_numpy(self.next_mask[indices]),
            torch.from_numpy(self.dones[indices]),
        )

    def __len__(self) -> int:
        """
        :return: how many transitions are currently stored.
        """
        return self.size
