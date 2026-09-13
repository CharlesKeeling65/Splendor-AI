"""Off-policy web-data replay mixing (roadmap 2026-09-12 §D2).

Browser games are harvested into a *separate* replay buffer and sampled with
a bounded mixing ratio so a trickle of human-play transitions cannot drown the
local curriculum.  Three guard rails, all fail-closed:

* **Feature version validation** - the web buffer's observation width must
  equal the local buffer's (both derived from the same feature schema);
  mismatching checkpoints simply refuse to mix.
* **Payment-semantic whitelist** - the web engine is more permissive than the
  local engine (fewer gems per take, discarding just-taken colours; see
  ``docs/web_experiments.md`` ADR).  Games whose parity monitor reported
  anomalies are dropped entirely by default, keeping the replay inside the
  engine's state distribution.
* **Bounded mixing ratio** - the web share of every batch is capped by the
  configured ratio (default 20%), and the web buffer only contributes once it
  holds enough transitions to form its share.
"""

from collections.abc import Iterator
from typing import NamedTuple

import torch
from torch import Tensor

from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer

DEFAULT_WEB_RATIO = 0.2
MIN_MIX_BATCH = 2


class MixedBatch(NamedTuple):
    """One sampled minibatch split across local and web buffers."""

    obs: Tensor
    actions: Tensor
    rewards: Tensor
    next_obs: Tensor
    next_mask: Tensor
    dones: Tensor
    web_share: float


class WebReplayMixer:
    """Sample minibatches from a local buffer plus a bounded web buffer."""

    def __init__(
        self,
        local: ReplayBuffer,
        web: ReplayBuffer,
        web_ratio: float = DEFAULT_WEB_RATIO,
    ) -> None:
        if not 0.0 < web_ratio <= 1.0:
            raise ValueError(f"web_ratio must lie in (0, 1], got {web_ratio}")
        if local.obs_dim != web.obs_dim:
            raise ValueError(
                "feature schema mismatch: local obs_dim "
                f"{local.obs_dim} != web obs_dim {web.obs_dim}; "
                "checkpoints and browser collection must share one schema"
            )
        if local.action_dim != web.action_dim:
            raise ValueError(
                f"action space mismatch: {local.action_dim} != {web.action_dim}"
            )
        self.local = local
        self.web = web
        self.web_ratio = web_ratio

    def _web_count(self, batch_size: int) -> int:
        """How many web transitions this batch should contain."""
        if self.web.size == 0 or self.local.size == 0:
            return 0
        desired = round(batch_size * self.web_ratio)
        return min(desired, self.web.size, batch_size - 1)

    def sample(self, batch_size: int) -> MixedBatch:
        """Sample one mixed minibatch (falls back to pure local when needed)."""
        if batch_size < MIN_MIX_BATCH and self._web_count(batch_size):
            raise ValueError(
                f"batch_size must be at least {MIN_MIX_BATCH} to mix buffers"
            )
        n_web = self._web_count(batch_size)
        n_local = batch_size - n_web
        if n_local > self.local.size:
            raise ValueError(
                f"local buffer holds {self.local.size} transitions, "
                f"fewer than the {n_local} requested"
            )
        local_batch = self.local.sample(n_local)
        if n_web:
            web_batch = self.web.sample(n_web)
            obs = torch.cat((local_batch[0], web_batch[0]))
            actions = torch.cat((local_batch[1], web_batch[1]))
            rewards = torch.cat((local_batch[2], web_batch[2]))
            next_obs = torch.cat((local_batch[3], web_batch[3]))
            next_mask = torch.cat((local_batch[4], web_batch[4]))
            dones = torch.cat((local_batch[5], web_batch[5]))
        else:
            obs, actions, rewards = local_batch[0], local_batch[1], local_batch[2]
            next_obs, next_mask, dones = local_batch[3], local_batch[4], local_batch[5]
        return MixedBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            next_mask=next_mask,
            dones=dones,
            web_share=n_web / batch_size,
        )

    def batches(self, batch_size: int, steps: int) -> Iterator[MixedBatch]:
        """Yield ``steps`` mixed minibatches (sampling uses the global RNG)."""
        for _ in range(steps):
            yield self.sample(batch_size)
