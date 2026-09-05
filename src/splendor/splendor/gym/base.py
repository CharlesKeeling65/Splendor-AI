"""
Unified environment protocol shared by every Splendor gym environment.

``SplendorEnvBase`` is the contract of the "single interface, two
environments" architecture: the local ``SplendorEnv`` (fast engine
simulation) and the browser-backed ``BrowserSplendorEnv`` (real web games)
both satisfy this protocol, so any policy trained against one of them runs
against the other without a single code change.

The protocol is a ``Protocol`` (structural subtyping) rather than an ABC on
purpose: ``SplendorEnv`` predates this contract and must not change its
inheritance tree, while a Protocol lets an existing class satisfy the
contract as-is.
"""

from typing import Protocol, runtime_checkable

import gymnasium as gym
from numpy.typing import NDArray


@runtime_checkable
class SplendorEnvBase(Protocol):
    """
    The minimal surface every Splendor environment must expose.

    :note: ``runtime_checkable`` makes ``isinstance(env, SplendorEnvBase)``
           verify member presence only - type signatures are still enforced
           by static type checkers, not at runtime.
    """

    observation_space: gym.spaces.Box  # shape (265,), dtype float32
    action_space: gym.spaces.Discrete  # len(ALL_ACTIONS) == 3510

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        """
        Start a new game.

        :return: the initial observation (shape (265,)) and an info dict
                 holding at least ``{"my_id": int}``.
        """
        ...

    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Take one action.

        :param action: index into the global action space (3510 actions).
        :param payment: which payment option to use when the action is a
                        purchase with several viable payments. Forward
                        compatibility for the tier-2 payment dimension
                        (phase 4); tier-1 implementations ignore it.
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        ...

    def get_legal_actions_mask(self) -> NDArray:
        """
        Binary array of shape (3510,); 1 marks a legal action.

        Must be re-fetched after every reset/step - the legal set changes
        with the state.
        """
        ...

    def get_payment_options(self, action: int) -> list[dict] | None:
        """
        Return the payment options of a purchase action.

        Tier-1 implementations always return ``None`` (a single greedy
        payment is implied). Tier-2 environments (payment enumeration,
        browser layer) return one dict per viable payment.
        """
        ...
