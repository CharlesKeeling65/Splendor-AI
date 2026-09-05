"""
Terminal win/loss reward wrapper (DQN_GUIDE §5.4).
"""

from typing import cast, override

import gymnasium as gym
from numpy.typing import NDArray

from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.splendor_env import SplendorEnv

from .constants import WIN_BONUS


class TerminalRewardWrapper(gym.Wrapper):
    """
    Keep the engine's score-delta shaping and add a terminal win/loss bonus.

    The engine's reward is a bare score delta with no win/loss signal - MC
    returns (PPO) can tolerate that, but TD bootstrapping without a terminal
    signal only learns a "remaining score" estimate, never the chance of
    winning. This wrapper appends ``+win_bonus / 0 / -win_bonus`` on terminal
    steps, with the win/loss judgment reusing ``calScore`` (including its
    +0.5 fewest-cards tie-break) - the exact function the evaluation path uses,
    so training and evaluation can never disagree about who won.

    :note: the same wrapper works unchanged on top of any SplendorEnvBase
           implementation (local engine or the future browser environment) -
           that is the point of the phase-0 protocol.
    """

    def __init__(self, env: gym.Env, win_bonus: float = WIN_BONUS) -> None:
        """
        Wrap the given Splendor environment with terminal reward shaping.

        :param env: the environment to wrap.
        :param win_bonus: the reward magnitude added/subtracted on terminal
                          steps (win / loss respectively).
        """
        super().__init__(env)
        self.win_bonus = win_bonus
        # Overridden by reset(); kept as an attribute so the terminal judgment
        # does not depend on wrapper construction order.
        self.my_id: int = -1

    @override
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        """
        Reset the wrapped environment and record which seat is ours.

        :param seed: forwarded to the wrapped environment.
        :param options: forwarded to the wrapped environment.
        :return: the initial observation and info of a new game.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.my_id = int(info["my_id"])
        return obs, info

    @override
    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Take a step in the wrapped environment, adding the terminal bonus.

        :param action: index into the global action space (3510 actions).
        :param payment: which payment option to use for a purchase action -
                        forwarded per the SplendorEnvBase protocol (tier-1
                        environments ignore it).
        :return: ``(obs, reward, terminated, truncated, info)`` where reward is
                 the engine's score delta plus the terminal win/loss bonus.
        """
        # gymnasium 0.29's stock wrappers (OrderEnforcing & co.) do not forward
        # keyword arguments, so payment is only passed through when it carries
        # information - the tier-1 default (None) keeps the plain path.
        if payment is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            env = cast(SplendorEnvBase, self.env)
            obs, reward, terminated, truncated, info = env.step(action, payment)
        reward = float(reward)

        if terminated:
            splendor_env = cast(SplendorEnv, self.env.unwrapped)
            state = splendor_env.state
            game_rule = splendor_env.game_rule

            my_score = float(game_rule.calScore(state, self.my_id))
            best_rival_score = max(
                (
                    float(game_rule.calScore(state, agent.id))
                    for agent in state.agents
                    if agent.id != self.my_id
                ),
                default=my_score,
            )
            if my_score > best_rival_score:
                reward += self.win_bonus
            elif my_score < best_rival_score:
                reward -= self.win_bonus

        return obs, reward, terminated, truncated, info
