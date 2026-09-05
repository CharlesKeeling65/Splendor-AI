"""
Unit tests for the terminal win/loss reward wrapper (phase-1 acceptance A1.1).

A fake environment mimicking the small SplendorEnv surface the wrapper relies
on (step/reset signatures, ``.state``, ``.game_rule``, reset info) keeps the
±win_bonus / draw semantics fully deterministic.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from splendor.agents.our_agents.dqn.reward_wrapper import TerminalRewardWrapper

OBS_SIZE = 4
BASE_REWARD = 1.0
WIN_BONUS = 10.0


@dataclass
class FakeAgentState:
    id: int
    score: float


class FakeGameRule:
    """``calScore`` stub returning preset per-agent scores (already
    tie-break adjusted - the wrapper must only compare them)."""

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores

    def calScore(self, game_state: object, agent_id: int) -> float:
        return self.scores[agent_id]


class FakeGameState:
    def __init__(self, agents: list[FakeAgentState]) -> None:
        self.agents = agents


class FakeSplendorEnv(gym.Env):
    """SplendorEnv surface stub: every step returns ``base_reward`` and can be
    made terminal or not."""

    def __init__(
        self, terminal_scores: dict[int, float], terminate_on_step: bool = True
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBS_SIZE,), np.float32
        )
        self.action_space = gym.spaces.Discrete(10)
        self.game_rule = FakeGameRule(terminal_scores)
        self.state = FakeGameState(
            [
                FakeAgentState(agent_id, score)
                for agent_id, score in terminal_scores.items()
            ]
        )
        self.my_turn = 0
        self.terminate_on_step = terminate_on_step
        self.received_payment: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return np.zeros(OBS_SIZE, dtype=np.float32), {"my_id": self.my_turn}

    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.received_payment = payment
        return (
            np.ones(OBS_SIZE, dtype=np.float32),
            BASE_REWARD,
            self.terminate_on_step,
            False,
            {},
        )


def test_non_terminal_reward_untouched():
    env = FakeSplendorEnv({0: 15.0, 1: 12.0}, terminate_on_step=False)
    wrapped = TerminalRewardWrapper(env, win_bonus=WIN_BONUS)
    wrapped.reset()

    _, reward, terminated, truncated, _ = wrapped.step(3)

    assert reward == BASE_REWARD
    assert terminated is False
    assert truncated is False


def test_terminal_win_adds_bonus():
    # 15.5 vs 15.0 models a tie-break win as calScore would produce it
    env = FakeSplendorEnv({0: 15.5, 1: 15.0})
    wrapped = TerminalRewardWrapper(env, win_bonus=WIN_BONUS)
    wrapped.reset()

    _, reward, terminated, _, _ = wrapped.step(3)

    assert terminated is True
    assert reward == BASE_REWARD + WIN_BONUS


def test_terminal_loss_subtracts_bonus():
    env = FakeSplendorEnv({0: 10.0, 1: 15.0})
    wrapped = TerminalRewardWrapper(env, win_bonus=WIN_BONUS)
    wrapped.reset()

    _, reward, terminated, _, _ = wrapped.step(3)

    assert terminated is True
    assert reward == BASE_REWARD - WIN_BONUS


def test_terminal_draw_adds_nothing():
    env = FakeSplendorEnv({0: 15.0, 1: 15.0})
    wrapped = TerminalRewardWrapper(env, win_bonus=WIN_BONUS)
    wrapped.reset()

    _, reward, terminated, _, _ = wrapped.step(3)

    assert terminated is True
    assert reward == BASE_REWARD


def test_payment_is_forwarded():
    env = FakeSplendorEnv({0: 15.0, 1: 12.0}, terminate_on_step=False)
    wrapped = TerminalRewardWrapper(env, win_bonus=WIN_BONUS)
    wrapped.reset()

    wrapped.step(3, payment=5)

    assert env.received_payment == 5


def test_reset_records_my_id():
    env = FakeSplendorEnv({0: 15.0, 1: 12.0})
    env.my_turn = 1
    wrapped = TerminalRewardWrapper(env, win_bonus=WIN_BONUS)
    assert wrapped.my_id == -1

    _, info = wrapped.reset()

    assert info["my_id"] == 1
    assert wrapped.my_id == 1
