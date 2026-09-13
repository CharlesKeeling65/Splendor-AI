"""Reward shaping tests (roadmap B2): telescoping, invariance, event terms."""

from __future__ import annotations

import random
from copy import deepcopy
from itertools import pairwise
from typing import cast

import gymnasium as gym
import numpy as np
import pytest

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.policy_imitation.shaping import (
    EventRewardShaper,
    EventShapingWrapper,
    EventWeights,
    PotentialRewardShaper,
    PotentialShapingWrapper,
    ShapingConfig,
    card_affordable,
    noble_progress,
    potential,
)
from splendor.splendor.gym.envs.splendor_env import SplendorEnv
from splendor.splendor.splendor_model import SplendorGameRule


def _seed_triple(seed: int) -> None:
    """AGENTS.md fact 6: engine draws ride the global random + numpy RNG."""
    random.seed(seed)
    np.random.seed(seed)


def _run_game(seed: int, shaped: bool, discount: float = 0.99):
    """Play one random-vs-random gym game; returns rewards, obs, potentials."""
    _seed_triple(seed)
    env: gym.Env = gym.make("splendor-v1", agents=[RandomAgent(1)])
    if shaped:
        env = PotentialShapingWrapper(env, discount_factor=discount)
    rng = np.random.default_rng(seed + 777)
    obs, _info = env.reset(seed=seed)
    rewards: list[float] = []
    observations = [np.asarray(obs)]
    terminated, truncated = False, False
    while not (terminated or truncated):
        mask = np.asarray(cast(SplendorEnv, env.unwrapped).get_legal_actions_mask())
        action = int(rng.choice(np.flatnonzero(mask)))
        _obs, reward, terminated, truncated, _info = env.step(action)
        rewards.append(float(reward))
        observations.append(np.asarray(obs))
    potentials = (
        list(env.get_wrapper_attr("potentials"))
        if isinstance(env, PotentialShapingWrapper)
        else []
    )
    env.close()
    return rewards, observations, potentials


@pytest.mark.parametrize("seed", [825_101, 825_102])
def test_shaping_preserves_observations_and_telescopes(seed: int) -> None:
    """Shaping must not touch observations and must telescope in gamma-space."""
    base_rewards, base_obs, _ = _run_game(seed, shaped=False)
    shaped_rewards, shaped_obs, potentials = _run_game(seed, shaped=True)
    assert len(shaped_rewards) == len(base_rewards) == len(potentials) - 1
    for index, (obs_a, obs_b) in enumerate(zip(base_obs, shaped_obs, strict=True)):
        assert np.array_equal(obs_a, obs_b), f"observation drift at step {index}"

    discount = 0.99
    bonuses = [
        shaped - base for shaped, base in zip(shaped_rewards, base_rewards, strict=True)
    ]
    weighted = sum(discount**t * bonus for t, bonus in enumerate(bonuses))
    expected = discount ** len(bonuses) * potentials[-1] - potentials[0]
    assert weighted == pytest.approx(expected, abs=1e-4)


def test_shaping_telescopes_undiscounted() -> None:
    """With gamma = 1 the bonus sum must equal phi(s_T) - phi(s_0) exactly."""
    _base_rewards, _base_obs, potentials = _run_game(825_103, shaped=True, discount=1.0)
    total_bonus = sum(
        phi_next - phi_prev for phi_prev, phi_next in pairwise(potentials)
    )
    assert total_bonus == pytest.approx(potentials[-1] - potentials[0], abs=1e-6)


def test_potential_reward_shaper_matches_wrapper_semantics() -> None:
    """The raw-loop shaper reproduces the same phi sequence on one state."""
    rule = SplendorGameRule(2)
    shaper = PotentialRewardShaper(kappa=0.05, discount_factor=0.99)
    first = shaper.reset(rule.current_game_state, 0, rule)
    assert first == pytest.approx(
        potential(rule.current_game_state, rule, 0, kappa=0.05)
    )
    bonus = shaper.advance(rule.current_game_state, 0, rule)
    assert bonus == pytest.approx(0.99 * first - first)
    with pytest.raises(ValueError):
        PotentialRewardShaper(discount_factor=0.0)


def test_noble_progress_and_potential_hand_computed() -> None:
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    # A fresh state has no scores and no cards: phi = kappa * noble_progress.
    assert noble_progress(state, 0) == 0.0
    assert potential(state, rule, 0, kappa=0.1) == pytest.approx(0.0)
    assert potential(state, rule, 1, kappa=0.1) == pytest.approx(0.0)

    # Give seat 0 cards towards the first noble; coverage drives the term.
    noble_cost = state.board.nobles[0][1]
    colour, need = next(iter(noble_cost.items()))
    agent = state.agents[0]
    card = deepcopy(state.board.dealt[0][0])
    agent.cards[colour].extend([card] * (need - 1))
    covered = min(need - 1, need)
    total = sum(noble_cost.values())
    assert noble_progress(state, 0) == pytest.approx(covered / total)

    own = float(rule.calScore(state, 0))
    rival = float(rule.calScore(state, 1))
    assert potential(state, rule, 0, kappa=0.1) == pytest.approx(
        own - rival + 0.1 * covered / total
    )


def test_card_affordable_uses_yellow_wildcards() -> None:
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    card = state.board.dealt[0][0]
    agent = state.agents[0]
    for colour, need in card.cost.items():
        agent.gems[colour] = max(need - 1, 0)
    total_missing = sum(max(need - agent.gems[c], 0) for c, need in card.cost.items())
    agent.gems["yellow"] = total_missing - 1
    assert not card_affordable(agent, dict(card.cost))
    agent.gems["yellow"] += 1
    assert card_affordable(agent, dict(card.cost))


def test_event_shaper_bonuses() -> None:
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    shaper = EventRewardShaper(EventWeights(buy=0.2, noble=1.5, denial_reserve=0.05))
    card = state.board.dealt[0][0]
    noble = {"code": "n1", "cost": {}, "points": 3}

    buy_action = {
        "type": "buy_available",
        "card": card,
        "noble": noble,
        "returned_gems": {},
    }
    assert shaper.bonus(buy_action, state, rule, 0) == pytest.approx(0.2 + 1.5)

    collect_action = {
        "type": "collect_diff",
        "collected_gems": {"red": 1},
        "returned_gems": {},
        "noble": None,
    }
    assert shaper.bonus(collect_action, state, rule, 0) == pytest.approx(0.0)

    reserve_action = {
        "type": "reserve",
        "card": card,
        "collected_gems": {},
        "returned_gems": {},
        "noble": None,
    }
    # Fresh state: nobody can afford anything, so no denial bonus.
    assert shaper.bonus(reserve_action, state, rule, 0) == pytest.approx(0.0)
    # Fund a rival up to the card cost: the denial term kicks in.
    rival = state.agents[1]
    for colour, need in card.cost.items():
        rival.gems[colour] = need
    assert shaper.bonus(reserve_action, state, rule, 0) == pytest.approx(0.05)


def test_event_shaping_wrapper_adds_bonuses() -> None:
    _seed_triple(825_104)
    raw_env = gym.make("splendor-v1", agents=[RandomAgent(1)])
    _seed_triple(825_104)
    shaped_env = EventShapingWrapper(gym.make("splendor-v1", agents=[RandomAgent(1)]))
    raw_rewards: list[float] = []
    shaped_rewards: list[float] = []
    for env, sink in ((raw_env, raw_rewards), (shaped_env, shaped_rewards)):
        _seed_triple(825_104)  # engine deals ride the global RNGs
        _obs, _info = env.reset(seed=825_104)
        local_rng = np.random.default_rng(42)
        terminated, truncated = False, False
        while not (terminated or truncated):
            mask = np.asarray(cast(SplendorEnv, env.unwrapped).get_legal_actions_mask())
            action = int(local_rng.choice(np.flatnonzero(mask)))
            _obs, reward, terminated, truncated, _info = env.step(action)
            sink.append(float(reward))
        env.close()
    assert len(raw_rewards) == len(shaped_rewards)
    assert any(s != r for s, r in zip(shaped_rewards, raw_rewards, strict=True))


def test_shaping_config_validation() -> None:
    assert ShapingConfig(kind="potential").kind == "potential"
    with pytest.raises(ValueError):
        ShapingConfig(kind="magic")


# ----- Roadmap E2: ranking utilities -----------------------------------------


def test_rank_utilities_map_and_average_ties() -> None:
    from splendor.agents.our_agents.policy_imitation.shaping import (
        RANK_UTILITIES,
        rank_of,
        rank_utility,
    )

    assert RANK_UTILITIES == {1: 1.0, 2: 0.0, 3: -0.5, 4: -1.0}
    scores = [15.0, 12.0, 8.0, 3.0]
    assert [rank_of(scores, s) for s in range(4)] == [1, 2, 3, 4]
    assert [rank_utility(scores, s) for s in range(4)] == [1.0, 0.0, -0.5, -1.0]
    # tied firsts share (1 + 0) / 2
    tied = [15.0, 15.0, 8.0, 3.0]
    assert rank_of(tied, 0) == 1 and rank_of(tied, 1) == 1
    assert rank_utility(tied, 0) == pytest.approx(0.5)
    assert rank_utility(tied, 1) == pytest.approx(0.5)
    assert rank_utility(tied, 2) == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        rank_of(scores, 4)


def test_rank_utility_wrapper_terminal_reward() -> None:
    from splendor.agents.our_agents.policy_imitation.shaping import (
        RankUtilityWrapper,
    )
    from splendor.template import Agent as Dummy

    class _Random(Dummy):
        pass

    env = gym.make(
        "splendor-v1", agents=[RandomAgent(1), RandomAgent(2), RandomAgent(3)]
    )
    wrapped = RankUtilityWrapper(env, terminal_scale=10.0)
    rng = np.random.default_rng(825_501)
    _seed_triple(825_501)
    _obs, _info = wrapped.reset(seed=825_501)
    terminated = truncated = False
    total = 0.0
    steps = 0
    while not (terminated or truncated):
        mask = np.asarray(wrapped.unwrapped.get_legal_actions_mask())
        action = int(rng.choice(np.flatnonzero(mask)))
        _obs, reward, terminated, truncated, _info = wrapped.step(action)
        total += float(reward)
        steps += 1
    wrapped.close()
    assert steps >= 1
    assert np.isfinite(total)
