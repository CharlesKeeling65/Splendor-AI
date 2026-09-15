"""T1.4 episodic PBRS contract and terminal-boundary tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from splendor.agents.generic.first_move import FirstActionAgent
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    OpponentPoolEntry,
    PolicyValueNetwork,
    PPOConfig,
    collect_ppo_game,
    formal_ppo_config_sha256,
)
from splendor.agents.our_agents.policy_imitation.protocol import isolated_seed
from splendor.agents.our_agents.policy_imitation.shaping import (
    DEFAULT_DISCOUNT,
    DEFAULT_KAPPA,
    SAFE_POTENTIAL_CONTRACT,
    SAFE_POTENTIAL_V1,
    TERMINAL_BIASED_POTENTIAL_CONTRACT,
    TERMINAL_BIASED_POTENTIAL_V1,
    PotentialRewardShaper,
    PotentialShapingWrapper,
    ShapingConfig,
    potential,
    potential_reward_contract,
)
from splendor.splendor.constants import ROUNDS_LIMIT
from splendor.splendor.splendor_model import SplendorGameRule
from splendor.splendor.utils import LimitRoundsGameRule


def test_reward_contracts_are_explicit_and_versioned() -> None:
    """O and O_bridge cannot silently resolve to the same terminal semantics."""
    assert SAFE_POTENTIAL_CONTRACT.version == SAFE_POTENTIAL_V1
    assert SAFE_POTENTIAL_CONTRACT.terminal_phi == 0.0
    assert SAFE_POTENTIAL_CONTRACT.policy_invariant
    assert TERMINAL_BIASED_POTENTIAL_CONTRACT.version == (TERMINAL_BIASED_POTENTIAL_V1)
    assert TERMINAL_BIASED_POTENTIAL_CONTRACT.terminal_phi is None
    assert not TERMINAL_BIASED_POTENTIAL_CONTRACT.policy_invariant
    assert potential_reward_contract(SAFE_POTENTIAL_V1) is SAFE_POTENTIAL_CONTRACT
    assert (
        potential_reward_contract(TERMINAL_BIASED_POTENTIAL_V1)
        is TERMINAL_BIASED_POTENTIAL_CONTRACT
    )
    with pytest.raises(ValueError, match="unknown potential reward version"):
        potential_reward_contract("potential")

    safe_config = ShapingConfig(kind="potential", potential_version=SAFE_POTENTIAL_V1)
    assert safe_config.kappa == DEFAULT_KAPPA
    assert safe_config.discount_factor == DEFAULT_DISCOUNT
    assert safe_config.potential_contract is SAFE_POTENTIAL_CONTRACT

    bridge_ppo = PPOConfig(shaping_kind="potential")
    safe_ppo = PPOConfig(shaping_kind="safe-potential")
    assert bridge_ppo.potential_reward_version == TERMINAL_BIASED_POTENTIAL_V1
    assert safe_ppo.potential_reward_version == SAFE_POTENTIAL_V1
    assert formal_ppo_config_sha256(bridge_ppo) != formal_ppo_config_sha256(safe_ppo)


def test_ppo_safe_potential_uses_zero_terminal_contract() -> None:
    """The PPO selector must apply safe PBRS rather than the legacy default."""
    seed = 825_271
    gamma = DEFAULT_DISCOUNT
    torch.manual_seed(17)
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    pool = [
        OpponentPoolEntry(
            "first",
            CandidateSpec("first", "fixed_baseline", FirstActionAgent),
        )
    ]
    common = {
        "hidden_layers": (8,),
        "updates": 1,
        "games_per_update": 1,
        "discount_factor": gamma,
        "terminal_value": 10.0,
    }
    base_record, base = collect_ppo_game(
        model,
        pool,
        seed=seed,
        seat=0,
        config=PPOConfig(shaping_kind="none", **common),
        update_index=0,
        game_index=0,
    )
    safe_record, safe = collect_ppo_game(
        model,
        pool,
        seed=seed,
        seat=0,
        config=PPOConfig(shaping_kind="safe-potential", **common),
        update_index=0,
        game_index=0,
    )
    assert base_record["action_trace_sha256"] == safe_record["action_trace_sha256"]
    assert len(base) == len(safe) > 0
    with isolated_seed(seed):
        initial_rule = LimitRoundsGameRule(2)
        phi_0 = potential(
            initial_rule.current_game_state,
            initial_rule,
            0,
            kappa=DEFAULT_KAPPA,
        )
    discounted_delta = sum(
        gamma**index * (safe_row.reward - base_row.reward)
        for index, (base_row, safe_row) in enumerate(zip(base, safe, strict=True))
    )
    assert discounted_delta == pytest.approx(-phi_0, abs=1e-5)


def test_legacy_default_keeps_nonzero_terminal_potential() -> None:
    """The unqualified historical entry remains O_bridge-compatible."""
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    state.agents[0].score = 2
    shaper = PotentialRewardShaper()
    initial_phi = shaper.reset(state, 0, rule)
    assert shaper.contract is TERMINAL_BIASED_POTENTIAL_CONTRACT

    state.agents[0].score = 15
    rule.current_agent_index = 0
    observed_terminal_phi = potential(state, rule, 0)
    assert rule.gameEnds() and observed_terminal_phi != 0.0
    bonus = shaper.advance(state, 0, rule)
    assert bonus == pytest.approx(
        DEFAULT_DISCOUNT * observed_terminal_phi - initial_phi
    )


def test_safe_potential_discounted_telescoping() -> None:
    """Discounted shaping return is -phi(s0) when terminal_phi is zero."""
    gamma = DEFAULT_DISCOUNT
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    state.agents[0].score = 2
    shaper = PotentialRewardShaper(contract=SAFE_POTENTIAL_CONTRACT)
    phi_0 = shaper.reset(state, 0, rule)

    state.agents[0].score = 5
    state.agents[1].score = 1
    bonus_0 = shaper.advance(state, 0, rule, episode_ended=False)
    phi_1 = potential(state, rule, 0)

    state.agents[0].score = 15
    rule.current_agent_index = 0
    bonus_1 = shaper.advance(state, 0, rule)
    assert rule.gameEnds()
    assert bonus_0 == pytest.approx(gamma * phi_1 - phi_0)
    assert bonus_1 == pytest.approx(-phi_1)
    assert bonus_0 + gamma * bonus_1 == pytest.approx(-phi_0)


def _normal_terminal(rule: SplendorGameRule) -> None:
    rule.current_game_state.agents[0].score = 15
    rule.current_agent_index = 0


def _deadlock_terminal(rule: SplendorGameRule) -> None:
    for agent in rule.current_game_state.agents:
        agent.passed = True


def _round_limit_terminal(rule: SplendorGameRule) -> None:
    for agent in rule.current_game_state.agents:
        agent.agent_trace.action_reward = [None] * ROUNDS_LIMIT


@pytest.mark.parametrize(
    ("rule_factory", "make_terminal"),
    [
        (SplendorGameRule, _normal_terminal),
        (SplendorGameRule, _deadlock_terminal),
        (LimitRoundsGameRule, _round_limit_terminal),
    ],
    ids=("normal", "deadlock", "round-limit"),
)
def test_engine_terminal_paths_resolve_phi_to_zero(
    rule_factory: Callable[[int], SplendorGameRule],
    make_terminal: Callable[[SplendorGameRule], None],
) -> None:
    """Every engine terminal kind uses the same absorbing-state contract."""
    rule = rule_factory(2)
    state = rule.current_game_state
    state.agents[0].score = 2
    shaper = PotentialRewardShaper(contract=SAFE_POTENTIAL_CONTRACT)
    phi_0 = shaper.reset(state, 0, rule)
    make_terminal(rule)
    assert rule.gameEnds()
    assert potential(state, rule, 0) != 0.0
    assert shaper.advance(state, 0, rule) == pytest.approx(-phi_0)


def test_runner_truncation_explicitly_resolves_phi_to_zero() -> None:
    """A time/resource truncation is terminal even when rule.gameEnds is false."""
    rule = SplendorGameRule(2)
    state = rule.current_game_state
    state.agents[0].score = 2
    shaper = PotentialRewardShaper(contract=SAFE_POTENTIAL_CONTRACT)
    phi_0 = shaper.reset(state, 0, rule)
    state.agents[0].score = 9
    assert not rule.gameEnds()
    assert potential(state, rule, 0) != 0.0
    assert shaper.finish(state, 0, rule) == pytest.approx(-phi_0)


class _EndingEnv(gym.Env[NDArray[np.float32], int]):
    """Small Gym surface that ends via either terminated or truncated."""

    metadata: ClassVar[dict[str, object]] = {}

    def __init__(self, *, terminated: bool, truncated: bool) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(1)
        self.game_rule = SplendorGameRule(2)
        self.state = self.game_rule.current_game_state
        self._terminated = terminated
        self._truncated = truncated

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, int]]:
        super().reset(seed=seed)
        del options
        self.game_rule = SplendorGameRule(2)
        self.state = self.game_rule.current_game_state
        self.state.agents[0].score = 2
        return np.zeros(1, dtype=np.float32), {"my_id": 0}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        assert action == 0
        self.state.agents[0].score = 9
        return (
            np.ones(1, dtype=np.float32),
            0.0,
            self._terminated,
            self._truncated,
            {},
        )


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
def test_safe_gym_wrapper_zeros_phi_for_both_done_flags(
    terminated: bool, truncated: bool
) -> None:
    env = PotentialShapingWrapper(
        _EndingEnv(terminated=terminated, truncated=truncated),
        contract=SAFE_POTENTIAL_CONTRACT,
    )
    _observation, _info = env.reset()
    phi_0 = env.potentials[0]
    _observation, reward, got_terminated, got_truncated, _info = env.step(0)
    assert (got_terminated, got_truncated) == (terminated, truncated)
    assert env.potentials == [phi_0, 0.0]
    assert reward == pytest.approx(-phi_0)
