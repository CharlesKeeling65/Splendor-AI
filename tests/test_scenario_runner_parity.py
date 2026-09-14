"""One ScenarioV1 enters Game, raw evaluation, and PPO without redealing."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from splendor.agents.generic.first_move import FirstActionAgent
from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.policy_imitation.evaluation import (
    DecisionProbe,
    play_game,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    OpponentPoolEntry,
    PolicyValueNetwork,
    PPOConfig,
    collect_ppo_game,
)
from splendor.agents.our_agents.policy_imitation.protocol import (
    FormalGameRng,
    isolated_python_seed,
)
from splendor.agents.our_agents.policy_imitation.scenario import (
    ScenarioV1,
    generate_scenario,
    rule_from_scenario,
    scenario_from_state,
    scenario_initial_state_sha256,
)
from splendor.game import Game
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.utils import LimitRoundsGameRule


def _scenario() -> ScenarioV1:
    return generate_scenario("ci_smoke", 825_010, selection_kind="ci-fixture")


def _candidate(name: str) -> CandidateSpec:
    return CandidateSpec(name, "fixed_baseline", FirstActionAgent)


def _mask(rule: LimitRoundsGameRule, seat: int) -> np.ndarray:
    state = rule.current_game_state
    return create_legal_actions_mask(rule.getLegalActions(state, seat), state, seat)


def test_game_accepts_prepared_rule_without_changing_legacy_default() -> None:
    scenario = _scenario()
    prepared = rule_from_scenario(scenario)
    game = Game(
        LimitRoundsGameRule,
        [FirstActionAgent(0), FirstActionAgent(1)],
        2,
        seed=999,
        displayer=None,
        prepared_game_rule=prepared,
    )
    assert game.game_rule is prepared
    assert scenario_initial_state_sha256(game.game_rule.current_game_state) == (
        scenario.scenario_id
    )


def test_loaded_snapshot_preserves_source_engine_legal_action_order() -> None:
    seed = 825_012
    with isolated_python_seed(seed):
        source = LimitRoundsGameRule(2)
    scenario = scenario_from_state(
        source.current_game_state,
        source_segment="ci_smoke",
        source_seed=seed,
        selection_kind="ci-fixture",
    )
    loaded = rule_from_scenario(scenario)

    def ordered_indices(rule: LimitRoundsGameRule) -> list[int]:
        state = rule.current_game_state
        legal = rule.getLegalActions(state, 0)
        return list(create_action_mapping(legal, state, 0))

    assert ordered_indices(loaded) == ordered_indices(source)


@pytest.mark.parametrize("feature_version", ["v1", "public-v2"])
def test_raw_game_evaluation_and_ppo_initial_observation_mask_parity(
    feature_version: str,
) -> None:
    scenario = _scenario()
    raw = rule_from_scenario(scenario)
    raw_state = raw.current_game_state
    raw_observation = extract_observation(raw_state, 0, feature_version)
    raw_mask = _mask(raw, 0)

    game = Game(
        LimitRoundsGameRule,
        [FirstActionAgent(0), FirstActionAgent(1)],
        2,
        seed=999,
        displayer=None,
        prepared_game_rule=rule_from_scenario(scenario),
    )
    game_state = game.game_rule.current_game_state
    assert scenario_initial_state_sha256(game_state) == scenario.scenario_id
    assert np.array_equal(
        extract_observation(game_state, 0, feature_version), raw_observation
    )
    assert np.array_equal(_mask(game.game_rule, 0), raw_mask)

    probes: list[DecisionProbe] = []
    record, steps = play_game(
        _candidate("candidate"),
        _candidate("opponent"),
        seed=123,
        seat=0,
        feature_version=feature_version,
        collect_trajectory=True,
        probe_sink=probes.append,
        scenario=scenario,
    )
    assert probes
    evaluation_state = probes[0].state
    evaluation_mask = create_legal_actions_mask(probes[0].actions, evaluation_state, 0)
    assert scenario_initial_state_sha256(evaluation_state) == scenario.scenario_id
    assert np.array_equal(
        extract_observation(evaluation_state, 0, feature_version), raw_observation
    )
    assert np.array_equal(evaluation_mask, raw_mask)
    assert record["scenario_id"] == scenario.scenario_id
    assert steps and all(step.deal_seed == scenario.source_seed for step in steps)

    torch.manual_seed(17)
    model = PolicyValueNetwork(
        len(raw_observation),
        feature_version=feature_version,
        hidden_layers=(8,),
    )
    context = FormalGameRng(
        experiment_id="task1-parity",
        phase="T1.2",
        coupling_group="replicate-0",
        replicate_id=0,
        treatment_id="O",
        scenario_id=scenario.scenario_id,
        seat=0,
        update=1,
        game_index=0,
    )
    ppo_record, transitions = collect_ppo_game(
        model,
        [OpponentPoolEntry("first", _candidate("first"))],
        seed=999_999,
        seat=0,
        config=PPOConfig(
            feature_version=feature_version,
            hidden_layers=(8,),
            updates=1,
            games_per_update=1,
        ),
        update_index=1,
        game_index=0,
        formal_rng=context,
        scenario=scenario,
    )
    assert transitions
    assert np.array_equal(transitions[0].observation, raw_observation)
    assert np.array_equal(transitions[0].legal_mask, raw_mask.astype(np.uint8))
    assert ppo_record["scenario_id"] == scenario.scenario_id
    assert ppo_record["canonical_state_sha256"] == scenario.scenario_id


def test_formal_ppo_rejects_missing_or_mismatched_snapshot() -> None:
    first = _scenario()
    second = generate_scenario("ci_smoke", 825_011, selection_kind="ci-fixture")
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    context = FormalGameRng(
        experiment_id="task1-parity",
        phase="T1.2",
        coupling_group="replicate-0",
        replicate_id=0,
        treatment_id="O",
        scenario_id=first.scenario_id,
        seat=0,
        update=1,
        game_index=0,
    )
    common = {
        "model": model,
        "opponent_pool": [OpponentPoolEntry("first", _candidate("first"))],
        "seed": 1,
        "seat": 0,
        "config": PPOConfig(hidden_layers=(8,), updates=1, games_per_update=1),
        "update_index": 1,
        "game_index": 0,
        "formal_rng": context,
    }
    with pytest.raises(ValueError, match="require an immutable ScenarioV1"):
        collect_ppo_game(**common)
    with pytest.raises(ValueError, match="scenario_id does not match"):
        collect_ppo_game(**common, scenario=second)
    with pytest.raises(ValueError, match="seat count does not match"):
        collect_ppo_game(
            **{
                **common,
                "formal_rng": None,
                "config": PPOConfig(
                    feature_version="public-v2-multi",
                    hidden_layers=(8,),
                    n_seats=3,
                    updates=1,
                    games_per_update=1,
                ),
            },
            scenario=first,
        )


def test_rule_loading_returns_fresh_board_objects() -> None:
    scenario = _scenario()
    left = rule_from_scenario(scenario)
    right = rule_from_scenario(scenario)
    left_card = left.current_game_state.board.dealt[0][0]
    right_card = right.current_game_state.board.dealt[0][0]
    original = deepcopy(right_card.cost)
    left_card.cost.clear()
    assert right_card.cost == original
