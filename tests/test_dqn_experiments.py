"""Public features, checkpoint schemas, RNG isolation, frozen history and search."""

import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.benchmark import benchmark, isolated_rng
from splendor.agents.our_agents.dqn.experiment import ExperimentConfig, train_variant
from splendor.agents.our_agents.dqn.features import V2_DIM, extract_observation
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.population import PopulationAgent
from splendor.agents.our_agents.dqn.search import sample_hidden, search_policy
from splendor.agents.our_agents.dqn.utils import load_saved_dqn, save_model
from splendor.splendor.gym.envs.utils import create_legal_actions_mask
from splendor.splendor.utils import LimitRoundsGameRule


def small_net(auxiliary: bool = True) -> QNetwork:
    return QNetwork(
        input_dim=V2_DIM,
        hidden_layers=(16, 16),
        feature_version="public-v2",
        auxiliary_heads=auxiliary,
    )


def test_public_features_ignore_hidden_information():
    with isolated_rng(32):
        rule = LimitRoundsGameRule(2)
        state = rule.current_game_state
        obs = extract_observation(state, 0, "public-v2")
        assert obs.shape == (V2_DIM,)
        state.board.decks[0].reverse()
        np.testing.assert_array_equal(obs, extract_observation(state, 0, "public-v2"))
        state.agents[1].gems["red"] += 1
        assert not np.array_equal(obs, extract_observation(state, 0, "public-v2"))
        with pytest.raises(ValueError):
            extract_observation(state, 0, "typo")


def test_checkpoint_roundtrip_and_frozen_statistics(tmp_path: Path):
    net = small_net()
    net.observe(torch.ones(V2_DIM))
    net.normalization_frozen = True
    before = deepcopy(net.state_dict())
    net.observe(torch.full((V2_DIM,), 100.0))
    for key, value in before.items():
        torch.testing.assert_close(net.state_dict()[key], value)
    save_model(net, tmp_path / "model.pth")
    loaded = load_saved_dqn(tmp_path / "model.pth")
    assert loaded.feature_version == "public-v2"
    assert loaded.normalization_frozen
    for key, value in before.items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_history_is_frozen_and_bounded():
    pool = PopulationAgent(0, 7, limit=2)
    net = small_net(False)
    for _ in range(3):
        pool.add_snapshot(net)
    assert len(pool.snapshots) == 2
    rival = pool.snapshots[0].net
    assert rival is not None
    assert next(rival.parameters()).data_ptr() != next(net.parameters()).data_ptr()
    assert not next(rival.parameters()).requires_grad


def test_search_legal_reproducible_and_no_mutation():
    with isolated_rng(7):
        rule = LimitRoundsGameRule(2)
        net = small_net()
        before = pickle.dumps(rule)
        py_rng, np_rng, torch_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        stats: dict[str, float | int] = {}
        pi = search_policy(net, rule, 8, np.random.default_rng(81), stats=stats)
        repeat = search_policy(net, rule, 8, np.random.default_rng(81))
        np.testing.assert_array_equal(pi, repeat)
        assert float(pi.sum()) == pytest.approx(1)
        assert stats["simulations"] == 8
        assert stats["tree_nodes"] >= 1
        mask = create_legal_actions_mask(
            rule.getLegalActions(rule.current_game_state, 0), rule.current_game_state, 0
        )
        assert np.all(pi[mask == 0] == 0)
        assert pickle.dumps(rule) == before
        assert random.getstate() == py_rng
        np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng)


def test_determinization_does_not_peek_at_true_reservation():
    with isolated_rng(17):
        rule = LimitRoundsGameRule(2)
        state = rule.current_game_state
        state.agents[1].cards["yellow"].append(state.board.decks[0].pop())
        changed = deepcopy(rule)
        other = changed.current_game_state
        other.agents[1].cards["yellow"][0], other.board.decks[0][0] = (
            other.board.decks[0][0],
            other.agents[1].cards["yellow"][0],
        )
        other.board.decks[0].reverse()
        one = sample_hidden(rule, 0, np.random.default_rng(18)).current_game_state
        two = sample_hidden(changed, 0, np.random.default_rng(18)).current_game_state
        assert [c.code for c in one.board.decks[0]] == [
            c.code for c in two.board.decks[0]
        ]
        assert (
            one.agents[1].cards["yellow"][0].code
            == two.agents[1].cards["yellow"][0].code
        )


def test_benchmark_pairs_seats_and_restores_rng(monkeypatch: pytest.MonkeyPatch):
    from splendor.splendor import features

    monkeypatch.setattr(features, "ROUNDS_LIMIT", 2)
    net = small_net(False)
    before = random.getstate()
    result = benchmark(net, "random", [817])
    assert result["games"] == 2
    assert [r["seat"] for r in result["records"]] == [0, 1]
    assert random.getstate() == before


@pytest.mark.parametrize(
    "variant", ["corrected", "frozen", "public", "population", "search"]
)
def test_each_experiment_variant_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
):
    from splendor.splendor import features

    monkeypatch.setattr(features, "ROUNDS_LIMIT", 2)
    config = ExperimentConfig(
        steps=20,
        warmup=4,
        batch_size=2,
        buffer_size=40,
        eval_every=20,
        validation_deals=1,
        test_deals=1,
        snapshot_every=8,
        search_every=2,
        simulations=2,
        device="cpu",
    )
    path = train_variant(tmp_path / variant, variant, 19, config)
    assert path.exists()
    assert (path.parent / "final.pth").exists()
