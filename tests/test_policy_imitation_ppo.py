"""Tests for terminal-aware imitation PPO and fixed opponent pools."""

from pathlib import Path
from typing import override

import numpy as np
import torch

from splendor.agents.our_agents.policy_imitation.bc_network import (
    BehaviorCloningNetwork,
)
from splendor.agents.our_agents.policy_imitation.bc_training import (
    BCConfig,
    save_bc_checkpoint,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    OpponentPoolEntry,
    PolicyValueNetwork,
    PPOConfig,
    PPOTransition,
    collect_ppo_game,
    compute_gae,
    train_ppo_selfplay,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class _FirstActionAgent(Agent):
    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        del game_state, game_rule
        return actions[0]


def _opponent() -> OpponentPoolEntry:
    return OpponentPoolEntry(
        "first",
        CandidateSpec("first", "fixed_baseline", _FirstActionAgent),
    )


def test_gae_stops_at_terminal_boundary() -> None:
    transitions: list[PPOTransition] = []
    for index, reward in enumerate((1.0, 2.0)):
        transitions.append(
            PPOTransition(
                np.zeros(265, dtype=np.float32),
                np.ones(3510, dtype=np.uint8),
                0,
                0.0,
                0.0,
                reward,
                index == 1,
                800701,
                0,
            )
        )
    advantages, returns = compute_gae(
        transitions,
        discount_factor=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (2,)
    assert returns.tolist() == advantages.tolist()
    assert np.isfinite(advantages).all()


def test_ppo_game_has_one_fixed_opponent_and_terminal_mapping() -> None:
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    config = PPOConfig(
        feature_version="v1",
        hidden_layers=(8,),
        games_per_update=1,
        updates=1,
        update_epochs=1,
        minibatch_size=32,
        terminal_value=10.0,
    )
    record, transitions = collect_ppo_game(
        model,
        [_opponent()],
        seed=800702,
        seat=0,
        config=config,
        update_index=1,
        game_index=0,
    )

    assert record["status"] == "completed"
    assert record["opponent"] == "first"
    assert record["teacher_queries"] == 0
    assert transitions
    assert transitions[-1].terminal
    assert np.isfinite([transition.reward for transition in transitions]).all()


def test_ppo_training_round_trip_from_bc(tmp_path: Path) -> None:
    bc = BehaviorCloningNetwork(265, feature_version="v1", hidden_layers=(8,))
    bc.fit_normalizer(torch.zeros((2, 265)))
    checkpoint = tmp_path / "bc.pth"
    save_bc_checkpoint(
        bc,
        checkpoint,
        epoch=1,
        config=BCConfig(feature_version="v1", hidden_layers=(8,), epochs=1),
        dataset_metadata={"feature_version": "v1"},
        metrics={},
    )
    config = PPOConfig(
        feature_version="v1",
        hidden_layers=(8,),
        learning_rate=1e-3,
        games_per_update=1,
        updates=1,
        update_epochs=1,
        minibatch_size=32,
        seed=23,
    )
    result = train_ppo_selfplay(
        checkpoint,
        tmp_path / "ppo-run",
        [800703],
        [_opponent()],
        config=config,
        source_manifest="ppo.json",
    )

    assert Path(result["best"]).is_file()
    assert Path(result["final"]).is_file()
    assert result["logs"][0]["teacher_queries"] == 0
    assert result["logs"][0]["training_failed_games"] == 0
