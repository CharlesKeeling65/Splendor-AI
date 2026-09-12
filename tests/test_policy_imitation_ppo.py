"""Tests for terminal-aware imitation PPO and fixed opponent pools."""

from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import distributions, optim

from splendor.agents.our_agents.policy_imitation import ppo_selfplay as ppo_module
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
    build_opponent_pool,
    collect_ppo_game,
    compute_gae,
    load_ppo_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
    train_ppo_selfplay,
    warmup_critic,
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


def test_return_critic_is_unbounded_and_terminal_target_is_ten() -> None:
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    assert model.value_mode == "return"
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.fill_(10.0)
        _, values = model(
            torch.zeros((1, 265)),
            torch.ones((1, 3510)),
        )
    assert values.item() == 10.0

    transition = PPOTransition(
        np.zeros(265, dtype=np.float32),
        np.ones(3510, dtype=np.uint8),
        0,
        0.0,
        0.0,
        10.0,
        True,
        800700,
        0,
    )
    _, returns = compute_gae([transition], discount_factor=0.99, gae_lambda=0.95)
    assert returns.tolist() == [10.0]


def test_legacy_checkpoint_without_value_mode_keeps_tanh_critic(tmp_path: Path) -> None:
    config = PPOConfig(hidden_layers=(8,), updates=1, games_per_update=1)
    model = PolicyValueNetwork(
        265,
        feature_version="v1",
        hidden_layers=(8,),
        value_mode="outcome",
    )
    checkpoint = tmp_path / "legacy.pth"
    save_ppo_checkpoint(
        model,
        checkpoint,
        update=1,
        config=config,
        source_bc="bc.pth",
        opponent_pool=[_opponent()],
        metrics={},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["value_mode"]
    del payload["config"]["value_mode"]
    torch.save(payload, checkpoint)

    loaded = load_ppo_checkpoint(checkpoint)
    assert loaded.value_mode == "outcome"
    _, values = loaded(torch.zeros((1, 265)), torch.ones((1, 3510)))
    assert float(values.detach().abs().max()) <= 1.0


def test_pool_history_bucket_is_fixed_and_zero_current_is_respected() -> None:
    fixed = [_opponent()]
    history = [
        OpponentPoolEntry(
            f"history-{index}",
            CandidateSpec(f"history-{index}", "fixed_baseline", _FirstActionAgent),
        )
        for index in range(6)
    ]
    current = OpponentPoolEntry(
        "current",
        CandidateSpec("current", "fixed_baseline", _FirstActionAgent),
    )
    pool = build_opponent_pool(
        fixed,
        history,
        current,
        current_weight=0.0,
        history_weight=2.0,
        history_limit=4,
    )
    weights = {entry.name: entry.weight for entry in pool}
    assert weights["first"] == 1.0
    assert weights["current"] == 0.0
    assert set(weights) == {"first", "current", "history-2", "history-3", "history-4", "history-5"}
    assert sum(weights[name] for name in weights if name.startswith("history-")) == 2.0
    assert all(weights[name] == 0.5 for name in weights if name.startswith("history-"))

    empty_history = build_opponent_pool(
        fixed,
        (),
        current,
        current_weight=0.0,
        history_weight=2.0,
        history_limit=4,
    )
    assert {entry.name: entry.weight for entry in empty_history}["current"] == 2.0


def _single_transition(model: PolicyValueNetwork) -> PPOTransition:
    observation = np.zeros(265, dtype=np.float32)
    legal_mask = np.ones(3510, dtype=np.uint8)
    with torch.no_grad():
        logits, value = model(
            torch.from_numpy(observation), torch.from_numpy(legal_mask.astype(np.float32))
        )
        log_probability = float(
            distributions.Categorical(logits=logits).log_prob(torch.tensor(0)).item()
        )
    return PPOTransition(
        observation,
        legal_mask,
        0,
        log_probability,
        float(value.item()),
        0.0,
        True,
        800704,
        0,
    )


def test_reference_kl_uses_the_legal_mask() -> None:
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    reference = BehaviorCloningNetwork(265, feature_version="v1", hidden_layers=(8,))
    transition = _single_transition(model)
    transition = PPOTransition(
        transition.observation,
        np.array([1, *([0] * 3509)], dtype=np.uint8),
        0,
        0.0,
        transition.old_value,
        0.0,
        True,
        transition.seed,
        transition.seat,
    )
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
    metrics = ppo_update(
        model,
        optim.Adam(model.parameters(), lr=1e-3),
        [transition],
        PPOConfig(
            hidden_layers=(8,),
            updates=1,
            games_per_update=1,
            update_epochs=1,
            minibatch_size=1,
            target_kl=None,
            reference_kl_coefficient=1.0,
            entropy_coefficient=0.0,
        ),
        update_seed=11,
        reference_model=reference,
    )
    assert np.isfinite(metrics["reference_kl"])
    assert metrics["reference_kl"] == 0.0


def test_critic_warmup_does_not_change_actor_or_trunk() -> None:
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    trunk_before = {name: value.detach().clone() for name, value in model.trunk.state_dict().items()}
    policy_before = {name: value.detach().clone() for name, value in model.policy_head.state_dict().items()}
    value_before = {name: value.detach().clone() for name, value in model.value_head.state_dict().items()}
    transition = _single_transition(model)
    transition = PPOTransition(
        transition.observation,
        transition.legal_mask,
        transition.action_index,
        transition.old_log_probability,
        transition.old_value,
        10.0,
        True,
        transition.seed,
        transition.seat,
    )
    metrics = warmup_critic(
        model,
        [transition],
        PPOConfig(
            hidden_layers=(8,),
            updates=1,
            games_per_update=1,
            critic_warmup_epochs=2,
        ),
    )
    assert metrics["optimizer_steps"] == 2
    assert all(torch.equal(value, model.trunk.state_dict()[name]) for name, value in trunk_before.items())
    assert all(torch.equal(value, model.policy_head.state_dict()[name]) for name, value in policy_before.items())
    assert any(not torch.equal(value, model.value_head.state_dict()[name]) for name, value in value_before.items())


def test_kl_stop_is_nonnegative_and_preoptimizer() -> None:
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    transition = _single_transition(model)
    transition = PPOTransition(
        transition.observation,
        transition.legal_mask,
        transition.action_index,
        transition.old_log_probability - 1.0,
        transition.old_value,
        transition.reward,
        transition.terminal,
        transition.seed,
        transition.seat,
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = ppo_update(
        model,
        optim.Adam(model.parameters(), lr=1e-3),
        [transition],
        PPOConfig(
            hidden_layers=(8,),
            updates=1,
            games_per_update=1,
            update_epochs=2,
            minibatch_size=1,
            target_kl=1e-4,
            entropy_coefficient=0.0,
        ),
        update_seed=12,
    )
    assert metrics["early_stopped"]
    assert metrics["optimizer_steps"] == 0
    assert metrics["approx_kl"] >= 0.0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize(
    "field",
    ["current_weight", "history_weight", "reference_kl_coefficient"],
)
def test_new_config_weights_reject_nan(field: str) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**{field: float("nan")})  # type: ignore[arg-type]


def test_configured_fake_fast_env_smoke_writes_incremental_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_collect(  # noqa: PLR0913 - mirror collector signature
        model: PolicyValueNetwork,
        opponent_pool: list[OpponentPoolEntry] | tuple[OpponentPoolEntry, ...],
        *,
        seed: int,
        seat: int,
        config: PPOConfig,
        update_index: int,
        game_index: int,
    ) -> tuple[dict[str, object], list[PPOTransition]]:
        del opponent_pool, config
        observation = np.zeros(265, dtype=np.float32)
        legal_mask = np.ones(3510, dtype=np.uint8)
        with torch.no_grad():
            logits, value = model(
                torch.from_numpy(observation),
                torch.from_numpy(legal_mask.astype(np.float32)),
            )
            log_probability = float(
                distributions.Categorical(logits=logits)
                .log_prob(torch.tensor(0))
                .item()
            )
        transition = PPOTransition(
            observation,
            legal_mask,
            0,
            log_probability,
            float(value.item()),
            0.0,
            True,
            seed,
            seat,
        )
        return (
            {
                "status": "completed",
                "opponent": "first",
                "seed": seed,
                "seat": seat,
                "game_index": game_index,
                "update": update_index,
            },
            [transition],
        )

    monkeypatch.setattr(ppo_module, "collect_ppo_game", fake_collect)
    output = tmp_path / "smoke"
    result = train_ppo_selfplay(
        checkpoint,
        output,
        [800705],
        [_opponent()],
        config=PPOConfig(
            hidden_layers=(8,),
            updates=2,
            games_per_update=1,
            update_epochs=1,
            minibatch_size=1,
            target_kl=None,
            current_weight=0.0,
            history_weight=1.0,
            history_limit=1,
            critic_warmup_epochs=2,
            eval_every=2,
        ),
    )

    assert result["status"] == "completed"
    assert [row["update"] for row in result["logs"]] == [0, 1, 2]
    assert (output / "initial.pth").is_file()
    assert (output / "status.json").is_file()
    assert (output / "result.json").is_file()
    assert result["logs"][1]["opponent_pool"][
        "history_weight_redistributed_to_current"
    ]
    assert not result["logs"][2]["opponent_pool"][
        "history_weight_redistributed_to_current"
    ]


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
