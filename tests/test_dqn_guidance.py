"""Training-only guidance, legal gradients, fading supervision and schema ablations."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.experiment import ExperimentConfig, train_variant
from splendor.agents.our_agents.dqn.guidance import (
    GuidanceBuffer,
    guidance_fraction,
    legal_margin_loss,
    teacher_action_index,
)
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.training import synchronize_normalization
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.utils import LimitRoundsGameRule


def test_legal_margin_ignores_illegal_max_and_has_correct_gradient():
    q = torch.tensor([[1.0, 2.0, 1000.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]])
    loss = legal_margin_loss(q, mask, torch.tensor([0]), margin=0.8)
    assert loss.item() == pytest.approx(1.8)
    loss.backward()
    torch.testing.assert_close(q.grad, torch.tensor([[-1.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="legal"):
        legal_margin_loss(q, mask, torch.tensor([2]))


def test_margin_zero_for_teacher_with_sufficient_advantage():
    q = torch.tensor([[3.0, 2.0, 1000.0]])
    assert (
        legal_margin_loss(q, torch.tensor([[1, 1, 0]]), torch.tensor([0])).item() == 0
    )


def test_guidance_really_exits_and_buffer_is_bounded():
    assert guidance_fraction(0, 100) == 1
    assert guidance_fraction(50, 100) == 0.5
    assert guidance_fraction(100, 100) == 0
    assert guidance_fraction(1000, 100) == 0
    buffer = GuidanceBuffer(2, 4, 3, 17)
    for i in range(5):
        buffer.add(np.full(4, i, dtype=np.float32), np.ones(3, np.float32), 1)
    assert buffer.size == 2
    assert buffer.masks.dtype == np.bool_
    net = QNetwork(input_dim=4, output_dim=3, hidden_layers=(8,))
    assert torch.isfinite(buffer.loss(net, 2))


def test_normalization_sync_does_not_copy_weights():
    online = QNetwork(hidden_layers=(8,))
    target = deepcopy(online)
    with torch.no_grad():
        online.input_norm.running_mean.add_(2)
        online.input_norm.running_var.add_(3)
        next(online.parameters()).add_(5)
    weight = next(target.parameters()).detach().clone()
    synchronize_normalization(online, target)
    torch.testing.assert_close(
        online.input_norm.running_mean, target.input_norm.running_mean
    )
    torch.testing.assert_close(
        online.input_norm.running_var, target.input_norm.running_var
    )
    torch.testing.assert_close(next(target.parameters()), weight)


def test_teacher_uses_existing_legal_mapping():
    rule = LimitRoundsGameRule(2)
    state = rule.current_game_state
    before = deepcopy(state.board.gems)
    action = teacher_action_index(state, rule, 0)
    mapping = create_action_mapping(rule.getLegalActions(state, 0), state, 0)
    assert action in mapping
    assert state.board.gems == before


@pytest.mark.parametrize("variant", ["public-ema", "public-sync", "public-demo"])
def test_new_variants_use_ema_and_train(
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
        demo_decay_steps=12,
        validation_opponents="minimax,heuristic",
        device="cpu",
    )
    path = train_variant(tmp_path / variant, variant, 18, config)
    net = load_saved_dqn(path)
    assert net.feature_version == "public-v2"
    assert not net.normalization_frozen
