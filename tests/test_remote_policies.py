"""Checkpoint dispatch and score-adapter tests for remote inference."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import save_model
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    PolicyValueNetwork,
    PPOConfig,
    save_ppo_checkpoint,
)
from splendor.remote.policies import (
    DQN_MODEL_TYPE,
    IMITATION_PPO_MODEL_TYPE,
    load_policies,
    load_policy,
)

ACTION_DIM = 3510
FEATURE_DIMS = {"v1": 265, "public-v2": 312}


def _mask() -> np.ndarray:
    """Return a small legal action mask shared by score tests."""
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    mask[[17, 1234, 3000]] = 1.0
    return mask


def _dqn_checkpoint(path: Path, feature_version: str = "v1") -> QNetwork:
    """Write a compact DQN checkpoint and return its source model."""
    model = QNetwork(
        input_dim=FEATURE_DIMS[feature_version],
        hidden_layers=(8,),
        feature_version=feature_version,
    )
    save_model(model, path)
    return model


def _ppo_checkpoint(path: Path, feature_version: str = "v1") -> PolicyValueNetwork:
    """Write a compact imitation-PPO checkpoint and return its source model."""
    model = PolicyValueNetwork(
        FEATURE_DIMS[feature_version],
        feature_version=feature_version,
        hidden_layers=(8,),
    )
    with torch.no_grad():
        model.normalizer.mean.fill_(2.0)
        model.normalizer.variance.fill_(3.0)
    model.normalizer.fitted = True
    save_ppo_checkpoint(
        model,
        path,
        update=4,
        config=PPOConfig(
            feature_version=feature_version,
            hidden_layers=(8,),
            device_name="cpu",
        ),
        source_bc="synthetic",
        opponent_pool=(),
        metrics={},
    )
    return model


@pytest.mark.parametrize("feature_version", ["v1", "public-v2"])
def test_load_dqn_adapter_and_scores_match_model(
    tmp_path: Path, feature_version: str
) -> None:
    """DQN metadata and masked scores are preserved through dispatch."""
    path = tmp_path / f"dqn-{feature_version}.pth"
    _dqn_checkpoint(path, feature_version)
    policy = load_policy(path)

    assert policy.kind == DQN_MODEL_TYPE
    assert policy.score_kind == "q"
    assert policy.feature_version == feature_version
    assert policy.input_dim == FEATURE_DIMS[feature_version]
    assert policy.output_dim == ACTION_DIM
    assert policy.device == torch.device("cpu")

    obs = np.linspace(-1.0, 1.0, policy.input_dim, dtype=np.float32)
    mask = _mask()
    scores = policy.scores(obs, mask)
    with torch.no_grad():
        expected = policy.model(torch.from_numpy(obs), torch.from_numpy(mask))
    torch.testing.assert_close(scores, expected.squeeze(0))
    assert int(scores.argmax().item()) in {17, 1234, 3000}


@pytest.mark.parametrize("feature_version", ["v1", "public-v2"])
def test_load_ppo_adapter_preserves_normalizer_and_scores(
    tmp_path: Path, feature_version: str
) -> None:
    """PPO policy logits are exposed without dropping its fitted normalizer."""
    path = tmp_path / f"ppo-{feature_version}.pth"
    source = _ppo_checkpoint(path, feature_version)
    policy = load_policy(path)

    assert policy.kind == IMITATION_PPO_MODEL_TYPE
    assert policy.score_kind == "policy_logit"
    assert policy.feature_version == feature_version
    assert policy.input_dim == FEATURE_DIMS[feature_version]
    assert policy.output_dim == ACTION_DIM
    assert isinstance(policy.model, PolicyValueNetwork)
    torch.testing.assert_close(policy.model.normalizer.mean, source.normalizer.mean)
    torch.testing.assert_close(
        policy.model.normalizer.variance, source.normalizer.variance
    )
    assert policy.model.normalizer.fitted

    obs = np.linspace(-1.0, 1.0, policy.input_dim, dtype=np.float32)
    mask = _mask()
    scores = policy.scores(obs, mask)
    with torch.no_grad():
        expected, _value = policy.model(
            torch.from_numpy(obs), torch.from_numpy(mask)
        )
    torch.testing.assert_close(scores, expected.squeeze(0))
    assert int(scores.argmax().item()) in {17, 1234, 3000}


def test_historical_unmarked_dqn_is_accepted(tmp_path: Path) -> None:
    """Pre-model_type DQN snapshots remain deployable after shape checks."""
    path = tmp_path / "legacy-dqn.pth"
    _dqn_checkpoint(path)
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["model_type"] == DQN_MODEL_TYPE
    payload.pop("model_type")
    torch.save(payload, path)

    policy = load_policy(path)
    assert policy.kind == DQN_MODEL_TYPE


def test_unknown_and_legacy_policy_checkpoints_fail_closed(tmp_path: Path) -> None:
    """Unknown, old PPO, and malformed model formats are never auto-loaded."""
    unknown = tmp_path / "unknown.pth"
    torch.save(
        {"model_type": "behavior_cloning", "model_state_dict": {}, "config": {}},
        unknown,
    )
    with pytest.raises(ValueError, match="unsupported checkpoint model_type"):
        load_policy(unknown)

    legacy_ppo = tmp_path / "legacy-ppo.pth"
    torch.save(
        {
            "model_state_dict": {
                "actor.weight": torch.zeros(2, 2),
                "critic.weight": torch.zeros(1, 2),
            },
            "config": {},
        },
        legacy_ppo,
    )
    with pytest.raises(ValueError, match="legacy PPO"):
        load_policy(legacy_ppo)

    malformed = tmp_path / "malformed.pth"
    torch.save({"model_type": DQN_MODEL_TYPE, "model_state_dict": {}}, malformed)
    with pytest.raises(ValueError, match="config mapping"):
        load_policy(malformed)


def test_load_policies_registers_mixed_checkpoint_types(tmp_path: Path) -> None:
    """The registry dispatches DQN and imitation-PPO files independently."""
    dqn = tmp_path / "dqn.pth"
    ppo = tmp_path / "ppo.pth"
    _dqn_checkpoint(dqn)
    _ppo_checkpoint(ppo)
    registry = load_policies([f"dqn={dqn}", f"ppo={ppo}"])

    assert registry["dqn"].kind == DQN_MODEL_TYPE
    assert registry["ppo"].kind == IMITATION_PPO_MODEL_TYPE


def test_load_policies_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    """A duplicate name must not silently replace a loaded policy."""
    dqn = tmp_path / "dqn.pth"
    _dqn_checkpoint(dqn)
    with pytest.raises(ValueError, match="duplicate model id 'dqn'"):
        load_policies([f"dqn={dqn}", f"dqn={dqn}"])


@pytest.mark.parametrize(
    "obs,mask,error",
    [
        (np.zeros(265, dtype=np.float32), np.zeros(ACTION_DIM), "at least one"),
        (np.full(265, np.nan, dtype=np.float32), np.ones(ACTION_DIM), "non-finite"),
        (np.zeros(265, dtype=np.float32), np.full(ACTION_DIM, 0.5), "0/1"),
    ],
)
def test_scores_reject_invalid_remote_inputs(
    tmp_path: Path,
    obs: np.ndarray,
    mask: np.ndarray,
    error: str,
) -> None:
    """Malformed observations/masks fail before an action can be selected."""
    path = tmp_path / "dqn.pth"
    _dqn_checkpoint(path)
    policy = load_policy(path)
    with pytest.raises(ValueError, match=error):
        policy.scores(obs, mask)
