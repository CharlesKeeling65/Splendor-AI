"""Checkpoint loading and score adapters for remote inference.

The remote service only needs an action score for each legal action.  DQN
models expose those scores directly as Q values, while imitation-PPO models
return a ``(masked_logits, value)`` pair.  This module keeps that difference
at the checkpoint boundary so consumers do not accidentally treat a PPO
policy logit as a Q value.

Only the current, feed-forward models are deployable through this seam:

* :class:`~splendor.agents.our_agents.dqn.network.QNetwork` (``dqn``);
* :class:`~splendor.agents.our_agents.policy_imitation.ppo_selfplay.PolicyValueNetwork`
  (``imitation_ppo_policy_value``).

The old course PPO, self-attention PPO, GRU, and LSTM checkpoints are
intentionally rejected.  They do not carry the feature/model contract needed
by the browser deployment and some of them require recurrent state.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.agents.our_agents.policy_imitation.bc_training import (
    DeviceName,
    resolve_device,
)
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    PolicyValueNetwork,
    load_ppo_checkpoint,
)
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

ACTION_DIM = len(ALL_ACTIONS)
MATRIX_RANK = 2

PolicyKind = Literal["dqn", "imitation_ppo_policy_value"]
ScoreKind = Literal["q", "policy_logit"]
DQN_MODEL_TYPE: PolicyKind = "dqn"
IMITATION_PPO_MODEL_TYPE: PolicyKind = "imitation_ppo_policy_value"
SupportedModel = QNetwork | PolicyValueNetwork
TensorInput = torch.Tensor | NDArray[Any]


class ScoredPolicy:
    """A frozen model with a common masked-action score interface.

    ``scores`` preserves the batch rank of its observation argument: a single
    observation returns ``(output_dim,)`` and a batch returns
    ``(batch_size, output_dim)``.  Scores are returned as a ``torch.Tensor`` on
    :attr:`device`, which lets callers perform ``argmax``/``topk`` without an
    unnecessary device round trip.

    The adapter never calls ``observe`` or updates normalization statistics.
    Both loaders restore the statistics embedded in their checkpoint before
    this object freezes the model in evaluation mode.  In particular, PPO's
    ``FixedNormalizer.mean`` and ``variance`` remain part of the loaded state.
    """

    def __init__(
        self,
        model: SupportedModel,
        *,
        kind: PolicyKind,
        score_kind: ScoreKind,
        device: torch.device | None = None,
    ) -> None:
        """Wrap a supported model and freeze it for inference."""
        if kind not in {DQN_MODEL_TYPE, IMITATION_PPO_MODEL_TYPE}:
            raise ValueError(f"unsupported policy kind {kind!r}")
        if score_kind not in {"q", "policy_logit"}:
            raise ValueError(f"unsupported score kind {score_kind!r}")
        if not isinstance(model, (QNetwork, PolicyValueNetwork)):
            raise TypeError(
                "remote inference supports only QNetwork and "
                "PolicyValueNetwork models"
            )
        expected_kind = (
            DQN_MODEL_TYPE
            if isinstance(model, QNetwork)
            else IMITATION_PPO_MODEL_TYPE
        )
        if kind != expected_kind:
            raise ValueError(
                f"policy kind {kind!r} does not match model {type(model).__name__}"
            )
        expected_score_kind: ScoreKind = (
            "q" if isinstance(model, QNetwork) else "policy_logit"
        )
        if score_kind != expected_score_kind:
            raise ValueError(
                f"score kind {score_kind!r} does not match model {type(model).__name__}"
            )

        target_device = device or next(model.parameters()).device
        self.model = model.to(target_device).eval()
        self.kind = kind
        self.score_kind = score_kind
        self.feature_version = str(model.feature_version)
        self.input_dim = int(model.input_dim)
        self.output_dim = int(model.output_dim)
        self.device = target_device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_dqn(
        cls, model: QNetwork, *, device: torch.device | None = None
    ) -> "ScoredPolicy":
        """Wrap a DQN whose masked scores are Q values."""
        return cls(
            model,
            kind=DQN_MODEL_TYPE,
            score_kind="q",
            device=device,
        )

    @classmethod
    def from_ppo(
        cls, model: PolicyValueNetwork, *, device: torch.device | None = None
    ) -> "ScoredPolicy":
        """Wrap an imitation-PPO policy, exposing masked policy logits."""
        return cls(
            model,
            kind=IMITATION_PPO_MODEL_TYPE,
            score_kind="policy_logit",
            device=device,
        )

    def scores(self, obs: TensorInput, mask: TensorInput) -> torch.Tensor:
        """Return finite masked action scores for one observation or a batch.

        Inputs are converted to float32 on the model device.  The adapter
        rejects malformed/non-finite observations, non-binary masks, and rows
        without a legal action before invoking a model.  These checks keep a
        bad remote frame from silently selecting an illegal action.
        """
        observations = _as_float_tensor(obs, self.device, "obs")
        masks = _as_float_tensor(mask, self.device, "mask")
        single = _validate_inputs(
            observations,
            masks,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        model_observations = (
            observations.unsqueeze(0) if single else observations
        )
        model_masks = masks.unsqueeze(0) if single else masks
        with torch.no_grad():
            if isinstance(self.model, QNetwork):
                output: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (
                    self.model(model_observations, model_masks)
                )
            else:
                output = self.model(model_observations, model_masks)
            score_tensor = output[0] if isinstance(output, tuple) else output
        if score_tensor.shape != model_masks.shape:
            raise ValueError(
                f"model scores shape {tuple(score_tensor.shape)} != mask shape "
                f"{tuple(model_masks.shape)}"
            )
        if not torch.isfinite(score_tensor).all():
            raise FloatingPointError("model produced non-finite action scores")
        return score_tensor[0] if single else score_tensor

    def act(self, obs: TensorInput, mask: TensorInput) -> int:
        """Return the greedy legal action index for one observation."""
        scores = self.scores(obs, mask)
        if scores.ndim != 1:
            raise ValueError("act expects one observation, not a batch")
        return int(scores.argmax(dim=-1).item())

    def metadata(self, model_id: str | None = None) -> dict[str, Any]:
        """Return JSON-friendly model metadata for a protocol handshake."""
        result: dict[str, Any] = {
            "feature_version": self.feature_version,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "kind": self.kind,
            "score_kind": self.score_kind,
            "device": str(self.device),
        }
        if model_id is not None:
            result["id"] = model_id
        return result


def load_policy(
    path: Path,
    *,
    device_name: DeviceName = "cpu",
) -> ScoredPolicy:
    """Load one supported checkpoint and wrap it as a :class:`ScoredPolicy`.

    Current checkpoints carry a top-level ``model_type``.  Historical DQN
    snapshots predate that field, so they are accepted only after their state
    dictionary is checked for the feed-forward DQN shape.  An unmarked PPO,
    recurrent, self-attention, BC, or arbitrary checkpoint fails closed.
    """
    checkpoint = _read_checkpoint(path)
    if "model_type" not in checkpoint:
        _validate_legacy_dqn_checkpoint(checkpoint)
        dqn_model = load_saved_dqn(path)
        return ScoredPolicy.from_dqn(
            dqn_model, device=resolve_device(device_name)
        )
    model_type = checkpoint["model_type"]
    if model_type == DQN_MODEL_TYPE:
        _validate_dqn_checkpoint(checkpoint)
        dqn_model = load_saved_dqn(path)
        return ScoredPolicy.from_dqn(
            dqn_model, device=resolve_device(device_name)
        )
    if model_type == IMITATION_PPO_MODEL_TYPE:
        _validate_ppo_checkpoint(checkpoint)
        ppo_model = load_ppo_checkpoint(path, device_name=device_name)
        return ScoredPolicy.from_ppo(
            ppo_model, device=resolve_device(device_name)
        )
    raise ValueError(
        f"unsupported checkpoint model_type {model_type!r}; supported types are "
        f"{DQN_MODEL_TYPE!r} and {IMITATION_PPO_MODEL_TYPE!r}"
    )


# ``load_checkpoint`` is a descriptive alias for callers that do not want to
# choose between a DQN/PPO-specific name at their call site.
load_checkpoint = load_policy


def load_policies(
    specs: Sequence[str] = (),
    models_dir: Path | None = None,
    *,
    device_name: DeviceName = "cpu",
) -> dict[str, ScoredPolicy]:
    """Build a named policy registry from ``name=path`` and/or a directory."""
    policies: dict[str, ScoredPolicy] = {}
    if models_dir is not None:
        for path in sorted(models_dir.glob("*.pth")):
            policies[path.stem] = load_policy(path, device_name=device_name)
    for spec in specs:
        name, separator, path_text = spec.partition("=")
        if not separator or not name or not path_text:
            raise ValueError(f"--model expects name=path, got {spec!r}")
        policies[name] = load_policy(Path(path_text), device_name=device_name)
    if not policies:
        raise ValueError("no models registered (use --model name=path / --models-dir)")
    return policies


# Keep the registry name parallel to the existing ``remote.server`` helper;
# the server can migrate to this implementation without changing its CLI.
load_models = load_policies


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Read one local checkpoint and require the repository mapping shape."""
    checkpoint = torch.load(
        str(path),
        weights_only=False,
        map_location="cpu",
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    return dict(checkpoint)


def _state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a checkpoint state dictionary or a useful format error."""
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint is missing a model_state_dict mapping")
    return state


def _config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a checkpoint config mapping or a useful format error."""
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint is missing a config mapping")
    return config


def _has_dqn_shape(state: Mapping[str, Any]) -> bool:
    """Recognize the feed-forward DQN state-key contract."""
    keys = {str(key) for key in state}
    has_trunk = any(key.startswith("net.") for key in keys)
    has_q_head = "q_head.weight" in keys
    has_dueling_heads = {
        "value_head.weight",
        "advantage_head.weight",
    }.issubset(keys)
    has_recurrent_or_legacy_policy = any(
        key.startswith(prefix)
        for key in keys
        for prefix in ("recurrent_unit.", "self_attention.", "actor.", "critic.")
    )
    return (
        has_trunk
        and (has_q_head or has_dueling_heads)
        and not has_recurrent_or_legacy_policy
    )


def _has_ppo_shape(state: Mapping[str, Any]) -> bool:
    """Recognize the current feed-forward imitation-PPO state-key contract."""
    keys = {str(key) for key in state}
    return {
        "normalizer.mean",
        "normalizer.variance",
        "policy_head.weight",
        "policy_head.bias",
    }.issubset(keys) and any(key.startswith("trunk.") for key in keys)


def _validate_dqn_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Fail before the DQN loader on a malformed or foreign checkpoint."""
    state = _state_dict(checkpoint)
    _config(checkpoint)
    if not _has_dqn_shape(state):
        raise ValueError(
            "checkpoint declares DQN but its state dictionary is not a "
            "feed-forward QNetwork"
        )


def _validate_legacy_dqn_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Validate an unmarked snapshot before allowing historical DQN loading."""
    state = _state_dict(checkpoint)
    _config(checkpoint)
    if _has_dqn_shape(state):
        return
    if _has_ppo_shape(state) or any(
        str(key).startswith(prefix)
        for key in state
        for prefix in ("actor.", "critic.", "recurrent_unit.", "self_attention.")
    ):
        raise ValueError(
            "unmarked checkpoint is a legacy PPO/GRU/LSTM/self-attention "
            "model; only historical unmarked DQN checkpoints are supported"
        )
    raise ValueError(
        "unmarked checkpoint is not a structurally valid historical DQN checkpoint"
    )


def _validate_ppo_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Fail before the PPO loader on a malformed or foreign checkpoint."""
    state = _state_dict(checkpoint)
    config = _config(checkpoint)
    if not _has_ppo_shape(state):
        raise ValueError(
            "checkpoint declares imitation PPO but its state dictionary is not "
            "a PolicyValueNetwork"
        )
    if "feature_version" not in config or "input_dim" not in config:
        raise ValueError(
            "imitation PPO checkpoint config must declare feature_version and input_dim"
        )


def _as_float_tensor(
    value: TensorInput,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Convert a numpy/tensor input to float32 on ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    try:
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted to a float tensor") from error


def _validate_inputs(
    observations: torch.Tensor,
    masks: torch.Tensor,
    *,
    input_dim: int,
    output_dim: int,
) -> bool:
    """Validate ranks, dimensions, finiteness, and legal mask values."""
    if observations.ndim not in {1, 2}:
        raise ValueError(f"obs must have rank 1 or 2, got {observations.ndim}")
    if masks.ndim != observations.ndim:
        raise ValueError(
            f"obs and mask ranks must match, got {observations.ndim} and {masks.ndim}"
        )
    if observations.shape[-1] != input_dim:
        raise ValueError(
            f"obs shape {tuple(observations.shape)} does not end in ({input_dim},)"
        )
    if masks.shape[-1] != output_dim:
        raise ValueError(
            f"mask shape {tuple(masks.shape)} does not end in ({output_dim},)"
        )
    if observations.ndim == MATRIX_RANK and observations.shape[0] != masks.shape[0]:
        raise ValueError(
            f"obs and mask batch sizes differ: {observations.shape[0]} and "
            f"{masks.shape[0]}"
        )
    if not torch.isfinite(observations).all():
        raise ValueError("obs contains non-finite values")
    if not torch.isfinite(masks).all():
        raise ValueError("mask contains non-finite values")
    if not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("mask must contain only 0/1 values")
    legal_counts = masks.sum(dim=-1)
    if not torch.all(legal_counts > 0):
        raise ValueError("each mask row must contain at least one legal action")
    return observations.ndim == 1


__all__ = [
    "ACTION_DIM",
    "DQN_MODEL_TYPE",
    "IMITATION_PPO_MODEL_TYPE",
    "PolicyKind",
    "ScoreKind",
    "ScoredPolicy",
    "load_checkpoint",
    "load_models",
    "load_policies",
    "load_policy",
]
