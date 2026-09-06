"""
Collection of utility functions.
"""

from pathlib import Path
from typing import Any

import torch

from .constants import HIDDEN_DIMS
from .network import QNetwork

DEFAULT_SAVED_DQN_PATH = Path(__file__).parent / "dqn_model.pth"


def save_model(
    model: QNetwork, path: Path, step: int = 0, config: dict[str, Any] | None = None
) -> None:
    """
    Save the weights of a Q-network into a file at the given path.

    The input normalization's running statistics are part of the model: if
    they are not stored & restored with the checkpoint, deployment normalizes
    observations differently than training did and every Q value is silently
    distorted (the symptom is just a weaker agent - very hard to trace back).
    The statistics are stored as (1, obs_dim), the PPO checkpoint convention,
    so future tooling can be shared between both agents.

    :param model: the model whose weights should be stored.
    :param path: where to store the weights.
    :param step: the global training step the model was saved at.
    :param config: the training configuration to store alongside the weights
                   (also used by ``load_saved_dqn`` to rebuild the network).
    """
    # Serialize tensors on CPU without moving the live network.  Moving the
    # model here would break a CUDA training loop immediately after the first
    # periodic checkpoint.
    model_state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    checkpoint: dict[str, Any] = {
        "model_state_dict": model_state_dict,
        "step": step,
        "config": {
            **(config or {}),
            "feature_version": model.feature_version,
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_layers": list(model.hidden_layers),
            "dueling": model.dueling,
            "use_input_norm": model.input_norm is not None,
            "auxiliary_heads": model.auxiliary_heads,
            "normalization_frozen": model.normalization_frozen,
        },
    }
    if model.input_norm is not None:
        checkpoint["running_mean"] = (
            model.input_norm.running_mean.detach().cpu().clone().reshape(1, -1)
        )
        checkpoint["running_var"] = (
            model.input_norm.running_var.detach().cpu().clone().reshape(1, -1)
        )

    torch.save(checkpoint, str(path))


def load_saved_dqn(path: Path | None = None) -> QNetwork:
    """
    Load the saved weights of a DQN model from a given path; if no path is
    given, the installed weights of the DQN agent are loaded.

    :param path: where to load the weights from.
    :return: the loaded model (in eval mode, ready for action selection).
    """
    if path is None:
        path = DEFAULT_SAVED_DQN_PATH

    checkpoint = torch.load(
        str(path),
        weights_only=False,
        map_location="cpu",
    )
    saved_config: dict[str, Any] = checkpoint.get("config") or {}

    net = QNetwork(
        input_dim=saved_config.get("input_dim", 265),
        output_dim=saved_config.get("output_dim", 3510),
        feature_version=saved_config.get("feature_version", "v1"),
        auxiliary_heads=saved_config.get("auxiliary_heads", False),
        hidden_layers=tuple(saved_config.get("hidden_layers", HIDDEN_DIMS)),
        use_input_norm=saved_config.get("use_input_norm", True),
        dueling=saved_config.get("dueling", True),
    )
    state_dict = checkpoint["model_state_dict"]
    if net.input_norm is not None:
        for key in ("running_mean", "running_var"):
            name = f"input_norm.{key}"
            state_dict[name] = state_dict[name].reshape(1, -1)
    net.load_state_dict(state_dict)
    net.normalization_frozen = saved_config.get("normalization_frozen", False)
    if net.input_norm is not None and "running_mean" in checkpoint:
        # both running_mean & running_var are stored as (1, obs_dim) rather
        # than (obs_dim,) - the PPO convention (mirrors ppo/utils.py).
        net.input_norm.running_mean = checkpoint["running_mean"].reshape(1, -1)
        net.input_norm.running_var = checkpoint["running_var"].reshape(1, -1)

    return net
