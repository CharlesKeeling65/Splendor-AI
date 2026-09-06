"""Reproducible held-out evaluation for DQN checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Literal

import torch

from splendor.agents.our_agents.ppo.arguments_parsing import (
    OPPONENTS_AGENTS_FACTORY,
)

from .training import evaluate
from .utils import load_saved_dqn

DeviceName = Literal["cuda", "cpu", "mps"]
DEVICE_CHOICES: tuple[DeviceName, ...] = ("cuda", "cpu", "mps")


def _resolve_checkpoint(path: Path) -> Path:
    """Accept either a checkpoint file or a DQN run directory."""
    if path.is_dir():
        candidates = (
            path / "models" / "dqn_model.pth",
            path / "dqn_model.pth",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no DQN checkpoint found below {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _select_device(name: DeviceName) -> torch.device:
    """Resolve a requested accelerator, falling back safely to CPU."""
    requested = torch.device(name)
    if getattr(torch, name).is_available():
        return requested
    return torch.device("cpu")


def evaluate_checkpoint(
    checkpoint_path: Path,
    opponent: str,
    games: int = 100,
    seed: int = 1234,
    device_name: DeviceName = "cpu",
) -> dict[str, float | int | str]:
    """Evaluate one checkpoint on a fixed, reproducible game set."""
    if games < 1:
        raise ValueError(f"games must be positive, got {games}")
    if opponent not in OPPONENTS_AGENTS_FACTORY:
        available = ", ".join(sorted(OPPONENTS_AGENTS_FACTORY))
        raise ValueError(f"unknown opponent {opponent!r}; available: {available}")

    resolved_path = _resolve_checkpoint(checkpoint_path)
    device = _select_device(device_name)
    network = load_saved_dqn(resolved_path).to(device)
    network.eval()

    result = evaluate(
        network,
        lambda: OPPONENTS_AGENTS_FACTORY[opponent](0),
        n_games=games,
        seed=seed,
    )
    checkpoint = torch.load(
        str(resolved_path), map_location="cpu", weights_only=False
    )
    step = checkpoint.get("step", 0)
    return {
        "checkpoint": str(resolved_path),
        "checkpoint_step": int(step),
        "opponent": opponent,
        "games": games,
        "seed": seed,
        "device": str(device),
        "wins": round(result["win"] * games),
        "draws": round(result["draw"] * games),
        "losses": round(result["loss"] * games),
        **result,
    }


def main() -> None:
    """Run the checkpoint evaluator and print a JSON result."""
    parser = argparse.ArgumentParser(
        prog="dqn-evaluate",
        description="Evaluate a DQN checkpoint against a held-out opponent.",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "-o",
        "--opponent",
        choices=tuple(sorted(OPPONENTS_AGENTS_FACTORY)),
        default="minimax",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        opponent=args.opponent,
        games=args.games,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
