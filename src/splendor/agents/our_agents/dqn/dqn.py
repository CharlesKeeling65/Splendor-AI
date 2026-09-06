"""
Entry-point for DQN training.
"""

import argparse
import json
import platform
import random
import sys
import time
import typing
from csv import writer as csv_writer
from datetime import datetime
from pathlib import Path
from typing import Literal, Required, TypedDict, cast

import gymnasium as gym
import numpy as np
import torch
from torch import optim

# import this to register splendor as one of gym's environments.
# pylint: disable=unused-import
import splendor.splendor.gym  # noqa: F401
from splendor.agents.our_agents.ppo.arguments_parsing import (
    DEFAULT_OPPONENT,
    DEFAULT_TEST_OPPONENT,
    OPPONENTS_AGENTS_FACTORY,
    OPPONENTS_CHOICES,
    SELF_OPPONENT,
    WORKING_DIR,
)
from splendor.template import Agent
from splendor.version import get_version

from .constants import (
    BATCH_SIZE,
    BUFFER_SIZE,
    DISCOUNT_FACTOR,
    EPS_DECAY_FRACTION,
    EPS_END,
    EPS_START,
    EVAL_EVERY,
    EVAL_GAMES,
    HIDDEN_DIMS,
    LEARNING_RATE,
    N_STEP,
    SAVE_EVERY,
    SEED,
    TOTAL_STEPS,
    WARMUP_STEPS,
    WIN_BONUS,
)
from .dqn_agent import DQNAgent
from .network import QNetwork
from .replay_buffer import ReplayBuffer
from .reward_wrapper import TerminalRewardWrapper
from .training import DQNParams, collect_one_step, dqn_update, epsilon_at, evaluate
from .utils import DEFAULT_SAVED_DQN_PATH, save_model

FOLDER_FORMAT = "%y-%m-%d_%H-%M-%S"
STATS_FILE = "stats.csv"
STATS_HEADERS = (
    "step",
    "episode",
    "epsilon",
    "loss",
    "q_mean",
    "train_score",
    "eval_wr",
    "eval_avg_score",
)
PROGRESS_FILE = "progress.csv"
PROGRESS_HEADERS = (
    "timestamp",
    "event",
    "step",
    "episode",
    "epsilon",
    "loss",
    "q_mean",
    "td_abs_mean",
    "train_score",
    "eval_win",
    "eval_draw",
    "eval_loss",
    "eval_avg_score",
    "buffer_size",
    "elapsed_sec",
    "steps_per_sec",
    "gpu_memory_mb",
)
CONFIG_FILE = "run_config.json"
STATUS_FILE = "run_status.json"

DeviceName = Literal["cuda", "cpu", "mps"]
DEVICE_NAME_CHOICES = typing.get_args(DeviceName)


def _write_json(path: Path, payload: dict[str, typing.Any]) -> None:
    """Write a small monitoring artifact atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _gpu_memory_mb(device: torch.device) -> float | None:
    """Return currently allocated CUDA memory, when the run is on CUDA."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return round(torch.cuda.memory_allocated(device) / (1024**2), 2)


class DQNArguments(TypedDict):
    """
    TypedDict representing the command-line arguments.
    """

    working_dir: Required[Path]
    learning_rate: Required[float]
    seed: Required[int]
    device_name: Required[DeviceName]
    opponent: Required[str]
    test_opponent: Required[str]
    total_steps: Required[int]
    buffer_size: Required[int]
    batch_size: Required[int]
    win_bonus: Required[float]
    save_every: Required[int]
    eval_every: Required[int]


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,too-many-positional-arguments
def train(  # noqa: C901,PLR0913,PLR0915,PLR0917
    working_dir: Path = WORKING_DIR,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    device_name: DeviceName = "cuda",
    opponent: str = DEFAULT_OPPONENT,
    test_opponent: str = DEFAULT_TEST_OPPONENT,
    total_steps: int = TOTAL_STEPS,
    buffer_size: int = BUFFER_SIZE,
    batch_size: int = BATCH_SIZE,
    win_bonus: float = WIN_BONUS,
    save_every: int = SAVE_EVERY,
    eval_every: int = EVAL_EVERY,
) -> QNetwork:
    """
    Train a DQN agent.

    :param working_dir: Where to store the statistics and weights.
    :param learning_rate: The learning rate of the gradient descent based learning.
    :param seed: Which seed to use during training.
    :param device_name: Name of the device used for mathematical computations.
    :param opponent: Opponent agent name that the DQN would train against
                     ("itself" trains in self-play with a shared network).
    :param test_opponent: Test opponent name that the DQN would be evaluated against.
    :param total_steps: How many environment steps to train for.
    :param buffer_size: The capacity of the replay buffer.
    :param batch_size: How many transitions are sampled per gradient step.
    :param win_bonus: The terminal win/loss reward magnitude.
    :param save_every: How often (in steps) to store a checkpoint.
    :param eval_every: How often (in steps) to run the greedy evaluation.
    :return: The trained model (DQN agent).
    """
    # the reproducibility trio - `env.reset(seed=...)` does NOT fix the dealing
    # order (global `random`) nor the seating order (global numpy RNG).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        device_name if getattr(torch, device_name).is_available() else "cpu"
    )

    if opponent in OPPONENTS_AGENTS_FACTORY:
        opponents = OPPONENTS_AGENTS_FACTORY[opponent](0)
    elif opponent == SELF_OPPONENT:
        # assume that the DQN is meant to train against itself; the shared
        # network is attached once it exists (mirrors the PPO precedent).
        opponents = [DQNAgent(0, load_net=False)]
    else:
        raise ValueError(f"Unknown opponent: {opponent}")

    if test_opponent not in OPPONENTS_AGENTS_FACTORY and test_opponent != SELF_OPPONENT:
        raise ValueError(f"Unknown test opponent: {test_opponent}")

    def make_eval_opponents() -> list[Agent]:
        if test_opponent in OPPONENTS_AGENTS_FACTORY:
            return OPPONENTS_AGENTS_FACTORY[test_opponent](0)
        # self-play evaluation: the rival shares the training network.
        rival = DQNAgent(0, load_net=False)
        rival.load_policy(q_net)
        return [rival]

    print(
        f"Training DQN against opponent: {opponent}"
        f" and evaluating against test opponent: {test_opponent}"
    )

    start_time = datetime.now()
    folder = working_dir / f"{start_time.strftime(FOLDER_FORMAT)}__dqn"
    models_folder = folder / "models"
    models_folder.mkdir(parents=True)

    train_env = TerminalRewardWrapper(
        gym.make("splendor-v1", agents=opponents), win_bonus=win_bonus
    )

    params = DQNParams(
        lr=learning_rate,
        batch_size=batch_size,
        warmup=WARMUP_STEPS,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay_steps=int(EPS_DECAY_FRACTION * total_steps),
        seed=seed,
        device=device,
    )

    q_net = QNetwork().float().to(device)
    if opponent == SELF_OPPONENT:
        for opponent_agent in opponents:
            cast(DQNAgent, opponent_agent).load_policy(q_net)

    target_net = QNetwork().float().to(device)
    target_net.load_state_dict(q_net.state_dict())

    buffer = ReplayBuffer(buffer_size, n_step=N_STEP, gamma=DISCOUNT_FACTOR)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    checkpoint_config = {
        "gamma": params.gamma,
        "lr": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "warmup": params.warmup,
        "eps_decay_steps": params.eps_decay_steps,
        "n_step": N_STEP,
        "win_bonus": win_bonus,
        "hidden_layers": list(HIDDEN_DIMS),
        "use_input_norm": True,
        "dueling": True,
        "opponent": opponent,
        "test_opponent": test_opponent,
        "seed": seed,
    }

    started_at = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    run_config: dict[str, typing.Any] = {
        **checkpoint_config,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "save_every": save_every,
        "eval_every": eval_every,
        "eval_games": EVAL_GAMES,
        "observation_dim": int(q_net.input_dim),
        "action_dim": int(q_net.output_dim),
        "device": str(device),
        "device_name": gpu_name or str(device),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "started_at": started_at.isoformat(timespec="seconds"),
    }
    _write_json(folder / CONFIG_FILE, run_config)

    episode = 0
    latest_update = {"loss": 0.0, "q_mean": 0.0, "td_abs_mean": 0.0}
    latest_eval = {"win": 0.0, "draw": 0.0, "loss": 0.0, "avg_score": 0.0}

    train_env.reset(seed=seed)

    status_path = folder / STATUS_FILE
    with (
        Path.open(
            folder / STATS_FILE,
            "w",
            buffering=1,
            newline="\n",
            encoding="ascii",
        ) as stats_file,
        Path.open(
            folder / PROGRESS_FILE,
            "w",
            buffering=1,
            newline="\n",
            encoding="ascii",
        ) as progress_file,
    ):
        stats_csv = csv_writer(stats_file)
        stats_csv.writerow(STATS_HEADERS)
        stats_file.flush()
        progress_csv = csv_writer(progress_file)
        progress_csv.writerow(PROGRESS_HEADERS)
        progress_file.flush()

        def write_progress(
            event: str,
            current_step: int,
            current_episode: int,
            train_score: float | None = None,
        ) -> None:
            """Append a live-monitoring row and publish the run status."""
            elapsed = max(time.monotonic() - started_monotonic, 1e-9)
            progress_csv.writerow(
                [
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    event,
                    current_step,
                    current_episode,
                    round(epsilon_at(current_step, params), 6),
                    latest_update["loss"],
                    latest_update["q_mean"],
                    latest_update["td_abs_mean"],
                    train_score,
                    latest_eval["win"],
                    latest_eval["draw"],
                    latest_eval["loss"],
                    latest_eval["avg_score"],
                    len(buffer),
                    round(elapsed, 3),
                    round(current_step / elapsed, 3),
                    _gpu_memory_mb(device),
                ]
            )
            progress_file.flush()
            _write_json(
                status_path,
                {
                    "status": "completed" if event == "complete" else "running",
                    "event": event,
                    "step": current_step,
                    "episode": current_episode,
                    "total_steps": total_steps,
                    "updated_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                },
            )

        write_progress("start", 0, 0)

        # Main training loop
        for step in range(total_steps):
            result = collect_one_step(train_env, q_net, buffer, params, step)

            if len(buffer) >= params.warmup:
                latest_update = dqn_update(
                    q_net, target_net, buffer, optimizer, params, step=step
                )

            if result["episode_ended"]:
                final_score = result["final_score"]
                train_score = (
                    float(final_score)
                    if isinstance(final_score, (float, int))
                    else None
                )
                stats_csv.writerow(
                    [
                        step,
                        episode,
                        round(epsilon_at(step, params), 4),
                        latest_update["loss"],
                        latest_update["q_mean"],
                        result["final_score"],
                        latest_eval["win"],
                        latest_eval["avg_score"],
                    ]
                )
                stats_file.flush()
                episode += 1
                write_progress("episode", step + 1, episode, train_score)

            if (step + 1) % eval_every == 0:
                latest_eval = evaluate(q_net, make_eval_opponents, EVAL_GAMES)
                print(
                    f"| Step: {step + 1} | Win: {latest_eval['win']:.2f} | "
                    f"Draw: {latest_eval['draw']:.2f} | Loss: {latest_eval['loss']:.2f} | "
                    f"Avg Score: {latest_eval['avg_score']:.2f} |",
                    flush=True,
                )
                write_progress("eval", step + 1, episode)

            if (step + 1) % save_every == 0:
                # explicit floor division inside the braces - an unbracketed
                # `step + 1 // save_every` would bind `1 // save_every` first
                # (the precedence bug ppo.py:293 suffers from, making its
                # checkpoints overwrite each other).
                save_model(
                    q_net,
                    models_folder / f"dqn_model_{(step // save_every):d}.pth",
                    step=step + 1,
                    config=checkpoint_config,
                )
                write_progress("checkpoint", step + 1, episode)

        save_model(
            q_net,
            models_folder / "dqn_model.pth",
            step=total_steps,
            config=checkpoint_config,
        )
        save_model(
            q_net, DEFAULT_SAVED_DQN_PATH, step=total_steps, config=checkpoint_config
        )
        write_progress("complete", total_steps, episode)

    print(f"Final model saved to {models_folder / 'dqn_model.pth'}")
    print(f"Deployed model saved to {DEFAULT_SAVED_DQN_PATH}")

    return q_net


def parse_args() -> DQNArguments:
    """
    Parse command-line arguments.

    :return: dictionary storing all the required arguments.
    """
    parser = argparse.ArgumentParser(
        prog="dqn",
        description="Train a DQN agent.",
    )
    parser.add_argument("--version", action="version", version=get_version())
    parser.add_argument(
        "-l",
        "--learning-rate",
        default=LEARNING_RATE,
        type=float,
        help="The learning rate to use during training with gradient descent",
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        default=WORKING_DIR,
        type=Path,
        help="Path to directory to work in (will create a directory with "
        "current timestamp for each run)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=SEED,
        type=int,
        help="Seed to set for numpy's, torch's and random's random number generators.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=DEVICE_NAME_CHOICES,
        dest="device_name",
        help="On which device to do heavy mathematical computation",
    )
    parser.add_argument(
        "-o",
        "--opponent",
        type=str,
        default=DEFAULT_OPPONENT,
        choices=OPPONENTS_CHOICES,
        help="Against whom the DQN should train",
    )
    parser.add_argument(
        "--test-opponent",
        type=str,
        default=DEFAULT_TEST_OPPONENT,
        choices=OPPONENTS_CHOICES,
        help="Against whom the DQN should be evaluated",
    )
    parser.add_argument(
        "--total-steps",
        default=TOTAL_STEPS,
        type=int,
        help="How many environment steps to train for",
    )
    parser.add_argument(
        "--buffer-size",
        default=BUFFER_SIZE,
        type=int,
        help="The capacity of the replay buffer",
    )
    parser.add_argument(
        "--batch-size",
        default=BATCH_SIZE,
        type=int,
        help="How many transitions are sampled per gradient step",
    )
    parser.add_argument(
        "--win-bonus",
        default=WIN_BONUS,
        type=float,
        help="The terminal win/loss reward magnitude",
    )
    parser.add_argument(
        "--save-every",
        default=SAVE_EVERY,
        type=int,
        help="How often (in steps) to store a checkpoint",
    )
    parser.add_argument(
        "--eval-every",
        default=EVAL_EVERY,
        type=int,
        help="How often (in steps) to run the greedy evaluation",
    )

    options: argparse.Namespace = parser.parse_args()

    return cast(DQNArguments, vars(options))


def main() -> None:
    """
    Entry-point for the ``dqn`` console script.
    """
    options = parse_args()
    train(**options)


if __name__ == "__main__":
    main()
