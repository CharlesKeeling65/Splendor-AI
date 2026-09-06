"""Reproducible staged DQN ablations; opt-in and never installs trained weights.

python -m splendor.agents.our_agents.dqn.experiment --help
Each stage is an independent equal-step training run, not sequential fine-tuning.
"""

import argparse
import csv
import json
import random
import subprocess
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

import splendor.splendor.gym  # noqa: F401
from splendor.splendor.gym.envs.splendor_env import SplendorEnv

from .benchmark import benchmark, isolated_rng
from .features import extract_observation, observation_dim
from .network import QNetwork
from .population import PopulationAgent
from .replay_buffer import ReplayBuffer
from .reward_wrapper import TerminalRewardWrapper
from .search import outcome, search_policy
from .training import DQNParams, dqn_update, epsilon_at
from .utils import load_saved_dqn, save_model

VARIANTS = ("corrected", "frozen", "public", "population", "search")
# Disjoint ranges reserved before any runs. Test seeds never select checkpoints.
VALIDATION_START = 700_000
TEST_START = 900_000


@dataclass
class ExperimentConfig:
    """All training and comparison budgets saved before sampling begins."""

    steps: int = 20_000
    warmup: int = 1_000
    batch_size: int = 128
    buffer_size: int = 30_000
    lr: float = 5e-5
    tau: float = 0.001
    eps_decay_steps: int = 8_000
    eval_every: int = 5_000
    validation_deals: int = 5
    test_deals: int = 25
    snapshot_every: int = 2_000
    simulations: int = 16
    search_every: int = 8
    device: str = "cuda"


def write_json(path: Path, value: object) -> None:
    """Atomically replace a status/result file inside a unique run directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


AuxRow = tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None, float
]


def auxiliary_update(
    net: QNetwork,
    rows: deque[AuxRow],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Outcome targets for all states, policy targets only for searched states."""
    device = next(net.parameters()).device
    batch = [rows[int(i)] for i in rng.integers(len(rows), size=min(32, len(rows)))]
    obs = torch.from_numpy(np.stack([r[0] for r in batch])).to(device)
    masks = torch.from_numpy(np.stack([r[1] for r in batch])).to(device)
    targets = torch.tensor([r[3] for r in batch], device=device)
    logits, values = net.policy_value(obs, masks)
    value_loss = F.mse_loss(values, targets)
    selected = [i for i, row in enumerate(batch) if row[2] is not None]
    policy_loss = logits.sum() * 0.0
    if selected:
        pi = torch.from_numpy(
            np.stack([cast(NDArray[np.float32], batch[i][2]) for i in selected])
        ).to(device)
        policy_loss = -(pi * logits[selected].log_softmax(-1)).sum(-1).mean()
    loss = 0.2 * (value_loss + policy_loss)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return {
        "value_loss": float(value_loss.item()),
        "policy_loss": float(policy_loss.item()),
    }


def train_variant(  # noqa: C901, PLR0912, PLR0915 - explicit experiment lifecycle
    folder: Path,
    variant: str,
    seed: int,
    config: ExperimentConfig,
) -> Path:
    """Train once, validate separately, save final and validation-selected models."""
    folder.mkdir(parents=True, exist_ok=False)
    if variant not in VARIANTS:
        raise ValueError(variant)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    version = "v1" if variant in {"corrected", "frozen"} else "public-v2"
    search = variant == "search"
    population = PopulationAgent(
        0, seed + 100_000, history=variant in {"population", "search"}
    )
    net = QNetwork(
        input_dim=observation_dim(version),
        feature_version=version,
        auxiliary_heads=search,
    ).to(device)
    target = deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size, obs_dim=net.input_dim, n_step=3)
    params = DQNParams(
        lr=config.lr,
        batch_size=config.batch_size,
        warmup=config.warmup,
        eps_decay_steps=config.eps_decay_steps,
        tau=config.tau,
        device=device,
    )
    env = TerminalRewardWrapper(
        gym.make("splendor-v1", agents=[population]), win_bonus=10
    )
    base = cast(SplendorEnv, env.unwrapped)
    env.reset(seed=seed)
    aux: deque[AuxRow] = deque(maxlen=4096)
    trajectory: list[
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]
    ] = []
    search_rng = np.random.default_rng(seed + 200_000)
    aux_rng = np.random.default_rng(seed + 300_000)
    metadata = {
        **asdict(config),
        "variant": variant,
        "seed": seed,
        "feature_version": version,
        "validation_start": VALIDATION_START,
        "test_start": TEST_START,
        "torch_version": torch.__version__,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write_json(folder / "config.json", metadata)
    episodes = 0
    best_validation = -1.0
    started = time.monotonic()
    stats: dict[str, float] = {}
    step = 0
    try:
        with (folder / "training.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = [
                "step",
                "episodes",
                "epsilon",
                "buffer",
                "loss",
                "q_mean",
                "td_abs_p90",
                "grad_norm",
                "value_loss",
                "policy_loss",
                "elapsed_seconds",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step in range(1, config.steps + 1):
                obs = extract_observation(base.state, base.my_turn, version)
                mask = base.get_legal_actions_mask().astype(np.float32)
                net.observe(torch.from_numpy(obs).to(device))
                if step == config.warmup and variant != "corrected":
                    # Estimate from the whole warmup population, not the last
                    # ~10 observations of EMA. Constant/rare features retain
                    # a finite physical scale instead of exploding later.
                    if net.input_norm is not None and len(buffer):
                        warmup_obs = torch.from_numpy(buffer.obs[: len(buffer)]).to(
                            device
                        )
                        net.input_norm.running_mean.copy_(
                            warmup_obs.mean(0, keepdim=True)
                        )
                        net.input_norm.running_var.copy_(
                            warmup_obs.var(0, unbiased=False, keepdim=True).clamp_min(
                                1.0
                            )
                        )
                    net.normalization_frozen = True
                    # No gradient updates preceded warmup; exact common scale.
                    target.load_state_dict(net.state_dict())
                    target.normalization_frozen = True
                pi = None
                if search and step > config.warmup and step % config.search_every == 0:
                    pi = search_policy(
                        net,
                        base.game_rule,
                        config.simulations,
                        search_rng,
                        root_noise=True,
                    )
                    action = int(search_rng.choice(len(pi), p=pi))
                elif random.random() < epsilon_at(step, params):
                    action = int(np.random.choice(np.flatnonzero(mask)))
                else:
                    action = net.act(
                        torch.from_numpy(obs).to(device),
                        torch.from_numpy(mask).to(device),
                    )
                if search:
                    trajectory.append((obs, mask, pi))
                _, reward, terminated, truncated, _ = env.step(action)
                ended = bool(terminated or truncated)
                following = extract_observation(base.state, base.my_turn, version)
                next_mask = (
                    np.zeros_like(mask)
                    if ended
                    else base.get_legal_actions_mask().astype(np.float32)
                )
                buffer.add(obs, action, float(reward), following, next_mask, ended)
                if ended:
                    episodes += 1
                    if search:
                        z = outcome(base.game_rule, base.my_turn)
                        aux.extend((o, m, p, z) for o, m, p in trajectory)
                        trajectory.clear()
                    env.reset()
                if step >= config.warmup and len(buffer) >= config.batch_size:
                    stats = dqn_update(net, target, buffer, optimizer, params, step)
                    if search and aux and step % 4 == 0:
                        stats.update(auxiliary_update(net, aux, optimizer, aux_rng))
                if step % config.snapshot_every == 0:
                    population.add_snapshot(net)
                if step % 100 == 0 or step == config.steps:
                    elapsed = time.monotonic() - started
                    writer.writerow(
                        {
                            "step": step,
                            "episodes": episodes,
                            "epsilon": epsilon_at(step, params),
                            "buffer": len(buffer),
                            **{key: stats.get(key, 0.0) for key in fields[4:-1]},
                            "elapsed_seconds": elapsed,
                        }
                    )
                    handle.flush()
                    write_json(
                        folder / "status.json",
                        {
                            "status": "running",
                            "step": step,
                            "total_steps": config.steps,
                            "episodes": episodes,
                            "elapsed_seconds": elapsed,
                            "opponent_games": dict(population.counts),
                            **stats,
                        },
                    )
                if step % config.eval_every == 0 or step == config.steps:
                    validation = benchmark(
                        net,
                        "minimax",
                        list(
                            range(
                                VALIDATION_START,
                                VALIDATION_START + config.validation_deals,
                            )
                        ),
                    )
                    write_json(folder / f"validation-{step}.json", validation)
                    if validation["win_rate"] > best_validation:
                        best_validation = validation["win_rate"]
                        save_model(net, folder / "best.pth", step, metadata)
                    save_model(net, folder / f"step-{step}.pth", step, metadata)
                    print(
                        f"{variant} seed={seed} step={step} validation={validation['win_rate']:.3f} Q={stats.get('q_mean', 0):.2f}",
                        flush=True,
                    )
        save_model(net, folder / "final.pth", step, metadata)
        write_json(
            folder / "status.json",
            {
                "status": "completed",
                "step": step,
                "total_steps": config.steps,
                "episodes": episodes,
                "elapsed_seconds": time.monotonic() - started,
                "opponent_games": dict(population.counts),
            },
        )
    except BaseException as error:
        save_model(net, folder / "interrupted.pth", step, metadata)
        write_json(
            folder / "status.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "step": step,
                "error": repr(error),
            },
        )
        raise
    finally:
        env.close()
    return folder / "best.pth"


def main() -> None:  # noqa: C901 - CLI validation and staged orchestration
    """Train all declared jobs first, then unseal the common test set once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2024])
    parser.add_argument("--baseline", nargs="*", type=Path, default=[])
    for name, field in ExperimentConfig.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )
    args = parser.parse_args()
    config = ExperimentConfig(
        **{name: getattr(args, name) for name in ExperimentConfig.__dataclass_fields__}
    )
    if (
        min(
            config.steps,
            config.warmup,
            config.batch_size,
            config.buffer_size,
            config.eval_every,
            config.validation_deals,
            config.test_deals,
            config.snapshot_every,
            config.search_every,
            config.simulations,
        )
        < 1
    ):
        parser.error("budgets must be positive")
    if (
        config.warmup >= config.steps
        or config.validation_deals >= TEST_START - VALIDATION_START
    ):
        parser.error(
            "warmup must be shorter than training; validation/test ranges must not overlap"
        )
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(
        args.variants
    ):
        parser.error("duplicate seeds/variants")
    if any(seed >= VALIDATION_START for seed in args.seeds):
        parser.error("training seeds must be below reserved evaluation ranges")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "config": asdict(config),
            "variants": args.variants,
            "seeds": args.seeds,
            "baseline": [str(p) for p in args.baseline],
            "selection": "best greedy minimax validation win rate; ties keep earliest",
            "test_protocol": "same deal seeds in both seats; all training completes before test",
        },
    )
    jobs: list[tuple[str, Path, int]] = []
    for variant in args.variants:
        for seed in args.seeds:
            with isolated_rng(seed):
                path = train_variant(
                    args.output / f"{variant}-{seed}", variant, seed, config
                )
            jobs.append(
                (
                    f"{variant}-{seed}",
                    path,
                    config.simulations if variant == "search" else 0,
                )
            )
    jobs.extend((f"archive-{i}", path, 0) for i, path in enumerate(args.baseline))
    results: dict[str, Any] = {}
    for name, path, simulations in jobs:
        net = load_saved_dqn(path).to(config.device)
        results[name] = {}
        for opponent in ("random", "heuristic", "minimax"):
            report = benchmark(
                net, opponent, list(range(TEST_START, TEST_START + config.test_deals))
            )
            results[name][opponent] = report
            print(
                f"TEST {name} {opponent}: {report['wins']}/{report['games']}",
                flush=True,
            )
            write_json(args.output / "results.json", results)
        if simulations:
            results[name]["minimax_search"] = benchmark(
                net,
                "minimax",
                list(range(TEST_START, TEST_START + config.test_deals)),
                simulations,
            )
            write_json(args.output / "results.json", results)
    write_json(
        args.output / "suite-status.json", {"status": "completed", "jobs": len(jobs)}
    )


if __name__ == "__main__":
    main()
