"""AZ-lite iteration driver (roadmap Z2): self-play -> train -> evaluate.

Each iteration:
1. collects ``games_per_iter`` self-play games in a process pool (workers load
   the current network once per task; engine deals are seeded from the
   ``z_training`` registry segment, search streams derived per game);
2. trains the policy/value network on the replay buffer with masked
   cross-entropy against the root visit distributions plus MSE against the
   terminal outcome ``z``;
3. evaluates the *greedy prior* (no search) against random and minimax on
   ``z_validation`` deals, both seat assignments;
4. saves a checkpoint and a per-iteration JSON log.

Gate semantics: the Z2 gate (vs minimax ~57%) is judged on the league CLI
with search enabled (``league_entries/az_search.py`` +
``AZ_SEARCH_CHECKPOINT``); the in-loop greedy numbers are the cheap
training-curve signal, reported separately and never used as the gate.

Usage::

    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python -m splendor.agents.our_agents.alphazero.train \
        --output runs/z2-az-lite --iterations 10 --games-per-iter 1000 \
        --workers 24 --bc-checkpoint runs/policy-imitation/formal-3.3-20260907/bc-dagger-2/best.pth
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.alphazero.selfplay import (
    OBS_DIM,
    SelfPlayConfig,
    TrainingSample,
    build_az_network,
    evaluate_greedy,
    pack_samples,
    play_game,
    unpack_samples,
    warm_start_from_bc,
)
from splendor.agents.our_agents.dqn.network import ACTION_DIM, QNetwork
from splendor.agents.our_agents.dqn.utils import save_model
from splendor.agents.our_agents.minmax import myAgent as MiniMaxAgent
from splendor.seed_registry import Z_TRAINING, Z_VALIDATION
from splendor.template import Agent

DEFAULT_BC = "runs/policy-imitation/formal-3.3-20260907/bc-dagger-2/best.pth"

#: Frozen network geometry; workers rebuild identical networks from it.
NET_CONFIG: dict[str, Any] = {
    "input_dim": OBS_DIM,
    "feature_version": "public-v2",
    "auxiliary_heads": True,
    "use_input_norm": True,
    "dueling": False,
    "hidden_layers": (128, 128, 128, 128),
}


def _random_factory(agent_id: int) -> Agent:
    return RandomAgent(agent_id)


def _minimax_factory(agent_id: int) -> Agent:
    return MiniMaxAgent(agent_id)


@dataclass
class AZConfig:
    """Full training-run configuration; stored into the manifest."""

    iterations: int = 10
    games_per_iter: int = 1000
    workers: int = 24
    simulations: int = 100
    n_trees: int = 4
    max_depth: int = 24
    temperature_moves: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iter: int = 2
    buffer_cap: int = 300_000
    eval_deals: int = 10
    device: str = "cuda"
    seed: int = 2026_0913
    bc_checkpoint: str | None = DEFAULT_BC
    seed_base: int = Z_TRAINING.start
    output: Path = field(default=Path("runs/z2-az-lite"))

    def selfplay(self) -> SelfPlayConfig:
        return SelfPlayConfig(
            simulations=self.simulations,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            temperature_moves=self.temperature_moves,
        )


def _selfplay_worker(
    payload: tuple[dict[str, Any], dict[str, torch.Tensor], list[int]],
) -> dict[str, Any]:
    """Process-pool entry: play one chunk of seeded games."""
    config_dict, state_dict, seeds = payload
    torch.set_num_threads(1)
    net = QNetwork(**NET_CONFIG)
    net.load_state_dict(state_dict)
    config = SelfPlayConfig(**config_dict)
    samples: list[TrainingSample] = []
    for seed in seeds:
        samples.extend(play_game(net, seed, config))
    return pack_samples(samples)


def _collect(
    executor: ProcessPoolExecutor,
    net: QNetwork,
    config: AZConfig,
    seeds: list[int],
) -> list[TrainingSample]:
    state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    chunk_size = max(1, len(seeds) // config.workers)
    chunks = [
        seeds[i : i + chunk_size] for i in range(0, len(seeds), chunk_size)
    ]
    payloads = [(asdict(config.selfplay()), state_dict, chunk) for chunk in chunks]
    samples: list[TrainingSample] = []
    for payload in executor.map(_selfplay_worker, payloads):
        samples.extend(unpack_samples(payload))
    return samples


def _train_batches(  # noqa: PLR0913, PLR0917 - one argument per concern
    net: QNetwork,
    buffer: deque[TrainingSample],
    config: AZConfig,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    """Run the iteration's epochs; return mean loss components."""
    net.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "steps": 0.0,
    }
    snapshot = list(buffer)  # deque indexing is O(n); snapshot once per call
    for _epoch in range(config.epochs_per_iter):
        order = rng.permutation(len(snapshot))
        for start in range(0, len(order) - config.batch_size + 1, config.batch_size):
            batch = [snapshot[i] for i in order[start : start + config.batch_size]]
            metrics = _train_step(net, batch, optimizer, device)
            for key in ("loss", "policy", "value"):
                totals[key] += metrics[key]
            totals["steps"] += 1.0
    net.eval()
    steps = max(totals["steps"], 1)
    return {key: totals[key] / steps for key in ("loss", "policy", "value")}


def _train_step(
    net: QNetwork,
    batch: list[TrainingSample],
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    observations = torch.from_numpy(
        np.stack([sample.observation for sample in batch])
    ).to(device)
    dense_target = torch.zeros(len(batch), ACTION_DIM, device=device)
    legal_mask = torch.zeros(len(batch), ACTION_DIM, device=device)
    for i, sample in enumerate(batch):
        dense_target[i, sample.indices] = torch.from_numpy(sample.target).to(device)
        legal_mask[i, sample.indices] = 1.0
    z = torch.tensor([sample.z for sample in batch], dtype=torch.float32, device=device)

    logits, value = net.policy_value(observations, legal_mask)
    log_probs = logits.log_softmax(dim=-1)
    policy_loss = -(dense_target * log_probs).sum(dim=-1).mean()
    value_loss = torch.nn.functional.mse_loss(value, z)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy": float(policy_loss.item()),
        "value": float(value_loss.item()),
    }


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - CLI driver
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/z2-az-lite"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--trees", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs-per-iter", type=int, default=2)
    parser.add_argument("--eval-deals", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026_0913)
    parser.add_argument("--bc-checkpoint", default=DEFAULT_BC)
    parser.add_argument(
        "--init-from",
        default=None,
        help="continue from a dqn-format QNetwork checkpoint (overrides BC warm start)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=Z_TRAINING.start,
        help="first self-play deal seed; continue past run-1 with 1040000+consumed",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="skip the BC warm start (from-scratch AZ ablation)",
    )
    args = parser.parse_args(argv)

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output dir: {args.output}")
    config = AZConfig(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        workers=args.workers,
        simulations=args.simulations,
        n_trees=args.trees,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs_per_iter=args.epochs_per_iter,
        eval_deals=args.eval_deals,
        device=args.device,
        seed=args.seed,
        bc_checkpoint=None if args.scratch else args.bc_checkpoint,
        seed_base=args.seed_base,
        output=args.output,
    )
    args.output.mkdir(parents=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    net = build_az_network()
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.normalization_frozen = checkpoint["config"].get(
            "normalization_frozen", False
        )
    elif config.bc_checkpoint:
        warm_start_from_bc(net, config.bc_checkpoint)
    train_net = net.to(config.device)
    optimizer = torch.optim.Adam(
        train_net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    buffer: deque[TrainingSample] = deque(maxlen=config.buffer_cap)
    rng = np.random.default_rng(config.seed)
    val_seeds = list(
        range(Z_VALIDATION.start, Z_VALIDATION.start + config.eval_deals)
    )
    spawn = multiprocessing.get_context("spawn")
    manifest: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "net_config": NET_CONFIG,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "seed_segment": {
            "training": Z_TRAINING.name,
            "validation": Z_VALIDATION.name,
            "seed_base": config.seed_base,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=spawn) as executor:
        for iteration in range(config.iterations):
            iter_start = time.perf_counter()
            seeds = [
                config.seed_base + iteration * config.games_per_iter + offset
                for offset in range(config.games_per_iter)
            ]
            if seeds[-1] >= Z_TRAINING.end:
                raise SystemExit("z_training segment exhausted; amend the registry")
            samples = _collect(executor, train_net, config, seeds)
            buffer.extend(samples)
            losses = (
                _train_batches(train_net, buffer, config, rng, optimizer, config.device)
                if len(buffer) >= config.batch_size
                else {"loss": float("nan"), "policy": float("nan"), "value": float("nan")}
            )
            # Evaluation always uses the CPU-side weights of the same net.
            eval_net = build_az_network()
            eval_net.load_state_dict(
                {k: v.detach().cpu() for k, v in train_net.state_dict().items()}
            )
            eval_net.normalization_frozen = True
            vs_random = evaluate_greedy(eval_net, _random_factory, val_seeds)
            vs_minimax = evaluate_greedy(eval_net, _minimax_factory, val_seeds)
            record = {
                "iteration": iteration,
                "games": config.games_per_iter,
                "moves": len(samples),
                "buffer": len(buffer),
                "elapsed_s": time.perf_counter() - iter_start,
                "losses": losses,
                "greedy_vs_random": vs_random,
                "greedy_vs_minimax": vs_minimax,
            }
            (args.output / f"iter_{iteration:03d}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            save_model(
                eval_net,
                args.output / f"iter_{iteration:03d}.pth",
                step=iteration,
                config={"source": "az-selfplay"},
            )
            print(
                f"[iter {iteration:03d}] moves={len(samples)} "
                f"loss={losses['loss']:.4f} "
                f"vs_random={vs_random['win_rate']:.2f} "
                f"vs_minimax={vs_minimax['win_rate']:.2f} "
                f"({record['elapsed_s']:.0f}s)",
                flush=True,
            )
    manifest["total_elapsed_s"] = time.perf_counter() - started
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"AZ training complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
