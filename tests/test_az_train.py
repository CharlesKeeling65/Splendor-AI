"""Z2 tests: replay/training step sanity and the full-CLI smoke run."""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from splendor.agents.our_agents.alphazero.selfplay import (
    TrainingSample,
    build_az_network,
)
from splendor.agents.our_agents.alphazero.train import (
    AZConfig,
    _selfplay_worker,
    _train_batches,
    main,
)


def _synthetic_buffer(n: int) -> list[TrainingSample]:
    rng = np.random.default_rng(3)
    samples: list[TrainingSample] = []
    for i in range(n):
        indices = sorted(rng.choice(3510, size=25, replace=False).tolist())
        raw = rng.random(25).astype(np.float32)
        samples.append(
            TrainingSample(
                seat=i % 2,
                observation=rng.normal(0, 2, 312).astype(np.float32),
                indices=indices,
                target=raw / raw.sum(),
                z=float(rng.choice([-1.0, 1.0])),
            )
        )
    return samples


def _tiny_config() -> AZConfig:
    return AZConfig(
        iterations=1,
        games_per_iter=2,
        epochs_per_iter=1,
        batch_size=8,
        eval_deals=1,
    )


def test_selfplay_worker_chunk():
    net = build_az_network()
    payload = (
        {
            "simulations": 4,
            "n_trees": 2,
            "max_depth": 12,
            "c_puct": 1.5,
            "dirichlet_alpha": 1.0,
            "dirichlet_epsilon": 0.25,
            "temperature_moves": 2,
        },
        {k: v.cpu() for k, v in net.state_dict().items()},
        [1_040_002, 1_040_003],
    )
    payload_out = _selfplay_worker(payload)
    assert len(payload_out["z"]) >= 20


def test_train_batches_loss_is_finite_and_repeatable():
    torch.manual_seed(0)
    net = build_az_network()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    buffer = deque(_synthetic_buffer(64))
    first = _train_batches(net, buffer, _tiny_config(), rng, optimizer, "cpu")
    assert all(np.isfinite(value) for value in first.values())
    second = _train_batches(net, buffer, _tiny_config(), rng, optimizer, "cpu")
    assert np.isfinite(second["loss"])


def test_train_cli_smoke(tmp_path: Path):
    exit_code = main(
        [
            "--output",
            str(tmp_path / "z2-smoke"),
            "--iterations",
            "1",
            "--games-per-iter",
            "2",
            "--workers",
            "2",
            "--simulations",
            "4",
            "--trees",
            "2",
            "--batch-size",
            "8",
            "--eval-deals",
            "1",
            "--device",
            "cpu",
            "--scratch",
            "--seed",
            "830_300",
        ]
    )
    assert exit_code == 0
    out = tmp_path / "z2-smoke"
    assert (out / "manifest.json").exists()
    assert (out / "iter_000.json").exists()
    assert (out / "iter_000.pth").exists()
    record = json.loads((out / "iter_000.json").read_text())
    assert record["games"] == 2
    assert record["greedy_vs_random"]["games"] == 2.0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["bc_checkpoint"] is None  # --scratch honoured
