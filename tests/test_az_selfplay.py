"""Z2 tests: self-play sampling, warm start and evaluation plumbing."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.alphazero.selfplay import (
    SelfPlayConfig,
    TrainingSample,
    build_az_network,
    evaluate_greedy,
    pack_samples,
    play_game,
    unpack_samples,
    warm_start_from_bc,
)

BC_CHECKPOINT = Path("runs/policy-imitation/formal-3.3-20260907/bc-dagger-2/best.pth")

TINY_CONFIG = SelfPlayConfig(
    simulations=4, n_trees=2, max_depth=12, temperature_moves=2
)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    samples = [
        TrainingSample(
            seat=i % 2,
            observation=rng.random(312, dtype=np.float32),
            indices=sorted(rng.choice(3510, size=20, replace=False).tolist()),
            target=(lambda v: v / v.sum())(rng.random(20).astype(np.float32)),
            z=float(i % 3 - 1),
        )
        for i in range(7)
    ]
    payload = pack_samples(samples)
    restored = unpack_samples(payload)
    assert len(restored) == len(samples)
    for original, back in zip(samples, restored, strict=True):
        assert back.seat == original.seat
        assert back.indices == original.indices
        np.testing.assert_allclose(back.target, original.target, rtol=1e-6)
        assert back.z == pytest.approx(original.z)
        np.testing.assert_allclose(back.observation, original.observation)


def test_play_game_produces_valid_samples():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    net = build_az_network()
    samples = play_game(net, seed=1_040_001, config=TINY_CONFIG)
    assert len(samples) >= 10  # a real game, not a single move
    for sample in samples:
        assert sample.observation.shape == (312,)
        assert sample.z in (-1.0, 0.0, 1.0)
        assert len(sample.indices) == len(sample.target)
        assert sample.target.sum() == pytest.approx(1.0, abs=1e-5)
    # Both seats appear (alternating turns).
    assert {sample.seat for sample in samples} == {0, 1}
    # Deterministic under the same seed and the same network weights.
    replay = play_game(net, seed=1_040_001, config=TINY_CONFIG)
    assert [s.indices for s in replay] == [s.indices for s in samples]
    assert all(
        np.allclose(a.target, b.target) for a, b in zip(samples, replay, strict=True)
    )


def test_greedy_agent_beats_never_crashes_and_reports():
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    net = build_az_network()
    from splendor.agents.generic.random import myAgent as RandomAgent

    report = evaluate_greedy(net, RandomAgent, seeds=[1_140_001, 1_140_002])
    assert report["games"] == 4.0
    assert 0.0 <= report["win_rate"] <= 1.0
    assert report["ties"] >= 0.0


def test_warm_start_copies_bc_weights():
    if not BC_CHECKPOINT.exists():
        pytest.skip("BC checkpoint not present on this machine")
    net = build_az_network()
    warm_start_from_bc(net, str(BC_CHECKPOINT))
    checkpoint = torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=False)
    bc_state = checkpoint["model_state_dict"]
    own = net.state_dict()
    torch.testing.assert_close(own["net.0.weight"], bc_state["trunk.0.weight"])
    torch.testing.assert_close(own["policy_head.weight"], bc_state["policy_head.weight"])
    torch.testing.assert_close(
        own["input_norm.running_mean"], bc_state["normalizer.mean"].reshape(1, -1)
    )
    assert net.normalization_frozen is True


def test_warm_start_rejects_foreign_checkpoint(tmp_path: Path):
    bad = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {"unrelated": torch.zeros(1)}}, bad)
    with pytest.raises(ValueError, match="lacks expected keys"):
        warm_start_from_bc(build_az_network(), str(bad))


def test_az_network_forward_shapes():
    net = build_az_network()
    logits, value = net.policy_value(
        torch.zeros(4, 312), torch.ones(4, 3510)
    )
    assert logits.shape == (4, 3510)
    assert value.shape == (4,)
    assert float(value.abs().max()) <= 1.0  # tanh-bounded outcome head
