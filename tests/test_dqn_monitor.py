"""Tests for live DQN artifacts and checkpoint device preservation."""

import json
from pathlib import Path

import pytest
import torch

from splendor.agents.our_agents.dqn.monitor import load_payload
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import save_model


def test_load_payload_reads_live_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run__dqn"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"total_steps": 100, "device": "cpu"}), encoding="utf-8"
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "step": 12}), encoding="utf-8"
    )
    (run_dir / "progress.csv").write_text(
        "timestamp,event,step,episode,epsilon\n"
        "2026-01-01T00:00:00+08:00,start,0,0,1.0\n"
        "2026-01-01T00:00:01+08:00,episode,12,1,0.9\n",
        encoding="utf-8",
    )

    payload = load_payload(run_dir)

    assert payload["status"]["status"] == "running"
    assert payload["status"]["step"] == 12
    assert payload["config"]["total_steps"] == 100
    assert payload["progress"][-1]["event"] == "episode"
    assert payload["progress"][-1]["step"] == 12


def test_load_payload_preserves_interrupted_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run__dqn"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "interrupted", "step": 42}), encoding="utf-8"
    )

    payload = load_payload(run_dir)

    assert payload["status"] == {"status": "interrupted", "step": 42}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_save_model_keeps_cuda_network_on_cuda(tmp_path: Path) -> None:
    model = QNetwork(hidden_layers=(8, 8)).to("cuda")
    save_model(model, tmp_path / "model.pth", step=1)

    assert next(model.parameters()).device.type == "cuda"
