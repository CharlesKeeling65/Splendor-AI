"""Smoke tests for the checkpoint league entry modules (roadmap C4)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from splendor.splendor.splendor_model import SplendorGameRule


def _env(monkeypatch: pytest.MonkeyPatch, name: str, missing: bool) -> None:
    if missing:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, "/tmp/definitely-not-a-checkpoint.pth")


@pytest.mark.parametrize(
    ("module", "env"),
    [
        (
            "splendor.agents.our_agents.league_entries.imitation_ppo",
            "SPLENDOR_PPO_CHECKPOINT",
        ),
        ("splendor.agents.our_agents.league_entries.dqn", "SPLENDOR_DQN_CHECKPOINT"),
    ],
)
def test_entries_require_explicit_checkpoint(
    monkeypatch: pytest.MonkeyPatch, module: str, env: str
) -> None:
    import importlib

    _env(monkeypatch, env, missing=True)
    entry = importlib.import_module(module)
    random.seed(0)
    with pytest.raises(ValueError, match=env):
        entry.myAgent(0)


def test_dqn_entry_plays_greedy_moves(tmp_path: Path) -> None:
    import torch

    from splendor.agents.our_agents.dqn.network import QNetwork
    from splendor.agents.our_agents.dqn.utils import save_model

    net = QNetwork(input_dim=265, output_dim=3510, feature_version="v1")
    path = tmp_path / "dqn.pth"
    save_model(net, path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("SPLENDOR_DQN_CHECKPOINT", str(path))
    try:
        from splendor.agents.our_agents.league_entries import dqn as entry

        agent = entry.myAgent(0)
        rule = SplendorGameRule(2)
        random.seed(825_700)
        for _ in range(5):
            state = rule.current_game_state
            turn = rule.getCurrentAgentIndex()
            legal = rule.getLegalActions(state, turn)
            action = agent.SelectAction(legal, state, rule)
            assert action in legal
            rule.update(action)
            if rule.gameEnds():
                break
        del torch
    finally:
        monkeypatch.undo()
