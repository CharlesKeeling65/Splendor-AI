"""Executable evidence for legacy gaps that task-1 opt-in paths must avoid."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from splendor.agents.generic.random import RandomAgent
from splendor.agents.our_agents.policy_imitation.protocol import isolated_seed
from splendor.agents.our_agents.policy_imitation.stabilization import (
    BaselineSpec,
    StabilizationConfig,
    _make_evaluation_jobs,
    make_training_jobs,
)
from splendor.game import Game
from splendor.league import AgentSpec, GameRecord, SeatMetrics, aggregate
from splendor.splendor.splendor_model import SplendorGameRule
from splendor.splendor.utils import LimitRoundsGameRule


def _board_signature(rule: SplendorGameRule) -> tuple[object, ...]:
    board = rule.current_game_state.board
    return (
        tuple(code for code, _cost in board.nobles),
        tuple(tuple(card.code for card in tier) for tier in board.dealt),
        tuple(tuple(card.code for card in deck) for deck in board.decks),
    )


def test_legacy_game_seed_is_not_the_raw_rule_deal_seed() -> None:
    seed = 1234
    random.seed(seed)
    raw = LimitRoundsGameRule(2)
    game = Game(
        LimitRoundsGameRule,
        [RandomAgent(0), RandomAgent(1)],
        2,
        seed=seed,
        displayer=None,
    )

    assert _board_signature(raw) != _board_signature(game.game_rule)


def test_legacy_whole_game_seed_couples_torch_action_randomness() -> None:
    torch.manual_seed(999)
    with isolated_seed(77):
        first = torch.rand(4)
    _ = torch.rand(20)
    with isolated_seed(77):
        second = torch.rand(4)

    assert torch.equal(first, second)


def _config(tmp_path: Path) -> StabilizationConfig:
    return StabilizationConfig(
        output=tmp_path / "run",
        initial_bc=tmp_path / "bc.pth",
        device="cpu",
        workers=1,
        updates=1,
        games_per_update=2,
        validation_deals=1,
        test_deals=1,
        seeds=(42,),
        variants=("fixed", "scratch"),
        baselines=(BaselineSpec("historical", "ppo", str(tmp_path / "old.pth")),),
        smoke=True,
        seed_base=824000,
    )


def test_legacy_stabilization_treatments_use_different_deals(tmp_path: Path) -> None:
    jobs = make_training_jobs(_config(tmp_path))

    assert jobs[0].model_seed == jobs[1].model_seed == 42
    assert jobs[0].training_seeds != jobs[1].training_seeds


def test_legacy_final_test_automatically_includes_every_candidate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    trained: dict[str, object] = {}
    for name in ("fixed-seed42", "scratch-seed42"):
        checkpoint = tmp_path / f"{name}.pth"
        checkpoint.touch()
        trained[name] = {"result": {"best": str(checkpoint)}}
    (tmp_path / "old.pth").touch()

    jobs = _make_evaluation_jobs(
        config,
        output_dir=tmp_path,
        training_payloads=trained,  # type: ignore[arg-type]
        test_seeds=(823000,),
    )

    assert [job.candidate.name for job in jobs] == [
        "fixed-seed42",
        "scratch-seed42",
        "historical",
    ]


def test_legacy_league_excludes_error_rows_from_agent_denominators() -> None:
    metric = SeatMetrics(1, 0, 0, 1, 0, None, 1.0, 1, 0, 0, 1)
    completed = GameRecord(0, (0, 1), ("a", "b"), 1, [1.0, 0.0], [0], [metric, metric])
    failed = GameRecord(0, (0, 1), ("a", "b"), 2, [], [], [], "boom")

    report = aggregate(
        [AgentSpec("a", "a.module"), AgentSpec("b", "b.module")],
        [completed, failed],
        2,
    )

    assert report["error_games"] == 1
    assert report["agents"]["a"]["games"] == 1
    assert len([completed, failed]) == 2
