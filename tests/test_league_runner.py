"""Tests for the league round-robin evaluator (roadmap A1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from splendor.league import (
    AgentSpec,
    GameTask,
    PairStats,
    aggregate,
    build_tasks,
    format_report,
    parse_agent_list,
    play_task_safe,
    run_league,
    wilson_interval,
)
from splendor.seed_registry import (
    CI_SMOKE,
    INDEPENDENT_TEST,
    SeedRegistryError,
    allocate_seeds,
    forbid_reserved,
    resolve_segment,
)

RANDOM_AGENT = "splendor.agents.generic.random"


def _random_specs() -> list[AgentSpec]:
    return parse_agent_list(f"a={RANDOM_AGENT},b={RANDOM_AGENT}")


def test_parse_agent_list_supports_names_and_bare_modules() -> None:
    specs = parse_agent_list("ppo=splendor.agents.our_agents.ppo.ppo_agent,random")
    assert specs[0].name == "ppo"
    assert specs[1].name == "random"
    with pytest.raises(SeedRegistryError):
        parse_agent_list("only=one.module")
    with pytest.raises(SeedRegistryError):
        parse_agent_list(f"x={RANDOM_AGENT},x={RANDOM_AGENT}")


def test_wilson_interval_basics() -> None:
    assert wilson_interval(0, 10) == (0.0, 0.0) or wilson_interval(0, 10)[0] == 0.0
    low, high = wilson_interval(6, 10)
    assert 0.0 <= low <= 6 / 10 <= high <= 1.0
    low_half, high_half = wilson_interval(7.5, 15)  # ties split as 0.5 successes
    assert 0.0 <= low_half <= 0.5 <= high_half <= 1.0
    with_more_games = wilson_interval(60, 100)
    assert with_more_games[1] - with_more_games[0] < high - low


def test_pair_stats_records_ties_and_losses() -> None:
    stats = PairStats()
    stats.record([0], 0)
    stats.record([1], 0)
    stats.record([0, 1], 0)
    assert (stats.wins, stats.losses, stats.ties, stats.games) == (1, 1, 1, 3)
    assert stats.score_rate == pytest.approx((1 + 0.5) / 3)


def test_build_tasks_enumerates_both_seat_orders() -> None:
    specs = _random_specs()
    seeds = allocate_seeds(CI_SMOKE, 4)
    tasks = build_tasks(specs, 2, 2, seeds)
    assignments = {task.seat_assignment for task in tasks}
    assert assignments == {(0, 1), (1, 0)}
    assert [task.seed for task in tasks] == seeds
    with pytest.raises(SeedRegistryError):
        build_tasks(specs, 2, 3, seeds)  # 6 games needed, only 4 seeds


def test_seed_registry_guards() -> None:
    assert resolve_segment("independent_test") is INDEPENDENT_TEST
    with pytest.raises(SeedRegistryError):
        allocate_seeds(INDEPENDENT_TEST, 51)  # sealed 50-seed segment
    wrapped = allocate_seeds(INDEPENDENT_TEST, 120, wrap=True)
    assert wrapped[0] == INDEPENDENT_TEST.start and len(wrapped) == 120
    with pytest.raises(SeedRegistryError):
        forbid_reserved([810_000])
    forbid_reserved([826_000])  # training segment start is fine


def test_play_task_smoke_random_vs_random() -> None:
    specs = _random_specs()
    seeds = allocate_seeds(CI_SMOKE, 2)
    tasks = build_tasks(specs, 2, 1, seeds)
    record = play_task_safe(tasks[0])
    assert record.error is None
    assert len(record.scores) == 2
    assert record.winner_seats, "every finished game has at least one winner"
    for seat, metrics in enumerate(record.metrics):
        assert metrics.turns >= 1
        assert metrics.raw_score >= 0
        assert metrics.final_score >= metrics.raw_score  # tie-break can add 0.5
        assert metrics.cards + metrics.nobles >= 0
        assert seat in (0, 1)


def test_paired_seed_reproducibility() -> None:
    specs = _random_specs()
    seeds = allocate_seeds(CI_SMOKE, 4)
    tasks = build_tasks(specs, 2, 2, seeds)
    first = [play_task_safe(task) for task in tasks]
    second = [play_task_safe(task) for task in tasks]
    for left, right in zip(first, second, strict=True):
        assert left.scores == right.scores
        assert left.winner_seats == right.winner_seats
        assert left.metrics == right.metrics


def test_play_task_safe_records_agent_load_failure() -> None:
    task = GameTask(
        matchup_index=0,
        seat_assignment=(0, 1),
        names=("x", "y"),
        modules=("splendor.template", RANDOM_AGENT),  # no myAgent in template
        seed=CI_SMOKE.start,
        time_limit=1.0,
        warning_limit=3,
    )
    record = play_task_safe(task)
    assert record.error is not None
    assert "myAgent" in record.error


def test_aggregate_and_report_smoke() -> None:
    specs = _random_specs()
    seeds = allocate_seeds(CI_SMOKE, 4)
    tasks = build_tasks(specs, 2, 2, seeds)
    records = [play_task_safe(task) for task in tasks]
    results = aggregate(specs, records, 2)
    assert results["error_games"] == 0
    for name in ("a", "b"):
        row = results["agents"][name]
        assert row["games"] == 4
        assert 0.0 <= row["score_rate"] <= 1.0
    assert set(results["pairs"]) == {"a vs b", "b vs a"}
    report = format_report(
        specs,
        2,
        {
            "created_utc": "t",
            "games_per_matchup": 2,
            "segment": CI_SMOKE.name,
            "seed_start": CI_SMOKE.start,
            "seed_count": 4,
            "git_revision": "x",
            "python_hash_seed": "0",
        },
        results,
    )
    assert "## Win-rate matrix" in report
    assert "| a |" in report


def test_run_league_three_agents_names_per_seat(tmp_path: Path) -> None:
    """Regression: rosters larger than the seat count must not crash.

    The matchup names are per-seat resolved (``record.names`` already is);
    re-indexing it with seat_assignment (roster indices) double-resolves and
    raises IndexError as soon as a roster index exceeds the seat range - the
    failure mode that hit the Z1 run with a 4-agent roster.
    """
    specs = parse_agent_list(f"a={RANDOM_AGENT},b={RANDOM_AGENT},c={RANDOM_AGENT}")
    results = run_league(specs, 2, 1, segment=CI_SMOKE, output_dir=tmp_path / "lg")
    assert len(results["matchups"]) == 6  # 3 agents x 2 seat orders
    for matchup in results["matchups"]:
        assert len(matchup["names"]) == 2
        for seat, roster_index in enumerate(matchup["seat_assignment"]):
            assert matchup["names"][seat] == specs[roster_index].name
