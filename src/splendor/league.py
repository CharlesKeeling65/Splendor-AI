"""League round-robin evaluator (roadmap 2026-09-12 §A1).

Runs every ordered seat assignment of the supplied agents (for two seats this
is the classic double-seat-balanced round robin), records per-game outcome and
behaviour metrics, and emits a JSON manifest plus a Markdown win-rate matrix
with Wilson intervals.

Determinism contract
--------------------
* Game seeds are allocated from a declared :mod:`splendor.seed_registry`
  segment and stamped into the manifest.
* The engine only consumes Python's global ``random``, which
  :class:`splendor.game.Game` re-seeds per game, so identical seeds reproduce
  identical games regardless of process pool layout.
* Runners should be launched with ``PYTHONHASHSEED=0`` (see the risk register
  in docs/IMPROVEMENT_ROADMAP_20260912.md); a mismatch is recorded in the
  manifest rather than silently ignored.

Usage::

    splendor-league -a splendor.agents.generic.random,\\
splendor.agents.our_agents.minmax -n 2 -m 10 --segment validation
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import itertools
import json
import math
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import git

from splendor.game import Game
from splendor.seed_registry import (
    VALIDATION,
    SeedRegistryError,
    SeedSegment,
    allocate_seeds,
    resolve_segment,
)
from splendor.splendor.splendor_model import SplendorState
from splendor.splendor.utils import LimitRoundsGameRule
from splendor.template import Agent

#: The engine supports 2-4 seats (observation slots assume MAX_RIVALS = 3).
MIN_SEATS = 2
MAX_SEATS = 4
#: Seat count for which the pairwise win-rate matrix is well defined.
PAIRWISE_SEATS = 2
#: Score a single agent needs to trigger the end of game.
WINNING_SCORE = 15

SCHEMA_VERSION = "splendor-league/1"


@dataclass(frozen=True)
class AgentSpec:
    """One league entrant: a display name plus its importable module."""

    name: str
    module: str


@dataclass(frozen=True)
class GameTask:
    """A fully determined game: seat assignment, seed and runner knobs."""

    matchup_index: int
    seat_assignment: tuple[int, ...]
    names: tuple[str, ...]
    modules: tuple[str, ...]
    seed: int
    time_limit: float
    warning_limit: int


@dataclass
class SeatMetrics:
    """Behaviour metrics for one seat in one finished game."""

    turns: int
    buys: int
    reserves: int
    collects: int
    passes: int
    rounds_to_15: int | None
    final_score: float
    raw_score: int
    cards: int
    nobles: int
    gems: int


@dataclass
class GameRecord:
    """Outcome of a single league game."""

    matchup_index: int
    seat_assignment: tuple[int, ...]
    names: tuple[str, ...]
    seed: int
    scores: list[float]
    winner_seats: list[int]
    metrics: list[SeatMetrics]
    error: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class PairStats:
    """Win/tie/loss aggregation from one agent's perspective."""

    games: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0

    def record(self, winner_seats: list[int], my_seat: int) -> None:
        self.games += 1
        if len(winner_seats) > 1:
            self.ties += 1
        elif my_seat in winner_seats:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def score_rate(self) -> float:
        """Points per game with ties split (1 win / 0.5 tie)."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.games


def parse_agent_list(raw: str) -> list[AgentSpec]:
    """Parse ``name=module`` pairs; bare modules take their tail as name.

    Repeating the same bare module is legitimate (e.g. ``random,random``);
    duplicate names get a ``_k`` suffix, mirroring the base runner's
    ``random0,random1`` convention.  Explicitly given names must stay unique.
    """
    specs: list[AgentSpec] = []
    explicit_names: set[str] = set()
    for raw_chunk in raw.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, module = chunk.split("=", 1)
            name = name.strip()
            if name in explicit_names:
                raise SeedRegistryError(f"duplicate agent name {name!r}")
            explicit_names.add(name)
            specs.append(AgentSpec(name, module.strip()))
        else:
            specs.append(AgentSpec(chunk.rsplit(".", 1)[-1], chunk))
    if len(specs) < MIN_SEATS:
        raise SeedRegistryError(
            f"at least {MIN_SEATS} agents are required, got {len(specs)}"
        )
    seen: dict[str, int] = {}
    deduped: list[AgentSpec] = []
    for spec in specs:
        count = seen.get(spec.name, 0)
        seen[spec.name] = count + 1
        deduped.append(
            spec if count == 0 else AgentSpec(f"{spec.name}_{count + 1}", spec.module)
        )
    return deduped


def load_agent_class(module_name: str) -> type[Agent]:
    """Import ``module`` and return its ``myAgent`` (repo loading convention)."""
    module = importlib.import_module(module_name)
    agent_class = getattr(module, "myAgent", None)
    if agent_class is None:
        raise ImportError(
            f"module {module_name!r} does not expose myAgent "
            "(see AGENTS.md agent interface convention)"
        )
    return agent_class  # type: ignore[no-any-return]


def seat_metrics(state: SplendorState, seat: int, final_score: float) -> SeatMetrics:
    """Derive behaviour metrics from the engine trace and final state."""
    agent = state.agents[seat]
    trace = agent.agent_trace.action_reward
    buys = reserves = collects = passes = 0
    rounds_to_15: int | None = None
    cumulative = 0
    for turn, (action, score_delta) in enumerate(trace, start=1):
        action_type = action["type"]
        if "buy" in action_type:
            buys += 1
        elif action_type == "reserve":
            reserves += 1
        elif "collect" in action_type:
            collects += 1
        elif action_type == "pass":
            passes += 1
        cumulative += score_delta
        if rounds_to_15 is None and cumulative >= WINNING_SCORE:
            rounds_to_15 = turn
    cards = sum(
        len(stack) for colour, stack in agent.cards.items() if colour != "yellow"
    )
    return SeatMetrics(
        turns=len(trace),
        buys=buys,
        reserves=reserves,
        collects=collects,
        passes=passes,
        rounds_to_15=rounds_to_15,
        final_score=final_score,
        raw_score=int(agent.score),
        cards=cards,
        nobles=len(agent.nobles),
        gems=int(sum(agent.gems.values())),
    )


def play_task(task: GameTask) -> GameRecord:
    """Play one league game in a fresh process-local state."""
    agent_classes = [load_agent_class(module) for module in task.modules]
    instances = [
        agent_classes[agent_index](seat)
        for seat, agent_index in enumerate(task.seat_assignment)
    ]
    rule = LimitRoundsGameRule
    game = Game(
        rule,
        instances,
        len(task.seat_assignment),
        seed=task.seed,
        time_limit=task.time_limit,
        warning_limit=task.warning_limit,
        displayer=None,
        agents_namelist=list(task.names),
    )
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        history = game.Run()
    final_state = game.game_rule.current_game_state
    scores = [float(history["scores"][seat]) for seat in range(len(instances))]
    top = max(scores)
    winners = [seat for seat, score in enumerate(scores) if score == top]
    metrics = [
        seat_metrics(final_state, seat, scores[seat]) for seat in range(len(instances))
    ]
    return GameRecord(
        matchup_index=task.matchup_index,
        seat_assignment=task.seat_assignment,
        names=task.names,
        seed=task.seed,
        scores=scores,
        winner_seats=winners,
        metrics=metrics,
    )


def play_task_safe(task: GameTask) -> GameRecord:
    """Like :func:`play_task` but converts crashes into error records.

    The offender is attributed from ``current_agent_index`` at raise time; the
    game is excluded from win-rate denominators by the aggregator.
    """
    try:
        return play_task(task)
    except Exception as exc:  # league must survive one bad matchup
        return GameRecord(
            matchup_index=task.matchup_index,
            seat_assignment=task.seat_assignment,
            names=task.names,
            seed=task.seed,
            scores=[],
            winner_seats=[],
            metrics=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def build_tasks(  # noqa: PLR0913 - task fields are one flat configuration
    specs: Sequence[AgentSpec],
    seats: int,
    games_per_matchup: int,
    seeds: Sequence[int],
    *,
    time_limit: float = 1.0,
    warning_limit: int = 3,
) -> list[GameTask]:
    """Enumerate ordered seat assignments and pin seeds to games."""
    if seats < MIN_SEATS or seats > MAX_SEATS:
        raise SeedRegistryError(
            f"seats must be in [{MIN_SEATS}, {MAX_SEATS}], got {seats}"
        )
    if len(specs) < seats:
        raise SeedRegistryError(
            f"{seats} seats need at least {seats} agents, got {len(specs)}"
        )
    matchups = list(itertools.permutations(range(len(specs)), seats))
    needed = len(matchups) * games_per_matchup
    if len(seeds) < needed:
        raise SeedRegistryError(
            f"{len(matchups)} matchups x {games_per_matchup} games = {needed} "
            f"seeds required, only {len(seeds)} allocated"
        )
    modules = tuple(spec.module for spec in specs)
    names = tuple(spec.name for spec in specs)
    tasks: list[GameTask] = []
    cursor = 0
    for matchup_index, assignment in enumerate(matchups):
        for _ in range(games_per_matchup):
            tasks.append(
                GameTask(
                    matchup_index=matchup_index,
                    seat_assignment=assignment,
                    names=names,
                    modules=modules,
                    seed=seeds[cursor],
                    time_limit=time_limit,
                    warning_limit=warning_limit,
                )
            )
            cursor += 1
    return tasks


def wilson_interval(
    successes: float, total: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; bounds clipped to [0, 1]."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    spread = (
        z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass
class AgentAggregate:
    """Per-agent totals across every game it played."""

    name: str
    games: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    rank_sum: int = 0
    score_sum: float = 0.0
    behaviour: dict[str, float] = field(default_factory=dict)
    behaviour_counts: dict[str, int] = field(default_factory=dict)

    def add_game(self, seat: int, record: GameRecord, seats: int) -> None:
        self.games += 1
        if len(record.winner_seats) > 1:
            self.ties += 1
        elif seat in record.winner_seats:
            self.wins += 1
        else:
            self.losses += 1
        ranking = sorted(
            range(seats),
            key=lambda s: (-record.scores[s], record.metrics[s].cards),
        )
        self.rank_sum += ranking.index(seat) + 1
        self.score_sum += record.scores[seat]
        metric = record.metrics[seat]
        samples: dict[str, float] = {
            "turns": float(metric.turns),
            "buys": float(metric.buys),
            "reserves": float(metric.reserves),
            "collects": float(metric.collects),
            "cards": float(metric.cards),
            "nobles": float(metric.nobles),
        }
        if metric.rounds_to_15 is not None:
            samples["rounds_to_15"] = float(metric.rounds_to_15)
        for key, value in samples.items():
            self.behaviour[key] = self.behaviour.get(key, 0.0) + value
            self.behaviour_counts[key] = self.behaviour_counts.get(key, 0) + 1

    def to_jsonable(self) -> dict[str, Any]:
        mean_behaviour = {
            key: self.behaviour[key] / self.behaviour_counts[key]
            for key in self.behaviour
        }
        return {
            "name": self.name,
            "games": self.games,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "win_rate": self.wins / self.games if self.games else 0.0,
            "score_rate": (self.wins + 0.5 * self.ties) / self.games
            if self.games
            else 0.0,
            "wilson": wilson_interval(self.wins + 0.5 * self.ties, self.games),
            "avg_rank": self.rank_sum / self.games if self.games else 0.0,
            "avg_score": self.score_sum / self.games if self.games else 0.0,
            "avg_behaviour": mean_behaviour,
        }


def aggregate(
    specs: Sequence[AgentSpec], records: Sequence[GameRecord], seats: int
) -> dict[str, Any]:
    """Build the aggregate tables that back the Markdown report."""
    by_agent = {spec.name: AgentAggregate(spec.name) for spec in specs}
    pair_stats: dict[tuple[str, str], PairStats] = {}
    error_count = 0
    for record in records:
        if record.error is not None:
            error_count += 1
            continue
        for seat, _rival in enumerate(record.seat_assignment):
            name = record.names[seat]
            by_agent[name].add_game(seat, record, seats)
            if seats == PAIRWISE_SEATS:
                rival_index = 1 - seat
                key = (name, record.names[rival_index])
                stats = pair_stats.setdefault(key, PairStats())
                stats.record(record.winner_seats, seat)
    return {
        "agents": {name: agg.to_jsonable() for name, agg in by_agent.items()},
        "pairs": {
            f"{row} vs {col}": {
                "games": stats.games,
                "wins": stats.wins,
                "ties": stats.ties,
                "losses": stats.losses,
                "score_rate": stats.score_rate,
                "wilson": wilson_interval(stats.wins + 0.5 * stats.ties, stats.games),
            }
            for (row, col), stats in pair_stats.items()
        },
        "error_games": error_count,
    }


def format_report(
    specs: Sequence[AgentSpec],
    seats: int,
    config: dict[str, Any],
    results: dict[str, Any],
) -> str:
    """Render the JSON aggregate as a compact Markdown report."""
    lines: list[str] = [
        "# League report",
        "",
        f"- generated: {config['created_utc']}",
        f"- agents: {', '.join(spec.name for spec in specs)}",
        f"- seats: {seats}, games per matchup: {config['games_per_matchup']}",
        f"- seed segment: `{config['segment']}` "
        f"[{config['seed_start']}, {config['seed_start'] + config['seed_count']})",
        f"- git revision: `{config['git_revision']}`",
        f"- PYTHONHASHSEED: `{config['python_hash_seed']}`",
        f"- error games: {results['error_games']}",
        "",
    ]
    standings = sorted(results["agents"].values(), key=lambda row: -row["score_rate"])
    lines += [
        "## Standings",
        "",
        "| agent | games | win rate | score rate | Wilson 95% | avg rank | avg score |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in standings:
        low, high = row["wilson"]
        lines.append(
            f"| {row['name']} | {row['games']} | {row['win_rate']:.1%} "
            f"| {row['score_rate']:.1%} | [{low:.1%}, {high:.1%}] "
            f"| {row['avg_rank']:.2f} | {row['avg_score']:.2f} |"
        )
    if seats == PAIRWISE_SEATS and results["pairs"]:
        lines += [
            "",
            "## Win-rate matrix (row beats column; ties split)",
            "",
        ]
        names = [spec.name for spec in specs]
        header = "| agent | " + " | ".join(names) + " |"
        divider = "|---" * (len(names) + 1) + "|"
        lines += [header, divider]
        for row_name in names:
            cells = []
            for col_name in names:
                if row_name == col_name:
                    cells.append("—")
                    continue
                stats = results["pairs"].get(f"{row_name} vs {col_name}")
                if stats is None or stats["games"] == 0:
                    cells.append("n/a")
                    continue
                low, high = stats["wilson"]
                cells.append(
                    f"{stats['score_rate']:.1%} "
                    f"[{low:.1%}, {high:.1%}] (g={stats['games']})"
                )
            lines.append(f"| {row_name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Behaviour metrics (per-game averages)",
        "",
        "| agent | turns | buys | reserves | collects | cards | nobles | rounds to 15 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    behaviour_keys = ("turns", "buys", "reserves", "collects", "cards", "nobles")
    for row in standings:
        avg = row["avg_behaviour"]
        cells = [f"{avg.get(key, 0.0):.2f}" for key in behaviour_keys]
        rounds = avg.get("rounds_to_15")
        cells.append(f"{rounds:.2f}" if rounds is not None else "n/a")
        lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def capture_runtime() -> dict[str, Any]:
    """Snapshot provenance fields for the manifest."""
    revision = "unknown"
    with contextlib.suppress(Exception):
        repo = git.Repo(
            Path(__file__).resolve().parent,
            search_parent_directories=True,
        )
        revision = str(repo.head.commit)[:12]
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": revision,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def run_league(  # noqa: PLR0913 - explicit knobs beat a config blob here
    specs: Sequence[AgentSpec],
    seats: int,
    games_per_matchup: int,
    *,
    segment: SeedSegment = VALIDATION,
    wrap_seeds: bool = False,
    time_limit: float = 1.0,
    warning_limit: int = 3,
    workers: int = 1,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full league and return the JSON-serialisable result bundle."""
    tasks = build_tasks(
        specs,
        seats,
        games_per_matchup,
        allocate_seeds(
            segment,
            _matchup_count(specs, seats) * games_per_matchup,
            wrap=wrap_seeds,
        ),
        time_limit=time_limit,
        warning_limit=warning_limit,
    )
    return _execute(
        specs, seats, games_per_matchup, tasks, segment, workers, output_dir
    )


def _matchup_count(specs: Sequence[AgentSpec], seats: int) -> int:
    """Number of ordered seat assignments for the given roster."""
    count = 1
    for offset in range(seats):
        count *= len(specs) - offset
    return count


def _execute(  # noqa: PLR0913 - mirrors run_league's signature split
    specs: Sequence[AgentSpec],
    seats: int,
    games_per_matchup: int,
    tasks: list[GameTask],
    segment: SeedSegment,
    workers: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    runtime = capture_runtime()
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        **runtime,
        "agents": [asdict(spec) for spec in specs],
        "seats": seats,
        "games_per_matchup": games_per_matchup,
        "game_rule": "LimitRoundsGameRule",
        "segment": segment.name,
        "seed_start": segment.start,
        "seed_count": len(tasks),
        "sealed_segment": segment.sealed,
        "workers": workers,
    }
    if workers > 1:
        records = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(play_task_safe, tasks):
                records.append(record)
    else:
        records = [play_task_safe(task) for task in tasks]
    results: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        **manifest,
        "matchups": [],
        "games": [record.to_jsonable() for record in records],
        **aggregate(specs, records, seats),
    }
    matchups: dict[int, dict[str, Any]] = {}
    for task, record in zip(tasks, records, strict=True):
        slot = matchups.setdefault(
            task.matchup_index,
            {
                "matchup_index": task.matchup_index,
                "seat_assignment": list(task.seat_assignment),
                "names": [record.names[seat] for seat in task.seat_assignment],
                "games": 0,
                "wins": [0] * seats,
                "ties": 0,
                "errors": 0,
            },
        )
        slot["games"] += 1
        if record.error is not None:
            slot["errors"] += 1
        elif len(record.winner_seats) > 1:
            slot["ties"] += 1
        else:
            for seat in record.winner_seats:
                slot["wins"][seat] += 1
    results["matchups"] = [matchups[key] for key in sorted(matchups)]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "league_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "league_report.md").write_text(
            format_report(specs, seats, manifest, results), encoding="utf-8"
        )
    return results


def load_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (exposed for tests)."""
    parser = argparse.ArgumentParser(
        prog="splendor-league",
        description="Round-robin league evaluator with behaviour metrics.",
    )
    parser.add_argument(
        "-a",
        "--agents",
        required=True,
        help="comma list of agent modules (name=module allowed); each must expose myAgent",
    )
    parser.add_argument(
        "-n", "--seats", type=int, default=2, help="seats per game (2-4)"
    )
    parser.add_argument(
        "-m", "--games", type=int, default=10, help="games per seat assignment"
    )
    parser.add_argument(
        "--segment",
        default=VALIDATION.name,
        help="seed segment to draw from (see docs/seed_registry.md)",
    )
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--warning-limit", type=int, default=3)
    parser.add_argument(
        "--workers", type=int, default=1, help="parallel game processes"
    )
    parser.add_argument(
        "--wrap-seeds",
        action="store_true",
        help="cycle the segment's seeds when the request exceeds capacity "
        "(acceptable across different matchups; recorded in the manifest)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory (default runs/league/<ts>)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``splendor-league``."""
    args = load_parser().parse_args(argv)
    specs = parse_agent_list(args.agents)
    segment = resolve_segment(args.segment)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "warning: PYTHONHASHSEED is not 0; determinism is not guaranteed "
            "(see docs/IMPROVEMENT_ROADMAP_20260912.md risk register)",
            file=sys.stderr,
        )
    output_dir = (
        Path(args.output)
        if args.output
        else Path("runs/league")
        / f"{datetime.now(UTC).strftime('%y-%m-%d_%H-%M-%S')}__league"
    )
    results = run_league(
        specs,
        args.seats,
        args.games,
        segment=segment,
        wrap_seeds=args.wrap_seeds,
        time_limit=args.time_limit,
        warning_limit=args.warning_limit,
        workers=args.workers,
        output_dir=output_dir,
    )
    print(f"league finished: {output_dir / 'league_report.md'}")
    standings = sorted(results["agents"].values(), key=lambda row: -row["score_rate"])
    for row in standings:
        low, high = row["wilson"]
        print(
            f"  {row['name']:<16} score_rate={row['score_rate']:6.1%} "
            f"win={row['win_rate']:6.1%} wilson=[{low:.1%}, {high:.1%}] "
            f"games={row['games']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
