"""
``play-web-remote`` console script (phase-6): browser control stays local,
inference runs on the remote server (TCP JSONL, see ``splendor.remote``).

Split rationale: the ego-browser adapter must own the local browser, while
checkpoints + Monte-Carlo win-rate rollouts belong on the inference machine.
The harness keeps the entire ``BrowserSplendorEnv`` surface local and swaps
only the two model touchpoints for remote calls:

* greedy action selection  -> ``client.act(model_id, obs, mask)``;
* win-rate estimation      -> ``client.estimate_winrate(...)`` per action.

Multi-bot: ``--bots N`` forks N worker processes - each owns one browser
task space and one seat (the session layer's one-process-per-seat contract),
and all share the same remote inference server through separate
connections. Models can differ per worker: ``--model`` takes one id for
every bot or a comma-separated list mapped to bots in order (checkpoint
matches between bot0 and bot1). Without ``--room-url`` bot 0 creates the
room and publishes its URL; the rest join and the owner starts the game
(self-play), or a pinned ``--room-url`` puts bots alongside humans
(etiquette still applies).

Event stream: every bot appends JSONL events (actions with per-seat win
rates before/after, gem-discard detections, parity anomalies, game ends) to
``events_dir/bot<i>.jsonl`` - the dashboard (``splendor.remote.dashboard``)
watches that directory.

Etiquette is inherited unchanged from play-web: human-paced clicks inside
the executor, random 5-15s rest between games.
"""

import argparse
import json
import multiprocessing
import random
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import extract_snapshot, waiting_seat
from splendor.browser.driver import BrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.browser.monitor import _describe_action
from splendor.browser.session import SessionManager
from splendor.remote.client import InferenceClient
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

REST_SECONDS = (5.0, 15.0)
ROOM_FILE = "room_url.txt"
JOIN_WAIT_SECONDS = 120.0
START_DELAY_SECONDS = 20.0


# ----- event stream -------------------------------------------------------------
class EventWriter:
    """Append-only JSONL event log, one file per bot (no cross-process locks)."""

    def __init__(self, events_dir: Path, bot_id: int) -> None:
        events_dir.mkdir(parents=True, exist_ok=True)
        self._path = events_dir / f"bot{bot_id}.jsonl"

    def emit(self, event: dict[str, Any]) -> None:
        event["ts"] = round(time.time(), 3)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


# ----- snapshot labelling ---------------------------------------------------------
def _panel_gem_total(panel: dict[str, Any]) -> int:
    return sum(panel["gems"].values())


def diff_label(
    prev_panel: dict[str, Any],
    cur_panel: dict[str, Any],
    prev_supply: dict[str, int],
    cur_supply: dict[str, int],
) -> str:
    """
    Heuristic seat-activity label from consecutive snapshots (rival actions
    have no action index - the DOM deltas are all we observe).

    Conservation makes the three main kinds separable: taking gems moves
    counters supply->panel, discarding moves panel->supply (the >10-gem
    return sub-flow), buying raises the score (with or without a payment).
    """
    score_delta = cur_panel["score"] - prev_panel["score"]
    gem_delta = _panel_gem_total(cur_panel) - _panel_gem_total(prev_panel)
    if score_delta > 0:
        return f"购买(得分+{score_delta})"
    if gem_delta > 0:
        return f"取宝石(+{gem_delta})"
    if gem_delta < 0:
        return f"弃宝石(-{-gem_delta})"
    if len(cur_panel["reserved_tiers"]) > len(prev_panel["reserved_tiers"]):
        return "预定"
    return "状态变化"


def _actor_seat_of(snapshot: Mapping[str, Any], my_seat: int) -> int:
    """Whose decision point the snapshot shows (falls back to my seat)."""
    seat = waiting_seat(snapshot["status"])
    if seat is not None:
        return seat
    if "等待你操作" in snapshot["status"]:
        return my_seat
    return my_seat


# ----- per-bot worker ---------------------------------------------------------------
def _model_for(bot_id: int, options: dict[str, Any]) -> str:
    """Resolve this bot's model: one shared id, or per-slot comma-separated."""
    ids = [part.strip() for part in options["model"].split(",") if part.strip()]
    return ids[0] if len(ids) == 1 else ids[bot_id]


def run_bot(bot_id: int, options: dict[str, Any]) -> None:
    """
    One seat, one browser task space, one remote connection. Runs in a child
    process (multiprocessing spawn): the ego-browser adapter shells out per
    call and the isolation keeps one bot's crash from the others.
    """
    # Each worker pins its own model (per-bot ids allow checkpoint matches);
    # the child's options dict is a private pickle copy, safe to specialize.
    options = {**options, "model": _model_for(bot_id, options)}
    events = EventWriter(Path(options["events_dir"]), bot_id)
    client = InferenceClient(
        options["server_host"], options["server_port"], timeout=options["timeout"]
    )
    ping = client.ping()
    models = {model["id"]: model for model in ping["models"]}
    if options["model"] not in models:
        raise SystemExit(
            f"model {options['model']!r} not served (available: {sorted(models)})"
        )
    feature_version = models[options["model"]]["feature_version"]
    events.emit({"type": "log", "level": "info",
                 "message": f"bot{bot_id} connected; server models={sorted(models)}"})

    driver = _driver_for(bot_id, options)
    session = SessionManager(driver, room_url=_room_for(bot_id, options))
    env = BrowserSplendorEnv(
        driver, session,
        poll_interval=options["poll"], step_timeout=options["timeout"],
        feature_version=feature_version,
    )

    for game_index in range(options["games"]):
        try:
            _run_game(env, client, events, bot_id, game_index, options)
        except Exception as error:  # one hiccup must not kill the bot
            events.emit({"type": "log", "level": "warn",
                         "message": f"game {game_index + 1} failed: {error}"})
            session.recover()
        if game_index < options["games"] - 1:
            rest = random.uniform(*REST_SECONDS)
            events.emit({"type": "log", "level": "info",
                         "message": f"resting {rest:.1f}s (etiquette)"})
            time.sleep(rest)
    client.close()


def _driver_for(bot_id: int, options: dict[str, Any]) -> BrowserDriver:
    room_url = _room_for(bot_id, options)
    return EgoBrowserDriver(
        f"{options['task_space']}-bot{bot_id}",
        room_url=room_url,
        profile_id=options["profile_ids"][bot_id],
    )


def _list_ego_profiles() -> list[dict[str, Any]]:
    """One ego-browser roundtrip listing browser profiles for isolation."""
    script = "(async () => console.log(JSON.stringify(await profiles())))()"
    completed = subprocess.run(
        ["ego-browser", "nodejs", "-e", script],
        capture_output=True, text=True, timeout=60.0, check=False,
    )
    try:
        profiles = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise SystemExit(
            "could not list ego-browser profiles "
            f"(exit {completed.returncode}): {completed.stderr[-300:]}"
        ) from error
    return profiles


def _resolve_profile_ids(options: dict[str, Any]) -> list[str]:
    """
    One isolated browser profile per bot: cookies live at profile level, so
    this is what lets two bots hold two accounts in the same room.
    """
    explicit = options.get("explicit_profile_ids") or []
    if explicit:
        if len(explicit) != options["bots"]:
            raise SystemExit(
                f"--profile-ids takes exactly {options['bots']} id(s), "
                f"got {len(explicit)}"
            )
        return explicit
    profiles = _list_ego_profiles()
    if len(profiles) < options["bots"]:
        raise SystemExit(
            f"{options['bots']} bots need {options['bots']} browser profiles "
            f"for separate logins, found {len(profiles)}: "
            f"{[p['id'] for p in profiles]}; import more with "
            "`ego-browser import --browser chrome --profile <dir>`"
        )
    return [str(profile["id"]) for profile in profiles[: options["bots"]]]


def _room_for(bot_id: int, options: dict[str, Any]) -> str | None:
    if options["room_url"]:
        return options["room_url"]
    if bot_id == 0:
        return None  # bot 0 creates the room and publishes it
    room_file = Path(options["events_dir"]) / ROOM_FILE
    deadline = time.monotonic() + JOIN_WAIT_SECONDS
    while time.monotonic() < deadline:
        if room_file.exists():
            return room_file.read_text(encoding="utf-8").strip()
        time.sleep(1.0)
    raise TimeoutError("room URL never published by bot 0")


def _coordinate_room(
    bot_id: int, session: SessionManager, options: dict[str, Any], events: EventWriter
) -> None:
    """Bot 0 creates + publishes the room; every bot joins a free seat."""
    if options["room_url"]:
        pass  # SessionManager already pins the room; just take a seat below
    elif bot_id == 0:
        room_url = session.create_room(seats=max(2, options["bots"]))
        room_file = Path(options["events_dir"]) / ROOM_FILE
        room_file.write_text(room_url, encoding="utf-8")
        events.emit({"type": "log", "level": "info",
                     "message": f"room created: {room_url}"})
    else:
        _room_for(bot_id, options)  # already resolved by _driver_for; no-op
    session.join_first_free_seat()
    if bot_id == 0 and not options["room_url"]:
        # Let the other bots join before the owner starts the game.
        time.sleep(START_DELAY_SECONDS)
    try:
        session.start_game()
    except ValueError:
        pass  # not the owner, or already started


@dataclass
class _GameContext:
    """Per-game state bundle: keeps the game loop's signature and locals sane."""

    bot_id: int
    game_index: int
    options: dict[str, Any]
    client: InferenceClient
    events: EventWriter
    my_seat: int = 0

    def emit(self, event: dict[str, Any]) -> None:
        self.events.emit({"bot": self.bot_id, "game": self.game_index + 1, **event})

    def estimate(
        self, snapshot: Mapping[str, Any], actor_seat: int
    ) -> list[float] | None:
        """Remote MC win-rate estimate; failure degrades to None, never aborts."""
        if not snapshot["panels"]:
            return None
        try:
            result = self.client.estimate_winrate(
                self.options["model"], snapshot, actor_seat,
                n_rollouts=self.options["n_rollouts"],
            )
            return result["win_rates"]
        except Exception as error:
            self.events.emit({"type": "log", "level": "warn",
                              "message": f"winrate estimate failed: {error}"})
            return None

    def emit_rival_events(
        self,
        prev_snapshot: Mapping[str, Any],
        cur_snapshot: Mapping[str, Any],
        before: list[float] | None,
        seq: int,
    ) -> None:
        """
        Label rival seats whose panel changed across the folded opponent
        turns. Their per-action win rate cannot be estimated without blocking
        the turn-poll loop, so they inherit the surrounding estimate
        (``stale: true``) - honest degradation, not fabricated precision.
        """
        for seat, (prev_panel, cur_panel) in enumerate(
            zip(prev_snapshot["panels"], cur_snapshot["panels"], strict=False),
            start=1,
        ):
            if seat == self.my_seat or prev_panel == cur_panel:
                continue
            label = diff_label(
                prev_panel, cur_panel,
                prev_snapshot["supply"], cur_snapshot["supply"],
            )
            if "弃宝石" in label:
                self.emit({
                    "type": "discard", "seq": seq, "seat": seat,
                    "gems": {
                        colour: prev_panel["gems"][colour] - cur_panel["gems"][colour]
                        for colour in prev_panel["gems"]
                        if cur_panel["gems"].get(colour, 0) < prev_panel["gems"][colour]
                    },
                })
            self.emit({
                "type": "action", "seq": seq, "seat": seat,
                "actor": f"座位{seat}", "desc": label,
                "before": before, "after": None, "stale": True,
            })


def _run_game(  # noqa: PLR0913, PLR0917, PLR0914 - deployment harness surface (repo noqa style)
    env: BrowserSplendorEnv,
    client: InferenceClient,
    events: EventWriter,
    bot_id: int,
    game_index: int,
    options: dict[str, Any],
) -> None:
    """One game: the play_web loop with remote inference + event emission."""
    _coordinate_room(bot_id, env.session, options, events)

    obs, info = env.reset()
    start_snapshot = extract_snapshot(env.driver)
    ctx = _GameContext(
        bot_id=bot_id, game_index=game_index, options=options,
        client=client, events=events, my_seat=int(info["my_id"]),
    )
    ctx.emit({"type": "game_start", "my_seat": ctx.my_seat,
              "seats": len(start_snapshot["panels"]) or 2})

    last_scores = {
        seat: float(panel["score"])
        for seat, panel in enumerate(start_snapshot["panels"], start=1)
    }
    anomalies = 0
    before = ctx.estimate(start_snapshot, ctx.my_seat)
    seq = 0

    for _step in range(options["max_steps"]):
        anomalies += _emit_parity(env, ctx)
        mask = env.get_legal_actions_mask()
        action = client.act(options["model"], obs, mask)
        desc = _describe_action(ALL_ACTIONS[action])
        prev_snapshot = extract_snapshot(env.driver)

        obs, _reward, terminated, _trunc, _info = env.step(action)
        seq += 1
        cur_snapshot = extract_snapshot(env.driver)
        for seat, panel in enumerate(cur_snapshot["panels"], start=1):
            last_scores[seat] = float(panel["score"])

        # Own discard sub-flow is exact (the action carries returned_gems).
        returned = ALL_ACTIONS[action].returned_gems or {}
        if returned:
            ctx.emit({"type": "discard", "seq": seq, "seat": ctx.my_seat,
                      "gems": dict(returned)})

        actor_seat = _actor_seat_of(cur_snapshot, ctx.my_seat)
        after = ctx.estimate(cur_snapshot, actor_seat)
        ctx.emit({
            "type": "action", "seq": seq, "seat": ctx.my_seat,
            "actor": f"bot{bot_id}", "desc": desc,
            "before": before, "after": after, "stale": after is None,
        })
        if after is not None:
            before = after
        ctx.emit_rival_events(prev_snapshot, cur_snapshot, before, seq)

        if terminated:
            break

    ctx.emit({
        "type": "game_end",
        "scores": {str(seat): score for seat, score in sorted(last_scores.items())},
        "result": _result_of(last_scores, ctx.my_seat), "anomalies": anomalies,
    })


def _emit_parity(env: BrowserSplendorEnv, ctx: _GameContext) -> int:
    """Forward the env's mask-parity report to the event stream; return count."""
    report = list(getattr(env, "last_parity_report", []))
    if report:
        ctx.emit({"type": "parity", "lines": report})
    return len(report)


def _result_of(last_scores: dict[int, float], my_seat: int) -> str:
    """play_web's convention: highest panel score wins, equality is a draw."""
    if not last_scores:
        return "aborted"
    mine = last_scores.get(my_seat, 0.0)
    best_rival = max(
        (score for seat, score in last_scores.items() if seat != my_seat),
        default=0.0,
    )
    if mine > best_rival:
        return "win"
    if mine < best_rival:
        return "loss"
    return "draw"


# ----- entry point -----------------------------------------------------------------
def main() -> None:
    """Entry point of the ``play-web-remote`` console script."""
    options = _parse_args()
    options["profile_ids"] = _resolve_profile_ids(options)
    if options["bots"] > 1 and options["room_url"]:
        # Pinned-room self-play with several identities would fight over the
        # same seats; multi-bot self-play needs a bot-owned room.
        print("note: --bots > 1 with --room-url joins the same pinned room")
    workers = [
        multiprocessing.Process(
            target=run_bot, args=(bot_id, options), daemon=False,
            name=f"play-web-remote-bot{bot_id}",
        )
        for bot_id in range(options["bots"])
    ]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        for worker in workers:
            worker.terminate()


def _parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="play-web-remote",
        description="Browser locally, inference on the remote server (phase-6).",
    )
    parser.add_argument("--server", required=True,
                        help="Inference server host:port (e.g. 10.0.0.8:8765).")
    parser.add_argument("--model", required=True,
                        help="model_id served remotely; one id for every bot, "
                             "or comma-separated ids mapped to bots in order "
                             "(e.g. --model bot1,bot2).")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--bots", type=int, default=1,
                        help="Number of concurrent bot workers (seats).")
    parser.add_argument("--room-url", default=None,
                        help="Join an existing room instead of self-play rooms.")
    parser.add_argument("--poll", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-step timeout; also the client socket timeout.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--n-rollouts", type=int, default=16,
                        help="Monte-Carlo rollouts per win-rate estimate.")
    parser.add_argument("--task-space", default="splendor-play-web")
    parser.add_argument(
        "--profile-ids", default=None,
        help="Comma-separated ego-browser profile ids, one per bot (default: "
             "auto-assign the first N profiles). Isolates cookies per bot.",
    )
    parser.add_argument("--events-dir", default="web_events",
                        help="Directory for bot<i>.jsonl events (dashboard input).")
    options = vars(parser.parse_args())

    model_ids = [part.strip() for part in options["model"].split(",") if part.strip()]
    if len(model_ids) not in {1, options["bots"]}:
        raise SystemExit(
            "--model takes one id for all bots, or one comma-separated id per "
            f"bot (got {len(model_ids)} id(s) for {options['bots']} bot(s))"
        )

    host_text, _, port_text = options["server"].rpartition(":")
    if not host_text or not port_text.isdigit():
        raise SystemExit("--server expects host:port")
    options["server_host"] = host_text
    options["server_port"] = int(port_text)
    options["explicit_profile_ids"] = (
        [part.strip() for part in options["profile_ids"].split(",") if part.strip()]
        if options["profile_ids"]
        else []
    )
    return options


if __name__ == "__main__":
    main()
