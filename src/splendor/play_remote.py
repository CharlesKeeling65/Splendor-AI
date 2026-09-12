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
room once, publishes its URL (``events_dir/room_url.txt``) and owns the
start; the rest join and report their seat through
``events_dir/seat<i>.ready``, then the owner starts and verifies the table
went live. One room is reused for every game of the run, matching the
measured end-of-game reality (the page returns to the room with the seats
still taken), or a pinned ``--room-url`` puts bots alongside humans
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import (
    extract_snapshot,
    is_my_turn,
    waiting_seat,
)
from splendor.browser.driver import BrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.browser.monitor import _describe_action, anomaly_count
from splendor.browser.session import SessionManager
from splendor.remote.client import InferenceClient
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

REST_SECONDS = (5.0, 15.0)
ROOM_FILE = "room_url.txt"
READY_FILE_FORMAT = "seat{bot}.ready"
JOIN_WAIT_SECONDS = 120.0
# The owner retries 开始游戏 at human pace until the table is live: the seat
# count it depends on is not readable from the DOM, so a single click (the
# old fixed-sleep design) silently produced rooms that never started.
START_RETRY_SECONDS = 5.0
START_WAIT_SECONDS = 60.0
# A seat click is only believed after the seat map shows this browser in a
# seat (SessionManager.my_seat); the server needs a beat to render the badge.
SEAT_VERIFY_SECONDS = 8.0
DEFAULT_LOGS_DIR = "logs"


# ----- event stream -------------------------------------------------------------
class EventWriter:
    """
    Append-only JSONL event log, one file per bot (no cross-process locks).

    Every emit is dual-written: the live dashboard stream
    (``events_dir/bot<i>.jsonl``) plus a per-game backup under
    ``logs_dir/bot<i>/game-NNN.jsonl`` so a finished run can be replayed
    even after the live files keep growing.
    """

    def __init__(
        self,
        events_dir: Path,
        bot_id: int,
        logs_dir: Path | None = None,
    ) -> None:
        events_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id
        self._path = events_dir / f"bot{bot_id}.jsonl"
        self._logs_dir = logs_dir
        self._game_path: Path | None = None

    def start_game(self, game_index: int) -> None:
        """Open a fresh per-game backup file (0-based index -> 1-based name)."""
        if self._logs_dir is None:
            return
        bot_dir = self._logs_dir / f"bot{self._bot_id}"
        bot_dir.mkdir(parents=True, exist_ok=True)
        self._game_path = bot_dir / f"game-{game_index + 1:03d}.jsonl"
        self._game_path.write_text("", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        event["ts"] = round(time.time(), 3)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        if self._game_path is not None:
            with self._game_path.open("a", encoding="utf-8") as handle:
                handle.write(line)


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
    events = EventWriter(
        Path(options["events_dir"]),
        bot_id,
        logs_dir=Path(options["logs_dir"]) if options.get("logs_dir") else None,
    )
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
    """One ego-browser roundtrip listing browser profiles for isolation.

    ``console.log`` inside the ego-browser node runtime lands on stderr
    (the CLI banners go to stdout), so both streams are scanned for the
    JSON array line.
    """
    script = "(async () => console.log(JSON.stringify(await profiles())))()"
    completed = subprocess.run(
        ["ego-browser", "nodejs", "-e", script],
        capture_output=True, text=True, timeout=60.0, check=False,
    )
    combined = (completed.stdout + "\n" + completed.stderr).splitlines()
    json_lines = [
        line for line in combined if line.lstrip().startswith("[")
    ]
    for line in reversed(json_lines):
        try:
            profiles = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(profiles, list) and profiles:
            return profiles
    raise SystemExit(
        "could not list ego-browser profiles "
        f"(exit {completed.returncode}): {completed.stderr[-300:]}"
    )


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
        if _fresh_marker(room_file, options):  # never a previous run's room
            return room_file.read_text(encoding="utf-8").strip()
        time.sleep(1.0)
    raise TimeoutError("room URL never published by bot 0")


def _fresh_marker(path: Path, options: dict[str, Any]) -> bool:
    """
    Whether ``path`` was written by *this* run.

    The coordination files live at fixed names so the dashboard and the human
    reader keep their familiar layout, which means a previous run's leftovers
    are still on disk when a new one starts. Measured 2026-09-11: a stale
    ``room_url.txt`` sent the peer into the *previous* room (it even reached
    ``game_start`` there while the owner was still creating a new room), and a
    stale seat marker satisfied the seat chain before the owner had done
    anything. Freshness is therefore decided by mtime against the run's start
    timestamp - ignoring an old file beats deleting it (no destructive step,
    and two runs in one directory stay harmless to each other).
    """
    try:
        return path.stat().st_mtime >= float(options["run_started"])
    except FileNotFoundError:
        return False


def _wait_for_predecessor_seat(bot_id: int, options: dict[str, Any]) -> None:
    """
    Chain the seat taking: bot N loads the room only after bot N-1 sits.

    Measured 2026-09-11 (the reason self-play could never start a game): both
    bots clicked 加入 on pages rendered *before* the other's join, so both
    clicks addressed seat 1 - the later one displaced the first bot into the
    spectator row, leaving a room with one player (``开始游戏`` refused) and
    one 观战中. Loading the room after the predecessor is seated makes every
    click land on the next genuinely free seat.
    """
    if options["room_url"] or bot_id == 0:
        return  # pinned human rooms keep their own seat order
    path = _ready_path(options, bot_id - 1)
    deadline = time.monotonic() + JOIN_WAIT_SECONDS
    while not _fresh_marker(path, options):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"bot{bot_id} waited {JOIN_WAIT_SECONDS:.0f}s for bot{bot_id - 1} "
                f"to sit down ({path.name} missing)"
            )
        time.sleep(1.0)


def _ready_path(options: dict[str, Any], bot_id: int) -> Path:
    return Path(options["events_dir"]) / READY_FILE_FORMAT.format(bot=bot_id)


def _announce_seated(options: dict[str, Any], bot_id: int, events: EventWriter) -> None:
    """Publish this bot's seat so the owner can wait for a full room."""
    _ready_path(options, bot_id).write_text(str(bot_id), encoding="utf-8")
    events.emit({"type": "log", "level": "info",
                 "message": f"bot{bot_id} took a seat; told the room owner"})


def _wait_for_seats(options: dict[str, Any], events: EventWriter) -> None:
    """
    Wait until every other bot has taken its seat.

    The seat handshake travels through the event directory rather than the
    DOM: the room *does* render a seat map (``div#userseat<i>``, see
    ``SessionManager.my_seat``), but a peer's occupancy there is only visible
    after the server pushes it, so the owner would be polling a picture it
    cannot trust. A file written by the peer itself, after its own seat click
    was verified, is the unambiguous signal.
    """
    deadline = time.monotonic() + JOIN_WAIT_SECONDS
    for bot_id in range(1, options["bots"]):
        path = _ready_path(options, bot_id)
        while not _fresh_marker(path, options):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"bot{bot_id} never reported a seat ({path.name} missing "
                    f"after {JOIN_WAIT_SECONDS:.0f}s); refusing to start an "
                    "empty room"
                )
            time.sleep(1.0)
    events.emit({"type": "log", "level": "info",
                 "message": f"all {options['bots']} seats are taken"})


def _game_running(snapshot: Mapping[str, Any]) -> bool:
    """
    Whether the room view has become a live table.

    The room page carries no turn-status sentence at all, so *any* measured
    等待(你|玩家N)… status - my decision point or a peer's - means the game is
    on. That is the only honest signal available before reset() starts its
    own, stricter (my-turn-only) wait.
    """
    status = snapshot["status"]
    return is_my_turn(status) or waiting_seat(status) is not None


def _start_until_running(
    env: BrowserSplendorEnv, events: EventWriter
) -> None:
    """Press 开始游戏 until the table is live, bounded and human-paced."""
    deadline = time.monotonic() + START_WAIT_SECONDS
    attempts = 0
    while True:
        attempts += 1
        try:
            env.session.start_game()
        except ValueError as error:
            # Owner-only button, or the room is not full yet: the page
            # refuses silently, so retrying is the only way to observe it.
            events.emit({"type": "log", "level": "info",
                         "message": f"开始游戏 not clickable yet: {error}"})
        if _game_running(extract_snapshot(env.driver)):
            events.emit({"type": "log", "level": "info",
                         "message": f"table live after {attempts} 开始游戏 click(s)"})
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"the room never started after {attempts} 开始游戏 click(s) in "
                f"{START_WAIT_SECONDS:.0f}s; check that every seat is taken"
            )
        time.sleep(START_RETRY_SECONDS)


def _seated_within(session: SessionManager, budget: float) -> int:
    """Poll :meth:`SessionManager.my_seat` until the seat badge shows up."""
    deadline = time.monotonic() + budget
    while True:
        seat = session.my_seat()
        if seat or time.monotonic() > deadline:
            return seat
        time.sleep(1.0)


def _take_seat(
    session: SessionManager,
    bot_id: int,
    game_index: int,
    options: dict[str, Any],
    events: EventWriter,
) -> None:
    """
    Click 加入 and *verify* the seat was actually taken.

    Two live-measured failure modes are handled here (2026-09-11):

    * a room entered with a rotated identity renders only 重连 - the seat
      click then finds nothing, and ``recover()`` (navigate + dismiss 重连)
      is the recipe that brings the 加入 buttons back;
    * a page rendered before a peer's join still offers 加入 for an occupied
      seat, and clicking it leaves the clicker spectating. Hence the seat map
      read afterwards: no 我 badge means no seat, whatever the click returned.
    """
    try:
        session.join_first_free_seat()
    except ValueError as error:
        if game_index != 0 or options["room_url"]:
            # Steady state from game 2 on (and in pinned rooms): the seat is
            # already held, so the room page renders no 加入 button at all.
            events.emit({"type": "log", "level": "info",
                         "message": f"bot{bot_id} already holds a seat"})
            return
        events.emit({"type": "log", "level": "warn",
                     "message": f"room offers no 加入 seat yet ({error}); "
                                "recovering the session"})
        session.recover()
        session.join_first_free_seat()

    if _seated_within(session, SEAT_VERIFY_SECONDS):
        return
    events.emit({"type": "log", "level": "warn",
                 "message": f"bot{bot_id} is still spectating after the 加入 click; "
                            "reloading the room for a fresh seat map"})
    session.recover()
    session.join_first_free_seat()
    if not _seated_within(session, SEAT_VERIFY_SECONDS):
        raise TimeoutError(
            f"bot{bot_id} never took a seat (still 观战中); the room may be "
            "full, or another player holds the same seat"
        )


def _coordinate_room(
    bot_id: int,
    game_index: int,
    env: BrowserSplendorEnv,
    options: dict[str, Any],
    events: EventWriter,
) -> None:
    """
    Get this bot onto the room page, seated, and let the owner start the game.

    Bot 0 owns the room (create + publish + start); every other bot reads the
    published URL and joins. The old design slept a fixed 20s and clicked
    开始游戏 once - both halves failed silently when a peer's browser start-up
    ran long, which is exactly how self-play dead-locked.
    """
    session = env.session
    room_url = session.room_url or options["room_url"]
    if room_url is not None:
        # Seats are chained (see _wait_for_predecessor_seat) so this page is
        # loaded *after* the previous bot sat: its 加入 click then addresses
        # the next genuinely free seat instead of a seat taken a moment ago.
        _wait_for_predecessor_seat(bot_id, options)
        env.driver.navigate(room_url)  # reconnection is automatic on load
    else:
        room_url = session.create_room(seats=max(2, options["bots"]))
        session.pin_room(room_url)
        Path(options["events_dir"], ROOM_FILE).write_text(room_url, encoding="utf-8")
        events.emit({"type": "log", "level": "info",
                     "message": f"room created: {room_url}"})
    _take_seat(session, bot_id, game_index, options, events)
    _announce_seated(options, bot_id, events)
    if bot_id == 0 and not options["room_url"]:
        _wait_for_seats(options, events)
        _start_until_running(env, events)
    else:
        try:
            session.start_game()
        except ValueError:
            pass  # not the owner (or already started) - the owner starts


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


def _board_summary(snapshot: Mapping[str, Any], my_seat: int) -> dict[str, Any]:
    """
    Compact human-readable board view for remote-act verification.

    This is what a human can cross-check against the live page (scores,
    supply, my gems, status) plus the mask size that was actually sent to
    the inference server - not the full 265-d obs vector.
    """
    panels = snapshot.get("panels") or []
    scores = {
        str(panel.get("seat", index + 1)): float(panel.get("score", 0))
        for index, panel in enumerate(panels)
    }
    my_panel = next(
        (p for p in panels if p.get("seat") == my_seat),
        panels[my_seat - 1] if 0 < my_seat <= len(panels) else None,
    )
    dealt_filled = [
        sum(1 for slot in row if slot is not None)
        for row in (snapshot.get("dealt") or [])
    ]
    return {
        "status": snapshot.get("status") or "",
        "scores": scores,
        "my_gems": dict(my_panel.get("gems") or {}) if my_panel else {},
        "my_reserved": len(my_panel.get("reserved_tiers") or []) if my_panel else 0,
        "supply": dict(snapshot.get("supply") or {}),
        "dealt_filled": dealt_filled,
        "deck_counts": list(snapshot.get("deck_counts") or []),
    }


def _ranking_payload(
    top: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Server ranking + Chinese descriptions for the dashboard panel."""
    ranked: list[dict[str, Any]] = []
    for item in top:
        idx = int(item["idx"])
        ranked.append(
            {
                "idx": idx,
                "q": float(item["q"]),
                "desc": _describe_action(ALL_ACTIONS[idx]),
            }
        )
    return ranked


def _run_game(  # noqa: PLR0913, PLR0914, PLR0917 - harness surface (repo style)
    env: BrowserSplendorEnv,
    client: InferenceClient,
    events: EventWriter,
    bot_id: int,
    game_index: int,
    options: dict[str, Any],
) -> None:
    """One game: the play_web loop with remote inference + event emission."""
    events.start_game(game_index)
    _coordinate_room(bot_id, game_index, env, options, events)

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
        decision = client.act(options["model"], obs, mask)
        action = decision.action
        desc = _describe_action(ALL_ACTIONS[action])
        act_snapshot = extract_snapshot(env.driver)
        board = _board_summary(act_snapshot, ctx.my_seat)
        ctx.emit({
            "type": "remote_act",
            "seq": seq + 1,
            "action": action,
            "desc": desc,
            "top": _ranking_payload(decision.top),
            "legal_count": int(np.count_nonzero(mask)),
            "board": board,
        })
        prev_snapshot = act_snapshot

        obs, _reward, terminated, _trunc, _info = env.step(action)
        seq += 1
        # env.step returning without raising means the executor completed the
        # click sequence for this same action index (no silent re-mapping).
        ctx.emit({
            "type": "browser_act",
            "seq": seq,
            "action": action,
            "desc": desc,
            "executed": True,
        })
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
    """
    Surface only the parity lines that need a human; return the anomaly count.

    Healthy decisions are silent: ``engine-only=0`` plus the E5/E6 and
    DOM-ONLY over-approximation buckets fire on almost every step and used to
    flood both the JSONL stream and the dashboard log. The report itself is
    emitted only when :func:`~splendor.browser.monitor.anomaly_count` is
    non-zero (page redesign / DOM extraction bug / ``engine-only > 0``), so a
    clean run stays readable and an anomaly is still fully attributed. The
    returned count is what ``game_end`` reports - never ``len(report)``, which
    would count the unconditional header.
    """
    report = list(getattr(env, "last_parity_report", []))
    count = anomaly_count(report)
    if count:
        ctx.emit({
            "type": "parity",
            "level": "warn",
            "anomalies": count,
            "lines": report,
        })
    return count


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
    # Freshness stamp for the coordination files: every worker ignores
    # markers older than this (see _fresh_marker), which is what keeps a
    # previous run's room URL and seat markers from being believed.
    options["run_started"] = time.time()
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
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR,
                        help="Per-game JSONL backup directory (logs/bot<i>/game-NNN.jsonl).")
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
