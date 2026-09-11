"""
``play-web`` console script: DQN checkpoint + BrowserSplendorEnv, playing
live games on game.hullqin.cn/ccbs.

Deployment is a different lifecycle than training (long unattended runs,
session recovery, reporting), so it lives in its own module instead of the
``dqn`` package: a training-loop refactor must never break a running
deployment and vice versa.

Etiquette is a hard constraint (roadmap R7, plan phase-2 M2.5): clicks are
human-paced inside the executor, and games are separated by a random
5-15s rest - "human-like" has to hold at both timescales.
"""

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import extract_snapshot
from splendor.browser.driver import BrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.browser.monitor import anomaly_count
from splendor.browser.session import SessionManager
from splendor.splendor.gym.base import SplendorEnvBase

REPORT_FILE = "s2r_games.json"
REST_SECONDS = (5.0, 15.0)


@dataclass
class GameReport:
    """Outcome of one web game (the s2r report's row unit)."""

    result: str  # "win" / "draw" / "loss" / "aborted"
    my_score: float
    rival_score: float
    steps: int
    duration: float
    mask_anomalies: int


def load_agent(checkpoint: Path) -> QNetwork:
    """
    Load a trained DQN checkpoint (identical loading path as the local agent
    - running stats included - so deployment normalizes exactly like
    training did).
    """
    return load_saved_dqn(checkpoint)


def _greedy_action(env: SplendorEnvBase, q_net: QNetwork, obs: np.ndarray) -> int:
    mask = env.get_legal_actions_mask()
    return q_net.act(torch.from_numpy(obs), torch.from_numpy(mask).float())


def run_game(
    env: BrowserSplendorEnv, q_net: QNetwork, max_steps: int = 200
) -> GameReport:
    """
    Play one web game with the greedy policy and report the outcome.

    Wins/losses use the engine's calScore convention on the final panel
    scores (highest score wins; ties broken toward fewer cards would need
    the full card counts - on the web only scores are visible, so panel
    equality counts as a draw).
    """
    start = time.monotonic()
    anomalies = 0
    steps = 0
    my_score, rival_score = 0.0, 0.0
    # The env's rewards are panel score deltas, so their sum telescopes to my
    # final score - the fallback for the measured room view (E3), where the
    # game ends with the panels gone. Snapshot reads stay authoritative while
    # a board is visible.
    telescoped_score = 0.0
    saw_board = False

    obs, info = env.reset()
    my_seat = int(info["my_id"])
    terminated = False
    for _ in range(max_steps):
        anomalies += anomaly_count(getattr(env, "last_parity_report", []))
        action = _greedy_action(env, q_net, obs)
        obs, reward, terminated, _truncated, _info = env.step(action)
        steps += 1
        telescoped_score += float(reward)
        snapshot_scores = _panel_scores(env, my_seat)
        if snapshot_scores is not None:
            saw_board = True
            my_score, rival_score = snapshot_scores
        if terminated:
            break

    if not saw_board:
        my_score = telescoped_score

    if not terminated:
        result = "aborted"  # max_steps hit: report honestly, never fake a win
    elif my_score > rival_score:
        result = "win"
    elif my_score < rival_score:
        result = "loss"
    else:
        result = "draw"
    return GameReport(
        result=result,
        my_score=my_score,
        rival_score=rival_score,
        steps=steps,
        duration=time.monotonic() - start,
        mask_anomalies=anomalies,
    )


def _panel_scores(env: BrowserSplendorEnv, my_seat: int) -> tuple[float, float] | None:
    """Best-effort (my, best-rival) panel scores from the env's last view."""
    try:
        snapshot = extract_snapshot(env.driver)
        if not snapshot["panels"]:
            return None
        mine = next(
            (p for p in snapshot["panels"] if p["seat"] == my_seat),
            snapshot["panels"][0],
        )
        rivals = [p for p in snapshot["panels"] if p["seat"] != my_seat]
        rival_best = max((p["score"] for p in rivals), default=0.0)
        return float(mine["score"]), float(rival_best)
    except Exception:  # reporting must never kill a deployment
        return None


def main() -> None:
    """Entry point of the ``play-web`` console script."""
    options = _parse_args()
    q_net = load_agent(options["checkpoint"])

    driver = options["driver_factory"]()
    session = SessionManager(
        driver, room_url=options["room_url"], seats=options["seats"]
    )
    env = BrowserSplendorEnv(
        driver, session, poll_interval=options["poll"], step_timeout=options["timeout"],
        feature_version=q_net.feature_version,
    )

    reports: list[GameReport] = []
    for game_index in range(options["games"]):
        try:
            report = run_game(env, q_net, max_steps=options["max_steps"])
        except Exception as error:  # one network hiccup must not kill the run
            print(f"game {game_index + 1} failed ({error}); recovering once")
            session.recover()
            try:
                report = run_game(env, q_net, max_steps=options["max_steps"])
            except Exception as error:
                print(f"game {game_index + 1} aborted after recovery: {error}")
                report = GameReport(
                    result="aborted", my_score=0.0, rival_score=0.0,
                    steps=0, duration=0.0, mask_anomalies=0,
                )
        if session.room_url:
            print(f"room: {session.room_url}  <- open this URL to watch")
        reports.append(report)
        print(
            f"game {game_index + 1}/{options['games']}: {report.result} "
            f"({report.my_score:.1f} vs {report.rival_score:.1f}, "
            f"{report.steps} steps, {report.duration:.0f}s, "
            f"{report.mask_anomalies} anomalies)"
        )
        if game_index < options["games"] - 1:
            rest = random.uniform(*REST_SECONDS)
            print(f"resting {rest:.1f}s (etiquette)")
            time.sleep(rest)

    summary = {
        "games": [asdict(report) for report in reports],
        "win_rate": sum(r.result == "win" for r in reports) / len(reports),
        "draw_rate": sum(r.result == "draw" for r in reports) / len(reports),
        "avg_score": sum(r.my_score for r in reports) / len(reports),
        "total_mask_anomalies": sum(r.mask_anomalies for r in reports),
    }
    out_path = options["working_dir"] / REPORT_FILE
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"win {summary['win_rate']:.0%} / draw {summary['draw_rate']:.0%} / "
        f"avg score {summary['avg_score']:.2f} / "
        f"anomalies {summary['total_mask_anomalies']}"
    )


def _parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="play-web",
        description="Deploy a trained DQN checkpoint to game.hullqin.cn/ccbs.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the DQN checkpoint (dqn_model.pth style).",
    )
    parser.add_argument("--games", type=int, default=50, help="How many games to play.")
    parser.add_argument("--seats", type=int, default=2, help="Seats per room.")
    parser.add_argument("--room-url", type=str, default=None,
                        help="Reuse an existing room instead of creating one.")
    parser.add_argument("--poll", type=float, default=0.4, help="Turn-poll interval (s).")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-step timeout (s).")
    parser.add_argument("--max-steps", type=int, default=200, help="Per-game step cap.")
    parser.add_argument("--task-space", type=str, default="splendor-play-web",
                        help="ego-browser task space to drive the page with.")
    parser.add_argument("--working-dir", type=Path, default=Path(),
                        help="Where to write the JSON report.")
    options = vars(parser.parse_args())

    def driver_factory() -> BrowserDriver:
        # The ego-browser adapter spawns a subprocess per call - the
        # reference (slow-but-tool-agnostic) path; swap in a CDP adapter here
        # for faster polling without touching any other layer.
        return EgoBrowserDriver(options["task_space"], room_url=options["room_url"])

    factory: Callable[[], BrowserDriver] = driver_factory
    options["driver_factory"] = factory
    return options


if __name__ == "__main__":
    main()
