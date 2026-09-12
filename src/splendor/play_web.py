"""
``play-web`` console script: DQN checkpoint + BrowserSplendorEnv, playing
live games on game.hullqin.cn/ccbs.

Deployment is a different lifecycle than training (long unattended runs,
session recovery, reporting), so it lives in its own module instead of the
``dqn`` package: a training-loop refactor must never break a running
deployment and vice versa.

Live reporting mirrors ``play_vs_humans``: every own decision prints the
Chinese action description, Q value, top alternatives, seat scores and the
Δscore reward. Mask-parity lines are printed only when they need a human
(``anomaly_count > 0``) so a healthy long run stays readable.

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

import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import BrowserDriver
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.browser.monitor import anomaly_count
from splendor.browser.session import SessionManager
from splendor.browser.state_builder import build_pseudo_state
from splendor.play_vs_humans import describe_action
from splendor.splendor.gym.base import SplendorEnvBase
from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule

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


def _select_action(  # noqa: PLR0913 - logging payload needs them
    env: SplendorEnvBase,
    q_net: QNetwork,
    obs: NDArray,
    snapshot: Snapshot,
    rule: SplendorGameRule | None,
    my_index: int,
    turns: int,
) -> tuple[int, float | None, str, list[tuple[int, float, str]]]:
    """
    Greedy action plus a human-readable account (play_vs_humans-style).

    Real checkpoints expose a full Q vector through ``forward``; offline
    stubs that only implement ``act`` still select an action (description
    and alternatives degrade to best-effort). The env's engine mask stays
    the legality authority and feeds the parity monitor.
    """
    mask = env.get_legal_actions_mask()
    obs_t = torch.from_numpy(obs)
    mask_t = torch.from_numpy(mask).float()

    q_values: torch.Tensor | None = None
    try:
        with torch.no_grad():
            q_values = q_net(obs_t, mask_t).squeeze(0)
        action_idx = int(q_values.argmax().item())
        q_value: float | None = float(q_values[action_idx].item())
    except Exception:  # stubs / unexpected tensor shapes: act-only path
        q_values = None
        action_idx = int(q_net.act(obs_t, mask_t))
        q_value = None

    description = "(动作描述不可用)"
    top: list[tuple[int, float, str]] = []
    try:
        if rule is not None:
            pseudo = build_pseudo_state(snapshot, my_index, turns=turns)
            legal = rule.getLegalActions(pseudo, my_index)
            mapping = create_action_mapping(legal, pseudo, my_index)
            if action_idx in mapping:
                description = describe_action(mapping[action_idx])
            if q_values is not None and mapping:
                ranked = sorted(
                    mapping, key=lambda idx: float(q_values[idx]), reverse=True
                )
                top = [
                    (idx, float(q_values[idx]), describe_action(mapping[idx]))
                    for idx in ranked[:3]
                ]
    except Exception as error:  # description is best-effort only
        print(f"[describe] 回退: {error}", flush=True)
    return action_idx, q_value, description, top


def _format_scores(snapshot: Snapshot, my_seat: int) -> str:
    """``座1=3 | 我=5`` from the live panel snapshot."""
    if not snapshot["panels"]:
        return "(无面板)"
    parts = []
    for panel in snapshot["panels"]:
        label = "我" if panel["seat"] == my_seat else f"座{panel['seat']}"
        parts.append(f"{label}={panel['score']}")
    return " | ".join(parts)


def _print_parity(env: BrowserSplendorEnv) -> int:
    """
    Print only the parity lines that need a human.

    Healthy over-approximation (E5/E6 buckets, dom-only PASS/RESERVE) stays
    silent so a 50-game run is not drowned in expected noise; the anomaly
    count still lands in GameReport.
    """
    report = list(getattr(env, "last_parity_report", []))
    count = anomaly_count(report)
    if count:
        for line in report:
            print(f"   ⚠ {line}", flush=True)
    return count


def _judge_result(terminated: bool, my_score: float, rival_score: float) -> str:
    if not terminated:
        return "aborted"  # max_steps hit: report honestly, never fake a win
    if my_score > rival_score:
        return "win"
    if my_score < rival_score:
        return "loss"
    return "draw"


def _print_step(  # noqa: PLR0913 - one console line needs them all
    label: str,
    step: int,
    description: str,
    q_value: float | None,
    scores_line: str,
    top: list[tuple[int, float, str]],
) -> None:
    q_text = f"Q={q_value:.2f}" if q_value is not None else "Q=—"
    print(
        f"▶ {label} 第 {step} 步 | 我执行: {description} | {q_text}",
        flush=True,
    )
    print(f"   比分: {scores_line}", flush=True)
    if top:
        alternatives = ", ".join(f"{text}({value:.2f})" for _, value, text in top)
        print(f"   备选: {alternatives}", flush=True)


def _build_rule(env: BrowserSplendorEnv) -> SplendorGameRule | None:
    try:
        panel_count = len(extract_snapshot(env.driver)["panels"])
        return SplendorGameRule(panel_count) if panel_count else None
    except Exception:  # logging must never break the game
        return None


def run_game(
    env: BrowserSplendorEnv,
    q_net: QNetwork,
    max_steps: int = 200,
    *,
    game_index: int = 1,
    total_games: int = 1,
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
    my_index = my_seat - 1
    label = f"对局 {game_index}/{total_games}"
    print(f"▶ {label} 开始 | 我的座位 {my_seat}", flush=True)

    rule = _build_rule(env)
    terminated = False
    for step in range(1, max_steps + 1):
        snapshot = extract_snapshot(env.driver)
        scores_line = _format_scores(snapshot, my_seat)
        action_idx, q_value, description, top = _select_action(
            env, q_net, obs, snapshot, rule, my_index, steps
        )
        anomalies += _print_parity(env)
        _print_step(label, step, description, q_value, scores_line, top)

        obs, reward, terminated, _truncated, _info = env.step(action_idx)
        steps += 1
        telescoped_score += float(reward)
        print(f"   奖励(Δscore): {reward:+.0f}", flush=True)

        snapshot_scores = _panel_scores(env, my_seat)
        if snapshot_scores is not None:
            saw_board = True
            my_score, rival_score = snapshot_scores
        if terminated:
            break
    else:
        print(f"⚠ {label} 达到步数上限，按中止处理", flush=True)

    if not saw_board:
        my_score = telescoped_score

    result = _judge_result(terminated, my_score, rival_score)
    duration = time.monotonic() - start
    result_cn = {"win": "胜", "loss": "负", "draw": "平", "aborted": "中止"}[result]
    print(
        f"🏁 {label}: {result_cn} | 我 {my_score:.1f} vs 对手最佳 "
        f"{rival_score:.1f} | {steps} 步 | {duration:.0f}s | 掩码异常 {anomalies}",
        flush=True,
    )
    return GameReport(
        result=result,
        my_score=my_score,
        rival_score=rival_score,
        steps=steps,
        duration=duration,
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
    print(
        f"✔ 模型已加载: {options['checkpoint']}"
        f"（feature_version={q_net.feature_version}）",
        flush=True,
    )

    driver = options["driver_factory"]()
    session = SessionManager(
        driver, room_url=options["room_url"], seats=options["seats"]
    )
    env = BrowserSplendorEnv(
        driver, session, poll_interval=options["poll"], step_timeout=options["timeout"],
        feature_version=q_net.feature_version,
    )
    print(
        f"▶ 开始部署 {options['games']} 局 | 座位数 {options['seats']}"
        + (f" | 钉定房间 {options['room_url']}" if options["room_url"] else ""),
        flush=True,
    )

    reports: list[GameReport] = []
    for game_index in range(options["games"]):
        n = game_index + 1
        try:
            report = run_game(
                env, q_net,
                max_steps=options["max_steps"],
                game_index=n,
                total_games=options["games"],
            )
        except Exception as error:  # one network hiccup must not kill the run
            print(f"⚠ 对局 {n}/{options['games']} 失败（{error}）；recover 一次", flush=True)
            session.recover()
            try:
                report = run_game(
                    env, q_net,
                    max_steps=options["max_steps"],
                    game_index=n,
                    total_games=options["games"],
                )
            except Exception as error:
                print(f"⚠ 对局 {n}/{options['games']} recover 后仍中止: {error}", flush=True)
                report = GameReport(
                    result="aborted", my_score=0.0, rival_score=0.0,
                    steps=0, duration=0.0, mask_anomalies=0,
                )
        if session.room_url:
            print(f"🔗 房间: {session.room_url}  <- 打开可旁观", flush=True)
        reports.append(report)
        if game_index < options["games"] - 1:
            rest = random.uniform(*REST_SECONDS)
            print(f"⏸ 休息 {rest:.1f}s（礼仪）", flush=True)
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
    print(f"📄 战报已保存: {out_path}", flush=True)
    print(
        f"📊 汇总: 胜 {summary['win_rate']:.0%} / 平 {summary['draw_rate']:.0%} / "
        f"均分 {summary['avg_score']:.2f} / 掩码异常 {summary['total_mask_anomalies']}",
        flush=True,
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
