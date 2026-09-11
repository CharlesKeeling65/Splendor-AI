"""
``play_vs_humans``: one AI seat (corrected-42 checkpoint) versus three human
seats in a pinned game.hullqin.cn/ccbs room.

Usage (from the repo root, ego-browser CLI logged in)::

    .venv/bin/python -m splendor.play_vs_humans --room-url <房间URL>

Flow: the humans create a 4-seat room and sit seats 1-3 (one of them is the
owner who presses 开始游戏); this process joins the first free seat, waits as
long as the humans need (turn timeout is hours by default - human thinking is
never rushed), and on its own turn runs the actor checkpoint's greedy policy
through the same BrowserSplendorEnv seam as ``play-web``.

Live reporting: every own decision is logged with the action description, Q
value and the top alternatives; while humans think, a throttled reporter
watches the polled snapshots and prints/appends every seat's current score
and a real-time win-probability estimate. Win probabilities come from a
second checkpoint (corrected-1234, the evaluator): for each seat we build the
pseudo state from that seat's perspective, mask its legal actions and read
max-Q as the expected *remaining* score gain, so
``estimate_i = score_i + max(Q_i, 0)`` and the four estimates go through a
softmax (temperature tunable). It is a heuristic, not a calibrated equity -
good enough for trend watching, not for claims of optimality.

Logs: one JSONL file per game (every event: my actions, waiting-state
analyses, score changes, final result) plus a summary JSON in the log dir.
"""

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.dom_extractor import (
    Snapshot,
    extract_snapshot,
    waiting_seat,
)
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.browser.monitor import anomaly_count
from splendor.browser.session import SessionManager
from splendor.browser.state_builder import build_pseudo_state
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import Card, SplendorGameRule

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ACTOR = (
    _REPO_ROOT / "runs/round2-ema-guidance-20260907/corrected-42/step-15000.pth"
)
DEFAULT_EVALUATOR = (
    _REPO_ROOT / "runs/round2-ema-guidance-20260907/corrected-1234/step-15000.pth"
)
DEFAULT_LOG_DIR = _REPO_ROOT / "runs/vs_humans"

MIN_PLAYERS = 2  # below this the page is not an in-game view

COLOR_CN: dict[str, str] = {
    "white": "白",
    "blue": "蓝",
    "green": "绿",
    "red": "红",
    "black": "黑",
    "yellow": "金",
}


# ---------------------------------------------------------------------------
# Human-readable rendering of engine actions (the dict format of ActionType)
# ---------------------------------------------------------------------------
def _gems_cn(counts: Mapping[str, int] | None) -> str:
    """``{'blue': 1, 'red': 2}`` -> ``蓝x1/红x2`` (zero counts dropped)."""
    if not counts:
        return ""
    return "/".join(
        f"{COLOR_CN.get(str(colour), str(colour))}x{count}"
        for colour, count in counts.items()
        if count
    )


def _card_cn(card: Card | None) -> str:
    tier = getattr(card, "deck_id", None)
    colour = COLOR_CN.get(str(getattr(card, "colour", "")), "?")
    points = getattr(card, "points", 0)
    tier_cn = f"{tier + 1}级" if isinstance(tier, int) else ""
    return f"{tier_cn}{colour}卡" + (f"{points}分" if points else "")


def describe_action(action: Mapping[str, Any]) -> str:
    """One-line Chinese description of an engine-format action dict."""
    atype = str(action.get("type", "?"))
    if atype in ("collect_diff", "collect_same"):
        text = f"拿取宝石 {_gems_cn(action.get('collected_gems'))}"
    elif atype == "reserve":
        yellow = (action.get("collected_gems") or {}).get("yellow", 0)
        text = f"预留 {_card_cn(action.get('card'))}" + ("(+1金)" if yellow else "")
    elif atype in ("buy_available", "buy_reserve"):
        where = "桌面" if atype == "buy_available" else "预留"
        text = f"购买{where} {_card_cn(action.get('card'))}"
        payment = _gems_cn(action.get("returned_gems"))
        if payment:
            text += f"(支付 {payment})"
    elif atype == "pass":
        text = "跳过"
    else:
        text = atype
    if action.get("noble"):
        text += ",并获得贵族"
    return text


# ---------------------------------------------------------------------------
# Logging plumbing
# ---------------------------------------------------------------------------
class JsonlLogger:
    """Append-only JSONL event log, flushed per record (crash-safe)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        record["wall_time"] = datetime.now().isoformat(timespec="seconds")
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass
class SeatEstimate:
    """One seat's live standing (score + Q-based projected win probability)."""

    seat: int
    score: int
    estimate: float  # projected final score
    win_prob: float
    source: str  # "q" = evaluator Q-max, "score" = fallback (no legal actions)


# ---------------------------------------------------------------------------
# Win-probability analysis (evaluator checkpoint, per-seat perspective)
# ---------------------------------------------------------------------------
class SeatAnalyzer:
    """
    Real-time standings for every seat, from the evaluator checkpoint.

    The rule instance is rebuilt whenever the observed player count changes
    (the engine rule itself is stateless for ``getLegalActions`` purposes);
    per-seat evaluation failures degrade to a pure score fallback instead of
    killing the reporting loop.
    """

    def __init__(
        self,
        evaluator: QNetwork,
        feature_version: str,
        temperature: float = 2.0,
    ) -> None:
        self._evaluator = evaluator
        self._feature_version = feature_version
        self._temperature = temperature
        self._rule: SplendorGameRule | None = None
        self._rule_player_count = 0

    def analyse(self, snapshot: Snapshot) -> list[SeatEstimate]:
        """Project every seat's final score and turn them into win probs."""
        panels = snapshot["panels"]
        count = len(panels)
        if count != self._rule_player_count:
            self._rule = SplendorGameRule(count)
            self._rule_player_count = count

        estimates: list[SeatEstimate] = []
        for index, panel in enumerate(panels):
            estimate = float(panel["score"])
            source = "score"
            try:
                assert self._rule is not None
                pseudo = build_pseudo_state(snapshot, index)
                legal = self._rule.getLegalActions(pseudo, index)
                if legal:
                    mask = create_legal_actions_mask(legal, pseudo, index)
                    obs = extract_observation(pseudo, index, self._feature_version)
                    q = self._evaluator(torch.from_numpy(obs), torch.from_numpy(mask))
                    estimate += max(float(q.max().item()), 0.0)
                    source = "q"
            except Exception as error:  # reporting never kills the game loop
                print(f"[analyzer] seat {index + 1} fallback: {error}", flush=True)
            estimates.append(
                SeatEstimate(
                    seat=panel["seat"],
                    score=panel["score"],
                    estimate=estimate,
                    win_prob=0.0,
                    source=source,
                )
            )

        values = np.array([item.estimate for item in estimates], dtype=np.float64)
        exponent = (values - values.max()) / self._temperature
        probs = np.exp(exponent)
        probs /= probs.sum()
        for item, prob in zip(estimates, probs, strict=True):
            item.win_prob = float(prob)
        return estimates


# ---------------------------------------------------------------------------
# Live reporter: fired from the env's wait loops (throttled)
# ---------------------------------------------------------------------------
class LiveReporter:
    """
    Snapshot listener for the long human turns.

    Called on every env poll; the expensive part (per-seat analysis) runs at
    most every ``report_interval`` seconds. All failures are swallowed here -
    a reporting glitch must never break the turn-waiting loop.
    """

    def __init__(
        self,
        analyzer: SeatAnalyzer,
        logger: JsonlLogger,
        report_interval: float = 15.0,
    ) -> None:
        self._analyzer = analyzer
        self._logger = logger
        self._report_interval = report_interval
        self._last_report = 0.0
        self._last_scores: dict[int, int] = {}
        self.latest_estimates: list[SeatEstimate] = []

    def last_scores(self) -> dict[int, int]:
        """Newest seat->score map seen while waiting (covers rival turns)."""
        return dict(self._last_scores)

    def on_snapshot(self, snapshot: Snapshot) -> None:
        try:
            self._handle(snapshot)
        except Exception as error:  # reporting is best-effort only
            print(f"[reporter] skipped a snapshot: {error}", flush=True)

    def _handle(self, snapshot: Snapshot) -> None:
        now = time.monotonic()
        if now - self._last_report < self._report_interval:
            return
        self._last_report = now

        panels = snapshot["panels"]
        in_game = bool(snapshot["deck_counts"]) and any(snapshot["deck_counts"])
        status = snapshot["status"] or "(等待开局)"
        seat_scores = {panel["seat"]: panel["score"] for panel in panels}
        scores_changed = seat_scores != self._last_scores
        self._last_scores = seat_scores

        if not in_game or len(panels) < MIN_PLAYERS:
            print(f"⏳ {status}(等待房主开始游戏)", flush=True)
            self._logger.write({"event": "waiting_start", "status": status})
            return

        self.latest_estimates = self._analyzer.analyse(snapshot)
        actor_seat = waiting_seat(status)
        headline = f"⏳ {status} | " + _format_estimates(self.latest_estimates)
        print(headline, flush=True)
        self._logger.write(
            {
                "event": "waiting",
                "status": status,
                "acting_seat": actor_seat,
                "scores": seat_scores,
                "estimates": [asdict(item) for item in self.latest_estimates],
                "scores_changed": scores_changed,
            }
        )


def _format_estimates(estimates: list[SeatEstimate]) -> str:
    parts = [f"座{item.seat}={item.score}分({item.win_prob:.1%})" for item in estimates]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# My own decision point
# ---------------------------------------------------------------------------
def select_action(  # noqa: PLR0913 - logging payload needs them
    env: BrowserSplendorEnv,
    actor: QNetwork,
    obs: NDArray,
    snapshot: Snapshot,
    rule: SplendorGameRule,
    my_index: int,
    my_turns: int,
) -> tuple[int, float, str, list[tuple[int, float, str]]]:
    """
    Greedy action from the actor checkpoint plus a human-readable account.

    The env's engine-rule mask is the legality authority (and feeds the parity
    monitor); the description mapping is rebuilt on my own pseudo state purely
    for logging, so a page race can only degrade the description, not the
    choice.
    """
    mask = env.get_legal_actions_mask()
    q_values = actor(
        torch.from_numpy(obs), torch.from_numpy(mask.astype(np.float32))
    ).squeeze(0)
    action_idx = int(q_values.argmax().item())

    description = "(动作描述不可用)"
    top: list[tuple[int, float, str]] = []
    try:
        pseudo = build_pseudo_state(snapshot, my_index, turns=my_turns)
        legal = rule.getLegalActions(pseudo, my_index)
        mapping = create_action_mapping(legal, pseudo, my_index)
        if action_idx in mapping:
            description = describe_action(mapping[action_idx])
        ranked = sorted(mapping, key=lambda idx: float(q_values[idx]), reverse=True)
        top = [
            (idx, float(q_values[idx]), describe_action(mapping[idx]))
            for idx in ranked[:3]
        ]
    except Exception as error:  # description is best-effort only
        print(f"[describe] fallback: {error}", flush=True)
    return action_idx, float(q_values[action_idx]), description, top


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------
def run_game(  # noqa: PLR0913 - one glue point per concern
    env: BrowserSplendorEnv,
    actor: QNetwork,
    analyzer: SeatAnalyzer,
    reporter: LiveReporter,
    logger: JsonlLogger,
    max_steps: int,
) -> dict[str, Any]:
    """Play one web game; returns the summary dict (also written to disk)."""
    start = time.monotonic()
    obs, info = env.reset()
    my_seat = int(info["my_id"])
    my_index = my_seat - 1
    print(f"✔ AI 已入座:座位 {my_seat}(其余为真人席位)", flush=True)

    rule = SplendorGameRule(len(extract_snapshot(env.driver)["panels"]))
    history: list[dict[str, Any]] = []
    anomalies = 0
    terminated = False
    my_turns = 0

    for step in range(1, max_steps + 1):
        snapshot = extract_snapshot(env.driver)
        estimates = analyzer.analyse(snapshot)
        action_idx, q_value, description, top = select_action(
            env, actor, obs, snapshot, rule, my_index, my_turns
        )
        anomalies += anomaly_count(getattr(env, "last_parity_report", []))

        print(
            f"▶ 第 {step} 步 | 我执行: {description} | Q={q_value:.2f}",
            flush=True,
        )
        print(f"   实时胜率: {_format_estimates(estimates)}", flush=True)
        alternatives = ", ".join(f"{text}({value:.2f})" for _, value, text in top)
        if alternatives:
            print(f"   备选: {alternatives}", flush=True)

        logger.write(
            {
                "event": "action",
                "step": step,
                "my_seat": my_seat,
                "action_idx": action_idx,
                "action_desc": description,
                "q_value": q_value,
                "top_alternatives": [
                    {"idx": idx, "q": value, "desc": text} for idx, value, text in top
                ],
                "estimates": [asdict(item) for item in estimates],
                "mask_anomalies": anomaly_count(
                    getattr(env, "last_parity_report", [])
                ),
            }
        )
        history.append(
            {
                "step": step,
                "action": description,
                "estimates": [asdict(e) for e in estimates],
            }
        )

        obs, reward, terminated, _truncated, _info = env.step(action_idx)
        my_turns += 1
        print(f"   奖励(Δscore): {reward:+.0f}", flush=True)

        if terminated:
            break
    else:
        print("⚠ 达到步数上限,按中止处理", flush=True)

    # Final standings: prefer the last in-game snapshot; the reporter's live
    # score tracking covers rivals' turns after my last observation.
    final = reporter.last_scores()
    final_snapshot = extract_snapshot(env.driver)
    if len(final_snapshot["panels"]) >= MIN_PLAYERS:
        final = {p["seat"]: p["score"] for p in final_snapshot["panels"]}
    result, ranking = _judge(final, my_seat)

    duration = time.monotonic() - start
    print(
        f"🏁 终局: {result} | 最终比分 {final} | 用时 {duration / 60:.1f} 分钟 "
        f"| 掩码异常 {anomalies}",
        flush=True,
    )
    summary = {
        "result": result,
        "my_seat": my_seat,
        "final_scores": final,
        "ranking": ranking,
        "steps": my_turns,
        "duration_seconds": duration,
        "mask_anomalies": anomalies,
        "win_prob_history": history,
    }
    logger.write(
        {
            "event": "game_over",
            **{k: v for k, v in summary.items() if k != "win_prob_history"},
        }
    )
    return summary


def _judge(final: dict[int, int], my_seat: int) -> tuple[str, list[tuple[int, int]]]:
    """(result, ranking) from the last observed scores; ties keep seat order."""
    if not final:
        return "unknown", []
    ranking = sorted(final.items(), key=lambda item: item[1], reverse=True)
    best = max(final.values())
    winners = [seat for seat, score in final.items() if score == best]
    result = (
        "win" if winners == [my_seat] else ("draw" if my_seat in winners else "loss")
    )
    return result, ranking


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point (``python -m splendor.play_vs_humans``)."""
    options = _parse_args()

    actor = load_saved_dqn(options["actor"])
    evaluator = load_saved_dqn(options["evaluator"])
    if actor.feature_version != evaluator.feature_version:
        raise SystemExit(
            "actor 与 evaluator 的 feature_version 不一致 "
            f"({actor.feature_version} vs {evaluator.feature_version})"
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = options["log_dir"] / f"session-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(log_dir / "game-1.jsonl")
    logger.write(
        {
            "event": "session_start",
            "room_url": options["room_url"],
            "actor": str(options["actor"]),
            "evaluator": str(options["evaluator"]),
            "temperature": options["temperature"],
        }
    )

    analyzer = SeatAnalyzer(evaluator, actor.feature_version, options["temperature"])
    reporter = LiveReporter(analyzer, logger, options["report_interval"])

    driver = EgoBrowserDriver(options["task_space"], room_url=options["room_url"])
    session = SessionManager(driver, room_url=options["room_url"], seats=4)
    env = BrowserSplendorEnv(
        driver,
        session,
        poll_interval=options["poll"],
        step_timeout=options["turn_timeout"],
        feature_version=actor.feature_version,
        snapshot_listener=reporter.on_snapshot,
    )

    print(f"▶ 加入房间: {options['room_url']}(AI 自动坐第一个空位)", flush=True)
    try:
        summary = run_game(env, actor, analyzer, reporter, logger, options["max_steps"])
    except TimeoutError as error:
        print(f"⚠ 等待超时(真人长时间未操作?): {error}", flush=True)
        logger.write({"event": "timeout", "detail": str(error)})
        summary = {"result": "aborted"}
    finally:
        logger.close()

    summary_path = log_dir / "game-1-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"📄 日志已保存: {logger.path}", flush=True)
    print(f"📄 战报已保存: {summary_path}", flush=True)


def _parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="play_vs_humans",
        description="1 个 AI 席位(corrected-42)对战 3 名真人的网页部署脚本。",
    )
    parser.add_argument("--room-url", required=True, help="真人创建好的房间 URL。")
    parser.add_argument(
        "--actor",
        type=Path,
        default=DEFAULT_ACTOR,
        help="出牌策略 checkpoint(默认 corrected-42 step-15000)。",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=DEFAULT_EVALUATOR,
        help="胜率评估 checkpoint(默认 corrected-1234 step-15000)。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="日志目录(默认 runs/vs_humans)。",
    )
    parser.add_argument(
        "--task-space",
        default="splendor-vs-humans",
        help="ego-browser task space 名称。",
    )
    parser.add_argument("--poll", type=float, default=1.0, help="回合轮询间隔(秒)。")
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=3600.0,
        help="等待人类一轮操作的超时(秒),默认 1 小时。",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=15.0,
        help="等待期间的实时分析输出间隔(秒)。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="胜率 softmax 温度(越小越两极分化)。",
    )
    parser.add_argument(
        "--max-steps", type=int, default=300, help="单局 AI 最大出手次数。"
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    main()
