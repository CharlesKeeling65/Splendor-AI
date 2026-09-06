"""Small dependency-free web monitor for DQN training runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

JsonObject = dict[str, Any]
MAX_ROWS = 4_000
STALE_AFTER_SECONDS = 90


def _coerce(value: str | None) -> object:
    """Convert CSV scalars to JSON-friendly values."""
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _read_rows(path: Path, limit: int = MAX_ROWS) -> list[JsonObject]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as csv_file:
            rows = [
                {key: _coerce(value) for key, value in row.items()}
                for row in csv.DictReader(csv_file)
            ]
    except (OSError, UnicodeError, csv.Error):
        return []
    return rows[-limit:]


def _read_json(path: Path) -> JsonObject:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _run_mtime(run_dir: Path) -> float:
    return max(
        _mtime(run_dir / name)
        for name in (
            "progress.csv",
            "stats.csv",
            "run_config.json",
            "run_status.json",
        )
    )


def find_latest_run(runs_dir: Path) -> Path | None:
    """Find the most recently updated DQN run below ``runs_dir``."""
    if not runs_dir.is_dir():
        return None
    candidates: set[Path] = set()
    for pattern in ("progress.csv", "stats.csv", "run_config.json"):
        candidates.update(path.parent for path in runs_dir.rglob(pattern))
    return max(candidates, key=_run_mtime) if candidates else None


def _status(run_dir: Path) -> JsonObject:
    stored = _read_json(run_dir / "run_status.json")
    stored_status = stored.get("status")
    if stored_status in {"completed", "failed"}:
        return stored
    if stored_status == "running":
        last_update = _run_mtime(run_dir)
        if last_update and time.time() - last_update <= STALE_AFTER_SECONDS:
            return stored
        return {**stored, "status": "stale"}

    if (run_dir / "models" / "dqn_model.pth").is_file():
        return {"status": "completed"}

    last_update = _run_mtime(run_dir)
    if last_update and time.time() - last_update <= STALE_AFTER_SECONDS:
        return {"status": "running"}
    return {"status": "stale" if last_update else "missing"}


def load_payload(run_dir: Path | None) -> JsonObject:
    """Load one monitor snapshot, tolerating files being written live."""
    if run_dir is None or not run_dir.is_dir():
        return {
            "run_dir": None,
            "status": {"status": "missing"},
            "config": {},
            "progress": [],
            "stats": [],
        }

    progress = _read_rows(run_dir / "progress.csv")
    stats = _read_rows(run_dir / "stats.csv")
    if not progress:
        # Older runs only have stats.csv.  Normalize its names so the same
        # dashboard remains useful when inspecting a pre-monitor checkpoint.
        progress = [
            {
                **row,
                "event": "episode",
                "eval_win": row.get("eval_wr"),
                "eval_avg_score": row.get("eval_avg_score"),
            }
            for row in stats
        ]
    return {
        "run_dir": str(run_dir),
        "status": _status(run_dir),
        "config": _read_json(run_dir / "run_config.json"),
        "progress": progress,
        "stats": stats,
    }


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Splendor-AI DQN Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101419;
      --panel: #171d24;
      --panel-2: #1d2630;
      --line: #30404d;
      --text: #e7edf2;
      --muted: #9aa9b5;
      --cyan: #50d2dc;
      --blue: #7fa8ff;
      --amber: #f7c66b;
      --green: #79d49b;
      --red: #ff8f8f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    .shell { max-width: 1440px; margin: 0 auto; padding: 28px 30px 50px; }
    .topbar { display: flex; justify-content: space-between; gap: 24px; align-items: end; }
    .eyebrow { color: var(--cyan); letter-spacing: .12em; text-transform: uppercase; font-size: 11px; }
    h1, h2, p { margin: 0; }
    h1 { font-size: 28px; font-weight: 650; letter-spacing: -.02em; margin-top: 5px; }
    h2 { font-size: 15px; font-weight: 600; margin-bottom: 14px; }
    .subtitle, .path, .muted { color: var(--muted); }
    .subtitle { margin-top: 7px; }
    .path { margin-top: 12px; font: 12px ui-monospace, SFMono-Regular, monospace; overflow-wrap: anywhere; }
    .status-wrap { text-align: right; min-width: 150px; }
    .status { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 5px 11px; text-transform: uppercase; letter-spacing: .08em; font-size: 11px; }
    .status[data-state="running"] { color: var(--green); border-color: #35694b; }
    .status[data-state="completed"] { color: var(--cyan); border-color: #2c6972; }
    .status[data-state="stale"], .status[data-state="missing"] { color: var(--amber); border-color: #6b5427; }
    .updated { color: var(--muted); font-size: 12px; margin-top: 8px; }
    .notice { min-height: 24px; color: var(--amber); margin: 20px 0 8px; }
    .metrics { display: grid; grid-template-columns: repeat(7, minmax(110px, 1fr)); gap: 10px; margin: 20px 0; }
    .metric, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .metric { padding: 14px 15px; min-height: 84px; }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { font-size: 22px; font-variant-numeric: tabular-nums; margin-top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .panel { padding: 18px; min-width: 0; }
    .wide { grid-column: 1 / -1; }
    canvas { display: block; width: 100%; height: 245px; }
    .config { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 0 22px; }
    .config-item { padding: 9px 0; border-bottom: 1px solid #26323d; min-width: 0; }
    .config-key { color: var(--muted); font-size: 12px; }
    .config-value { margin-top: 3px; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, monospace; }
    .events { display: grid; gap: 7px; }
    .event { display: grid; grid-template-columns: 100px 80px 1fr; gap: 12px; color: var(--muted); font: 12px ui-monospace, SFMono-Regular, monospace; }
    .event strong { color: var(--text); font-weight: 500; }
    @media (max-width: 1050px) { .metrics { grid-template-columns: repeat(4, 1fr); } .config { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 700px) { .shell { padding: 22px 16px 35px; } .topbar { display: block; } .status-wrap { text-align: left; margin-top: 18px; } .metrics, .grid { grid-template-columns: 1fr 1fr; } .wide { grid-column: auto; } .config { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div>
        <div class="eyebrow">Splendor-AI / DQN</div>
        <h1>训练过程监控</h1>
        <p class="subtitle">每 2 秒刷新一次; 数据来自训练目录中的 progress.csv 和 run_config.json。</p>
        <p class="path" id="run-path">正在查找训练目录…</p>
      </div>
      <div class="status-wrap">
        <div class="status" id="status" data-state="missing">MISSING</div>
        <div class="updated" id="updated">等待数据</div>
      </div>
    </header>

    <div class="notice" id="notice" role="status" aria-live="polite"></div>

    <section class="metrics" aria-label="训练摘要">
      <div class="metric"><div class="metric-label">训练步数</div><div class="metric-value" id="step">—</div></div>
      <div class="metric"><div class="metric-label">回合数</div><div class="metric-value" id="episode">—</div></div>
      <div class="metric"><div class="metric-label">Epsilon</div><div class="metric-value" id="epsilon">—</div></div>
      <div class="metric"><div class="metric-label">评估胜率</div><div class="metric-value" id="eval-win">—</div></div>
      <div class="metric"><div class="metric-label">评估平均分</div><div class="metric-value" id="eval-score">—</div></div>
      <div class="metric"><div class="metric-label">吞吐 steps/s</div><div class="metric-value" id="throughput">—</div></div>
      <div class="metric"><div class="metric-label">GPU 显存 MB</div><div class="metric-value" id="gpu-memory">—</div></div>
    </section>

    <div class="grid">
      <section class="panel"><h2>优化过程 · loss / Q / TD</h2><canvas id="optimization-chart" aria-label="loss, q mean and TD error over training"></canvas></section>
      <section class="panel"><h2>评估结果 · win rate / average score</h2><canvas id="evaluation-chart" aria-label="evaluation win rate and average score over training"></canvas></section>
      <section class="panel"><h2>探索过程 · epsilon</h2><canvas id="exploration-chart" aria-label="epsilon over training"></canvas></section>
      <section class="panel"><h2>最近事件</h2><div class="events" id="events"><span class="muted">等待训练事件…</span></div></section>
      <section class="panel wide"><h2>运行参数与环境</h2><div class="config" id="config"><span class="muted">等待配置…</span></div></section>
    </div>
  </div>

  <script>
    let lastData = null;
    const colors = { loss: "#50d2dc", q_mean: "#7fa8ff", target_q_mean: "#b38cff", td_abs_mean: "#f7c66b", td_abs_p90: "#ff8f8f", eval_win: "#79d49b", eval_avg_score: "#ff9f7a", epsilon: "#c59cff" };
    const $ = (id) => document.getElementById(id);
    const num = (value) => { const n = Number(value); return Number.isFinite(n) ? n : null; };
    const fmt = (value, digits = 2) => { const n = num(value); return n === null ? "—" : n.toFixed(digits); };
    const text = (id, value) => { $(id).textContent = value; };

    function drawChart(id, rows, definitions) {
      const canvas = $(id);
      const width = Math.max(300, canvas.clientWidth || 700);
      const height = 245;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio; canvas.height = height * ratio;
      const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const pad = { left: 46, right: 14, top: 26, bottom: 30 };
      const series = definitions.map((definition) => ({
        ...definition,
        points: rows.map((row) => [num(row.step), num(row[definition.key])]).filter((point) => point[0] !== null && point[1] !== null)
      })).filter((definition) => definition.points.length > 0);
      if (!series.length) {
        ctx.fillStyle = "#9aa9b5"; ctx.font = "13px ui-sans-serif"; ctx.fillText("等待训练数据…", pad.left, height / 2);
        return;
      }
      const allPoints = series.flatMap((definition) => definition.points);
      let xmin = Math.min(...allPoints.map((point) => point[0])), xmax = Math.max(...allPoints.map((point) => point[0]));
      let ymin = Math.min(...allPoints.map((point) => point[1])), ymax = Math.max(...allPoints.map((point) => point[1]));
      if (xmin === xmax) xmax = xmin + 1;
      if (ymin === ymax) { ymin -= 1; ymax += 1; }
      const x = (value) => pad.left + ((value - xmin) / (xmax - xmin)) * (width - pad.left - pad.right);
      const y = (value) => height - pad.bottom - ((value - ymin) / (ymax - ymin)) * (height - pad.top - pad.bottom);
      ctx.strokeStyle = "#2b3945"; ctx.lineWidth = 1; ctx.font = "11px ui-monospace"; ctx.fillStyle = "#81909c";
      for (let i = 0; i <= 4; i++) {
        const yy = pad.top + i * (height - pad.top - pad.bottom) / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(width - pad.right, yy); ctx.stroke();
        const label = ymax - i * (ymax - ymin) / 4; ctx.fillText(label.toFixed(2), 3, yy + 4);
      }
      ctx.fillText(String(Math.round(xmin)), pad.left, height - 8); ctx.textAlign = "right"; ctx.fillText(String(Math.round(xmax)), width - pad.right, height - 8); ctx.textAlign = "left";
      for (const definition of series) {
        ctx.strokeStyle = definition.color; ctx.lineWidth = 2; ctx.beginPath();
        definition.points.forEach((point, index) => { const px = x(point[0]), py = y(point[1]); if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
        ctx.stroke();
      }
      let legendX = pad.left;
      for (const definition of series) {
        ctx.fillStyle = definition.color; ctx.fillRect(legendX, 8, 14, 3); ctx.fillText(definition.label, legendX + 19, 13); legendX += definition.label.length * 7 + 45;
      }
    }

    function renderConfig(config) {
      const container = $("config"); container.replaceChildren();
      const keys = Object.keys(config).sort();
      if (!keys.length) { container.innerHTML = '<span class="muted">尚未发现 run_config.json</span>'; return; }
      for (const key of keys) {
        const item = document.createElement("div"); item.className = "config-item";
        const name = document.createElement("div"); name.className = "config-key"; name.textContent = key;
        const value = document.createElement("div"); value.className = "config-value"; value.textContent = typeof config[key] === "object" ? JSON.stringify(config[key]) : String(config[key]);
        item.append(name, value); container.append(item);
      }
    }

    function renderEvents(rows) {
      const container = $("events"); container.replaceChildren();
      const recent = rows.slice(-10).reverse();
      if (!recent.length) { container.innerHTML = '<span class="muted">等待训练事件…</span>'; return; }
      for (const row of recent) {
        const item = document.createElement("div"); item.className = "event";
        item.innerHTML = `<strong>${row.event || "event"}</strong><span>step ${row.step ?? "—"}</span><span>${row.timestamp || ""}</span>`;
        container.append(item);
      }
    }

    function render(data) {
      lastData = data;
      const config = data.config || {}, status = data.status || {}, rows = data.progress && data.progress.length ? data.progress : (data.stats || []);
      const latest = rows.length ? rows[rows.length - 1] : {};
      const step = status.step ?? latest.step;
      const total = num(config.total_steps ?? status.total_steps);
      const stepText = step === undefined || step === null ? "—" : `${step}${total ? ` / ${total}` : ""}`;
      text("step", stepText); text("episode", status.episode ?? latest.episode ?? "—"); text("epsilon", fmt(latest.epsilon, 3));
      text("eval-win", fmt(latest.eval_win, 2)); text("eval-score", fmt(latest.eval_avg_score, 2)); text("throughput", fmt(latest.steps_per_sec, 1)); text("gpu-memory", fmt(latest.gpu_memory_mb, 1));
      const state = status.status || "missing"; const statusElement = $("status"); statusElement.dataset.state = state; statusElement.textContent = state.toUpperCase();
      text("run-path", data.run_dir || "尚未发现训练目录"); text("updated", status.updated_at || latest.timestamp || "等待数据");
      $("notice").textContent = data.run_dir ? (rows.length ? "监控已连接; 训练运行时页面会自动更新。" : "已连接训练目录, 等待第一条进度事件。") : "请先启动 dqn, 或用 --runs-dir 指向训练输出目录。";
      drawChart("optimization-chart", rows, [{ key: "loss", label: "loss", color: colors.loss }, { key: "q_mean", label: "q mean", color: colors.q_mean }, { key: "target_q_mean", label: "target q", color: colors.target_q_mean }, { key: "td_abs_mean", label: "td abs", color: colors.td_abs_mean }, { key: "td_abs_p90", label: "td p90", color: colors.td_abs_p90 }]);
      drawChart("evaluation-chart", rows.filter((row) => row.event === "eval" || num(row.eval_win) !== null), [{ key: "eval_win", label: "win rate", color: colors.eval_win }, { key: "eval_avg_score", label: "avg score", color: colors.eval_avg_score }]);
      drawChart("exploration-chart", rows, [{ key: "epsilon", label: "epsilon", color: colors.epsilon }]);
      renderEvents(rows); renderConfig(config);
    }

    async function refresh() {
      try {
        const response = await fetch(`/api/data?now=${Date.now()}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        render(await response.json());
      } catch (error) {
        $("notice").textContent = `监控读取失败: ${error.message}`;
      }
    }
    window.addEventListener("resize", () => { if (lastData) render(lastData); });
    refresh(); setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def _handler(resolve_run: Callable[[], Path | None]) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        """Serve the dashboard and a live JSON snapshot."""

        server_version = "SplendorDQNMonitor/1.0"

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route in {"/", "/index.html"}:
                payload = PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/api/data":
                payload = json.dumps(
                    load_payload(resolve_run()), ensure_ascii=False
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format_string: str, *args: object) -> None:
            del format_string, args

    return MonitorHandler


def main() -> None:
    """Run the local DQN monitor server."""
    parser = argparse.ArgumentParser(description="Monitor Splendor-AI DQN runs")
    parser.add_argument(
        "--run",
        type=Path,
        help="Exact training directory; otherwise monitor the newest run",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing timestamped DQN run directories",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    exact_run = args.run.resolve() if args.run is not None else None
    runs_dir = args.runs_dir.resolve()

    def resolve_run() -> Path | None:
        return exact_run if exact_run is not None else find_latest_run(runs_dir)

    server = ThreadingHTTPServer((args.host, args.port), _handler(resolve_run))
    print(
        f"DQN monitor: http://{args.host}:{args.port} "
        f"(runs-dir: {runs_dir})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
