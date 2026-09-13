"""
Local real-time dashboard (phase-6): watches the bots' JSONL event streams
and serves one HTML page - the per-seat win-rate line chart (before/after
every action), cumulative stats and the action/discard log feed.

stdlib-only on purpose: the dashboard runs on the deployment machine next
to the bot harness, and the project runtime has no web framework. The page
polls ``/api/state`` every 1.5s - trivially cheap for the event volumes of
a Splendor deployment and immune to SSE plumbing.

Server-side cost control: each ``bot*.jsonl`` is parsed once and reused
until its ``(mtime_ns, size)`` changes, so a 1.5s poll of a quiet
directory is a handful of ``stat`` calls, not a full re-read. Only the
latest ``remote_act`` keeps its Q-ranking payload; older ones are served
with ``top: []`` (the page never looks past the newest decision, and the
ranking lists are the fattest field in the stream).

Run: ``play-dashboard --events-dir web_events --port 8899`` then open
``http://localhost:8899``.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Full reload per poll is bounded by the cap; older events stop mattering
# once the chart window and the log feed are filled.
MAX_EVENTS_SERVED = 3000
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


class _EventStore:
    """Merged view over ``bot<i>.jsonl`` files + the published room URL."""

    def __init__(self, events_dir: Path) -> None:
        self._events_dir = events_dir
        # file name -> (mtime_ns, size, parsed events); invalidated on stat change
        self._file_cache: dict[str, tuple[int, int, list[dict[str, Any]]]] = {}
        self._html_cache: bytes | None = None
        self._html_mtime_ns: int = -1

    def dashboard_html(self) -> bytes:
        """Dashboard page bytes, re-read only when the file on disk changes."""
        mtime_ns = DASHBOARD_HTML.stat().st_mtime_ns
        if self._html_cache is None or mtime_ns != self._html_mtime_ns:
            self._html_cache = DASHBOARD_HTML.read_bytes()
            self._html_mtime_ns = mtime_ns
        return self._html_cache

    def snapshot(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        if self._events_dir.exists():
            for path in sorted(self._events_dir.glob("bot*.jsonl")):
                events.extend(self._read_cached(path))
        events.sort(key=lambda event: (event.get("ts", 0.0), event.get("bot", 0)))
        served = _prune_rankings(events)[-MAX_EVENTS_SERVED:]
        room_url = None
        room_file = self._events_dir / "room_url.txt"
        if room_file.exists():
            room_url = room_file.read_text(encoding="utf-8").strip() or None
        return {
            "events": served,
            "total_events": len(events),
            "room_url": room_url,
        }

    def _read_cached(self, path: Path) -> list[dict[str, Any]]:
        """Parse ``path`` once per ``(mtime_ns, size)``; torn lines are skipped."""
        try:
            stat = path.stat()
        except OSError:
            self._file_cache.pop(path.name, None)
            return []
        cached = self._file_cache.get(path.name)
        if (
            cached is not None
            and cached[0] == stat.st_mtime_ns
            and cached[1] == stat.st_size
        ):
            return cached[2]
        events: list[dict[str, Any]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue  # a torn last line (mid-write) is normal
        self._file_cache[path.name] = (stat.st_mtime_ns, stat.st_size, events)
        return events


def _prune_rankings(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep the Q-ranking only on the newest ``remote_act``; copy-on-write.

    The page's decision panel reads ``top`` from the latest remote decision
    alone. Every older ranking is pure payload weight on a 1.5s poll, so
    they are replaced by ``top: []``. Cached event dicts are shared across
    snapshots - never mutate them in place.
    """
    latest_index: int | None = None
    for index, event in enumerate(events):
        if event.get("type") == "remote_act":
            latest_index = index
    if latest_index is None:
        return events
    pruned: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if index != latest_index and event.get("type") == "remote_act" and event.get("top"):
            pruned.append({**event, "top": []})
        else:
            pruned.append(event)
    return pruned


def _make_handler(store: _EventStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path in {"/", "/index.html"}:
                self._serve_bytes(
                    store.dashboard_html(), "text/html; charset=utf-8"
                )
            elif path == "/api/state":
                payload = json.dumps(store.snapshot(), ensure_ascii=False)
                self._serve_bytes(payload.encode("utf-8"),
                                  "application/json; charset=utf-8")
            else:
                self.send_error(404)

        def _serve_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Local ops console: a refresh during a run must pick up HTML edits
            # and never serve a stale API payload from the browser cache.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # keep the console quiet under 1.5s polling

    return Handler


def main() -> None:
    """Entry point of the ``play-dashboard`` console script."""
    parser = argparse.ArgumentParser(
        prog="play-dashboard",
        description="Real-time win-rate dashboard over bot event streams.",
    )
    parser.add_argument("--events-dir", default="web_events")
    parser.add_argument("--port", type=int, default=8899)
    options = parser.parse_args()

    store = _EventStore(Path(options.events_dir))
    server = ThreadingHTTPServer(("0.0.0.0", options.port), _make_handler(store))
    print(f"[dashboard] http://localhost:{options.port} "
          f"(events: {Path(options.events_dir).resolve()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
