"""
Local real-time dashboard (phase-6): watches the bots' JSONL event streams
and serves one HTML page - the per-seat win-rate line chart (before/after
every action), cumulative stats and the action/discard log feed.

stdlib-only on purpose: the dashboard runs on the deployment machine next
to the bot harness, and the project runtime has no web framework. The page
polls ``/api/state`` (full event list, capped) every 1.5s - trivially cheap
for the event volumes of a Splendor deployment and immune to SSE plumbing.

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

    def snapshot(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        if self._events_dir.exists():
            for path in sorted(self._events_dir.glob("bot*.jsonl")):
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        events.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue  # a torn last line (mid-write) is normal
        events.sort(key=lambda event: (event.get("ts", 0.0), event.get("bot", 0)))
        room_url = None
        room_file = self._events_dir / "room_url.txt"
        if room_file.exists():
            room_url = room_file.read_text(encoding="utf-8").strip() or None
        return {
            "events": events[-MAX_EVENTS_SERVED:],
            "total_events": len(events),
            "room_url": room_url,
        }


def _make_handler(store: _EventStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._serve_bytes(
                    DASHBOARD_HTML.read_bytes(), "text/html; charset=utf-8"
                )
            elif self.path == "/api/state":
                payload = json.dumps(store.snapshot(), ensure_ascii=False)
                self._serve_bytes(payload.encode("utf-8"), "application/json")
            else:
                self.send_error(404)

        def _serve_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
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
