"""
Local dashboard for the browser advisor (plan phase-7 §3.5).

stdlib-only on purpose (mirrors ``remote/dashboard.py``): the dashboard
runs next to the advisor on the player's machine and the runtime has no
web framework. The page polls ``/api/state`` every second; the deep-mode
button posts ``/api/deep`` and the result lands in the same payload.

Thread model: the poll thread calls :meth:`AdvisorStore.set_state` with a
fresh JSON-ready dict each adopted frame; the deep worker publishes via
:meth:`AdvisorStore.set_deep_result`. Both only swap references, guarded
by one lock.
"""

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from splendor.splendor.splendor_model import SplendorState

DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


@dataclass
class DeepJob:
    """The latest my-turn reconstruction, waiting for a deep request."""

    state: SplendorState | None = None  # latest my-turn reconstruction
    depth: int = 2


@dataclass
class AdvisorStore:
    """Shared state between the poll thread, the deep worker and HTTP."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _state: dict[str, Any] | None = None
    _deep_source: DeepJob | None = None
    _deep_running: bool = False
    _deep_result: dict[str, Any] | None = None
    _deep_error: str | None = None
    _reset_requested: bool = False
    _deep_wakeup: threading.Event = field(default_factory=threading.Event)

    # ----- poll thread -----------------------------------------------------
    def set_state(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._state = payload

    def set_deep_source(self, state: SplendorState, depth: int) -> None:
        """Publish the latest my-turn reconstruction for deep requests."""
        with self._lock:
            self._deep_source = DeepJob(state=state, depth=depth)

    def pop_reset_request(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    # ----- HTTP thread -------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state) if self._state is not None else None
            deep = dict(self._deep_result) if self._deep_result else None
        if state is None:
            return {"ready": False, "deep": self._deep_envelope(deep)}
        state["ready"] = True
        state["deep"] = self._deep_envelope(deep)
        return state

    @staticmethod
    def _deep_envelope(deep: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "running": False,
            "error": None,
            "rows": [],
            "elapsed_ms": None,
            **(deep or {}),
        }

    def request_deep(self) -> dict[str, Any]:
        """Queue a deep computation; returns a small status dict."""
        with self._lock:
            if self._deep_running:
                return {"queued": False, "reason": "deep already running"}
            if self._deep_source is None:
                return {"queued": False, "reason": "no my-turn state yet"}
            self._deep_running = True
            self._deep_error = None
            self._deep_wakeup.set()
            return {"queued": True}

    def reset_tracker_request(self) -> None:
        with self._lock:
            self._reset_requested = True

    # ----- deep worker -------------------------------------------------------
    def wait_deep_request(self, timeout: float) -> DeepJob | None:
        """Block until a deep request arrives (or timeout); None on timeout."""
        if not self._deep_wakeup.wait(timeout):
            return None
        with self._lock:
            self._deep_wakeup.clear()
            job = self._deep_source
            return job

    def set_deep_result(
        self, rows: list[dict[str, Any]], elapsed_ms: float | None = None
    ) -> None:
        with self._lock:
            self._deep_running = False
            self._deep_result = {
                "running": False,
                "error": None,
                "rows": rows,
                "elapsed_ms": elapsed_ms,
            }

    def set_deep_error(self, message: str) -> None:
        with self._lock:
            self._deep_running = False
            self._deep_error = message
            self._deep_result = {
                "running": False,
                "error": message,
                "rows": [],
                "elapsed_ms": None,
            }


def make_handler(store: AdvisorStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path in {"/", "/index.html"}:
                self._serve_bytes(
                    DASHBOARD_HTML.read_bytes(), "text/html; charset=utf-8"
                )
            elif path == "/api/state":
                payload = json.dumps(store.snapshot(), ensure_ascii=False)
                self._serve_bytes(
                    payload.encode("utf-8"), "application/json; charset=utf-8"
                )
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/deep":
                payload = json.dumps(store.request_deep(), ensure_ascii=False)
                self._serve_bytes(
                    payload.encode("utf-8"), "application/json; charset=utf-8"
                )
            elif path == "/api/reset":
                store.reset_tracker_request()
                self._serve_bytes(b'{"ok": true}', "application/json; charset=utf-8")
            else:
                self.send_error(404)

        def _serve_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Local ops console: never serve a stale API payload.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # keep the console quiet under 1s polling

    return Handler


def start_server(store: AdvisorStore, port: int) -> ThreadingHTTPServer:
    """Bind the dashboard server; caller runs ``serve_forever`` in a thread."""
    return ThreadingHTTPServer(("0.0.0.0", port), make_handler(store))
