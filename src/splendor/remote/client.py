"""
Synchronous inference client (phase-6): the local bot harness's view of the
remote server. Used from the bot loop (blocking request -> response), which
is fine because a bot acts at human pace and every request is sub-second
except ``winrate`` (seconds - still small against a turn).

Reconnect discipline: one transparent reconnect-and-retry per call - a
blipped Wi-Fi must not abort a deployment, but an infinite retry loop must
not silently wedge a bot either (the second failure surfaces to the caller,
which already has per-game recovery).
"""

import io
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import (
    MAX_FRAME_BYTES,
    OP_ACT,
    OP_PING,
    OP_WINRATE,
    decode_message,
    encode_message,
    make_request,
)

DEFAULT_TIMEOUT = 120.0
DEFAULT_TOP_K = 5


class InferenceClientError(RuntimeError):
    """The server answered with an error frame, or the link failed twice."""


@dataclass(frozen=True)
class ActDecision:
    """Greedy action plus the server's legal-action Q ranking."""

    action: int
    top: tuple[dict[str, Any], ...]  # [{"idx": int, "q": float}, ...]


class InferenceClient:
    """Blocking JSONL-over-TCP client; one connection, reconnect-once."""

    def __init__(self, host: str, port: int, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._file: io.BufferedReader | None = None
        self._next_id = 1

    # ----- public operations -----------------------------------------------------
    def ping(self) -> dict[str, Any]:
        """Health check; returns {"models": [...], "device": ...}."""
        return self._call(OP_PING)

    def act(
        self,
        model_id: str,
        obs: np.ndarray,
        mask: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ) -> ActDecision:
        """
        Greedy action for one observation under one legal mask.

        :returns: :class:`ActDecision` - the chosen ``ALL_ACTIONS`` index plus
                  the server's top-k legal Q ranking (for logging / dashboard).
        """
        response = self._call(
            OP_ACT,
            model_id=model_id,
            obs=np.asarray(obs, dtype=np.float32).tolist(),
            mask=np.asarray(mask, dtype=np.int64).tolist(),
            top_k=int(top_k),
        )
        action = response["action"]
        if not isinstance(action, int):
            raise InferenceClientError(f"malformed act response: {response!r}")
        raw_top = response.get("top") or []
        top: list[dict[str, Any]] = []
        for item in raw_top:
            if not isinstance(item, dict) or "idx" not in item or "q" not in item:
                raise InferenceClientError(f"malformed act ranking: {item!r}")
            top.append({"idx": int(item["idx"]), "q": float(item["q"])})
        return ActDecision(action=action, top=tuple(top))

    def estimate_winrate(
        self,
        model_id: str,
        snapshot: Mapping[str, Any],
        actor_seat: int,
        n_rollouts: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Monte-Carlo win-rate estimate of the current position (see
        ``rollout`` module for the definition). Returns a dict with
        ``win_rates`` (per page seat order), ``draw_rate``, ``rollouts``.
        """
        payload: dict[str, Any] = {
            "model_id": model_id,
            "snapshot": snapshot,
            "actor_seat": int(actor_seat),
        }
        if n_rollouts is not None:
            payload["n_rollouts"] = int(n_rollouts)
        if max_steps is not None:
            payload["max_steps"] = int(max_steps)
        return self._call(OP_WINRATE, **payload)

    def close(self) -> None:
        """Drop the connection (idempotent)."""
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._file = None

    def __enter__(self) -> "InferenceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- transport -----------------------------------------------------------
    def _call(self, op: str, **payload: Any) -> dict[str, Any]:  # noqa: ANN401
        request = make_request(self._next_id, op, **payload)
        self._next_id += 1
        try:
            return self._roundtrip(request)
        except OSError:
            self.close()  # stale link: reconnect once, then surface
            return self._roundtrip(request)

    def _roundtrip(self, request: dict[str, Any]) -> dict[str, Any]:
        sock = self._ensure_connected()
        sock.sendall(encode_message(request))
        reader = self._file
        assert reader is not None  # set together with the socket
        line = reader.readline(MAX_FRAME_BYTES + 1)
        if not line:
            raise OSError("server closed the connection")
        if len(line) > MAX_FRAME_BYTES:
            raise InferenceClientError("response frame exceeds MAX_FRAME_BYTES")
        response = decode_message(line)
        if response.get("id") != request["id"]:
            raise InferenceClientError(
                f"response id {response.get('id')!r} != request {request['id']}"
            )
        if not response.get("ok", False):
            raise InferenceClientError(
                f"server error: {response.get('error', 'unknown')}"
            )
        return response

    def _ensure_connected(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.create_connection((self._host, self._port), self._timeout)
        sock.settimeout(self._timeout)
        self._sock = sock
        self._file = sock.makefile("rb")
        return sock
