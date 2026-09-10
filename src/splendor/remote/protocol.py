"""
Wire format for the local<->remote inference protocol (phase-6).

Framing: one UTF-8 JSON object per line (JSONL) over a stream socket.
A 265-d obs plus a 3510-d mask serialises to ~30 KB - trivially cheap on a
LAN, and staying human-readable means every message can be inspected with
``nc``/``tcpdump`` without a decoder. ``MAX_FRAME_BYTES`` is a run-away
guard, not a tuning knob.

Envelope: every request carries a client-chosen integer ``id``; the reply
echoes it, so a client may pipeline requests (the bot harness does not,
keeping failure attribution one-to-one).

* success: ``{"id": 1, "ok": true,  ...op-specific fields}``
* failure: ``{"id": 1, "ok": false, "error": "..."}``

Operations (request -> response payload):

* ``ping``    -> ``{"models": [...], "device": "cpu"}``
* ``act``     -> ``{"action": int}``           (obs 265-d + mask 3510-d in)
* ``winrate`` -> ``{"win_rates": [float per seat], "draw_rate": float, ...}``
"""

import json
from typing import Any

MAX_FRAME_BYTES = 16 * 1024 * 1024  # run-away guard; real frames are ~30 KB

OP_PING = "ping"
OP_ACT = "act"
OP_WINRATE = "winrate"


class ProtocolError(ValueError):
    """A frame violated the envelope contract (malformed JSON/envelope)."""


def encode_message(message: dict[str, Any]) -> bytes:
    """One JSON object + newline (the JSONL frame)."""
    return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")


def decode_message(raw: bytes) -> dict[str, Any]:
    """
    Parse one frame into an envelope dict.

    :raises ProtocolError: on undecodable JSON or a non-object frame.
    """
    try:
        message = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"undecodable frame: {error}") from error
    if not isinstance(message, dict):
        raise ProtocolError(f"frame must be a JSON object, got {type(message)}")
    return message


def make_request(request_id: int, op: str, **payload: Any) -> dict[str, Any]:  # noqa: ANN401 - JSON payload by design (see module doc)
    """Client-side request envelope."""
    return {"id": request_id, "op": op, **payload}


def make_response(request_id: int, **payload: Any) -> dict[str, Any]:  # noqa: ANN401 - JSON payload by design (see module doc)
    """Server-side success envelope (``ok: true``)."""
    return {"id": request_id, "ok": True, **payload}


def make_error(request_id: int, error: str) -> dict[str, Any]:
    """Server-side failure envelope (``ok: false``); never raises."""
    return {"id": request_id, "ok": False, "error": str(error)}


def require(message: dict[str, Any], *keys: str, what: str = "message") -> None:
    """
    Enforce required envelope keys server/client-side.

    :raises ProtocolError: naming the first missing key.
    """
    for key in keys:
        if key not in message:
            raise ProtocolError(f"{what} missing required field {key!r}")
