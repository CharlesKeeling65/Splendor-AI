"""
Remote inference: the local/remote split seam for browser deployment.

The browser control plane (DOM extraction, executor, session, parity
monitor) must run on the machine that owns the browser; model inference
(checkpoints + Monte-Carlo win-rate rollouts) belongs on a GPU/CPU server.
This package implements the TCP JSONL protocol between the two halves
(plan/phase-6-remote-inference.md):

* :mod:`.protocol` - message framing and schema helpers (zero dependency);
* :mod:`.client`   - synchronous client used by the local bot harness;
* :mod:`.server`   - asyncio server hosting the checkpoint registry and the
  batched Monte-Carlo win-rate estimator (:mod:`.rollout`).
"""

from .client import InferenceClient
from .protocol import (
    MAX_FRAME_BYTES,
    decode_message,
    encode_message,
    make_error,
    make_request,
    make_response,
)

__all__ = [
    "MAX_FRAME_BYTES",
    "InferenceClient",
    "decode_message",
    "encode_message",
    "make_error",
    "make_request",
    "make_response",
]
