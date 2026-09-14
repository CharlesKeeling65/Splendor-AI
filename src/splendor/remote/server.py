"""
Remote inference server (phase-6): hosts DQN checkpoints + the Monte-Carlo
win-rate estimator behind the JSONL protocol, on the inference machine
(e.g. Z8). The browser control plane never touches this module.

Concurrency: one asyncio task per connection (multi-bot = several bot
processes, each with its own connection). ``act`` is a single small forward
- handled inline; ``winrate`` is seconds of CPU - handed to a worker thread
under a lock, because the engine paths consume the global RNG streams and
one estimate must not interleave with another (AGENTS.md fact 6 discipline).

Run: ``inference-server --model round2=runs/.../dqn_model.pth [--model ...]``
"""

import argparse
import asyncio
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.policy_imitation.bc_training import DeviceName
from splendor.browser.dom_extractor import Snapshot, validate_snapshot

from .policies import ScoredPolicy, load_policies
from .protocol import (
    MAX_FRAME_BYTES,
    OP_ACT,
    OP_PING,
    OP_WINRATE,
    ProtocolError,
    decode_message,
    encode_message,
    make_error,
    make_response,
    require,
)
from .rollout import DEFAULT_MAX_STEPS, RolloutResult, WinRateEstimator

DEFAULT_TOP_K = 5


class InferenceServer:
    """TCP JSONL server: ping / act / winrate over a checkpoint registry."""

    def __init__(  # noqa: PLR0913 - registry + tunables, one construction site
        self,
        models: Mapping[str, ScoredPolicy | QNetwork],
        host: str = "0.0.0.0",
        port: int = 8765,
        default_rollouts: int = 16,
        default_max_steps: int = DEFAULT_MAX_STEPS,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._models = {
            name: _as_scored_policy(model) for name, model in models.items()
        }
        self._host = host
        self._port = port
        self._default_rollouts = _positive_int(default_rollouts, "default_rollouts")
        self._default_max_steps = _positive_int(
            default_max_steps, "default_max_steps"
        )
        self._default_top_k = _positive_int(default_top_k, "default_top_k")
        self._estimators = {
            name: WinRateEstimator(model) for name, model in self._models.items()
        }
        # The estimator (and engine deal paths) consume the global RNG; the
        # lock serialises estimates across worker threads.
        self._estimate_lock = threading.Lock()
        self._estimate_serial = 0

    # ----- connection handling ---------------------------------------------------
    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        print(f"[inference] client connected: {peer}", flush=True)
        try:
            while line := await reader.readline():
                await self._handle_line(line, writer)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass  # client hangup is a normal lifecycle event
        except Exception as error:  # never crash the server on one frame
            print(f"[inference] connection error {peer}: {error}", file=sys.stderr)
        finally:
            writer.close()
            print(f"[inference] client disconnected: {peer}", flush=True)

    async def _handle_line(
        self, line: bytes, writer: asyncio.StreamWriter
    ) -> None:
        if len(line) > MAX_FRAME_BYTES:
            raise ProtocolError("frame exceeds MAX_FRAME_BYTES")
        request: dict[str, Any] = {}
        try:
            request = decode_message(line)
            require(request, "id", "op", what="request")
            response = await self._dispatch(request)
        except Exception as error:
            request_id = request.get("id", -1)
            response = make_error(request_id, f"{type(error).__name__}: {error}")
        writer.write(encode_message(response))
        await writer.drain()

    # ----- operations --------------------------------------------------------------
    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request["id"]
        op = request["op"]
        if op == OP_PING:
            return make_response(
                request_id,
                models=[
                    model.metadata(name)
                    for name, model in sorted(self._models.items())
                ],
                # Keep the historical top-level field for old clients.  The
                # per-model metadata is authoritative for mixed registries.
                device=self._device_label(),
            )
        if op == OP_ACT:
            return self._handle_act(request_id, request)
        if op == OP_WINRATE:
            return await self._handle_winrate(request_id, request)
        raise ProtocolError(f"unknown op {op!r}")

    def _handle_act(self, request_id: int, request: dict[str, Any]) -> dict[str, Any]:
        require(request, "model_id", "obs", "mask", what="act request")
        policy = self._model(request["model_id"])
        try:
            obs = np.asarray(request["obs"], dtype=np.float32)
            mask = np.asarray(request["mask"], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ProtocolError(
                f"act inputs cannot be converted to arrays: {error}"
            ) from error
        if obs.shape != (policy.input_dim,):
            raise ProtocolError(
                f"obs shape {obs.shape} != ({policy.input_dim},)"
            )
        if mask.shape != (policy.output_dim,):
            raise ProtocolError(
                f"mask shape {mask.shape} != ({policy.output_dim},)"
            )
        _validate_act_arrays(obs, mask)
        top_k = _positive_int(request.get("top_k", self._default_top_k), "top_k")
        if top_k > policy.output_dim:
            raise ProtocolError(
                f"top_k must be <= {policy.output_dim}, got {top_k}"
            )

        scores = policy.scores(obs, mask)
        if scores.ndim != 1 or scores.shape != (policy.output_dim,):
            raise ProtocolError(
                f"policy scores shape {tuple(scores.shape)} != "
                f"({policy.output_dim},)"
            )
        score_array = _scores_to_numpy(scores)
        legal_indices = np.flatnonzero(mask == 1)
        if not len(legal_indices):  # defensive; _validate_act_arrays checks this
            raise ProtocolError("act mask must contain at least one legal action")
        legal_scores = score_array[legal_indices]
        action = int(legal_indices[int(np.argmax(legal_scores))])
        order = np.argsort(-legal_scores, kind="stable")[
            : min(top_k, len(legal_indices))
        ]
        top = [
            {
                "idx": int(legal_indices[position]),
                "score": float(legal_scores[position]),
                # ``q`` is a compatibility alias for the pre-PPO client and
                # dashboard.  score_kind is the semantic source of truth.
                "q": float(legal_scores[position]),
            }
            for position in order
        ]
        return make_response(
            request_id,
            action=action,
            top=top,
            kind=policy.kind,
            score_kind=policy.score_kind,
        )

    async def _handle_winrate(
        self, request_id: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        require(request, "model_id", "snapshot", "actor_seat", what="winrate request")
        model_id = request["model_id"]
        self._model(model_id)
        estimator = self._estimators[model_id]
        snapshot = request["snapshot"]
        validate_snapshot(snapshot)
        actor_seat = _positive_int(request["actor_seat"], "actor_seat")
        n_rollouts = _positive_int(
            request.get("n_rollouts", self._default_rollouts), "n_rollouts"
        )
        max_steps = _positive_int(
            request.get("max_steps", self._default_max_steps), "max_steps"
        )
        result = await asyncio.to_thread(
            self._estimate_serialized,
            estimator,
            snapshot,
            actor_seat,
            n_rollouts,
            max_steps,
        )
        return make_response(
            request_id,
            win_rates=result.win_rates,
            draw_rate=result.draw_rate,
            rollouts=result.rollouts,
            aborted=result.aborted,
            elapsed_s=round(result.elapsed_s, 2),
        )

    # ----- helpers -------------------------------------------------------------------
    def _model(self, model_id: str) -> ScoredPolicy:
        if model_id not in self._models:
            raise ProtocolError(
                f"unknown model_id {model_id!r} (available: {sorted(self._models)})"
            )
        return self._models[model_id]

    def _estimate_serialized(
        self,
        estimator: WinRateEstimator,
        snapshot: Snapshot,
        actor_seat: int,
        n_rollouts: int,
        max_steps: int,
    ) -> RolloutResult:
        """Run one RNG-consuming estimate while keeping lock waits off-loop."""
        with self._estimate_lock:
            self._estimate_serial += 1
            seed = 20260910 + self._estimate_serial
            return estimator.estimate(
                snapshot, actor_seat, n_rollouts, max_steps, seed
            )

    def _device_label(self) -> str:
        devices = {str(policy.device) for policy in self._models.values()}
        if len(devices) == 1:
            return next(iter(devices))
        return "mixed"

    async def start(
        self, host: str = "127.0.0.1", port: int = 0
    ) -> asyncio.Server:
        """
        Bind and return the raw asyncio server (port 0 = ephemeral). Tests
        use this against loopback; production entry is :meth:`serve`.
        """
        return await asyncio.start_server(
            self.handle, host, port, limit=MAX_FRAME_BYTES
        )

    async def serve(self) -> None:
        server = await self.start(self._host, self._port)
        print(
            f"[inference] serving on {self._host}:{self._port} "
            f"models={sorted(self._models)}",
            flush=True,
        )
        async with server:
            await server.serve_forever()


def load_models(
    specs: list[str],
    models_dir: Path | None,
    *,
    device_name: DeviceName = "cpu",
) -> dict[str, ScoredPolicy]:
    """
    Model registry from ``name=path`` specs and/or a directory of checkpoints
    (name = file stem, e.g. ``round2.pth`` -> ``round2``).
    """
    return load_policies(specs or [], models_dir, device_name=device_name)


def _as_scored_policy(model: ScoredPolicy | QNetwork) -> ScoredPolicy:
    """Keep direct-QNetwork construction compatible with phase-6 callers."""
    if isinstance(model, ScoredPolicy):
        return model
    if isinstance(model, QNetwork):
        return ScoredPolicy.from_dqn(model)
    raise TypeError(f"unsupported remote policy {type(model).__name__}")


def _validate_act_arrays(obs: np.ndarray, mask: np.ndarray) -> None:
    """Reject malformed protocol arrays before invoking a model."""
    if not np.isfinite(obs).all():
        raise ProtocolError("obs contains non-finite values")
    if not np.isfinite(mask).all():
        raise ProtocolError("mask contains non-finite values")
    if not np.isin(mask, (0.0, 1.0)).all():
        raise ProtocolError("mask must contain only 0/1 values")
    if not np.any(mask == 1):
        raise ProtocolError("mask must contain at least one legal action")


def _scores_to_numpy(scores: torch.Tensor) -> np.ndarray:
    """Detach a policy result from any accelerator for protocol ranking."""
    values = scores.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ProtocolError("policy produced non-finite action scores")
    return values.numpy()


def _positive_int(value: object, name: str) -> int:
    """Parse a strictly positive integer request option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be a positive integer")
    result = value
    if result <= 0:
        raise ProtocolError(f"{name} must be a positive integer, got {result}")
    return result


def main() -> None:
    """Entry point of the ``inference-server`` console script."""
    parser = argparse.ArgumentParser(
        prog="inference-server",
        description="Serve DQN act/winrate for the browser deployment (phase-6).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model", action="append", default=[],
        help="Register one checkpoint as name=path (repeatable).",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None,
        help="Register every *.pth in a directory (name = file stem).",
    )
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument("--max-rollout-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default="cpu",
        help="Inference device; unavailable accelerators safely fall back to CPU.",
    )
    options = parser.parse_args()

    models = load_models(
        options.model, options.models_dir, device_name=options.device
    )
    server = InferenceServer(
        models,
        host=options.host,
        port=options.port,
        default_rollouts=options.n_rollouts,
        default_max_steps=options.max_rollout_steps,
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("[inference] stopped", flush=True)


if __name__ == "__main__":
    main()
