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
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import load_saved_dqn
from splendor.browser.dom_extractor import validate_snapshot
from splendor.splendor.gym.envs.actions import ALL_ACTIONS

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
from .rollout import DEFAULT_MAX_STEPS, WinRateEstimator


class InferenceServer:
    """TCP JSONL server: ping / act / winrate over a checkpoint registry."""

    def __init__(
        self,
        models: dict[str, QNetwork],
        host: str = "0.0.0.0",
        port: int = 8765,
        default_rollouts: int = 16,
        default_max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._models = models
        self._host = host
        self._port = port
        self._default_rollouts = default_rollouts
        self._default_max_steps = default_max_steps
        self._estimators = {
            name: WinRateEstimator(model) for name, model in models.items()
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
                    {
                        "id": name,
                        "feature_version": model.feature_version,
                        "input_dim": model.input_dim,
                    }
                    for name, model in sorted(self._models.items())
                ],
                device="cuda" if self._any_cuda() else "cpu",
            )
        if op == OP_ACT:
            return self._handle_act(request_id, request)
        if op == OP_WINRATE:
            return await self._handle_winrate(request_id, request)
        raise ProtocolError(f"unknown op {op!r}")

    def _handle_act(self, request_id: int, request: dict[str, Any]) -> dict[str, Any]:
        require(request, "model_id", "obs", "mask", what="act request")
        model = self._model(request["model_id"])
        obs = np.asarray(request["obs"], dtype=np.float32)
        mask = np.asarray(request["mask"], dtype=np.float32)
        if obs.shape != (model.input_dim,):
            raise ProtocolError(
                f"obs shape {obs.shape} != ({model.input_dim},)"
            )
        if mask.shape != (len(ALL_ACTIONS),):
            raise ProtocolError(
                f"mask shape {mask.shape} != ({len(ALL_ACTIONS)},)"
            )
        action = model.act(torch.from_numpy(obs), torch.from_numpy(mask))
        return make_response(request_id, action=int(action))

    async def _handle_winrate(
        self, request_id: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        require(request, "model_id", "snapshot", "actor_seat", what="winrate request")
        estimator = self._estimators[request["model_id"]]  # KeyError -> error frame
        snapshot = request["snapshot"]
        validate_snapshot(snapshot)
        with self._estimate_lock:
            self._estimate_serial += 1
            seed = 20260910 + self._estimate_serial
            result = await asyncio.to_thread(
                estimator.estimate,
                snapshot,
                int(request["actor_seat"]),
                int(request.get("n_rollouts", self._default_rollouts)),
                int(request.get("max_steps", self._default_max_steps)),
                seed,
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
    def _model(self, model_id: str) -> QNetwork:
        if model_id not in self._models:
            raise ProtocolError(
                f"unknown model_id {model_id!r} (available: {sorted(self._models)})"
            )
        return self._models[model_id]

    def _any_cuda(self) -> bool:
        return any(
            next(model.parameters()).is_cuda for model in self._models.values()
        )

    async def start(
        self, host: str = "127.0.0.1", port: int = 0
    ) -> asyncio.AbstractServer:
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


def load_models(specs: list[str], models_dir: Path | None) -> dict[str, QNetwork]:
    """
    Model registry from ``name=path`` specs and/or a directory of checkpoints
    (name = file stem, e.g. ``round2.pth`` -> ``round2``).
    """
    models: dict[str, QNetwork] = {}
    if models_dir is not None:
        for path in sorted(models_dir.glob("*.pth")):
            models[path.stem] = load_saved_dqn(path)
    for spec in specs or []:
        name, _, path_text = spec.partition("=")
        if not name or not path_text:
            raise ValueError(f"--model expects name=path, got {spec!r}")
        models[name] = load_saved_dqn(Path(path_text))
    if not models:
        raise ValueError("no models registered (use --model name=path / --models-dir)")
    return models


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
    options = parser.parse_args()

    models = load_models(options.model, options.models_dir)
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
