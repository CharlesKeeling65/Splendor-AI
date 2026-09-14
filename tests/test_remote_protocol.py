"""
Remote inference protocol tests (phase-6): JSONL framing + a real client
against an in-process server on loopback (no external network; the random
initial weights keep the deployment plumbing under test, not the policy).
"""

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.policy_imitation.ppo_selfplay import (
    PolicyValueNetwork,
)
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import MockBrowserDriver
from splendor.remote.client import InferenceClient, InferenceClientError
from splendor.remote.policies import ScoredPolicy
from splendor.remote.protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    make_error,
    make_request,
    make_response,
)
from splendor.remote.rollout import RolloutResult, WinRateEstimator
from splendor.remote.server import InferenceServer

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"


# ----- framing -------------------------------------------------------------------
def test_roundtrip() -> None:
    message = {"id": 7, "ok": True, "action": 42, "note": "中文"}
    assert decode_message(encode_message(message)) == message


def test_frame_ends_with_newline() -> None:
    assert encode_message({"id": 1}).endswith(b"\n")


def test_garbage_frame_raises() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b"not json\n")


def test_non_object_frame_raises() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b"[1, 2, 3]\n")


def test_error_envelope_shape() -> None:
    assert make_error(3, "boom") == {"id": 3, "ok": False, "error": "boom"}
    assert make_response(3, action=1)["id"] == 3
    assert make_request(3, "act")["op"] == "act"


# ----- live server on loopback -----------------------------------------------------
@pytest.fixture(scope="module")
def server_port() -> Iterator[int]:
    model = QNetwork(input_dim=265, output_dim=3510, feature_version="v1")
    model.eval()
    server = InferenceServer({"m1": model})
    loop = asyncio.new_event_loop()
    started = threading.Event()
    holder: dict[str, int] = {}

    def _run() -> None:
        asyncio.set_event_loop(loop)

        async def _start() -> None:
            raw = await server.start("127.0.0.1", 0)
            holder["port"] = raw.sockets[0].getsockname()[1]

        loop.run_until_complete(_start())
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait(10.0)
    yield holder["port"]
    loop.call_soon_threadsafe(loop.stop)


def _snapshot_of(fixture: str) -> Snapshot:
    driver = MockBrowserDriver()
    driver.set_html((FIXTURES / fixture).read_text(encoding="utf-8"))
    return extract_snapshot(driver)


def test_ping_lists_models(server_port: int) -> None:
    with InferenceClient("127.0.0.1", server_port) as client:
        reply = client.ping()
    assert [model["id"] for model in reply["models"]] == ["m1"]
    assert reply["models"][0]["feature_version"] == "v1"


def test_act_respects_the_mask(server_port: int) -> None:
    obs = np.zeros(265, dtype=np.float32)
    mask = np.zeros(3510, dtype=np.int64)
    mask[1234] = 1  # the only legal action must be the answer
    with InferenceClient("127.0.0.1", server_port) as client:
        decision = client.act("m1", obs, mask)
    assert decision.action == 1234
    assert decision.top and decision.top[0]["idx"] == 1234
    assert isinstance(decision.top[0]["q"], float)


def test_act_shape_violation_is_an_error_frame(server_port: int) -> None:
    with InferenceClient("127.0.0.1", server_port) as client:
        with pytest.raises(InferenceClientError, match="mask shape"):
            client.act("m1", np.zeros(265), np.zeros(10))


@pytest.mark.parametrize(
    "mask,error",
    [
        (np.zeros(3510), "at least one legal action"),
        (np.full(3510, 0.5), "0/1"),
        (np.full(3510, np.nan), "non-finite"),
    ],
)
def test_act_rejects_bad_masks(mask: np.ndarray, error: str) -> None:
    """The protocol boundary rejects masks before model execution."""
    server = _mixed_server()
    with pytest.raises(ProtocolError, match=error):
        server._handle_act(  # noqa: SLF001 - direct boundary validation seam
            1,
            {
                "model_id": "dqn",
                "obs": np.zeros(265, dtype=np.float32).tolist(),
                "mask": mask.tolist(),
            },
        )


def test_unknown_model_is_an_error_frame(server_port: int) -> None:
    with InferenceClient("127.0.0.1", server_port) as client:
        with pytest.raises(InferenceClientError, match="unknown model_id"):
            client.act("nope", np.zeros(265), np.zeros(3510))


def _mixed_server() -> InferenceServer:
    """Build a small mixed registry without loading external checkpoints."""
    dqn = QNetwork(input_dim=265, output_dim=3510, feature_version="v1")
    ppo = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    return InferenceServer(
        {
            "dqn": ScoredPolicy.from_dqn(dqn),
            "ppo": ScoredPolicy.from_ppo(ppo),
        }
    )


def test_ping_exposes_mixed_policy_metadata() -> None:
    """Mixed registries advertise the score semantics needed by clients."""
    response = asyncio.run(
        _mixed_server()._dispatch(  # noqa: SLF001 - direct dispatch validation seam
            {"id": 1, "op": "ping"}
        )
    )
    models = {model["id"]: model for model in response["models"]}
    assert models["dqn"]["kind"] == "dqn"
    assert models["dqn"]["score_kind"] == "q"
    assert models["ppo"]["kind"] == "imitation_ppo_policy_value"
    assert models["ppo"]["score_kind"] == "policy_logit"
    assert models["ppo"]["input_dim"] == 265
    assert models["ppo"]["output_dim"] == 3510
    assert models["dqn"]["device"] == "cpu"
    assert response["device"] == "cpu"


def test_ppo_act_matches_local_masked_argmax() -> None:
    """Remote PPO selection must match its local PolicyValueNetwork."""
    model = PolicyValueNetwork(265, feature_version="v1", hidden_layers=(8,))
    policy = ScoredPolicy.from_ppo(model)
    server = InferenceServer({"ppo": policy})
    obs = np.linspace(-1.0, 1.0, 265, dtype=np.float32)
    mask = np.zeros(3510, dtype=np.float32)
    mask[[19, 777, 2048]] = 1.0
    with torch.no_grad():
        expected = int(
            model(
                torch.from_numpy(obs), torch.from_numpy(mask)
            )[0].argmax(dim=-1).item()
        )
    response = server._handle_act(  # noqa: SLF001 - direct dispatch validation seam
        1, {"model_id": "ppo", "obs": obs.tolist(), "mask": mask.tolist()}
    )
    assert response["action"] == expected
    assert response["score_kind"] == "policy_logit"
    assert response["top"]
    assert all("score" in item and "q" in item for item in response["top"])
    assert all(mask[item["idx"]] == 1 for item in response["top"])


@pytest.mark.parametrize("top_k", [True, 1.5, 3511])
def test_act_rejects_non_integer_or_oversized_top_k(top_k: object) -> None:
    """Ranking requests have a strict integer bound at the server boundary."""
    server = _mixed_server()
    obs = np.zeros(265, dtype=np.float32)
    mask = np.zeros(3510, dtype=np.float32)
    mask[0] = 1.0
    with pytest.raises(ProtocolError, match="top_k"):
        server._handle_act(  # noqa: SLF001 - direct dispatch validation seam
            1,
            {
                "model_id": "dqn",
                "obs": obs.tolist(),
                "mask": mask.tolist(),
                "top_k": top_k,
            },
        )


@pytest.mark.parametrize("field", ["n_rollouts", "max_steps"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_winrate_options_require_positive_json_int(
    field: str, value: object
) -> None:
    """Invalid rollout budgets fail before a worker thread is scheduled."""
    server = InferenceServer(
        {"m1": QNetwork(input_dim=265, output_dim=3510, feature_version="v1")}
    )
    request: dict[str, object] = {
        "model_id": "m1",
        "snapshot": _snapshot_of("opening.html"),
        "actor_seat": 1,
        field: value,
    }
    with pytest.raises(ProtocolError, match=field):
        asyncio.run(
            server._handle_winrate(  # noqa: SLF001 - direct dispatch validation seam
                1, request
            )
        )


class _SlowEstimator:
    """Thread-observable estimator used to test the async lock boundary."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def estimate(
        self,
        snapshot: Snapshot,
        actor_seat: int,
        n_rollouts: int,
        max_steps: int,
        seed: int,
    ) -> RolloutResult:
        del snapshot, actor_seat, n_rollouts, max_steps, seed
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.15)
        with self._state_lock:
            self.active -= 1
        return RolloutResult([1.0, 0.0], 0.0, 1, 0, 0.15)


def test_winrate_lock_wait_does_not_block_event_loop() -> None:
    """A queued winrate must not prevent ping while another is running."""
    server = InferenceServer(
        {"m1": QNetwork(input_dim=265, output_dim=3510, feature_version="v1")}
    )
    estimator = _SlowEstimator()
    server._estimators["m1"] = cast(  # noqa: SLF001 - concurrency seam
        WinRateEstimator, estimator
    )
    snapshot = _snapshot_of("opening.html")
    request = {
        "model_id": "m1",
        "snapshot": snapshot,
        "actor_seat": 1,
        "n_rollouts": 1,
        "max_steps": 1,
    }

    async def exercise() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        first = asyncio.create_task(
            server._handle_winrate(  # noqa: SLF001 - direct concurrency seam
                1, request
            )
        )
        await asyncio.sleep(0.03)
        second = asyncio.create_task(
            server._handle_winrate(  # noqa: SLF001 - direct concurrency seam
                2, request
            )
        )
        await asyncio.sleep(0.03)
        ping = await asyncio.wait_for(
            server._dispatch(  # noqa: SLF001 - direct concurrency seam
                {"id": 3, "op": "ping"}
            ),
            timeout=0.1,
        )
        results = list(await asyncio.gather(first, second))
        return ping, results

    ping, results = asyncio.run(exercise())
    assert ping["ok"] is True
    assert len(results) == 2
    assert estimator.max_active == 1


def test_winrate_smoke(server_port: int) -> None:
    snapshot = _snapshot_of("opening.html")
    with InferenceClient("127.0.0.1", server_port) as client:
        reply = client.estimate_winrate(
            "m1", snapshot, actor_seat=snapshot["my_seat"] or 1,
            n_rollouts=2, max_steps=50,
        )
    assert len(reply["win_rates"]) == 2
    assert all(0.0 <= rate <= 1.0 for rate in reply["win_rates"])
    assert reply["rollouts"] == 2


def test_winrate_rejects_bad_actor_seat(server_port: int) -> None:
    snapshot = _snapshot_of("opening.html")
    with InferenceClient("127.0.0.1", server_port) as client:
        with pytest.raises(InferenceClientError):
            client.estimate_winrate("m1", snapshot, actor_seat=9)
