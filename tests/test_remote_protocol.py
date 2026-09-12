"""
Remote inference protocol tests (phase-6): JSONL framing + a real client
against an in-process server on loopback (no external network; the random
initial weights keep the deployment plumbing under test, not the policy).
"""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import MockBrowserDriver
from splendor.remote.client import InferenceClient, InferenceClientError
from splendor.remote.protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    make_error,
    make_request,
    make_response,
)
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


def test_unknown_model_is_an_error_frame(server_port: int) -> None:
    with InferenceClient("127.0.0.1", server_port) as client:
        with pytest.raises(InferenceClientError, match="unknown model_id"):
            client.act("nope", np.zeros(265), np.zeros(3510))


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
