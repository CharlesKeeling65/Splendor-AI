"""
Offline tests for the advisor dashboard plumbing (plan phase-7 T7.6):
AdvisorStore thread-safe handoffs, the HTTP surface (stdlib server on an
ephemeral port) and the engine's parity report - no network beyond
loopback, no browser.
"""

import json
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from splendor.browser.advisor.engine import AdvisorEngine
from splendor.browser.advisor.server import AdvisorStore, start_server
from splendor.browser.advisor.tracker import ReservationTracker
from splendor.browser.dom_extractor import Snapshot, extract_snapshot
from splendor.browser.driver import MockBrowserDriver
from splendor.splendor.constants import NUMBER_OF_TIERS

REPO = Path(__file__).parent.parent
FIXTURES = REPO / "src" / "splendor" / "browser" / "fixtures"
OPENING = FIXTURES / "opening.html"


def _opening_snapshot() -> Snapshot:
    driver = MockBrowserDriver()
    driver.set_html(OPENING.read_text(encoding="utf-8"))
    return extract_snapshot(driver)


def _payload() -> dict[str, Any]:
    return {"frame_seq": 1, "status": "等待你操作", "advice": {"ga": []}}


# ----- store -------------------------------------------------------------------
def test_store_roundtrip_and_deep_flow() -> None:
    store = AdvisorStore()
    assert store.snapshot()["ready"] is False

    store.set_state(_payload())
    served = store.snapshot()
    assert served["ready"] is True
    assert served["status"] == "等待你操作"
    assert served["deep"]["rows"] == []  # deep envelope always present

    # Deep without a my-turn source -> honest error, never a crash.
    status = store.request_deep()
    assert status == {"queued": False, "reason": "no my-turn state yet"}

    store.set_deep_source(state=None, depth=2)
    assert store.request_deep() == {"queued": True}
    job = store.wait_deep_request(timeout=0.1)
    assert job is not None and job.depth == 2
    store.set_deep_error("暂无可深算的局面（等待我的回合）")
    served = store.snapshot()
    assert served["deep"]["error"].startswith("暂无可深算")


def test_store_deep_busy_and_reset_flag() -> None:
    store = AdvisorStore()
    store.set_deep_source(state=object(), depth=2)
    assert store.request_deep() == {"queued": True}
    assert store.request_deep() == {"queued": False, "reason": "deep already running"}

    assert store.pop_reset_request() is False
    store.reset_tracker_request()
    assert store.pop_reset_request() is True
    assert store.pop_reset_request() is False


# ----- HTTP surface ---------------------------------------------------------------
@pytest.fixture(name="server_url")
def server_url_fixture() -> Generator[str, None, None]:
    store = AdvisorStore()
    store.set_state(_payload())
    server = start_server(store, 0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_http_serves_dashboard_and_state(server_url: str) -> None:
    page = urlopen(f"{server_url}/", timeout=2).read().decode("utf-8")
    assert "璀璨宝石" in page and "/api/state" in page
    state = json.loads(urlopen(f"{server_url}/api/state", timeout=2).read())
    assert state["ready"] is True
    with pytest.raises(HTTPError):
        urlopen(f"{server_url}/api/nope", timeout=2)


def test_http_post_endpoints(server_url: str) -> None:
    response = urlopen(f"{server_url}/api/reset", timeout=2, data=b"")
    assert response.read() == b'{"ok": true}'
    deep = urlopen(f"{server_url}/api/deep", timeout=2, data=b"").read()
    assert b"queued" in deep


# ----- parity report -----------------------------------------------------------------
def test_parity_report_on_opening_snapshot() -> None:
    engine = AdvisorEngine(2, 0, seed=1)
    report = engine.parity_report(_opening_snapshot())
    assert report and report[0].startswith("mask parity: engine-legal=")
    # Direction-safe ruling: the engine mask is a subset of DOM affordances.
    assert any("OK" in line or "direction safe" in line for line in report)


def test_deck_hist_rows_match_tier_count() -> None:
    engine = AdvisorEngine(2, 0, seed=1)
    histogram = engine.deck_histogram(_opening_snapshot(), ReservationTracker())
    assert len(histogram.rows) == NUMBER_OF_TIERS
