"""Roadmap D2 tests: web replay mixing and parity-whitelisted harvesting."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import splendor.splendor.gym  # noqa: F401  # registers the gym env id
from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer
from splendor.agents.our_agents.dqn.training import collect_from_browser
from splendor.agents.our_agents.dqn.web_replay import (
    DEFAULT_WEB_RATIO,
    WebReplayMixer,
)
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.driver import SNAPSHOT_JS_MARKER, MockBrowserDriver
from splendor.browser.session import BASE_URL

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"
_OPENING = (FIXTURES / "opening.html").read_text(encoding="utf-8")


def _make_buffers(obs_dim: int = 265) -> tuple[ReplayBuffer, ReplayBuffer]:
    local = ReplayBuffer(capacity=512, obs_dim=obs_dim)
    web = ReplayBuffer(capacity=512, obs_dim=obs_dim)
    rng = np.random.default_rng(825_400)
    for _ in range(64):
        local.add(
            rng.random(obs_dim, dtype=np.float32),
            int(rng.integers(0, 100)),
            float(rng.random()),
            rng.random(obs_dim, dtype=np.float32),
            np.zeros(3510, dtype=np.float32),
            False,
        )
    for _ in range(16):
        web.add(
            rng.random(obs_dim, dtype=np.float32),
            int(rng.integers(0, 100)),
            float(rng.random()),
            rng.random(obs_dim, dtype=np.float32),
            np.zeros(3510, dtype=np.float32),
            False,
        )
    return local, web


def test_mixed_batch_respects_web_ratio() -> None:
    local, web = _make_buffers()
    mixer = WebReplayMixer(local, web, web_ratio=0.25)
    batch = mixer.sample(40)
    assert batch.actions.shape[0] == 40
    assert 0.0 < batch.web_share <= 0.25
    # web buffer only holds 16 transitions, so the share is bounded by data
    assert batch.web_share == pytest.approx(min(10, 16) / 40)
    assert torch.isfinite(batch.rewards).all()


def test_mixed_batch_falls_back_to_pure_local() -> None:
    local, web = _make_buffers()
    empty_web = ReplayBuffer(capacity=16, obs_dim=local.obs_dim)
    mixer = WebReplayMixer(local, empty_web, web_ratio=DEFAULT_WEB_RATIO)
    batch = mixer.sample(32)
    assert batch.web_share == 0.0
    assert batch.actions.shape[0] == 32
    del web


def test_mixer_validates_feature_schema_and_ratio() -> None:
    local, web = _make_buffers()
    wrong_web = ReplayBuffer(capacity=16, obs_dim=312)
    with pytest.raises(ValueError, match="feature schema mismatch"):
        WebReplayMixer(local, wrong_web)
    with pytest.raises(ValueError, match="web_ratio"):
        WebReplayMixer(local, web, web_ratio=0.0)
    with pytest.raises(ValueError, match="web_ratio"):
        WebReplayMixer(local, web, web_ratio=1.5)


def _strip_between(page: str, start_marker: str, end_marker: str) -> str:
    """Remove the block [first start_marker, first end_marker) - the fixture
    generator emits the rows and the supply row as contiguous blocks (the
    same slicing test_play_web.py uses)."""
    start = page.index(start_marker)
    end = page.index(end_marker, start)
    return page[:start] + page[end:]


_SCORED = _OPENING.replace(
    '<span class="ccbs-score">0分</span>',
    '<span class="ccbs-score">3分</span>',
    1,
).replace("等待你操作", "等待玩家2操作", 1)
_SCORED_MY_TURN = _SCORED.replace("等待玩家2操作", "等待你操作", 1)
_WAITING = _SCORED
_TERMINAL = _strip_between(
    _SCORED_MY_TURN,
    '<div class="flex justify-center origin-top">',
    '<div class="ccbs-noble',
)
_TERMINAL = _strip_between(
    _TERMINAL,
    '<div class="mt-4 flex items-center justify-center space-x-6">',
    '<div class="flex flex-wrap items-center justify-center my-2">',
)


class _RotatingDriver(MockBrowserDriver):
    """Serve a fixed page list per snapshot read, holding the last forever."""

    def __init__(self, pages: list[str]) -> None:
        super().__init__()
        self._pages = pages
        self._reads = 0

    def evaluate(self, js: str) -> object:
        if SNAPSHOT_JS_MARKER in js:
            index = min(self._reads, len(self._pages) - 1)
            self._reads += 1
            self.set_html(self._pages[index])
        return super().evaluate(js)


class _NoSession:
    def new_game(self) -> None:
        return None


def _harvest_env() -> BrowserSplendorEnv:
    # Read ledger mirrors test_play_web: several opening reads while the pass
    # policy acts, then the opponent turn, the +3 board, finally the room view
    # (game over) - so every harvest terminates regardless of read counts.
    driver = _RotatingDriver([_OPENING] * 40 + [_WAITING, _SCORED_MY_TURN, _TERMINAL])
    driver.set_html(_OPENING, url=f"{BASE_URL}/gt02")
    return BrowserSplendorEnv(
        driver,
        _NoSession(),  # type: ignore[arg-type]
        click_delay=(0, 0),
        poll_interval=0.0,
    )


class _PassPolicy:
    input_dim = 265
    output_dim = 3510

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        del obs
        return 0  # ALL_ACTIONS[0] = PASS

    def parameters(self) -> Iterator[torch.Tensor]:
        yield torch.zeros(1)


class _SilentMonitor:
    """Stand-in for a monitor whose whitelist covers every observed drift."""

    def check(self, engine_mask: NDArray, dom_affordances_set: set[int]) -> list[str]:
        del engine_mask, dom_affordances_set
        return []


def _silent_env() -> BrowserSplendorEnv:
    driver = _RotatingDriver([_OPENING] * 60 + [_WAITING, _SCORED_MY_TURN, _TERMINAL])
    driver.set_html(_OPENING, url=f"{BASE_URL}/gt02")
    return BrowserSplendorEnv(
        driver,
        _NoSession(),  # type: ignore[arg-type]
        click_delay=(0, 0),
        poll_interval=0.0,
        monitor=_SilentMonitor(),  # type: ignore[arg-type]
    )


def test_collect_from_browser_folds_clean_game() -> None:
    env = _silent_env()
    buffer = ReplayBuffer(capacity=256, obs_dim=265)
    stats = collect_from_browser(env, buffer, n_games=1, q_net=_PassPolicy())
    assert stats["games"] == 1.0
    assert stats["steps"] >= 1.0
    assert stats["anomalous_games_dropped"] == 0.0
    assert buffer.size == pytest.approx(stats["steps"])
    # the fixture grants +3 points on the scored board; the sum of panel
    # deltas telescopes to the final score
    assert stats["avg_score"] == pytest.approx(3.0)


def test_collect_from_browser_drops_parity_anomalous_game() -> None:
    class _AnomalousMonitor:
        def check(
            self, engine_mask: NDArray, dom_affordances_set: set[int]
        ) -> list[str]:
            del engine_mask, dom_affordances_set
            return ["DOM EXTRACTION BUG suspected: 3 engine-legal actions unsupported (test stub)"]

    driver = _RotatingDriver([_OPENING] * 40 + [_WAITING, _SCORED_MY_TURN, _TERMINAL])
    driver.set_html(_OPENING, url=f"{BASE_URL}/gt03")
    env = BrowserSplendorEnv(
        driver,
        _NoSession(),  # type: ignore[arg-type]
        click_delay=(0, 0),
        poll_interval=0.0,
        monitor=_AnomalousMonitor(),  # type: ignore[arg-type]
    )
    buffer = ReplayBuffer(capacity=256, obs_dim=265)
    stats = collect_from_browser(env, buffer, n_games=1, q_net=_PassPolicy())
    assert stats["anomalous_games_dropped"] == 1.0
    assert buffer.size == 0


def test_collect_from_browser_keeps_when_whitelist_disabled() -> None:
    class _AnomalousMonitor:
        def check(
            self, engine_mask: NDArray, dom_affordances_set: set[int]
        ) -> list[str]:
            del engine_mask, dom_affordances_set
            return ["DOM EXTRACTION BUG suspected: 3 engine-legal actions unsupported (test stub)"]

    driver = _RotatingDriver([_OPENING] * 40 + [_WAITING, _SCORED_MY_TURN, _TERMINAL])
    driver.set_html(_OPENING, url=f"{BASE_URL}/gt04")
    env = BrowserSplendorEnv(
        driver,
        _NoSession(),  # type: ignore[arg-type]
        click_delay=(0, 0),
        poll_interval=0.0,
        monitor=_AnomalousMonitor(),  # type: ignore[arg-type]
    )
    buffer = ReplayBuffer(capacity=256, obs_dim=265)
    stats = collect_from_browser(
        env, buffer, n_games=1, q_net=_PassPolicy(), drop_parity_anomalous=False
    )
    assert stats["anomalous_games_dropped"] == 0.0
    assert buffer.size == pytest.approx(stats["steps"])
