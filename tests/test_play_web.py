"""
Phase-3 offline tests: the play-web harness and the experience back-flow,
both against the mock browser driver (no network).
"""

from pathlib import Path

import numpy as np
import torch

import splendor.splendor.gym  # noqa: F401  # registers the gym env id
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.replay_buffer import ReplayBuffer
from splendor.agents.our_agents.dqn.training import collect_from_browser
from splendor.browser.browser_env import BrowserSplendorEnv
from splendor.browser.driver import SNAPSHOT_JS_MARKER, MockBrowserDriver
from splendor.play_web import GameReport, load_agent, run_game

FIXTURES = Path(__file__).parent.parent / "src" / "splendor" / "browser" / "fixtures"
BASE_URL = "https://game.hullqin.cn/ccbs"
_OPENING = (FIXTURES / "opening.html").read_text(encoding="utf-8")
# my panel is the first my-2 block in the fixture: its score bump reads +3
_SCORED = _OPENING.replace(
    '<span class="ccbs-score">0分</span>', '<span class="ccbs-score">3分</span>', 1
).replace("等待你操作", "等待玩家2操作", 1)
_SCORED_MY_TURN = _SCORED.replace("等待玩家2操作", "等待你操作", 1)
_WAITING = _SCORED  # opponent acting: my turn text flipped, scores unchanged
def _strip_between(page: str, start_marker: str, end_marker: str) -> str:
    """Remove the block [first start_marker, first end_marker) - the fixture
    generator emits the rows and the supply row as contiguous blocks, so
    marker slicing is exact (regex nesting would mis-eat panels)."""
    start = page.index(start_marker)
    end = page.index(end_marker, start)
    return page[:start] + page[end:]


# measured E3 room view: the board (rows + supply) is gone, the game is over
_TERMINAL = _strip_between(
    _SCORED_MY_TURN, '<div class="flex justify-center origin-top">', '<div class="ccbs-noble'
)
_TERMINAL = _strip_between(
    _TERMINAL,
    '<div class="mt-4 flex items-center justify-center space-x-6">',
    '<div class="flex flex-wrap items-center justify-center my-2">',
)


class _RotatingDriver(MockBrowserDriver):
    """
    Serves an explicit page list on snapshot reads after :meth:`arm`,
    advancing one page per read and holding the last page forever - an
    offline stand-in for the temporal page sequence (my turn -> opponent
    acting -> my turn scored -> room view). Tests pass an explicit read
    ledger (see the comments at each use) because every extract counts.
    """

    def __init__(self, pages: list[str]) -> None:
        super().__init__()
        self._pages = pages
        self._armed = False
        self._reads = 0

    def arm(self) -> None:
        self._armed = True

    def reset_rotation(self) -> None:
        self._reads = 0

    def evaluate(self, js: str) -> object:
        if self._armed and SNAPSHOT_JS_MARKER in js:
            index = min(self._reads, len(self._pages) - 1)
            self._reads += 1
            self.set_html(self._pages[index])
        return super().evaluate(js)


class _PassPolicy:
    """Greedy-policy stand-in that always plays PASS (deterministic ledger)."""

    input_dim = 265
    output_dim = 3510

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        return 0  # ALL_ACTIONS[0] = PASS


class _StubSession:
    """new_game() reloads the opening page and rewinds the rotation."""

    def __init__(self, driver: _RotatingDriver) -> None:
        self.driver = driver

    def new_game(self) -> None:
        self.driver.set_html(_OPENING, url=f"{BASE_URL}/gt01")
        self.driver.reset_rotation()

    def recover(self) -> None:  # pragma: no cover - not exercised here
        raise AssertionError("recover is not part of these tests")


def _make_env(pages: list[str]) -> BrowserSplendorEnv:
    """Rotation is armed immediately; new_game() rewinds it (see session)."""
    driver = _RotatingDriver(pages)
    driver.set_html(_OPENING, url=f"{BASE_URL}/gt01")
    driver.arm()
    return BrowserSplendorEnv(
        driver, _StubSession(driver), click_delay=(0, 0), poll_interval=0.0
    )


def test_run_game_reports_and_survives_terminal() -> None:
    """
    One action, then the page rotates to a +3-scored board and finally to the
    measured room view (game over). run_game must translate the panel delta
    into the report and stop cleanly.

    Read ledger of the first step: reset's start-wait (1) + the greedy mask
    (1) + step's pre-action extract (1) -> three opening reads; the score
    bump appears on poll 4 and the room view on poll 5.
    """
    # reads: reset-wait(1) + mask(1) + pre-action(1) stay on the opening
    # board; polls then see opponent-acting -> scored my-turn -> room view.
    env = _make_env([
        _OPENING, _OPENING, _OPENING,
        _WAITING, _SCORED_MY_TURN, _TERMINAL,
    ])
    report = run_game(env, _PassPolicy(), max_steps=5)

    assert isinstance(report, GameReport)
    assert report.result == "win"  # 3 vs 0 on the final visible board
    assert report.my_score == 3.0
    assert report.rival_score == 0.0
    assert report.steps >= 1
    assert report.duration >= 0.0


def test_load_agent_roundtrip_preserves_q_values(tmp_path: Path) -> None:
    from splendor.agents.our_agents.dqn.utils import save_model

    net = QNetwork()
    path = tmp_path / "dqn_model.pth"
    save_model(net, path, step=123, config={"hidden_layers": (128, 128, 128, 128)})
    loaded = load_agent(path)

    obs = torch.randn(4, net.input_dim)
    mask = torch.ones(4, net.output_dim)
    with torch.no_grad():
        reference = net(obs, mask)
        restored = loaded(obs, mask)
    assert torch.allclose(reference, restored)


def test_collect_from_browser_fills_buffer() -> None:
    """
    Web transitions land in the replay buffer with the right shapes.

    Read ledger of the first (and only) step: reset wait + mask + pre-action
    = three opening reads, then the scored board, then the room view.
    """
    env = _make_env([_OPENING, _WAITING, _SCORED_MY_TURN, _TERMINAL])
    buffer = ReplayBuffer(capacity=1000, n_step=1)

    stats = collect_from_browser(env, buffer, n_games=1, q_net=None)

    assert stats["games"] == 1.0
    assert stats["steps"] >= 1
    assert len(buffer) == stats["steps"]
    obs, _action, reward, next_obs, next_mask, done = buffer.sample(1)
    assert obs.shape == (1, 265)
    assert next_obs.shape == (1, 265)
    assert next_mask.shape == (1, 3510)
    assert float(done[0]) == 1.0  # the harvest ran until the game ended
    assert torch.isfinite(reward[0])


def test_collect_from_browser_random_policy_is_seeded() -> None:
    """Same seed -> same random-policy action stream (AGENTS.md fact 6)."""
    import random

    env = _make_env([_OPENING, _WAITING, _SCORED_MY_TURN, _TERMINAL])
    buffer = ReplayBuffer(capacity=1000, n_step=1)
    random.seed(7)
    np.random.seed(7)
    first = collect_from_browser(env, buffer, n_games=1, q_net=None)

    env = _make_env([_OPENING, _WAITING, _SCORED_MY_TURN, _TERMINAL])
    buffer2 = ReplayBuffer(capacity=1000, n_step=1)
    random.seed(7)
    np.random.seed(7)
    second = collect_from_browser(env, buffer2, n_games=1, q_net=None)

    assert first["steps"] == second["steps"]
