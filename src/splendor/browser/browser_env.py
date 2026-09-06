"""
BrowserSplendorEnv: one seat of the game.hullqin.cn/ccbs page as a gym.Env.

This is the sim-to-real seam: it satisfies the same ``SplendorEnvBase``
protocol as the local ``SplendorEnv``, so a phase-1 DQN checkpoint plays live
web games through the identical reset/step/mask surface with no code changes.

Design rulings (IMPLEMENTATION_SPEC §0.3-1, plan phase-2 §3.6):

* the legal-action mask comes from *engine rule reuse* on the pseudo state -
  never from DOM recalculation; DOM affordances are collected alongside and
  cross-checked by :class:`~splendor.browser.monitor.MaskParityMonitor`;
* the reward is the "N分" panel differential - the same Δscore semantics as
  the local env (splendor_env.py:142-161) - so P1's terminal-reward wrapper
  applies unchanged;
* ``truncated`` is always False, mirroring the local env.
"""

import random
import time

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.features import extract_observation, observation_dim
from splendor.splendor import features
from splendor.splendor.gym.envs.actions import ALL_ACTIONS
from splendor.splendor.gym.envs.utils import create_legal_actions_mask
from splendor.splendor.splendor_model import SplendorGameRule

from .action_executor import HUMAN_CLICK_DELAY, ActionExecutor, parse_pill
from .dom_extractor import (
    DEFAULT_GAME_OVER_MARKERS,
    Snapshot,
    extract_snapshot,
    is_my_turn,
    looks_like_game_over,
)
from .driver import BrowserDriver
from .monitor import MaskParityMonitor, dom_affordances
from .session import SessionManager
from .state_builder import build_pseudo_state


class BrowserSplendorEnv(gym.Env):
    """Wraps one web seat into the unified Splendor environment contract."""

    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=features.METRICS_WITH_CARDS_SHAPE,
        dtype=np.float32,
    )
    action_space = gym.spaces.Discrete(len(ALL_ACTIONS))

    # The first five parameters are the spec'd surface (IMPLEMENTATION_SPEC
    # §3 T2.5); the keyword-only rest are tuning knobs, not new coupling.
    def __init__(  # noqa: PLR0913
        self,
        driver: BrowserDriver,
        session: SessionManager,
        rule: SplendorGameRule | None = None,
        poll_interval: float = 0.4,
        step_timeout: float = 120.0,
        *,
        monitor: MaskParityMonitor | None = None,
        game_over_markers: tuple[str, ...] = DEFAULT_GAME_OVER_MARKERS,
        click_delay: tuple[float, float] = HUMAN_CLICK_DELAY,
        feature_version: str = "v1",
    ) -> None:
        """
        :param driver: the browser abstraction to drive the page with.
        :param session: room lifecycle manager (reset() calls new_game()).
        :param rule: engine rule instance; rebuilt on reset to match the
            snapshot's player count when omitted.
        :param poll_interval: seconds between turn-status polls.
        :param step_timeout: hard wait budget per step before the degraded
            pass rescue (once) and then a TimeoutError.
        :param game_over_markers: terminal-text markers; E3 pending, hence
            configurable instead of hard-coded.
        """
        super().__init__()
        self.feature_version = feature_version
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(observation_dim(feature_version),), dtype=np.float32,
        )
        self._driver = driver
        self._session = session
        self._rule = rule
        self._poll_interval = poll_interval
        self._step_timeout = step_timeout
        self._monitor = monitor if monitor is not None else MaskParityMonitor()
        self._game_over_markers = game_over_markers
        self._executor = ActionExecutor(driver, click_delay=click_delay)

        self._my_seat = 0
        self._turns = 0
        self._last_obs = np.zeros(
            (observation_dim(feature_version),), dtype=np.float32
        )
        self._last_seen_score: float | None = None
        self.last_parity_report: list[str] = []

    # ----- SplendorEnvBase protocol -----------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        """
        Start a new web game: session.new_game() -> wait for the opening ->
        first snapshot.

        :return: (obs(265,), {"my_id": seat}) where ``my_id`` is the 1-based
                 page seat number (page semantics; the pseudo-state agent
                 index is seat - 1).
        """
        _ = options
        if seed is not None:
            # Engine rule construction draws on the global RNG streams
            # (SplendorGameRule.__init__ -> initialGameState); keep the
            # seed discipline (AGENTS.md fact 6).
            random.seed(seed)
            np.random.seed(seed)

        self._session.new_game()
        # one round trip: the start-wait loop already produced the snapshot
        snapshot = self._wait_for_game_start()
        self._my_seat = snapshot["my_seat"]
        self._rule = SplendorGameRule(len(snapshot["panels"]))
        self._turns = 0
        return self._observe(snapshot), {"my_id": self._my_seat}

    def step(
        self, action: int, payment: int | None = None
    ) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Execute one action, then block until it is my turn again or the game
        ends (the opponent's whole turn - including its own payment/return
        sub-steps - happens inside this wait, mirroring how the local env
        folds opponent turns into step()).

        :param payment: tier-1 forward-compatibility parameter, ignored: the
                        engine-implied greedy payment is what the executor
                        selects.
        """
        _ = payment
        if self._rule is None:
            raise RuntimeError("call reset() before step()")

        snapshot = extract_snapshot(self._driver)
        pseudo_state = build_pseudo_state(
            snapshot, self._my_index, turns=self._turns
        )
        previous_score = snapshot["panels"][self._my_index]["score"]

        self._executor.execute(action, snapshot, pseudo_state)
        self._turns += 1
        self._last_seen_score = None

        final_snapshot = self._wait_for_my_turn()
        if self._last_seen_score is not None:
            current_score = self._last_seen_score
        else:
            # Game over returned to the room page before any poll saw a board
            # (E3): the score-differential window closes with the last view.
            current_score = previous_score
        reward = float(current_score - previous_score)
        terminated = looks_like_game_over(
            final_snapshot["status"],
            self._game_over_markers,
            board_present=self._board_present(final_snapshot),
        )
        # A finished game returns to the room page (E3): no table rows, hence
        # no observable board - the last in-game observation stands in.
        return self._observe(final_snapshot), reward, terminated, False, {}

    def get_legal_actions_mask(self) -> NDArray:
        """
        Engine-rule mask on the pseudo state (the sole legality authority),
        with the DOM affordance set collected for the parity monitor.
        """
        if self._rule is None:
            raise RuntimeError("call reset() before get_legal_actions_mask()")
        snapshot = extract_snapshot(self._driver)
        pseudo_state = build_pseudo_state(
            snapshot, self._my_index, turns=self._turns
        )
        legal_actions = self._rule.getLegalActions(pseudo_state, self._my_index)
        mask = create_legal_actions_mask(legal_actions, pseudo_state, self._my_index)
        self.last_parity_report = self._monitor.check(
            mask, dom_affordances(snapshot)
        )
        return mask

    def get_payment_options(self, action: int) -> list[dict] | None:
        """
        Payment options of a pending purchase, or None.

        Tier-1 policy never chooses (the executor picks the engine-greedy
        pill); this method exists for the tier-2 payment dimension and for
        debugging the live page.
        """
        _ = action
        snapshot = extract_snapshot(self._driver)
        pills = snapshot["payment_options"]
        if not pills:
            return None
        return [parse_pill(pill) for pill in pills]

    # ----- internals ----------------------------------------------------------
    @property
    def driver(self) -> BrowserDriver:
        """
        The underlying browser driver.

        Harness-level introspection (play-web's reporting reads the last
        page view through it); normal environment users never need this.
        """
        return self._driver

    @property
    def _my_index(self) -> int:
        """Agent index of my seat (page seats are 1-based, panels in order)."""
        return self._my_seat - 1

    @staticmethod
    def _board_present(snapshot: Snapshot) -> bool:
        """Measured E3 terminal signal: the room page renders no board."""
        return any(count > 0 for count in snapshot["deck_counts"]) or any(
            slot is not None for row in snapshot["dealt"] for slot in row
        )

    def _observe(self, snapshot: Snapshot) -> NDArray:
        if not self._board_present(snapshot) or not snapshot["panels"]:
            # Room page (game over or not yet started): nothing to vectorize.
            return self._last_obs
        pseudo_state = build_pseudo_state(
            snapshot, self._my_index, turns=self._turns
        )
        self._last_obs = extract_observation(
            pseudo_state, self._my_index, self.feature_version
        )
        return self._last_obs

    def _wait_for_game_start(self) -> Snapshot:
        # Only my turn counts as "started": the room page (before/after a
        # game) is a waiting state too, not a playable board (measured E3).
        deadline = time.monotonic() + self._step_timeout
        while True:
            snapshot = extract_snapshot(self._driver)
            if is_my_turn(snapshot["status"]) and self._board_present(snapshot):
                return snapshot
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"game did not start within {self._step_timeout}s "
                    f"(status: {snapshot['status']!r})"
                )
            time.sleep(self._poll_interval)

    def _wait_for_my_turn(self) -> Snapshot:
        """
        Poll the turn status until it is my decision point again.

        ``saw_other_turn`` guards against the page merely re-rendering my own
        turn right after my action (the server needs a moment to process and
        rotate). On the hard timeout the degraded rescue fires *once* - a
        stuck pass keeps the seat alive; a second timeout surfaces to the
        caller (which should run session.recover()).
        """
        deadline = time.monotonic() + self._step_timeout
        saw_other_turn = False
        rescued = False
        while True:
            snapshot = extract_snapshot(self._driver)
            if self._my_index in range(len(snapshot["panels"])):
                # Remember the newest in-game panel score: on game over the
                # board (and the panels with it) vanishes, and the final
                # reward must not lose the last scoring transition.
                self._last_seen_score = float(
                    snapshot["panels"][self._my_index]["score"]
                )
            status = snapshot["status"]
            if looks_like_game_over(
                status,
                self._game_over_markers,
                board_present=self._board_present(snapshot),
            ):
                return snapshot
            if is_my_turn(status):
                if saw_other_turn:
                    return snapshot
                if time.monotonic() > deadline:
                    if not rescued:
                        # My own input is stuck (e.g. a half-finished
                        # selection): pass to keep the seat alive.
                        self._executor.force_pass()
                        rescued = True
                        deadline = time.monotonic() + self._step_timeout
                    else:
                        raise TimeoutError(
                            f"turn did not return within {self._step_timeout}s "
                            "after the degraded pass; run session.recover()"
                        )
            else:
                saw_other_turn = True
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"opponents did not finish within {self._step_timeout}s "
                        f"(status: {status!r}); run session.recover()"
                    )
            time.sleep(self._poll_interval)
