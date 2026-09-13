"""
Monte-Carlo position win-rate estimator (phase-6, server side).

Definition (dashboard + docs use the same wording):

    实时局面胜率(玩家 i, 局面 s) = 在从 s 重建的完整引擎状态上, 用当前
    DQN 策略为所有座位自对弈模拟 N 局至终局, 玩家 i 以最高分终局的频率
    (同分按终局卡数 tie-break, 仍并列各计 0.5 胜)。

Seat counts: the v1 metric block carries MAX_RIVALS (=3) rival-score slots on
each side of the observer, so the reconstruction and the observation cover
2-4 seats and the estimator accepts all of them. The ``public-v2`` schema
hard-codes exactly two seats. Note the honesty caveat: a checkpoint trained
on 2-player self-play still produces a well-defined value on 3-4 seats, but
that value is an out-of-distribution *proxy* for strength, not a calibrated
win probability.

Unknown information is filled by uniform sampling from the card registry
(the 90-card library minus visible faces): remaining deck compositions and
rivals' reserved-card faces. This makes the estimate an approximation of
"policy strength from this position", not an exact solving - the sampling
seeds are fixed per request so the estimate is reproducible.

Batching: rollouts step in lockstep so each tick performs ONE batched
``forward`` over all active rollouts - the same trick as vectorised RL
envs. Engine-side work (legal actions, successor, features) stays
per-rollout; that is the irreducible Python cost.

Engine-termination semantics are mirrored from ``SplendorGameRule.gameEnds``
(score >= 15 lands exactly when the round wraps to agent 0, or every agent
passed) on explicit states - no rule-instance mutation, since rollouts run
on per-rollout deepcopies instead of ``current_game_state``.
"""

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.dqn.features import (
    PLAYERS,
    extract_observation,
)
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.browser.card_registry import CARD_REGISTRY, lookup_card
from splendor.browser.dom_extractor import Snapshot
from splendor.browser.state_builder import build_pseudo_state
from splendor.splendor.constants import MAX_RIVALS, WINNING_SCORE_TRESHOLD
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import (
    Card,
    SplendorGameRule,
    SplendorState,
)

# A rollout that has not ended after this many *actions* is aborted (credit
# 0 for every seat) - only pathological stalls (all-pass loops) hit it.
DEFAULT_MAX_STEPS = 400

# Supported seat counts: 2 players up to the engine's rival-slot capacity
# (MAX_RIVALS rival slots per side of the observer => MAX_RIVALS + 1 seats).
MIN_SEATS = 2
MAX_SEATS = MAX_RIVALS + 1


@dataclass(frozen=True)
class RolloutResult:
    """Per-seat win estimates; all rates sum to 1 across seats."""

    win_rates: list[float]
    draw_rate: float
    rollouts: int
    aborted: int
    elapsed_s: float


class WinRateEstimator:
    """Batched Monte-Carlo win-rate estimation for one loaded checkpoint."""

    def __init__(
        self,
        model: QNetwork,
        rule: SplendorGameRule | None = None,
    ) -> None:
        """
        :param model: the policy evaluated for every seat (self-play).
        :param rule: engine rule reused for legality/successor calls; a
            scratch instance is built when omitted (never mutated per call -
            states are passed explicitly everywhere).
        """
        self._model = model
        self._feature_version = model.feature_version
        self._rule = rule if rule is not None else SplendorGameRule(2)

    def estimate(
        self,
        snapshot: Snapshot,
        actor_seat: int,
        n_rollouts: int = 16,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
    ) -> RolloutResult:
        """
        Estimate the per-seat win probabilities of ``snapshot``.

        :param actor_seat: 1-based page seat to move next (the env knows it
            from the status text; the reconstruction mirrors the engine's
            round-wrap termination rule off this).
        :param seed: seeds the global RNG streams for the unknown-information
            sampling (AGENTS.md fact 6 discipline), making one request's
            estimate reproducible.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        start = time.monotonic()
        n_players = len(snapshot["panels"])
        # v1's metric block reserves MAX_RIVALS score slots on each side of the
        # observer, so 2-4 seats all vectorize; public-v2-multi generalizes the
        # rival panels to MAX_RIVALS slots (roadmap B1/E4). Only the legacy
        # public-v2 schema remains hard-coded to exactly two players.
        if not MIN_SEATS <= n_players <= MAX_SEATS:
            raise ValueError(
                f"win-rate estimation supports {MIN_SEATS}..{MAX_SEATS} seats, "
                f"got {n_players}"
            )
        if self._feature_version == "public-v2" and n_players != PLAYERS:
            raise ValueError(
                f"feature schema {self._feature_version!r} encodes exactly "
                f"{PLAYERS} seats, got {n_players}"
            )
        if actor_seat not in range(1, n_players + 1):
            raise ValueError(f"actor_seat {actor_seat} outside 1..{n_players}")
        actor_index = actor_seat - 1

        rollouts: list[tuple[SplendorState, int]] = []
        for rollout_index in range(n_rollouts):
            state = self._reconstruct_state(snapshot, actor_index, rollout_index)
            rollouts.append((state, actor_index))

        win_credits, draws, finished = self._run_rollouts(rollouts, max_steps)
        return RolloutResult(
            win_rates=[float(rate / n_rollouts) for rate in win_credits],
            draw_rate=draws / n_rollouts,
            rollouts=n_rollouts,
            aborted=n_rollouts - finished,
            elapsed_s=time.monotonic() - start,
        )

    # ----- rollout stepping ------------------------------------------------------
    def _run_rollouts(
        self, rollouts: list[tuple[SplendorState, int]], max_steps: int
    ) -> tuple[NDArray, int, int]:
        """
        Lockstep stepping to termination.

        :returns: (win credits per seat, draw count, finished count).
            Every finished rollout distributes exactly one credit (draws
            split it 0.5/0.5); aborted rollouts distribute none.
        """
        n_players = len(rollouts[0][0].agents)
        win_credits = np.zeros(n_players, dtype=np.float64)
        draws = 0
        finished = 0
        step = 0
        while step < max_steps and rollouts:
            ongoing: list[tuple[SplendorState, int]] = []
            for state, next_index in self._advance(rollouts):
                if self._game_over(state, next_index):
                    finished += 1
                    winners = self._winners(state)
                    draws += 1 if len(winners) > 1 else 0
                    for seat in winners:
                        win_credits[seat] += 1.0 / len(winners)
                else:
                    ongoing.append((state, next_index))
            rollouts = ongoing
            step += 1
        return win_credits, draws, finished

    # ----- state reconstruction ------------------------------------------------
    @staticmethod
    def _reconstruct_state(
        snapshot: Snapshot, actor_index: int, rollout_index: int
    ) -> SplendorState:
        """
        Full engine state from the DOM snapshot: known facts restored via the
        pseudo-state builder, unknown facts (deck compositions, rivals'
        reserved faces) sampled from the unseen registry remainder.
        """
        my_index = snapshot["my_seat"] - 1
        state = build_pseudo_state(snapshot, my_index)
        unseen = _unseen_cards(snapshot)
        rng = random.Random(hash(("winrate", actor_index, rollout_index)) & 0xFFFF)

        # Rivals' reserved backs leak their tier; sample real faces of that
        # tier so their buy actions behave like real cards.
        for agent_index, agent in enumerate(state.agents):
            if agent_index == my_index:
                continue
            tiers = snapshot["panels"][agent_index]["reserved_tiers"]
            sampled: list[Card] = []
            for tier in tiers:
                card = _pop_random(unseen, tier, rng)
                if card is not None:
                    sampled.append(card)
            agent.cards["yellow"] = sampled

        # Decks: whatever remains unseen, per tier, shuffled (deal order is
        # unknown information by definition).
        state.board.decks = [[], [], []]
        for card in unseen:
            state.board.decks[card.deck_id].append(card)
        for deck in state.board.decks:
            rng.shuffle(deck)

        state.agent_to_move = actor_index
        return state

    # ----- rollout stepping ------------------------------------------------------
    def _advance(
        self, rollouts: list[tuple[SplendorState, int]]
    ) -> list[tuple[SplendorState, int]]:
        """One lockstep tick: batched forward, per-rollout engine step."""
        if not rollouts:
            return []
        batch_obs: list[NDArray] = []
        batch_mask: list[NDArray] = []
        prepared: list[tuple[SplendorState, int, list]] = []
        for state, index in rollouts:
            legal = self._rule.getLegalActions(state, index)
            mask = create_legal_actions_mask(legal, state, index)
            batch_obs.append(extract_observation(state, index, self._feature_version))
            batch_mask.append(mask)
            prepared.append((state, index, legal))
        obs = torch.from_numpy(np.stack(batch_obs)).float()
        masks = torch.from_numpy(np.stack(batch_mask)).float()
        q_values = self._model.forward(obs, masks)
        action_indices = q_values.argmax(dim=-1).tolist()

        stepped: list[tuple[SplendorState, int]] = []
        for (state, index, legal), action_index in zip(
            prepared, action_indices, strict=True
        ):
            mapping = create_action_mapping(legal, state, index)
            engine_action = mapping[action_index]
            # generateSuccessor mutates in place; the rollout state is a
            # private deepcopy so no rollback pairing is needed here.
            self._rule.generateSuccessor(state, engine_action, index)
            stepped.append((state, (index + 1) % len(state.agents)))
        return stepped

    # ----- termination & scoring -------------------------------------------------
    @staticmethod
    def _game_over(state: SplendorState, next_index: int) -> bool:
        """Mirror of SplendorGameRule.gameEnds on an explicit state."""
        passed = 0
        for agent in state.agents:
            passed += 1 if agent.passed else 0
            if agent.score >= WINNING_SCORE_TRESHOLD and next_index == 0:
                return True
        return passed == len(state.agents)

    @staticmethod
    def _winners(state: SplendorState) -> list[int]:
        """Seat indices sharing the top calScore (tie-break already inside)."""
        # calScore reads only its game_state argument; a bare instance avoids
        # SplendorGameRule.__init__'s RNG-consuming initial state build.
        scratch = object.__new__(SplendorGameRule)
        scores = [scratch.calScore(state, seat) for seat in range(len(state.agents))]
        top = max(scores)
        return [seat for seat, score in enumerate(scores) if score == top]


# ---- module-level helpers (kept outside the class: pure functions) --------------
def _unseen_cards(snapshot: Snapshot) -> list[Card]:
    """Registry cards not visible in the snapshot (dealt + my reserved)."""
    visible_codes: set[str] = set()
    for row in snapshot["dealt"]:
        for info in row:
            if info is not None:
                visible_codes.add(lookup_card(**_registry_args(info)).code)
    visible_codes.update(
        lookup_card(**_registry_args(info)).code for info in snapshot["my_reserved"]
    )
    return [card for card in CARD_REGISTRY.values() if card.code not in visible_codes]


def _registry_args(info: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot CardInfo -> lookup_card kwargs."""
    return {
        "deck_id": info["tier"],
        "colour": info["colour"],
        "points": info["points"],
        "cost": info["cost"],
    }


def _pop_random(unseen: list[Card], tier: int, rng: random.Random) -> Card | None:
    """Remove and return a random unseen card of ``tier`` (None if empty)."""
    candidates = [i for i, card in enumerate(unseen) if card.deck_id == tier]
    if not candidates:
        return None
    position = candidates[rng.randrange(len(candidates))]
    return unseen.pop(position)
