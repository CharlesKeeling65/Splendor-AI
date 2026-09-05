"""
Feature/mask parity test: the quality gate of the whole sim-to-real pipeline
(plan phase-2 §3.4, acceptance A2.1).

It verifies not "the code runs" but "the two worlds are isomorphic": for
randomly played decision points,

1. the 265-d observation extracted from the *engine* state is bitwise equal
   to the observation extracted from ``pseudo_state(build pseudo from
   project_visible(state))`` - guarding the feature pipeline (colour maps,
   tier direction, cost reading, reserved handling, turn counting);
2. the legal-action mask computed by the engine on the real state equals the
   one computed by the *same engine rule* on the pseudo state - the direct
   evidence for ruling F9 (getLegalActions only reads panel-level fields)
   and for the placeholder-card construction (F7).

``project_visible`` is deliberately the only bridge: one piece of code, two
uses - the test projector here and the reference for offline fixtures.
"""

import random

import numpy as np
import pytest
from numpy.typing import NDArray

from splendor.browser.dom_extractor import (
    CardInfo,
    NobleInfo,
    PanelInfo,
    Snapshot,
)
from splendor.browser.state_builder import build_pseudo_state
from splendor.splendor.constants import NORMAL_COLORS
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.envs.utils import create_legal_actions_mask
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState

# Acceptance A2.1 requires >=1000 decision points; the check itself costs
# ~6s for 1000 points on a laptop, far below the ~60s budget, so the target
# is applied verbatim (raise here if the budget ever grows).
PARITY_POINTS_TARGET = 1000
MAX_TIER_CARDS = 4
NUMBER_OF_TIERS = 3


def project_visible(state: SplendorState, my_index: int) -> Snapshot:
    """
    Project an engine state down to exactly what the page DOM can show.

    Everything not DOM-visible (deck contents, rival reserved faces, rival
    gems, opponent card identities) is intentionally dropped; the parity test
    asserts the dropped parts are exactly the parts features/rules never read.
    """
    dealt: list[list[CardInfo | None]] = []
    for deck_id in range(NUMBER_OF_TIERS):
        row: list[CardInfo | None] = []
        for card in state.board.dealt[deck_id]:
            if card is None:
                row.append(None)
            else:
                row.append(_card_info(card.deck_id, card.colour, card.points, card.cost))
        dealt.append(row)

    panels: list[PanelInfo] = []
    for index, agent in enumerate(state.agents):
        panels.append(
            {
                "seat": index + 1,
                "score": agent.score,
                "card_counts": {c: len(agent.cards[c]) for c in NORMAL_COLORS},
                "gems": dict(agent.gems),
                "reserved_tiers": [c.deck_id for c in agent.cards["yellow"]],
            }
        )

    nobles: list[NobleInfo] = [
        {"requirements": dict(cost)} for _, cost in state.board.nobles
    ]

    my_agent = state.agents[my_index]
    return Snapshot(
        dealt=dealt,
        deck_counts=[len(deck) for deck in state.board.decks],
        nobles=nobles,
        supply=dict(state.board.gems),
        panels=panels,
        my_seat=my_index + 1,
        my_reserved=[
            _card_info(c.deck_id, c.colour, c.points, c.cost)
            for c in my_agent.cards["yellow"]
        ],
        status="等待你操作",
        payment_options=None,
        noble_options=None,
    )


def _card_info(deck_id: int, colour: str, points: int, cost: dict) -> CardInfo:
    return CardInfo(tier=deck_id, colour=colour, points=points, cost=dict(cost))


def _collect_parity_points(
    seed: int, target: int
) -> list[tuple[SplendorState, SplendorGameRule, int, int]]:
    """
    Play random games and record every decision point (state, rule, agent,
    turn count) until ``target`` points are gathered.

    Seeding follows the three-RNG discipline (AGENTS.md fact 6): dealing uses
    the global ``random`` module, so its stream is seeded explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)

    points: list[tuple[SplendorState, SplendorGameRule, int, int]] = []
    games = 0
    while len(points) < target and games < 100:
        games += 1
        rule = SplendorGameRule(2)
        state = rule.initialGameState()
        guard = 0
        while not rule.gameEnds() and guard < 500 and len(points) < target:
            guard += 1
            agent_index = rule.current_agent_index
            legal = rule.getLegalActions(state, agent_index)
            if not legal:  # pragma: no cover - engine always yields pass
                break
            turns = len(state.agents[agent_index].agent_trace.action_reward)
            points.append((state, rule, agent_index, turns))
            random.choice(legal)
            rule.update(random.choice(legal))
    return points


def test_obs_and_mask_parity() -> None:
    """
    A2.1: over >=1000 random decision points,
    (i) engine obs == projected-pseudo obs bitwise (np.allclose, zero tol);
    (ii) engine legal mask == pseudo-state legal mask exactly.
    """
    points = _collect_parity_points(seed=20260905, target=PARITY_POINTS_TARGET)
    assert len(points) >= PARITY_POINTS_TARGET, (
        f"only {len(points)} decision points collected; the random-play "
        "driver must reach the acceptance target"
    )

    for point_index, (state, rule, agent_index, turns) in enumerate(points):
        snapshot = project_visible(state, agent_index)
        pseudo_state = build_pseudo_state(
            snapshot, agent_index, turns=turns
        )

        engine_obs = extract_metrics_with_cards(state, agent_index)
        pseudo_obs = extract_metrics_with_cards(pseudo_state, agent_index)
        if not np.allclose(engine_obs, pseudo_obs, rtol=0, atol=0):
            _fail_obs(point_index, engine_obs, pseudo_obs)

        engine_mask = create_legal_actions_mask(
            rule.getLegalActions(state, agent_index), state, agent_index
        )
        pseudo_mask: NDArray = create_legal_actions_mask(
            rule.getLegalActions(pseudo_state, agent_index),
            pseudo_state,
            agent_index,
        )
        if not np.array_equal(engine_mask, pseudo_mask):
            _fail_mask(point_index, engine_mask, pseudo_mask)


def _fail_obs(point_index: int, engine_obs: NDArray, pseudo_obs: NDArray) -> None:
    differing = np.flatnonzero(engine_obs != pseudo_obs)
    first = int(differing[0])
    pytest.fail(
        f"obs parity broken at decision point {point_index}: dimension {first} "
        f"engine={engine_obs[first]!r} pseudo={pseudo_obs[first]!r} "
        f"({len(differing)} differing dimensions total)"
    )


def _fail_mask(point_index: int, engine_mask: NDArray, pseudo_mask: NDArray) -> None:
    engine_only = np.flatnonzero(engine_mask & ~pseudo_mask)
    pseudo_only = np.flatnonzero(pseudo_mask & ~engine_mask)
    pytest.fail(
        f"mask parity broken at decision point {point_index}: "
        f"{len(engine_only)} engine-only indices {engine_only[:10].tolist()}, "
        f"{len(pseudo_only)} pseudo-only indices {pseudo_only[:10].tolist()}"
    )


def test_projection_drops_exactly_the_invisible() -> None:
    """
    Sanity for the projector itself: a projected+rebuilt state must preserve
    every panel-level fact the features read, for BOTH seats.
    """
    random.seed(7)
    np.random.seed(7)
    rule = SplendorGameRule(2)
    state = rule.initialGameState()
    for _ in range(30):
        legal = rule.getLegalActions(state, rule.current_agent_index)
        random.choice(legal)
        rule.update(random.choice(legal))
        if rule.gameEnds():
            break

    for agent_index in range(len(state.agents)):
        pseudo = build_pseudo_state(
            project_visible(state, agent_index),
            agent_index,
            turns=len(state.agents[agent_index].agent_trace.action_reward),
        )
        assert pseudo.agents[agent_index].score == state.agents[agent_index].score
        assert pseudo.board.gems == state.board.gems
        assert pseudo.board.nobles == state.board.nobles
        for colour in NORMAL_COLORS:
            assert len(pseudo.agents[agent_index].cards[colour]) == len(
                state.agents[agent_index].cards[colour]
            )
        assert len(pseudo.agents[agent_index].cards["yellow"]) == len(
            state.agents[agent_index].cards["yellow"]
        )
