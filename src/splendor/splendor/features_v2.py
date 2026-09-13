"""Shared public-information feature schemas (roadmap 2026-09-12 §B1).

``public-v2`` started life inside the DQN package as a two-player-only
extension of the 265-dim v1 observation.  Phase B1 lifts it into the engine
layer so every trainer (DQN, BC, PPO) and the browser stack share one
implementation, and adds a seat-count-agnostic layout.

Layouts
-------
``v1`` (265): ``features.extract_metrics_with_cards`` — untouched, locked by
``tests/test_feature_parity.py``.

``public-v2`` (312, two seats only — byte-for-byte identical to the original
DQN implementation)::

    [0:265)    v1 base
    [265:271)  board supply gems (5 colours + yellow)
    [271:283)  self panel (gems incl. yellow, cards per colour, yellow cards)
    [283:295)  rival panel (same shape)
    [295:310)  noble costs (3 x 5 colours)
    [310:312)  seat, endgame flag

``public-v2-multi`` (337, any seat count 2..4)::

    [0:312)    exactly the ``public-v2`` layout above, so a two-seat
               ``public-v2-multi`` observation is bit-identical to the legacy
               one on its full support
    [312:313)  number of agents (explicit seat-count feature)
    [313:325)  rival slot 1 (cyclic seat order, zeros when absent)
    [325:337)  rival slot 2 (cyclic seat order, zeros when absent)

Rival slots follow cyclic seat order after ``seat``; absent rivals are
zero-padded, mirroring how the v1 metric block pads ``MAX_RIVALS`` score
slots.  Rival reserved-card *counts* stay public; identities, deck order and
purchased-card identities never enter any v2 schema.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from splendor.splendor.constants import (
    MAX_RIVALS,
    NORMAL_COLORS,
    WINNING_SCORE_TRESHOLD,
)
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.splendor_model import SplendorState

V1_DIM: Final = 265
V2_DIM: Final = 312
V2_MULTI_DIM: Final = 337
#: Per-agent panel: 6 gems (incl. yellow) + 5 card colours + 1 yellow count.
PANEL_DIM: Final = len(NORMAL_COLORS) + 6 + 1
SCHEMA_V1 = "v1"
SCHEMA_PUBLIC_V2 = "public-v2"
SCHEMA_PUBLIC_V2_MULTI = "public-v2-multi"
KNOWN_SCHEMAS: Final = (SCHEMA_V1, SCHEMA_PUBLIC_V2, SCHEMA_PUBLIC_V2_MULTI)
COLORS = (*NORMAL_COLORS, "yellow")
#: The legacy two-seat schema is fixed to exactly two agents.
LEGACY_PLAYERS: Final = 2
#: Minimum seat count for the multi schema (the engine caps at MAX_RIVALS + 1).
MIN_PLAYERS: Final = 2


def observation_dim(version: str) -> int:
    """Validate the schema name rather than silently reading wrong weights."""
    if version not in KNOWN_SCHEMAS:
        raise ValueError(f"Unknown feature schema: {version}")
    if version == SCHEMA_V1:
        return V1_DIM
    if version == SCHEMA_PUBLIC_V2:
        return V2_DIM
    return V2_MULTI_DIM


def agent_panel(state: SplendorState, index: int) -> list[float]:
    """Public panel of one agent: gems, per-colour cards, yellow card count."""
    agent = state.agents[index]
    panel = [float(agent.gems.get(c, 0)) for c in COLORS]
    panel.extend(float(len(agent.cards[c])) for c in NORMAL_COLORS)
    panel.append(float(len(agent.cards["yellow"])))
    return panel


def noble_costs(state: SplendorState) -> list[float]:
    """Flattened costs of up to three nobles, zero-padded like the engine."""
    costs: list[float] = []
    for index in range(MAX_RIVALS):
        cost = state.board.nobles[index][1] if index < len(state.board.nobles) else {}
        costs.extend(float(cost.get(c, 0)) for c in NORMAL_COLORS)
    return costs


def extract_observation_public_v2(
    state: SplendorState, seat: int
) -> NDArray[np.float32]:
    """Legacy two-seat 312-dim schema (byte-identical to the DQN original)."""
    if len(state.agents) != LEGACY_PLAYERS:
        raise ValueError("public-v2 currently supports two players only")
    base = extract_metrics_with_cards(state, seat).astype(np.float32)
    extra = [float(state.board.gems.get(c, 0)) for c in COLORS]
    for index in (seat, 1 - seat):
        extra.extend(agent_panel(state, index))
    extra.extend(noble_costs(state))
    extra.extend(
        [
            float(seat),
            float(any(a.score >= WINNING_SCORE_TRESHOLD for a in state.agents)),
        ]
    )
    return np.concatenate((base, np.asarray(extra, dtype=np.float32)))


def extract_observation_public_v2_multi(
    state: SplendorState, seat: int
) -> NDArray[np.float32]:
    """Seat-count-agnostic 337-dim schema (see module docstring for layout)."""
    n_agents = len(state.agents)
    if not MIN_PLAYERS <= n_agents <= MAX_RIVALS + 1:
        raise ValueError(f"public-v2-multi supports 2..{MAX_RIVALS + 1} players")
    if not 0 <= seat < n_agents:
        raise ValueError(f"seat {seat} out of range for {n_agents} players")
    legacy = extract_observation_public_v2_multi_prefix(state, seat)
    extra = [float(n_agents)]
    for slot in range(1, MAX_RIVALS):
        rival = seat + slot + 1
        if rival < seat + n_agents:
            extra.extend(agent_panel(state, rival % n_agents))
        else:
            extra.extend([0.0] * PANEL_DIM)
    return np.concatenate(
        (legacy, np.asarray(extra, dtype=np.float32)),
    )


def extract_observation_public_v2_multi_prefix(
    state: SplendorState, seat: int
) -> NDArray[np.float32]:
    """First 312 dims of the multi layout for any seat count.

    Identical in content to :func:`extract_observation_public_v2` except the
    single rival panel is the *first* cyclic rival instead of assuming two
    players.  For two players the two functions agree bit for bit.
    """
    n_agents = len(state.agents)
    if not 0 <= seat < n_agents:
        raise ValueError(f"seat {seat} out of range for {n_agents} players")
    base = extract_metrics_with_cards(state, seat).astype(np.float32)
    extra = [float(state.board.gems.get(c, 0)) for c in COLORS]
    extra.extend(agent_panel(state, seat))
    extra.extend(agent_panel(state, (seat + 1) % n_agents))
    extra.extend(noble_costs(state))
    extra.extend(
        [
            float(seat),
            float(any(a.score >= WINNING_SCORE_TRESHOLD for a in state.agents)),
        ]
    )
    return np.concatenate((base, np.asarray(extra, dtype=np.float32)))


def extract_observation_v2(
    state: SplendorState, seat: int, version: str
) -> NDArray[np.float32]:
    """Dispatch between the versioned public schemas (v1 handled upstream)."""
    if version == SCHEMA_PUBLIC_V2:
        return extract_observation_public_v2(state, seat)
    if version == SCHEMA_PUBLIC_V2_MULTI:
        return extract_observation_public_v2_multi(state, seat)
    raise ValueError(f"Unknown public feature schema: {version}")
