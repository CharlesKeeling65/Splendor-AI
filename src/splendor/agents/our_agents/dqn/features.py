"""Versioned, public-information-only DQN observations; v1 stays unchanged."""

import numpy as np
from numpy.typing import NDArray

from splendor.splendor.constants import NORMAL_COLORS, WINNING_SCORE_TRESHOLD
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.splendor_model import SplendorState

V1_DIM = 265
V2_DIM = 312
COLORS = (*NORMAL_COLORS, "yellow")
PLAYERS = 2


def observation_dim(version: str) -> int:
    """Validate the schema name rather than silently reading wrong weights."""
    if version not in {"v1", "public-v2"}:
        raise ValueError(f"Unknown DQN feature schema: {version}")
    return V1_DIM if version == "v1" else V2_DIM


def extract_observation(
    state: SplendorState, seat: int, version: str = "v1"
) -> NDArray[np.float32]:
    """Append supply, both panels, noble requirements, seat and endgame flag.

    Rival reserved *count* is public; identities, deck order and purchased-card
    identities never enter v2. Two-player-only v2 keeps seat semantics explicit.
    The original v1 prefix is byte-for-byte unchanged.
    """
    observation_dim(version)
    base = extract_metrics_with_cards(state, seat).astype(np.float32)
    if version == "v1":
        return base
    if len(state.agents) != PLAYERS:
        raise ValueError("public-v2 currently supports two players only")
    extra = [float(state.board.gems.get(c, 0)) for c in COLORS]
    for index in (seat, 1 - seat):
        agent = state.agents[index]
        extra.extend(float(agent.gems.get(c, 0)) for c in COLORS)
        extra.extend(float(len(agent.cards[c])) for c in NORMAL_COLORS)
        extra.append(float(len(agent.cards["yellow"])))
    nobles = state.board.nobles
    for index in range(3):
        cost = nobles[index][1] if index < len(nobles) else {}
        extra.extend(float(cost.get(c, 0)) for c in NORMAL_COLORS)
    extra.extend(
        [
            float(seat),
            float(any(a.score >= WINNING_SCORE_TRESHOLD for a in state.agents)),
        ]
    )
    return np.concatenate((base, np.asarray(extra, dtype=np.float32)))
