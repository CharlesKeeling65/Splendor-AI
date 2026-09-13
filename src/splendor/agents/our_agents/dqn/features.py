"""Versioned, public-information-only DQN observations; v1 stays unchanged.

The public-v2 implementations live in the engine layer
(:mod:`splendor.splendor.features_v2`, roadmap 2026-09-12 §B1); this module
re-exports them so every existing importer keeps working unchanged.
"""

import numpy as np
from numpy.typing import NDArray

from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.features_v2 import (
    COLORS,
    KNOWN_SCHEMAS,
    LEGACY_PLAYERS,
    PANEL_DIM,
    SCHEMA_PUBLIC_V2,
    SCHEMA_PUBLIC_V2_MULTI,
    SCHEMA_V1,
    V1_DIM,
    V2_DIM,
    V2_MULTI_DIM,
    agent_panel,
    extract_observation_public_v2,
    extract_observation_public_v2_multi,
    extract_observation_v2,
    noble_costs,
    observation_dim,
)
from splendor.splendor.splendor_model import SplendorState

#: Backward-compatible alias; the legacy 312-dim schema is two-seat only.
PLAYERS = LEGACY_PLAYERS

__all__ = [
    "COLORS",
    "KNOWN_SCHEMAS",
    "LEGACY_PLAYERS",
    "PANEL_DIM",
    "PLAYERS",
    "SCHEMA_PUBLIC_V2",
    "SCHEMA_PUBLIC_V2_MULTI",
    "SCHEMA_V1",
    "V1_DIM",
    "V2_DIM",
    "V2_MULTI_DIM",
    "agent_panel",
    "extract_observation",
    "extract_observation_public_v2",
    "extract_observation_public_v2_multi",
    "extract_observation_v2",
    "noble_costs",
    "observation_dim",
]


def extract_observation(
    state: SplendorState, seat: int, version: str = "v1"
) -> NDArray[np.float32]:
    """Append supply, panels, noble requirements, seat and endgame flag.

    Rival reserved *count* is public; identities, deck order and purchased-card
    identities never enter v2. The original v1 prefix is byte-for-byte
    unchanged.
    """
    if version == SCHEMA_V1:
        return extract_metrics_with_cards(state, seat).astype(np.float32)
    return extract_observation_v2(state, seat, version)
