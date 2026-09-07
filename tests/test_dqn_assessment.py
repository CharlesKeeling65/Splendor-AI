"""Reject plausible-looking rates when their underlying game records disagree."""

from copy import deepcopy
from typing import Any

import pytest

from splendor.agents.our_agents.dqn.assessment import audit_report


def report() -> dict[str, Any]:
    return {
        "wins": 1,
        "draws": 0,
        "losses": 1,
        "games": 2,
        "win_rate": 0.5,
        "mean_score": 12,
        "records": [
            {"seed": 10, "seat": 0, "score": 16, "rival_score": 10, "outcome": 1},
            {"seed": 10, "seat": 1, "score": 8, "rival_score": 15, "outcome": -1},
        ],
    }


def test_audit_recomputes_counts_and_rates():
    audit_report(report(), [10])
    bad = deepcopy(report())
    bad["wins"] = 2
    with pytest.raises(ValueError, match="W/D/L"):
        audit_report(bad, [10])


def test_audit_rejects_duplicates_and_wrong_outcome():
    bad = report()
    bad["records"][1]["seat"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        audit_report(bad, [10])
    bad = report()
    bad["records"][1]["outcome"] = 1
    with pytest.raises(ValueError, match="outcome"):
        audit_report(bad, [10])
