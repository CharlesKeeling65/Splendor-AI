"""Tests for the isolated same-weight MCTS gate adapter."""

from pathlib import Path

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.dqn.features import V2_DIM
from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.utils import save_model
from splendor.agents.our_agents.policy_imitation.evaluation import evaluate_candidate
from splendor.agents.our_agents.policy_imitation.mcts_gate import (
    build_mcts_candidate,
    summarize_search_cost,
)
from splendor.agents.our_agents.policy_imitation.policies import CandidateSpec


def test_mcts_candidate_records_search_work(tmp_path: Path) -> None:
    checkpoint = tmp_path / "search.pth"
    save_model(
        QNetwork(
            input_dim=V2_DIM,
            hidden_layers=(16, 16),
            feature_version="public-v2",
            auxiliary_heads=True,
        ),
        checkpoint,
    )
    records: list[dict[str, float | int]] = []
    candidate = build_mcts_candidate(
        checkpoint,
        simulations=1,
        stats_sink=records.append,
    )
    opponent = CandidateSpec("random", "fixed_baseline", RandomAgent)

    result = evaluate_candidate(candidate, opponent, [812345])

    assert result["games"] == 2
    assert result["failed_games"] == 0
    assert result["candidate_illegal_actions"] == 0
    assert records
    assert all(record["simulations"] == 1 for record in records)
    assert all(int(record["tree_nodes"]) >= 1 for record in records)
    assert summarize_search_cost(records)["fallback_queries"] == 0
