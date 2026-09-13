"""Roadmap F2 tests: multi-determinized-tree search machinery."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from splendor.agents.our_agents.dqn.network import QNetwork
from splendor.agents.our_agents.dqn.search import (
    allocate_tree_budget,
    multi_tree_search_policy,
)
from splendor.splendor.splendor_model import SplendorGameRule


def _aux_net(seed: int) -> QNetwork:
    torch.manual_seed(seed)
    return QNetwork(auxiliary_heads=True).eval()


def test_allocate_tree_budget_uniform_and_priority() -> None:
    uniform = allocate_tree_budget(100, 4, "uniform")
    assert uniform.tolist() == [25, 25, 25, 25]
    remainder = allocate_tree_budget(101, 4, "uniform")
    assert remainder.sum() == 101 and remainder.max() - remainder.min() <= 1
    priority = allocate_tree_budget(
        100, 4, "priority", root_value_stds=np.array([0.4, 0.3, 0.2, 0.1])
    )
    assert priority.sum() == 100
    assert priority[0] > priority[-1]  # higher spread gets more work
    flat = allocate_tree_budget(
        100, 4, "priority", root_value_stds=np.array([0.0, 0.0, 0.0, 0.0])
    )
    assert flat.sum() == 100 and flat.min() >= 1
    with pytest.raises(ValueError, match="unknown allocation"):
        allocate_tree_budget(100, 4, "magic")
    with pytest.raises(ValueError, match="root value stds"):
        allocate_tree_budget(100, 4, "priority")


def _nonterminal_rule() -> SplendorGameRule:
    rule = SplendorGameRule(2)
    assert not rule.gameEnds()
    return rule


def test_multi_tree_search_returns_normalized_root_policy() -> None:
    net = _aux_net(11)
    rule = _nonterminal_rule()
    rng = np.random.default_rng(825_600)
    stats: dict[str, float | int] = {}
    policy = multi_tree_search_policy(
        net,
        rule,
        8,
        rng,
        n_trees=4,
        allocation="uniform",
        max_depth=3,
        stats=stats,
    )
    assert policy.shape == (3510,)
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert policy[policy > 0].sum() == pytest.approx(1.0)
    assert stats["n_trees"] == 4
    assert stats["simulations"] == 8
    assert stats["tree_nodes"] >= 4


def test_multi_tree_priority_allocation_runs() -> None:
    net = _aux_net(13)
    rule = _nonterminal_rule()
    rng = np.random.default_rng(825_601)
    stats: dict[str, float | int] = {}
    policy = multi_tree_search_policy(
        net,
        rule,
        10,
        rng,
        n_trees=4,
        allocation="priority",
        max_depth=3,
        stats=stats,
    )
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert stats["allocation"] == 1
    assert stats["budget_max"] >= stats["budget_min"]


def test_multi_tree_rejects_invalid_configuration() -> None:
    net = _aux_net(17)
    rule = _nonterminal_rule()
    with pytest.raises(ValueError, match="simulations >= n_trees"):
        multi_tree_search_policy(net, rule, 2, np.random.default_rng(1), n_trees=4)
    plain = QNetwork(auxiliary_heads=False).eval()
    with pytest.raises(ValueError, match="policy/value heads"):
        multi_tree_search_policy(plain, rule, 8, np.random.default_rng(1), n_trees=4)
