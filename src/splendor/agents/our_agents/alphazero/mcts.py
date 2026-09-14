"""AlphaZero-style PUCT search over determinization ensembles.

Differences from :mod:`splendor.agents.our_agents.dqn.search` (roadmap F2):

* **Pluggable evaluator** - Z1 runs uniform priors + zero leaf values (pure
  search anchor), Z2 plugs the trained policy/value network;
* **In-place traversal** - each determinized tree deep-copies the rule *once*
  (inside ``sample_hidden``) and reuses it across all simulations through the
  exact apply/undo :class:`~.state_utils.Transactor`, instead of one deepcopy
  per simulation;
* **Action objects stored on nodes** - ``getLegalActions`` deep-copies, so
  expansion builds the legal-action table once per node and traversal applies
  the stored action objects directly;
* **Raw visit aggregation** - trees are combined by summing raw visit counts
  (the self-play training target; equals the F2 normalized average under
  uniform budget allocation).

Root Dirichlet noise is drawn *once* per search call and mixed into every
tree root: the root public state (hence prior and legal set) is identical
across determinizations, and one noise realization is the intended
exploration directive, not a per-tree arbitrary one.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from splendor.agents.our_agents.alphazero.evaluator import Evaluator
from splendor.agents.our_agents.alphazero.state_utils import (
    Transactor,
    TransitionSnapshot,
    legal_action_table,
    state_fingerprint,
)
from splendor.agents.our_agents.dqn.features import PLAYERS, extract_observation
from splendor.agents.our_agents.dqn.network import ACTION_DIM
from splendor.agents.our_agents.dqn.search import (
    allocate_tree_budget,
    outcome,
    sample_hidden,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType


@dataclass
class _Node:
    """Edge values use the perspective of the player acting at this node."""

    seat: int
    indices: list[int]
    actions: list[ActionType]
    prior: NDArray[np.float64]
    visits: NDArray[np.float64]
    total: NDArray[np.float64]

@dataclass
class AzSearchResult:
    """Root visit distribution plus aggregate search work counters."""

    indices: list[int]
    visits: NDArray[np.float64]
    pi: NDArray[np.float32]
    stats: dict[str, float | int] = field(default_factory=dict)

def _transposition_key(
    state: SplendorState, seat: int, indices: list[int], kind: str
) -> bytes:
    if kind == "fingerprint":
        body = state_fingerprint(state, include_last_action=False)
    elif kind == "obs":
        body = extract_observation(state, seat, "public-v2").tobytes()
    else:  # pragma: no cover - guarded by the evaluator contract
        raise ValueError(f"unknown transposition key kind {kind!r}")
    return bytes([seat]) + body + np.asarray(indices, dtype=np.int64).tobytes()

def _select_child(node: _Node, c_puct: float) -> int:
    q = node.total / np.maximum(node.visits, 1.0)
    exploration = (
        c_puct
        * node.prior
        * np.sqrt(float(node.visits.sum()) + 1.0)
        / (1.0 + node.visits)
    )
    return int(np.argmax(q + exploration))

def az_search(  # noqa: C901, PLR0913, PLR0915 - one keyword per search knob
    rule: SplendorGameRule,
    evaluator: Evaluator,
    rng: np.random.Generator,
    *,
    simulations: int = 100,
    n_trees: int = 4,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 1.0,
    dirichlet_epsilon: float = 0.25,
    root_noise: bool = False,
    max_depth: int = 24,
) -> AzSearchResult:
    """Search the current position; return root visits over legal actions.

    ``n_trees`` determinizations of the hidden state (unseen decks + rival
    face-down reservations) are sampled once each; the budget is split across
    trees (uniform) and every tree runs PUCT on its own in-place mutable copy,
    undoing each simulation back to the root. Requires a two-player
    non-terminal position.
    """
    if simulations < n_trees or n_trees < 1:
        raise ValueError("need simulations >= n_trees >= 1")
    if rule.num_of_agent != PLAYERS:
        raise ValueError("az_search supports two-player games only")
    if rule.gameEnds():
        raise ValueError("cannot search a terminal position")

    key_kind = evaluator.key_kind
    transactor = Transactor()
    root_seat = rule.getCurrentAgentIndex()
    started = time.perf_counter()

    def expand(
        current: SplendorGameRule, seat: int, tree: dict[bytes, _Node]
    ) -> tuple[_Node, bool, float]:
        indices, actions = legal_action_table(current, seat)
        key = _transposition_key(
            current.current_game_state, seat, indices, key_kind
        )
        existing = tree.get(key)
        if existing is not None:
            return existing, False, 0.0
        prior, value = evaluator.evaluate(
            current.current_game_state, seat, indices, actions
        )
        node = _Node(
            seat=seat,
            indices=indices,
            actions=actions,
            prior=prior,
            visits=np.zeros(len(indices), dtype=np.float64),
            total=np.zeros(len(indices), dtype=np.float64),
        )
        tree[key] = node
        return node, True, value

    def simulate(
        determinization: SplendorGameRule, tree: dict[bytes, _Node]
    ) -> tuple[bool, int]:
        """One PUCT simulation on a mutable determinization; undo afterwards."""
        path: list[tuple[_Node, int]] = []
        undo_log: list[tuple[ActionType, int, TransitionSnapshot]] = []
        seat = root_seat
        value = 0.0
        value_seat = root_seat
        terminal = False
        for _depth in range(max_depth):
            if determinization.gameEnds():
                value = outcome(determinization, seat)
                value_seat = seat
                terminal = True
                break
            node, fresh, leaf_value = expand(determinization, seat, tree)
            if fresh:
                value, value_seat = leaf_value, seat
                break
            edge = _select_child(node, c_puct)
            path.append((node, edge))
            action = node.actions[edge]
            undo_log.append(
                (action, seat, transactor.apply(determinization, action, seat))
            )
            seat = 1 - seat
        else:
            indices, actions = legal_action_table(determinization, seat)
            _prior, value = evaluator.evaluate(
                determinization.current_game_state, seat, indices, actions
            )
            value_seat = seat
        for node, edge in path:
            node.visits[edge] += 1.0
            node.total[edge] += value if node.seat == value_seat else -value
        for action, acted_seat, snapshot in reversed(undo_log):
            transactor.undo(determinization, action, acted_seat, snapshot)
        return terminal, len(path)

    trees: list[dict[bytes, _Node]] = []
    roots: list[_Node] = []
    determinizations: list[SplendorGameRule] = []
    for _tree in range(n_trees):
        determinization = sample_hidden(rule, root_seat, rng)
        tree: dict[bytes, _Node] = {}
        root_node, _fresh, _value = expand(determinization, root_seat, tree)
        determinizations.append(determinization)
        trees.append(tree)
        roots.append(root_node)

    root_indices = roots[0].indices
    if root_noise:
        noise = rng.dirichlet(np.full(len(root_indices), dirichlet_alpha))
        for root_node in roots:
            root_node.prior = (
                1.0 - dirichlet_epsilon
            ) * root_node.prior + dirichlet_epsilon * noise

    budget = allocate_tree_budget(simulations, n_trees, "uniform")
    terminal_hits = 0
    max_depth_reached = 0
    for tree_index, tree_budget in enumerate(budget):
        for _sim in range(int(tree_budget)):
            terminal, depth = simulate(determinizations[tree_index], trees[tree_index])
            terminal_hits += int(terminal)
            max_depth_reached = max(max_depth_reached, depth)

    combined = np.zeros(ACTION_DIM, dtype=np.float64)
    for root_node in roots:
        combined[root_node.indices] += root_node.visits
    total = float(combined.sum())
    if total <= 0:
        raise RuntimeError("search produced no root visits")
    pi = np.zeros(ACTION_DIM, dtype=np.float32)
    pi[root_indices] = (combined[root_indices] / total).astype(np.float32)
    stats: dict[str, float | int] = {
        "simulations": simulations,
        "n_trees": n_trees,
        "tree_nodes": sum(len(tree) for tree in trees),
        "root_legal_actions": len(root_indices),
        "max_depth_reached": max_depth_reached,
        "terminal_hits": terminal_hits,
        "elapsed_s": time.perf_counter() - started,
    }
    return AzSearchResult(
        indices=root_indices, visits=combined[root_indices], pi=pi, stats=stats
    )

def select_action(
    result: AzSearchResult, temperature: float, rng: np.random.Generator
) -> int:
    """Sample a root action index by visit counts raised to 1/temperature.

    ``temperature <= 0`` degenerates to the argmax (evaluation / late-game
    play). Returns the global action index, not the position in ``indices``.
    """
    if temperature <= 0:
        position = int(np.argmax(result.visits))
        return result.indices[position]
    weights = np.power(np.maximum(result.visits, 0.0), 1.0 / temperature)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        position = int(np.argmax(result.visits))
        return result.indices[position]
    position = int(rng.choice(len(result.indices), p=weights / weights.sum()))
    return result.indices[position]
