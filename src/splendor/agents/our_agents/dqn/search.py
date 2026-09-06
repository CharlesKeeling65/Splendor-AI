"""Experimental sampled-hidden-state PUCT; never bootstrap from score-return Q.

This is a local information-set approximation, not exact imperfect-information
AlphaZero: unseen rival reservations are resampled with the remaining deck.
Search requires full public purchase history (real card identities) and is not
enabled on browser pseudo states. Original state, rule and global RNGs are inert.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule

from .features import PLAYERS, extract_observation
from .network import ACTION_DIM, QNetwork


@dataclass
class Node:
    """Edge values use the perspective of the player acting at this node."""

    seat: int
    actions: NDArray[np.int64]
    prior: NDArray[np.float64]
    visits: NDArray[np.float64]
    total: NDArray[np.float64]


def outcome(rule: SplendorGameRule, seat: int) -> float:
    """Use the engine's completed-game tiebreak, not a score threshold."""
    state = rule.current_game_state
    return float(np.sign(rule.calScore(state, seat) - rule.calScore(state, 1 - seat)))


def sample_hidden(
    rule: SplendorGameRule, root_seat: int, rng: np.random.Generator
) -> SplendorGameRule:
    """Remove deck-order and rival-reservation knowledge before each simulation.

    Sort the unseen union before shuffling: equal public histories yield the
    same samples even if the real deck order/reserved identities differ.
    """
    copied = deepcopy(rule)
    state = copied.current_game_state
    rival = state.agents[1 - root_seat]
    for tier, deck in enumerate(state.board.decks):
        slots = [i for i, c in enumerate(rival.cards["yellow"]) if c.deck_id == tier]
        unseen = sorted(
            [*deck, *(rival.cards["yellow"][i] for i in slots)], key=lambda c: c.code
        )
        rng.shuffle(unseen)
        for index in slots:
            rival.cards["yellow"][index] = unseen.pop()
        state.board.decks[tier] = unseen
    return copied


@torch.no_grad()
def search_policy(  # noqa: C901, PLR0913, PLR0915 - bounded PUCT traversal
    net: QNetwork,
    rule: SplendorGameRule,
    simulations: int,
    rng: np.random.Generator,
    *,
    max_depth: int = 24,
    root_noise: bool = False,
) -> NDArray[np.float32]:
    """Return root visits over legal actions with fresh hidden sampling per rollout."""
    if simulations < 1 or max_depth < 1:
        raise ValueError("search budget and depth must be positive")
    if rule.num_of_agent != PLAYERS or not net.auxiliary_heads:
        raise ValueError("search needs two players and trained policy/value heads")
    if rule.gameEnds():
        raise ValueError("cannot search a terminal position")
    device = next(net.parameters()).device
    tree: dict[bytes, Node] = {}

    def leaf(current: SplendorGameRule) -> tuple[bytes, Node, float]:
        seat = current.current_agent_index
        state = current.current_game_state
        mapping = create_action_mapping(
            current.getLegalActions(state, seat), state, seat
        )
        indices = np.asarray(sorted(mapping), dtype=np.int64)
        obs = extract_observation(state, seat, net.feature_version)
        key = bytes([seat]) + obs.tobytes() + indices.tobytes()
        if key in tree:
            return key, tree[key], 0.0
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        mask[indices] = 1
        logits, value = net.policy_value(
            torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device)
        )
        prior = logits.softmax(-1)[0, indices].cpu().numpy().astype(np.float64)
        prior /= prior.sum()
        node = Node(
            seat, indices, prior, np.zeros(len(indices)), np.zeros(len(indices))
        )
        tree[key] = node
        return key, node, float(value.item())

    root_seat = rule.current_agent_index
    _, root, _ = leaf(rule)
    if root_noise:
        root.prior = 0.75 * root.prior + 0.25 * rng.dirichlet(
            np.full(len(root.prior), 0.3)
        )
    for _ in range(simulations):
        current = sample_hidden(rule, root_seat, rng)
        path: list[tuple[Node, int]] = []
        value = 0.0
        value_seat = root_seat
        for _depth in range(max_depth):
            value_seat = current.current_agent_index
            if current.gameEnds():
                value = outcome(current, value_seat)
                break
            previous_size = len(tree)
            _, node, value = leaf(current)
            if len(tree) > previous_size:
                break
            scores = node.total / np.maximum(node.visits, 1) + (
                1.5 * node.prior * np.sqrt(node.visits.sum() + 1) / (1 + node.visits)
            )
            edge = int(np.argmax(scores))
            path.append((node, edge))
            state = current.current_game_state
            mapping = create_action_mapping(
                current.getLegalActions(state, value_seat), state, value_seat
            )
            current.update(mapping[int(node.actions[edge])])
        else:
            value_seat = current.current_agent_index
            if current.gameEnds():
                value = outcome(current, value_seat)
            else:
                obs = extract_observation(
                    current.current_game_state, value_seat, net.feature_version
                )
                _, predicted = net.policy_value(
                    torch.from_numpy(obs).to(device),
                    torch.ones(ACTION_DIM, device=device),
                )
                value = float(predicted.item())
        for node, edge in path:
            node.visits[edge] += 1
            node.total[edge] += value if node.seat == value_seat else -value
    pi = np.zeros(ACTION_DIM, dtype=np.float32)
    pi[root.actions] = root.visits / root.visits.sum()
    return pi
