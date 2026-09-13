"""Exact in-place state transitions for search: fingerprint, snapshot, undo.

The engine's ``generateSuccessor`` mutates the state in place and
``generatePredecessor`` reverts it (the minmax.py search pattern). Two details
are *not* restored by the engine itself:

* ``board.nobles`` ordering - the successor deletes a taken noble from the
  middle of the list while the predecessor appends it back at the end, so a
  taken noble that was not last leaves the list permuted;
* ``agent.last_action`` - both directions overwrite it with the same action,
  so after an undo it holds the reverted action instead of the pre-action
  value.

:class:`Transactor` snapshots exactly those two around every forward step so
one mutable determinization can be reused across all MCTS simulations instead
of deep-copying per rollout. :func:`state_fingerprint` serializes the full
state canonically (deck order included) and is the exactness oracle used by
the unit tests: after ``undo(apply(s))`` the fingerprint must equal the
original, byte for byte.
"""

from dataclasses import dataclass

from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType

_COLOUR_ORDER = ("black", "red", "yellow", "green", "blue", "white")


def _action_repr(action: ActionType | None) -> str:
    """Canonical one-line serialization of an action (cards by code)."""
    if action is None:
        return "-"
    parts = [str(action["type"])]
    if action.get("card") is not None:
        parts.append(str(action["card"].code))
    for key in ("collected_gems", "returned_gems"):
        gems = action.get(key) or {}
        parts.append(",".join(f"{c}:{gems[c]}" for c in _COLOUR_ORDER if c in gems))
    position = action.get("card_position")
    if position is not None:
        parts.append(f"{position[0]},{position[1]}")
    noble = action.get("noble")
    if noble is not None:
        parts.append(str(noble[0]))
    return "|".join(parts)


def state_fingerprint(
    state: SplendorState, *, include_last_action: bool = True
) -> bytes:
    """Serialize the full state canonically; equal states -> equal bytes.

    Deck *order* is part of the fingerprint: the undo-exactness oracle must
    catch ordering drift even when the card multisets match. Agent traces are
    folded in by length + last reward only (actions are re-derivable and the
    replayed action repr is captured via ``last_action``).

    ``include_last_action=False`` is the search *transposition* variant: no
    engine decision path reads ``last_action`` (verified for
    getLegalActions/gameEnds/calScore and both feature extractors), so two
    positions differing only in it are strategically identical and should
    share a tree node. The undo oracle keeps it included - it is exactly one
    of the two details ``generatePredecessor`` leaves unrestored.
    """
    out: list[str] = []
    board = state.board
    for tier in range(3):
        out.append(
            "deck:" + ",".join(card.code for card in board.decks[tier])
        )
        out.append(
            "dealt:"
            + ",".join(
                card.code if card is not None else "-" for card in board.dealt[tier]
            )
        )
    out.append(
        "gems:" + ",".join(f"{c}:{board.gems[c]}" for c in _COLOUR_ORDER)
    )
    out.append("nobles:" + ",".join(str(n[0]) for n in board.nobles))
    for agent in state.agents:
        out.append(
            f"agent:{agent.id}:{agent.score}:{int(agent.passed)}:"
            + ",".join(f"{c}:{agent.gems[c]}" for c in _COLOUR_ORDER)
        )
        for colour in _COLOUR_ORDER:
            out.append(
                f"cards:{agent.id}:{colour}:"
                + ",".join(card.code for card in agent.cards[colour])
            )
        out.append(
            "agentnobles:" + ",".join(str(n[0]) for n in agent.nobles)
        )
        trace = agent.agent_trace.action_reward
        out.append(f"trace:{agent.id}:{len(trace)}")
        if include_last_action:
            out.append(f"last:{agent.id}:{_action_repr(agent.last_action)}")
    return "|".join(out).encode("utf-8")


@dataclass(frozen=True)
class TransitionSnapshot:
    """Engine-unrestored details captured before one in-place forward step."""

    nobles: tuple[tuple[str, dict[str, int]], ...]
    last_action: ActionType | None


class Transactor:
    """Apply/undo engine transitions with exact-state snapshots.

    ``apply`` mirrors ``SplendorGameRule.update`` but does not advance
    ``current_agent_index``/``action_counter`` (search tracks the acting seat
    itself); ``undo`` reverts via ``generatePredecessor`` and restores the two
    details the engine leaves permuted. The public seat bookkeeping of the
    underlying rule is therefore untouched by a full apply/undo cycle.
    """

    def apply(
        self, rule: SplendorGameRule, action: ActionType, seat: int
    ) -> TransitionSnapshot:
        state = rule.current_game_state
        agent = state.agents[seat]
        snapshot = TransitionSnapshot(
            nobles=tuple(state.board.nobles), last_action=agent.last_action
        )
        rule.generateSuccessor(state, action, seat)
        return snapshot

    def undo(
        self,
        rule: SplendorGameRule,
        action: ActionType,
        seat: int,
        snapshot: TransitionSnapshot,
    ) -> None:
        state = rule.current_game_state
        rule.generatePredecessor(state, action, seat)
        state.board.nobles[:] = list(snapshot.nobles)
        state.agents[seat].last_action = snapshot.last_action


def legal_action_table(
    rule: SplendorGameRule, seat: int
) -> tuple[list[int], list[ActionType]]:
    """Sorted legal-action indices plus the aligned action objects.

    Indices are sorted for deterministic expansion order (the engine's set
    enumeration is hash-order sensitive); ``actions[i]`` is the engine action
    for ``indices[i]``. The action objects reference persistent engine cards,
    so they stay valid across in-place apply/undo cycles and are stored on
    tree nodes to avoid re-running ``getLegalActions`` (which deep-copies)
    during traversal.
    """
    mapping = create_action_mapping(
        rule.getLegalActions(rule.current_game_state, seat),
        rule.current_game_state,
        seat,
    )
    indices = sorted(mapping)
    return indices, [mapping[index] for index in indices]
