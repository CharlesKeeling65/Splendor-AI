"""
Collection of useful utility functions used in the implementation of SplendorEnv.
"""

import numpy as np
from numpy.typing import NDArray

from splendor.splendor.constants import MAX_TIER_CARDS, NUMBER_OF_TIERS, RESERVED
from splendor.splendor.splendor_model import SplendorState
from splendor.splendor.types import ActionType, GemsCount

from .actions import ALL_ACTIONS, Action, ActionEnum, CardPosition


def _gems_key(gems: GemsCount | None) -> tuple[tuple[str, int], ...] | None:
    """
    Convert a gems count (dict) into a hashable key.

    ``None`` and the empty dict must stay distinguishable: buy actions carry
    ``collected_gems=None`` while "no gems to return" is an empty dict - the
    two are different actions.
    """
    return None if gems is None else tuple(sorted(gems.items()))


def action_key(action: Action) -> tuple:
    """
    Convert an Action into a hashable key, preserving the exact equality
    semantics of the Action dataclass (field-by-field comparison).

    This is the identity used by ACTION_INDEX: ``Action`` holds dict fields
    (unhashable), so dict lookups must go through this flattened key.
    """
    position_key = (
        (action.position.tier, action.position.card_index, action.position.reserved_index)
        if action.position is not None
        else None
    )
    return (
        action.type_enum,
        _gems_key(action.collected_gems),
        _gems_key(action.returned_gems),
        position_key,
        action.noble_index,
    )


# Built once at import time (O(3510)); every later lookup is O(1).
ACTION_INDEX: dict[tuple, int] = {
    action_key(action): index for index, action in enumerate(ALL_ACTIONS)
}

# A key collision means ALL_ACTIONS contains duplicate actions - better to
# fail loudly at import than to silently mis-index actions later.
assert len(ACTION_INDEX) == len(ALL_ACTIONS), (
    "duplicate actions detected in ALL_ACTIONS (index keys collide)"
)


def _index_of(action_element: Action) -> int:
    """
    Return the index of the given action in ALL_ACTIONS (O(1) cache lookup).

    Raises a ValueError carrying the offending action when it isn't part of
    ALL_ACTIONS - the previous list.index() based implementation raised a
    bare ValueError deep inside the per-action loop, without any detail
    about which action failed to match.
    """
    try:
        return ACTION_INDEX[action_key(action_element)]
    except KeyError as err:
        raise ValueError(f"action not in ALL_ACTIONS: {action_element}") from err


def _valid_position(state: SplendorState, position: CardPosition) -> bool:
    """
    check if the given card position is a valid position in the given state.
    useful for validating that a position of a card can be purchased/reserved.
    """
    if position.tier not in range(NUMBER_OF_TIERS):
        return False
    if (
        position.card_index not in range(MAX_TIER_CARDS)
        or state.board.dealt[position.tier][position.card_index] is None
    ):
        return False
    return True


def _valid_reserved_position(
    state: SplendorState, position: CardPosition, agent_index: int
) -> bool:
    """
    check if the given reserved card position is a valid position in the given state.
    useful for validating that a position of a reserved card can be purchased.
    """
    return (
        position.reserved_index in range(len(state.agents[agent_index].cards[RESERVED]))
        and state.agents[agent_index].cards[RESERVED][position.reserved_index]
    )


def build_action(
    action_index: int,
    state: SplendorState,
    agent_index: int,
) -> dict:
    """
    Construct the action to be taken from it's action index in the ALL_ACTION list.

    :return: the corresponding action to the action_index, in the format required
             by SplendorGameRule.

    :note: when using this function for building a buying action the function doesn't
           takes into account the wildcard gems (yellow) and the owned cards for the
           conclusion of the returned_gems - this can lead to a broken state where a
           player have a negative amount of gems...
    """
    if action_index not in range(len(ALL_ACTIONS)):
        raise ValueError(f"The action {action_index} isn't a valid action")

    action = ALL_ACTIONS[action_index]

    noble = (
        state.board.nobles[action.noble_index]
        if action.noble_index is not None
        and action.noble_index in range(len(state.board.nobles))
        else None
    )
    card = (
        state.board.dealt[action.position.tier][action.position.card_index]
        if action.position and _valid_position(state, action.position)
        else None
    )
    reserved_card = (
        state.agents[agent_index].cards[RESERVED][action.position.reserved_index]
        if action.position
        and _valid_reserved_position(state, action.position, agent_index)
        else None
    )

    match action.type_enum:
        case ActionEnum.PASS:
            action_to_execute = {
                "type": "pass",
                "noble": noble,
            }
        case ActionEnum.COLLECT_SAME:
            action_to_execute = {
                "type": "collect_same",
                "noble": noble,
                "collected_gems": action.collected_gems,
                "returned_gems": action.returned_gems,
            }
        case ActionEnum.COLLECT_DIFF:
            action_to_execute = {
                "type": "collect_diff",
                "noble": noble,
                "collected_gems": action.collected_gems,
                "returned_gems": action.returned_gems,
            }
        case ActionEnum.RESERVE:
            action_to_execute = {
                "type": "reserve",
                "noble": noble,
                "card": card,
                "collected_gems": action.collected_gems,
                "returned_gems": action.returned_gems,
            }
        case ActionEnum.BUY_AVAILABLE:
            if card is None:
                # this might happen when buying a card but with a
                # wrong index (there is no card at that position).
                raise ValueError(
                    f"Can't build action {action} since there is not card to buy!"
                )
            returned_gems = card.cost

            action_to_execute = {
                "type": "buy_available",
                "noble": noble,
                "card": card,
                "returned_gems": returned_gems,
            }
        case ActionEnum.BUY_RESERVE:
            if reserved_card is None:
                # this might happen when buying a reserved card but with a
                # wrong index.
                raise ValueError(
                    f"Can't build action {action} since there is not card to buy!"
                )
            returned_gems = reserved_card.cost

            action_to_execute = {
                "type": "buy_reserve",
                "noble": noble,
                "card": reserved_card,
                "returned_gems": returned_gems,
            }
        case _:
            raise ValueError(
                f"Unknown action type: {action.type_enum} of the action {action}"
            )

    return action_to_execute


def _slow_create_legal_actions_mask(
    legal_actions: list[ActionType],
    state: SplendorState,
    agent_index: int,
) -> NDArray:
    """
    The pre-cache implementation of create_legal_actions_mask, kept only as
    the reference oracle for the equivalence tests (O(3510) scan per action).
    """
    mask = np.zeros(len(ALL_ACTIONS))

    for legal_action in legal_actions:
        action_element = Action.to_action_element(legal_action, state, agent_index)
        mask[ALL_ACTIONS.index(action_element)] = 1

    return mask


def create_legal_actions_mask(
    legal_actions: list[ActionType],
    state: SplendorState,
    agent_index: int,
) -> NDArray:
    """
    Create an array of shape (len(ALL_ACTIONS),) whose values are 0's or 1's.
    If the at the i'th index the mask[i] == 1 then the i'th action is legal,
    otherwise it's illegal.
    """
    mask = np.zeros(len(ALL_ACTIONS))

    for legal_action in legal_actions:
        action_element = Action.to_action_element(legal_action, state, agent_index)
        mask[_index_of(action_element)] = 1

    return mask


def _slow_create_action_mapping(
    legal_actions: list[ActionType], state: SplendorState, agent_index: int
) -> dict[int, ActionType]:
    """
    The pre-cache implementation of create_action_mapping, kept only as
    the reference oracle for the equivalence tests (O(3510) scan per action).
    """
    mapping = {
        ALL_ACTIONS.index(
            Action.to_action_element(legal_action, state, agent_index)
        ): legal_action
        for legal_action in legal_actions
    }

    return mapping


def create_action_mapping(
    legal_actions: list[ActionType], state: SplendorState, agent_index: int
) -> dict[int, ActionType]:
    """
    Create the mapping between action indices to legal actions.
    This would be in use by both SplendorEnv & by the PPO agent.
    """
    mapping = {
        _index_of(
            Action.to_action_element(legal_action, state, agent_index)
        ): legal_action
        for legal_action in legal_actions
    }

    return mapping
