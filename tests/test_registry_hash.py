"""Stable primitive hashes for ScenarioV1 engine registries."""

from dataclasses import replace

from splendor.agents.our_agents.policy_imitation.scenario import (
    action_registry_hash,
    action_registry_hash_for,
    card_registry_hash,
)
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, Action, ActionEnum
from splendor.splendor.splendor_utils import CARDS, NOBLES


def test_card_and_action_registry_hashes_are_complete_and_stable() -> None:
    assert len(CARDS) == 90
    assert len(NOBLES) == 10
    assert len(ALL_ACTIONS) == 3510
    assert card_registry_hash() == (
        "4b392c9f743b992a753d6400ddcce053215545c58c83b07c580186440507cd02"
    )
    assert action_registry_hash() == (
        "105d4efcf27a258107d3a3c028c7f0ba3fbe4bf924728e2612c6b10b2253fe16"
    )


def test_action_registry_hash_binds_index_order() -> None:
    swapped = [ALL_ACTIONS[1], ALL_ACTIONS[0], *ALL_ACTIONS[2:]]
    assert action_registry_hash_for(swapped) != action_registry_hash()
    assert action_registry_hash_for(ALL_ACTIONS) == action_registry_hash()


def test_action_encoding_distinguishes_none_from_empty_gems() -> None:
    absent = Action(type_enum=ActionEnum.PASS, collected_gems=None)
    empty = replace(absent, collected_gems={})
    assert action_registry_hash_for([absent]) != action_registry_hash_for([empty])
