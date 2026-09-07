"""Safe, instrumented calls into the repository's legacy Agent interface."""

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class TeacherDecisionError(RuntimeError):
    """A teacher failed to provide one of the currently legal actions."""

    def __init__(
        self,
        message: str,
        *,
        illegal: bool,
        elapsed_seconds: float,
        search_nodes: int,
    ) -> None:
        super().__init__(message)
        self.illegal = illegal
        self.elapsed_seconds = elapsed_seconds
        self.search_nodes = search_nodes


class CountingRuleProxy:
    """Delegate the legacy rule while counting successor calls made by search."""

    def __init__(self, rule: SplendorGameRule) -> None:
        self._rule = rule
        self.successor_calls = 0

    def generateSuccessor(
        self, state: SplendorState, action: ActionType, agent_id: int
    ) -> SplendorState:
        self.successor_calls += 1
        return self._rule.generateSuccessor(state, action, agent_id)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - legacy delegation
        return getattr(self._rule, name)


@dataclass(frozen=True)
class DecisionResult:
    """A validated action plus cost counters for one policy query."""

    action: ActionType
    action_index: int
    elapsed_seconds: float
    search_nodes: int


def _find_action_index(
    mapping: dict[int, ActionType], selected: ActionType
) -> int | None:
    """Compare action dictionaries without hashing cards or nested dictionaries."""
    for index, legal_action in mapping.items():
        if legal_action == selected:
            return index
    return None


def select_action(
    agent: Agent,
    actions: list[ActionType],
    game_state: SplendorState,
    game_rule: SplendorGameRule,
) -> DecisionResult:
    """Call an agent on defensive copies, then validate its action by the mask.

    The engine mutates states in some search paths. Passing copies ensures a
    teacher cannot corrupt the game being evaluated; the returned action is
    normalized against the original legal-action list before execution.
    """
    if not actions:
        raise TeacherDecisionError(
            "teacher received no legal actions",
            illegal=True,
            elapsed_seconds=0.0,
            search_nodes=0,
        )
    began = time.perf_counter()
    proxy = CountingRuleProxy(deepcopy(game_rule))
    try:
        selected = agent.SelectAction(
            deepcopy(actions),
            deepcopy(game_state),
            proxy,  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise TeacherDecisionError(
            f"teacher {agent.__class__.__name__} raised {type(exc).__name__}: {exc}",
            illegal=False,
            elapsed_seconds=time.perf_counter() - began,
            search_nodes=proxy.successor_calls,
        ) from exc
    elapsed = time.perf_counter() - began
    mapping = create_action_mapping(actions, game_state, agent.id)
    action_index = _find_action_index(mapping, selected)
    if action_index is None:
        raise TeacherDecisionError(
            f"teacher {agent.__class__.__name__} returned an illegal action",
            illegal=True,
            elapsed_seconds=elapsed,
            search_nodes=proxy.successor_calls,
        )
    return DecisionResult(
        action=selected,
        action_index=action_index,
        elapsed_seconds=elapsed,
        search_nodes=proxy.successor_calls,
    )
