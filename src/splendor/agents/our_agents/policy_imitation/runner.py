"""Safe, instrumented calls into the repository's legacy Agent interface."""

import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from splendor.splendor.gym.envs.utils import create_action_mapping
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .protocol import SeedLineage, isolated_seed

_LEGACY_RNG_ADAPTER_LOCK = threading.RLock()


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

    def __deepcopy__(self, memo: dict[int, Any]) -> "CountingRuleProxy":
        """Copy the wrapped rule without recursing through ``__getattr__``."""
        copied = CountingRuleProxy(deepcopy(self._rule, memo))
        copied.successor_calls = self.successor_calls
        return copied

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
    *,
    rng_lineage: SeedLineage | None = None,
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
    # Copy the query inputs as one object graph.  Legal actions contain card
    # objects from the state, and legacy search code may read the state from
    # the rule instead of from its explicit argument.  Rebind the copied rule
    # after deepcopy as a defensive guard for callers that supplied a rule and
    # state copied independently (as the probe API does).
    copied_actions, copied_state, copied_rule = deepcopy(
        (actions, game_state, game_rule)
    )
    copied_rule.current_game_state = copied_state
    proxy = CountingRuleProxy(copied_rule)
    try:
        # Legacy agents do not accept an RNG object.  The formal runner gives
        # each decision its own derived seed and restores every global stream
        # immediately afterwards; legacy callers retain the original path.
        rng_scope = (
            isolated_seed(rng_lineage.seed63)
            if rng_lineage is not None
            else nullcontext()
        )
        if rng_lineage is None:
            with rng_scope:
                selected = agent.SelectAction(
                    copied_actions,
                    copied_state,
                    proxy,  # type: ignore[arg-type]
                )
        else:
            # Global RNG compatibility is necessarily process-wide.  Formal
            # workers use spawn, and this lock prevents accidental threads in
            # one worker from interleaving the save/seed/restore transaction.
            with _LEGACY_RNG_ADAPTER_LOCK, rng_scope:
                selected = agent.SelectAction(
                    copied_actions,
                    copied_state,
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
