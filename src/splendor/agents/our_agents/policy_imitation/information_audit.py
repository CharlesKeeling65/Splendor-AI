"""Empirical information-set audits for teacher candidates."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from splendor.agents.our_agents.dqn.features import extract_observation
from splendor.splendor.gym.envs.utils import create_legal_actions_mask
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType

from .evaluation import DecisionProbe, collect_decision_probes
from .policies import CandidateSpec
from .protocol import isolated_seed
from .runner import TeacherDecisionError, select_action

StatePerturbation = Callable[[SplendorState, int], bool]


def _shuffle_hidden_decks(state: SplendorState, _seat: int) -> bool:
    """Reverse unseen deck tails while keeping the displayed board unchanged."""
    changed = False
    for deck in state.board.decks:
        if len(deck) > 1:
            deck.reverse()
            changed = True
    return changed


def _swap_opponent_reserved_card(state: SplendorState, seat: int) -> bool:
    """Replace an opponent's reserved identity without changing its public count."""
    reserved = state.agents[1 - seat].cards["yellow"]
    if not reserved:
        return False
    for deck in state.board.decks:
        if deck:
            reserved[0], deck[0] = deck[0], reserved[0]
            return True
    return False


def _select_probe_action(  # noqa: PLR0913 - audit inputs are explicit
    candidate: CandidateSpec,
    state: SplendorState,
    rule: SplendorGameRule,
    actions: list[ActionType],
    *,
    seat: int,
    seed: int,
) -> tuple[int | None, str | None]:
    """Query a fresh policy copy under a repeatable RNG state."""
    with isolated_seed(seed):
        try:
            decision = select_action(
                candidate.build(seat), actions, state, rule
            )
        except TeacherDecisionError as exc:
            return None, str(exc)
    return decision.action_index, None


def _probe_variant(
    candidate: CandidateSpec,
    probe: DecisionProbe,
    perturbation: StatePerturbation,
    feature_version: str,
    *,
    ordinal: int,
) -> dict[str, Any]:
    """Compare one hidden-state perturbation after checking obs/mask equality."""
    base_state = deepcopy(probe.state)
    base_rule = deepcopy(probe.rule)
    base_rule.current_game_state = base_state
    changed_state = deepcopy(base_state)
    changed = perturbation(changed_state, probe.seat)
    if not changed:
        return {"attempted": False, "reason": "state has no perturbable hidden component"}
    changed_rule = deepcopy(base_rule)
    changed_rule.current_game_state = changed_state
    changed_actions = changed_rule.getLegalActions(changed_state, probe.seat)
    base_observation = extract_observation(base_state, probe.seat, feature_version)
    changed_observation = extract_observation(
        changed_state, probe.seat, feature_version
    )
    base_mask = create_legal_actions_mask(
        probe.actions, base_state, probe.seat
    ).astype(np.uint8)
    changed_mask = create_legal_actions_mask(
        changed_actions, changed_state, probe.seat
    ).astype(np.uint8)
    if not np.array_equal(base_observation, changed_observation) or not np.array_equal(
        base_mask, changed_mask
    ):
        return {
            "attempted": True,
            "projection_match": False,
            "reason": "perturbation changed the student's observation or legal mask",
        }
    base_action, base_error = _select_probe_action(
        candidate,
        base_state,
        base_rule,
        probe.actions,
        seat=probe.seat,
        seed=probe.seed + ordinal * 2,
    )
    changed_action, changed_error = _select_probe_action(
        candidate,
        changed_state,
        changed_rule,
        changed_actions,
        seat=probe.seat,
        seed=probe.seed + ordinal * 2,
    )
    if base_action is None or changed_action is None:
        return {
            "attempted": True,
            "projection_match": True,
            "comparable": False,
            "base_error": base_error,
            "changed_error": changed_error,
        }
    return {
        "attempted": True,
        "projection_match": True,
        "comparable": True,
        "base_action": base_action,
        "changed_action": changed_action,
        "changed_decision": base_action != changed_action,
    }


def audit_candidate_information(  # noqa: PLR0913 - audit scope is explicit
    candidate: CandidateSpec,
    opponent: CandidateSpec,
    seeds: Sequence[int],
    *,
    feature_version: str | None = None,
    seats: Sequence[int] = (0, 1),
    max_states: int = 24,
) -> dict[str, Any]:
    """Empirically flag dependence on hidden deck or opponent-card identities.

    A changed action under a projection-preserving perturbation is reported as
    a *potential* information leak. It is evidence for follow-up review, not a
    proof that the legacy teacher intentionally used hidden information.
    """
    if max_states < 1:
        raise ValueError("max_states must be positive")
    schema = feature_version or candidate.feature_version
    probes = collect_decision_probes(
        candidate, opponent, seeds, seats=seats, limit=max_states
    )
    perturbations: dict[str, StatePerturbation] = {
        "hidden_deck_order": _shuffle_hidden_decks,
        "opponent_reserved_identity": _swap_opponent_reserved_card,
    }
    details: dict[str, list[dict[str, Any]]] = {
        name: [] for name in perturbations
    }
    projection_mismatches = 0
    comparable = 0
    changed_decisions = 0
    decision_errors = 0
    for ordinal, probe in enumerate(probes):
        for name, perturbation in perturbations.items():
            result = _probe_variant(
                candidate,
                probe,
                perturbation,
                schema,
                ordinal=ordinal,
            )
            details[name].append(
                {
                    "seed": probe.seed,
                    "seat": probe.seat,
                    "ply": probe.ply,
                    **result,
                }
            )
            if result.get("projection_match") is False:
                projection_mismatches += 1
            if result.get("comparable"):
                comparable += 1
                changed_decisions += int(result["changed_decision"])
            elif result.get("attempted") and result.get("projection_match"):
                decision_errors += 1
    attempted = sum(
        int(result.get("attempted", False))
        for records in details.values()
        for result in records
    )
    stable_rate = (comparable - changed_decisions) / comparable if comparable else None
    risks: list[str] = []
    if changed_decisions:
        risks.append(
            "potential hidden-state dependence detected; this audit signal is not proof of a code defect"
        )
    if decision_errors:
        risks.append("some probe actions could not be replayed and require manual review")
    if projection_mismatches:
        risks.append("some perturbations were not projection-preserving and were excluded")
    status = "inconclusive" if not comparable else ("risk" if risks else "pass")
    return {
        "candidate": candidate.name,
        "candidate_snapshot": candidate.snapshot,
        "opponent_for_state_access": opponent.name,
        "feature_version": schema,
        "states_checked": len(probes),
        "perturbations_attempted": attempted,
        "projection_mismatches": projection_mismatches,
        "comparable_probes": comparable,
        "changed_decisions": changed_decisions,
        "information_stable_rate": stable_rate,
        "decision_errors": decision_errors,
        "status": status,
        "risks": risks,
        "details": details,
    }


def audit_information_matrix(
    candidates: Sequence[CandidateSpec],
    opponent: CandidateSpec,
    seeds: Sequence[int],
    *,
    feature_version: str | None = None,
    max_states: int = 24,
) -> dict[str, dict[str, Any]]:
    """Run the same projection audit for every candidate."""
    return {
        candidate.name: audit_candidate_information(
            candidate,
            opponent,
            seeds,
            feature_version=feature_version,
            max_states=max_states,
        )
        for candidate in candidates
    }
