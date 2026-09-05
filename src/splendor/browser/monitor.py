"""
Mask parity monitoring: two independent measurements of the legal-action set.

The engine mask (``getLegalActions`` on the pseudo state) is the sole
authority for legality (IMPLEMENTATION_SPEC §0.3-1 ruling): DOM interactive
elements prove "clickable", never "the complete legal set" - whether taking
fewer gems voluntarily is legal is a rules question, not a UI question. The
DOM affordance set is therefore demoted to a cross-validation signal, which
comes with a free permanent detector for the rule-parity risk (R2): the two
sets must agree, and every disagreement is either

1. a *known rule difference* registered from the T0.4 experiments (E5/E6),
2. a *DOM extraction bug* (an action the engine allows but the measured DOM
   cannot support), or
3. a *page redesign* (whole UI regions gone - e.g. the action buttons).

Without this attribution, differences are debugged by hand through logs; with
it, only a *new* category needs human attention (phase-3 acceptance: zero
unexplained differences over 10 games).
"""

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from splendor.splendor.gym.envs.actions import (
    ALL_ACTIONS,
    Action,
    ActionEnum,
)
from splendor.splendor.types import GemsCount

from .dom_extractor import Snapshot

REPORT_SAMPLE_SIZE = 5


def dom_affordances(snapshot: Snapshot) -> set[int]:
    """
    Indices of ALL_ACTIONS the *measured DOM facts* could possibly support.

    This is deliberately rule-free: each pre-enumerated action is checked
    against mechanical page facts only - chips present for the gems it takes,
    a card sitting at the position it targets. It never decides legality
    (that is the engine's monopoly) and it intentionally over-approximates:
    affordability, the >=4 same-colour stack rule, the reservation limit and
    the 7-cards-per-colour cap are engine knowledge a DOM reading must not
    duplicate.
    """
    facts = _DomActionFacts(snapshot)
    return {
        index for index, action in enumerate(ALL_ACTIONS) if facts.supports(action)
    }


class _DomActionFacts:
    """Mechanical page facts an ALL_ACTIONS entry is checked against."""

    def __init__(self, snapshot: Snapshot) -> None:
        self._supply = snapshot["supply"]
        self._dealt = snapshot["dealt"]
        self._my_reserved_count = len(snapshot["my_reserved"])

    def supports(self, action: Action) -> bool:
        kind = action.type_enum
        if kind is ActionEnum.PASS:
            return True
        if kind in (ActionEnum.COLLECT_SAME, ActionEnum.COLLECT_DIFF):
            return self._chips_support(action.collected_gems)
        if kind is ActionEnum.RESERVE or kind is ActionEnum.BUY_AVAILABLE:
            return self._board_position_supported(action)
        if kind is ActionEnum.BUY_RESERVE:
            position = action.position
            return position is not None and (
                position.reserved_index < self._my_reserved_count
            )
        return False  # pragma: no cover - ActionEnum is a closed set

    def _chips_support(self, collected: GemsCount | None) -> bool:
        if not collected:
            return False
        return all(
            self._supply.get(colour, 0) >= count
            for colour, count in collected.items()
        )

    def _board_position_supported(self, action: Action) -> bool:
        position = action.position
        if position is None or position.tier not in range(len(self._dealt)):
            return False
        row = self._dealt[position.tier]
        return position.card_index < len(row) and row[position.card_index] is not None


class KnownDifference:
    """A registered, explainable engine-vs-page rule divergence."""

    def __init__(
        self, code: str, description: str, matches: Callable[[Action], bool]
    ) -> None:
        self.code = code
        self.description = description
        self.matches = matches


def _is_collect(action: Action) -> bool:
    return action.type_enum in (ActionEnum.COLLECT_SAME, ActionEnum.COLLECT_DIFF)


def _is_buy(action: Action) -> bool:
    return action.type_enum in (ActionEnum.BUY_AVAILABLE, ActionEnum.BUY_RESERVE)


# Registered from the T0.4 experiment agenda (docs/web_experiments.md):
# outcomes of E5/E6 that contradict the engine land here once confirmed.
DEFAULT_KNOWN_DIFFERENCES: tuple[KnownDifference, ...] = (
    KnownDifference(
        "E5",
        "web may allow taking fewer gems than the engine's forced minimum "
        "(voluntary partial take)",
        _is_collect,
    ),
    KnownDifference(
        "E6",
        "web may allow buying beyond the engine's 7-cards-per-colour cap",
        _is_buy,
    ),
)


class MaskParityMonitor:
    """Attributes engine-mask vs DOM-affordance differences to a cause."""

    def __init__(
        self,
        known_differences: Iterable[KnownDifference] | None = None,
    ) -> None:
        self._known_differences = (
            tuple(known_differences)
            if known_differences is not None
            else DEFAULT_KNOWN_DIFFERENCES
        )

    def check(
        self, engine_mask: NDArray, dom_affordances_set: set[int]
    ) -> list[str]:
        """
        Compare both measurements and return a human-readable report.

        Empty list is never returned for a mismatch: every line names a
        category, counts and sample actions, so phase-3 acceptance ("zero
        unexplained differences") can be judged mechanically.
        """
        engine_indices = {int(i) for i in np.flatnonzero(engine_mask)}
        dom_indices = set(dom_affordances_set)

        shared = engine_indices & dom_indices
        missing = sorted(engine_indices - dom_indices)
        extra = sorted(dom_indices - engine_indices)

        report: list[str] = [
            f"mask parity: engine-legal={len(engine_indices)} "
            f"dom-affordable={len(dom_indices)} shared={len(shared)} "
            f"engine-only={len(missing)} dom-only={len(extra)}"
        ]
        if not missing and not extra:
            report.append("OK: both measurements agree on the legal-action set")
            return report

        if not dom_indices:
            report.append(
                "PAGE REDESIGN suspected: the DOM affordance set is empty "
                f"while {len(engine_indices)} engine-legal actions exist - "
                "the extraction JS probably matches nothing anymore"
            )
            return report

        if any(ALL_ACTIONS[i].type_enum is ActionEnum.PASS for i in missing):
            report.append(
                "PAGE REDESIGN suspected: PASS is engine-legal but the page "
                "offers no pass affordance - top action buttons missing"
            )

        extraction_bug = [i for i in missing if ALL_ACTIONS[i].type_enum
                          is not ActionEnum.PASS]
        if extraction_bug:
            report.append(
                f"DOM EXTRACTION BUG suspected: {len(extraction_bug)} "
                "engine-legal actions unsupported by the measured DOM "
                f"(e.g. {_describe_samples(extraction_bug)})"
            )

        matched: dict[str, list[int]] = {}
        unmatched: list[int] = []
        for index in extra:
            action = ALL_ACTIONS[index]
            code = next(
                (
                    known.code
                    for known in self._known_differences
                    if known.matches(action)
                ),
                None,
            )
            if code is None:
                unmatched.append(index)
            else:
                matched.setdefault(code, []).append(index)

        for known in self._known_differences:
            if known.code in matched:
                samples = matched[known.code]
                report.append(
                    f"KNOWN RULE DIFFERENCE {known.code}: {known.description} - "
                    f"{len(samples)} dom-only action(s) "
                    f"(e.g. {_describe_samples(samples)})"
                )
        if unmatched:
            # Affordances over-approximate legality *by design* (a DOM reading
            # cannot know affordability, the pass restriction, or which
            # sub-flows the engine would open), so dom-only residuals are
            # expected. They stay in the report - grouped by action type -
            # so a systematic drift (e.g. every reserve missing) is visible
            # instead of silently folded into noise.
            report.append(
                f"DOM-ONLY over-approximation (expected unless systematic): "
                f"{len(unmatched)} action(s) - "
                f"{_group_by_type(unmatched)} "
                f"(e.g. {_describe_samples(unmatched)})"
            )
        return report


def _describe_samples(indices: list[int]) -> str:
    samples = indices[:REPORT_SAMPLE_SIZE]
    return ", ".join(
        f"#{index} {_describe_action(ALL_ACTIONS[index])}" for index in samples
    )


def _group_by_type(indices: list[int]) -> str:
    counts: dict[str, int] = {}
    for index in indices:
        name = ALL_ACTIONS[index].type_enum.name
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))


def _describe_action(action: Action) -> str:
    label = action.type_enum.name
    details: list[str] = []
    if action.collected_gems:
        details.append(
            "takes "
            + ",".join(f"{c}x{n}" for c, n in sorted(action.collected_gems.items()))
        )
    if action.returned_gems:
        details.append(
            "returns "
            + ",".join(f"{c}x{n}" for c, n in sorted(action.returned_gems.items()))
        )
    if action.position is not None:
        details.append(
            f"pos(tier={action.position.tier}, col={action.position.card_index}, "
            f"res={action.position.reserved_index})"
        )
    if action.noble_index is not None:
        details.append(f"noble={action.noble_index}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"{label}{suffix}"
