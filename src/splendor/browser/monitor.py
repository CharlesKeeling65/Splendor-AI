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

How to read a report, in one place, because two fields invite misreading:

* ``engine-only`` is the safety-critical count and must be 0. The policy
  samples the *engine* mask, so a non-zero value means it could be handed an
  action the page cannot execute today.
* ``dom-only`` is expected to be large and is *not* a problem list. The DOM
  affordance collector is rule-free by design (see :func:`dom_affordances`), so
  it never knows affordability, the >=4 same-colour stack rule, the reservation
  limit or the 7-card cap. It therefore counts *reachable-but-illegal*
  candidates, which the policy can never select.
* The ``E5``/``E6`` buckets are matched by **action type** (every collect /
  every buy), not by the specific divergence they name, so their counts are
  candidate-class sizes rather than confirmed divergence counts. Measured on
  the ``opening.html`` fixture (empty hand, full board): of 3465 dom-only
  actions, 89% carry a ``returned_gems`` combo and 84% carry a ``noble_index``
  - both invisible to the collector - while only 0.43% have the genuine
  "voluntary partial take" shape. The triage signal is therefore not these
  counts but *whether a dom-only action's type belongs to no registered class
  at all*.

:func:`anomaly_count` turns all of the above into the single number the
phase-3 acceptance is judged on: the lines carrying an ``ANOMALY_PREFIXES``
tag, i.e. exactly the "needs a human" categories above. Never count
``len(report)`` - the header is unconditional, so that reports perfect parity
as 2 anomalies.
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
    duplicate. Nor does it inspect the *return* side of an action (whether
    returning those gems is possible at all) or collapse the pre-enumerated
    noble slots - and those two blind spots dominate the residual: measured on
    the ``opening.html`` fixture, 89% of the dom-only set carries a
    ``returned_gems`` combo and 84% carries a ``noble_index``.
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

        # The safety-critical direction is `engine-only`: a non-zero count means
        # the engine would let the policy choose an action the page cannot
        # execute. The reverse (`dom-only`) is unreachable by construction (the
        # policy samples the engine mask) and dom_affordances over-approximates
        # on purpose, so it is expected rather than alarming. Stated in the
        # header line so the one field that matters is never read out of context.
        direction_note = (
            " [direction safe: engine mask is a subset of DOM affordances]"
            if not missing
            else ""
        )
        report: list[str] = [
            f"mask parity: engine-legal={len(engine_indices)} "
            f"dom-affordable={len(dom_indices)} shared={len(shared)} "
            f"engine-only={len(missing)} dom-only={len(extra)}{direction_note}"
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
                    f"KNOWN-DIFFERENCE CLASS {known.code} "
                    f"(type-matched bucket, not confirmed instances): "
                    f"{known.description} - {len(samples)} dom-only action(s) "
                    f"(e.g. {_describe_samples(samples)})"
                )
        if unmatched:
            # Affordances over-approximate legality *by design* (a DOM reading
            # cannot know affordability, the pass restriction, which
            # sub-flows the engine would open, whether a return combo is
            # possible at all, or which noble slot is still free), so dom-only
            # residuals are expected and are NOT anomalies. They stay in the
            # report - grouped by action type - so a systematic drift (e.g.
            # every reserve missing) is visible instead of silently folded
            # into noise.
            report.append(
                f"DOM-ONLY over-approximation (expected, not counted as "
                f"anomaly): {len(unmatched)} action(s) - "
                f"{_group_by_type(unmatched)} "
                f"(e.g. {_describe_samples(unmatched)})"
            )
        return report


# Report lines that need a human. Everything else in a report documents the
# by-design over-approximation described above. Kept as *prefixes* rather than
# a structured return type so that every consumer (play_web, play_vs_humans,
# play_remote) can filter the same list the dashboard renders, without
# re-deriving the underlying sets.
ANOMALY_PREFIXES: tuple[str, ...] = (
    "PAGE REDESIGN",
    "DOM EXTRACTION BUG",
)


def anomaly_count(report: Iterable[str]) -> int:
    """
    How many of a report's lines need a human.

    This is the number phase-3 acceptance ("zero unexplained differences over
    10 games") is judged on. Counting ``len(report)`` instead is a bug that
    reports *perfect* parity as 2 anomalies - the header line is always there,
    and an agreeing report appends "OK" to it.
    """
    return sum(1 for line in report if line.startswith(ANOMALY_PREFIXES))


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
