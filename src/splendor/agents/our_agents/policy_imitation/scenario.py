"""Immutable two-player opening snapshots for formal task-1 experiments.

``source_seed`` is provenance only.  Once generated, a scenario is identified
by the canonical SHA-256 of the complete replay-defining opening state.  This
lets every runner load exactly the same board instead of hoping that an integer
seed has the same meaning across different construction paths.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any, Final, Literal, cast

from splendor.seed_registry import check_seeds_declared, resolve_segment
from splendor.splendor.constants import NORMAL_COLORS, ROUNDS_LIMIT
from splendor.splendor.gym.envs.actions import ALL_ACTIONS, Action
from splendor.splendor.splendor_model import Card, SplendorState
from splendor.splendor.splendor_utils import CARDS, COLOURS, NOBLES
from splendor.splendor.utils import LimitRoundsGameRule

from .protocol import isolated_python_seed, sha256_canonical_json

SCENARIO_SCHEMA_VERSION: Final = "splendor-scenario/1"
SCENARIO_RULE_ID: Final = "splendor.rules.LimitRoundsGameRule/2p-v1"
SCENARIO_DECK_TOP: Final = "list_end"
SCENARIO_SEATS: Final = 2
SCENARIO_TIER_COUNTS: Final = (40, 30, 20)
SCENARIO_DEALT_PER_TIER: Final = 4
SCENARIO_DECK_COUNTS: Final = (36, 26, 16)
SCENARIO_NOBLES: Final = 3
NOBLE_TUPLE_SIZE: Final = 2
SHA256_HEX_LENGTH: Final = 64
LOW_COST_LIMIT: Final = 7
GEM_ORDER: Final = tuple(COLOURS.values())
INITIAL_GEMS_2P: Final = {
    "black": 4,
    "red": 4,
    "yellow": 5,
    "green": 4,
    "blue": 4,
    "white": 4,
}
SelectionKind = Literal["iid", "natural-deal-srswor", "stress-balanced", "ci-fixture"]
StrataValue = int | float


class ScenarioValidationError(ValueError):
    """Raised when a snapshot cannot be replayed under the frozen contract."""


def _is_int(value: object) -> bool:
    return type(value) is int


def _require_string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ScenarioValidationError(f"scenario {field} must be a non-empty string")
    return value


def _require_integer(value: object, field: str) -> int:
    if not _is_int(value):
        raise ScenarioValidationError(f"scenario {field} must be an integer")
    return cast(int, value)


def _pairs_to_dict(
    pairs: Sequence[tuple[str, int]],
    *,
    field: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in pairs:
        if type(key) is not str or not key or not _is_int(value):
            raise ScenarioValidationError(
                f"scenario {field} must contain string/integer pairs"
            )
        if key in result:
            raise ScenarioValidationError(f"scenario {field} contains duplicate {key}")
        result[key] = value
    return result


def _normalise_cost(cost: Mapping[Any, int]) -> dict[str, int]:
    return {str(colour): int(value) for colour, value in sorted(cost.items())}


@cache
def card_registry_hash() -> str:
    """Hash all card and noble primitives, independent of object ``repr``."""
    payload = {
        "version": "splendor-card-noble-registry/1",
        "cards": [
            {
                "code": code,
                "colour": colour,
                "cost": _normalise_cost(cost),
                "deck_id": deck_id - 1,
                "points": points,
            }
            for code, (colour, cost, deck_id, points) in sorted(CARDS.items())
        ],
        "nobles": [
            {"code": code, "cost": _normalise_cost(cost)}
            for code, cost in sorted(NOBLES, key=lambda item: item[0])
        ],
    }
    return sha256_canonical_json(payload)


def _gems_payload(gems: Mapping[Any, int] | None) -> dict[str, int] | None:
    return None if gems is None else _normalise_cost(gems)


def _action_payload(index: int, action: Action) -> dict[str, object]:
    position = action.position
    return {
        "index": index,
        "type": action.type_enum.name,
        "collected_gems": _gems_payload(action.collected_gems),
        "returned_gems": _gems_payload(action.returned_gems),
        "position": None
        if position is None
        else {
            "tier": position.tier,
            "card_index": position.card_index,
            "reserved_index": position.reserved_index,
        },
        "noble_index": action.noble_index,
    }


@cache
def action_registry_hash() -> str:
    """Hash the exact 3510-action index order with lossless field encoding."""
    return sha256_canonical_json(
        {
            "version": "splendor-action-registry/1",
            "actions": [
                _action_payload(index, action)
                for index, action in enumerate(ALL_ACTIONS)
            ],
        }
    )


def action_registry_hash_for(actions: Sequence[Action]) -> str:
    """Hash an explicit action sequence; primarily useful for drift audits."""
    return sha256_canonical_json(
        {
            "version": "splendor-action-registry/1",
            "actions": [
                _action_payload(index, action) for index, action in enumerate(actions)
            ],
        }
    )


def _quantile(values: Sequence[int], probability: float) -> float:
    """Deterministic linear quantile (NumPy's historical ``linear`` rule)."""
    ordered = sorted(values)
    if not ordered:
        raise ScenarioValidationError("scenario descriptors need visible cards")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _rounded(value: float) -> float:
    return round(float(value), 12)


def compute_strata_v1(
    dealt: Sequence[Sequence[str]],
    nobles_in_order: Sequence[str],
) -> tuple[tuple[str, StrataValue], ...]:
    """Compute versioned opening-only descriptors with explicit formulas.

    Token deficit is the sum of a card's coloured costs for an empty player.
    Entropy is Shannon entropy of each card's cost shares divided by ``log(5)``;
    concentration is that card's largest cost share.  Noble overlap is mean
    pairwise weighted Jaccard.  Alignment is cosine similarity between visible
    bonus-colour counts and aggregate noble demand.  A low-cost/high-point card
    costs at most seven tokens and is worth at least one point.
    """
    codes = [code for tier in dealt for code in tier]
    if len(codes) != SCENARIO_DEALT_PER_TIER * len(SCENARIO_TIER_COUNTS):
        raise ScenarioValidationError("strata_v1 requires exactly 12 visible cards")
    card_rows = [CARDS.get(code) for code in codes]
    if any(row is None for row in card_rows):
        raise ScenarioValidationError("strata_v1 references an unknown card")
    cards = [cast(tuple[str, dict[str, int], int, int], row) for row in card_rows]
    noble_by_code = dict(NOBLES)
    try:
        noble_costs = [noble_by_code[code] for code in nobles_in_order]
    except KeyError as exc:
        raise ScenarioValidationError(
            f"strata_v1 references unknown noble {exc.args[0]!r}"
        ) from exc

    deficits = [sum(cost.values()) for _colour, cost, _tier, _points in cards]
    entropies: list[float] = []
    concentrations: list[float] = []
    bonus_counts: dict[str, int] = dict.fromkeys(NORMAL_COLORS, 0)
    for colour, cost, _tier, _points in cards:
        total = sum(cost.values())
        shares = [value / total for value in cost.values() if value]
        entropies.append(
            -sum(share * math.log(share) for share in shares) / math.log(5)
        )
        concentrations.append(max(shares))
        bonus_counts[colour] += 1

    pair_overlaps: list[float] = []
    for left_index in range(len(noble_costs)):
        for right_index in range(left_index + 1, len(noble_costs)):
            left = noble_costs[left_index]
            right = noble_costs[right_index]
            numerator = sum(
                min(left.get(colour, 0), right.get(colour, 0))
                for colour in NORMAL_COLORS
            )
            denominator = sum(
                max(left.get(colour, 0), right.get(colour, 0))
                for colour in NORMAL_COLORS
            )
            pair_overlaps.append(numerator / denominator if denominator else 0.0)

    noble_demand = {
        colour: sum(cost.get(colour, 0) for cost in noble_costs)
        for colour in NORMAL_COLORS
    }
    dot = sum(bonus_counts[colour] * noble_demand[colour] for colour in NORMAL_COLORS)
    bonus_norm = math.sqrt(sum(value * value for value in bonus_counts.values()))
    noble_norm = math.sqrt(sum(value * value for value in noble_demand.values()))
    alignment = dot / (bonus_norm * noble_norm) if bonus_norm and noble_norm else 0.0

    descriptors: dict[str, StrataValue] = {
        "visible_point_density": _rounded(
            sum(points for _colour, _cost, _tier, points in cards) / len(cards)
        ),
        "token_deficit_min": min(deficits),
        "token_deficit_q25": _rounded(_quantile(deficits, 0.25)),
        "token_deficit_q50": _rounded(_quantile(deficits, 0.50)),
        "token_deficit_q75": _rounded(_quantile(deficits, 0.75)),
        "cost_colour_entropy_mean": _rounded(sum(entropies) / len(entropies)),
        "cost_colour_concentration_mean": _rounded(
            sum(concentrations) / len(concentrations)
        ),
        "noble_pair_weighted_jaccard_mean": _rounded(
            sum(pair_overlaps) / len(pair_overlaps)
        ),
        "visible_noble_alignment_cosine": _rounded(alignment),
        "low_cost_high_point_count": sum(
            sum(cost.values()) <= LOW_COST_LIMIT and points >= 1
            for _colour, cost, _tier, points in cards
        ),
    }
    return tuple(sorted(descriptors.items()))


def _state_payload(  # noqa: PLR0913 - every replay-defining field is explicit
    *,
    n_seats: int,
    rule_id: str,
    rounds_limit: int,
    card_hash: str,
    action_hash: str,
    nobles_in_order: Sequence[str],
    dealt: Sequence[Sequence[str]],
    decks: Sequence[Sequence[str]],
    deck_top: str,
    initial_gems: Mapping[str, int],
    current_agent_index: int,
) -> dict[str, object]:
    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "n_seats": n_seats,
        "rule_id": rule_id,
        "rounds_limit": rounds_limit,
        "card_registry_hash": card_hash,
        "action_registry_hash": action_hash,
        "nobles_in_order": list(nobles_in_order),
        "dealt": [list(tier) for tier in dealt],
        "decks": [list(tier) for tier in decks],
        "deck_top": deck_top,
        "initial_gems": dict(initial_gems),
        "current_agent_index": current_agent_index,
        "action_counter": 0,
        "agent_state": "empty-v1",
    }


@dataclass(frozen=True)
class ScenarioV1:
    """JSON-safe, content-addressed two-player initial state."""

    schema_version: str
    scenario_id: str
    n_seats: int
    source_segment: str
    source_seed: int
    rule_id: str
    rounds_limit: int
    card_registry_hash: str
    action_registry_hash: str
    nobles_in_order: tuple[str, ...]
    dealt: tuple[tuple[str, ...], ...]
    decks: tuple[tuple[str, ...], ...]
    deck_top: str
    initial_gems: tuple[tuple[str, int], ...]
    current_agent_index: int
    canonical_state_sha256: str
    strata_v1: tuple[tuple[str, StrataValue], ...]
    selection_kind: SelectionKind = "iid"
    inclusion_probability: float = 1.0
    selection_stratum: str | None = None
    selection_design_sha256: str | None = None

    def state_payload(self) -> dict[str, object]:
        """Return only fields that define replay semantics and identity."""
        return _state_payload(
            n_seats=self.n_seats,
            rule_id=self.rule_id,
            rounds_limit=self.rounds_limit,
            card_hash=self.card_registry_hash,
            action_hash=self.action_registry_hash,
            nobles_in_order=self.nobles_in_order,
            dealt=self.dealt,
            decks=self.decks,
            deck_top=self.deck_top,
            initial_gems=_pairs_to_dict(self.initial_gems, field="initial_gems"),
            current_agent_index=self.current_agent_index,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation, including provenance."""
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "n_seats": self.n_seats,
            "source_segment": self.source_segment,
            "source_seed": self.source_seed,
            "rule_id": self.rule_id,
            "rounds_limit": self.rounds_limit,
            "card_registry_hash": self.card_registry_hash,
            "action_registry_hash": self.action_registry_hash,
            "nobles_in_order": list(self.nobles_in_order),
            "dealt": [list(tier) for tier in self.dealt],
            "decks": [list(tier) for tier in self.decks],
            "deck_top": self.deck_top,
            "initial_gems": dict(self.initial_gems),
            "current_agent_index": self.current_agent_index,
            "canonical_state_sha256": self.canonical_state_sha256,
            "strata_v1": dict(self.strata_v1),
            "selection_kind": self.selection_kind,
            "inclusion_probability": self.inclusion_probability,
            "selection_stratum": self.selection_stratum,
            "selection_design_sha256": self.selection_design_sha256,
        }

    @classmethod
    def from_dict(  # noqa: C901 - strict schema parsing keeps all fields fail-closed
        cls, raw: Mapping[str, object]
    ) -> ScenarioV1:
        """Parse a strict JSON mapping; unknown or missing fields fail closed."""
        required = {
            "schema_version",
            "scenario_id",
            "n_seats",
            "source_segment",
            "source_seed",
            "rule_id",
            "rounds_limit",
            "card_registry_hash",
            "action_registry_hash",
            "nobles_in_order",
            "dealt",
            "decks",
            "deck_top",
            "initial_gems",
            "current_agent_index",
            "canonical_state_sha256",
            "strata_v1",
            "selection_kind",
            "inclusion_probability",
            "selection_stratum",
            "selection_design_sha256",
        }
        if set(raw) != required:
            missing = sorted(required - set(raw))
            unknown = sorted(set(raw) - required)
            raise ScenarioValidationError(
                f"scenario fields mismatch; missing={missing}, unknown={unknown}"
            )

        def string_tuple(value: object, field: str) -> tuple[str, ...]:
            if not isinstance(value, list) or any(
                type(item) is not str for item in value
            ):
                raise ScenarioValidationError(f"scenario {field} must be a string list")
            return tuple(cast(list[str], value))

        def nested_strings(value: object, field: str) -> tuple[tuple[str, ...], ...]:
            if not isinstance(value, list):
                raise ScenarioValidationError(f"scenario {field} must be a nested list")
            return tuple(string_tuple(tier, field) for tier in value)

        gems_raw = raw["initial_gems"]
        strata_raw = raw["strata_v1"]
        if not isinstance(gems_raw, Mapping):
            raise ScenarioValidationError("scenario initial_gems must be a mapping")
        if not isinstance(strata_raw, Mapping):
            raise ScenarioValidationError("scenario strata_v1 must be a mapping")
        gems: list[tuple[str, int]] = []
        for key, value in sorted(gems_raw.items(), key=lambda item: str(item[0])):
            if type(key) is not str or not _is_int(value):
                raise ScenarioValidationError(
                    "scenario initial_gems must map strings to integers"
                )
            gems.append((key, cast(int, value)))
        strata: list[tuple[str, StrataValue]] = []
        for key, value in sorted(strata_raw.items(), key=lambda item: str(item[0])):
            if type(key) is not str or type(value) not in {int, float}:
                raise ScenarioValidationError(
                    "scenario strata_v1 must map strings to finite numbers"
                )
            strata.append((key, cast(StrataValue, value)))
        selection_stratum = raw["selection_stratum"]
        if selection_stratum is not None and type(selection_stratum) is not str:
            raise ScenarioValidationError(
                "scenario selection_stratum must be a string or null"
            )
        selection_design_sha256 = raw["selection_design_sha256"]
        if (
            selection_design_sha256 is not None
            and type(selection_design_sha256) is not str
        ):
            raise ScenarioValidationError(
                "scenario selection_design_sha256 must be a string or null"
            )
        inclusion = raw["inclusion_probability"]
        if type(inclusion) not in {int, float}:
            raise ScenarioValidationError(
                "scenario inclusion_probability must be numeric"
            )
        scenario = cls(
            schema_version=_require_string(raw["schema_version"], "schema_version"),
            scenario_id=_require_string(raw["scenario_id"], "scenario_id"),
            n_seats=_require_integer(raw["n_seats"], "n_seats"),
            source_segment=_require_string(raw["source_segment"], "source_segment"),
            source_seed=_require_integer(raw["source_seed"], "source_seed"),
            rule_id=_require_string(raw["rule_id"], "rule_id"),
            rounds_limit=_require_integer(raw["rounds_limit"], "rounds_limit"),
            card_registry_hash=_require_string(
                raw["card_registry_hash"], "card_registry_hash"
            ),
            action_registry_hash=_require_string(
                raw["action_registry_hash"], "action_registry_hash"
            ),
            nobles_in_order=string_tuple(raw["nobles_in_order"], "nobles_in_order"),
            dealt=nested_strings(raw["dealt"], "dealt"),
            decks=nested_strings(raw["decks"], "decks"),
            deck_top=_require_string(raw["deck_top"], "deck_top"),
            initial_gems=tuple(gems),
            current_agent_index=_require_integer(
                raw["current_agent_index"], "current_agent_index"
            ),
            canonical_state_sha256=_require_string(
                raw["canonical_state_sha256"], "canonical_state_sha256"
            ),
            strata_v1=tuple(strata),
            selection_kind=cast(
                SelectionKind,
                _require_string(raw["selection_kind"], "selection_kind"),
            ),
            inclusion_probability=float(cast(int | float, inclusion)),
            selection_stratum=cast(str | None, selection_stratum),
            selection_design_sha256=cast(str | None, selection_design_sha256),
        )
        validate_scenario(scenario)
        return scenario


def _validate_initial_agents(state: SplendorState) -> None:
    if len(state.agents) != SCENARIO_SEATS:
        raise ScenarioValidationError("ScenarioV1 requires exactly two agents")
    for expected_id, agent in enumerate(state.agents):
        if agent.id != expected_id or agent.score != 0 or agent.passed:
            raise ScenarioValidationError("scenario agents must be fresh and ordered")
        if agent.last_action is not None or agent.nobles:
            raise ScenarioValidationError("scenario agents must have no prior actions")
        if agent.agent_trace.action_reward:
            raise ScenarioValidationError("scenario agent traces must be empty")
        if set(agent.gems) != set(GEM_ORDER) or any(
            agent.gems[colour] != 0 for colour in GEM_ORDER
        ):
            raise ScenarioValidationError("scenario agents must start without gems")
        if set(agent.cards) != set(GEM_ORDER) or any(
            agent.cards[colour] for colour in GEM_ORDER
        ):
            raise ScenarioValidationError("scenario agents must start without cards")


def _validate_initial_board(  # noqa: C901 - every source primitive is audited
    state: SplendorState,
) -> None:
    board = state.board
    if list(board.gems) != list(GEM_ORDER) or board.gems != INITIAL_GEMS_2P:
        raise ScenarioValidationError(
            "scenario source board must use the exact ordered 2p gem supply"
        )
    if len(board.dealt) != len(SCENARIO_TIER_COUNTS) or any(
        len(tier) != SCENARIO_DEALT_PER_TIER for tier in board.dealt
    ):
        raise ScenarioValidationError("scenario source dealt cards must have shape 3x4")
    if tuple(len(tier) for tier in board.decks) != SCENARIO_DECK_COUNTS:
        raise ScenarioValidationError("scenario source deck lengths must be 36/26/16")

    codes: list[str] = []
    for tier_index, tier in enumerate((*board.dealt, *board.decks)):
        expected_tier = tier_index % len(SCENARIO_TIER_COUNTS)
        for card in tier:
            if not isinstance(card, Card) or card.code not in CARDS:
                raise ScenarioValidationError(
                    "scenario source contains an unknown card"
                )
            colour, cost, deck_id, points = CARDS[card.code]
            if (
                card.colour != colour
                or card.cost != cost
                or card.deck_id != deck_id - 1
                or card.deck_id != expected_tier
                or card.points != points
            ):
                raise ScenarioValidationError(
                    f"scenario source card {card.code!r} primitive fields were modified"
                )
            codes.append(card.code)
    if (
        len(codes) != len(CARDS)
        or len(set(codes)) != len(CARDS)
        or set(codes) != set(CARDS)
    ):
        raise ScenarioValidationError("scenario source cards are missing or duplicated")

    if len(board.nobles) != SCENARIO_NOBLES:
        raise ScenarioValidationError("scenario source must contain three nobles")
    noble_registry = dict(NOBLES)
    noble_codes: list[str] = []
    for noble in board.nobles:
        if (
            not isinstance(noble, tuple)
            or len(noble) != NOBLE_TUPLE_SIZE
            or type(noble[0]) is not str
            or noble[0] not in noble_registry
            or not isinstance(noble[1], Mapping)
            or dict(noble[1]) != noble_registry[noble[0]]
        ):
            raise ScenarioValidationError(
                "scenario source noble primitive fields were modified"
            )
        noble_codes.append(noble[0])
    if len(set(noble_codes)) != SCENARIO_NOBLES:
        raise ScenarioValidationError("scenario source nobles must be unique")


def scenario_from_state(  # noqa: PLR0913 - provenance/selection are explicit
    state: SplendorState,
    *,
    source_segment: str,
    source_seed: int,
    selection_kind: SelectionKind = "iid",
    inclusion_probability: float = 1.0,
    selection_stratum: str | None = None,
) -> ScenarioV1:
    """Freeze a pristine engine state without consulting any object repr."""
    _validate_initial_agents(state)
    _validate_initial_board(state)
    dealt = tuple(
        tuple(card.code if card is not None else "" for card in tier)
        for tier in state.board.dealt
    )
    decks = tuple(tuple(card.code for card in tier) for tier in state.board.decks)
    nobles = tuple(code for code, _cost in state.board.nobles)
    gems = tuple(
        sorted((str(key), int(value)) for key, value in state.board.gems.items())
    )
    payload = _state_payload(
        n_seats=len(state.agents),
        rule_id=SCENARIO_RULE_ID,
        rounds_limit=ROUNDS_LIMIT,
        card_hash=card_registry_hash(),
        action_hash=action_registry_hash(),
        nobles_in_order=nobles,
        dealt=dealt,
        decks=decks,
        deck_top=SCENARIO_DECK_TOP,
        initial_gems=dict(gems),
        current_agent_index=state.agent_to_move,
    )
    state_hash = sha256_canonical_json(payload)
    scenario = ScenarioV1(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=state_hash,
        n_seats=len(state.agents),
        source_segment=source_segment,
        source_seed=source_seed,
        rule_id=SCENARIO_RULE_ID,
        rounds_limit=ROUNDS_LIMIT,
        card_registry_hash=card_registry_hash(),
        action_registry_hash=action_registry_hash(),
        nobles_in_order=nobles,
        dealt=dealt,
        decks=decks,
        deck_top=SCENARIO_DECK_TOP,
        initial_gems=gems,
        current_agent_index=state.agent_to_move,
        canonical_state_sha256=state_hash,
        strata_v1=compute_strata_v1(dealt, nobles),
        selection_kind=selection_kind,
        inclusion_probability=inclusion_probability,
        selection_stratum=selection_stratum,
    )
    validate_scenario(scenario)
    return scenario


def generate_scenario(
    source_segment: str,
    source_seed: int,
    *,
    selection_kind: SelectionKind = "iid",
    inclusion_probability: float = 1.0,
    selection_stratum: str | None = None,
) -> ScenarioV1:
    """Generate one opening through the engine while restoring global RNG."""
    segment = resolve_segment(source_segment)
    owner = check_seeds_declared([source_seed])
    if owner != segment:
        raise ScenarioValidationError(
            f"source seed {source_seed} belongs to {owner.name!r}, not {segment.name!r}"
        )
    with isolated_python_seed(source_seed):
        state = SplendorState(SCENARIO_SEATS)
    return scenario_from_state(
        state,
        source_segment=segment.name,
        source_seed=source_seed,
        selection_kind=selection_kind,
        inclusion_probability=inclusion_probability,
        selection_stratum=selection_stratum,
    )


def validate_scenario(  # noqa: C901,PLR0912,PLR0915 - complete invariant gate
    scenario: ScenarioV1,
) -> None:
    """Recompute every invariant required for deterministic replay."""
    if scenario.schema_version != SCENARIO_SCHEMA_VERSION:
        raise ScenarioValidationError("unsupported scenario schema_version")
    if scenario.n_seats != SCENARIO_SEATS:
        raise ScenarioValidationError("ScenarioV1 requires n_seats=2")
    if scenario.rule_id != SCENARIO_RULE_ID or scenario.rounds_limit != ROUNDS_LIMIT:
        raise ScenarioValidationError("scenario rule or rounds_limit has drifted")
    if scenario.deck_top != SCENARIO_DECK_TOP:
        raise ScenarioValidationError("scenario deck_top must be 'list_end'")
    if scenario.current_agent_index not in range(SCENARIO_SEATS):
        raise ScenarioValidationError("scenario current_agent_index is invalid")
    if scenario.card_registry_hash != card_registry_hash():
        raise ScenarioValidationError("scenario card registry hash mismatch")
    if scenario.action_registry_hash != action_registry_hash():
        raise ScenarioValidationError("scenario action registry hash mismatch")
    if len(scenario.dealt) != len(SCENARIO_TIER_COUNTS) or any(
        len(tier) != SCENARIO_DEALT_PER_TIER for tier in scenario.dealt
    ):
        raise ScenarioValidationError("scenario dealt must have shape 3x4")
    if tuple(len(tier) for tier in scenario.decks) != SCENARIO_DECK_COUNTS:
        raise ScenarioValidationError("scenario deck lengths must be 36/26/16")
    if (
        len(scenario.nobles_in_order) != SCENARIO_NOBLES
        or len(set(scenario.nobles_in_order)) != SCENARIO_NOBLES
    ):
        raise ScenarioValidationError("scenario must contain three unique nobles")
    known_nobles = {code for code, _cost in NOBLES}
    if not set(scenario.nobles_in_order) <= known_nobles:
        raise ScenarioValidationError("scenario contains an unknown noble")

    all_codes = [code for tier in (*scenario.dealt, *scenario.decks) for code in tier]
    if len(all_codes) != len(CARDS) or len(set(all_codes)) != len(all_codes):
        raise ScenarioValidationError("scenario cards are missing or duplicated")
    if set(all_codes) != set(CARDS):
        raise ScenarioValidationError("scenario cards do not cover the registry")
    for tier_index, tier in enumerate((*scenario.dealt, *scenario.decks)):
        expected_tier = tier_index % len(SCENARIO_TIER_COUNTS)
        if any(CARDS[code][2] - 1 != expected_tier for code in tier):
            raise ScenarioValidationError("scenario card is stored in the wrong tier")
    if _pairs_to_dict(scenario.initial_gems, field="initial_gems") != INITIAL_GEMS_2P:
        raise ScenarioValidationError("scenario initial gems do not match the 2p rules")

    state_hash = sha256_canonical_json(scenario.state_payload())
    if (
        scenario.canonical_state_sha256 != state_hash
        or scenario.scenario_id != state_hash
    ):
        raise ScenarioValidationError("scenario canonical state hash mismatch")
    expected_strata = compute_strata_v1(scenario.dealt, scenario.nobles_in_order)
    if scenario.strata_v1 != expected_strata:
        raise ScenarioValidationError("scenario strata_v1 mismatch")
    if scenario.selection_kind not in {
        "iid",
        "natural-deal-srswor",
        "stress-balanced",
        "ci-fixture",
    }:
        raise ScenarioValidationError("scenario selection_kind is invalid")
    if not math.isfinite(scenario.inclusion_probability) or not (
        0.0 < scenario.inclusion_probability <= 1.0
    ):
        raise ScenarioValidationError(
            "scenario inclusion_probability must lie in (0, 1]"
        )
    if scenario.selection_kind in {"iid", "ci-fixture"} and (
        scenario.inclusion_probability != 1.0
        or scenario.selection_stratum is not None
        or scenario.selection_design_sha256 is not None
    ):
        raise ScenarioValidationError(
            "unselected scenarios require probability 1 and no selection metadata"
        )
    if scenario.selection_kind == "natural-deal-srswor" and (
        scenario.selection_stratum is None
        or scenario.selection_design_sha256 is None
        or len(scenario.selection_design_sha256) != SHA256_HEX_LENGTH
        or any(
            character not in "0123456789abcdef"
            for character in scenario.selection_design_sha256
        )
    ):
        raise ScenarioValidationError(
            "natural-deal SRSWOR scenarios require stratum and selection-design SHA-256"
        )
    if scenario.selection_kind == "stress-balanced" and not scenario.selection_stratum:
        raise ScenarioValidationError(
            "stress-balanced scenarios require a selection stratum"
        )
    if scenario.selection_kind == "stress-balanced" and (
        scenario.selection_design_sha256 is None
        or len(scenario.selection_design_sha256) != SHA256_HEX_LENGTH
        or any(
            character not in "0123456789abcdef"
            for character in scenario.selection_design_sha256
        )
    ):
        raise ScenarioValidationError(
            "stress-balanced scenarios require a selection-design SHA-256"
        )

    try:
        segment = resolve_segment(scenario.source_segment)
        owner = check_seeds_declared([scenario.source_seed])
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc
    if segment != owner:
        raise ScenarioValidationError("scenario source seed/segment mismatch")


def _new_card(code: str) -> Card:
    colour, cost, deck_id, points = CARDS[code]
    return Card(colour, code, dict(cost), deck_id - 1, points)


def state_from_scenario(scenario: ScenarioV1) -> SplendorState:
    """Build a fresh engine object graph without invoking a random constructor."""
    validate_scenario(scenario)
    state = cast(SplendorState, object.__new__(SplendorState))
    board = cast(SplendorState.BoardState, object.__new__(SplendorState.BoardState))
    cards = {code: _new_card(code) for code in CARDS}
    board.dealt = [[cards[code] for code in tier] for tier in scenario.dealt]
    board.decks = [[cards[code] for code in tier] for tier in scenario.decks]
    noble_registry = dict(NOBLES)
    board.nobles = [
        (code, dict(noble_registry[code])) for code in scenario.nobles_in_order
    ]
    initial_gems = _pairs_to_dict(scenario.initial_gems, field="initial_gems")
    # The legacy engine enumerates ``board.gems`` in insertion order when it
    # constructs legal actions.  Rebuild that mapping in the engine's frozen
    # colour order so a loaded snapshot reproduces not just the legal set but
    # also the action-list order seen by legacy opponents.
    board.gems = {colour: initial_gems[colour] for colour in GEM_ORDER}
    state.board = board
    state.agents = [
        SplendorState.AgentState(index) for index in range(scenario.n_seats)
    ]
    state.agent_to_move = scenario.current_agent_index
    _validate_initial_agents(state)
    return state


def install_scenario(
    rule: LimitRoundsGameRule,
    scenario: ScenarioV1,
) -> LimitRoundsGameRule:
    """Install a fresh snapshot into an existing compatible rule instance."""
    if type(rule) is not LimitRoundsGameRule:
        raise ScenarioValidationError("ScenarioV1 requires the exact frozen rule type")
    if rule.num_of_agent != scenario.n_seats:
        raise ScenarioValidationError("rule seat count does not match scenario")
    rule.current_game_state = state_from_scenario(scenario)
    rule.current_agent_index = scenario.current_agent_index
    rule.action_counter = 0
    rule.private_information = None
    return rule


def rule_from_scenario(
    scenario: ScenarioV1,
    rule_type: type[LimitRoundsGameRule] = LimitRoundsGameRule,
) -> LimitRoundsGameRule:
    """Create a rule without first constructing and discarding a random board."""
    validate_scenario(scenario)
    if rule_type is not LimitRoundsGameRule:
        raise ScenarioValidationError("ScenarioV1 requires the exact frozen rule type")
    rule = cast(LimitRoundsGameRule, object.__new__(rule_type))
    rule.num_of_agent = scenario.n_seats
    rule.current_game_state = state_from_scenario(scenario)
    rule.current_agent_index = scenario.current_agent_index
    rule.action_counter = 0
    rule.private_information = None
    return rule


def scenario_collection_sha256(scenarios: Sequence[ScenarioV1]) -> str:
    """Hash a complete, provenance-bearing scenario collection canonically."""
    if not scenarios:
        raise ScenarioValidationError("scenario collection must not be empty")
    for scenario in scenarios:
        validate_scenario(scenario)
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ScenarioValidationError("scenario collection contains duplicate states")
    # Hash each row before sorting so streaming readers only need to retain
    # ``(scenario_id, row_digest)`` pairs rather than every decoded row.
    rows = sorted(
        (
            scenario.scenario_id,
            sha256_canonical_json(scenario.to_dict()),
        )
        for scenario in scenarios
    )
    return sha256_canonical_json(rows)


def scenario_state_set_sha256(scenarios: Sequence[ScenarioV1]) -> str:
    """Hash only the sorted state identities for split-overlap audits."""
    if not scenarios:
        raise ScenarioValidationError("scenario state set must not be empty")
    for scenario in scenarios:
        validate_scenario(scenario)
    ids = sorted(scenario.scenario_id for scenario in scenarios)
    if len(ids) != len(set(ids)):
        raise ScenarioValidationError("scenario state set contains duplicates")
    return sha256_canonical_json(ids)


def with_sampling_metadata(
    scenario: ScenarioV1,
    *,
    selection_kind: SelectionKind,
    inclusion_probability: float,
    selection_stratum: str | None,
    selection_design_sha256: str | None = None,
) -> ScenarioV1:
    """Attach selection metadata without changing replay-state identity."""
    updated = replace(
        scenario,
        selection_kind=selection_kind,
        inclusion_probability=inclusion_probability,
        selection_stratum=selection_stratum,
        selection_design_sha256=selection_design_sha256,
    )
    validate_scenario(updated)
    return updated


def scenario_initial_state_sha256(state: SplendorState) -> str:
    """Hash a pristine loaded state using the same replay payload as ScenarioV1."""
    frozen = scenario_from_state(
        state,
        source_segment="ci_smoke",
        source_seed=resolve_segment("ci_smoke").start,
        selection_kind="ci-fixture",
    )
    return frozen.canonical_state_sha256


def scenario_from_json(raw: str) -> ScenarioV1:
    """Parse one canonical-compatible JSON scenario."""
    import json  # noqa: PLC0415 - keep schema dependencies at module boundary

    payload: Any = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ScenarioValidationError("scenario JSON root must be an object")
    return ScenarioV1.from_dict(payload)


__all__ = [
    "INITIAL_GEMS_2P",
    "SCENARIO_DECK_COUNTS",
    "SCENARIO_SCHEMA_VERSION",
    "ScenarioV1",
    "ScenarioValidationError",
    "action_registry_hash",
    "action_registry_hash_for",
    "card_registry_hash",
    "compute_strata_v1",
    "generate_scenario",
    "install_scenario",
    "rule_from_scenario",
    "scenario_collection_sha256",
    "scenario_from_json",
    "scenario_from_state",
    "scenario_initial_state_sha256",
    "scenario_state_set_sha256",
    "state_from_scenario",
    "validate_scenario",
    "with_sampling_metadata",
]
