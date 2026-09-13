"""
Advice engine for the browser advisor (plan phase-7 §3.3/§3.4).

Three deterministic, inference-free scorers over the observed position:

* **GA top-k** - the evolved heuristic (`GeneAlgoAgent`'s shipped weights,
  pure numpy) applied to *every* legal action instead of only the argmax,
  with per-action feature-delta attributions as the explanation lines.
* **minimax deep mode** - an alpha-beta wrapper that mirrors the minmax.py
  search pattern but (a) collects *all* root values so the human gets a
  ranking, not just the winner, (b) sorts deterministically instead of
  shuffling, and (c) searches every root action with a full window so the
  ranking is exact. 2-player only (same constraint as the baseline).
* **deck composition** - registry minus every face ever seen (the
  tracker's seen set): the per-tier colour histogram of the *undealt* deck
  plus a grey bucket for unseen faces that left it, and a conservation
  check against the page's own deck counts.

All scoring runs on a *determinized reconstruction* of the position, never
on the raw pseudo state - verified engine facts make the pseudo state
unusable for successor generation: its decks are empty
(``state_builder.py`` F9) so post-buy refills silently do nothing, and the
engine enumerates no deck-reserve actions at all. The reconstruction
follows ``remote/rollout.py``'s precedent with one upgrade: rivals'
reserved slots take the tracker's *identified* faces first and only sample
for the grey entries.
"""

import random
import zlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from splendor.agents.our_agents.alphazero.state_utils import (
    Transactor,
    state_fingerprint,
)
from splendor.agents.our_agents.genetic_algorithm.genes import StrategyGene
from splendor.agents.our_agents.genetic_algorithm.genetic_algorithm_agent import (
    GeneAlgoAgent,
)
from splendor.agents.our_agents.minmax import MiniMaxAgent
from splendor.browser.card_registry import CARD_REGISTRY
from splendor.browser.dom_extractor import CardInfo, Snapshot
from splendor.browser.state_builder import _placeholder_cards, build_pseudo_state
from splendor.splendor.action_text import COLOR_CN, describe_action
from splendor.splendor.features import (
    METRICS_SHAPE,
    extract_metrics,
    normalize_metrics,
)
from splendor.splendor.splendor_model import Card, SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType

from .tracker import ReservationTracker, TrackedReserved

DEFAULT_TOP_K = 5
_DEFAULT_DEPTH = 2
# Mirror of minmax.EXPECTED_AMOUNT_OF_PLAYERS - the deep mode is zero-sum
# 2-seat by construction (opponent seat = 1 - my_index).
TWO_PLAYER_SEATS = 2
_REASON_EPSILON = 1e-4


@dataclass(frozen=True)
class Advice:
    """One ranked move: the action, its score, and why it scores that."""

    action: ActionType
    value: float
    text: str
    reasons: list[str]
    source: str  # "ga" | "minimax"
    best_reply: str | None = None  # minimax only: the opponent's hardest answer


@dataclass(frozen=True)
class DeckRow:
    """Per-tier histogram row: undealt colours plus the grey bucket."""

    tier: int  # 0-based deck id
    counts: dict[str, int]  # undealt cards with known-absent faces, per colour
    grey: int  # unseen faces already out of the deck (unknown reserves,
    # pre-join history) - cannot be attributed to a colour honestly
    deck_count: int  # the page's own remaining-deck reading


@dataclass(frozen=True)
class DeckHistogram:
    rows: tuple[DeckRow, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AffordRow:
    """One purchasable card and what my seat is still missing for it."""

    text: str  # "2级蓝卡3分"
    source: str  # "dealt" | "reserved"
    missing: dict[str, int]  # colour -> count still short after production+gems
    gold_covers: int  # shortfall the hand's gold can cover
    affordable: bool


# Labels mirroring splendor.features.METRICS_SHAPE (display only - the
# extraction itself stays the engine's, per the single-source discipline).
_METRIC_LABEL_GROUPS: tuple[str, ...] = (
    "常量",
    "我的回合数",
    "我的分数",
    "我是否达到15分",
    "我已购卡数",
    "我的预留数",
    "购买力方差",
    "我的黄金数",
    "我的宝石总数",
    "我已购卡(按色)",
    "我的购买力(按色)",
    "我的递减购买力(按色)",
    "在我之前对手的分数",
    "在我之后对手的分数",
    "T1桌面卡宝石距离",
    "T2桌面卡宝石距离",
    "T3桌面卡宝石距离",
    "T1桌面卡回合距离",
    "T2桌面卡回合距离",
    "T3桌面卡回合距离",
    "预留卡宝石距离",
    "预留卡回合距离",
    "贵族宝石距离",
    "贵族卡数距离",
)


def _expanded_labels() -> list[str]:
    """Per-metric labels from the group labels + METRICS_SHAPE widths."""
    labels: list[str] = []
    for label, width in zip(_METRIC_LABEL_GROUPS, METRICS_SHAPE, strict=True):
        if width == 1:
            labels.append(label)
        else:
            labels.extend(f"{label}#{i + 1}" for i in range(width))
    return labels


_METRIC_LABELS: list[str] = _expanded_labels()


def _action_sort_key(action: ActionType) -> tuple:
    """Total, engine-independent ordering: the deterministic replacement
    for minmax.py's ``random.shuffle`` (whose only job was tie variety)."""
    fields = cast(dict[str, Any], action)  # heterogeneous TypedDict access
    card = fields.get("card")
    return (
        str(fields.get("type", "")),
        str(card.code) if card is not None else "",
        tuple(sorted((fields.get("collected_gems") or {}).items())),
        tuple(sorted((fields.get("returned_gems") or {}).items())),
        str(fields.get("noble") or ""),
    )


def _card_label(deck_id: int, colour: str, points: int) -> str:
    tier_cn = f"{deck_id + 1}级" if deck_id >= 0 else ""
    points_cn = f"{points}分" if points else ""
    return f"{tier_cn}{COLOR_CN.get(colour, colour)}卡{points_cn}"


def _missing_text(missing: dict[str, int], gold_covers: int) -> str:
    if not missing:
        return "已可支付"
    parts = [f"{COLOR_CN.get(c, c)}{n}" for c, n in sorted(missing.items()) if n]
    text = "还差 " + ("".join(parts) if parts else "0")
    if gold_covers:
        text += f"（金可补{gold_covers}）"
    return text


class AdvisorEngine:
    """
    Deterministic advice computation for one seat of one game.

    Owns a scratch :class:`SplendorGameRule` (never advanced - states are
    passed explicitly, and the reconstruction is installed as
    ``current_game_state`` only so :class:`Transactor` can apply/undo on it).
    """

    def __init__(
        self,
        panel_count: int,
        my_index: int,
        *,
        seed: int = 0,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if panel_count < TWO_PLAYER_SEATS:
            raise ValueError(f"panel_count must be >= 2, got {panel_count}")
        if my_index not in range(panel_count):
            raise ValueError(f"my_index {my_index} outside 0..{panel_count - 1}")
        self._my_index = my_index
        self._top_k = top_k
        self._seed = seed
        self._rule = SplendorGameRule(panel_count)
        self._trans = Transactor()
        self._ga = GeneAlgoAgent(my_index)
        self._ga_strategies = (
            self._ga.stategy_gene_1,
            self._ga.stategy_gene_2,
            self._ga.stategy_gene_3,
        )
        # The baseline's evaluator is reused deliberately (the deep mode
        # must score exactly like the minmax agent it mirrors), without
        # subclassing or modifying the baseline file.
        self._minimax_eval = MiniMaxAgent(my_index)._evaluation_function  # noqa: SLF001

    # ----- determinized reconstruction (§3.3) --------------------------------
    def build_reconstruction(
        self,
        snapshot: Snapshot,
        tracker: ReservationTracker,
        *,
        turns: int = 0,
    ) -> SplendorState:
        """
        Full engine state from the snapshot: pseudo state + tracker-backed
        unknown fill. Same state -> same reconstruction (rng keyed on the
        pseudo-state fingerprint + the engine seed).
        """
        state = build_pseudo_state(snapshot, self._my_index, turns=turns)
        fingerprint = state_fingerprint(state, include_last_action=False)
        rng = random.Random(zlib.crc32(fingerprint) ^ self._seed)
        seen = tracker.seen_face_codes()
        unseen = [card for card in CARD_REGISTRY.values() if card.code not in seen]

        for agent_index, agent in enumerate(state.agents):
            if agent_index == self._my_index:
                continue
            agent.cards["yellow"] = self._rival_reserved_faces(
                snapshot, tracker, agent_index, unseen, rng
            )

        state.board.decks = [[], [], []]
        for card in unseen:
            state.board.decks[card.deck_id].append(card)
        for deck in state.board.decks:
            rng.shuffle(deck)
        self._rule.current_game_state = state
        return state

    def _rival_reserved_faces(
        self,
        snapshot: Snapshot,
        tracker: ReservationTracker,
        agent_index: int,
        unseen: list[Card],
        rng: random.Random,
    ) -> list[Card]:
        """Tracker faces first; grey entries sample from the unseen pool."""
        seat = agent_index + 1
        entries = tracker.reserved(seat)
        if not entries:
            # Seat never seen by the tracker: fall back to the page's backs.
            tiers = list(snapshot["panels"][agent_index]["reserved_tiers"])
            entries = [
                TrackedReserved(tier=tier, card=None, frame_seq=-1) for tier in tiers
            ]
        faces: list[Card] = []
        for entry in entries:
            if entry.card is not None:
                faces.append(entry.card)
                continue
            tier = entry.tier
            candidates = [i for i, c in enumerate(unseen) if c.deck_id == tier]
            if candidates:
                faces.append(unseen.pop(candidates[rng.randrange(len(candidates))]))
            else:
                # Exhausted unseen pool (mid-join drift): keep a count-correct
                # placeholder so the state stays structurally valid.
                faces.append(_placeholder_cards("yellow", 1)[0])
        return faces

    # ----- GA top-k (§3.4) -----------------------------------------------------
    def ga_top_k(self, state: SplendorState) -> list[Advice]:
        """
        Rank every legal action with the evolved heuristic; the top-1 must
        equal ``GeneAlgoAgent.SelectAction`` on the same state (tested).
        """
        legal = self._rule.getLegalActions(state, self._my_index)
        if not legal:
            return []
        # Transactor applies on rule.current_game_state - install the state
        # under test so apply/undo mutate exactly the object being scored.
        self._rule.current_game_state = state
        before = normalize_metrics(extract_metrics(state, self._my_index))
        strategy: StrategyGene = self._ga.manager_gene.select_strategy(
            before, self._ga_strategies
        )
        dna = strategy.dna
        advice: list[Advice] = []
        for action in legal:
            snap = self._trans.apply(self._rule, action, self._my_index)
            after = normalize_metrics(extract_metrics(state, self._my_index))
            self._trans.undo(self._rule, action, self._my_index, snap)
            value = float(np.matmul(after, dna))
            advice.append(
                Advice(
                    action=action,
                    value=value,
                    text=describe_action(action),
                    reasons=self._reasons(dna, after - before),
                    source="ga",
                )
            )
        advice.sort(key=lambda a: a.value, reverse=True)
        return advice[: self._top_k]

    @staticmethod
    def _reasons(dna: NDArray, delta: NDArray) -> list[str]:
        contribution = dna * delta
        order = np.argsort(-np.abs(contribution))
        reasons = []
        for index in order[:3]:
            weight = float(contribution[index])
            if abs(weight) <= _REASON_EPSILON:
                break
            sign = "+" if weight >= 0 else "−"
            reasons.append(f"{_METRIC_LABELS[int(index)]} {sign}{abs(weight):.2f}")
        return reasons

    # ----- minimax deep mode (§3.4) --------------------------------------------
    def minimax_top_k(self, state: SplendorState, depth: int = _DEFAULT_DEPTH) -> list[Advice]:
        """
        Alpha-beta over the reconstruction with every root value collected.

        Each root action is searched with a *full window* so the ranking is
        exact (root-level alpha carry would return bounds, not values, for
        later actions). 2-player only - the evaluation is zero-sum seating.
        """
        if len(state.agents) != TWO_PLAYER_SEATS:
            raise ValueError(
                f"minimax deep mode supports {TWO_PLAYER_SEATS} seats, "
                f"got {len(state.agents)}"
            )
        legal = self._rule.getLegalActions(state, self._my_index)
        if not legal:
            return []
        self._rule.current_game_state = state  # see ga_top_k: Transactor's target
        ranked: list[Advice] = []
        for action in sorted(legal, key=_action_sort_key):
            snap = self._trans.apply(self._rule, action, self._my_index)
            value = self._search(state, depth - 1, False, -np.inf, np.inf)
            self._trans.undo(self._rule, action, self._my_index, snap)
            ranked.append(
                Advice(
                    action=action,
                    value=value,
                    text=describe_action(action),
                    reasons=[],
                    source="minimax",
                )
            )
        ranked.sort(key=lambda a: a.value, reverse=True)
        top = ranked[: self._top_k]
        if top:
            best = top[0]
            reply = self._best_reply(state, best.action, depth - 1)
            top[0] = Advice(
                action=best.action,
                value=best.value,
                text=best.text,
                reasons=[f"对手最狠回应: {reply}" if reply else "对手无合法动作"],
                source=best.source,
                best_reply=reply,
            )
        return top

    def _search(
        self,
        state: SplendorState,
        depth: int,
        is_maximizing: bool,
        alpha: float,
        beta: float,
    ) -> float:
        """The minmax.py recursion, transplanted onto apply/undo + fixed
        ordering (mirror of ``_select_action_recursion`` minus shuffle)."""
        if depth <= 0:
            return float(self._minimax_eval(state))
        seat = self._my_index if is_maximizing else 1 - self._my_index
        actions = self._rule.getLegalActions(state, seat)
        if not actions:  # all-pass / terminal-ish: evaluate as a leaf
            return float(self._minimax_eval(state))
        best = -np.inf if is_maximizing else np.inf
        for action in sorted(actions, key=_action_sort_key):
            snap = self._trans.apply(self._rule, action, seat)
            value = self._search(state, depth - 1, not is_maximizing, alpha, beta)
            self._trans.undo(self._rule, action, seat, snap)
            if is_maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            if beta <= alpha:
                break
        return float(best)

    def _best_reply(self, state: SplendorState, root: ActionType, depth: int) -> str | None:
        """The opponent's hardest answer to ``root`` (for the top-1 line)."""
        opponent = 1 - self._my_index
        snap = self._trans.apply(self._rule, root, self._my_index)
        try:
            actions = self._rule.getLegalActions(state, opponent)
            if not actions:
                return None
            worst: tuple[float, ActionType] | None = None
            for action in sorted(actions, key=_action_sort_key):
                inner = self._trans.apply(self._rule, action, opponent)
                value = self._search(state, max(depth - 1, 0), True, -np.inf, np.inf)
                self._trans.undo(self._rule, action, opponent, inner)
                if worst is None or value < worst[0]:
                    worst = (value, action)
            assert worst is not None
            return describe_action(worst[1])
        finally:
            self._trans.undo(self._rule, root, self._my_index, snap)

    # ----- deck composition + planning (§3.4) -----------------------------------
    def deck_histogram(
        self, snapshot: Snapshot, tracker: ReservationTracker
    ) -> DeckHistogram:
        """
        Undealt-deck colour histogram from the registry minus the seen set,
        with a per-tier grey bucket and a conservation warning.
        """
        seen = tracker.seen_face_codes()
        remaining: list[dict[str, int]] = [{} for _ in range(3)]
        for card in CARD_REGISTRY.values():
            if card.code not in seen:
                counts = remaining[card.deck_id]
                counts[card.colour] = counts.get(card.colour, 0) + 1
        warnings: list[str] = []
        rows: list[DeckRow] = []
        for tier in range(3):
            total_unseen = sum(remaining[tier].values())
            deck_count = snapshot["deck_counts"][tier]
            grey = total_unseen - deck_count
            if grey < 0:
                warnings.append(
                    f"T{tier + 1}: 页面牌堆计数 {deck_count} 低于注册表推算 "
                    f"{total_unseen}（抽取或解析异常）"
                )
                grey = 0
            rows.append(
                DeckRow(
                    tier=tier,
                    counts=dict(remaining[tier]),
                    grey=grey,
                    deck_count=deck_count,
                )
            )
        return DeckHistogram(rows=tuple(rows), warnings=tuple(warnings))

    def affordability(self, snapshot: Snapshot) -> list[AffordRow]:
        """What each purchasable card still costs my seat (gold-aware)."""
        my_index = self._my_index
        panel = snapshot["panels"][my_index]
        production = panel["card_counts"]
        gems = panel["gems"]
        rows: list[AffordRow] = []
        purchasable: list[tuple[str, CardInfo]] = [
            ("dealt", info)
            for row in snapshot["dealt"]
            for info in row
            if info is not None
        ] + [("reserved", info) for info in snapshot["my_reserved"]]
        for source, info in purchasable:
            missing: dict[str, int] = {}
            for colour, cost in info["cost"].items():
                if colour == "yellow":
                    continue  # gold costs are wildcard-covered below
                short = cost - production.get(colour, 0) - gems.get(colour, 0)
                if short > 0:
                    missing[colour] = short
            shortfall = sum(missing.values())
            gold_covers = min(gems.get("yellow", 0), shortfall)
            rows.append(
                AffordRow(
                    text=_card_text(info),
                    source=source,
                    missing=missing,
                    gold_covers=gold_covers,
                    affordable=shortfall <= gold_covers,
                )
            )
        return rows

    def noble_progress(self, snapshot: Snapshot) -> list[dict[str, Any]]:
        """Per-noble visit progress for my seat (colour -> still short)."""
        panel = snapshot["panels"][self._my_index]
        production = panel["card_counts"]
        progress: list[dict[str, Any]] = []
        for noble in snapshot["nobles"]:
            missing = {
                colour: cost - production.get(colour, 0)
                for colour, cost in noble["requirements"].items()
                if cost - production.get(colour, 0) > 0
            }
            progress.append({"requirements": noble["requirements"], "missing": missing})
        return progress


def _card_text(info: CardInfo) -> str:
    tier = info["tier"]
    points = info["points"]
    colour = COLOR_CN.get(info["colour"], info["colour"])
    return f"{tier + 1}级{colour}卡" + (f"{points}分" if points else "")


__all__ = [
    "DEFAULT_TOP_K",
    "Advice",
    "AdvisorEngine",
    "AffordRow",
    "DeckHistogram",
    "DeckRow",
]
