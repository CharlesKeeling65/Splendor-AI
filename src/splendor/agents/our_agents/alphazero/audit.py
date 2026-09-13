"""Information audit for the AZ search path: hidden-order invariance.

Contract under audit: an evaluator (and the search harness around it) must
never observe the *true* hidden information - unseen deck order and rival
face-down reservation identities. Two positions identical in all public
information but differing in the hidden assignment must yield byte-identical
search outputs under a fixed rng seed.

The audit runs two checks per position:

1. **Clean check** - the evaluator under test produces identical visit
   distributions on the original position and on a re-hidden copy (true
   hidden assignment re-sampled, public information untouched);
2. **Positive control** - a deliberately leaking evaluator (it biases priors
   by reading the *true* deck top of the rule it was constructed with) must
   break the equality. Without this control a vacuously-passing audit (e.g.
   both runs degenerate to identical uniform output) would look like success.

This mirrors the teacher information audit used for the GA DAgger labels
(positive control included; 6/77 leak episodes were caught there).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from splendor.agents.our_agents.alphazero.evaluator import Evaluator, UniformEvaluator
from splendor.agents.our_agents.alphazero.mcts import az_search
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType


def reshuffle_hidden(
    rule: SplendorGameRule, root_seat: int, rng: np.random.Generator
) -> None:
    """Re-sample the true hidden assignment in place (public info unchanged).

    Same sort-then-shuffle contract as
    :func:`splendor.agents.our_agents.dqn.search.sample_hidden`: the unseen
    union (decks + rival face-down reservations) gets a canonical order
    before shuffling, so the result depends only on (public info, rng) -
    never on the previous hidden order. Mutates the rule; callers pass a
    sacrificial copy.
    """
    state = rule.current_game_state
    rival = state.agents[1 - root_seat]
    for tier, deck in enumerate(state.board.decks):
        slots = [i for i, c in enumerate(rival.cards["yellow"]) if c.deck_id == tier]
        unseen = sorted(
            [*deck, *(rival.cards["yellow"][i] for i in slots)], key=lambda c: c.code
        )
        rng.shuffle(unseen)
        for index in slots:
            rival.cards["yellow"][index] = unseen.pop()
        state.board.decks[tier] = unseen


class DeckPeekingEvaluator:
    """Positive control: leaks the true deck order into the prior.

    Biases prior mass heavily toward reserving board cards of the tier whose
    *true* next draw (top of the hidden deck) has the lexicographically
    largest card code - a direct function of the hidden order, so two
    positions differing only in that order receive different priors.
    Constructed with the rule it should peek at - the reference run peeks
    the original, the re-hidden run peeks the re-hidden copy, so the two
    disagree exactly when hidden order leaks into outputs.
    """

    key_kind = "fingerprint"

    def __init__(self, true_rule: SplendorGameRule) -> None:
        self._true_rule = true_rule

    def evaluate(
        self,
        state: SplendorState,
        seat: int,
        indices: list[int],
        actions: list[ActionType],
    ) -> tuple[NDArray[np.float64], float]:
        del state, seat
        true_state = self._true_rule.current_game_state
        tops = {
            tier: (deck[-1].code if deck else "")
            for tier, deck in enumerate(true_state.board.decks)
        }
        best_tier = max(tops, key=lambda tier: tops[tier])
        prior = np.ones(len(indices), dtype=np.float64)
        for i, action in enumerate(actions):
            card = action.get("card")
            if action.get("type") == "reserve" and card is not None:
                prior[i] += 50.0 * (card.deck_id == best_tier)
        prior /= prior.sum()
        return prior, 0.0


def _search_visits(
    rule: SplendorGameRule,
    evaluator: Evaluator,
    seed: int,
    simulations: int,
    n_trees: int,
) -> NDArray[np.float64]:
    result = az_search(
        rule,
        evaluator,
        np.random.default_rng(seed),
        simulations=simulations,
        n_trees=n_trees,
    )
    return result.visits


def run_information_audit(
    evaluator: Evaluator,
    positions: list[SplendorGameRule],
    *,
    seed: int = 831_000,
    simulations: int = 16,
    n_trees: int = 2,
) -> dict[str, Any]:
    """Run the clean check + positive control over the given positions."""
    clean_equal = True
    control_differs = False
    per_position: list[dict[str, Any]] = []
    for index, rule in enumerate(positions):
        seat = rule.getCurrentAgentIndex()
        if isinstance(evaluator, DeckPeekingEvaluator):
            reference_evaluator: Evaluator = DeckPeekingEvaluator(rule)
        else:
            reference_evaluator = evaluator
        reference = _search_visits(
            rule, reference_evaluator, seed + index, simulations, n_trees
        )

        perturbed_rule = deepcopy(rule)
        reshuffle_hidden(
            perturbed_rule, seat, np.random.default_rng(seed + 7 * index + 1)
        )
        perturbed_evaluator = evaluator
        if isinstance(evaluator, DeckPeekingEvaluator):
            perturbed_evaluator = DeckPeekingEvaluator(perturbed_rule)
        perturbed = _search_visits(
            perturbed_rule, perturbed_evaluator, seed + index, simulations, n_trees
        )
        max_abs_diff = float(np.max(np.abs(reference - perturbed)))
        equal = bool(np.array_equal(reference, perturbed))
        clean_equal &= equal
        if isinstance(evaluator, DeckPeekingEvaluator):
            control_differs |= not equal
        per_position.append(
            {"position": index, "equal": equal, "max_abs_diff": max_abs_diff}
        )
    if isinstance(evaluator, DeckPeekingEvaluator):
        return {
            "check": "positive_control",
            "positions": per_position,
            "control_differs": control_differs,
        }
    return {
        "check": "clean",
        "positions": per_position,
        "clean_equal": clean_equal,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/z0-audit"))
    parser.add_argument("--seed", type=int, default=830_000)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--trees", type=int, default=2)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rule = SplendorGameRule(2)
    moves = 0
    positions: list[SplendorGameRule] = []
    while len(positions) < args.positions and moves < 200:
        if moves >= args.warmup and not rule.gameEnds():
            positions.append(deepcopy(rule))
        if rule.gameEnds():
            rule = SplendorGameRule(2)
            moves = 0
            continue
        state = rule.current_game_state
        seat = rule.getCurrentAgentIndex()
        legal = rule.getLegalActions(state, seat)
        rule.update(legal[int(np.random.randint(len(legal)))])
        moves += 1

    report: dict[str, Any] = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "seed": args.seed,
        "n_positions": len(positions),
        "simulations": args.simulations,
        "n_trees": args.trees,
    }
    report["clean"] = run_information_audit(
        UniformEvaluator(),
        positions,
        seed=args.seed + 100_000,
        simulations=args.simulations,
        n_trees=args.trees,
    )
    report["clean"]["evaluator"] = "UniformEvaluator"
    report["positive_control"] = run_information_audit(
        DeckPeekingEvaluator(positions[0]),
        positions,
        seed=args.seed + 100_000,
        simulations=args.simulations,
        n_trees=args.trees,
    )
    report["positive_control"]["evaluator"] = "DeckPeekingEvaluator"
    passed = bool(report["clean"]["clean_equal"]) and bool(
        report["positive_control"]["control_differs"]
    )
    report["verdict"] = "pass" if passed else "FAIL"
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "audit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clean", "positive_control", "verdict")}, indent=2))
    print(f"Audit report written to {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
