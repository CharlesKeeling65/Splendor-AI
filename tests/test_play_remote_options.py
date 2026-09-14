"""Offline tests for the phase-6 harness model and provenance plumbing."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from splendor.browser.dom_extractor import Snapshot
from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.play_remote import (
    EventWriter,
    _GameContext,
    _model_for,
    _parse_args,
    _ranking_payload,
    _resolve_profile_ids,
    _seat_compatibility_listener,
)


def test_model_for_maps_comma_list_by_bot_slot() -> None:
    options: dict[str, Any] = {"model": "bot1,bot2"}
    assert _model_for(0, options) == "bot1"
    assert _model_for(1, options) == "bot2"


def test_model_for_single_id_shared_by_all_bots() -> None:
    options: dict[str, Any] = {"model": "round2"}
    assert _model_for(0, options) == "round2"
    assert _model_for(3, options) == "round2"


def test_resolve_profile_ids_requires_explicit_count_match() -> None:
    options: dict[str, Any] = {
        "bots": 2, "explicit_profile_ids": ["Profile 1"],
    }
    with pytest.raises(SystemExit, match="exactly 2"):
        _resolve_profile_ids(options)


def test_driver_profile_space_name_and_script() -> None:
    legacy = EgoBrowserDriver("space")
    assert legacy._space_name() == "space"  # noqa: SLF001
    assert "useOrCreateTaskSpace" in legacy._select_task_space()  # noqa: SLF001

    pinned = EgoBrowserDriver("space", profile_id="Default")
    assert pinned._space_name() == "space@Default"  # noqa: SLF001
    script = pinned._select_task_space()  # noqa: SLF001
    assert "listTaskSpaces" in script  # find-or-create: profileId only at creation
    assert "taskSpace" in script
    assert '"Default"' in script
    assert "profileId" in script


def test_parse_args_accepts_global_winrate_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "play-web-remote",
            "--server",
            "localhost:8765",
            "--model",
            "actor0,actor1",
            "--bots",
            "2",
            "--winrate-model",
            "evaluator",
        ],
    )
    options = _parse_args()
    assert options["model"] == "actor0,actor1"
    assert options["winrate_model"] == "evaluator"


def test_ranking_payload_accepts_canonical_and_legacy_scores() -> None:
    ranked = _ranking_payload(
        [{"idx": 0, "score": 1.5}, {"idx": 1, "q": -0.25}]
    )
    assert ranked[0]["score"] == ranked[0]["q"] == 1.5
    assert ranked[1]["score"] == ranked[1]["q"] == -0.25
    assert ranked[0]["desc"]


def _board_snapshot(seats: int) -> Snapshot:
    return {
        "dealt": [[None, None, None, None] for _ in range(3)],
        "deck_counts": [1, 1, 1],
        "nobles": [],
        "supply": {},
        "panels": [
            {
                "seat": seat,
                "score": 0,
                "card_counts": {},
                "gems": {},
                "reserved_tiers": [],
            }
            for seat in range(1, seats + 1)
        ],
        "my_seat": 1,
        "my_reserved": [],
        "status": "等待你操作",
        "payment_options": None,
        "noble_options": None,
    }


def test_seat_guard_uses_actual_board_panels_for_both_models() -> None:
    models = {
        "actor": {"feature_version": "v1"},
        "evaluator": {"feature_version": "public-v2-multi"},
    }
    guard = _seat_compatibility_listener(models, "actor", "evaluator")
    guard(_board_snapshot(3))
    guard(_board_snapshot(4))

    # The evaluator is checked independently of the acting model.
    mixed = {
        "actor": {"feature_version": "v1"},
        "evaluator": {"feature_version": "public-v2"},
    }
    guard = _seat_compatibility_listener(mixed, "actor", "evaluator")
    with pytest.raises(ValueError, match="exactly 2"):
        guard(_board_snapshot(3))

    public_v2 = {"model": {"feature_version": "public-v2"}}
    guard = _seat_compatibility_listener(public_v2, "model", "model")
    with pytest.raises(ValueError, match="exactly 2"):
        guard(_board_snapshot(3))
    guard = _seat_compatibility_listener(public_v2, "model", "model")
    guard(_board_snapshot(2))

    # A board-less room snapshot does not invent a seat count or fail early.
    guard = _seat_compatibility_listener(public_v2, "model", "model")
    empty = _board_snapshot(0)
    empty["deck_counts"] = [0, 0, 0]
    guard(empty)


class _RecordingEstimatorClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def estimate_winrate(
        self,
        model_id: str,
        snapshot: Snapshot,
        actor_seat: int,
        *,
        n_rollouts: int,
    ) -> dict[str, Any]:
        del snapshot, n_rollouts
        self.calls.append((model_id, actor_seat))
        return {"win_rates": [0.5, 0.5]}


def test_context_estimate_uses_winrate_model_and_emits_provenance(
    tmp_path: Path,
) -> None:
    client = _RecordingEstimatorClient()
    options: dict[str, Any] = {
        "model": "actor",
        "winrate_model": "evaluator",
        "n_rollouts": 2,
    }
    events = EventWriter(tmp_path, 0)
    context = _GameContext(
        bot_id=0,
        game_index=0,
        options=options,
        client=client,  # type: ignore[arg-type]
        events=events,
        my_seat=1,
    )
    assert context.estimate(_board_snapshot(2), 2) == [0.5, 0.5]
    assert client.calls == [("evaluator", 2)]

    context.emit({"type": "remote_act", "score_kind": "policy_logit"})
    payload = json.loads((tmp_path / "bot0.jsonl").read_text(encoding="utf-8"))
    assert payload["acting_model_id"] == "actor"
    assert payload["winrate_model_id"] == "evaluator"
    assert payload["score_kind"] == "policy_logit"
    assert payload["winrate_mode"] == "homogeneous_selfplay_proxy"
