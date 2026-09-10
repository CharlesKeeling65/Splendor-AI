"""Offline tests for the phase-6 harness: per-bot model & profile plumbing."""

from typing import Any

import pytest

from splendor.browser.ego_driver import EgoBrowserDriver
from splendor.play_remote import (
    _model_for,  # noqa: PLC2701
    _resolve_profile_ids,  # noqa: PLC2701
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
    assert "taskSpace" in script
    assert '"Default"' in script
    assert "profileId" in script
