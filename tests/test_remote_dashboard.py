"""Offline dashboard compatibility tests (canonical scores + legacy streams)."""

from pathlib import Path

from splendor.remote.dashboard import _prune_rankings


def test_prune_rankings_preserves_canonical_score_fields() -> None:
    events = [
        {"type": "remote_act", "top": [{"idx": 1, "score": 2.0, "q": 2.0}]},
        {"type": "remote_act", "top": [{"idx": 2, "score": 3.0, "q": 3.0}]},
    ]
    pruned = _prune_rankings(events)
    assert pruned[0]["top"] == []
    assert pruned[1]["top"][0]["score"] == 3.0
    assert pruned[1]["top"][0]["q"] == 3.0


def test_dashboard_has_dynamic_score_and_proxy_labels() -> None:
    html_path = (
        Path(__file__).parents[1]
        / "src"
        / "splendor"
        / "remote"
        / "dashboard.html"
    )
    html = html_path.read_text(encoding="utf-8")
    assert "function scoreKindLabel" in html
    assert 'kind === "policy_logit"' in html
    assert "item && item.score" in html
    assert "homogeneous_selfplay_proxy" in html
    assert "lastStart.seats" in html
    assert "Q 值排序" not in html
