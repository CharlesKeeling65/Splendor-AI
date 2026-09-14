"""Z0 smoke: the benchmark CLI runs end to end and writes its report."""

import json
from pathlib import Path

from splendor.agents.our_agents.alphazero.benchmark import main


def test_benchmark_main_smoke(tmp_path: Path):
    exit_code = main(
        [
            "--output",
            str(tmp_path / "z0"),
            "--seed",
            "830_200",
            "--checkpoints",
            "0",
            "4",
            "--sims",
            "8",
            "--trees",
            "2",
            "--repeats",
            "2",
            "--devices",
            "cpu",
        ]
    )
    assert exit_code == 0
    report = json.loads((tmp_path / "z0" / "z0_report.json").read_text())
    assert set(report["positions"]) == {"move_0", "move_4"}
    for entry in report["positions"].values():
        assert entry["apply_undo"]["fingerprint_mismatches"] == 0
        assert "search_sims8_trees2" in entry
