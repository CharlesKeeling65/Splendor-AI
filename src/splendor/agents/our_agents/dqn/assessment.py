"""Audit staged experiment artifacts and optionally confirm on fresh deals.

Confirmation is a new, declared comparison, not a new checkpoint-selection pass.
All three corrected training seeds and both archived controls are retained.
"""

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmark import benchmark
from .experiment import write_json
from .utils import load_saved_dqn

CONFIRMATION_START = 920_000
CONFIRMATION_DEALS = 50


def paired_difference(reports: dict[str, Any]) -> dict[str, Any]:
    """Exploratory two-way bootstrap: training replicas and paired deal clusters.

    Three replicas are too few for a strong population significance claim;
    this interval illustrates uncertainty, it is not a promotion criterion.
    """
    control = {
        (r["seed"], r["seat"]): float(r["outcome"] > 0)
        for r in reports["archive-1"]["records"]
    }
    seeds = sorted({seed for seed, _ in control})
    names = sorted(name for name in reports if name.startswith("corrected-"))
    differences = []
    for name in names:
        records = {
            (r["seed"], r["seat"]): float(r["outcome"] > 0)
            for r in reports[name]["records"]
        }
        if records.keys() != control.keys():
            raise ValueError("comparison games are not paired")
        differences.append(
            [
                [records[(seed, seat)] - control[(seed, seat)] for seat in (0, 1)]
                for seed in seeds
            ]
        )
    array = np.asarray(differences)
    rng = np.random.default_rng(20260907)
    bootstrap = []
    for _ in range(10_000):
        replicas = rng.integers(len(names), size=len(names))
        deals = rng.integers(len(seeds), size=len(seeds))
        bootstrap.append(float(array[replicas][:, deals].mean()))
    return {
        "models": names,
        "control": "archive-1",
        "mean_win_rate_difference": float(array.mean()),
        "per_model_difference": {
            name: float(array[i].mean()) for i, name in enumerate(names)
        },
        "exploratory_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "bootstrap_seed": 20260907,
        "replicates": 10_000,
        "method": "resample training models and paired deal clusters; keep both seats; only 3 trained replicas",
    }


def audit_report(report: dict[str, Any], seeds: list[int]) -> None:
    """Recompute every rate/count from unique seed-seat game records."""
    rows = report["records"]
    expected = {(seed, seat) for seed in seeds for seat in (0, 1)}
    if len(rows) != len(expected) or {(r["seed"], r["seat"]) for r in rows} != expected:
        raise ValueError("missing/duplicate/unexpected seed-seat records")
    for row in rows:
        scores = [row["score"], row["rival_score"]]
        if not np.isfinite(scores).all() or row["outcome"] != np.sign(
            scores[0] - scores[1]
        ):
            raise ValueError("outcome disagrees with engine scores")
    counts = [sum(r["outcome"] == outcome for r in rows) for outcome in (1, 0, -1)]
    if counts != [report["wins"], report["draws"], report["losses"]]:
        raise ValueError("reported W/D/L disagrees with raw records")
    if report["games"] != len(rows) or not np.isclose(
        report["win_rate"], counts[0] / len(rows)
    ):
        raise ValueError("bad rate denominator")
    if not np.isclose(report["mean_score"], np.mean([r["score"] for r in rows])):
        raise ValueError("bad mean score")


def audit_suite(folder: Path) -> dict[str, Any]:  # noqa: C901, PLR0912 - explicit artifact checks
    """Require all runs complete, uniform budgets, finite logs and fresh test seeds."""
    manifest = json.loads((folder / "manifest.json").read_text())
    results = json.loads((folder / "results.json").read_text())
    status = json.loads((folder / "suite-status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("suite has not completed")
    expected_names = {
        f"{v}-{s}" for v in manifest["variants"] for s in manifest["seeds"]
    }
    expected_names.update(f"archive-{i}" for i in range(len(manifest["baseline"])))
    if set(results) != expected_names:
        raise ValueError("missing or unexpected models")
    runs: dict[str, Any] = {}
    for variant in manifest["variants"]:
        for seed in manifest["seeds"]:
            name = f"{variant}-{seed}"
            run = folder / name
            config = json.loads((run / "config.json").read_text())
            if any(config[key] != value for key, value in manifest["config"].items()):
                raise ValueError(f"run budget/settings disagree with manifest: {name}")
            run_status = json.loads((run / "status.json").read_text())
            if (
                run_status["status"] != "completed"
                or run_status["step"] != manifest["config"]["steps"]
            ):
                raise ValueError(f"incomplete training: {name}")
            with (run / "training.csv").open() as handle:
                logs = list(csv.DictReader(handle))
            if not logs or int(logs[-1]["step"]) != config["steps"]:
                raise ValueError(f"incomplete log: {name}")
            if any(not np.isfinite(float(v)) for row in logs for v in row.values()):
                raise ValueError(f"nonfinite training statistic: {name}")
            if [int(row["step"]) for row in logs] != sorted(
                {int(row["step"]) for row in logs}
            ):
                raise ValueError(f"nonmonotonic step: {name}")
            if config["steps"] < config["buffer_size"] and config["steps"] - int(
                logs[-1]["buffer"]
            ) not in range(3):
                raise ValueError(f"unexpected replay sample loss: {name}")
            test_seeds = list(
                range(config["test_start"], config["test_start"] + config["test_deals"])
            )
            if set(test_seeds) & set(
                range(
                    config["validation_start"],
                    config["validation_start"] + config["validation_deals"],
                )
            ):
                raise ValueError("validation/test leakage")
            for report in results[name].values():
                audit_report(report, test_seeds)
            selected = torch.load(
                run / "best.pth", map_location="cpu", weights_only=False
            )["step"]
            validation_steps = sorted(
                set(
                    range(
                        config["eval_every"], config["steps"] + 1, config["eval_every"]
                    )
                )
                | {config["steps"]}
            )
            validations = [
                (step, json.loads((run / f"validation-{step}.json").read_text()))
                for step in validation_steps
            ]
            for _, validation in validations:
                components = validation.get("opponents", {"single": validation})
                if not np.isclose(
                    validation["win_rate"],
                    np.mean([r["win_rate"] for r in components.values()]),
                ):
                    raise ValueError("validation macro average is incorrect")
                for component in components.values():
                    audit_report(
                        component,
                        list(
                            range(
                                config["validation_start"],
                                config["validation_start"] + config["validation_deals"],
                            )
                        ),
                    )
            if selected != max(validations, key=lambda item: item[1]["win_rate"])[0]:
                raise ValueError(
                    f"selected checkpoint disagrees with validation rule: {name}"
                )
            runs[name] = {
                "selected_step": selected,
                "elapsed_seconds": run_status["elapsed_seconds"],
                "final_q": float(logs[-1]["q_mean"]),
                "max_q": max(float(r["q_mean"]) for r in logs),
                "final_td_p90": float(logs[-1]["td_abs_p90"]),
                "buffer": int(logs[-1]["buffer"]),
                "opponent_games": run_status["opponent_games"],
            }
    for i in range(len(manifest["baseline"])):
        for report in results[f"archive-{i}"].values():
            audit_report(report, test_seeds)
    summary: dict[str, Any] = {}
    for variant in manifest["variants"]:
        names = [f"{variant}-{seed}" for seed in manifest["seeds"]]
        summary[variant] = {}
        for opponent in results[names[0]]:
            reports = [results[name][opponent] for name in names]
            summary[variant][opponent] = {
                "wins": sum(r["wins"] for r in reports),
                "games": sum(r["games"] for r in reports),
                "seed_win_rates": [r["win_rate"] for r in reports],
                "training_seed_sd": float(
                    np.std([r["win_rate"] for r in reports], ddof=1)
                )
                if len(reports) > 1
                else None,
                "mean_action_seconds": float(
                    np.mean([r["action_seconds_mean"] for r in reports])
                ),
            }
    return {
        "assessment": "share with caveats",
        "models_checked": len(results),
        "games_checked": sum(
            r["games"] for reports in results.values() for r in reports.values()
        ),
        "summary": summary,
        "runs": runs,
        "caveats": [
            "10k-step pilots, not convergence",
            "only three trained replicas",
            "paired/shared deals are not independent Bernoulli trials",
            "archived controls have different budgets and opponent distributions",
            "later stages inherit frozen normalization; feature contribution not isolated",
            "search adds compute; no equal-decision-time superiority claim",
        ],
    }


def confirm_job(path: Path) -> dict[str, Any]:
    """Fixed checkpoint, 50 new deals in both seats, one opponent."""
    torch.set_num_threads(1)
    net = load_saved_dqn(path).to("cuda")
    seeds = list(range(CONFIRMATION_START, CONFIRMATION_START + CONFIRMATION_DEALS))
    report = benchmark(net, "minimax", seeds)
    audit_report(report, seeds)
    return report


def main() -> None:
    """Keep audit/confirmation artifacts separate from the original test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit = audit_suite(args.suite)
    write_json(args.suite / "audit.json", audit)
    print(json.dumps(audit["summary"], indent=2))
    if not args.confirm:
        confirmation = args.suite / "confirmation"
        if (confirmation / "status.json").exists():
            if (
                json.loads((confirmation / "status.json").read_text())["status"]
                != "completed"
            ):
                raise ValueError("confirmation is incomplete")
            existing_reports = json.loads((confirmation / "results.json").read_text())
            for report in existing_reports.values():
                audit_report(
                    report,
                    list(
                        range(
                            CONFIRMATION_START, CONFIRMATION_START + CONFIRMATION_DEALS
                        )
                    ),
                )
            if "archive-1" in existing_reports:
                write_json(
                    confirmation / "comparison.json",
                    paired_difference(existing_reports),
                )
        return
    destination = args.suite / "confirmation"
    destination.mkdir(exist_ok=False)
    manifest = json.loads((args.suite / "manifest.json").read_text())
    jobs = {
        f"corrected-{s}": args.suite / f"corrected-{s}" / "best.pth"
        for s in manifest["seeds"]
    }
    jobs.update(
        {f"archive-{i}": Path(path) for i, path in enumerate(manifest["baseline"])}
    )
    write_json(
        destination / "manifest.json",
        {
            "purpose": "confirm corrected versus historical models without further selection",
            "models": {name: str(path) for name, path in jobs.items()},
            "seed_start": CONFIRMATION_START,
            "deals": CONFIRMATION_DEALS,
            "seats": [0, 1],
            "opponent": "minimax",
        },
    )
    reports: dict[str, Any] = {}
    with ProcessPoolExecutor(
        max_workers=5, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        pending = {
            executor.submit(confirm_job, path): name for name, path in jobs.items()
        }
        for future in as_completed(pending):
            name = pending[future]
            reports[name] = future.result()
            print(
                f"CONFIRM {name}: {reports[name]['wins']}/{reports[name]['games']}",
                flush=True,
            )
            write_json(destination / "results.json", reports)
    write_json(destination / "status.json", {"status": "completed"})
    if "archive-1" in reports:
        write_json(destination / "comparison.json", paired_difference(reports))


if __name__ == "__main__":
    main()
