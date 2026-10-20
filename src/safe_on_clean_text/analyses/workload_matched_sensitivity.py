from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.analyses.resplit_scores import in_study_order, load_run_scores, operating_point, run_tasks
from safe_on_clean_text.analyses.workload_matched import ALPHA_GRID, thresholds
from safe_on_clean_text.analyses.zero_shot_ranking import ZERO_SHOT
from safe_on_clean_text.evaluation.aggregation import aggregate
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS

# DESCRIPTIVE sensitivity of H7 (the H7 verdict stays as fixed). Exact workload matching on the test half's clean
# workload by linear interpolation along an alpha grid: (a) trained models, M1's c1 and workload interpolated over the
# H7 grid (0.02-0.20) to M0's clean workload at alpha = 0.10; (b) zero-shot models, M0's alpha raised (grid 0.10-0.99)
# until its clean workload equals M1's at alpha = 0.10. Of several crossings, the one with alpha nearest to 0.10 is used
# (the smaller alpha on a tie); resplits without a crossing are unmatched and left out. Variant match_on="calibration"
# (trained models): the crossing is found on the calibration half's in-sample clean workloads, as in H7, and the
# test-half c1 and workload are interpolated at that alpha.
M0_GRID = [round(0.10 + 0.01 * i, 2) for i in range(90)]


def crossing(workloads: np.ndarray, target: float, grid: list[float]) -> tuple[int, float] | None:
    """
    (i, t) with (1 - t) * workloads[i] + t * workloads[i + 1] = target, the crossing nearest to alpha 0.10; None if
    the target is never reached.
    """

    found = []
    for i in range(len(grid) - 1):
        low, high = workloads[i], workloads[i + 1]
        if min(low, high) <= target <= max(low, high):
            t = 0.0 if high == low else (target - low) / (high - low)
            alpha = (1 - t) * grid[i] + t * grid[i + 1]
            found.append((round(abs(alpha - ALPHA), 12), alpha, i, t))  # rounded: exact ties -> smaller alpha
    return min(found)[2:] if found else None


def run(task: dict[str, Any]) -> pd.DataFrame:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    labels = run_scores.labels
    zero_shot = task["model"] in ZERO_SHOT
    moved, fixed = ("M0", "M1") if zero_shot else ("M1", "M0")
    grid = M0_GRID if zero_shot else ALPHA_GRID
    rows = []
    for replicate, (calibration, test) in enumerate(task["replicates"]):
        clean_calibration, calibration_labels, test_labels = (
            run_scores.scores["clean"][calibration],
            labels[calibration],
            labels[test],
        )
        fixed_q = thresholds(clean_calibration, calibration_labels, fixed, ALPHA)
        moved_q = [thresholds(clean_calibration, calibration_labels, moved, alpha) for alpha in grid]
        # Matching on the test half's clean workload, or on the calibration half's in-sample clean workload.
        clean, clean_labels = (
            (clean_calibration, calibration_labels)
            if task.get("match_on", "test") == "calibration"
            else (run_scores.scores["clean"][test], test_labels)
        )
        target = operating_point(clean, clean_labels, fixed_q)["workload_per_1000"]
        clean_curve = np.array([operating_point(clean, clean_labels, q)["workload_per_1000"] for q in moved_q])
        found = crossing(clean_curve, target, grid)
        rows.append(
            {
                "model": task["model"],
                "seed": run_scores.seed,
                "replicate": replicate,
                "condition": "all",
                "unmatched": float(found is None),
                "target_clean_workload": target,
                "moved_min_clean_workload": float(clean_curve.min()),
                "moved_max_clean_workload": float(clean_curve.max()),
            }
        )
        if found is None:
            continue
        i, t = found
        for condition in task["conditions"]:
            test_scores = run_scores.scores[condition][test]
            at_fixed = operating_point(test_scores, test_labels, fixed_q)
            low, high = (operating_point(test_scores, test_labels, moved_q[j]) for j in [i, i + 1])
            c1 = {fixed: at_fixed["c1"], moved: (1 - t) * low["c1"] + t * high["c1"]}
            workload_moved = (1 - t) * low["workload_per_1000"] + t * high["workload_per_1000"]
            rows.append(
                {
                    "model": task["model"],
                    "seed": run_scores.seed,
                    "replicate": replicate,
                    "condition": condition,
                    "moved_method": moved,
                    "alpha_star": (1 - t) * grid[i] + t * grid[i + 1],
                    "c1_M0": c1["M0"],
                    "c1_M1": c1["M1"],
                    "c1_difference_M1_minus_M0": c1["M1"] - c1["M0"],
                    "workload_gap_M1_minus_M0": (workload_moved - at_fixed["workload_per_1000"])
                    * (1 if moved == "M1" else -1),
                }
            )
    return pd.DataFrame(rows)


def workload_matched_sensitivity_table(
    config: dict, models: list[str] = DREADDIT_MODELS, match_on: str = "test"
) -> pd.DataFrame:
    """
    Per model and condition: the interpolated alpha* of the moved method, c1 of M0 and M1 at exactly matched clean
    workload (on the test half, or on the calibration half for match_on="calibration"), their paired difference and the
    remaining test-half workload gap (0 on clean when matched on the test half); rows with condition "all": the share of
    unmatched (seed, resplit) pairs, the target clean workload and the lowest / highest clean workload the moved method
    reaches on its grid (on the matching half). Primary aggregation.
    """

    tasks = [{**task, "match_on": match_on} for task in run_tasks(config, models)]
    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        rows = pd.concat(executor.map(run, tasks), ignore_index=True)
    per_run = rows[rows["condition"] == "all"].dropna(axis=1, how="all")
    matched = rows[rows["condition"] != "all"].dropna(axis=1, how="all")
    moved = matched[["model", "moved_method"]].drop_duplicates()
    table = pd.concat(
        [
            aggregate(per_run, ["model", "condition"]),
            aggregate(matched.drop(columns="moved_method"), ["model", "condition"]),
        ],
        ignore_index=True,
    )
    table = table.merge(moved, on="model", how="left")
    table["moved_method"] = table["moved_method"].fillna(table["model"].map(lambda m: "M0" if m in ZERO_SHOT else "M1"))
    return in_study_order(table, ["all", *config["conditions"]])
