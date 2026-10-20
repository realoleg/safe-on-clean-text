from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.analyses.resplit_scores import (
    in_study_order,
    load_run_scores,
    operating_point,
    run_tasks,
    true_scores,
)
from safe_on_clean_text.conformal.thresholds import conformal_threshold
from safe_on_clean_text.evaluation.aggregation import aggregate, main_alpha_rows
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS

# H7 (criteria fixed before running): M1 at a workload-matched alpha' against M0 at alpha = 0.10. alpha' is chosen on
# the calibration half of each resplit (per run): the grid value nearest to 0.10 at which M1's in-sample clean workload
# (pi = 0.10) is not lower than M0's at 0.10 (the smaller alpha on a tie; 0.02 and flagged unmatched if none is).
ALPHA_GRID = [round(0.02 + 0.01 * i, 2) for i in range(19)]
CURVE_CONDITIONS = ["clean", "head_25"]
CURVE_METHODS = ["M0", "M1", "M3"]
CHECK_METHODS = ["M0", "M1", "M3"]
IDENTITY_TOLERANCE = 1e-12


def thresholds(scores: np.ndarray, labels: np.ndarray, method: str, alpha: float) -> np.ndarray:
    """
    (q_0, q_1) of M0 (one threshold over all rows) or of M1 / M3 (one per class) from calibration rows.
    """

    true = true_scores(scores, labels)
    if method == "M0":
        q = conformal_threshold(true, alpha)
        return np.array([q, q])
    return np.array([conformal_threshold(true[labels == label], alpha) for label in [0, 1]])


def choose_alpha_prime(workload_m1: dict[float, float], target: float) -> tuple[float, bool]:
    matched = [alpha for alpha in ALPHA_GRID if workload_m1[alpha] >= target]
    if not matched:
        return ALPHA_GRID[0], False
    return min(matched, key=lambda alpha: (abs(alpha - ALPHA), alpha)), True


def run(task: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    labels = run_scores.labels
    main_rows, curve_rows = [], []
    for replicate, (calibration, test) in enumerate(task["replicates"]):
        keys = {"model": task["model"], "seed": run_scores.seed, "replicate": replicate}
        clean_calibration, calibration_labels = run_scores.scores["clean"][calibration], labels[calibration]
        test_labels = labels[test]

        m0 = thresholds(clean_calibration, calibration_labels, "M0", ALPHA)
        m1 = {alpha: thresholds(clean_calibration, calibration_labels, "M1", alpha) for alpha in ALPHA_GRID}
        # In-sample workloads on the calibration half (clean) select alpha'.
        target = operating_point(clean_calibration, calibration_labels, m0)["workload_per_1000"]
        in_sample = {
            alpha: operating_point(clean_calibration, calibration_labels, q)["workload_per_1000"]
            for alpha, q in m1.items()
        }
        alpha_prime, matched = choose_alpha_prime(in_sample, target)

        for condition in task["conditions"]:
            test_scores = run_scores.scores[condition][test]
            at_m0 = operating_point(test_scores, test_labels, m0)
            at_m1 = operating_point(test_scores, test_labels, m1[alpha_prime])
            main_rows.append(
                {
                    **keys,
                    "condition": condition,
                    "alpha_prime": alpha_prime,
                    "unmatched": float(not matched),
                    "calibration_workload_M0": target,
                    "calibration_workload_M1_alpha_prime": in_sample[alpha_prime],
                    "c1_M0": at_m0["c1"],
                    "c1_M1_alpha_prime": at_m1["c1"],
                    "c1_difference": at_m1["c1"] - at_m0["c1"],
                    "workload_M0": at_m0["workload_per_1000"],
                    "workload_M1_alpha_prime": at_m1["workload_per_1000"],
                    "workload_gap": at_m1["workload_per_1000"] - at_m0["workload_per_1000"],
                }
            )

        for condition in CURVE_CONDITIONS:
            test_scores = run_scores.scores[condition][test]
            matched_calibration = run_scores.scores[condition][calibration]
            for alpha in ALPHA_GRID:
                for method in CURVE_METHODS:
                    source = matched_calibration if method == "M3" else clean_calibration
                    q = thresholds(source, calibration_labels, method, alpha)
                    point = operating_point(test_scores, test_labels, q)
                    curve_rows.append(
                        {
                            **keys,
                            "condition": condition,
                            "method": method,
                            "alpha": alpha,
                            "c1": point["c1"],
                            "workload_per_1000": point["workload_per_1000"],
                        }
                    )
    return pd.DataFrame(main_rows), pd.DataFrame(curve_rows)


def check_against_long_tables(config: dict, curves: pd.DataFrame) -> float:
    """
    Largest |difference| between the curve points at alpha = 0.10 and the stored long-table c1 / workload.
    """

    worst = 0.0
    at_alpha = curves[curves["alpha"] == ALPHA].set_index(["seed", "replicate", "condition", "method"])
    for model, part in at_alpha.groupby("model"):
        long = main_alpha_rows(config, model)
        long = long[long["method"].isin(CHECK_METHODS) & long["condition"].isin(CURVE_CONDITIONS)]
        long = long.set_index(["seed", "replicate", "condition", "method"])
        joined = part.join(long[["c1", "workload_per_1000_pi=0.1"]], rsuffix="_stored", how="left")
        assert len(joined) == len(part) and joined["c1_stored"].notna().all(), model
        worst = max(
            worst,
            float((joined["c1"] - joined["c1_stored"]).abs().max()),
            float((joined["workload_per_1000"] - joined["workload_per_1000_pi=0.1"]).abs().max()),
        )
    return worst


def workload_matched_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    H7 on Dreaddit (all 7 models, 1,000 resplits): per model and condition, alpha', the clean workload gap
    M1(alpha') - M0, c1 of M0 and M1(alpha') and their paired difference; and the c1-vs-workload curves of M0, M1 and
    M3 over the alpha grid on clean and head_25. Primary aggregation (seeds averaged within each resplit).
    """

    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        results = list(executor.map(run, run_tasks(config, DREADDIT_MODELS)))
    main = pd.concat([r[0] for r in results], ignore_index=True)
    curves = pd.concat([r[1] for r in results], ignore_index=True)
    identity = check_against_long_tables(config, curves)
    assert identity <= IDENTITY_TOLERANCE, identity

    metrics = [c for c in main.columns if c not in ["model", "seed", "replicate", "condition"]]
    table = aggregate(main, ["model", "condition"], metrics=metrics)
    curve_table = aggregate(curves, ["model", "condition", "method", "alpha"], metrics=["c1", "workload_per_1000"])
    print(f"H7: identity vs long tables at alpha 0.10, max |difference| {identity:.1e}")
    return (
        in_study_order(table, config["conditions"]).assign(identity_max_abs_difference=identity),
        in_study_order(curve_table, config["conditions"]),
    )
