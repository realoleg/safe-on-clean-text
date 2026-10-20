from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from safe_on_clean_text.analyses.resplit_scores import (
    in_study_order,
    load_run_scores,
    operating_point,
    run_tasks,
    set_metrics,
)
from safe_on_clean_text.analyses.workload_matched import thresholds
from safe_on_clean_text.conformal.scores import RISK_CLASS
from safe_on_clean_text.conformal.thresholds import uncorrected_threshold
from safe_on_clean_text.evaluation.aggregation import aggregate
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS

# H9 (criteria fixed before running): oracle = risk-only rule calibrated on the degraded calibration half of the same
# condition: flag when s_1 <= t, t = ceil(0.9 n_1)-th smallest risk score (the smallest threshold with in-sample
# c1 >= 0.90, uncorrected as in B1), dismiss otherwise. Ratios use workloads averaged over seeds within each resplit.
METHODS = ["oracle", "M1", "M3"]


def run(task: dict[str, Any]) -> pd.DataFrame:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    labels = run_scores.labels
    rows = []
    for replicate, (calibration, test) in enumerate(task["replicates"]):
        calibration_labels, test_labels = labels[calibration], labels[test]
        m1 = thresholds(run_scores.scores["clean"][calibration], calibration_labels, "M1", ALPHA)
        for condition in task["conditions"]:
            matched, test_scores = run_scores.scores[condition][calibration], run_scores.scores[condition][test]
            t = uncorrected_threshold(matched[calibration_labels == RISK_CLASS, RISK_CLASS], ALPHA)
            # Risk-only sets: {risk} when flagged, {not risk} otherwise (as M5 / B1).
            flag = test_scores[:, RISK_CLASS] <= t
            sets = np.zeros((len(test), 2), dtype=bool)
            sets[:, RISK_CLASS], sets[:, 1 - RISK_CLASS] = flag, ~flag
            points = {
                "oracle": set_metrics(sets, test_labels),
                "M1": operating_point(test_scores, test_labels, m1),
                "M3": operating_point(test_scores, test_labels, thresholds(matched, calibration_labels, "M3", ALPHA)),
            }
            rows.append(
                {
                    "model": task["model"],
                    "seed": run_scores.seed,
                    "replicate": replicate,
                    "condition": condition,
                    "oracle_risk_in_set_if_p_at_least": float(expit(-t)),
                    **{f"c1_{method}": points[method]["c1"] for method in METHODS},
                    **{f"workload_{method}": points[method]["workload_per_1000"] for method in METHODS},
                }
            )
    return pd.DataFrame(rows)


def oracle_cost_rows(config: dict) -> pd.DataFrame:
    """
    Per (model, seed, resplit, condition) rows of the oracle, M1 and M3 on Dreaddit (all 7 models, 1,000 resplits).
    """

    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        return pd.concat(executor.map(run, run_tasks(config, DREADDIT_MODELS)), ignore_index=True)


def oracle_cost_table(rows: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """
    H9: per model and condition, c1 and workload of the oracle, M1 and M3 (rows of oracle_cost_rows), and the ratios
    M3 / oracle and M1 / oracle of the seed-averaged workloads per resplit. Primary aggregation.
    """

    metrics = [c for c in rows.columns if c not in ["model", "seed", "replicate", "condition"]]
    table = aggregate(rows, ["model", "condition"], metrics=metrics)

    per_replicate = rows.groupby(["model", "condition", "replicate"])[[f"workload_{m}" for m in METHODS]].mean()
    ratios = pd.DataFrame(
        {
            f"workload_ratio_{method}_over_oracle": per_replicate[f"workload_{method}"]
            / per_replicate["workload_oracle"]
            for method in ["M3", "M1"]
        }
    ).reset_index()
    ratio_table = aggregate(ratios.assign(seed="seed-averaged"), ["model", "condition"])
    return in_study_order(pd.concat([table, ratio_table], ignore_index=True), conditions)


def m3_cost_ratio_table(rows: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """
    DESCRIPTIVE: M3 workload per 1,000 in each condition divided by M1's clean workload, and by M1's workload in the
    same condition (alpha 0.10, pi = 0.10; rows of oracle_cost_rows), as ratios of the seed-averaged workloads per
    resplit; median and 2.5-97.5% over resplits per model and condition.
    """

    per_replicate = rows.groupby(["model", "condition", "replicate"])[["workload_M3", "workload_M1"]].mean()
    m1_clean = per_replicate.xs("clean", level="condition")["workload_M1"]
    m1_clean = m1_clean.reindex(per_replicate.index.droplevel("condition")).to_numpy()
    ratios = pd.DataFrame(
        {
            "workload_ratio_M3_over_M1_clean": per_replicate["workload_M3"] / m1_clean,
            "workload_ratio_M3_over_M1_same_condition": per_replicate["workload_M3"] / per_replicate["workload_M1"],
        }
    ).reset_index()
    return in_study_order(aggregate(ratios.assign(seed="seed-averaged"), ["model", "condition"]), conditions)
