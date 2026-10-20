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
from safe_on_clean_text.analyses.workload_matched import thresholds
from safe_on_clean_text.conformal.scores import RISK_CLASS
from safe_on_clean_text.conformal.thresholds import label_shift_thresholds
from safe_on_clean_text.degradations import make_stream
from safe_on_clean_text.evaluation.aggregation import aggregate
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS

# H8 (criteria fixed before running): calibration at 10% risk. In each resplit the risk rows of the calibration half
# are thinned to round(0.10 * n_1), drawn without replacement once per resplit (stream seeded by the base seed and the
# resplit, shared by all models); the test half is unchanged. M0-LS targets the test half's risk share.
RISK_FRACTION = 0.1
METHODS = ["M0", "M1", "M0-LS"]


def thinned_calibration(calibration: np.ndarray, labels: np.ndarray, base_seed: int, replicate: int) -> np.ndarray:
    risk = calibration[labels[calibration] == RISK_CLASS]
    kept = make_stream(base_seed, replicate, "calibration_risk_thinning").choice(
        risk, size=int(round(RISK_FRACTION * len(risk))), replace=False
    )
    return np.sort(np.concatenate([calibration[labels[calibration] != RISK_CLASS], kept]))


def run(task: dict[str, Any]) -> pd.DataFrame:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    labels = run_scores.labels
    rows = []
    for replicate, (calibration, test) in enumerate(task["replicates"]):
        calibration = thinned_calibration(calibration, labels, task["base_seed"], replicate)
        clean_calibration, calibration_labels, test_labels = (
            run_scores.scores["clean"][calibration],
            labels[calibration],
            labels[test],
        )
        q = {"M0": thresholds(clean_calibration, calibration_labels, "M0", ALPHA)}
        q["M1"] = thresholds(clean_calibration, calibration_labels, "M1", ALPHA)
        q["M0-LS"] = np.array(
            label_shift_thresholds(
                true_scores(clean_calibration, calibration_labels),
                calibration_labels,
                float(np.mean(test_labels == RISK_CLASS)),
                ALPHA,
            )
        )
        risk_m0, risk_m1 = q["M0"][RISK_CLASS], q["M1"][RISK_CLASS]
        in_sample_c1 = operating_point(clean_calibration, calibration_labels, q["M0"])["c1"]
        keys = {
            "model": task["model"],
            "seed": run_scores.seed,
            "replicate": replicate,
            "n_calibration_1": int((calibration_labels == RISK_CLASS).sum()),
            "n_calibration_0": int((calibration_labels != RISK_CLASS).sum()),
            "q_M0": risk_m0,
            "q_M1_risk": risk_m1,
            "q_M1_below_q_M0": float(risk_m1 < risk_m0),
            "q_M1_equal_q_M0": float(risk_m1 == risk_m0),
            "q_M1_above_q_M0": float(risk_m1 > risk_m0),
            "calibration_slack_c1_M0": in_sample_c1 - (1 - ALPHA),
        }
        for condition in task["conditions"]:
            test_scores = run_scores.scores[condition][test]
            points = {method: operating_point(test_scores, test_labels, q[method]) for method in METHODS}
            rows.append(
                {
                    **keys,
                    "condition": condition,
                    **{f"c1_{method}": points[method]["c1"] for method in METHODS},
                    **{f"workload_{method}": points[method]["workload_per_1000"] for method in METHODS},
                    "c1_difference_M1_minus_M0": points["M1"]["c1"] - points["M0"]["c1"],
                }
            )
    return pd.DataFrame(rows)


def low_prevalence_calibration_table(config: dict) -> pd.DataFrame:
    """
    H8 on Dreaddit (all 7 models, 1,000 resplits, calibration risk rows thinned to 10%): per model and condition, c1 and
    workload of M0, M1 and M0-LS, the paired c1(M1) - c1(M0), the shares of (seed, resplit) pairs with
    q_M1 <, =, > q_M0 (column mean) and the calibration-half slack c1(M0) - 0.90. Primary aggregation.
    """

    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        rows = pd.concat(executor.map(run, run_tasks(config, DREADDIT_MODELS)), ignore_index=True)
    metrics = [c for c in rows.columns if c not in ["model", "seed", "replicate", "condition"]]
    return in_study_order(aggregate(rows, ["model", "condition"], metrics=metrics), config["conditions"])
