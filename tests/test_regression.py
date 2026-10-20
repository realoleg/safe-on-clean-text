from typing import Any

import numpy as np
import pandas as pd
from conftest import REPO

from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.conformal.metrics import triage_frame, summarize_indicators
from safe_on_clean_text.conformal.thresholds import conformal_rank, conformal_threshold

# Regression on the DistilBERT predictions of the original code (calibration 454 rows, test 715 rows), in two parts
# replacing a plain "same numbers as the old code" check:
#   (a) new M0 with the conformal rule (k-th smallest score, k = ceil((n + 1)(1 - alpha))) -> frozen new constants;
#   (b) the same set construction with the (k + 1)-th score reproduces the old code's output exactly.
# The old code used np.quantile(scores, k / n, method="higher"), which returns the (k + 1)-th score
# whenever k < n (one rank too conservative); that rank is the only difference between old and new code.
# Fixture: the calibration and test rows of results/predictions/transformer_predictions.csv of the original
# code, reduced to id, split, label and the two probabilities (values copied as written, no text).
REGRESSION_FIXTURE = REPO / "tests/fixtures/regression_569d312.csv"


REGRESSION_NEW_M0 = {
    0.1: {
        "threshold": 0.93093027,
        "coverage": 0.8825174825174825,
        "set_size": 1.2503496503496503,
        "singleton": 0.7496503496503496,
        "empty": 0.0,
        "full": 0.25034965034965034,
    },
    0.05: {
        "threshold": 0.97843152,
        "coverage": 0.9482517482517483,
        "set_size": 1.5188811188811189,
        "singleton": 0.4811188811188811,
        "empty": 0.0,
        "full": 0.5188811188811189,
    },
}


# Old code output (results/tables/conformal_metrics.csv), reproduced with the old code before it was deleted.
REGRESSION_OLD_CODE = {
    0.1: {
        "threshold": 0.945091568,
        "coverage": 0.8923076923076924,
        "set_size": 1.283916083916084,
        "singleton": 0.7160839160839161,
        "empty": 0.0,
        "full": 0.2839160839160839,
    },
    0.05: {
        "threshold": 0.978802262,
        "coverage": 0.9482517482517483,
        "set_size": 1.5216783216783216,
        "singleton": 0.4783216783216783,
        "empty": 0.0,
        "full": 0.5216783216783217,
    },
}


REGRESSION_TOLERANCE = 1e-12


def check_regression() -> list[tuple[str, Any]]:
    """
    Two-part regression on the DistilBERT predictions of the original code:
    (a) new M0 (k-th score) matches the frozen new constants and its threshold is the k-th score;
    (b) the (k + 1)-th score with the same set construction reproduces the old code's output.
    """

    old = pd.read_csv(REGRESSION_FIXTURE)

    def frame(split_name: str) -> pd.DataFrame:
        split = old[old["split"] == split_name]
        return pd.DataFrame(
            {
                "label": split["label"].astype(int).to_numpy(),
                "score_0": 1.0 - split["prob_not_stress"].to_numpy(dtype=np.float64),
                "score_1": 1.0 - split["prob_stress"].to_numpy(dtype=np.float64),
            }
        )

    calibration, test = frame("calibration"), frame("test")
    labels = test["label"].to_numpy()
    rows: list[tuple[str, Any]] = [("n_calibration", len(calibration)), ("n_test", len(test))]
    all_ok = len(calibration) == 454 and len(test) == 715

    scores = np.sort(np.where(calibration["label"] == 1, calibration["score_1"], calibration["score_0"]))
    test_scores = test[["score_0", "score_1"]].to_numpy()

    def observed(sets: np.ndarray, threshold: float) -> dict[str, float]:
        summary = summarize_indicators(triage_frame(sets, labels), labels)
        return {
            "threshold": threshold,
            **{name: summary[name] for name in ["coverage", "set_size", "singleton", "empty", "full"]},
        }

    for alpha in REGRESSION_NEW_M0:
        k = conformal_rank(len(scores), alpha)

        # (a) New M0 through the library path; its threshold must be the k-th score.
        sets, _ = predict_sets(calibration, test, "M0", alpha)
        threshold = conformal_threshold(scores, alpha)
        new_ok = threshold == scores[k - 1] and np.array_equal(sets, test_scores <= threshold)
        rows.append((f"a.alpha={alpha}.threshold_is_kth_score_ok", bool(new_ok)))
        all_ok &= new_ok

        # (b) Same set construction with the (k + 1)-th score, as the old code did.
        old_threshold = float(scores[k])
        parts = {
            "a": (REGRESSION_NEW_M0[alpha], observed(sets, threshold)),
            "b": (REGRESSION_OLD_CODE[alpha], observed(test_scores <= old_threshold, old_threshold)),
        }
        for part, (expected, values) in parts.items():
            for name, value in values.items():
                ok = abs(value - expected[name]) <= REGRESSION_TOLERANCE
                all_ok &= ok
                rows.append((f"{part}.alpha={alpha}.{name}", value))
                rows.append((f"{part}.alpha={alpha}.{name}_ok", ok))

    rows.append(("ok", all_ok))
    return rows


def test_two_part_regression_against_the_original_code():
    """
    The original code used np.quantile(scores, k / n, method="higher"), i.e. the (k + 1)-th smallest score whenever
    k < n. (a) The k-th score reproduces the frozen M0 numbers; (b) the (k + 1)-th score reproduces the original output.
    """

    rows = dict(check_regression())
    assert rows["ok"]
