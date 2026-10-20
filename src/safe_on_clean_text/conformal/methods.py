from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from safe_on_clean_text.conformal.metrics import group_codes
from safe_on_clean_text.conformal.scores import RISK_CLASS, true_class_scores
from safe_on_clean_text.conformal.thresholds import conformal_threshold, uncorrected_threshold


@dataclass(frozen=True)
class Method:
    """
    A split-conformal method: where it calibrates and how its thresholds are grouped.
    calibration: "clean" (clean calibration), "matched" (calibration in the test condition) or "mix" (each calibration
    example in its mix-assigned condition); by_class: one threshold per candidate class; risk_only: risk-class flag only.
    """

    name: str
    description: str
    calibration: str
    group_columns: tuple[str, ...] = ()
    by_class: bool = True
    risk_only: bool = False


METHODS: dict[str, Method] = {
    "M0": Method("marginal", "One threshold over all calibration examples.", "clean", by_class=False),
    "M1": Method("class-conditional", "One threshold per class (Mondrian).", "clean"),
    "M2": Method("domain × class", "One threshold per (domain, class).", "clean", ("domain",)),
    "M3": Method("matched degradation", "Class-conditional, calibrated on the test condition.", "matched"),
    "M3-mix": Method(
        "degradation mixture",
        "Per (length bucket, class) on a random mix of the degradations.",
        "mix",
        ("length_bucket",),
    ),
    "M5": Method(
        "risk-only, one-sided", "Flag when the risk class passes its class threshold.", "clean", risk_only=True
    ),
}

# Display label of every method in figures, tables, captions and the digest: "<code> (<name>)". M0-LS (label-shift M0,
# analyses/prevalence.py) and the baselines B1, B2 (evaluation/runs.py) and B3 (evaluation/pipeline.py) are not split
# conformal methods of the registry above.
METHOD_LABELS: dict[str, str] = {
    **{code: f"{code} ({method.name})" for code, method in METHODS.items()},
    "M0-LS": "M0-LS (marginal, label-shift weighted)",
    "B1": "B1 (validation threshold)",
    "B2": "B2 (temperature scaling + abstention)",
    "B3": "B3 (seed ensemble)",
}


def jitter_seed(base_seed: int, resplit_id: int | str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{resplit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def predict_sets(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    alpha: float,
    jitter: np.random.Generator | None = None,
    jitter_scale: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """
    Prediction sets (n_test x 2 boolean) of one method; class y is included if s(x, y) <= threshold of its group.
    Groups without calibration examples get +inf. Returns the sets and the number of +inf thresholds used.
    Optional jitter (appendix tie-breaking): U(0, jitter_scale) added to every calibration and test score.
    """

    spec = METHODS[method]

    calibration_scores = true_class_scores(calibration)
    test_scores = test[["score_0", "score_1"]].to_numpy(dtype=np.float64)
    if jitter is not None:
        calibration_scores = calibration_scores + jitter.uniform(0, jitter_scale, size=calibration_scores.shape)
        test_scores = test_scores + jitter.uniform(0, jitter_scale, size=test_scores.shape)

    calibration_groups, test_groups, names = group_codes(calibration, test, list(spec.group_columns))
    n_classes = 2 if spec.by_class else 1
    calibration_keys = calibration_groups * n_classes
    if spec.by_class:
        calibration_keys = calibration_keys + calibration["label"].to_numpy()

    thresholds = np.array(
        [
            conformal_threshold(calibration_scores[calibration_keys == key], alpha)
            for key in range(len(names) * n_classes)
        ]
    )

    sets = np.zeros((len(test), 2), dtype=bool)
    infinite_keys: set[int] = set()
    candidate_classes = [RISK_CLASS] if spec.risk_only else [0, 1]

    for candidate in candidate_classes:
        test_keys = test_groups * n_classes + (candidate if spec.by_class else 0)
        used = np.unique(test_keys)
        infinite_keys |= set(used[np.isinf(thresholds[used])].tolist())
        sets[:, candidate] = test_scores[:, candidate] <= thresholds[test_keys]

    if spec.risk_only:
        # M5 screening: flag -> {risk}; otherwise -> {not risk}.
        sets[:, 1 - RISK_CLASS] = ~sets[:, RISK_CLASS]

    return sets, len(infinite_keys)


def uncorrected_sets(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    jitter: np.random.Generator | None = None,
    jitter_scale: float = 1e-9,
) -> np.ndarray:
    """
    B1: threshold on the risk score from validation risk cases (no correction); flag -> {risk}, else {not risk}.
    Optional jitter (appendix tie-breaking): U(0, jitter_scale) added to the validation and test risk scores.
    """

    risk_scores = validation.loc[validation["label"] == RISK_CLASS, f"score_{RISK_CLASS}"].to_numpy(dtype=np.float64)
    test_scores = test[f"score_{RISK_CLASS}"].to_numpy(dtype=np.float64)
    if jitter is not None:
        risk_scores = risk_scores + jitter.uniform(0, jitter_scale, size=risk_scores.shape)
        test_scores = test_scores + jitter.uniform(0, jitter_scale, size=test_scores.shape)
    threshold = uncorrected_threshold(risk_scores, alpha)

    sets = np.zeros((len(test), 2), dtype=bool)
    sets[:, RISK_CLASS] = test_scores <= threshold
    sets[:, 1 - RISK_CLASS] = ~sets[:, RISK_CLASS]
    return sets


def temperature_refusal_sets(margins: np.ndarray, temperature: float, n_refuse: int) -> np.ndarray:
    """
    B2: temperature-scaled probabilities; the n_refuse cases with the lowest max-probability are refused
    (full set -> refer), the others get {argmax} (ties at 0.5 -> class 0, as elsewhere).
    """

    prob_risk = expit(margins / temperature)
    confidence = np.maximum(prob_risk, 1.0 - prob_risk)
    refused = np.argsort(confidence, kind="stable")[:n_refuse]

    sets = np.zeros((len(margins), 2), dtype=bool)
    predicted = (prob_risk > 0.5).astype(int)
    sets[np.arange(len(margins)), predicted] = True
    sets[refused] = True
    return sets


def calibration_frame(
    method: str,
    calibration_by_condition: dict[str, pd.DataFrame],
    mix_calibration: pd.DataFrame,
    test_condition: str,
) -> pd.DataFrame:
    """
    Calibration rows a method uses for a test condition.
    """

    source = METHODS[method].calibration
    if source == "clean":
        return calibration_by_condition["clean"]
    if source == "matched":
        return calibration_by_condition[test_condition]
    return mix_calibration
