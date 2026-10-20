from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from safe_on_clean_text.conformal.scores import RISK_CLASS


def conformal_rank(n: int, alpha: float) -> int:
    """
    Rank of the conformal threshold: k = ceil((n + 1)(1 - alpha)), computed exactly.
    """

    return math.ceil((n + 1) * (1 - Fraction(str(alpha))))


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """
    k-th smallest calibration score; +inf if k > n (the group is too small, every class is included).
    """

    n = len(scores)
    k = conformal_rank(n, alpha)
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def label_shift_thresholds(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    prevalence: float,
    alpha: float,
) -> tuple[float, float]:
    """
    Exact split CP under label shift with known class weights (Podkopaev & Ramdas, 2021) for the M0 scores:
    w1 = pi / pi_hat, w0 = (1 - pi) / (1 - pi_hat) with pi_hat the calibration risk share; for candidate label y the
    threshold is the weighted (1 - alpha) quantile of the calibration scores with an extra mass w(y) at +inf.
    Returns the thresholds for candidate labels (0, 1); with pi = pi_hat both equal the M0 threshold.
    """

    pi_hat = float(np.mean(calibration_labels == RISK_CLASS))
    class_weight = {RISK_CLASS: prevalence / pi_hat, 1 - RISK_CLASS: (1 - prevalence) / (1 - pi_hat)}
    weights = np.where(calibration_labels == RISK_CLASS, class_weight[RISK_CLASS], class_weight[1 - RISK_CLASS])
    order = np.argsort(calibration_scores, kind="stable")
    sorted_scores = calibration_scores[order]
    cumulative = np.cumsum(weights[order])

    thresholds = []
    for label in [0, 1]:
        total = cumulative[-1] + class_weight[label]
        # First score whose cumulative weight reaches (1 - alpha) of the total (relative tolerance for float sums).
        index = int(np.searchsorted(cumulative, (1 - alpha) * total * (1 - 1e-12), side="left"))
        thresholds.append(float(sorted_scores[index]) if index < len(sorted_scores) else math.inf)
    return thresholds[0], thresholds[1]


def uncorrected_threshold(scores: np.ndarray, alpha: float) -> float:
    """
    B1: k'-th smallest score with k' = ceil(n (1 - alpha)), no finite-sample correction
    (the smallest threshold whose in-sample coverage is at least 1 - alpha).
    """

    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil(n * (1 - Fraction(str(alpha))))
    return float(np.sort(scores)[k - 1])


def weighted_thresholds(
    calibration_scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Weighted split CP under covariate shift (Tibshirani et al., 2019): for a test point with weight w(x), the
    (1 - alpha) quantile of sum_i p_i delta(V_i) + p_x delta(+inf) with p_i = w_i / (sum_j w_j + w(x)), p_x = w(x) / (...).
    Returns one threshold per test point (+inf if the calibration mass never reaches 1 - alpha); with equal weights it
    equals conformal_threshold.
    """

    if len(calibration_scores) == 0:
        return np.full(len(test_weights), math.inf)
    order = np.argsort(calibration_scores, kind="stable")
    sorted_scores = calibration_scores[order]
    cumulative = np.cumsum(calibration_weights[order])
    # First score whose cumulative weight reaches (1 - alpha) of the total (relative tolerance for float sums).
    index = np.searchsorted(cumulative, (1 - alpha) * (cumulative[-1] + test_weights) * (1 - 1e-12), side="left")
    out = np.full(len(test_weights), math.inf)
    finite = index < len(sorted_scores)
    out[finite] = sorted_scores[index[finite]]
    return out
