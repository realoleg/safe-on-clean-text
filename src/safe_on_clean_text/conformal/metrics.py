from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from safe_on_clean_text.conformal.scores import RISK_CLASS

# Per-example indicators; every metric below is a mean of one of them (so it can be reweighted by class).
INDICATORS = [
    "covered",
    "set_size",
    "singleton",
    "empty",
    "full",
    "flag",
    "dismiss",
    "workload",
    "uncertainty_referral",
]


@lru_cache(maxsize=None)
def reference_band(n_calibration: int, m_test: int, alpha: float, level: float = 0.95) -> tuple[float, float]:
    """
    Central `level` interval of test coverage when the guarantee holds:
    BetaBinomial(m, n + 1 - l, l) / m with l = floor((n + 1) alpha). If l = 0 the threshold is +inf and coverage is 1.
    """

    if n_calibration == 0 or m_test == 0:
        return math.nan, math.nan

    ell = math.floor((n_calibration + 1) * Fraction(str(alpha)))
    if ell == 0:
        return 1.0, 1.0

    distribution = betabinom(m_test, n_calibration + 1 - ell, ell)
    tail = (1 - level) / 2
    return float(distribution.ppf(tail) / m_test), float(distribution.ppf(1 - tail) / m_test)


GROUP_TYPES = {
    "all": [],
    "class": ["label"],
    "domain": ["domain"],
    "domain_class": ["domain", "label"],
    "agreement": ["agreement"],
}


def group_codes(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint integer codes of the group columns for calibration and test rows, and the sorted group names
    ("|"-joined values; "all" without columns).
    """

    if not columns:
        return np.zeros(len(calibration), dtype=int), np.zeros(len(test), dtype=int), np.array(["all"])

    def keys(frame: pd.DataFrame) -> np.ndarray:
        joined = frame[columns[0]].astype(str)
        for column in columns[1:]:
            joined = joined + "|" + frame[column].astype(str)
        return joined.to_numpy()

    codes, names = pd.factorize(np.concatenate([keys(calibration), keys(test)]), sort=True)
    return codes[: len(calibration)], codes[len(calibration) :], np.asarray(names)


def triage_indicators(sets: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Per-example indicators (columns in INDICATORS order). Triage: {not risk} -> dismiss, {risk} -> flag,
    full or empty -> refer to a human. workload = not dismissed (flag + refer); uncertainty_referral = full or empty set.
    """

    size = sets.sum(axis=1)
    flag = sets[:, RISK_CLASS] & ~sets[:, 1 - RISK_CLASS]
    dismiss = sets[:, 1 - RISK_CLASS] & ~sets[:, RISK_CLASS]
    return np.column_stack(
        [
            sets[np.arange(len(labels)), labels],
            size,
            size == 1,
            size == 0,
            size == 2,
            flag,
            dismiss,
            ~dismiss,
            (size == 0) | (size == 2),
        ]
    ).astype(np.float64)


def triage_frame(sets: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(triage_indicators(sets, labels), columns=INDICATORS)


def summarize_groups(
    matrix: np.ndarray,
    labels: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
) -> list[dict[str, float]]:
    """
    Per group: overall means of the indicators plus class-conditional means (suffix _0 / _1).
    c0, c1 = class coverage; sensitivity = share of risk cases not dismissed. Empty classes give NaN.
    All indicators are 0/1 or set sizes, so the sums are exact and the means do not depend on summation order.
    """

    counts = np.bincount(codes, minlength=n_groups)
    sums = np.column_stack(
        [np.bincount(codes, weights=matrix[:, j], minlength=n_groups) for j in range(matrix.shape[1])]
    )
    class_codes = codes * 2 + labels
    class_counts = np.bincount(class_codes, minlength=2 * n_groups)
    class_sums = np.column_stack(
        [np.bincount(class_codes, weights=matrix[:, j], minlength=2 * n_groups) for j in range(matrix.shape[1])]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]
        class_means = class_sums / class_counts[:, None]

    out = []
    for group in range(n_groups):
        summary: dict[str, float] = {"n": int(counts[group])}
        summary.update({name: float(means[group, j]) for j, name in enumerate(INDICATORS)})
        for label in [0, 1]:
            row = 2 * group + label
            summary[f"n_{label}"] = int(class_counts[row])
            summary.update({f"{name}_{label}": float(class_means[row, j]) for j, name in enumerate(INDICATORS)})

        summary["coverage"] = summary.pop("covered")
        summary["c0"] = summary.pop("covered_0")
        summary["c1"] = summary.pop("covered_1")
        summary["sensitivity"] = summary[f"workload_{RISK_CLASS}"]
        out.append(summary)

    return out


def summarize_indicators(indicators: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """
    Overall and class-conditional means of all examples (see summarize_groups).
    """

    assert list(indicators.columns) == INDICATORS
    return summarize_groups(indicators.to_numpy(), labels, np.zeros(len(labels), dtype=int), 1)[0]


def reweight(summary: dict[str, float], prevalence: float) -> dict[str, float]:
    """
    Metric at risk prevalence pi: pi * metric(risk class) + (1 - pi) * metric(non-risk class).
    Works on a dict of scalars or on a DataFrame of runs (column-wise).
    """

    class_metrics = {"coverage": ("c0", "c1")} | {
        name: (f"{name}_0", f"{name}_1") for name in INDICATORS if name != "covered"
    }
    return {
        name: prevalence * summary[risk] + (1 - prevalence) * summary[non_risk]
        for name, (non_risk, risk) in class_metrics.items()
    }


def group_summaries(
    indicators: pd.DataFrame,
    test: pd.DataFrame,
    calibration: pd.DataFrame,
    alpha: float,
    group_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Metrics per group (all, class, domain, domain x class, agreement bucket) for the groups present in test,
    each with n_calibration, n_test and the reference band for coverage at those sizes.
    """

    assert list(indicators.columns) == INDICATORS
    matrix = indicators.to_numpy()
    labels = test["label"].to_numpy()

    rows = []
    for group_type in group_types or list(GROUP_TYPES):
        calibration_codes, test_codes, names = group_codes(calibration, test, GROUP_TYPES[group_type])
        calibration_sizes = np.bincount(calibration_codes, minlength=len(names))
        summaries = summarize_groups(matrix, labels, test_codes, len(names))

        for group, name in enumerate(names):
            if summaries[group]["n"] == 0:
                continue
            band_low, band_high = reference_band(int(calibration_sizes[group]), summaries[group]["n"], alpha)
            rows.append(
                {
                    "group_type": group_type,
                    "group": str(name),
                    "n_calibration": int(calibration_sizes[group]),
                    "band_low": band_low,
                    "band_high": band_high,
                    **summaries[group],
                }
            )

    return pd.DataFrame(rows)
