from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.methods import uncorrected_sets, predict_sets
from safe_on_clean_text.conformal.metrics import triage_frame, reference_band, reweight, summarize_indicators
from safe_on_clean_text.conformal.scores import score_frame
from safe_on_clean_text.conformal.thresholds import conformal_threshold

SYNTHETIC_REPEATS = 1000
SYNTHETIC_SIZES = (700, 715)
SYNTHETIC_BAND_HIT_RANGE = (0.93, 0.97)

LOG_ODDS_REPEATS = 200
LOG_ODDS_METHODS = ["M0", "M1", "M2", "M5"]
LOG_ODDS_DOMAINS = ["abuse", "social", "anxiety", "ptsd", "financial"]

BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_TOLERANCE = 0.005
BOOTSTRAP_RUN = ("distilbert", 42)
REWEIGHT_METRICS = ["coverage", "workload", "uncertainty_referral", "set_size"]


def synthetic_scores(prob_stress: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": labels.astype(int),
            "score_0": prob_stress.astype(np.float64),
            "score_1": 1.0 - prob_stress.astype(np.float64),
        }
    )


def check_synthetic(alpha: float, rng: np.random.Generator) -> list[tuple[str, Any]]:
    """
    Exchangeable synthetic data (p ~ Beta(2, 2), y ~ Bernoulli(p)), random calibration/test split.
    Share of repeats whose M0 coverage (and M1 c0, c1) falls inside the reference band.
    """

    n_calibration, m_test = SYNTHETIC_SIZES
    hits = {"M0_coverage": 0, "M1_c0": 0, "M1_c1": 0}

    for _ in range(SYNTHETIC_REPEATS):
        prob_stress = rng.beta(2, 2, size=n_calibration + m_test)
        labels = (rng.random(n_calibration + m_test) < prob_stress).astype(int)
        order = rng.permutation(n_calibration + m_test)
        calibration = synthetic_scores(prob_stress[order[:n_calibration]], labels[order[:n_calibration]])
        test = synthetic_scores(prob_stress[order[n_calibration:]], labels[order[n_calibration:]])
        test_labels = test["label"].to_numpy()

        sets, _ = predict_sets(calibration, test, "M0", alpha)
        coverage = sets[np.arange(m_test), test_labels].mean()
        low, high = reference_band(n_calibration, m_test, alpha)
        hits["M0_coverage"] += low <= coverage <= high

        sets, _ = predict_sets(calibration, test, "M1", alpha)
        for label in [0, 1]:
            in_class = test_labels == label
            class_coverage = sets[in_class, label].mean()
            low, high = reference_band(int((calibration["label"] == label).sum()), int(in_class.sum()), alpha)
            hits[f"M1_c{label}"] += low <= class_coverage <= high

    rows: list[tuple[str, Any]] = []
    lower, upper = SYNTHETIC_BAND_HIT_RANGE
    for name, count in hits.items():
        share = count / SYNTHETIC_REPEATS
        rows.append((f"{name}_share_in_band", round(share, 4)))
        rows.append((f"{name}_ok", lower <= share <= upper))
    return rows


def check_log_odds_equivalence(rng: np.random.Generator) -> list[tuple[str, Any]]:
    """
    Log-odds LAC scores (s_y = log p_other - log p_y) give the same sets as the planned 1 - p_y on synthetic data
    without precision issues (p ~ U(0.02, 0.98)), for M0, M1, M2 (domain x class), M5 and B1 at both alphas;
    and with +-inf scores (exact 0 / 1 probabilities) a +inf threshold includes every label, never NaN.
    """

    n_calibration, m_test = SYNTHETIC_SIZES
    n_validation = 400
    compared = changed = 0
    for _ in range(LOG_ODDS_REPEATS):
        n = n_calibration + m_test + n_validation
        prob_stress = rng.uniform(0.02, 0.98, size=n)
        labels = (rng.random(n) < prob_stress).astype(int)
        domains = rng.choice(LOG_ODDS_DOMAINS, size=n)
        margin = np.log(prob_stress) - np.log1p(-prob_stress)
        planned = synthetic_scores(prob_stress, labels).assign(domain=domains)
        log_odds = pd.DataFrame({"label": labels, "score_0": margin, "score_1": -margin, "domain": domains})

        parts = {}
        for name, frame in [("planned", planned), ("log_odds", log_odds)]:
            calibration, test = frame.iloc[:n_calibration], frame.iloc[n_calibration : n_calibration + m_test]
            validation = frame.iloc[n_calibration + m_test :]
            for alpha in [0.1, 0.05]:
                for method in LOG_ODDS_METHODS:
                    parts[(name, method, alpha)] = predict_sets(calibration, test, method, alpha)[0]
                parts[(name, "B1", alpha)] = uncorrected_sets(validation, test, alpha)
        for (name, method, alpha), sets in parts.items():
            if name == "planned":
                difference = (sets != parts[("log_odds", method, alpha)]).any(axis=1)
                compared += len(difference)
                changed += int(difference.sum())

    # +-inf scores: 30% of the calibration true-class scores are +inf, so at alpha 0.10 the threshold is +inf.
    true_scores = np.r_[np.zeros(70), np.full(30, np.inf)]
    calibration = pd.DataFrame({"label": 0, "score_0": true_scores, "score_1": -true_scores})
    test = pd.DataFrame(
        {"label": [0, 1, 0, 1], "score_0": [np.inf, -np.inf, 0.0, 0.0], "score_1": [-np.inf, np.inf, 0.0, 0.0]}
    )
    infinite_sets, n_infinite = predict_sets(calibration, test, "M0", 0.1)

    return [
        ("set_rows_compared", compared),
        ("set_rows_changed", changed),
        ("ok", changed == 0),
        ("infinite_threshold_used", n_infinite),
        ("infinite_threshold_includes_every_label_ok", bool(infinite_sets.all()) and n_infinite == 1),
    ]


def check_infinite_threshold(alpha: float) -> list[tuple[str, Any]]:
    """
    Boundary of the +inf rule (k > n) and its effect through a grouped method (M2-like grouping).
    """

    rows: list[tuple[str, Any]] = []

    # Smallest calibration size with a finite threshold: n >= 1/alpha - 1.
    n_finite = math.ceil(1 / Fraction(str(alpha)) - 1)
    for n, expect_infinite in [(0, True), (n_finite - 1, True), (n_finite, False)]:
        threshold = conformal_threshold(np.linspace(0.0, 0.5, n), alpha)
        rows.append((f"n={n}_infinite", math.isinf(threshold)))
        rows.append((f"n={n}_ok", math.isinf(threshold) == expect_infinite))

    rng = np.random.default_rng(0)
    small, large = n_finite - 1, 200
    calibration = pd.concat(
        [
            synthetic_scores(rng.random(small), np.repeat([0, 1], small // 2 + 1)[:small]).assign(domain="small"),
            synthetic_scores(rng.random(2 * large), np.repeat([0, 1], large)).assign(domain="large"),
        ],
        ignore_index=True,
    )
    test = pd.concat(
        [
            synthetic_scores(rng.random(20), np.repeat([0, 1], 10)).assign(domain="small"),
            synthetic_scores(rng.random(20), np.repeat([0, 1], 10)).assign(domain="unseen"),
            synthetic_scores(rng.random(20), np.repeat([0, 1], 10)).assign(domain="large"),
        ],
        ignore_index=True,
    )

    sets, n_infinite = predict_sets(calibration, test, "M2", alpha)
    full_small_unseen = sets[:40].all()
    rows.append(("grouped_small_and_unseen_groups_full_sets", bool(full_small_unseen)))
    rows.append(("grouped_n_infinite_thresholds", n_infinite))
    rows.append(("grouped_ok", bool(full_small_unseen) and n_infinite == 4))
    return rows


def check_reweighting(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    prevalence_grid: list[float],
    rng: np.random.Generator,
) -> list[tuple[str, Any]]:
    """
    Analytic prevalence reweighting vs a stratified bootstrap of the test set at prevalence pi.
    """

    rows: list[tuple[str, Any]] = []
    labels = test["label"].to_numpy()
    index_by_class = {label: np.flatnonzero(labels == label) for label in [0, 1]}
    max_difference = 0.0

    for method in ["M0", "M1"]:
        sets, _ = predict_sets(calibration, test, method, alpha)
        indicators = triage_frame(sets, labels).rename(columns={"covered": "coverage"})
        summary = summarize_indicators(triage_frame(sets, labels), labels)

        for prevalence in prevalence_grid:
            analytic = reweight(summary, prevalence)
            n_risk = round(len(labels) * prevalence)
            sample = np.concatenate(
                [
                    rng.choice(index_by_class[1], size=(BOOTSTRAP_REPEATS, n_risk)),
                    rng.choice(index_by_class[0], size=(BOOTSTRAP_REPEATS, len(labels) - n_risk)),
                ],
                axis=1,
            )
            for metric in REWEIGHT_METRICS:
                bootstrap = indicators[metric].to_numpy()[sample].mean(axis=1).mean()
                difference = abs(bootstrap - analytic[metric])
                max_difference = max(max_difference, difference)
                rows.append((f"{method}.pi={prevalence}.{metric}.analytic", round(analytic[metric], 6)))
                rows.append((f"{method}.pi={prevalence}.{metric}.bootstrap", round(float(bootstrap), 6)))

    rows.append(("max_abs_difference", round(max_difference, 6)))
    rows.append(("ok", max_difference <= BOOTSTRAP_TOLERANCE))
    return rows


def run_checks(config: dict) -> None:
    """
    Synthetic coverage validity, the +inf threshold rule, prevalence reweighting against a bootstrap and the log-odds
    equivalence of the scores; writes conformal_checks.csv to the checks folder and fails if any check fails.
    """

    output_dir = Path(config["paths"]["checks_dir"])

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conformal_config = config["conformal"]
    main_alpha = float(conformal_config["alpha_values"][0])
    rng = np.random.default_rng(int(config["seed"]))

    metadata = pd.concat([pd.read_csv(data_dir / f"{name}.csv") for name in ["calibration", "test"]])
    model, seed = BOOTSTRAP_RUN
    predictions = pd.read_csv(runs_dir / model / f"seed_{seed}" / "predictions.csv.gz")
    clean = predictions[predictions["condition"] == "clean"]
    calibration = score_frame(clean[clean["split"] == "calibration"], metadata)
    test = score_frame(clean[clean["split"] == "test"], metadata)

    sections: dict[str, list[tuple[str, Any]]] = {}
    for alpha in conformal_config["alpha_values"]:
        sections[f"synthetic_alpha={alpha}"] = check_synthetic(float(alpha), rng)
        sections[f"infinite_threshold_alpha={alpha}"] = check_infinite_threshold(float(alpha))
    sections[f"reweighting_{model}_seed_{seed}_alpha={main_alpha}"] = check_reweighting(
        calibration, test, main_alpha, [float(p) for p in conformal_config["prevalence_grid"]], rng
    )
    # Own generator, so the checks above keep their random streams.
    sections["log_odds_equivalence"] = check_log_odds_equivalence(np.random.default_rng([int(config["seed"]), 1]))

    report = pd.DataFrame(
        [(section, key, value) for section, rows in sections.items() for key, value in rows],
        columns=["section", "key", "value"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "conformal_checks.csv", index=False)

    verdicts = report[report["key"].str.endswith("ok")]
    print(report[~report["key"].str.contains("analytic|bootstrap")].to_string(index=False))
    print(f"\nSaved report to: {(output_dir / 'conformal_checks.csv').resolve()}")

    failed = verdicts[verdicts["value"].astype(str) != "True"]
    if not failed.empty:
        raise RuntimeError(f"Conformal checks failed:\n{failed.to_string(index=False)}")
