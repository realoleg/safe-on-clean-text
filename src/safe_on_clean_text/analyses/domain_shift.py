from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import betabinom
from sklearn.metrics import f1_score

from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.conformal.metrics import reference_band
from safe_on_clean_text.conformal.scores import score_frame
from safe_on_clean_text.degradations import make_stream
from safe_on_clean_text.models.predictions import predicted_labels
from safe_on_clean_text.provenance import timestamp

METRICS = ["coverage", "c0", "c1"]


def held_out_frames(
    predictions: pd.DataFrame, metadata: pd.DataFrame, domain: str, from_main_run: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean calibration (the other four domains) and test (the held-out domain's official test rows) frames.
    LODO runs already hold exactly these rows; main runs (LLMs) are filtered by domain.
    """

    clean = predictions[predictions["condition"] == "clean"]
    calibration = score_frame(clean[clean["split"] == "calibration"], metadata)
    test = score_frame(clean[clean["split"] == "test"], metadata)
    if from_main_run:
        calibration = calibration[calibration["domain"] != domain].reset_index(drop=True)
        test = test[test["domain"] == domain].reset_index(drop=True)
    assert (calibration["domain"] != domain).all() and (test["domain"] == domain).all()
    return calibration, test


def class_metrics(sets: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    covered = sets[np.arange(len(labels)), labels]
    return {"coverage": covered.mean(), "c0": covered[labels == 0].mean(), "c1": covered[labels == 1].mean()}


def below_band_probability(n_calibration: int, m_test: int, alpha: float) -> float:
    """
    P(coverage < lower band edge) for one domain when the guarantee holds (BetaBinomial(m, n + 1 - l, l) / m).
    """

    ell = math.floor((n_calibration + 1) * Fraction(str(alpha)))
    distribution = betabinom(m_test, n_calibration + 1 - ell, ell)
    low = distribution.ppf(0.025)
    return float(distribution.cdf(low - 1))


def coverage_tail_probability(observed: float, n_calibration: int, m_test: int, alpha: float) -> float:
    """
    P(coverage <= observed) under the reference BetaBinomial(m, n + 1 - l, l) / m (the guarantee holding).
    """

    ell = math.floor((n_calibration + 1) * Fraction(str(alpha)))
    return float(betabinom(m_test, n_calibration + 1 - ell, ell).cdf(math.floor(observed * m_test + 1e-9)))


def leave_one_domain_out(config: dict) -> None:
    """
    Leave-one-domain-out: M0 / M1 coverage of each held-out domain over calibration bootstraps for the LODO encoder
    runs and the LLM S-tok runs -> lodo_long.parquet, lodo_summary.csv (with reference bands) and lodo_report.csv.
    """

    lodo = config["lodo"]
    base_seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    runs_root = Path(config["paths"]["runs_dir"])
    output_dir = Path(config["evaluation"]["output_dir"])
    alphas = [float(a) for a in config["conformal"]["alpha_values"]]
    main_alpha = alphas[0]

    metadata = pd.concat([pd.read_csv(data_dir / f"{name}.csv") for name in ["calibration", "test"]])
    posts = metadata.set_index("id")["post_id"]

    runs = [(lodo["model"], str(seed), False) for seed in lodo["seeds"]] + [
        (name, "P1", True) for name in lodo["llm_runs"]
    ]
    rows: list[dict[str, Any]] = []
    report: list[tuple[str, str, str, Any]] = []
    for domain in lodo["domains"]:
        for model, seed, from_main_run in runs:
            if from_main_run:
                path = runs_root / dataset / model / f"seed_{seed}" / "predictions.csv.gz"
            else:
                path = runs_root / f"{dataset}_lodo" / domain / model / f"seed_{seed}" / "predictions.csv.gz"
            calibration, test = held_out_frames(pd.read_csv(path), metadata, domain, from_main_run)
            labels = test["label"].to_numpy()
            calibration_posts = posts.loc[calibration["id"]].to_numpy()
            positions_by_post = pd.Series(np.arange(len(calibration))).groupby(calibration_posts).apply(np.asarray)
            post_ids = positions_by_post.index.to_numpy()

            for b in range(int(lodo["n_bootstraps"])):
                rng = make_stream(base_seed, b, f"lodo_bootstrap_{domain}")
                drawn = np.concatenate(
                    positions_by_post.loc[rng.choice(post_ids, size=len(post_ids), replace=True)].to_numpy()
                )
                calibration_b = calibration.iloc[drawn]
                for alpha in alphas:
                    for method in lodo["methods"]:
                        sets, _ = predict_sets(calibration_b, test, method, alpha)
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "domain": domain,
                                "method": method,
                                "alpha": alpha,
                                "bootstrap": b,
                                **class_metrics(sets, labels),
                            }
                        )

            # Bands at the domain's sizes: the original four-domain calibration (n) and the held-out test (m).
            cal_labels = calibration["label"].to_numpy()
            for metric, n, m in [
                ("coverage", len(cal_labels), len(labels)),
                ("c0", int((cal_labels == 0).sum()), int((labels == 0).sum())),
                ("c1", int((cal_labels == 1).sum()), int((labels == 1).sum())),
            ]:
                for alpha in alphas:
                    low, high = reference_band(n, m, alpha)
                    report.append(
                        ("band", f"{model}/{seed}/{domain}", f"{metric}.alpha={alpha}", f"{low:.4f}|{high:.4f}|{n}|{m}")
                    )
            if not from_main_run:
                # Held-out macro-F1 vs the in-distribution main run of the same seed on the same domain's test rows.
                main = pd.read_csv(runs_root / dataset / model / f"seed_{seed}" / "predictions.csv.gz")
                main_test = main[(main["split"] == "test") & (main["condition"] == "clean")]
                main_test = main_test[main_test["id"].isin(test["id"])]
                report.append(
                    (
                        "macro_f1",
                        f"{model}/{seed}/{domain}",
                        "held_out",
                        round(float(f1_score(labels, predicted_labels(test), average="macro")), 4),
                    )
                )
                report.append(
                    (
                        "macro_f1",
                        f"{model}/{seed}/{domain}",
                        "in_distribution",
                        round(float(f1_score(main_test["label"], predicted_labels(main_test), average="macro")), 4),
                    )
                )
            print(f"[{timestamp()}] LODO {domain}: {model} seed {seed} done", flush=True)

    long = pd.DataFrame(rows)
    # Per-bootstrap rows (gitignored), e.g. for seed-averaged aggregation in the paper tables.
    long.to_parquet(output_dir / "lodo_long.parquet", index=False)
    bands = pd.DataFrame([r for r in report if r[0] == "band"], columns=["section", "scope", "key", "value"])
    split_scope = bands["scope"].str.split("/", expand=True)
    split_key = bands["key"].str.split(".alpha=", expand=True)
    split_value = bands["value"].str.split("|", expand=True)
    bands = pd.DataFrame(
        {
            "model": split_scope[0],
            "seed": split_scope[1],
            "domain": split_scope[2],
            "metric": split_key[0],
            "alpha": split_key[1].astype(float),
            "band_low": split_value[0].astype(float),
            "band_high": split_value[1].astype(float),
            "n_calibration": split_value[2].astype(int),
            "m_test": split_value[3].astype(int),
        }
    )

    # Per (model, seed | all seeds, domain, method, alpha, metric): median and 95% interval over bootstraps, share below band.
    summaries = []
    for seed_scope, frame in [("per_seed", long), ("all", long.assign(seed="all"))]:
        melted = frame.melt(
            id_vars=["model", "seed", "domain", "method", "alpha", "bootstrap"], value_vars=METRICS, var_name="metric"
        )
        band_seed = (
            bands
            if seed_scope == "per_seed"
            else bands.groupby(["model", "domain", "metric", "alpha"], as_index=False).first().assign(seed="all")
        )
        melted = melted.merge(band_seed, on=["model", "seed", "domain", "metric", "alpha"], how="left")
        melted["below_band"] = melted["value"] < melted["band_low"]
        grouped = melted.groupby(["model", "seed", "domain", "method", "alpha", "metric"], sort=True)
        summaries.append(
            pd.concat(
                [
                    grouped["value"].median().rename("median"),
                    grouped["value"].quantile(0.025).rename("q025"),
                    grouped["value"].quantile(0.975).rename("q975"),
                    grouped["below_band"].mean().rename("share_below_band"),
                    grouped["band_low"].first(),
                    grouped["band_high"].first(),
                    grouped["n_calibration"].first(),
                    grouped["m_test"].first(),
                    grouped.size().rename("n_values"),
                ],
                axis=1,
            ).reset_index()
        )
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary[
        ~((summary["seed"] != "all") & summary["model"].isin(lodo["llm_runs"]))
    ]  # LLMs: one prompt, one row
    summary.insert(0, "dataset", dataset)
    summary.to_csv(output_dir / "lodo_summary.csv", index=False)

    # H4: DistilBERT, M0, main alpha, all seeds: is the worst domain's median coverage below its band?
    h4 = summary[
        (summary["model"] == lodo["model"])
        & (summary["seed"] == "all")
        & (summary["method"] == "M0")
        & (summary["alpha"] == main_alpha)
        & (summary["metric"] == "coverage")
    ].set_index("domain")
    worst = h4["median"].idxmin()
    report += [
        ("h4", lodo["model"], "worst_domain", worst),
        ("h4", lodo["model"], "worst_median_coverage", round(float(h4.loc[worst, "median"]), 4)),
        ("h4", lodo["model"], "worst_band_low", round(float(h4.loc[worst, "band_low"]), 4)),
        ("h4", lodo["model"], "confirmed", bool(h4.loc[worst, "median"] < h4.loc[worst, "band_low"])),
    ]
    # Context: rate of "at least one of the 5 domains below its band" if the guarantee held, from the bands.
    probabilities = [
        below_band_probability(int(h4.loc[d, "n_calibration"]), int(h4.loc[d, "m_test"]), main_alpha) for d in h4.index
    ]
    report += [("h4_context", lodo["model"], f"p_below_band.{d}", round(p, 4)) for d, p in zip(h4.index, probabilities)]
    report.append(
        (
            "h4_context",
            lodo["model"],
            "expected_rate_any_of_5_below_band",
            round(1 - float(np.prod([1 - p for p in probabilities])), 4),
        )
    )
    report.append(("h4_context", lodo["model"], "approximation_1_minus_0.975^5", round(1 - 0.975**5, 4)))

    # Descriptive strength: per model, method and domain, P(coverage <= observed median) under the reference
    # BetaBinomial(m, n + 1 - l, l) / m at that domain's m and calibration n; worst-of-5 adjusted 1 - (1 - p_min)^5.
    coverage = summary[
        (summary["seed"] == "all") & (summary["alpha"] == main_alpha) & (summary["metric"] == "coverage")
    ]
    for (model, method), frame in coverage.groupby(["model", "method"], sort=True):
        tail = {}
        for _, row in frame.iterrows():
            tail[row["domain"]] = coverage_tail_probability(
                row["median"], int(row["n_calibration"]), int(row["m_test"]), main_alpha
            )
            report.append(
                (
                    "h4_strength",
                    f"{model}/{method}",
                    f"p_coverage_le_median.{row['domain']}",
                    round(tail[row["domain"]], 6),
                )
            )
        p_min = min(tail.values())
        report.append(("h4_strength", f"{model}/{method}", "worst_domain", min(tail, key=tail.get)))
        report.append(
            ("h4_strength", f"{model}/{method}", "worst_of_5_adjusted", round(1 - (1 - p_min) ** len(tail), 6))
        )
    report.append(("checks", dataset, "domains_computed", int(summary["domain"].nunique())))

    out = pd.DataFrame([r for r in report if r[0] != "band"], columns=["section", "scope", "key", "value"])
    out.to_csv(output_dir / "lodo_report.csv", index=False)
    print(out.to_string(index=False))
