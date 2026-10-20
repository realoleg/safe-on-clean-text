from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from safe_on_clean_text.conformal.metrics import reference_band
from safe_on_clean_text.conformal.scores import true_class_scores
from safe_on_clean_text.conformal.thresholds import conformal_threshold
from safe_on_clean_text.degradations import load_condition_texts
from safe_on_clean_text.evaluation.aggregation import (
    aggregate,
    main_alpha_rows,
    metric_columns,
    spearman_correlation,
)
from safe_on_clean_text.evaluation.resplits import load_pool, make_replicates, mix_for_conditions, pool_frames
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS, SDCNL_MODELS
from safe_on_clean_text.text import word_count

# EXPLORATORY, POST HOC (added after the hypothesis tally): descriptive tables built only from
# results that were already computed; no new runs, models or data variants; no criterion is changed.
# Paper aggregation: per resplit the quantity is averaged over a model's seeds, then the median
# and 2.5-97.5% over resplits.
LABEL = "exploratory_post_hoc"
SEEDS = {"tfidf_logreg": ["42"], "distilbert": ["13", "42", "101", "2024", "7777"], "qwen_s_tok_P1": ["P1"]}
MECHANISM_CONDITIONS = ["clean", "head_25", "tail_25"]


def seed_average(values: pd.Series) -> pd.Series:
    return values.groupby(level="replicate").mean()


def addendum(dreaddit: dict, sdcnl: dict) -> None:
    """
    Post hoc exploratory addendum across both datasets (SDCNL conditions, head vs tail truncation, class-coverage
    asymmetry vs the prevalence drop, mechanism by dataset) -> exploratory_addendum.csv in the SDCNL evaluation folder.
    """

    rows: list[dict[str, Any]] = []

    # (1) SDCNL, all 11 conditions: c1 under M0 / M1 / M3-mix, share of resplits below the c1 band, paired M1 - M0.
    # Below-band share = share of resplits whose seed-averaged c1 is below the c1 band low.
    for model in SDCNL_MODELS:
        long = main_alpha_rows(sdcnl, model)
        long = (
            long[long["method"].isin(["M0", "M1", "M3-mix"])]
            .set_index(["condition", "method", "seed", "replicate"])
            .sort_index()
        )
        for condition in sdcnl["conditions"]:
            row = {
                "label": LABEL,
                "section": "sdcnl_conditions",
                "dataset": "sdcnl",
                "model": model,
                "condition": condition,
            }
            for method in ["M0", "M1", "M3-mix"]:
                part = long.loc[(condition, method)]
                low, _ = reference_band(
                    int(round(part["n_calibration_1"].median())), int(round(part["n_1"].median())), ALPHA
                )
                c1 = seed_average(part["c1"])
                row[f"{method}_c1_median"] = float(c1.median())
                row[f"{method}_c1_below_band_share"] = float((c1 < low).mean())
            difference = seed_average(long.loc[(condition, "M1"), "c1"] - long.loc[(condition, "M0"), "c1"])
            row["M1_minus_M0_c1_median"] = difference.median()
            row["M1_minus_M0_c1_q025"], row["M1_minus_M0_c1_q975"] = np.quantile(difference, [0.025, 0.975])
            rows.append(row)

    # (2) head_25 vs tail_25: SDCNL text = title + ". " + selftext, so head_25 starts with the title, tail_25 drops it.
    raw = pd.read_csv(sdcnl["data"]["raw_path"])
    title_words = raw["title"].astype(str).str.split().str.len()
    for scope, names in [
        ("calibration+test", ["calibration", "test"]),
        ("all_splits", ["train", "validation", "calibration", "test"]),
    ]:
        texts = pd.concat([pd.read_csv(Path(sdcnl["data"]["processed_dir"]) / f"{name}.csv") for name in names])
        share = title_words.loc[texts["id"]].to_numpy() / word_count(texts["text"]).to_numpy()
        rows.append(
            {
                "label": LABEL,
                "section": "head_vs_tail",
                "dataset": "sdcnl",
                "scope": scope,
                "metric": "share_title_at_least_25pct_of_words",
                "value": float(np.mean(share >= 0.25)),
                "n": len(texts),
            }
        )
        rows.append(
            {
                "label": LABEL,
                "section": "head_vs_tail",
                "dataset": "sdcnl",
                "scope": scope,
                "metric": "median_title_share_of_words",
                "value": float(np.median(share)),
                "n": len(texts),
            }
        )
    for config in [dreaddit, sdcnl]:
        conditions = load_condition_texts(Path(config["data"]["processed_dir"]) / "conditions.csv")
        for condition in ["clean", "head_25", "tail_25"]:
            words = word_count(conditions[conditions["condition"] == condition]["text"])
            rows.append(
                {
                    "label": LABEL,
                    "section": "head_vs_tail",
                    "dataset": config["data"]["name"],
                    "scope": "calibration+test",
                    "metric": f"median_words_{condition}",
                    "value": float(words.median()),
                    "n": len(words),
                }
            )

    # (3) Class-coverage asymmetry |c1 - c0| (M0, clean; median over resplits) next to the M0 coverage drop at pi = 0.05.
    for config in [dreaddit, sdcnl]:
        models = sorted(
            {
                path.stem.removeprefix(f"{config['data']['name']}_").rsplit("_seed", 1)[0]
                for path in (Path(config["evaluation"]["output_dir"]) / "long").glob("*.parquet")
            }
        )
        for model in models:
            long = main_alpha_rows(config, model)
            long = long[(long["condition"] == "clean") & (long["method"] == "M0")]
            if long.empty:
                continue
            m0 = aggregate(long, ["method"], metrics=metric_columns(long)).set_index("metric")["median"]
            gap = seed_average(
                (long["c1"] - long["c0"]).abs().set_axis(pd.MultiIndex.from_frame(long[["seed", "replicate"]]))
            )
            rows.append(
                {
                    "label": LABEL,
                    "section": "asymmetry_vs_prevalence_drop",
                    "dataset": config["data"]["name"],
                    "model": model,
                    "median_abs_c1_minus_c0": float(gap.median()),
                    "M0_coverage_clean": m0["coverage"],
                    "M0_coverage_pi=0.05": m0["coverage_pi=0.05"],
                    "M0_drop_clean_minus_pi=0.05": m0["coverage"] - m0["coverage_pi=0.05"],
                }
            )

    # (4) The score mechanism analysis on both datasets (SDCNL vs Dreaddit): median p(risk) of risk / non-risk posts (calibration +
    # test, seeds pooled), M0 / M1 thresholds as p cutoffs (median over the resplits, clean calibration) and the share of
    # risk posts above each cutoff. Question: does truncation shift risk-class scores less on SDCNL than on Dreaddit?
    for config in [dreaddit, sdcnl]:
        data_dir = Path(config["data"]["processed_dir"])
        pool = load_pool(data_dir, config["evaluation"]["group_column"])
        replicates = make_replicates(
            pool,
            int(config["evaluation"]["n_replicates"]),
            int(config["evaluation"]["resplit_folds"]),
            int(config["seed"]),
        )["resplit"]
        runs_dir = Path(config["paths"]["runs_dir"]) / config["data"]["name"]
        for model in SDCNL_MODELS:
            predictions = {
                seed: pd.read_csv(runs_dir / model / f"seed_{seed}" / "predictions.csv.gz") for seed in SEEDS[model]
            }
            # Also on the log-odds scale (risk in the set iff log-odds >= -q), where saturated LLM probabilities stay distinct.
            # Per seed one value per resplit; the cutoff is averaged over seeds per resplit, then the median over resplits.
            cutoffs: dict[str, list[list[float]]] = {
                "M0_risk_in_set_if_p_at_least": [],
                "M1_risk_in_set_if_p_at_least": [],
                "M0_risk_in_set_if_logodds_at_least": [],
                "M1_risk_in_set_if_logodds_at_least": [],
            }
            for frame_predictions in predictions.values():
                for values in cutoffs.values():
                    values.append([])
                clean = frame_predictions[frame_predictions["condition"] == "clean"]
                mix = mix_for_conditions(
                    pd.DataFrame({"id": pool["id"], "condition": "clean"}),
                    pool,
                    ["clean"],
                    ["clean"],
                    int(config["seed"]),
                )
                frame = pool_frames(clean, pool, mix, ["clean"])[0]["clean"]
                scores, labels = true_class_scores(frame), frame["label"].to_numpy()
                for calibration_positions, _ in replicates:
                    s_cal, y_cal = scores[calibration_positions], labels[calibration_positions]
                    for method, q in [
                        ("M0", conformal_threshold(s_cal, ALPHA)),
                        ("M1", conformal_threshold(s_cal[y_cal == 1], ALPHA)),
                    ]:
                        cutoffs[f"{method}_risk_in_set_if_p_at_least"][-1].append(float(expit(-q)))
                        cutoffs[f"{method}_risk_in_set_if_logodds_at_least"][-1].append(-q)
            median_cutoff = {name: float(np.median(np.mean(values, axis=0))) for name, values in cutoffs.items()}
            pooled = pd.concat(predictions.values())
            pooled = pooled[pooled["split"].isin(["calibration", "test"])]
            for condition in MECHANISM_CONDITIONS:
                part = pooled[pooled["condition"] == condition]
                risk, non_risk = part[part["label"] == 1]["prob_stress"], part[part["label"] == 0]["prob_stress"]
                logodds = part["logit_stress"] - part["logit_not_stress"]
                rows.append(
                    {
                        "label": LABEL,
                        "section": "mechanism_by_dataset",
                        "dataset": config["data"]["name"],
                        "model": model,
                        "condition": condition,
                        "median_p_risk_posts": float(risk.median()),
                        "median_p_non_risk_posts": float(non_risk.median()),
                        "median_logodds_risk_posts": float(logodds[part["label"] == 1].median()),
                        "median_logodds_non_risk_posts": float(logodds[part["label"] == 0].median()),
                        **median_cutoff,
                        "share_risk_above_M0_cutoff": float(
                            (risk >= median_cutoff["M0_risk_in_set_if_p_at_least"]).mean()
                        ),
                        "share_risk_above_M1_cutoff": float(
                            (risk >= median_cutoff["M1_risk_in_set_if_p_at_least"]).mean()
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    output = Path(sdcnl["evaluation"]["output_dir"]) / "exploratory_addendum.csv"
    table.to_csv(output, index=False)
    print(f"EXPLORATORY, POST HOC -> {output}")
    for section, frame in table.groupby("section", sort=False):
        print(f"\n== {section}")
        print(frame.dropna(axis=1, how="all").drop(columns=["label", "section"]).round(4).to_string(index=False))


def slack_analysis(dreaddit: dict, sdcnl: dict) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    EXPLORATORY, POST HOC (mechanism-derived): per run, x = clean M0 c1 (median over resplits)
    - 0.90 and y = paired head_25 c1(M1) - c1(M0) (median and 2.5-97.5% over resplits); Spearman per dataset and overall.
    """

    rows = []
    for config, models in [(dreaddit, DREADDIT_MODELS), (sdcnl, SDCNL_MODELS)]:
        for model in models:
            long = main_alpha_rows(config, model)
            c1 = long[long["method"].isin(["M0", "M1"]) & long["condition"].isin(["clean", "head_25"])]
            c1 = c1.set_index(["seed", "condition", "method", "replicate"])["c1"].sort_index()
            for seed in sorted(c1.index.get_level_values("seed").unique()):
                difference = c1.loc[(seed, "head_25", "M1")] - c1.loc[(seed, "head_25", "M0")]
                rows.append(
                    {
                        "label": "exploratory_post_hoc",
                        "dataset": config["data"]["name"],
                        "model": model,
                        "seed": seed,
                        "clean_M0_c1_median": float(c1.loc[(seed, "clean", "M0")].median()),
                        "slack": float(c1.loc[(seed, "clean", "M0")].median()) - 0.90,
                        "difference_median": float(difference.median()),
                        "difference_q025": float(np.quantile(difference, 0.025)),
                        "difference_q975": float(np.quantile(difference, 0.975)),
                    }
                )
    points = pd.DataFrame(rows)
    correlations = [
        {
            "label": "exploratory_post_hoc",
            "scope": scope,
            "n_runs": len(part),
            "spearman_slack_vs_difference_median": spearman_correlation(
                part["slack"].to_numpy(), part["difference_median"].to_numpy()
            ),
        }
        for scope, part in [
            ("dreaddit", points[points["dataset"] == "dreaddit"]),
            ("sdcnl", points[points["dataset"] == "sdcnl"]),
            ("overall", points),
        ]
    ]
    return points, correlations
