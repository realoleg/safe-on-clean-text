from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safe_on_clean_text.analyses.band_decomposition import band_decomposition_table
from safe_on_clean_text.analyses.exploratory import slack_analysis
from safe_on_clean_text.analyses.low_prevalence_calibration import low_prevalence_calibration_table
from safe_on_clean_text.analyses.natural_shift import natural_shift_tables
from safe_on_clean_text.analyses.oracle_cost import m3_cost_ratio_table, oracle_cost_rows, oracle_cost_table
from safe_on_clean_text.analyses.weighted_conformal import clipped_weights_table, weighted_conformal_table
from safe_on_clean_text.analyses.workload_matched import workload_matched_tables
from safe_on_clean_text.analyses.workload_matched_sensitivity import workload_matched_sensitivity_table
from safe_on_clean_text.analyses.zero_shot_ranking import ENCODERS, zero_shot_ranking_tables
from safe_on_clean_text.conformal.methods import METHOD_LABELS
from safe_on_clean_text.degradations import length_bucket
from safe_on_clean_text.evaluation.aggregation import (
    aggregate,
    available_models,
    median_size_band,
    lookup,
    primary_summary,
)
from safe_on_clean_text.paper import FIGURES, PAPER_DIR, TABLES
from safe_on_clean_text.paper.digest import results_digest
from safe_on_clean_text.paper.figures import (
    PAPER_MODELS,
    plot_c1_workload_curves,
    plot_cut_not_short,
    plot_class_coverage,
    plot_degradation_curves,
    plot_grouped_bars,
    plot_lodo,
    plot_prevalence_curves,
    plot_reliability,
    plot_mechanism_schematic,
    plot_score_mechanism,
    plot_risk_coverage_vs_workload,
    plot_slack,
    plot_subreddit_prevalence,
    plot_triage_scheme,
)
from safe_on_clean_text.paper.hypotheses import (
    hypothesis_table,
    matched_workload_hypotheses_table,
    natural_shift_hypotheses_table,
)
from safe_on_clean_text.paper.tables import (
    copy_fixed_split_tables,
    jitter_table,
    conformal_table,
    group_rows,
    data_table,
    classification_metrics_table,
    llm_table,
)
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS, ENSEMBLES, LLM_VARIANTS, SDCNL_MODELS
from safe_on_clean_text.text import word_count

FIGURE5_METHODS = ["M0", "M1", "M3-mix", "M5", "B1"]
# Figure 4 x position: share of words kept (%) under truncation, deletion / typo probability (%) otherwise.
DEGRADATION_STRENGTH = {
    "head_75": 75,
    "head_50": 50,
    "head_25": 25,
    "tail_75": 75,
    "tail_50": 50,
    "tail_25": 25,
    "del_15": 15,
    "del_30": 30,
    "typo_05": 5,
    "typo_10": 10,
}
CLEAN_STRENGTH = {"head": 100, "tail": 100, "del": 0, "typo": 0}


# ---------------------------------------------------------------- Table 6 (hypotheses) and its sensitivity appendix


# ---------------------------------------------------------------- Tables 1-5


# ---------------------------------------------------------------- figures 1-5, slack, mechanism


def class_coverage_figure(
    summary: pd.DataFrame, models: list[str], output: Path, alpha: float = ALPHA, title: str = ""
) -> pd.DataFrame:
    points = []
    for model in models:
        for metric in ["coverage", "c0", "c1"]:
            low, high = median_size_band(summary, model, "clean", "M0", metric, alpha)
            for method in ["M0", "M1"]:
                points.append(
                    {
                        "model": model,
                        "method": method,
                        "metric": metric,
                        "band_low": low,
                        "band_high": high,
                        **{
                            s: lookup(summary, model, "clean", method, metric, alpha, s)
                            for s in ["median", "q025", "q975", "seed_min", "seed_max"]
                        },
                    }
                )
    points = pd.DataFrame(points)
    points.to_csv(output.with_suffix(".csv"), index=False)
    plot_class_coverage(points, output, title, nominal=1 - alpha)
    return points


def prevalence_figure(dreaddit: dict, sdcnl: dict, sdcnl_summary: pd.DataFrame) -> None:
    """
    Coverage and workload per 1,000 against pi: M0 and M1 calibrated at the original prevalence and M0-LS
    (Dreaddit); SDCNL analogue from the reweighted long tables (no M0-LS run).
    """

    prevalence = pd.read_parquet(Path(dreaddit["evaluation"]["output_dir"]) / "prevalence_long.parquet")
    prevalence = prevalence[prevalence["model"].isin(DREADDIT_MODELS) & (prevalence["alpha"] == ALPHA)]
    prevalence["scenario"] = np.where(
        prevalence["scenario"] == "label_shift", prevalence["method"] + "-LS", prevalence["method"]
    )
    table = aggregate(
        prevalence,
        ["model", "scenario", "prevalence"],
        metrics=["coverage", "workload_per_1000", "uncertainty_referrals_per_1000"],
    )
    table.to_csv(FIGURES / "figure3_prevalence.csv", index=False)
    plot_prevalence_curves(table, FIGURES / "figure3_prevalence")

    rows = []
    for model in SDCNL_MODELS:
        for method in ["M0", "M1"]:
            for p in [0.05, 0.1, 0.2, 0.35, 0.52]:
                for metric, name in [
                    (f"coverage_pi={p}", "coverage"),
                    (f"workload_per_1000_pi={p}", "workload_per_1000"),
                ]:
                    rows.append(
                        {
                            "model": model,
                            "scenario": method,
                            "prevalence": p,
                            "metric": name,
                            **{
                                s: lookup(sdcnl_summary, model, "clean", method, metric, statistic=s)
                                for s in ["median", "q025", "q975"]
                            },
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(FIGURES / "appendix_sdcnl_figure3_prevalence.csv", index=False)
    plot_prevalence_curves(table, FIGURES / "appendix_sdcnl_figure3_prevalence")


def degradation_figure(
    summary: pd.DataFrame, model: str, output: Path, alpha: float = ALPHA, title: str = "", height: float = 4.0
) -> None:
    rows = []
    for family in ["head", "tail", "del", "typo"]:
        for condition in ["clean"] + [c for c in DEGRADATION_STRENGTH if c.startswith(family)]:
            low, high = median_size_band(summary, model, condition, "M1", "c1", alpha)
            for method in ["M0", "M1", "M3", "M3-mix"]:
                for metric, name in [("c1", "c1"), ("workload_per_1000_pi=0.1", "workload_per_1000")]:
                    rows.append(
                        {
                            "model": model,
                            "alpha": alpha,
                            "family": family,
                            "condition": condition,
                            "strength": CLEAN_STRENGTH[family]
                            if condition == "clean"
                            else DEGRADATION_STRENGTH[condition],
                            "method": method,
                            "metric": name,
                            "band_low": low,
                            "band_high": high,
                            **{
                                s: lookup(summary, model, condition, method, metric, alpha, s)
                                for s in ["median", "q025", "q975", "seed_min", "seed_max"]
                            },
                        }
                    )
    points = pd.DataFrame(rows)
    points.to_csv(output.with_suffix(".csv"), index=False)
    plot_degradation_curves(points, output, title, height)


def risk_coverage_workload_figure(summary: pd.DataFrame) -> None:
    points = []
    for model in DREADDIT_MODELS:
        for condition in ["clean", "head_25"]:
            for method in FIGURE5_METHODS:
                points.append(
                    {
                        "model": model,
                        "condition": condition,
                        "method": method,
                        "c1": lookup(summary, model, condition, method, "c1"),
                        "c1_q025": lookup(summary, model, condition, method, "c1", statistic="q025"),
                        "c1_q975": lookup(summary, model, condition, method, "c1", statistic="q975"),
                        "workload_per_1000_pi=0.1": lookup(
                            summary, model, condition, method, "workload_per_1000_pi=0.1"
                        ),
                    }
                )
    points = pd.DataFrame(points)
    points.to_csv(FIGURES / "figure5_points.csv", index=False)
    # The clean panel's dense region (encoders' M1 / M5 / B1 points near 0.90) is shown enlarged.
    plot_risk_coverage_vs_workload(
        points, FIGURES / "figure5_risk_coverage_vs_workload", inset={"clean": (295, 440, 0.855, 0.965)}
    )


def risk_coverage_workload_curves_figure() -> None:
    """
    Figure 5 with thin c1-vs-workload curves of M0 and M1 over the alpha grid 0.02-0.20 (H7 curves) for every model;
    points and inset as in Figure 5, which stays unchanged.
    """

    points = pd.read_csv(FIGURES / "figure5_points.csv")
    curves = pd.read_csv(FIGURES / "appendix_h7_c1_vs_workload.csv")
    curves = curves[curves["method"].isin(["M0", "M1"])]
    curves.to_csv(FIGURES / "figure5_risk_coverage_vs_workload_curves.csv", index=False)
    plot_risk_coverage_vs_workload(
        points,
        FIGURES / "figure5_risk_coverage_vs_workload_curves",
        inset={"clean": (295, 440, 0.855, 0.965)},
        curves=curves,
    )


def slack_figure(dreaddit: dict, sdcnl: dict) -> None:
    points, correlations = slack_analysis(dreaddit, sdcnl)
    points.to_csv(TABLES / "exploratory_slack_points.csv", index=False)
    pd.DataFrame(correlations).to_csv(TABLES / "exploratory_slack_spearman.csv", index=False)
    plot_slack(points, FIGURES / "exploratory_slack")


def score_mechanism_figures(dreaddit: dict) -> None:
    directory = Path(dreaddit["evaluation"]["output_dir"]) / "mechanisms"
    histograms = pd.read_csv(directory / "score_histograms.csv")
    thresholds = pd.read_parquet(directory / "raw" / "thresholds.parquet")
    thresholds = aggregate(thresholds, ["model", "method", "length_bucket"], metrics=["stress_in_set_if_p_at_least"])
    thresholds.to_csv(FIGURES / "figure_mechanism_thresholds.csv", index=False)
    cutoff = thresholds.set_index(["model", "method", "length_bucket"])["median"]
    # M3-mix thresholds depend on the length bucket: each panel uses the bucket of the condition's median length.
    conditions = pd.read_csv(Path(dreaddit["data"]["processed_dir"]) / "conditions.csv", keep_default_na=False)
    buckets = {
        condition: length_bucket(
            pd.Series([word_count(conditions[conditions["condition"] == condition]["text"]).median()])
        ).iloc[0]
        for condition in ["clean", "head_25", "tail_25"]
    }
    for model in DREADDIT_MODELS:
        lines = {
            condition: {
                "M0": float(cutoff.loc[(model, "M0", "all")]),
                "M1": float(cutoff.loc[(model, "M1", "all")]),
                "M3-mix": float(cutoff.loc[(model, "M3-mix", bucket)]),
            }
            for condition, bucket in buckets.items()
        }
        name = "figure_mechanism_distilbert" if model == "distilbert" else f"appendix_mechanism_{model}"
        plot_score_mechanism(histograms[histograms["model"] == model], lines, FIGURES / name)


def mechanism_schematic_figure() -> None:
    """
    Conceptual mechanism figure; the two cut-offs are DistilBERT's median M0 / M1 risk thresholds from the mechanism
    figure's threshold table.
    """

    thresholds = pd.read_csv(FIGURES / "figure_mechanism_thresholds.csv")
    thresholds = thresholds[(thresholds["model"] == "distilbert") & (thresholds["length_bucket"] == "all")]
    cutoffs = thresholds.set_index("method")["median"]
    plot_mechanism_schematic({m: float(cutoffs[m]) for m in ["M0", "M1"]}, FIGURES / "figure_schematic")


# ---------------------------------------------------------------- appendix figures and tables


def domain_shift_appendix(dreaddit: dict) -> None:
    output_dir = Path(dreaddit["evaluation"]["output_dir"])
    lodo = pd.read_parquet(output_dir / "lodo_long.parquet")
    table = aggregate(
        lodo, ["model", "domain", "method", "alpha"], replicate="bootstrap", metrics=["coverage", "c0", "c1"]
    )
    bands = pd.read_csv(output_dir / "lodo_summary.csv")
    bands = bands[bands["seed"].astype(str) == "all"][
        ["model", "domain", "method", "alpha", "metric", "band_low", "band_high"]
    ]
    table = table.merge(bands, on=["model", "domain", "method", "alpha", "metric"], how="left", validate="one_to_one")
    table.to_csv(TABLES / "appendix_lodo.csv", index=False)
    plot_lodo(table[(table["alpha"] == ALPHA) & (table["metric"] == "coverage")], FIGURES / "appendix_lodo")


def mechanism_appendix(dreaddit: dict, summary: pd.DataFrame) -> None:
    """
    Mechanism analyses under the primary aggregation from the raw per-(seed, replicate) tables.
    """

    raw = Path(dreaddit["evaluation"]["output_dir"]) / "mechanisms" / "raw"
    # Agreement buckets (first 200 resplits, clean, M0 / M1) and H6 AUROCs + logistic regression of a referral (full or empty set) on agreement.
    buckets = []
    for model in DREADDIT_MODELS:
        groups = group_rows(dreaddit, model, "agreement")
        groups = groups[
            (groups["condition"] == "clean") & groups["method"].isin(["M0", "M1"]) & (groups["alpha"] == ALPHA)
        ]
        buckets.append(
            aggregate(groups, ["method", "group"], metrics=["n", "coverage", "c1", "workload"]).assign(model=model)
        )
    buckets = pd.concat(buckets, ignore_index=True)
    buckets.to_csv(TABLES / "appendix_agreement_buckets.csv", index=False)
    distilbert = buckets[(buckets["model"] == "distilbert") & (buckets["metric"] == "coverage")]
    distilbert = distilbert.assign(method=distilbert["method"].map(METHOD_LABELS))
    plot_grouped_bars(
        distilbert,
        "group",
        "method",
        "median",
        "q025",
        "q975",
        FIGURES / "appendix_agreement_distilbert",
        "coverage (median over resplits)",
        "DistilBERT, clean, by annotator agreement",
        reference=0.90,
    )
    auroc = aggregate(pd.read_parquet(raw / "auroc.parquet"), ["model", "method", "score"], metrics=["auroc"])
    auroc.to_csv(TABLES / "appendix_agreement_auroc.csv", index=False)
    regression = pd.read_parquet(raw / "regression.parquet")
    aggregate(
        regression,
        ["model", "method"],
        metrics=[c for c in regression.columns if c not in ["model", "seed", "replicate", "method"]],
    ).to_csv(TABLES / "appendix_agreement_regression.csv", index=False)

    # Unanimous-only main table, confident-miss LIWC, rows with missing agreement.
    aggregate(pd.read_parquet(raw / "unanimous.parquet"), ["model", "subset", "condition", "method"]).to_csv(
        TABLES / "appendix_unanimous.csv", index=False
    )
    aggregate(pd.read_parquet(raw / "liwc.parquet"), ["model", "method"]).to_csv(
        TABLES / "appendix_confident_miss_liwc.csv", index=False
    )
    aggregate(pd.read_parquet(raw / "missing.parquet"), ["model", "rows"]).to_csv(
        TABLES / "appendix_missing_agreement.csv", index=False
    )

    # Subreddits vs prevalence: subreddit coverage against its stress share, with pi * c1 + (1 - pi) * c0 of the clean run.
    subreddit = aggregate(pd.read_parquet(raw / "subreddit.parquet"), ["model", "subreddit"])
    subreddit.to_csv(TABLES / "appendix_subreddits.csv", index=False)
    wide = (
        subreddit[subreddit["model"] == "distilbert"]
        .pivot(index="subreddit", columns="metric", values="median")
        .reset_index()
    )
    wide = wide.rename(columns={"n": "n_median"})
    for method in ["M0", "M1"]:
        c0, c1 = (
            lookup(summary, "distilbert", "clean", method, "c0"),
            lookup(summary, "distilbert", "clean", method, "c1"),
        )
        wide[f"{method}_line"] = wide["stress_share"] * c1 + (1 - wide["stress_share"]) * c0
    plot_subreddit_prevalence(wide, FIGURES / "appendix_subreddits_distilbert")

    # Sentence occlusion (DistilBERT, RoBERTa; Dreaddit test; mean over posts, +/- SD over seeds).
    occlusion = pd.read_csv(Path(dreaddit["evaluation"]["output_dir"]) / "mechanisms" / "sentence_occlusion.csv")
    occlusion["series"] = occlusion["model"].map({"distilbert": "DistilBERT", "roberta": "RoBERTa"}) + occlusion[
        "label"
    ].map({0: ", not stress", 1: ", stress"})
    occlusion["low"] = occlusion["mean_delta"] - occlusion["mean_delta_sd_over_seeds"]
    occlusion["high"] = occlusion["mean_delta"] + occlusion["mean_delta_sd_over_seeds"]
    plot_grouped_bars(
        occlusion,
        "position",
        "series",
        "mean_delta",
        "low",
        "high",
        FIGURES / "appendix_occlusion",
        "mean change in $p(\\text{stress})$\nwhen the sentence is removed",
        "Sentence occlusion (test, clean)",
        reference=0.0,
    )


def reliability_appendix(dreaddit: dict, sdcnl: dict) -> None:
    """
    Reliability bins (15 equal-width bins of raw p(stress), fixed test split, clean): per run the bin's mean p and stress
    rate, then the mean over seeds per bin.
    """

    rows = []
    for config, models in [(dreaddit, DREADDIT_MODELS), (sdcnl, SDCNL_MODELS)]:
        for model in models:
            for path in sorted(
                (Path(config["paths"]["runs_dir"]) / config["data"]["name"] / model).glob("seed_*/predictions.csv.gz")
            ):
                test = pd.read_csv(path)
                test = test[(test["split"] == "test") & (test["condition"] == "clean")]
                index = np.minimum((test["prob_stress"].to_numpy() * 15).astype(int), 14)
                for b, part in test.groupby(index):
                    rows.append(
                        {
                            "dataset": config["data"]["name"],
                            "model": model,
                            "seed": path.parent.name.removeprefix("seed_"),
                            "bin": b,
                            "n": len(part),
                            "mean_p": part["prob_stress"].mean(),
                            "stress_rate": part["label"].mean(),
                        }
                    )
    bins = (
        pd.DataFrame(rows)
        .groupby(["dataset", "model", "bin"], sort=True)[["n", "mean_p", "stress_rate"]]
        .mean()
        .reset_index()
    )
    bins.to_csv(TABLES / "appendix_reliability_bins.csv", index=False)
    for dataset in ["dreaddit", "sdcnl"]:
        positive = "stress" if dataset == "dreaddit" else "risk"
        plot_reliability(bins[bins["dataset"] == dataset], FIGURES / f"appendix_reliability_{dataset}", positive)


CAPTIONS = """# Figure captions

Conformal quantities use alpha = 0.10 unless stated otherwise. Each is the median over 1,000 grouped resplits of
calibration + test, and bars or bands show the 2.5-97.5% range over resplits. For encoders, the metric is first averaged
over the 5 seeds within each resplit, and the median and interval are then taken over resplits. The tables report the
between-seed spread. The LLMs are zero-shot with prompt P1 and one run each. The grey band is the 95% beta-binomial
reference band of the coverage that the guarantee implies at the calibration and test sizes of that population. "Risk"
denotes the positive class of both datasets, which is stress on Dreaddit and suicidal ideation on SDCNL. Figures that
show Dreaddit alone say "stress" and "not stress". Method labels: M0 (marginal), M1 (class-conditional), M2 (domain ×
class), M3 (matched degradation), M3-mix (degradation mixture), M5 (risk-only, one-sided), M0-LS (marginal, label-shift
weighted), B1 (validation threshold), B2 (temperature scaling + abstention), B3 (seed ensemble).

**Figure 1.** Triage scheme. The prediction set decides the action ({not risk} -> dismiss, {risk} -> flag, full or
empty set -> refer to a human). Every per-class metric is reweighted to a risk prevalence pi as pi * metric(risk) +
(1 - pi) * metric(not risk).

**Figure 2.** Coverage, not-risk coverage c0 and risk-class coverage c1 of each model on clean Dreaddit under M0
(marginal) and M1 (class-conditional), shown against each metric's reference band. The appendix repeats the figure for
alpha = 0.05 and for SDCNL.

**Figure 3.** Coverage (top) and workload per 1,000 screened (bottom) as functions of the risk prevalence pi. M0
(marginal) and M1 (class-conditional) are calibrated at the original prevalence. M0-LS (marginal, label-shift weighted,
Podkopaev & Ramdas, 2021) knows pi. An appendix version covers SDCNL with M0 and M1 only, since M0-LS was not run there.

**Figure 4.** DistilBERT risk-class coverage c1 (top) and workload per 1,000 at pi = 0.10 (bottom) against degradation
strength (head / tail truncation, word deletion, letter typos). Methods are M0 (marginal) and M1 (class-conditional),
both calibrated on clean texts, plus M3 (matched degradation) and M3-mix (degradation mixture, with length bucket x
class groups). The grey band is the c1 reference band. Appendix versions show the other models, alpha = 0.05 and SDCNL.

**Figure 5.** Risk-class coverage c1 against workload per 1,000 screened at pi = 0.10. The left panel shows clean texts
with the dense region enlarged, and the right panel shows texts truncated to their first 25% of words. Each line joins
the operating points of one model. Markers identify the method among M0 (marginal), M1 (class-conditional), M3-mix
(degradation mixture), M5 (risk-only, one-sided) and B1 (validation threshold). M5 coincides with M1 because it flags
exactly the cases whose M1 set contains the risk class, which makes c1 and workload equal. The dotted line marks the
nominal 0.90.

**Mechanism figure (main text).** DistilBERT p(stress) of stress and not stress posts on clean texts and after
truncation to the first or last 25% of words, pooled over calibration + test with shares averaged over the 5 seeds. The
vertical lines mark the p(stress) at or above which each method puts the stress class in the prediction set. These
cut-offs are averaged over seeds per resplit, and the figure shows their median over 1,000 resplits. M0 (marginal) has
a single threshold and M1 (class-conditional) contributes its stress-class threshold. M3-mix (degradation mixture)
thresholds depend on the text-length bin, so each panel shows the threshold of the bin that contains the condition's
median length (> 80 words for clean, <= 25 words for head/tail 25%). The number at the top of each line is its
cut-off. The appendix shows the other six models.

**Schematic figure (conceptual).** How degradation moves at-risk posts into the band between the two cut-offs. The two
curves are illustrative beta-shaped densities of the model score p(risk) of at-risk posts for clean and degraded text,
not estimates from data. The vertical lines are DistilBERT's median M0 (marginal) and M1 (class-conditional) risk
cut-offs from the mechanism figure. Degraded at-risk posts that land in the shaded band between them are still caught
by M0 (their set contains risk) but missed by M1.

**Exploratory figure (post hoc, mechanism-derived).** Each point is one run, covering every encoder seed and the single
runs of TF-IDF and LLM S-tok P1. Circles are Dreaddit runs and triangles SDCNL runs, with hollow markers for LLMs. The x
axis gives the slack, defined as the clean-data c1 under M0 (median over resplits) minus 0.90. On the y axis is the
paired difference c1(M1) - c1(M0) at head_25, as a median with bars for the 2.5-97.5% range over resplits. The analysis
was defined after the main results and is descriptive only.

**Appendix figures.** Leave-one-domain-out coverage of the held-out domain under M0 and M1, from 1,000 calibration
bootstraps (H4). Coverage by annotator-agreement bucket for DistilBERT over the first 200 resplits. For DistilBERT,
subreddit coverage is plotted against the subreddit's stress share together with the reweighting line. Sentence
occlusion shows the mean change in p(stress) on the test split when the first, a middle or the last sentence is
removed, with +/- SD over seeds. Reliability diagrams use 15 equal-width bins on the fixed test split, with seeds
averaged per bin.
"""

TABLE_CAPTIONS = """# Table captions

Conformal tables give medians and 2.5-97.5% quantiles (q025, q975) over 1,000 grouped resplits of calibration + test.
SDCNL resplits are grouped by author. For encoders, each resplit is first averaged over the 5 seeds (primary
aggregation). The columns seed_min and seed_max give the between-seed spread as the min-max of the per-seed medians and
are empty for single-run models. Results are reported for alpha = 0.10 and 0.05 (column alpha). Methods appear as the
codes of the labels in figures/captions.md, and <model>_ensemble rows are B3 (seed ensemble). Metrics: coverage, c0 / c1
(coverage of the not-risk / risk class), workload_per_1000_pi=0.1 (posts not dismissed, that is flagged or referred, per
1,000 screened at risk prevalence 0.10), uncertainty_referrals_per_1000_pi=0.1 (referrals, meaning full or empty sets,
per 1,000 at prevalence 0.10) and coverage_pi=0.1 (coverage reweighted to prevalence 0.10).

**Table 1 (table1_data.csv).** Data per dataset and split. Columns give rows, posts (Dreaddit) or author groups
(SDCNL), risk rows, risk share (stress on Dreaddit, suicidal ideation on SDCNL) and median words. For Dreaddit the table
adds rows and risk share per domain and rows per annotator-agreement bucket.

**Table 2 (table2_classification.csv).** Classification on clean texts of the fixed validation and test splits. Reported
are macro-F1, AUROC, Brier score, NLL and ECE (15 equal-width and 15 equal-mass bins of p(risk)), each as the median and
min-max over seeds (n_seeds). The column probabilities marks raw or temperature-scaled values. Temperature scaling, with
the temperature fitted on validation, is applied to Dreaddit only.

**Table 3 (table3_conformal.csv).** Dreaddit conformal results per model, method and scenario. The four scenarios are
clean texts, clean texts reweighted to risk prevalence 0.10 (pi=0.1), texts truncated to their first 25% of words
(head_25) and "worst in-distribution domain (min of 5; clean; first 200 resplits)". The last one takes, per resplit, the
lowest coverage among the 5 Dreaddit domains (abuse, anxiety, financial, PTSD, social) on that resplit's test half, over
the first 200 resplits. Every model in it is trained and calibrated on all domains, so it is an in-distribution group
check. The leave-one-domain-out analysis (H4, appendix_lodo.csv) differs in that it removes a domain from training and
calibration.

**Table 4 (table4_llms.csv).** Zero-shot LLMs by confidence method and prompt (P1-P3). S-tok uses answer-token
probabilities, S-verb a verbalised probability and S-cons self-consistency over 10 samples. The table gives validation
and test macro-F1 and test AUROC on the clean fixed split, and the mean probability mass on the answer tokens (S-tok,
clean test). It also lists the conformal operating points on clean texts at alpha = 0.10 under M0 and M1, as the median
and 2.5-97.5% range over resplits for the single run of each.

**Table 5 (table5_sdcnl.csv).** SDCNL replication in the format of Table 3, for TF-IDF + LogReg, DistilBERT (5 seeds)
and Qwen3-4B S-tok P1. It covers methods M0, M1, M3, M3-mix, M5 and B1 and has no domain scenario.

**Table 6 (table6_hypotheses.csv).** Criteria of hypotheses H1-H6 and of their SDCNL replications R1-R3 under the
primary aggregation. The table has one row per model-level quantity, per tally of models meeting a criterion and per
verdict. Column value holds the quantity, and margin is its distance to the criterion's threshold, positive when the
criterion is met. The meets column records whether the criterion is met, and seed_spread_min_max_of_medians gives the
between-seed spread. Reference bands are the 95% beta-binomial bands of the coverage the guarantee implies at the median
calibration and test sizes.

**Appendix tables.**
- appendix_table6_sensitivity_pooled.csv lists the Table 6 tallies and verdicts that differ when an encoder's seeds are
  pooled, with quantiles over all seed x resplit values in place of seed-averaged resplits.
- Leave-one-domain-out results (H4) are in appendix_lodo.csv. It gives coverage, c0 and c1 of the held-out domain under
  M0 and M1 as the median and 2.5-97.5% over 1,000 post-level bootstraps of the four-domain calibration (for DistilBERT,
  3 seeds averaged per bootstrap), with the reference band (band_low, band_high) at that domain's sizes.
- appendix_agreement_buckets.csv reports n, coverage, c1 and workload by annotator-agreement bucket (clean, M0 and M1,
  first 200 resplits).
- The AUROC for a non-unanimous annotation, with a full M0 / M1 prediction set or 1 - max p as the predictor (H6), is in
  appendix_agreement_auroc.csv.
- appendix_agreement_regression.csv holds a logistic regression of a referral (full or empty set) on the agreement
  bucket (reference level unanimous, 1.00), the domain (reference level abuse) and log word count, with coefficients
  over resplits.
- In appendix_unanimous.csv, calibration and test are restricted to unanimously annotated rows (subset unanimous). The
  main metrics appear next to those for all rows, with the remaining sizes.
- appendix_confident_miss_liwc.csv compares stress posts whose set is exactly {not stress} (confident misses) with
  stress posts whose set contains stress, by counts and standardised mean differences (smd_*) of LIWC features.
- Official-test rows without an agreement value are set against rows with one in appendix_missing_agreement.csv.
- appendix_subreddits.csv gives size, stress share and coverage per subreddit (DistilBERT).
- The reliability-diagram bins (15 equal-width bins of p(risk), clean fixed test split, seeds averaged per bin) are
  stored in appendix_reliability_bins.csv.
- appendix_jitter.csv sets the tie-jitter variant, with U(0, 1e-9) added to the scores, beside the main result for the
  discrete-score LLM runs (S-verb, S-cons).
- For each run, appendix_official_test_permutation.csv holds the one-sided p-value of the official-test c0 under M1
  (clean, alpha = 0.10) against the 1,000 resplits.
- appendix_llm_mass.csv summarises LLM answers per split and condition, with unparsed answers, mean answer-token mass,
  the shares of rows with mass below 0.9 and 0.5, and rows with a missing answer-token variant.
- The post hoc slack analysis (figures/captions.md, exploratory figure) and its Spearman correlations are in
  exploratory_slack_points.csv and exploratory_slack_spearman.csv.
- appendix_band_decomposition.csv is exploratory and post hoc. It decomposes the risk-class coverage gap
  c1(M0) - c1(M1) at alpha = 0.10 (resplit protocol, 1,000 resplits, all 11 conditions) for the Table 3 and Table 5
  runs other than the seed ensembles. Both methods are calibrated on clean data and put the risk label in the set when
  its score is at most the threshold, q0 under M0 and q1 under M1. The gap therefore equals the signed share of risk
  test posts whose score lies between q1 and q0 (band_mass), which is negative when the M1 cut-off is looser than the
  M0 cut-off. band_mass_change_vs_clean is the paired difference band mass(condition) - band mass(clean) within each
  resplit. The clean rows also give q0 and q1 and the same thresholds as p(risk) cut-offs, the risk label being in the
  set when p(risk) is at least the cut-off. Rows with seed = seed-averaged use the primary aggregation, and each
  multi-seed model also has one row group per seed. ratio_of_medians is band mass(condition) / band mass(clean) computed
  from the aggregated medians, not per resplit, and is empty when the clean median is below 0.005.
  identity_max_abs_difference is the largest absolute difference between band_mass and the stored c1(M0) - c1(M1) over
  the seeds and resplits of the row group.
"""

MATCHED_WORKLOAD_CAPTIONS = """# Captions of the H7-H10 tables and figure

H7-H10 were added after the main analysis, with their criteria fixed before running (README, "Criteria H7-H10"). All
use Dreaddit, the 7 models of Table 6, alpha = 0.10 and the 1,000 grouped resplits of calibration + test. For encoders,
each resplit is first averaged over the 5 seeds, and the tables give the median, mean and 2.5-97.5% quantiles (q025,
q975) over resplits, with the between-seed spread as seed_min and seed_max. Workload is the share of posts not
dismissed (flagged or referred) per 1,000 screened at risk prevalence 0.10, computed on the test half.

**Table 7 (table7_hypotheses_h7_h10.csv).** Criteria of H7-H10 in the format of Table 6, with one row per model-level
quantity, per count of models or cells meeting a criterion and per verdict. Rows marked descriptive have no criterion.

**appendix_h7_workload_matched.csv (H7).** M1 at a workload-matched alpha' against M0 at alpha = 0.10, with M1 and M0
calibrated on clean data. In each resplit and for each seed, alpha' is chosen on the calibration half as the value of
the grid 0.02, 0.03, ..., 0.20 nearest to 0.10 at which M1's in-sample clean workload is not lower than M0's (the smaller
alpha on a tie). When no grid value reaches M0's workload, alpha' is 0.02 and the resplit counts as unmatched
(column unmatched, its mean being the share). Metrics per condition are alpha', the calibration-half workloads of M0 and
of M1 at alpha', c1 of M0 and of M1 at alpha' on the test half, their paired difference c1_difference = c1(M1 at alpha')
- c1(M0), both test-half workloads and the workload gap M1(alpha') - M0. identity_max_abs_difference is the largest
difference between the recomputed alpha = 0.10 points and the stored main results.

**appendix_h7_c1_vs_workload (figure and CSV, H7).** Risk-class coverage c1 against workload per 1,000 over the alpha
grid 0.02-0.20 for M0 and M1 (clean calibration) and M3 (calibration in the test condition), on clean texts and under
head_25, as medians over resplits. The enlarged point on each line is alpha = 0.10, and the dotted line marks 0.90.

**appendix_h8_calibration_risk_10pct.csv (H8).** In each resplit the risk rows of the calibration half are thinned to
round(0.10 n1) rows (about 37), drawn without replacement once per resplit and shared by all models, while the test half
is unchanged. Rows give c1 and workload of M0, M1 and M0-LS (label-shift M0 targeting the test half's risk share) in
every condition, and the paired c1(M1) - c1(M0). The threshold comparison columns q_M1_below_q_M0, q_M1_equal_q_M0 and
q_M1_above_q_M0 are 0/1 per seed and resplit, so their mean is the share of resplits (q_M1 is M1's risk-class threshold
and q_M0 the M0 threshold, both log-odds scores). calibration_slack_c1_M0 is the in-sample c1 of M0 on the thinned
calibration half minus 0.90. The threshold and slack columns do not depend on the condition.

**appendix_h9_oracle_cost.csv (H9).** The oracle flags a post when its risk score is at most t, where t is the
ceil(0.9 n1)-th smallest risk score of the calibration half in the same condition, and dismisses it otherwise. This is
the smallest threshold with in-sample c1 of at least 0.90, uncorrected as in B1. oracle_risk_in_set_if_p_at_least is
that threshold as a p(risk) cut-off. Rows give c1 and workload of the oracle, M1 (clean calibration) and M3 (calibration
in the test condition), and workload_ratio_M3_over_oracle and workload_ratio_M1_over_oracle, the ratios of the
seed-averaged workloads within each resplit. For calibration halves of 369-370 risk rows, ceil(0.9 n1) equals the
conformal rank ceil(0.9 (n1 + 1)), so the oracle threshold coincides with M3's risk-class threshold, and the ratio
differs from 1 only through M3's empty sets.

**appendix_h10_weighted_conformal.csv (H10).** Weighted split conformal prediction (Tibshirani et al., 2019) with
density-ratio weights from a logistic-regression domain classifier fitted on the validation split only. Its features
are word count and the share of lower-cased words outside the training vocabulary, computed on validation texts
degraded by the study's degradation code. The source of the classifier is the same validation posts, each in one random
condition as for M3-mix (variant a), or clean (variant b), and its target is those posts in the test condition. Weights
use text features only, and shifts in model scores at fixed length and vocabulary are not corrected. Variant a weights
the M3-mix calibration set and variant b the clean calibration set. Method WCP is the marginal and WCP-class the
class-conditional weighted method. Metrics are c1, workload per 1,000, the share of full sets, the share of infinite
thresholds (over test posts, and over test posts x candidate classes for WCP-class), the effective sample size
ess = (sum w)^2 / sum w^2 of the calibration weights (ess_risk for its risk rows) and max_weight_ratio, the largest over
smallest weight among the calibration and test posts of a resplit. Rows with variant "-" give M0, M1 and M3-mix from the
main results. c1_band_low and c1_band_high are the c1 reference band at the median risk-class calibration and test sizes.
"""

FOLLOW_UP_CAPTIONS = """# Captions of the H7-H10 follow-up tables and figure

Descriptive follow-ups and sensitivity checks without criteria; the H7-H10 verdicts stay as fixed. Same data, models,
alpha = 0.10, 1,000 resplits and primary aggregation as the H7-H10 tables (captions_h7_h10.md). Encoders are
DistilBERT, RoBERTa and MentalRoBERTa; zero-shot models are the three LLMs with S-tok P1.

**appendix_zero_shot_ranking.csv.** Paired differences per resplit, zero-shot model minus encoder (the encoder averaged
over its seeds within the resplit), under head_25, tail_25, del_30 and typo_10, as the median and 2.5-97.5% range over
resplits. Metric oracle_workload is the H9 oracle workload per 1,000 at c1 = 0.90 (lower is cheaper, see
appendix_h9_oracle_cost.csv); metric auroc is the test-half AUROC of the risk log-odds (higher ranks better). verdict
says which side is better when the interval excludes 0.

**appendix_auroc_by_condition.csv.** Test-half AUROC of all 7 models in all 11 conditions (metric auroc) and the paired
difference to the best encoder of that condition, the one with the highest median AUROC (metric
auroc_minus_best_encoder, column best_encoder).

**appendix_h7_sensitivity.csv.** H7 with exact workload matching on the test half's clean workload by linear
interpolation along an alpha grid. For the trained models (moved_method M1), M1's c1 and workload are interpolated over
the grid 0.02-0.20 to M0's clean workload at alpha = 0.10. For the zero-shot models (moved_method M0), M0's alpha is
raised over the grid 0.10-0.99 until its clean workload equals M1's at alpha = 0.10. alpha_star is the interpolated
alpha (the crossing nearest to 0.10), c1_difference_M1_minus_M0 the paired difference and workload_gap_M1_minus_M0 the
remaining workload gap (0 on clean by construction). Resplits without a crossing are left out. Rows with condition
"all" give their share (metric unmatched), the target clean workload and the lowest and highest clean workload the moved
method reaches on its grid.

**appendix_h10_clipped_weights.csv.** H10 variant a, class-conditional (WCP-class), with calibration and test weights
clipped at 11 = 1/pi_k, the bound on the true target / mixture density ratio when the target condition has weight 1/11
in the mixture, next to the unclipped results (column weights). Metrics as in appendix_h10_weighted_conformal.csv. Rows
with model "all" give, per condition, the shares of calibration and test weights above 11 (the weights do not depend
on the model).

**figure5_risk_coverage_vs_workload_curves (figure and CSV).** Figure 5 with thin c1-vs-workload curves over the alpha
grid 0.02-0.20 for M0 and M1 of every model (colour and line style of the model), from the H7 curves. Markers are the
alpha = 0.10 operating points of Figure 5. The CSV holds the curve points; the markers are in figure5_points.csv.
"""

FOLLOW_UP_CAPTIONS_2 = """# Captions of the second set of H7-H10 follow-up tables

Descriptive, without criteria; the H7-H10 verdicts stay as fixed. Same data, models, alpha = 0.10, 1,000 resplits and
primary aggregation as captions_follow_up.md.

**appendix_h7_sensitivity_calibration_matched.csv.** As appendix_h7_sensitivity.csv for the trained models (TF-IDF,
DistilBERT, RoBERTa, MentalRoBERTa; moved_method M1), but the workloads are matched on the calibration half's
in-sample clean data, as in H7: alpha_star is where M1's in-sample clean workload, interpolated along the grid
0.02-0.20, equals M0's at alpha = 0.10. c1 and workload on the test half are interpolated at that alpha, so the clean
workload gap on the test half is no longer 0 by construction. Rows with condition "all" give the share of unmatched
resplits, the calibration-half target workload and the range M1 reaches on the grid.

**appendix_m3_cost_ratios.csv.** Workload per 1,000 at pi = 0.10 of M3 (calibration in the test condition) divided by
M1's clean workload (metric workload_ratio_M3_over_M1_clean) and by M1's workload in the same condition (metric
workload_ratio_M3_over_M1_same_condition), both methods at alpha = 0.10. Ratios of the seed-averaged workloads within
each resplit, median and 2.5-97.5% over resplits. On clean texts M3 equals M1, so both ratios are 1.
"""

NATURAL_SHIFT_CAPTIONS = """# Captions of the H11-H13 tables

H11-H13 were added after the main analysis, with their criteria fixed before running (README, "Criteria H11-H13").
Dreaddit excerpts are cut from longer posts: "We partition the posts into contiguous five-sentence chunks for labeling"
(Turcan and McKeown, 2019, section 3.2), so short excerpts are a natural counterpart of truncation. Length is the clean
word count. Long excerpts have at least the median length of the pool (calibration + test, 81 words) and short ones at
most its lower quartile (65 words). The truncation level k = 50 was chosen mechanically, being the level of 75, 50 and
25 whose median truncated length of long excerpts (51 words) is closest to the median length of short excerpts (55
words). Dreaddit, the 7 models of Table 6, alpha = 0.10, 1,000 grouped resplits, encoders averaged over their 5 seeds
within each resplit, then median, mean and 2.5-97.5% quantiles over resplits, with seed_min and seed_max the
between-seed spread. Workload is per 1,000 screened at risk prevalence 0.10.

**Table 8 (table8_hypotheses_h11_h13.csv).** Criteria of H11-H13 in the format of Table 6.

**appendix_natural_shift_descriptives.csv.** Length quantiles of the pool, size, median length, risk share and domain
shares of the long and short groups, the number of short risk excerpts per test half and of long risk excerpts per
calibration half over the resplits, and the number of excerpts of at most 40 words (with their risk count). The
secondary analysis on excerpts of at most 40 words needs at least 30 risk excerpts per test half and is not run, because
there are 12 (median over resplits).

**appendix_natural_shift.csv.** c1 and workload per 1,000 per (analysis, test set, method), with the median risk
counts of calibration (n_calibration_1) and test (n_test_1) and the c1 reference band at those counts. Analysis "long
calibration": M0 and M1 calibrated on the clean long excerpts of the calibration half and M3-mix on those excerpts in
their M3-mix condition, with (length bucket, class) groups, tested on the short clean excerpts of the test half (H11,
H13), on its long clean excerpts and on its long excerpts truncated to head_50 and tail_50. Analysis "cross-fitted
prediction" (H12): within the calibration half the long excerpts are split at random by post into halves A and B, and
each is cut to head_50 or tail_50 with probability 1/2 (both draws once per resplit, shared by all models). M0 or M1
fitted on clean A predicts c1 as the share of B's risk excerpts, in their truncated versions, whose risk label is in
the set, and the two folds are averaged after swapping A and B. Analysis "long calibration, domain-reweighted"
(sensitivity): weighted split conformal prediction with the long calibration excerpts weighted by domain so that their
domain shares match those of the short test excerpts (label-free), for M0, M1 and M3-mix, with domain_weight_ess the
effective sample size (sum w)^2 / sum w^2 of the calibration weights. Analysis "standard calibration" (secondary):
calibration on the whole clean calibration half (M3-mix on its M3-mix assignment), tested on the short clean
excerpts of the test half, which gives coverage conditional on length. Its "all clean" rows equal the main results and
serve as a check.

**appendix_natural_shift_differences.csv.** Paired differences per resplit (within seed, then averaged over seeds): c1
of M1 minus M0 on short excerpts; for each method, short-excerpt c1 minus c1 on the long test excerpts (clean, head_50,
tail_50), all with long calibration; the observed band mass c1(M0) - c1(M1) on short excerpts and the predicted band
mass from the cross-fitted prediction; and the prediction error, predicted minus observed c1 of M1.
"""

CUT_NOT_SHORT_CAPTION = """# Caption of figure_cut_not_short

**Figure (figure_cut_not_short).** Risk-class coverage c1 of M1 (class-conditional) calibrated on the long clean
excerpts of the calibration half (at least 81 words, the pool median), on four kinds of test excerpts of the test half:
long clean excerpts, naturally short excerpts (at most 65 words, the pool lower quartile), and long excerpts cut to
their first or last 50% of words (median 51 words, against 55 for the short excerpts). Markers are medians over 1,000
grouped resplits with 2.5-97.5% bars, encoders averaged over their 5 seeds within each resplit, alpha = 0.10. The light
grey strip is the c1 reference band for the short test excerpts (at the median risk counts of the long calibration and
short test excerpts), and the dotted line marks 0.90. Values are in appendix_natural_shift.csv and
figure_cut_not_short.csv.
"""

ASSET_LIST = [
    # (item, type, output, source files)
    (
        "Figure 1: triage scheme and prevalence formula",
        "main figure",
        "figures/figure1_triage",
        "safe_on_clean_text/paper/figures.py (diagram)",
    ),
    (
        "Figure 2: coverage and class coverage per model vs the band (H1, H5)",
        "main figure",
        "figures/figure2_class_coverage",
        "results/evaluation/dreaddit/long/*.parquet",
    ),
    (
        "Figure 3: coverage and workload vs pi, M0 vs M1 (+ M0-LS) (H2)",
        "main figure",
        "figures/figure3_prevalence",
        "results/evaluation/dreaddit/prevalence_long.parquet",
    ),
    (
        "Figure 4: c1 vs degradation strength, clean / matched / mix calibration, + workload (H3a-c)",
        "main figure",
        "figures/figure4_degradation_distilbert",
        "results/evaluation/dreaddit/long/*.parquet",
    ),
    (
        "Figure 5: risk-class coverage c1 vs workload per 1,000 at pi = 0.10",
        "main figure",
        "figures/figure5_risk_coverage_vs_workload",
        "results/evaluation/dreaddit/long/*.parquet",
    ),
    (
        "Mechanism figure: DistilBERT p(stress), clean / head_25 / tail_25, M0 / M1 / M3-mix",
        "main figure",
        "figures/figure_mechanism_distilbert",
        "results/evaluation/dreaddit/mechanisms/score_histograms.csv, mechanisms/raw/thresholds.parquet",
    ),
    (
        "Schematic figure: conceptual mechanism, degraded at-risk posts fall between the M0 and M1 cut-offs",
        "main figure",
        "figures/figure_schematic",
        "figures/figure_mechanism_thresholds.csv (DistilBERT M0 / M1 cut-offs); densities schematic",
    ),
    (
        "Exploratory (post hoc): slack vs head_25 M1 - M0",
        "exploratory figure + tables",
        "figures/exploratory_slack; tables/exploratory_slack_*.csv",
        "results/evaluation/{dreaddit,sdcnl}/long/*.parquet",
    ),
    (
        "Exploratory (post hoc): band decomposition of the M1 - M0 risk-class coverage gap",
        "appendix table",
        "tables/appendix_band_decomposition.csv",
        "results/runs/{dreaddit,sdcnl}/*/seed_*/predictions.csv.gz, results/evaluation/{dreaddit,sdcnl}/long/*.parquet",
    ),
    (
        "Appendix: leave-one-domain-out (H4)",
        "appendix figure + table",
        "figures/appendix_lodo; tables/appendix_lodo.csv",
        "results/evaluation/dreaddit/lodo_long.parquet",
    ),
    (
        "Appendix: annotator agreement (H6)",
        "appendix figure + tables",
        "figures/appendix_agreement_distilbert; tables/appendix_agreement_*.csv",
        "results/evaluation/dreaddit/groups/*.parquet, mechanisms/raw/{auroc,regression}.parquet",
    ),
    (
        "Appendix: subreddits vs prevalence",
        "appendix figure + table",
        "figures/appendix_subreddits_distilbert; tables/appendix_subreddits.csv",
        "mechanisms/raw/subreddit.parquet",
    ),
    (
        "Appendix: p(stress) distributions, other models",
        "appendix figures",
        "figures/appendix_mechanism_<model>",
        "mechanisms/score_histograms.csv, raw/thresholds.parquet",
    ),
    (
        "Appendix: sentence occlusion",
        "appendix figure",
        "figures/appendix_occlusion",
        "results/evaluation/dreaddit/mechanisms/sentence_occlusion.csv",
    ),
    (
        "Appendix: ECE and reliability diagrams",
        "appendix figures + table",
        "figures/appendix_reliability_{dreaddit,sdcnl}; tables/appendix_reliability_bins.csv; ECE in table2",
        "results/runs/*/*/seed_*/predictions.csv.gz",
    ),
    (
        "Appendix: alpha = 0.05",
        "appendix figures",
        "figures/appendix_figure2_alpha005, appendix_figure4_distilbert_alpha005; table3 (alpha column)",
        "results/evaluation/dreaddit/long/*.parquet",
    ),
    (
        "Appendix: SDCNL analogues of figures 2-4",
        "appendix figures",
        "figures/appendix_sdcnl_figure{2,3,4}_*",
        "results/evaluation/sdcnl/long/*.parquet",
    ),
    (
        "Appendix: figure 4 for the other Dreaddit models",
        "appendix figures",
        "figures/appendix_figure4_degradation_<model>",
        "results/evaluation/dreaddit/long/*.parquet",
    ),
    (
        "Table 1: data (sizes, balance, domains, agreement)",
        "table",
        "tables/table1_data.csv",
        "data/processed/dreaddit/{train,validation,calibration,test}.csv, data/processed/sdcnl/split_report.csv",
    ),
    (
        "Table 2: classification (macro-F1, AUROC, Brier, NLL, ECE; median and min-max over seeds)",
        "table",
        "tables/table2_classification.csv",
        "results/evaluation/dreaddit/classification.csv; SDCNL from results/runs/sdcnl/*/seed_*/predictions.csv.gz",
    ),
    (
        "Table 3: main conformal (model x method x {clean, pi = 0.10, head_25, worst in-distribution domain})",
        "table",
        "tables/table3_conformal.csv",
        "results/evaluation/dreaddit/long/*.parquet, groups/*.parquet",
    ),
    (
        "Table 4: LLMs, S-tok vs S-verb vs S-cons (+ prompts P1-P3)",
        "table",
        "tables/table4_llms.csv",
        "results/evaluation/dreaddit/long/*.parquet, classification.csv, results/llm/dreaddit/*_run_summary.csv",
    ),
    ("Table 5: SDCNL replication", "table", "tables/table5_sdcnl.csv", "results/evaluation/sdcnl/long/*.parquet"),
    (
        "Table 6: hypotheses H1-H6, R1-R3 (primary aggregation)",
        "table",
        "tables/table6_hypotheses.csv",
        "long tables, LODO, mechanisms/raw/auroc.parquet, classification",
    ),
    (
        "Appendix: Table 6 under the pooled seed x resplit aggregation (verdicts that differ)",
        "appendix table",
        "tables/appendix_table6_sensitivity_pooled.csv",
        "as Table 6",
    ),
    (
        "Appendix tables: unanimous rows, confident-miss LIWC, missing agreement, jitter, permutation p, LLM mass",
        "appendix tables",
        "tables/appendix_{unanimous,confident_miss_liwc,missing_agreement,jitter,official_test_permutation,llm_mass}.csv",
        "mechanisms/raw/*.parquet, long tables, official_test_permutation.csv, results/llm/*/*_run_summary.csv",
    ),
    (
        "Table and figure captions",
        "captions",
        "tables/captions.md; figures/captions.md",
        "safe_on_clean_text/paper/build.py",
    ),
    (
        "Results digest: every cited number with its source file",
        "digest",
        "results_digest.md",
        "results/paper/tables, figures/*.csv, results/checks, sdcnl addendum",
    ),
    (
        "Appendix: SDCNL exploratory addendum (post hoc)",
        "appendix table",
        "results/evaluation/sdcnl/exploratory_addendum.csv (safe_on_clean_text/analyses/exploratory.py)",
        "SDCNL / Dreaddit long tables and predictions",
    ),
]


def matched_workload_analyses(dreaddit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    H7-H10 (criteria fixed before running; Dreaddit, alpha 0.10, workload at pi = 0.10): the analysis tables, the H7
    curves figure, Table 7 and the captions file of these outputs. Returns the oracle rows and the H10 table for the
    follow-up analyses.
    """

    h7, curves = workload_matched_tables(dreaddit)
    h7.to_csv(TABLES / "appendix_h7_workload_matched.csv", index=False)
    curves.to_csv(FIGURES / "appendix_h7_c1_vs_workload.csv", index=False)
    plot_c1_workload_curves(curves, FIGURES / "appendix_h7_c1_vs_workload")
    h8 = low_prevalence_calibration_table(dreaddit)
    h8.to_csv(TABLES / "appendix_h8_calibration_risk_10pct.csv", index=False)
    oracle_rows = oracle_cost_rows(dreaddit)
    h9 = oracle_cost_table(oracle_rows, dreaddit["conditions"])
    h9.to_csv(TABLES / "appendix_h9_oracle_cost.csv", index=False)
    h10 = weighted_conformal_table(dreaddit)
    h10.to_csv(TABLES / "appendix_h10_weighted_conformal.csv", index=False)
    table = matched_workload_hypotheses_table(h7, h8, h9, h10, dreaddit["conditions"])
    table.to_csv(TABLES / "table7_hypotheses_h7_h10.csv", index=False)
    (TABLES / "captions_h7_h10.md").write_text(MATCHED_WORKLOAD_CAPTIONS, encoding="utf-8")
    print(table[table["scope"].isin(["tally", "verdict"])][["hypothesis", "quantity", "value"]].to_string(index=False))
    return oracle_rows, h10


def follow_up_analyses(dreaddit: dict, oracle_rows: pd.DataFrame, h10: pd.DataFrame) -> None:
    """
    Descriptive follow-ups of H7-H10 (no criteria; the H7-H10 verdicts stay as fixed): zero-shot vs encoder ranking,
    exact workload matching for H7, clipped H10 weights, Figure 5 with c1-vs-workload curves, and their captions file.
    """

    pairs, auroc = zero_shot_ranking_tables(dreaddit, oracle_rows)
    pairs.to_csv(TABLES / "appendix_zero_shot_ranking.csv", index=False)
    auroc.to_csv(TABLES / "appendix_auroc_by_condition.csv", index=False)
    workload_matched_sensitivity_table(dreaddit).to_csv(TABLES / "appendix_h7_sensitivity.csv", index=False)
    clipped_weights_table(dreaddit, h10).to_csv(TABLES / "appendix_h10_clipped_weights.csv", index=False)
    risk_coverage_workload_curves_figure()
    (TABLES / "captions_follow_up.md").write_text(FOLLOW_UP_CAPTIONS, encoding="utf-8")
    workload_matched_sensitivity_table(dreaddit, ["tfidf_logreg", *ENCODERS], match_on="calibration").to_csv(
        TABLES / "appendix_h7_sensitivity_calibration_matched.csv", index=False
    )
    m3_cost_ratio_table(oracle_rows, dreaddit["conditions"]).to_csv(TABLES / "appendix_m3_cost_ratios.csv", index=False)
    (TABLES / "captions_follow_up_2.md").write_text(FOLLOW_UP_CAPTIONS_2, encoding="utf-8")


def natural_shift_analyses(dreaddit: dict) -> None:
    """
    H11-H13 (criteria fixed before running): descriptives of long and short excerpts, coverage on naturally short posts
    with the synthetic analogues, cross-fitted prediction, domain reweighting and standard calibration, Table 8 and the
    captions file of these outputs.
    """

    descriptives, table, differences = natural_shift_tables(dreaddit)
    descriptives.to_csv(TABLES / "appendix_natural_shift_descriptives.csv", index=False)
    table.to_csv(TABLES / "appendix_natural_shift.csv", index=False)
    differences.to_csv(TABLES / "appendix_natural_shift_differences.csv", index=False)
    hypotheses = natural_shift_hypotheses_table(table)
    hypotheses.to_csv(TABLES / "table8_hypotheses_h11_h13.csv", index=False)
    (TABLES / "captions_h11_h13.md").write_text(NATURAL_SHIFT_CAPTIONS, encoding="utf-8")
    print(
        hypotheses[hypotheses["scope"].isin(["tally", "verdict"])][["hypothesis", "quantity", "value"]].to_string(
            index=False
        )
    )


def cut_not_short_figure() -> None:
    """
    Main-text figure from the natural-shift table: c1 of M1 calibrated on long clean excerpts on long clean, naturally
    short, head-cut and tail-cut long test excerpts, with the band of the short test excerpts; points CSV and caption.
    """

    table = pd.read_csv(TABLES / "appendix_natural_shift.csv")
    tests = ["long clean", "short clean", "long head_50", "long tail_50"]
    points = table[
        (table["analysis"] == "long calibration")
        & (table["method"] == "M1")
        & (table["metric"] == "c1")
        & table["test"].isin(tests)
    ][["model", "test", "median", "q025", "q975", "seed_min", "seed_max", "c1_band_low", "c1_band_high"]]
    points.to_csv(FIGURES / "figure_cut_not_short.csv", index=False)
    plot_cut_not_short(points, FIGURES / "figure_cut_not_short")
    (FIGURES / "captions_figure_cut_not_short.md").write_text(CUT_NOT_SHORT_CAPTION, encoding="utf-8")


def build(dreaddit: dict, sdcnl: dict) -> None:
    """
    Every paper table, figure, caption file, the results digest and the asset list into results/paper/.
    """

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    dreaddit_models = [m for m in DREADDIT_MODELS + ENSEMBLES + LLM_VARIANTS if m in available_models(dreaddit)]
    dreaddit_models = list(dict.fromkeys(dreaddit_models))
    summary = primary_summary(dreaddit, dreaddit_models)
    sdcnl_summary = primary_summary(sdcnl, SDCNL_MODELS)

    hypothesis_table(dreaddit, sdcnl)
    data_table(dreaddit, sdcnl)
    classification_metrics_table(dreaddit, sdcnl)
    conformal_table(dreaddit, summary, DREADDIT_MODELS + ENSEMBLES, domains=True).to_csv(
        TABLES / "table3_conformal.csv", index=False
    )
    llm_table(dreaddit, summary)
    conformal_table(sdcnl, sdcnl_summary, SDCNL_MODELS, domains=False).to_csv(TABLES / "table5_sdcnl.csv", index=False)

    plot_triage_scheme(FIGURES / "figure1_triage")
    class_coverage_figure(summary, DREADDIT_MODELS, FIGURES / "figure2_class_coverage")
    class_coverage_figure(
        summary, DREADDIT_MODELS, FIGURES / "appendix_figure2_alpha005", alpha=0.05, title=r"$\alpha = 0.05$"
    )
    class_coverage_figure(sdcnl_summary, SDCNL_MODELS, FIGURES / "appendix_sdcnl_figure2_class_coverage", title="SDCNL")
    prevalence_figure(dreaddit, sdcnl, sdcnl_summary)
    degradation_figure(summary, "distilbert", FIGURES / "figure4_degradation_distilbert")
    degradation_figure(
        summary,
        "distilbert",
        FIGURES / "appendix_figure4_distilbert_alpha005",
        alpha=0.05,
        title=r"DistilBERT, $\alpha = 0.05$",
    )
    for model in DREADDIT_MODELS:
        if model != "distilbert":
            degradation_figure(
                summary, model, FIGURES / f"appendix_figure4_degradation_{model}", title=PAPER_MODELS[model][0]
            )
    for model in SDCNL_MODELS:
        degradation_figure(
            sdcnl_summary,
            model,
            FIGURES / f"appendix_sdcnl_figure4_degradation_{model}",
            title="" if model == "distilbert" else f"SDCNL, {PAPER_MODELS[model][0]}",
            height=3.5,
        )
    risk_coverage_workload_figure(summary)
    slack_figure(dreaddit, sdcnl)
    score_mechanism_figures(dreaddit)
    mechanism_schematic_figure()

    domain_shift_appendix(dreaddit)
    mechanism_appendix(dreaddit, summary)
    reliability_appendix(dreaddit, sdcnl)
    jitter_table(summary)
    copy_fixed_split_tables(dreaddit)
    band_decomposition_table(dreaddit, sdcnl).to_csv(TABLES / "appendix_band_decomposition.csv", index=False)
    follow_up_analyses(dreaddit, *matched_workload_analyses(dreaddit))
    natural_shift_analyses(dreaddit)
    cut_not_short_figure()

    (FIGURES / "captions.md").write_text(CAPTIONS, encoding="utf-8")
    (TABLES / "captions.md").write_text(TABLE_CAPTIONS, encoding="utf-8")
    results_digest(dreaddit, sdcnl)
    pd.DataFrame(ASSET_LIST, columns=["item", "type", "output (under results/paper)", "source"]).to_csv(
        PAPER_DIR / "asset_list.csv", index=False
    )
    print(f"tables: {len(list(TABLES.glob('*.csv')))}, figures: {len(list(FIGURES.glob('*.png')))} -> {PAPER_DIR}")
