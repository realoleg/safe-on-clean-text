from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from safe_on_clean_text.conformal.methods import METHODS, calibration_frame, predict_sets
from safe_on_clean_text.conformal.metrics import triage_frame, reweight, summarize_indicators
from safe_on_clean_text.conformal.scores import RISK_CLASS, agreement_bucket, true_class_scores
from safe_on_clean_text.conformal.thresholds import conformal_threshold
from safe_on_clean_text.evaluation.resplits import load_pool, make_replicates, mix_for_conditions, pool_frames
from safe_on_clean_text.provenance import timestamp

# Dreaddit only, alpha 0.10. DistilBERT (5 seeds pooled) is primary; the other runs are descriptive.
PRIMARY = "distilbert"
RUNS = {
    "distilbert": [13, 42, 101, 2024, 7777],
    "tfidf_logreg": [42],
    "roberta": [13, 42, 101, 2024, 7777],
    "mentalroberta": [13, 42, 101, 2024, 7777],
    "gpt_s_tok_P1": ["P1"],
    "qwen_s_tok_P1": ["P1"],
    "gemma_s_tok_P1": ["P1"],
}
OCCLUSION_MODELS = ["distilbert", "roberta"]
ALPHA = 0.1
PREVALENCE = 0.1
MAIN_CONDITIONS = ["clean", "head_25"]
MECHANISM_CONDITIONS = ["clean", "head_25", "tail_25"]
CP_METHODS = list(METHODS)
LIWC = ["negemo", "anx", "sad", "i", "focuspast", "social"]
UNANIMOUS = 1.0
DOMAIN_REFERENCE = "abuse"
AGREEMENT_REFERENCE = "1.00"


def set_codes(sets: np.ndarray) -> np.ndarray:
    return np.where(
        sets.sum(axis=1) == 2,
        "full",
        np.where(sets.sum(axis=1) == 0, "empty", np.where(sets[:, RISK_CLASS], "stress", "not_stress")),
    )


def main_metrics(sets: np.ndarray, test: pd.DataFrame) -> dict[str, float]:
    """
    Main-table metrics of one set matrix: coverage, c0, c1, worst-domain coverage; workload and uncertainty referrals
    per 1,000 and coverage reweighted to the main prevalence.
    """

    labels = test["label"].to_numpy()
    summary = summarize_indicators(triage_frame(sets, labels), labels)
    at_prevalence = reweight(summary, PREVALENCE)
    covered = sets[np.arange(len(labels)), labels]
    domains = test["domain"].to_numpy()
    return {
        "coverage": summary["coverage"],
        "c0": summary["c0"],
        "c1": summary["c1"],
        "coverage_pi=0.1": at_prevalence["coverage"],
        "workload_per_1000_pi=0.1": 1000 * at_prevalence["workload"],
        "uncertainty_referrals_per_1000_pi=0.1": 1000 * at_prevalence["uncertainty_referral"],
        "worst_domain_coverage": min(covered[domains == d].mean() for d in np.unique(domains)),
    }


def thresholds_as_probabilities(calibration: pd.DataFrame, method: str) -> list[dict[str, Any]]:
    """
    p(stress) cutoffs of a method's thresholds on log-odds scores: stress is in the set iff p >= expit(-q_stress),
    not-stress iff p <= expit(q_not). M0: one q; M1: per class; M3-mix: per (length bucket, class).
    """

    scores, labels = true_class_scores(calibration), calibration["label"].to_numpy()
    rows = []
    if method == "M0":
        q = conformal_threshold(scores, ALPHA)
        groups = [("all", q, q)]
    elif method == "M1":
        groups = [
            ("all", conformal_threshold(scores[labels == 0], ALPHA), conformal_threshold(scores[labels == 1], ALPHA))
        ]
    else:
        buckets = calibration["length_bucket"].to_numpy()
        groups = [
            (
                b,
                conformal_threshold(scores[(buckets == b) & (labels == 0)], ALPHA),
                conformal_threshold(scores[(buckets == b) & (labels == 1)], ALPHA),
            )
            for b in np.unique(buckets)
        ]
    for bucket, q_not, q_stress in groups:
        rows.append(
            {
                "method": method,
                "length_bucket": bucket,
                "stress_in_set_if_p_at_least": float(expit(-q_stress)),
                "not_stress_in_set_if_p_at_most": float(expit(q_not)),
            }
        )
    return rows


def missing_agreement_rows(
    keys: dict[str, Any], sets: dict[str, np.ndarray], labels: np.ndarray, prob: np.ndarray, known: np.ndarray
) -> list[dict[str, Any]]:
    """
    Rows with missing vs known agreement: n, stress share, accuracy (p > 0.5), coverage under M0 and M1.
    """

    rows = []
    for group, mask in [("missing", ~known), ("known", known)]:
        if not mask.any():
            continue
        row = {
            **keys,
            "rows": group,
            "n": int(mask.sum()),
            "stress_share": float(labels[mask].mean()),
            "accuracy": float(((prob[mask] > 0.5).astype(int) == labels[mask]).mean()),
        }
        for method in ["M0", "M1"]:
            row[f"{method}_coverage"] = float(sets[method][mask][np.arange(mask.sum()), labels[mask]].mean())
        rows.append(row)
    return rows


def mechanism_run(task: dict[str, Any]) -> dict[str, pd.DataFrame]:
    start = time.perf_counter()
    predictions = pd.read_csv(task["path"])
    pool = task["pool"]
    mix_assignment = mix_for_conditions(
        task["mix_assignment"], pool, task["conditions"], task["conditions"], task["base_seed"]
    )
    frames, mix = pool_frames(predictions, pool, mix_assignment, task["conditions"])
    extra = task["extra"]  # pool-aligned subreddit and LIWC columns
    confidence = pool["confidence"].to_numpy()
    keys = {"model": task["model"], "seed": str(task["seed"])}
    out: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ["auroc", "regression", "unanimous", "subreddit", "thresholds", "liwc", "missing"]
    }

    for replicate, (calibration_positions, test_positions) in enumerate(task["replicates"]):
        row_keys = {**keys, "replicate": replicate}
        calibration_by_condition = {c: f.iloc[calibration_positions] for c, f in frames.items()}
        test = frames["clean"].iloc[test_positions]
        labels = test["label"].to_numpy()
        sets = {m: predict_sets(calibration_by_condition["clean"], test, m, ALPHA)[0] for m in ["M0", "M1"]}
        test_confidence = confidence[test_positions]
        known = ~np.isnan(test_confidence)
        prob = test["prob_stress"].to_numpy()

        # (1) Agreement: AUROC of "full set" and of 1 - max p for a non-unanimous label; referral regression.
        non_unanimous = test_confidence[known] < UNANIMOUS
        for method in ["M0", "M1"]:
            full = sets[method].sum(axis=1)[known] == 2
            out["auroc"].append(
                {
                    **row_keys,
                    "method": method,
                    "score": "full_set",
                    "auroc": roc_auc_score(non_unanimous, full)
                    if non_unanimous.any() and (~non_unanimous).any()
                    else np.nan,
                }
            )
        out["auroc"].append(
            {
                **row_keys,
                "method": "none",
                "score": "one_minus_max_p",
                "auroc": roc_auc_score(non_unanimous, 1 - np.maximum(prob, 1 - prob)[known]),
            }
        )
        design = pd.get_dummies(
            pd.DataFrame(
                {
                    "agreement": agreement_bucket(pd.Series(test_confidence[known])).to_numpy(),
                    "domain": test["domain"].to_numpy()[known],
                }
            ),
            dtype=float,
        )
        design = design.drop(columns=[f"agreement_{AGREEMENT_REFERENCE}", f"domain_{DOMAIN_REFERENCE}"])
        design["log_words"] = np.log(test["n_words"].to_numpy()[known])
        for method in ["M0", "M1"]:
            referral = (sets[method].sum(axis=1) != 1)[known]
            if referral.all() or not referral.any():
                continue
            model = LogisticRegression(penalty=None, max_iter=2000).fit(design.to_numpy(), referral)
            out["regression"].append(
                {
                    **row_keys,
                    "method": method,
                    "intercept": model.intercept_[0],
                    **dict(zip(design.columns, model.coef_[0])),
                }
            )

        # (2) Unanimous only: calibration and test restricted to confidence = 1, main table; full data alongside.
        unanimous_calibration = calibration_positions[confidence[calibration_positions] == UNANIMOUS]
        unanimous_test = test_positions[confidence[test_positions] == UNANIMOUS]
        for subset, cal_pos, test_pos in [
            ("all_rows", calibration_positions, test_positions),
            ("unanimous", unanimous_calibration, unanimous_test),
        ]:
            sub_calibration = {c: f.iloc[cal_pos] for c, f in frames.items()}
            sub_mix = mix.iloc[cal_pos]
            cal_labels = sub_calibration["clean"]["label"].to_numpy()
            for condition in MAIN_CONDITIONS:
                sub_test = frames[condition].iloc[test_pos]
                for method in CP_METHODS:
                    calibration = calibration_frame(method, sub_calibration, sub_mix, condition)
                    method_sets = predict_sets(calibration, sub_test, method, ALPHA)[0]
                    out["unanimous"].append(
                        {
                            **row_keys,
                            "subset": subset,
                            "condition": condition,
                            "method": method,
                            "n_calibration": len(cal_pos),
                            "n_calibration_1": int((cal_labels == 1).sum()),
                            "n": len(test_pos),
                            "n_1": int((sub_test["label"] == 1).sum()),
                            **main_metrics(method_sets, sub_test),
                        }
                    )

        # (3) Subreddits: coverage under M0 / M1 against the subreddit's stress share (test fold).
        subreddits = extra["subreddit"].to_numpy()[test_positions]
        for subreddit in np.unique(subreddits):
            mask = subreddits == subreddit
            row = {**row_keys, "subreddit": subreddit, "n": int(mask.sum()), "stress_share": float(labels[mask].mean())}
            for method in ["M0", "M1"]:
                row[f"{method}_coverage"] = float(sets[method][mask][np.arange(mask.sum()), labels[mask]].mean())
            out["subreddit"].append(row)

        # (4) Thresholds of M0, M1 (clean calibration) and M3-mix (mix calibration) as p(stress) cutoffs.
        for method in ["M0", "M1", "M3-mix"]:
            calibration = calibration_frame(method, calibration_by_condition, mix.iloc[calibration_positions], "clean")
            out["thresholds"] += [{**row_keys, **r} for r in thresholds_as_probabilities(calibration, method)]

        # (6) Confident misses (primary model): stress posts with set exactly {not_stress} vs caught (stress in set).
        if task["model"] == PRIMARY:
            stress = labels == 1
            liwc = extra[[f"lex_liwc_{name}" for name in LIWC]].to_numpy()[test_positions][stress]
            for method in ["M0", "M1"]:
                codes = set_codes(sets[method])[stress]
                missed, caught = codes == "not_stress", sets[method][stress, RISK_CLASS]
                row = {**row_keys, "method": method, "n_missed": int(missed.sum()), "n_caught": int(caught.sum())}
                for index, name in enumerate(LIWC):
                    a, b = liwc[missed, index], liwc[caught, index]
                    pooled = (
                        np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
                        if len(a) > 1
                        else np.nan
                    )
                    row[f"smd_{name}"] = (a.mean() - b.mean()) / pooled if pooled and pooled > 0 else np.nan
                out["liwc"].append(row)

        # (7) Rows with missing agreement vs known agreement in the test fold.
        out["missing"] += missing_agreement_rows(row_keys, sets, labels, prob, known)

    # (7) also on the fixed official split (original calibration / official test).
    split = pool["split"].to_numpy()
    calibration, test = (
        frames["clean"].iloc[np.flatnonzero(split == "calibration")],
        frames["clean"].iloc[np.flatnonzero(split == "test")],
    )
    fixed_sets = {m: predict_sets(calibration, test, m, ALPHA)[0] for m in ["M0", "M1"]}
    out["missing"] += missing_agreement_rows(
        {**keys, "replicate": "official"},
        fixed_sets,
        test["label"].to_numpy(),
        test["prob_stress"].to_numpy(),
        ~np.isnan(confidence[split == "test"]),
    )

    print(f"[{timestamp()}] {task['model']} seed {task['seed']}: {time.perf_counter() - start:.0f} s", flush=True)
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


def pooled_summary(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=True)[values]
    return (
        pd.concat(
            {"median": grouped.median(), "q025": grouped.quantile(0.025), "q975": grouped.quantile(0.975)}, axis=1
        )
        .stack(level=1, future_stack=True)
        .rename_axis([*keys, "metric"])
        .reset_index()
    )


def mechanisms(config: dict) -> None:
    """
    Mechanism analyses on the resplits (agreement, unanimous rows, subreddits, score thresholds, occlusion,
    confident-miss LIWC, missing agreement) -> <output_dir>/mechanisms/*.csv and per-replicate tables in mechanisms/raw/.
    """

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    output_dir = Path(config["evaluation"]["output_dir"]) / "mechanisms"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_seed = int(config["seed"])

    pool = load_pool(data_dir)
    extra = pd.concat([pd.read_csv(data_dir / f"{name}.csv") for name in ["calibration", "test"]], ignore_index=True)
    assert np.array_equal(extra["id"].to_numpy(), pool["id"].to_numpy())
    extra = extra[["subreddit", *[f"lex_liwc_{name}" for name in LIWC]]]
    replicates = make_replicates(
        pool, int(config["evaluation"]["n_replicates"]), int(config["evaluation"]["resplit_folds"]), base_seed
    )["resplit"]
    common = {
        "pool": pool,
        "extra": extra,
        "replicates": replicates,
        "base_seed": base_seed,
        "conditions": config["conditions"],
        "mix_assignment": pd.read_csv(data_dir / "mix_assignment.csv"),
    }
    tasks = [
        {**common, "model": model, "seed": seed, "path": runs_dir / model / f"seed_{seed}" / "predictions.csv.gz"}
        for model, seeds in RUNS.items()
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        results = list(executor.map(mechanism_run, tasks))
    tables = {
        name: pd.concat([r[name] for r in results if not r[name].empty], ignore_index=True) for name in results[0]
    }
    # Per-replicate rows (gitignored), e.g. for seed-averaged aggregation in the paper tables.
    (output_dir / "raw").mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.astype({"seed": str, "replicate": str}).to_parquet(output_dir / "raw" / f"{name}.parquet", index=False)

    # (1) Agreement: bucket metrics (already computed, groups summary), AUROC and regression over resplits.
    groups = pd.read_csv(Path(config["evaluation"]["output_dir"]) / "pooled_groups.csv")
    buckets = groups[
        (groups["group_type"] == "agreement")
        & (groups["protocol"] == "resplit")
        & (groups["condition"] == "clean")
        & (groups["alpha"] == ALPHA)
        & groups["method"].isin(["M0", "M1"])
        & groups["model"].isin(list(RUNS))
    ]
    auroc = pooled_summary(tables["auroc"], ["model", "method", "score"], ["auroc"])
    regression_values = [c for c in tables["regression"].columns if c not in ["model", "seed", "replicate", "method"]]
    regression = pooled_summary(tables["regression"], ["model", "method"], regression_values)
    pd.concat(
        [
            buckets.assign(part="bucket_metrics_first_200_replicates"),
            auroc.assign(part="auroc_non_unanimous_label"),
            regression.assign(part="referral_regression_coefficients"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "agreement.csv", index=False)

    # (2) Unanimous only: main table, sizes and bands at the median sizes.
    metrics = [
        "coverage",
        "c0",
        "c1",
        "coverage_pi=0.1",
        "workload_per_1000_pi=0.1",
        "uncertainty_referrals_per_1000_pi=0.1",
        "worst_domain_coverage",
        "n_calibration",
        "n_calibration_1",
        "n",
        "n_1",
    ]
    unanimous = pooled_summary(tables["unanimous"], ["model", "subset", "condition", "method"], metrics)
    unanimous.to_csv(output_dir / "unanimous_only.csv", index=False)

    # (3) Subreddits: medians over resplits, and the reweighting line from the model's overall clean c0 / c1.
    subreddit = (
        tables["subreddit"]
        .groupby(["model", "subreddit"])
        .agg(
            n_median=("n", "median"),
            stress_share=("stress_share", "median"),
            M0_coverage=("M0_coverage", "median"),
            M1_coverage=("M1_coverage", "median"),
        )
        .reset_index()
    )
    overall = tables["unanimous"][
        (tables["unanimous"]["subset"] == "all_rows") & (tables["unanimous"]["condition"] == "clean")
    ]
    class_coverage = overall.groupby(["model", "method"])[["c0", "c1"]].median()
    for method in ["M0", "M1"]:
        c = class_coverage.xs(method, level="method")
        subreddit[f"{method}_line"] = subreddit["stress_share"] * subreddit["model"].map(c["c1"]) + (
            1 - subreddit["stress_share"]
        ) * subreddit["model"].map(c["c0"])
    subreddit.to_csv(output_dir / "subreddit_prevalence.csv", index=False)

    # (4) Mechanism: p(stress) histograms (stress and not-stress, pooled calibration + test, seeds pooled) and thresholds.
    histograms = []
    edges = np.linspace(0, 1, 51)
    for model, seeds in RUNS.items():
        predictions = pd.concat([pd.read_csv(runs_dir / model / f"seed_{s}" / "predictions.csv.gz") for s in seeds])
        predictions = predictions[predictions["split"].isin(["calibration", "test"])]
        for condition in MECHANISM_CONDITIONS:
            for label in [0, 1]:
                values = predictions[(predictions["condition"] == condition) & (predictions["label"] == label)][
                    "prob_stress"
                ].to_numpy()
                counts, _ = np.histogram(values, bins=edges)
                histograms += [
                    {
                        "model": model,
                        "condition": condition,
                        "label": label,
                        "bin_low": lo,
                        "bin_high": hi,
                        "share": n / len(values),
                    }
                    for lo, hi, n in zip(edges[:-1], edges[1:], counts)
                ]
    pd.DataFrame(histograms).to_csv(output_dir / "score_histograms.csv", index=False)
    thresholds = pooled_summary(
        tables["thresholds"],
        ["model", "method", "length_bucket"],
        ["stress_in_set_if_p_at_least", "not_stress_in_set_if_p_at_most"],
    )
    thresholds.to_csv(output_dir / "score_thresholds.csv", index=False)

    # (5) Occlusion (computed at training time): delta p(stress) by sentence position and class.
    occlusion = []
    for model in OCCLUSION_MODELS:
        for seed in RUNS[model]:
            frame = pd.read_csv(runs_dir / model / f"seed_{seed}" / "occlusion.csv")
            grouped = frame.groupby(["label", "position"])["delta_prob_stress"]
            occlusion.append(
                pd.concat(
                    [
                        grouped.size().rename("n"),
                        grouped.mean().rename("mean_delta"),
                        grouped.median().rename("median_delta"),
                        grouped.apply(lambda d: (d < 0).mean()).rename("share_negative"),
                    ],
                    axis=1,
                )
                .reset_index()
                .assign(model=model, seed=seed)
            )
    occlusion = pd.concat(occlusion, ignore_index=True)
    occlusion_summary = (
        occlusion.groupby(["model", "label", "position"])
        .agg(
            n=("n", "first"),
            mean_delta=("mean_delta", "mean"),
            mean_delta_sd_over_seeds=("mean_delta", "std"),
            median_delta=("median_delta", "mean"),
            share_negative=("share_negative", "mean"),
        )
        .reset_index()
    )
    occlusion_summary.to_csv(output_dir / "sentence_occlusion.csv", index=False)

    # (6) Confident misses vs caught stress posts: standardized LIWC differences over resplits (primary model).
    pooled_summary(tables["liwc"], ["model", "method"], ["n_missed", "n_caught", *[f"smd_{n}" for n in LIWC]]).to_csv(
        output_dir / "confident_miss_liwc.csv", index=False
    )

    # (7) Missing agreement: medians over resplits, plus the fixed official split (seed values, then their mean).
    missing = tables["missing"]
    resplit_part = pooled_summary(
        missing[missing["replicate"] != "official"],
        ["model", "rows"],
        ["n", "stress_share", "accuracy", "M0_coverage", "M1_coverage"],
    ).assign(part="resplits")
    official = (
        missing[missing["replicate"] == "official"]
        .groupby(["model", "rows"])[["n", "stress_share", "accuracy", "M0_coverage", "M1_coverage"]]
        .mean()
    )
    official_part = (
        official.stack()
        .rename("official_split_mean_over_seeds")
        .rename_axis(["model", "rows", "metric"])
        .reset_index()
        .assign(part="official_split")
    )
    pd.concat([resplit_part, official_part], ignore_index=True).to_csv(
        output_dir / "missing_agreement.csv", index=False
    )
