from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.analyses.resplit_scores import load_run_scores, run_tasks, set_metrics, true_scores
from safe_on_clean_text.analyses.workload_matched import thresholds
from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.conformal.metrics import reference_band
from safe_on_clean_text.conformal.scores import RISK_CLASS
from safe_on_clean_text.conformal.thresholds import weighted_thresholds
from safe_on_clean_text.degradations import length_bucket, load_condition_texts, make_stream
from safe_on_clean_text.evaluation.aggregation import aggregate, main_alpha_rows
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS

# H11-H13 (criteria fixed before running): naturally short Dreaddit excerpts as a natural counterpart of truncation.
# Length = clean word count; long = at least the pool median, short = at most the pool lower quartile (pool =
# calibration + test). K = 50 was chosen mechanically (median truncated length of long posts closest to the median
# length of short posts). Descriptive additions: paired differences, synthetic head_K / tail_K on long test posts, band
# mass, domain reweighting of the long calibration posts, and coverage on short posts under standard calibration.
K = 50
SYNTHETIC = [f"head_{K}", f"tail_{K}"]
METHODS = ["M0", "M1", "M3-mix"]
IDENTITY_TOLERANCE = 1e-12
# Secondary analysis on excerpts of at most VERY_SHORT words only with at least MIN_VERY_SHORT_RISK risk posts per test half.
VERY_SHORT, MIN_VERY_SHORT_RISK = 40, 30


def length_groups(n_words: np.ndarray) -> np.ndarray:
    return np.where(n_words >= np.median(n_words), "long", np.where(n_words <= np.quantile(n_words, 0.25), "short", ""))


def score_frame(scores: np.ndarray, labels: np.ndarray, n_words: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score_0": scores[:, 0],
            "score_1": scores[:, 1],
            "label": labels,
            "length_bucket": length_bucket(pd.Series(n_words)),
        }
    )


def weighted_sets(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_groups: np.ndarray,
    calibration_weights: np.ndarray,
    test_scores: np.ndarray,
    test_groups: np.ndarray,
    test_weights: np.ndarray,
    by_class: bool,
) -> np.ndarray:
    """
    Weighted split-CP sets with thresholds per (group, class) (by_class) or per group; groups without calibration rows
    get +inf.
    """

    true = true_scores(calibration_scores, calibration_labels)
    limits = np.full(test_scores.shape, np.inf)
    for group in np.unique(test_groups):
        rows, in_group = test_groups == group, calibration_groups == group
        if not by_class:
            q = weighted_thresholds(true[in_group], calibration_weights[in_group], test_weights[rows], ALPHA)
            limits[rows, :] = q[:, None]
            continue
        for label in [0, 1]:
            selected = in_group & (calibration_labels == label)
            limits[rows, label] = weighted_thresholds(
                true[selected], calibration_weights[selected], test_weights[rows], ALPHA
            )
    return test_scores <= limits


def run(task: dict[str, Any]) -> pd.DataFrame:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    labels, scores, n_words = run_scores.labels, run_scores.scores, task["n_words"]
    long, short = task["length_group"] == "long", task["length_group"] == "short"
    domains, posts = task["pool"]["domain"].to_numpy(), task["pool"]["group"].to_numpy()
    rows: list[dict[str, Any]] = []

    for replicate, (calibration, test) in enumerate(task["replicates"]):
        keys = {"model": task["model"], "seed": run_scores.seed, "replicate": replicate}

        def record(analysis: str, test_set: str, method: str, sets: np.ndarray, positions: np.ndarray, n_cal_1: int):
            point = set_metrics(sets, labels[positions])
            rows.append(
                {
                    **keys,
                    "analysis": analysis,
                    "test": test_set,
                    "method": method,
                    "c1": point["c1"],
                    "workload_per_1000": point["workload_per_1000"],
                    "n_calibration_1": n_cal_1,
                    "n_test_1": int((labels[positions] == RISK_CLASS).sum()),
                }
            )

        def three_methods(calibration_rows: np.ndarray, test_rows: np.ndarray, condition: str) -> dict[str, np.ndarray]:
            clean, calibration_labels = scores["clean"][calibration_rows], labels[calibration_rows]
            test_scores = scores[condition][test_rows]
            mix = predict_sets(
                score_frame(run_scores.mix[calibration_rows], calibration_labels, n_words["mix"][calibration_rows]),
                score_frame(test_scores, labels[test_rows], n_words[condition][test_rows]),
                "M3-mix",
                ALPHA,
            )[0]
            return {
                "M0": test_scores <= thresholds(clean, calibration_labels, "M0", ALPHA),
                "M1": test_scores <= thresholds(clean, calibration_labels, "M1", ALPHA),
                "M3-mix": mix,
            }

        long_calibration, short_test, long_test = calibration[long[calibration]], test[short[test]], test[long[test]]
        n_long_1 = int((labels[long_calibration] == RISK_CLASS).sum())

        # H11 / H13: calibration on long posts; natural short test posts and synthetic analogues on long test posts.
        for test_set, test_rows, condition in [
            ("short clean", short_test, "clean"),
            ("long clean", long_test, "clean"),
            *[(f"long {c}", long_test, c) for c in SYNTHETIC],
        ]:
            for method, sets in three_methods(long_calibration, test_rows, condition).items():
                record("long calibration", test_set, method, sets, test_rows, n_long_1)

        # Secondary: standard calibration on all lengths; coverage on the short test posts (and the identity check on
        # the whole clean test half).
        n_all_1 = int((labels[calibration] == RISK_CLASS).sum())
        for test_set, test_rows in [("short clean", short_test), ("all clean", test)]:
            for method, sets in three_methods(calibration, test_rows, "clean").items():
                record("standard calibration", test_set, method, sets, test_rows, n_all_1)
        very_short = test[task["n_words"]["clean"][test] <= VERY_SHORT]
        if int((labels[very_short] == RISK_CLASS).sum()) >= MIN_VERY_SHORT_RISK:
            for method, sets in three_methods(calibration, very_short, "clean").items():
                record("standard calibration", f"<= {VERY_SHORT} words clean", method, sets, very_short, n_all_1)

        # Sensitivity: long calibration posts reweighted by domain to the short test posts' domain shares (label-free).
        short_share = pd.Series(domains[short_test]).value_counts(normalize=True)
        long_share = pd.Series(domains[long_calibration]).value_counts(normalize=True)
        ratio = (short_share / long_share).fillna(0.0)
        calibration_weights = ratio.reindex(domains[long_calibration]).fillna(0.0).to_numpy()
        test_weights = ratio.reindex(domains[short_test]).fillna(0.0).to_numpy()
        clean_long, calibration_labels = scores["clean"][long_calibration], labels[long_calibration]
        test_scores = scores["clean"][short_test]
        no_groups_calibration, no_groups_test = np.zeros(len(long_calibration)), np.zeros(len(short_test))
        mix_buckets = length_bucket(pd.Series(n_words["mix"][long_calibration])).to_numpy()
        test_buckets = length_bucket(pd.Series(n_words["clean"][short_test])).to_numpy()
        reweighted = {
            "M0": weighted_sets(
                clean_long,
                calibration_labels,
                no_groups_calibration,
                calibration_weights,
                test_scores,
                no_groups_test,
                test_weights,
                False,
            ),
            "M1": weighted_sets(
                clean_long,
                calibration_labels,
                no_groups_calibration,
                calibration_weights,
                test_scores,
                no_groups_test,
                test_weights,
                True,
            ),
            "M3-mix": weighted_sets(
                run_scores.mix[long_calibration],
                calibration_labels,
                mix_buckets,
                calibration_weights,
                test_scores,
                test_buckets,
                test_weights,
                True,
            ),
        }
        ess = calibration_weights.sum() ** 2 / (calibration_weights**2).sum()
        for method, sets in reweighted.items():
            record("long calibration, domain-reweighted", "short clean", method, sets, short_test, n_long_1)
            rows[-1]["domain_weight_ess"] = ess

        # H12: cross-fitted prediction from synthetic truncation of the long calibration posts (no test labels).
        groups = np.unique(posts[long_calibration])
        make_stream(task["base_seed"], replicate, "natural_shift_split").shuffle(groups)
        in_a = np.isin(posts[long_calibration], groups[: len(groups) // 2])
        head = make_stream(task["base_seed"], replicate, "natural_shift_truncation").random(len(long_calibration)) < 0.5
        truncated = np.where(
            head[:, None], scores[SYNTHETIC[0]][long_calibration], scores[SYNTHETIC[1]][long_calibration]
        )
        for method in ["M0", "M1"]:
            folds = []
            for fit, predict in [(in_a, ~in_a), (~in_a, in_a)]:
                q = thresholds(clean_long[fit], calibration_labels[fit], method, ALPHA)
                risk = predict & (calibration_labels == RISK_CLASS)
                folds.append(float(np.mean(truncated[risk, RISK_CLASS] <= q[RISK_CLASS])))
            rows.append(
                {
                    **keys,
                    "analysis": "cross-fitted prediction",
                    "test": f"long calibration posts, {SYNTHETIC[0]} / {SYNTHETIC[1]}",
                    "method": method,
                    "c1": float(np.mean(folds)),
                }
            )
    return pd.DataFrame(rows)


def check_against_long_tables(config: dict, rows: pd.DataFrame) -> float:
    """
    Largest |difference| between the standard-calibration c1 on the whole clean test half and the stored long tables.
    """

    worst = 0.0
    part = rows[(rows["analysis"] == "standard calibration") & (rows["test"] == "all clean")]
    for model, frame in part.groupby("model"):
        long = main_alpha_rows(config, model)
        long = long[(long["condition"] == "clean") & long["method"].isin(METHODS)].set_index(
            ["seed", "replicate", "method"]
        )
        joined = frame.set_index(["seed", "replicate", "method"]).join(long[["c1"]], rsuffix="_stored", how="left")
        assert len(joined) == len(frame) and joined["c1_stored"].notna().all(), model
        worst = max(worst, float((joined["c1"] - joined["c1_stored"]).abs().max()))
    return worst


def paired_differences(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Per (model, seed, replicate): c1(M1) - c1(M0) on short posts, natural short minus synthetic long c1 per method,
    observed and predicted band mass c1(M0) - c1(M1), and the prediction error predicted - observed c1(M1).
    """

    c1 = rows.pivot_table(index=["model", "seed", "replicate"], columns=["analysis", "test", "method"], values="c1")
    short = ("long calibration", "short clean")
    predicted = ("cross-fitted prediction", f"long calibration posts, {SYNTHETIC[0]} / {SYNTHETIC[1]}")
    out = {
        "short: c1(M1) - c1(M0)": c1[(*short, "M1")] - c1[(*short, "M0")],
        "observed band mass c1(M0) - c1(M1)": c1[(*short, "M0")] - c1[(*short, "M1")],
        "predicted band mass c1(M0) - c1(M1)": c1[(*predicted, "M0")] - c1[(*predicted, "M1")],
        "prediction error c1(M1): predicted - observed": c1[(*predicted, "M1")] - c1[(*short, "M1")],
    }
    for condition in ["clean", *SYNTHETIC]:
        for method in METHODS:
            out[f"{method}: short c1 - long {condition} c1"] = (
                c1[(*short, method)] - c1[("long calibration", f"long {condition}", method)]
            )
    return pd.DataFrame(out).rename_axis(columns="quantity").stack().rename("value").reset_index()


def descriptives(pool: pd.DataFrame, n_words: np.ndarray, groups: np.ndarray, replicates: list) -> pd.DataFrame:
    rows = [("pool", "", "rows", len(pool))]
    for q in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        rows.append(("pool", "", f"word count quantile {q:g}", float(np.quantile(n_words, q))))
    labels = pool["label"].to_numpy()
    for group in ["long", "short"]:
        members = groups == group
        rows += [
            (group, "", "rows", int(members.sum())),
            (group, "", "median words", float(np.median(n_words[members]))),
            (group, "", "risk share", float(labels[members].mean())),
        ]
        rows += [
            (group, domain, "domain share", float(share))
            for domain, share in pd.Series(pool["domain"][members]).value_counts(normalize=True).sort_index().items()
        ]
    per_half = {
        "short risk posts per test half": [
            int(((groups[t] == "short") & (labels[t] == RISK_CLASS)).sum()) for _, t in replicates
        ],
        "short posts per test half": [int((groups[t] == "short").sum()) for _, t in replicates],
        "long risk posts per calibration half": [
            int(((groups[c] == "long") & (labels[c] == RISK_CLASS)).sum()) for c, _ in replicates
        ],
        f"<= {VERY_SHORT}-word risk posts per test half": [
            int(((n_words[t] <= VERY_SHORT) & (labels[t] == RISK_CLASS)).sum()) for _, t in replicates
        ],
    }
    for name, values in per_half.items():
        rows += [
            ("resplits", "", f"{name}: {s}", float(f(values)))
            for s, f in [
                ("median", np.median),
                ("q025", lambda v: np.quantile(v, 0.025)),
                ("q975", lambda v: np.quantile(v, 0.975)),
            ]
        ]
    very_short = n_words <= VERY_SHORT
    rows += [
        ("pool", "", f"<= {VERY_SHORT} words", int(very_short.sum())),
        ("pool", "", f"<= {VERY_SHORT} words, risk", int(labels[very_short].sum())),
    ]
    return pd.DataFrame(rows, columns=["group", "domain", "quantity", "value"])


def natural_shift_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    H11-H13 on Dreaddit (all 7 models, 1,000 resplits, alpha 0.10, workload at pi = 0.10): the descriptives, the
    aggregated c1 / workload per (analysis, test set, method) with the c1 reference band at the median risk counts, and
    the aggregated paired differences. Primary aggregation.
    """

    tasks = run_tasks(config, DREADDIT_MODELS)
    pool = tasks[0]["pool"]
    data_dir = Path(config["data"]["processed_dir"])
    texts = load_condition_texts(data_dir / "conditions.csv").set_index(["split", "condition", "id"])["n_words"]
    position = list(zip(pool["split"], pool["id"]))
    n_words = {
        condition: texts.loc[[(split, condition, i) for split, i in position]].to_numpy()
        for condition in ["clean", *SYNTHETIC]
    }
    mix = tasks[0]["mix_assignment"]["condition"].to_numpy()
    n_words["mix"] = texts.loc[[(split, condition, i) for (split, i), condition in zip(position, mix)]].to_numpy()
    groups = length_groups(n_words["clean"])

    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        rows = pd.concat(
            executor.map(run, [{**task, "n_words": n_words, "length_group": groups} for task in tasks]),
            ignore_index=True,
        )
    identity = check_against_long_tables(config, rows)
    assert identity <= IDENTITY_TOLERANCE, identity
    print(f"H11-H13: identity vs long tables (standard calibration, clean), max |difference| {identity:.1e}")

    keys = ["model", "analysis", "test", "method"]
    table = aggregate(
        rows, keys, metrics=["c1", "workload_per_1000", "n_calibration_1", "n_test_1", "domain_weight_ess"]
    )
    table = table.dropna(subset=["median"])
    sizes = table[table["metric"].isin(["n_calibration_1", "n_test_1"])].pivot_table(
        index=keys, columns="metric", values="median"
    )
    bands = pd.DataFrame(
        [
            reference_band(int(round(a)), int(round(b)), ALPHA)
            for a, b in sizes[["n_calibration_1", "n_test_1"]].to_numpy()
        ],
        index=sizes.index,
        columns=["c1_band_low", "c1_band_high"],
    ).reset_index()
    table = table.merge(bands, on=keys, how="left")
    difference_table = aggregate(paired_differences(rows), ["model", "quantity"], metrics=["value"]).drop(
        columns="metric"
    )
    return (
        descriptives(pool, n_words["clean"], groups, tasks[0]["replicates"]),
        in_model_order(table),
        in_model_order(difference_table),
    )


def in_model_order(table: pd.DataFrame) -> pd.DataFrame:
    order = {model: i for i, model in enumerate(DREADDIT_MODELS)}
    return table.sort_values("model", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
