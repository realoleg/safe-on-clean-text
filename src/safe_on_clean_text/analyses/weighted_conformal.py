from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safe_on_clean_text.analyses.resplit_scores import (
    in_study_order,
    load_run_scores,
    run_tasks,
    set_metrics,
    true_scores,
)
from safe_on_clean_text.conformal.metrics import reference_band
from safe_on_clean_text.conformal.scores import RISK_CLASS
from safe_on_clean_text.conformal.thresholds import weighted_thresholds
from safe_on_clean_text.degradations import apply_condition, assign_mix, load_condition_texts
from safe_on_clean_text.evaluation.aggregation import aggregate, main_alpha_rows
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS
from safe_on_clean_text.text import word_count

# H10 (criteria fixed before running): weighted split CP (Tibshirani et al., 2019) with density-ratio weights from a
# logistic-regression domain classifier fitted on the validation split only: source = validation posts degraded as for
# M3-mix (variant a; one assign_mix condition per post) or clean (variant b), target = the same posts in the test
# condition, so the classes are balanced and the odds are the density ratio. Weights use text features only (word count,
# share of lower-cased words outside the training vocabulary); shifts in model scores at fixed length and vocabulary are
# not corrected. Weights do not depend on the model. Variant a weights the M3-mix calibration set, variant b the clean one.
VARIANTS = {"a": "mix", "b": "clean"}
# Sensitivity: weights clipped at 1 / pi_k = 11, the bound on the true target / mixture density ratio when the target
# condition has weight 1/11 in the mixture (variant a, WCP-class only).
WEIGHT_CLIP = 11.0
WEIGHTED_METHODS = ["WCP", "WCP-class"]
REFERENCE_METHODS = ["M0", "M1", "M3-mix"]


def text_features(texts: pd.Series, vocabulary: set[str]) -> np.ndarray:
    """
    (n x 2): word count and the share of lower-cased words outside the vocabulary.
    """

    out_of_vocabulary = [
        np.mean([word.lower() not in vocabulary for word in text.split()]) if text.split() else 0.0 for text in texts
    ]
    return np.column_stack([word_count(texts).to_numpy(dtype=np.float64), np.asarray(out_of_vocabulary)])


def density_ratio_weights(config: dict, pool: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """
    Per (variant, test condition): weights of every pool row as a calibration row (text in its calibration condition:
    the M3-mix assignment for variant a, clean for b) and as a test row (text in the test condition).
    """

    base_seed, conditions = int(config["seed"]), config["conditions"]
    data_dir = Path(config["data"]["processed_dir"])
    vocabulary = {
        word.lower() for text in pd.read_csv(data_dir / "train.csv")["text"].astype(str) for word in text.split()
    }

    validation = pd.read_csv(data_dir / "validation.csv")
    validation_features = {c: text_features(apply_condition(validation, c, base_seed), vocabulary) for c in conditions}
    validation_mix = assign_mix(validation["id"], conditions, base_seed).to_numpy()
    validation_source = {
        "a": np.vstack([validation_features[c][i] for i, c in enumerate(validation_mix)]),
        "b": validation_features["clean"],
    }

    texts = load_condition_texts(data_dir / "conditions.csv")
    pool_features = {}
    for condition in conditions:
        part = texts[texts["condition"] == condition].set_index(["split", "id"])["text"]
        aligned = part.loc[list(zip(pool["split"], pool["id"]))].reset_index(drop=True)
        pool_features[condition] = text_features(aligned, vocabulary)
    mix = pd.read_csv(data_dir / "mix_assignment.csv")
    assert np.array_equal(mix["id"].to_numpy(), pool["id"].to_numpy())
    pool_mix = mix["condition"].to_numpy()
    pool_calibration = {
        "a": np.vstack([pool_features[c][i] for i, c in enumerate(pool_mix)]),
        "b": pool_features["clean"],
    }

    weights = {}
    for variant in VARIANTS:
        for condition in conditions:
            features = np.vstack([validation_source[variant], validation_features[condition]])
            domain = np.repeat([0, 1], len(validation))
            classifier = make_pipeline(StandardScaler(), LogisticRegression()).fit(features, domain)
            # Balanced classes: the odds P(target | x) / P(source | x) are the density ratio target / source.
            weights[(variant, condition)] = (
                np.exp(classifier.decision_function(pool_calibration[variant])),
                np.exp(classifier.decision_function(pool_features[condition])),
            )
    return weights


def run(task: dict[str, Any]) -> pd.DataFrame:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    labels = run_scores.labels
    calibration_scores = {"a": run_scores.mix, "b": run_scores.scores["clean"]}
    rows = []
    for replicate, (calibration, test) in enumerate(task["replicates"]):
        calibration_labels, test_labels = labels[calibration], labels[test]
        for variant in [v for v in VARIANTS if (v, "clean") in task["weights"]]:
            true = true_scores(calibration_scores[variant][calibration], calibration_labels)
            for condition in task["conditions"]:
                calibration_weights, test_weights = task["weights"][(variant, condition)]
                calibration_weights, test_weights = calibration_weights[calibration], test_weights[test]
                test_scores = run_scores.scores[condition][test]
                marginal = weighted_thresholds(true, calibration_weights, test_weights, ALPHA)
                by_class = np.column_stack(
                    [
                        weighted_thresholds(
                            true[calibration_labels == y],
                            calibration_weights[calibration_labels == y],
                            test_weights,
                            ALPHA,
                        )
                        for y in [0, 1]
                    ]
                )
                all_weights = np.concatenate([calibration_weights, test_weights])
                risk_weights = calibration_weights[calibration_labels == RISK_CLASS]
                shared = {
                    "model": task["model"],
                    "seed": run_scores.seed,
                    "replicate": replicate,
                    "variant": variant,
                    "condition": condition,
                    "n_calibration_1": int((calibration_labels == RISK_CLASS).sum()),
                    "n_1": int((test_labels == RISK_CLASS).sum()),
                    "ess": calibration_weights.sum() ** 2 / (calibration_weights**2).sum(),
                    "ess_risk": risk_weights.sum() ** 2 / (risk_weights**2).sum(),
                    "max_weight_ratio": all_weights.max() / all_weights.min(),
                }
                for method, q in [("WCP", marginal[:, None]), ("WCP-class", by_class)]:
                    point = set_metrics(test_scores <= q, test_labels)
                    rows.append(
                        {
                            **shared,
                            "method": method,
                            "c1": point["c1"],
                            "workload_per_1000": point["workload_per_1000"],
                            "full": point["full"],
                            "infinite_threshold_share": float(np.isinf(q).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def weighted_conformal_table(config: dict) -> pd.DataFrame:
    """
    H10 on Dreaddit (all 7 models, 1,000 resplits): per model, variant (a: mix source on the M3-mix calibration set;
    b: clean source on the clean calibration set), method (WCP marginal, WCP-class class-conditional) and condition,
    c1, workload per 1,000 at pi = 0.10, share of full sets, share of infinite thresholds, ESS (all calibration rows and
    risk rows) and the maximum weight ratio; M0, M1 and M3-mix from the long tables next to them (variant "-"); c1 band
    at the median risk-class calibration and test sizes. Primary aggregation.
    """

    tasks = run_tasks(config, DREADDIT_MODELS)
    weights = density_ratio_weights(config, tasks[0]["pool"])
    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        rows = pd.concat(executor.map(run, [{**task, "weights": weights} for task in tasks]), ignore_index=True)
    keys = ["model", "variant", "method", "condition"]
    table = aggregate(rows, keys, metrics=[c for c in rows.columns if c not in [*keys, "seed", "replicate"]])

    references = []
    for model in DREADDIT_MODELS:
        long = main_alpha_rows(config, model)
        long = long[long["method"].isin(REFERENCE_METHODS)].rename(
            columns={"workload_per_1000_pi=0.1": "workload_per_1000"}
        )
        references.append(
            aggregate(
                long.assign(model=model, variant="-"),
                keys,
                metrics=["c1", "workload_per_1000", "full", "n_calibration_1", "n_1"],
            )
        )
    table = pd.concat([table, *references], ignore_index=True)

    sizes = table[table["metric"].isin(["n_calibration_1", "n_1"])].pivot_table(
        index=keys, columns="metric", values="median"
    )
    bands = pd.DataFrame(
        [
            reference_band(int(round(n_cal)), int(round(n_test)), ALPHA)
            for n_cal, n_test in sizes[["n_calibration_1", "n_1"]].to_numpy()
        ],
        index=sizes.index,
        columns=["c1_band_low", "c1_band_high"],
    ).reset_index()
    table = table.merge(bands, on=keys, how="left")
    order = {
        "variant": {"a": 0, "b": 1, "-": 2},
        "method": {m: i for i, m in enumerate(WEIGHTED_METHODS + REFERENCE_METHODS)},
    }
    table = table.sort_values(["variant", "method"], key=lambda s: s.map(order[s.name]), kind="stable")
    return in_study_order(table, config["conditions"])


def clipped_weights_table(config: dict, unclipped: pd.DataFrame) -> pd.DataFrame:
    """
    DESCRIPTIVE sensitivity of H10 (the verdict stays as fixed): variant a, WCP-class, with calibration and test weights
    clipped at WEIGHT_CLIP; per model and condition c1, workload per 1,000, ESS (all / risk rows), full and infinite-
    threshold shares next to the unclipped rows of weighted_conformal_table (column weights), and per condition the
    shares of calibration and test weights above the clip (model "all", the weights do not depend on the model).
    """

    tasks = run_tasks(config, DREADDIT_MODELS)
    weights = density_ratio_weights(config, tasks[0]["pool"])
    clipped = {
        key: (np.minimum(calibration, WEIGHT_CLIP), np.minimum(test, WEIGHT_CLIP))
        for key, (calibration, test) in weights.items()
        if key[0] == "a"
    }
    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        rows = pd.concat(executor.map(run, [{**task, "weights": clipped} for task in tasks]), ignore_index=True)
    rows = rows[rows["method"] == "WCP-class"]
    metrics = ["c1", "workload_per_1000", "ess", "ess_risk", "full", "infinite_threshold_share", "max_weight_ratio"]
    keys = ["model", "variant", "method", "condition"]
    table = aggregate(rows, keys, metrics=metrics).assign(weights=f"clipped at {WEIGHT_CLIP:g}")
    bands = unclipped[["model", "condition", "c1_band_low", "c1_band_high"]].drop_duplicates(["model", "condition"])
    table = table.merge(bands, on=["model", "condition"], how="left")
    reference = unclipped[
        (unclipped["variant"] == "a") & (unclipped["method"] == "WCP-class") & unclipped["metric"].isin(metrics)
    ].assign(weights="unclipped")

    shares = []
    for replicate, (calibration, test) in enumerate(tasks[0]["replicates"]):
        for condition in config["conditions"]:
            calibration_weights, test_weights = weights[("a", condition)]
            shares.append(
                {
                    "model": "all",
                    "seed": "all",
                    "replicate": replicate,
                    "condition": condition,
                    "clipped_share_calibration": float(np.mean(calibration_weights[calibration] > WEIGHT_CLIP)),
                    "clipped_share_test": float(np.mean(test_weights[test] > WEIGHT_CLIP)),
                }
            )
    shares = aggregate(pd.DataFrame(shares), ["model", "condition"]).assign(
        variant="a", method="WCP-class", weights=f"clipped at {WEIGHT_CLIP:g}"
    )
    table = pd.concat([shares, table, reference], ignore_index=True)
    order = {"weights": {"unclipped": 0, f"clipped at {WEIGHT_CLIP:g}": 1}}
    table = table.sort_values("weights", key=lambda s: s.map(order[s.name]), kind="stable")
    table = in_study_order(table, config["conditions"])  # model "all" (clipped shares) sorts last
    first = ["weights", "model", "variant", "method", "condition", "metric"]
    return table[[*first, *[c for c in table.columns if c not in first]]].reset_index(drop=True)
