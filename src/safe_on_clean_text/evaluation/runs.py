from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from safe_on_clean_text.conformal.calibration import fit_temperature, temperature_margin
from safe_on_clean_text.conformal.methods import (
    METHODS,
    uncorrected_sets,
    temperature_refusal_sets,
    calibration_frame,
    jitter_seed,
    predict_sets,
)
from safe_on_clean_text.conformal.metrics import (
    INDICATORS,
    group_summaries,
    triage_frame,
    reference_band,
    reweight,
    summarize_indicators,
)
from safe_on_clean_text.conformal.scores import score_frame, logit_margin, true_class_scores
from safe_on_clean_text.evaluation.resplits import (
    ensemble_predictions,
    mix_for_conditions,
    pool_frames,
    available_conditions,
)
from safe_on_clean_text.provenance import timestamp

# Evaluation of one run (all replicates x conditions x methods x alphas) and the per-replicate table helpers;
# used by safe_on_clean_text.evaluation.pipeline and by the public reliability_report.
CLASS_INDICATORS = [name for name in INDICATORS if name != "covered"]
LONG_METRICS = (
    ["n", "n_0", "n_1", "coverage", "c0", "c1", "sensitivity"]
    + CLASS_INDICATORS
    + [f"{name}_{label}" for name in CLASS_INDICATORS for label in [0, 1]]
)
GROUP_METRICS = [
    "n",
    "n_0",
    "n_1",
    "n_calibration",
    "band_low",
    "band_high",
    "coverage",
    "c0",
    "c1",
    "set_size",
    "full",
    "empty",
    "workload",
    "uncertainty_referral",
    "sensitivity",
]
# Appendix tie-breaking variant (runs in evaluation.jitter_runs): every score-threshold method, as "<method>+jitter".
JITTER_METHODS = [*METHODS, "B1"]


def evaluate_run(task: dict[str, Any]) -> dict[str, Any]:
    """
    All replicates x conditions x methods x alphas of one run; writes its long and group tables (parquet).
    """

    start = time.perf_counter()
    model, seed = task["model"], task["seed"]
    if len(task["predictions_paths"]) > 1:
        predictions = ensemble_predictions(task["predictions_paths"])
    else:
        predictions = pd.read_csv(task["predictions_paths"][0])
    # LLM runs with four conditions use those conditions and the M3-mix draw over them.
    conditions = available_conditions(predictions, task["conditions"])
    mix_assignment = mix_for_conditions(
        task["mix_assignment"], task["pool"], conditions, task["conditions"], task["base_seed"]
    )
    frames, mix = pool_frames(predictions, task["pool"], mix_assignment, conditions)

    # B1 threshold and B2 temperature come from the clean validation split (outside the pool).
    validation = score_frame(predictions[predictions["split"] == "validation"], task["validation_metadata"])
    temperature = None
    if "B2" in task["methods"]:
        temperature = fit_temperature(temperature_margin(validation), validation["label"].to_numpy())

    long_rows: list[dict[str, Any]] = []
    group_tables: list[pd.DataFrame] = []

    for protocol, replicates in task["replicates"].items():
        for replicate, (calibration_positions, test_positions) in enumerate(replicates):
            with_groups = replicate < task["n_group_replicates"] and bool(task["group_types"])
            # Appendix tie-breaking: one generator per (protocol, replicate), consumed in a fixed loop order.
            jitter = (
                np.random.default_rng(jitter_seed(task["base_seed"], f"{protocol}/{replicate}"))
                if task["jitter"]
                else None
            )
            calibration_by_condition = {
                condition: frame.iloc[calibration_positions] for condition, frame in frames.items()
            }
            mix_calibration = mix.iloc[calibration_positions]

            for condition in conditions:
                test = frames[condition].iloc[test_positions]
                labels = test["label"].to_numpy()
                refusals: dict[float, int] = {}

                for method in task["methods"]:
                    if method in METHODS:
                        calibration = calibration_frame(method, calibration_by_condition, mix_calibration, condition)
                        calibration_name = METHODS[method].calibration
                    else:
                        # B1 / B2 are tuned on validation; sizes and group bands use the replicate's clean calibration.
                        calibration = calibration_by_condition["clean"]
                        calibration_name = "validation"
                    calibration_labels = calibration["label"].to_numpy()
                    keys = {
                        "dataset": task["dataset"],
                        "model": model,
                        "seed": seed,
                        "protocol": protocol,
                        "replicate": replicate,
                        "condition": condition,
                        "method": method,
                        "calibration": calibration_name,
                    }

                    for alpha in task["alphas"]:
                        if method in METHODS:
                            sets, n_infinite = predict_sets(calibration, test, method, alpha)
                        elif method == "B1":
                            sets, n_infinite = uncorrected_sets(validation, test, alpha), 0
                        else:
                            # B2 refuses exactly as many cases as M1 refers (full or empty sets) on this test.
                            # Sets from the unclipped margin; the temperature (fitted on the clipped margin) only rescales.
                            sets, n_infinite = (
                                temperature_refusal_sets(logit_margin(test), temperature, refusals[alpha]),
                                0,
                            )
                        if method == "M1":
                            refusals[alpha] = int((sets.sum(axis=1) != 1).sum())
                        variants = [(method, sets, n_infinite)]
                        if jitter is not None and method in JITTER_METHODS:
                            scale = task["tie_jitter"]
                            if method in METHODS:
                                jittered = predict_sets(
                                    calibration, test, method, alpha, jitter=jitter, jitter_scale=scale
                                )
                            else:
                                jittered = (
                                    uncorrected_sets(validation, test, alpha, jitter=jitter, jitter_scale=scale),
                                    0,
                                )
                            variants.append((f"{method}+jitter", *jittered))

                        for name, variant_sets, variant_infinite in variants:
                            indicators = triage_frame(variant_sets, labels)
                            summary = summarize_indicators(indicators, labels)
                            long_rows.append(
                                {
                                    **keys,
                                    "method": name,
                                    "alpha": alpha,
                                    "n_calibration": len(calibration_labels),
                                    "n_calibration_0": int((calibration_labels == 0).sum()),
                                    "n_calibration_1": int((calibration_labels == 1).sum()),
                                    "n_infinite_thresholds": variant_infinite,
                                    **{metric: summary[metric] for metric in LONG_METRICS},
                                }
                            )

                            if with_groups:
                                groups = group_summaries(indicators, test, calibration, alpha, task["group_types"])
                                group_tables.append(
                                    groups[["group_type", "group", *GROUP_METRICS]].assign(
                                        **{**keys, "method": name}, alpha=alpha
                                    )
                                )

            if (replicate + 1) % 100 == 0:
                print(f"[{timestamp()}] {model} seed {seed}: {protocol} {replicate + 1}/{len(replicates)}", flush=True)

    long_table = pd.DataFrame(long_rows)
    tables = [(long_table, task["long_path"])]
    if group_tables:
        tables.append((pd.concat(group_tables, ignore_index=True), task["groups_path"]))
    for table, path in tables:
        for column in table.select_dtypes(include=["object", "str"]).columns:
            table[column] = table[column].astype("category")
        table.to_parquet(path, index=False, compression="zstd")

    # Descriptive exchangeability diagnostic on the original split: true-class M0 scores, calibration vs test.
    clean = frames["clean"]
    is_calibration = task["pool"]["split"].to_numpy() == "calibration"
    scores = true_class_scores(clean)
    ks = ks_2samp(scores[is_calibration], scores[~is_calibration])

    return {
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "n_calibration": int(is_calibration.sum()),
        "n_test": int((~is_calibration).sum()),
        "long_rows": len(long_table),
        "group_rows": sum(len(t) for t in group_tables),
        "seconds": round(time.perf_counter() - start, 1),
    }


def add_band_flags(table: pd.DataFrame) -> pd.DataFrame:
    """
    Per replicate: is coverage (c0, c1) below / inside / above its reference band at that replicate's sizes.
    Sensitivity uses the class-1 band at the replicate's sizes for every method (also B1 / B2).
    Adds the columns in place.
    """

    out = table
    for metric, n_calibration, n_test in [
        ("coverage", "n_calibration", "n"),
        ("c0", "n_calibration_0", "n_0"),
        ("c1", "n_calibration_1", "n_1"),
        ("sensitivity", "n_calibration_1", "n_1"),
    ]:
        sizes = out[[n_calibration, n_test, "alpha"]].drop_duplicates()
        bands = {
            (a, b, c): reference_band(int(a), int(b), float(c)) for a, b, c in sizes.itertuples(index=False, name=None)
        }
        low_high = np.array([bands[key] for key in zip(out[n_calibration], out[n_test], out["alpha"])])
        out[f"{metric}_below_band"] = (out[metric] < low_high[:, 0]).astype(float)
        out[f"{metric}_above_band"] = (out[metric] > low_high[:, 1]).astype(float)
        out[f"{metric}_in_band"] = 1.0 - out[f"{metric}_below_band"] - out[f"{metric}_above_band"]
    return out


def add_prevalence_metrics(table: pd.DataFrame, prevalence_grid: list[float]) -> pd.DataFrame:
    out = table
    for prevalence in prevalence_grid:
        at_prevalence = reweight(out, prevalence)
        out[f"coverage_pi={prevalence}"] = at_prevalence["coverage"]
        out[f"workload_per_1000_pi={prevalence}"] = 1000 * at_prevalence["workload"]
        out[f"uncertainty_referrals_per_1000_pi={prevalence}"] = 1000 * at_prevalence["uncertainty_referral"]
    return out


def quantile_summary(table: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """
    Long summary: one row per (keys, metric) with the number of values, mean, median, 2.5% and 97.5% quantiles.
    Computed on the wide table (no melt), so memory stays proportional to the input.
    """

    grouped = table.groupby(keys, observed=True, sort=True)[metrics]
    quantiles = grouped.quantile([0.025, 0.5, 0.975])
    statistics = {
        "mean": grouped.mean(),
        "q025": quantiles.xs(0.025, level=-1),
        "median": quantiles.xs(0.5, level=-1),
        "q975": quantiles.xs(0.975, level=-1),
    }
    out = (
        pd.concat({name: frame.stack() for name, frame in statistics.items()}, axis=1)
        .rename_axis([*keys, "metric"])
        .reset_index()
    )
    sizes = grouped.size().rename("n_values").reset_index()
    out = out.merge(sizes, on=keys, how="left")
    return out[[*keys, "metric", "n_values", "mean", "median", "q025", "q975"]]
