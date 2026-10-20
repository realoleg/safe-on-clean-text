from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.conformal.metrics import (
    INDICATORS,
    triage_frame,
    triage_indicators,
    reweight,
    summarize_indicators,
)
from safe_on_clean_text.conformal.scores import RISK_CLASS, true_class_scores
from safe_on_clean_text.conformal.thresholds import label_shift_thresholds
from safe_on_clean_text.degradations import make_stream
from safe_on_clean_text.evaluation.resplits import ensemble_predictions, load_pool, make_replicates, pool_frames
from safe_on_clean_text.provenance import timestamp

# Scenario "original": calibration at the replicate's own prevalence (M0, M1), metrics reweighted to pi analytically.
# Scenario "label_shift": M0 with the exact label-shift CP at the target pi (Podkopaev & Ramdas, 2021).
SCENARIOS = [("M0", "original"), ("M1", "original"), ("M0", "label_shift")]
CHECK_METRICS = ["coverage", "workload", "uncertainty_referral"]
METRIC_COLUMNS = ["c0", "c1", "coverage", "workload_per_1000", "uncertainty_referrals_per_1000"]


def label_shift_sets(calibration: pd.DataFrame, test: pd.DataFrame, prevalence: float, alpha: float) -> np.ndarray:
    thresholds = label_shift_thresholds(
        true_class_scores(calibration), calibration["label"].to_numpy(), prevalence, alpha
    )
    scores = test[["score_0", "score_1"]].to_numpy(dtype=np.float64)
    return scores <= np.array(thresholds)[None, :]


def prevalence_run(task: dict[str, Any]) -> dict[str, Any]:
    """
    Clean resplits of one run: class metrics of M0 / M1 (original calibration) and of the label-shift M0 at every pi,
    reweighted to pi; on the first check replicates, one stratified bootstrap test sample per (scenario, pi, alpha).
    """

    start = time.perf_counter()
    paths = task["predictions_paths"]
    predictions = ensemble_predictions(paths) if len(paths) > 1 else pd.read_csv(paths[0])
    clean_only = pd.DataFrame({"id": task["pool"]["id"], "condition": "clean"})
    frames, _ = pool_frames(predictions[predictions["condition"] == "clean"], task["pool"], clean_only, ["clean"])
    frame = frames["clean"]

    rows: list[dict[str, Any]] = []
    checks: dict[tuple, list[np.ndarray]] = {}
    for replicate, (calibration_positions, test_positions) in enumerate(task["replicates"]):
        calibration, test = frame.iloc[calibration_positions], frame.iloc[test_positions]
        labels = test["label"].to_numpy()
        with_check = replicate < task["check_replicates"]
        rng = make_stream(task["base_seed"], replicate, "prevalence_check") if with_check else None
        class_index = {label: np.flatnonzero(labels == label) for label in [0, 1]}

        for alpha in task["alphas"]:
            original = {method: predict_sets(calibration, test, method, alpha)[0] for method in ["M0", "M1"]}
            for prevalence in task["prevalence_grid"]:
                for method, scenario in SCENARIOS:
                    sets = (
                        original[method]
                        if scenario == "original"
                        else label_shift_sets(calibration, test, prevalence, alpha)
                    )
                    summary = summarize_indicators(triage_frame(sets, labels), labels)
                    at_prevalence = reweight(summary, prevalence)
                    rows.append(
                        {
                            "model": task["model"],
                            "seed": task["seed"],
                            "replicate": replicate,
                            "alpha": alpha,
                            "method": method,
                            "scenario": scenario,
                            "prevalence": prevalence,
                            "c0": summary["c0"],
                            "c1": summary["c1"],
                            "coverage": at_prevalence["coverage"],
                            "workload_per_1000": 1000 * at_prevalence["workload"],
                            "uncertainty_referrals_per_1000": 1000 * at_prevalence["uncertainty_referral"],
                        }
                    )
                    if with_check:
                        # Stratified bootstrap test samples of the original size with n1 = round(pi * m) risk cases;
                        # the replicate's bootstrap value is the mean over check_draws samples.
                        n_risk = round(prevalence * len(labels))
                        matrix = triage_indicators(sets, labels)
                        draws = []
                        for _ in range(task["check_draws"]):
                            sample = np.concatenate(
                                [
                                    rng.choice(class_index[RISK_CLASS], size=n_risk),
                                    rng.choice(class_index[1 - RISK_CLASS], size=len(labels) - n_risk),
                                ]
                            )
                            draws.append(
                                [
                                    matrix[sample, INDICATORS.index(name if name != "coverage" else "covered")].mean()
                                    for name in CHECK_METRICS
                                ]
                            )
                        bootstrap = np.mean(draws, axis=0)
                        analytic = [at_prevalence[name] for name in CHECK_METRICS]
                        checks.setdefault((method, scenario, alpha, prevalence), []).append(
                            np.array([bootstrap, analytic])
                        )

    check_rows = []
    for (method, scenario, alpha, prevalence), values in checks.items():
        values = np.array(values)  # replicates x (bootstrap, analytic) x metrics
        mean = values.mean(axis=0)
        differences = values[:, 0, :] - values[:, 1, :]
        for index, name in enumerate(CHECK_METRICS):
            # Monte Carlo standard error of the mean difference over replicates.
            standard_error = differences[:, index].std(ddof=1) / np.sqrt(len(values))
            check_rows.append(
                {
                    "model": task["model"],
                    "seed": task["seed"],
                    "method": method,
                    "scenario": scenario,
                    "alpha": alpha,
                    "prevalence": prevalence,
                    "metric": name,
                    "n_replicates": len(values),
                    "bootstrap_mean": mean[0, index],
                    "analytic_mean": mean[1, index],
                    "abs_difference": abs(mean[0, index] - mean[1, index]),
                    "monte_carlo_se": standard_error,
                    "z": (mean[0, index] - mean[1, index]) / standard_error if standard_error > 0 else 0.0,
                }
            )
    print(f"[{timestamp()}] {task['model']} seed {task['seed']}: {time.perf_counter() - start:.0f} s", flush=True)
    return {"rows": pd.DataFrame(rows), "checks": pd.DataFrame(check_rows)}


def prevalence_shift(config: dict) -> None:
    """
    Prevalence shift: per-class metrics reweighted to every pi of the grid, with M0 calibrated at the original pi and
    label-shift M0 (M0-LS) -> prevalence_long.parquet, prevalence_summary.csv; bootstrap check of the reweighting ->
    <checks_dir>/prevalence_bootstrap.csv.
    """

    base_seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    evaluation = config["evaluation"]
    output_dir = Path(evaluation["output_dir"])
    tolerance = float(config["prevalence"]["check_tolerance"])

    pool = load_pool(data_dir)
    replicates = make_replicates(pool, int(evaluation["n_replicates"]), int(evaluation["resplit_folds"]), base_seed)[
        "resplit"
    ]
    common = {
        "pool": pool,
        "replicates": replicates,
        "base_seed": base_seed,
        "alphas": [float(a) for a in config["conformal"]["alpha_values"]],
        "prevalence_grid": [float(p) for p in config["conformal"]["prevalence_grid"]],
        "check_replicates": int(config["prevalence"]["check_replicates"]),
        "check_draws": int(config["prevalence"]["check_draws"]),
    }
    runs_by_model: dict[str, list[Path]] = {}
    for path in sorted(runs_dir.glob("*/seed_*/predictions.csv.gz")):
        if "prob_stress" in pd.read_csv(path, nrows=1).columns:
            runs_by_model.setdefault(path.parent.parent.name, []).append(path)
    tasks = []
    for model, paths in runs_by_model.items():
        tasks += [
            {**common, "model": model, "seed": path.parent.name.removeprefix("seed_"), "predictions_paths": [path]}
            for path in paths
        ]
        if len(paths) > 1:
            tasks.append({**common, "model": f"{model}_ensemble", "seed": "ensemble", "predictions_paths": paths})

    with ProcessPoolExecutor(max_workers=int(evaluation["workers"])) as executor:
        results = list(executor.map(prevalence_run, tasks))
    rows = pd.concat([result["rows"] for result in results], ignore_index=True)
    checks = pd.concat([result["checks"] for result in results], ignore_index=True)
    checks["ok"] = checks["abs_difference"] <= tolerance
    # Per-replicate rows (gitignored), e.g. for seed-averaged aggregation in the paper.
    rows.to_parquet(output_dir / "prevalence_long.parquet", index=False)

    # Summary pooled over seeds (median and 95% interval over all seed x replicate values).
    keys = ["model", "method", "scenario", "alpha", "prevalence"]
    grouped = rows.groupby(keys, sort=True)[METRIC_COLUMNS]
    summary = (
        pd.concat(
            {
                "mean": grouped.mean(),
                "median": grouped.median(),
                "q025": grouped.quantile(0.025),
                "q975": grouped.quantile(0.975),
            },
            axis=1,
        )
        .stack(level=1, future_stack=True)
        .rename_axis([*keys, "metric"])
        .reset_index()
    )
    summary.insert(0, "dataset", dataset)
    summary.to_csv(output_dir / "prevalence_summary.csv", index=False)
    checks_dir = Path(config["paths"]["checks_dir"])
    checks_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(checks_dir / "prevalence_bootstrap.csv", index=False)

    # Reported, not raised (see monte_carlo_se, z): with one draw per replicate the Monte Carlo error of each comparison
    # was ~0.0014 and 8 of 2,880 comparisons exceeded 0.005; check_draws = 10 shrinks it by about sqrt(10).
    print(
        f"Bootstrap check: {int(checks['ok'].sum())}/{len(checks)} within {tolerance}; "
        f"max |difference| {checks['abs_difference'].max():.5f}; median Monte Carlo SE {checks['monte_carlo_se'].median():.5f}; "
        f"|z| > 3: {int((checks['z'].abs() > 3).sum())}, |z| > 4: {int((checks['z'].abs() > 4).sum())}"
    )
