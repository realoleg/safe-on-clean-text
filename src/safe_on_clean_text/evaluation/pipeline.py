from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.evaluation.resplits import load_metadata, load_pool, make_replicates
from safe_on_clean_text.evaluation.runs import (
    add_band_flags,
    add_prevalence_metrics,
    quantile_summary,
    evaluate_run,
)
from safe_on_clean_text.provenance import timestamp

SUMMARY_METRICS = [
    "coverage",
    "c0",
    "c1",
    "sensitivity",
    "set_size",
    "singleton",
    "empty",
    "full",
    "flag",
    "workload",
    "uncertainty_referral",
    "n_infinite_thresholds",
    "n_calibration",
    "n",
]
PER_SEED_METRICS = ["coverage", "c0", "c1", "sensitivity", "workload"]
KEYS = ["dataset", "model", "protocol", "condition", "method", "calibration", "alpha"]
COVERAGE_CHECK_TOLERANCE = 0.01


def summarize_model(
    long_paths: list[Path], group_paths: list[Path], prevalence_grid: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates of one model over its seeds: over all (seed, replicate) and per seed; groups over all.
    """

    long_table = pd.concat([pd.read_parquet(path) for path in long_paths], ignore_index=True)
    long_table = add_prevalence_metrics(add_band_flags(long_table), prevalence_grid)
    for column in KEYS:
        long_table[column] = long_table[column].astype(str) if column != "alpha" else long_table[column]

    band_metrics = [
        f"{m}_{flag}"
        for m in ["coverage", "c0", "c1", "sensitivity"]
        for flag in ["below_band", "in_band", "above_band"]
    ]
    prevalence_metrics = [
        f"{name}_pi={p}"
        for p in prevalence_grid
        for name in ["coverage", "workload_per_1000", "uncertainty_referrals_per_1000"]
    ]
    overall = quantile_summary(long_table, KEYS, SUMMARY_METRICS + band_metrics + prevalence_metrics).assign(seed="all")
    per_seed_metrics = PER_SEED_METRICS + [
        "workload_per_1000_pi=0.1",
        "coverage_below_band",
        "c1_below_band",
        "sensitivity_below_band",
        "sensitivity_above_band",
    ]
    per_seed = quantile_summary(
        long_table.assign(seed=long_table["seed"].astype(str)), KEYS + ["seed"], per_seed_metrics
    )
    summary = pd.concat([overall, per_seed], ignore_index=True)

    group_paths = [path for path in group_paths if path.exists()]
    if not group_paths:
        return summary, pd.DataFrame()
    groups = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    *KEYS,
                    "group_type",
                    "group",
                    "n",
                    "n_calibration",
                    "band_low",
                    "band_high",
                    "coverage",
                    "c1",
                    "workload",
                    "sensitivity",
                ],
            )
            for path in group_paths
        ],
        ignore_index=True,
    )
    for column in [*KEYS, "group_type", "group"]:
        groups[column] = groups[column].astype(str) if column != "alpha" else groups[column]
    groups["coverage_below_band"] = (groups["coverage"] < groups["band_low"]).astype(float)
    grouped = groups.groupby([*KEYS, "group_type", "group"], sort=True)
    group_summary = pd.concat(
        [
            grouped.size().rename("n_values"),
            grouped["n"].median().rename("n_test_median"),
            grouped["n_calibration"].median().rename("n_calibration_median"),
            grouped["coverage"].median().rename("coverage_median"),
            grouped["coverage"].quantile(0.025).rename("coverage_q025"),
            grouped["coverage"].quantile(0.975).rename("coverage_q975"),
            grouped["coverage_below_band"].mean().rename("coverage_below_band_share"),
            grouped["c1"].median().rename("c1_median"),
            grouped["workload"].median().rename("workload_median"),
            grouped["sensitivity"].median().rename("sensitivity_median"),
        ],
        axis=1,
    ).reset_index()

    return summary, group_summary


def evaluate(config: dict) -> None:
    """
    Main evaluation of every run in results/runs/<dataset>: all conformal methods on every replicate of each protocol
    (long and group tables, parquet), pooled_summary.csv, pooled_groups.csv and diagnostics.csv in evaluation.output_dir.
    """

    base_seed = int(config["seed"])
    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    conditions = config["conditions"]
    alphas = [float(alpha) for alpha in config["conformal"]["alpha_values"]]
    prevalence_grid = [float(p) for p in config["conformal"]["prevalence_grid"]]
    evaluation = config["evaluation"]
    output_dir = Path(evaluation["output_dir"])
    (output_dir / "long").mkdir(parents=True, exist_ok=True)
    (output_dir / "groups").mkdir(parents=True, exist_ok=True)

    # Dataset-specific settings (SDCNL: author groups, resplits only, a method subset, no group metrics).
    pool = load_pool(data_dir, evaluation["group_column"])
    validation_metadata = load_metadata(data_dir / "validation.csv")
    mix_assignment = pd.read_csv(data_dir / "mix_assignment.csv")
    methods = evaluation["methods"]
    ensemble_methods = evaluation["ensemble_methods"]

    replicates = make_replicates(pool, int(evaluation["n_replicates"]), int(evaluation["resplit_folds"]), base_seed)
    replicates = {protocol: replicates[protocol] for protocol in evaluation["protocols"]}
    print(
        f"[{timestamp()}] Built replicates: " + ", ".join(f"{protocol} {len(r)}" for protocol, r in replicates.items()),
        flush=True,
    )

    common = {
        "dataset": dataset,
        "pool": pool,
        "validation_metadata": validation_metadata,
        "base_seed": base_seed,
        "tie_jitter": float(config["conformal"]["tie_jitter"]),
        "mix_assignment": mix_assignment,
        "conditions": conditions,
        "alphas": alphas,
        "replicates": replicates,
        "n_group_replicates": int(evaluation["n_group_replicates"]),
        "group_types": evaluation["group_types"],
    }
    runs_by_model: dict[str, list[Path]] = {}
    for predictions_path in sorted(runs_dir.glob("*/seed_*/predictions.csv.gz")):
        if "prob_stress" not in pd.read_csv(predictions_path, nrows=1).columns:
            continue  # LinearSVC has no probabilities and is not part of conformal prediction.
        runs_by_model.setdefault(predictions_path.parent.parent.name, []).append(predictions_path)

    tasks = []
    for model, paths in runs_by_model.items():
        for path in paths:
            tasks.append(
                {
                    **common,
                    "model": model,
                    "seed": path.parent.name.removeprefix("seed_"),
                    "predictions_paths": [path],
                    "methods": methods,
                    "jitter": model in evaluation["jitter_runs"],
                }
            )
        if len(paths) > 1 and ensemble_methods:
            # B3: ensemble of the model's seeds (mean probability), M0 and M1 on top.
            tasks.append(
                {
                    **common,
                    "model": f"{model}_ensemble",
                    "seed": "ensemble",
                    "predictions_paths": paths,
                    "methods": ensemble_methods,
                    "jitter": False,
                }
            )
    for task in tasks:
        task["long_path"] = output_dir / "long" / f"{task['model']}_seed{task['seed']}.parquet"
        task["groups_path"] = output_dir / "groups" / f"{task['model']}_seed{task['seed']}.parquet"

    with ProcessPoolExecutor(max_workers=int(evaluation["workers"])) as executor:
        run_results = list(executor.map(evaluate_run, tasks))
    print(f"[{timestamp()}] All runs evaluated", flush=True)

    summaries, group_summaries_all = [], []
    for model in sorted({task["model"] for task in tasks}):
        model_tasks = [task for task in tasks if task["model"] == model]
        summary, group_summary = summarize_model(
            [task["long_path"] for task in model_tasks], [task["groups_path"] for task in model_tasks], prevalence_grid
        )
        summaries.append(summary)
        group_summaries_all.append(group_summary)

    summary = pd.concat(summaries, ignore_index=True)
    summary = summary[["seed", *[c for c in summary.columns if c != "seed"]]]
    summary.to_csv(output_dir / "pooled_summary.csv", index=False)
    group_summary = pd.concat(group_summaries_all, ignore_index=True)
    if not group_summary.empty:
        group_summary.to_csv(output_dir / "pooled_groups.csv", index=False)

    # Check: clean M0 median coverage over resplits within 0.90 +/- 0.01 for every model (all seeds pooled).
    report_rows: list[tuple[str, str, str, Any]] = []
    main_alpha = alphas[0]
    m0_clean = summary[
        (summary["protocol"] == "resplit")
        & (summary["condition"] == "clean")
        & (summary["method"] == "M0")
        & (summary["alpha"] == main_alpha)
        & (summary["metric"] == "coverage")
    ]
    for _, row in m0_clean.iterrows():
        report_rows.append(
            (
                "coverage_check",
                f"{row['model']}/seed={row['seed']}",
                "M0_clean_median_coverage",
                round(row["median"], 4),
            )
        )
    pooled = m0_clean[m0_clean["seed"] == "all"]
    target = 1 - main_alpha
    for _, row in pooled.iterrows():
        report_rows.append(
            (
                "coverage_check",
                f"{row['model']}/seed=all",
                "within_0.90_pm_0.01",
                bool(abs(row["median"] - target) <= COVERAGE_CHECK_TOLERANCE),
            )
        )

    sizes = np.array([[len(c), len(t)] for c, t in replicates["resplit"]])
    report_rows += [
        ("replicates", "resplit", "n_calibration_min", int(sizes[:, 0].min())),
        ("replicates", "resplit", "n_calibration_max", int(sizes[:, 0].max())),
        ("replicates", "resplit", "n_test_min", int(sizes[:, 1].min())),
        ("replicates", "resplit", "n_test_max", int(sizes[:, 1].max())),
    ]
    if "official_test" in replicates:
        report_rows += [
            (
                "replicates",
                "official_test",
                "n_calibration_min",
                int(min(len(c) for c, _ in replicates["official_test"])),
            ),
            (
                "replicates",
                "official_test",
                "n_calibration_max",
                int(max(len(c) for c, _ in replicates["official_test"])),
            ),
        ]
    for result in run_results:
        name = f"{result['model']}/seed={result['seed']}"
        if result["temperature"] is not None:
            report_rows.append(("temperature", name, "validation_nll_temperature", round(result["temperature"], 4)))
        report_rows += [
            ("ks_descriptive", name, "ks_statistic", round(result["ks_statistic"], 4)),
            ("ks_descriptive", name, "ks_pvalue", round(result["ks_pvalue"], 4)),
            ("runtime", name, "long_rows", result["long_rows"]),
            ("runtime", name, "group_rows", result["group_rows"]),
        ]

    report = pd.DataFrame(report_rows, columns=["section", "scope", "key", "value"])
    report.to_csv(output_dir / "diagnostics.csv", index=False)
    print(report[report["section"] != "runtime"].to_string(index=False))
