from __future__ import annotations

from pathlib import Path

import pandas as pd

from safe_on_clean_text.config import load_config
from safe_on_clean_text.evaluation.resplits import load_metadata, load_pool, make_replicates
from safe_on_clean_text.evaluation.runs import add_band_flags, add_prevalence_metrics, quantile_summary, evaluate_run

# Public entry point: one predictions file -> reliability report, through the conformal methods and the grouped
# resplit protocol of the paper (evaluate_run, resplit protocol only, no group metrics).
REQUIRED_COLUMNS = ["id", "split", "condition", "label", "n_words", "prob_stress", "logit_stress", "logit_not_stress"]
DEFAULT_METHODS = ["M0", "M1", "M3", "M3-mix", "M5"]
REPORT_METRICS = [
    "coverage",
    "c0",
    "c1",
    "c1_below_band",
    "workload_per_1000_pi=0.1",
    "uncertainty_referrals_per_1000_pi=0.1",
]


def reliability_report(
    predictions_path: str | Path,
    output_dir: str | Path,
    config_path: str = "configs/dreaddit.yaml",
    methods: list[str] | None = None,
    n_replicates: int | None = None,
    model: str = "model",
) -> pd.DataFrame:
    """
    Reliability report of one model's predictions on a dataset of this repository.

    predictions_path: a CSV (optionally .gz) in the format of results/runs/<dataset>/<model>/seed_<seed>/predictions.csv.gz:
        one row per (id, split, condition) with the columns in REQUIRED_COLUMNS, covering every calibration and test id of
        the config's dataset for at least the clean condition (more conditions -> per-condition rows and M3 / M3-mix on
        them); validation rows are needed only for the baselines B1 / B2.
    output_dir: receives long.parquet (per resplit x condition x method x alpha), summary.csv (median and 2.5-97.5% over
        resplits of every metric) and report.csv (the headline table returned here).
    methods: conformal methods (default M0, M1, M3, M3-mix, M5; B1 / B2 also accepted).
    n_replicates: number of grouped resplits (default: the config's, 1,000).

    Returns one row per (condition, method, alpha, metric) with the median and 2.5-97.5% over resplits for coverage,
    class coverage c0 / c1, the share of resplits whose c1 is below its reference band (expected about 2.5% when the
    class-conditional guarantee holds), and workload (not dismissed) and referrals (full or empty sets) per 1,000 screened at prevalence 0.10.
    """

    config = load_config(config_path)
    evaluation = config["evaluation"]
    data_dir = Path(config["data"]["processed_dir"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_splits = [name for name in ["calibration", "test", "validation"] if not (data_dir / f"{name}.csv").exists()]
    if missing_splits:
        raise FileNotFoundError(
            f"{data_dir} lacks the prepared splits {missing_splits}; run `safe-on-clean-text prepare` first"
        )

    predictions = pd.read_csv(predictions_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions file lacks columns {missing}")
    if "clean" not in set(predictions["condition"]):
        raise ValueError("predictions file has no clean condition")

    pool = load_pool(data_dir, evaluation["group_column"])
    replicates = make_replicates(
        pool, int(n_replicates or evaluation["n_replicates"]), int(evaluation["resplit_folds"]), int(config["seed"])
    )
    long_path = output_dir / "long.parquet"
    task = {
        "dataset": config["data"]["name"],
        "model": model,
        "seed": "user",
        "pool": pool,
        "validation_metadata": load_metadata(data_dir / "validation.csv"),
        "base_seed": int(config["seed"]),
        "tie_jitter": float(config["conformal"]["tie_jitter"]),
        "mix_assignment": pd.read_csv(data_dir / "mix_assignment.csv"),
        "conditions": config["conditions"],
        "alphas": [float(a) for a in config["conformal"]["alpha_values"]],
        "replicates": {"resplit": replicates["resplit"]},
        "n_group_replicates": 0,
        "group_types": [],
        "predictions_paths": [Path(predictions_path)],
        "methods": methods or DEFAULT_METHODS,
        "jitter": False,
        "long_path": long_path,
        "groups_path": output_dir / "groups.parquet",
    }
    evaluate_run(task)

    long = pd.read_parquet(long_path)
    for column in long.select_dtypes(include="category").columns:
        long[column] = long[column].astype(str)
    long = add_prevalence_metrics(add_band_flags(long), [float(p) for p in config["conformal"]["prevalence_grid"]])
    metrics = [
        c
        for c in long.columns
        if c not in ["dataset", "model", "seed", "protocol", "replicate", "condition", "method", "calibration", "alpha"]
    ]
    summary = quantile_summary(long, ["condition", "method", "alpha"], metrics)
    summary.to_csv(output_dir / "summary.csv", index=False)
    report = summary[summary["metric"].isin(REPORT_METRICS)].copy()
    # For the band flag the mean over resplits is the share of resplits below the band.
    report.loc[report["metric"] == "c1_below_band", "median"] = report.loc[report["metric"] == "c1_below_band", "mean"]
    report = report.rename(columns={"median": "value"})[
        ["condition", "method", "alpha", "metric", "value", "q025", "q975", "n_values"]
    ]
    report.loc[report["metric"] == "c1_below_band", ["q025", "q975"]] = float("nan")
    report.to_csv(output_dir / "report.csv", index=False)
    return report
