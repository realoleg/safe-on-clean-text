from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.metrics import reference_band
from safe_on_clean_text.study import ALPHA

# Paper aggregation: per resplit the metric is averaged over a model's seeds, then the
# median and 2.5-97.5% over resplits; the between-seed spread is the min-max of the per-seed medians.
PREVALENCES = [0.05, 0.1, 0.2, 0.35, 0.52]
BASE_METRICS = [
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
    "workload_0",
    "workload_1",
    "uncertainty_referral_0",
    "uncertainty_referral_1",
    "n",
    "n_0",
    "n_1",
    "n_calibration",
    "n_calibration_0",
    "n_calibration_1",
]
GROUP_KEYS = ["protocol", "condition", "method", "alpha"]


def load_replicate_rows(config: dict, model: str, protocols: tuple[str, ...] = ("resplit",)) -> pd.DataFrame:
    """
    Per-(seed, replicate) rows of one model (all its seeds) with the reweighted prevalence metrics added.
    """

    directory = Path(config["evaluation"]["output_dir"]) / "long"
    columns = ["seed", "replicate", *GROUP_KEYS, *BASE_METRICS]
    paths = sorted(directory.glob(f"{model}_seed*.parquet"))
    long = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    for column in ["seed", "protocol", "condition", "method"]:
        long[column] = long[column].astype(str)
    long = long[long["protocol"].isin(protocols)]
    for p in PREVALENCES:
        long[f"coverage_pi={p}"] = p * long["c1"] + (1 - p) * long["c0"]
        long[f"workload_per_1000_pi={p}"] = 1000 * (p * long["workload_1"] + (1 - p) * long["workload_0"])
        long[f"uncertainty_referrals_per_1000_pi={p}"] = 1000 * (
            p * long["uncertainty_referral_1"] + (1 - p) * long["uncertainty_referral_0"]
        )
    return long


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ["seed", "replicate", *GROUP_KEYS]]


def aggregate(
    frame: pd.DataFrame,
    keys: list[str],
    replicate: str = "replicate",
    seed: str = "seed",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Primary aggregation of per-(seed, replicate) rows: mean over seeds per replicate, then median / mean / 2.5% / 97.5%
    over replicates; seed_min / seed_max = range of the per-seed medians (NaN for single-run models).
    Returns one row per (keys, metric).
    """

    metrics = metrics or [
        c for c in frame.columns if c not in [*keys, replicate, seed] and pd.api.types.is_numeric_dtype(frame[c])
    ]
    per_replicate = frame.groupby([*keys, replicate], observed=True, sort=True)[metrics].mean()
    grouped = per_replicate.groupby(level=keys, observed=True)
    statistics = {
        "median": grouped.median(),
        "mean": grouped.mean(),
        "q025": grouped.quantile(0.025),
        "q975": grouped.quantile(0.975),
    }
    per_seed = frame.groupby([*keys, seed], observed=True, sort=True)[metrics].median()
    n_seeds = frame.groupby(keys, observed=True)[seed].nunique()
    seed_groups = per_seed.groupby(level=keys, observed=True)
    statistics["seed_min"] = seed_groups.min().where(n_seeds > 1)
    statistics["seed_max"] = seed_groups.max().where(n_seeds > 1)
    out = pd.concat({name: table.stack(future_stack=True) for name, table in statistics.items()}, axis=1)
    return out.rename_axis([*keys, "metric"]).reset_index()


def primary_summary(config: dict, models: list[str], protocols: tuple[str, ...] = ("resplit",)) -> pd.DataFrame:
    """
    Primary-aggregated summary (one row per model, protocol, condition, method, alpha, metric).
    """

    tables = []
    for model in models:
        long = load_replicate_rows(config, model, protocols)
        tables.append(aggregate(long, GROUP_KEYS, metrics=metric_columns(long)).assign(model=model))
    summary = pd.concat(tables, ignore_index=True)
    return summary[["model", *GROUP_KEYS, "metric", "median", "mean", "q025", "q975", "seed_min", "seed_max"]]


def lookup(
    summary: pd.DataFrame,
    model: str,
    condition: str,
    method: str,
    metric: str,
    alpha: float = 0.1,
    statistic: str = "median",
    protocol: str = "resplit",
) -> float:
    row = summary[
        (summary["model"] == model)
        & (summary["protocol"] == protocol)
        & (summary["condition"] == condition)
        & (summary["method"] == method)
        & (summary["alpha"] == alpha)
        & (summary["metric"] == metric)
    ]
    assert len(row) == 1, (model, condition, method, metric, len(row))
    return float(row[statistic].iloc[0])


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


SIZE_COLUMNS = {"coverage": ("n_calibration", "n"), "c0": ("n_calibration_0", "n_0"), "c1": ("n_calibration_1", "n_1")}


def available_models(config: dict) -> set[str]:
    directory = Path(config["evaluation"]["output_dir"]) / "long"
    return {path.stem.rsplit("_seed", 1)[0] for path in directory.glob("*.parquet")}


def main_alpha_rows(config: dict, model: str) -> pd.DataFrame:
    long = load_replicate_rows(config, model)
    return long[long["alpha"] == ALPHA]


def median_size_band(
    summary: pd.DataFrame, model: str, condition: str, method: str, metric: str, alpha: float = ALPHA
) -> tuple[float, float]:
    """
    Reference band of a coverage metric at the median calibration / test sizes of that metric's population.
    """

    n_calibration, n_test = (
        int(round(lookup(summary, model, condition, method, size, alpha))) for size in SIZE_COLUMNS[metric]
    )
    return reference_band(n_calibration, n_test, alpha)
