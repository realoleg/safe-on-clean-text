from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.metrics import summarize_groups, triage_indicators
from safe_on_clean_text.evaluation.resplits import available_conditions, load_pool, make_replicates, pool_frames
from safe_on_clean_text.study import DREADDIT_MODELS

# Shared inputs of the H7-H10 analyses (analyses/{workload_matched,low_prevalence_calibration,oracle_cost,
# weighted_conformal}.py): per run, the log-odds scores of every pool row in every condition, and the resplits.
PREVALENCE = 0.1


@dataclass
class RunScores:
    """
    Scores of one run on the resplit pool (calibration rows, then test rows): labels, (n x 2) score matrices
    [s_0, s_1] per condition and for the M3-mix frame (each row in its assigned condition).
    """

    seed: str
    labels: np.ndarray
    scores: dict[str, np.ndarray]
    mix: np.ndarray


def load_run_scores(path: Path, pool: pd.DataFrame, mix_assignment: pd.DataFrame, conditions: list[str]) -> RunScores:
    predictions = pd.read_csv(path)
    assert available_conditions(predictions, conditions) == conditions, path
    frames, mix = pool_frames(predictions, pool, mix_assignment, conditions)
    columns = ["score_0", "score_1"]
    return RunScores(
        seed=path.parent.name.removeprefix("seed_"),
        labels=frames["clean"]["label"].to_numpy(),
        scores={condition: frame[columns].to_numpy(dtype=np.float64) for condition, frame in frames.items()},
        mix=mix[columns].to_numpy(dtype=np.float64),
    )


def run_tasks(config: dict, models: list[str]) -> list[dict[str, Any]]:
    """
    One task per (model, seed) run with the pool, its mix assignment, the conditions and the resplits.
    """

    data_dir = Path(config["data"]["processed_dir"])
    pool = load_pool(data_dir, config["evaluation"]["group_column"])
    common = {
        "pool": pool,
        "mix_assignment": pd.read_csv(data_dir / "mix_assignment.csv"),
        "conditions": config["conditions"],
        "replicates": make_replicates(
            pool,
            int(config["evaluation"]["n_replicates"]),
            int(config["evaluation"]["resplit_folds"]),
            int(config["seed"]),
        )["resplit"],
        "base_seed": int(config["seed"]),
    }
    runs_dir = Path(config["paths"]["runs_dir"]) / config["data"]["name"]
    return [
        {**common, "model": model, "path": path}
        for model in models
        for path in sorted((runs_dir / model).glob("seed_*/predictions.csv.gz"))
    ]


def true_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return scores[np.arange(len(labels)), labels]


def operating_point(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    """
    Metrics of the sets {y : s_y <= threshold_y} (thresholds of shape (2,) or one pair per row, (n, 2)).
    """

    return set_metrics(scores <= thresholds, labels)


def set_metrics(sets: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    c0, c1, share of full sets and workload per 1,000 reweighted to PREVALENCE (as in the paper tables) of (n x 2) sets.
    """

    summary = summarize_groups(triage_indicators(sets, labels), labels, np.zeros(len(labels), dtype=int), 1)[0]
    return {
        "c0": summary["c0"],
        "c1": summary["c1"],
        "full": summary["full"],
        "workload_per_1000": 1000 * (PREVALENCE * summary["workload_1"] + (1 - PREVALENCE) * summary["workload_0"]),
    }


def in_study_order(table: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """
    Rows sorted by model (study order), then condition (config order); other row orders are kept.
    """

    order = {
        "model": {m: i for i, m in enumerate(DREADDIT_MODELS)},
        "condition": {c: i for i, c in enumerate(conditions)},
    }
    return table.sort_values(["model", "condition"], key=lambda s: s.map(order[s.name]), kind="stable").reset_index(
        drop=True
    )
