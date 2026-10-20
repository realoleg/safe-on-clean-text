from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from safe_on_clean_text.analyses.resplit_scores import in_study_order, load_run_scores, run_tasks
from safe_on_clean_text.conformal.scores import RISK_CLASS
from safe_on_clean_text.evaluation.aggregation import aggregate
from safe_on_clean_text.study import DREADDIT_MODELS

# DESCRIPTIVE follow-up (no criterion): do the zero-shot LLMs rank degraded text better than the encoders? Primary:
# the H9 oracle workload at c1 = 0.90 (lower = cheaper); secondary: test-half AUROC of the risk log-odds. Paired
# differences per resplit, encoders averaged over their seeds within each resplit.
ZERO_SHOT = ["qwen_s_tok_P1", "gemma_s_tok_P1", "gpt_s_tok_P1"]
ENCODERS = ["distilbert", "roberta", "mentalroberta"]
PAIR_CONDITIONS = ["head_25", "tail_25", "del_30", "typo_10"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Rank AUROC (Mann-Whitney, ties counted half); scores may be +-inf.
    """

    ranks = rankdata(scores)
    positive = labels == RISK_CLASS
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def run(task: dict[str, Any]) -> pd.DataFrame:
    run_scores = load_run_scores(task["path"], task["pool"], task["mix_assignment"], task["conditions"])
    rows = []
    for replicate, (_, test) in enumerate(task["replicates"]):
        for condition in task["conditions"]:
            rows.append(
                {
                    "model": task["model"],
                    "seed": run_scores.seed,
                    "replicate": replicate,
                    "condition": condition,
                    # Risk log-odds = -s_1.
                    "auroc": auroc(-run_scores.scores[condition][test, RISK_CLASS], run_scores.labels[test]),
                }
            )
    return pd.DataFrame(rows)


def verdict(metric: str, low: float, high: float) -> str:
    better = high < 0 if metric == "oracle_workload" else low > 0
    worse = low > 0 if metric == "oracle_workload" else high < 0
    return "zero-shot better" if better else "encoder better" if worse else "interval includes 0"


def zero_shot_ranking_tables(config: dict, oracle_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (1) Paired zero-shot minus encoder differences per resplit of the oracle workload per 1,000 (rows of
    oracle_cost.oracle_cost_rows) and of the AUROC under head_25, tail_25, del_30 and typo_10, with a verdict per row
    (interval excluding 0). (2) AUROC of all 7 models in all 11 conditions and the paired difference to the best encoder
    of that condition (highest median AUROC). Primary aggregation.
    """

    with ProcessPoolExecutor(max_workers=int(config["evaluation"]["workers"])) as executor:
        rows = pd.concat(executor.map(run, run_tasks(config, DREADDIT_MODELS)), ignore_index=True)
    per_replicate = (
        rows.merge(
            oracle_rows[["model", "seed", "replicate", "condition", "workload_oracle"]], how="left", validate="1:1"
        )
        .rename(columns={"workload_oracle": "oracle_workload"})
        .groupby(["model", "condition", "replicate"])[["oracle_workload", "auroc"]]
        .mean()
    )
    assert per_replicate.notna().all().all()

    pairs = []
    for zero_shot in ZERO_SHOT:
        for encoder in ENCODERS:
            for condition in PAIR_CONDITIONS:
                difference = per_replicate.loc[(zero_shot, condition)] - per_replicate.loc[(encoder, condition)]
                for metric in ["oracle_workload", "auroc"]:
                    low, high = np.quantile(difference[metric], [0.025, 0.975])
                    pairs.append(
                        {
                            "zero_shot": zero_shot,
                            "encoder": encoder,
                            "condition": condition,
                            "metric": metric,
                            "median": float(difference[metric].median()),
                            "q025": float(low),
                            "q975": float(high),
                            "verdict": verdict(metric, low, high),
                        }
                    )

    table = aggregate(rows, ["model", "condition"], metrics=["auroc"])
    medians = table.set_index(["model", "condition"])["median"]
    differences = []
    for condition in config["conditions"]:
        best = max(ENCODERS, key=lambda encoder: medians[(encoder, condition)])
        reference = per_replicate.loc[(best, condition), "auroc"]
        for model in DREADDIT_MODELS:
            difference = per_replicate.loc[(model, condition), "auroc"] - reference
            differences.append(
                pd.DataFrame(
                    {"model": model, "condition": condition, "replicate": difference.index, "seed": "seed-averaged"}
                ).assign(auroc_minus_best_encoder=difference.to_numpy(), best_encoder=best)
            )
    differences = pd.concat(differences, ignore_index=True)
    best_encoder = differences[["condition", "best_encoder"]].drop_duplicates()
    difference_table = aggregate(differences.drop(columns="best_encoder"), ["model", "condition"])
    auroc_table = pd.concat([table, difference_table], ignore_index=True).merge(best_encoder, on="condition")
    return pd.DataFrame(pairs), in_study_order(auroc_table, config["conditions"])
