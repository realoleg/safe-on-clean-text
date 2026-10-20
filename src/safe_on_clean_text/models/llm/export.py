from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.models.encoders import check_probabilities
from safe_on_clean_text.models.llm.cache import AnswerCache
from safe_on_clean_text.models.llm.confidence import NUMBER_PATTERN
from safe_on_clean_text.models.llm.run import API_MODEL, answer_examples, planned_examples
from safe_on_clean_text.models.predictions import save_predictions
from safe_on_clean_text.provenance import commit_hash, library_versions, timestamp, save_json
from safe_on_clean_text.text import word_count

POOL_SPLITS = ["calibration", "test"]


class CachedAnswersOnly:
    """
    Stands in for a model during export: every answer must come from the cache (export never calls a model).
    """

    calls = 0

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Cache miss ({name}): the export never calls a model; run `safe-on-clean-text llm run` first."
        )


def prediction_frame(
    rows: list[dict[str, Any]], examples: list[tuple[Any, str, str, str]], method: str, labels: pd.Series
) -> pd.DataFrame:
    """
    Rows in the shared schema (no text): id, split, condition, label, n_words, prob_not_stress, prob_stress and
    log-probability columns logit_not_stress / logit_stress from the most exact cached quantity (their difference is the
    log-odds; the LAC scores use it): S-tok log P(no), log P(yes); S-verb log(100 - v), log v for the answer v;
    S-cons log(n - k), log k for k yes-votes of n recognized (all unrecognized: 0, 0, i.e. p = 0.5). Exact 0 / 1 -> -inf.
    S-tok probabilities are recomputed from P(yes) and P(no), so both tails keep full precision.
    """

    frame = pd.DataFrame(
        {
            "id": [example[0] for example in examples],
            "split": [example[1] for example in examples],
            "condition": [example[2] for example in examples],
        }
    )
    frame["label"] = labels.loc[frame["id"]].to_numpy()
    frame["n_words"] = word_count(pd.Series([example[3] for example in examples])).to_numpy()
    if method == "s_tok":
        p_yes = np.array([row["p_yes"] for row in rows], dtype=np.float64)
        p_no = np.array([row["p_no"] for row in rows], dtype=np.float64)
        frame["prob_not_stress"] = p_no / (p_yes + p_no)
        frame["prob_stress"] = p_yes / (p_yes + p_no)
        weights = np.column_stack([p_no, p_yes])
    else:
        prob = np.array([row["p_stress"] for row in rows], dtype=np.float64)
        frame["prob_not_stress"] = 1.0 - prob
        frame["prob_stress"] = prob
        if method == "s_verb":
            values = np.array(
                [
                    float(NUMBER_PATTERN.search(row["answer"]).group()) if not np.isnan(row["p_stress"]) else np.nan
                    for row in rows
                ]
            )
            weights = np.column_stack([100.0 - values, values])
        else:
            yes = np.array([round(row["p_stress"] * row["n_recognized"]) for row in rows], dtype=np.float64)
            recognized = np.array([row["n_recognized"] for row in rows], dtype=np.float64)
            weights = np.where(recognized[:, None] > 0, np.column_stack([recognized - yes, yes]), 1.0)
    with np.errstate(divide="ignore"):
        frame["logit_not_stress"] = np.log(weights[:, 0])
        frame["logit_stress"] = np.log(weights[:, 1])
    return frame


def exclude_unparsed(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    An id unparsed (NaN) in any condition is excluded from calibration and test in all conditions,
    so all conditions share one id set; unparsed validation rows are dropped as well. Returns the counts.
    """

    unparsed = frame["prob_stress"].isna()
    pool_ids = set(frame.loc[unparsed & frame["split"].isin(POOL_SPLITS), "id"])
    excluded = (frame["split"].isin(POOL_SPLITS) & frame["id"].isin(pool_ids)) | (
        unparsed & (frame["split"] == "validation")
    )
    counts = {
        "unparsed_rows": int(unparsed.sum()),
        "excluded_pool_ids": len(pool_ids),
        "excluded_validation_rows": int((unparsed & (frame["split"] == "validation")).sum()),
    }
    return frame[~excluded].reset_index(drop=True), counts


def export(config: dict) -> None:
    """
    Cached LLM answers -> one run folder per (LLM, confidence method, prompt) with predictions and run_meta.json (no
    model calls).
    """

    llm_config = config["llm"]
    data_dir = Path(config["data"]["processed_dir"])
    runs_dir = Path(config["paths"]["runs_dir"]) / config["data"]["name"]
    cache_dir = Path(llm_config["cache_dir"])
    labels = pd.concat([pd.read_csv(data_dir / f"{name}.csv") for name in ["validation", *POOL_SPLITS]]).set_index(
        "id"
    )["label"]

    # The dataset's LLMs (SDCNL: Qwen only).
    for key in llm_config["models"]:
        model_name = llm_config["api_model"] if key == API_MODEL else llm_config["local_models"][key]
        prefix = llm_config["answer_prefix"] if key in llm_config["answer_prefix_models"] else ""
        for entry in llm_config["run_plan"]:
            method = entry["method"]
            if method == "s_cons" and key != API_MODEL:
                continue
            examples = planned_examples(config, data_dir, entry["conditions"])
            cache = AnswerCache(cache_dir / key / f"{method}.jsonl.gz")
            for prompt_id in entry["prompts"]:
                rows = answer_examples(
                    CachedAnswersOnly(),
                    model_name,
                    method,
                    prompt_id,
                    examples,
                    llm_config,
                    cache,
                    answer_prefix=prefix if method == "s_tok" else "",
                )
                frame, counts = exclude_unparsed(prediction_frame(rows, examples, method, labels))
                check_probabilities(frame[["prob_not_stress", "prob_stress"]].to_numpy(), f"{key}/{method}/{prompt_id}")
                margin = frame["logit_stress"].to_numpy() - frame["logit_not_stress"].to_numpy()
                assert not np.isnan(margin).any(), f"{key}/{method}/{prompt_id}: NaN log-odds"

                run_name = f"{key}_{method}_{prompt_id}"
                run_dir = runs_dir / run_name / f"seed_{prompt_id}"
                run_dir.mkdir(parents=True, exist_ok=True)
                save_predictions(frame, run_dir / "predictions.csv.gz")

                pool_clean = frame[frame["split"].isin(POOL_SPLITS) & (frame["condition"] == "clean")]["prob_stress"]
                save_json(
                    {
                        "dataset": config["data"]["name"],
                        "model": run_name,
                        "llm": key,
                        "model_name": model_name,
                        "method": method,
                        "prompt": prompt_id,
                        "seed": prompt_id,
                        "answer_prefix": prefix if method == "s_tok" else "",
                        "conditions": sorted(set(frame["condition"]), key=config["conditions"].index),
                        "n_rows": {
                            split: int((frame["split"] == split).sum()) for split in ["validation", *POOL_SPLITS]
                        },
                        **counts,
                        # Tie diagnostics (clean calibration + test): share of distinct p(stress) values, exact 0 / 1.
                        "distinct_score_share_clean": round(pool_clean.nunique() / len(pool_clean), 6),
                        "exact_0_or_1_share_clean": round(float(pool_clean.isin([0.0, 1.0]).mean()), 6),
                        "cache": str(cache.path),
                        "commit": commit_hash(),
                        "exported_at": timestamp(),
                        "versions": library_versions(),
                    },
                    run_dir / "run_meta.json",
                )
                print(f"[{timestamp()}] {run_name}: {len(frame)} rows, {counts}", flush=True)
