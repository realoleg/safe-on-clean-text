from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold

from safe_on_clean_text.conformal.scores import score_frame
from safe_on_clean_text.degradations import assign_mix, make_stream

PROTOCOLS = ["resplit", "official_test"]


def with_metadata_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Datasets without domains or annotator agreement (SDCNL) get one domain "none" and missing confidence.
    """

    defaults = {"domain": "none", "confidence": np.nan}
    return frame.assign(**{column: value for column, value in defaults.items() if column not in frame.columns})


def load_metadata(path: Path) -> pd.DataFrame:
    return with_metadata_defaults(pd.read_csv(path))[["id", "domain", "confidence"]]


def load_pool(data_dir: Path, group_column: str = "post_id") -> pd.DataFrame:
    """
    The resplit pool: calibration rows, then test rows, with the columns the evaluation needs;
    "group" = the column resplits and bootstraps keep together (Dreaddit: post_id; SDCNL: author_group).
    """

    pool = pd.concat(
        [pd.read_csv(data_dir / f"{name}.csv").assign(split=name) for name in ["calibration", "test"]],
        ignore_index=True,
    )
    pool = with_metadata_defaults(pool).assign(group=pool[group_column])
    return pool[["id", "post_id", "group", "label", "domain", "confidence", "split"]]


def make_replicates(
    pool: pd.DataFrame,
    n_replicates: int,
    n_folds: int,
    base_seed: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """
    (calibration positions, test positions) in the pool (calibration rows, then test rows) for both protocols.
    resplit r: StratifiedGroupKFold(n_folds, shuffle=True, random_state=r) by group (post / author), fold 0 ->
    calibration, fold 1 -> test. official_test b: calibration groups drawn with replacement (all their rows),
    test = the official test.
    """

    labels = pool["label"].to_numpy()
    posts = pool["group"].to_numpy()
    replicates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {protocol: [] for protocol in PROTOCOLS}

    for r in range(n_replicates):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=r)
        folds = [test_index for _, test_index in splitter.split(pool, labels, groups=posts)]
        replicates["resplit"].append((np.sort(folds[0]), np.sort(folds[1])))

    calibration_positions = np.flatnonzero(pool["split"].to_numpy() == "calibration")
    test_positions = np.flatnonzero(pool["split"].to_numpy() == "test")
    positions_by_post = pd.Series(calibration_positions).groupby(posts[calibration_positions]).apply(np.asarray)
    calibration_posts = positions_by_post.index.to_numpy()

    for b in range(n_replicates):
        rng = make_stream(base_seed, b, "bootstrap")
        drawn = rng.choice(calibration_posts, size=len(calibration_posts), replace=True)
        calibration = np.concatenate(positions_by_post.loc[drawn].to_numpy())
        replicates["official_test"].append((calibration, test_positions))

    return replicates


def available_conditions(predictions: pd.DataFrame, conditions: list[str]) -> list[str]:
    """
    Conditions a run has predictions for, in config order (LLM S-verb, S-cons and prompts P2-P3 have four).
    """

    present = set(predictions["condition"])
    return [condition for condition in conditions if condition in present]


def mix_for_conditions(
    mix_assignment: pd.DataFrame,
    pool: pd.DataFrame,
    conditions: list[str],
    all_conditions: list[str],
    base_seed: int,
) -> pd.DataFrame:
    """
    M3-mix assignment of the pool: the saved one for runs with every condition; for runs with fewer conditions,
    the same per-id draw over the available conditions.
    """

    if conditions == all_conditions:
        return mix_assignment
    return pd.DataFrame(
        {"id": pool["id"].to_numpy(), "condition": assign_mix(pool["id"], conditions, base_seed).to_numpy()}
    )


def pool_frames(
    predictions: pd.DataFrame,
    pool: pd.DataFrame,
    mix_assignment: pd.DataFrame,
    conditions: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Evaluation frames of the pool (calibration rows, then test rows) for every condition,
    and the M3-mix frame (each example in its assigned condition).
    """

    metadata = pool[["id", "domain", "confidence"]]
    frames = {}
    for condition in conditions:
        rows = predictions[predictions["condition"] == condition]
        frame = pd.concat(
            [score_frame(rows[rows["split"] == name], metadata) for name in ["calibration", "test"]],
            ignore_index=True,
        )
        assert np.array_equal(frame["id"].to_numpy(), pool["id"].to_numpy())
        frames[condition] = frame

    assert np.array_equal(mix_assignment["id"].to_numpy(), pool["id"].to_numpy())
    assigned = mix_assignment["condition"].to_numpy()
    mix = (
        pd.concat(
            [frames[condition].assign(position=np.arange(len(pool)))[assigned == condition] for condition in conditions]
        )
        .sort_values("position")
        .drop(columns="position")
        .reset_index(drop=True)
    )
    assert np.array_equal(mix["id"].to_numpy(), pool["id"].to_numpy())

    return frames, mix


def ensemble_predictions(paths: list[Path]) -> pd.DataFrame:
    """
    Ensemble of runs of one model: mean of the class probabilities per (split, condition, id), computed in float64
    from the members' logits; its log-probabilities (logsumexp of the members' log-softmax minus log n) are stored
    as logit columns, so the log-odds scores never pass through a rounded probability. Members must share rows and order.
    """

    members = [pd.read_csv(path) for path in paths]
    keys = ["id", "split", "condition", "label", "n_words"]
    for member in members[1:]:
        assert member[keys].equals(members[0][keys])

    logits = np.stack([member[["logit_not_stress", "logit_stress"]].to_numpy(dtype=np.float64) for member in members])
    log_probs = logits - logsumexp(logits, axis=2, keepdims=True)
    mean_log_probs = logsumexp(log_probs, axis=0) - np.log(len(members))

    out = members[0][keys].copy()
    out["prob_not_stress"] = np.exp(mean_log_probs[:, 0])
    out["prob_stress"] = np.exp(mean_log_probs[:, 1])
    out["logit_not_stress"] = mean_log_probs[:, 0]
    out["logit_stress"] = mean_log_probs[:, 1]
    return out
