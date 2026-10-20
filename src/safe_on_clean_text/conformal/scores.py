from __future__ import annotations

import numpy as np
import pandas as pd

from safe_on_clean_text.degradations import length_bucket

RISK_CLASS = 1


# Annotator-agreement buckets of Dreaddit `confidence` (missing = confidence 0 in the raw data).
AGREEMENT_BINS = [0.0, 0.60, 0.87, 1.0]


AGREEMENT_LABELS = ["0.43-0.60", "0.67-0.86", "1.00"]


AGREEMENT_MISSING = "missing"


def agreement_bucket(confidence: pd.Series) -> pd.Series:
    buckets = pd.cut(confidence, bins=AGREEMENT_BINS, labels=AGREEMENT_LABELS).astype(object)
    assert buckets[confidence.notna()].notna().all(), "confidence value outside the agreement buckets"
    return buckets.where(confidence.notna(), AGREEMENT_MISSING).astype(str)


def score_frame(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Predictions of one (split, condition) with scores and grouping columns: domain, agreement bucket, length bucket
    of the (degraded) text, and LAC scores in log-odds, s_y = log p_other - log p_y, from the stored logits / log-probabilities
    (for numerical precision: a strictly increasing transform of 1 - p_y in the binary case, so the sets are the
    LAC sets, without the float64 rounding of 1 - p near 0 / 1). Exact 0 / 1 probabilities give +-inf, never NaN.
    """

    out = predictions.merge(metadata[["id", "domain", "confidence"]], on="id", how="left", validate="one_to_one")
    assert len(out) == len(predictions) and out["domain"].notna().all()

    margin = logit_margin(out)
    out["score_0"] = margin
    out["score_1"] = -margin
    assert not np.isnan(margin).any(), "log-odds score is NaN"
    out["agreement"] = agreement_bucket(out["confidence"])
    out["length_bucket"] = length_bucket(out["n_words"])
    return out.reset_index(drop=True)


def true_class_scores(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame["label"].to_numpy() == 1, frame["score_1"].to_numpy(), frame["score_0"].to_numpy())


def logit_margin(frame: pd.DataFrame) -> np.ndarray:
    return frame["logit_stress"].to_numpy(dtype=np.float64) - frame["logit_not_stress"].to_numpy(dtype=np.float64)
