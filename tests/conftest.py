from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
requires_prepared_data = pytest.mark.skipif(
    not (REPO / "data/processed/dreaddit/test.csv").exists(), reason="run `safe-on-clean-text prepare` first"
)


def lac_frame(prob_risk: np.ndarray, labels: np.ndarray, **columns) -> pd.DataFrame:
    """Scores 1 - p_y as in the LAC definition (s_0 = p(risk), s_1 = 1 - p(risk))."""
    return pd.DataFrame({"label": labels.astype(int), "score_0": prob_risk, "score_1": 1.0 - prob_risk, **columns})


def log_odds_frame(prob_risk: np.ndarray, labels: np.ndarray, **columns) -> pd.DataFrame:
    """The same scores in log-odds: s_0 = margin, s_1 = -margin with margin = log p - log(1 - p)."""
    margin = np.log(prob_risk) - np.log1p(-prob_risk)
    return pd.DataFrame({"label": labels.astype(int), "score_0": margin, "score_1": -margin, **columns})


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
