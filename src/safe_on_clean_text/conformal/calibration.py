from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from safe_on_clean_text.conformal.scores import logit_margin

# Temperature fitting / scaling uses the logit margin clipped to +-logit(1 - eps), i.e. p clipped to [eps, 1 - eps]
# (needed for LLM scores; no encoder / LogReg margin reaches it); sets always use the unclipped scores.
TEMPERATURE_CLIP = 1e-6


def temperature_margin(frame: pd.DataFrame) -> np.ndarray:
    """
    Margin for temperature fitting / scaling: the logit margin clipped to +-logit(1 - TEMPERATURE_CLIP),
    i.e. p clipped to [TEMPERATURE_CLIP, 1 - TEMPERATURE_CLIP] (finite for LLM p in {0, 1}).
    """

    bound = math.log((1 - TEMPERATURE_CLIP) / TEMPERATURE_CLIP)
    return np.clip(logit_margin(frame), -bound, bound)


def fit_temperature(margins: np.ndarray, labels: np.ndarray) -> float:
    """
    Temperature T minimizing the binary NLL of sigmoid(margin / T) (bounded search over log T in [-3, 3]).
    """

    signed = np.where(labels == 1, margins, -margins)

    def nll(log_temperature: float) -> float:
        return float(np.mean(np.logaddexp(0.0, -signed / np.exp(log_temperature))))

    return float(np.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded").x))


def calibration_metrics(prob_risk: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """
    Calibration of p(stress): Brier, NLL, ECE with n_bins equal-width bins on [0, 1] (1.0 in the last bin)
    and with n_bins equal-mass bins (sorted probabilities split into n_bins nearly equal parts).
    """

    prob_risk = prob_risk.astype(np.float64)
    labels = labels.astype(np.float64)
    clipped = np.clip(prob_risk, 1e-15, 1 - 1e-15)

    def ece(bins: list[np.ndarray]) -> float:
        return float(sum(len(b) * abs(labels[b].mean() - prob_risk[b].mean()) for b in bins if len(b)) / len(labels))

    width_index = np.minimum((prob_risk * n_bins).astype(int), n_bins - 1)
    width_bins = [np.flatnonzero(width_index == b) for b in range(n_bins)]
    mass_bins = np.array_split(np.argsort(prob_risk, kind="stable"), n_bins)

    return {
        "brier": float(np.mean((prob_risk - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        f"ece_width{n_bins}": ece(width_bins),
        f"ece_mass{n_bins}": ece(mass_bins),
    }
