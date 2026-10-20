import numpy as np
import pandas as pd
import pytest
from conftest import REPO, log_odds_frame, lac_frame

from safe_on_clean_text.conformal.methods import uncorrected_sets, predict_sets
from safe_on_clean_text.conformal.scores import score_frame

METHODS = ["M0", "M1", "M2", "M5"]
DOMAINS = np.array(["abuse", "social", "anxiety", "ptsd", "financial"])


def synthetic(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = rng.uniform(0.02, 0.98, size=n)
    return prob, (rng.random(n) < prob).astype(int), rng.choice(DOMAINS, size=n)


def all_sets(frame: pd.DataFrame, n_calibration: int, n_test: int) -> dict:
    calibration, test = frame.iloc[:n_calibration], frame.iloc[n_calibration : n_calibration + n_test]
    validation = frame.iloc[n_calibration + n_test :]
    out = {}
    for alpha in [0.1, 0.05]:
        for method in METHODS:
            out[(method, alpha)] = predict_sets(calibration, test, method, alpha)[0]
        out[("B1", alpha)] = uncorrected_sets(validation, test, alpha)
    return out


def test_log_odds_scores_give_the_lac_sets(rng):
    """s_y = log p_other - log p_y is a strictly increasing transform of 1 - p_y: identical sets."""
    for _ in range(20):
        prob, labels, domains = synthetic(rng, 700 + 715 + 400)
        planned = all_sets(lac_frame(prob, labels, domain=domains), 700, 715)
        log_odds = all_sets(log_odds_frame(prob, labels, domain=domains), 700, 715)
        for key, sets in planned.items():
            assert np.array_equal(sets, log_odds[key]), key


def test_infinite_scores_never_give_nan_and_an_infinite_threshold_includes_every_label():
    true_scores = np.r_[np.zeros(70), np.full(30, np.inf)]
    calibration = pd.DataFrame({"label": 0, "score_0": true_scores, "score_1": -true_scores})
    test = pd.DataFrame(
        {"label": [0, 1, 0, 1], "score_0": [np.inf, -np.inf, 0.0, 0.0], "score_1": [-np.inf, np.inf, 0.0, 0.0]}
    )
    sets, n_infinite = predict_sets(calibration, test, "M0", 0.1)
    assert sets.all() and n_infinite == 1


@pytest.mark.parametrize("temperature", [0.3, 1.7, 7.0])
def test_temperature_scaling_does_not_change_any_set(rng, temperature):
    prob, labels, domains = synthetic(rng, 700 + 715 + 400)
    frame = log_odds_frame(prob, labels, domain=domains)
    scaled = frame.assign(score_0=frame["score_0"] / temperature, score_1=frame["score_1"] / temperature)
    original, rescaled = all_sets(frame, 700, 715), all_sets(scaled, 700, 715)
    for key in original:
        assert np.array_equal(original[key], rescaled[key]), key


def test_temperature_scaling_does_not_change_the_sets_of_a_stored_run():
    predictions = pd.read_csv(REPO / "results/runs/dreaddit/distilbert/seed_42/predictions.csv.gz")
    metadata = pd.read_csv(REPO / "data/processed/dreaddit/mix_assignment.csv")[["id"]].assign(
        domain="all", confidence=np.nan
    )
    clean = predictions[predictions["condition"] == "clean"]
    calibration = score_frame(clean[clean["split"] == "calibration"], metadata)
    test = score_frame(clean[clean["split"] == "test"], metadata)
    for method in ["M0", "M1", "M5"]:
        original = predict_sets(calibration, test, method, 0.1)[0]
        for temperature in [0.5, 1.3, 4.0]:
            scale = {"score_0": calibration["score_0"] / temperature, "score_1": calibration["score_1"] / temperature}
            scaled_test = {"score_0": test["score_0"] / temperature, "score_1": test["score_1"] / temperature}
            assert np.array_equal(
                original, predict_sets(calibration.assign(**scale), test.assign(**scaled_test), method, 0.1)[0]
            )
