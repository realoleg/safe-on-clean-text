import numpy as np
import pytest
from conftest import lac_frame
from scipy.stats import betabinom

from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.conformal.metrics import reference_band

N_CALIBRATION, N_TEST, REPEATS = 700, 715, 300


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_coverage_on_exchangeable_data_falls_inside_the_reference_band(rng, alpha):
    """About 95% of draws inside the band, for marginal (M0) and class-conditional (M1) coverage."""
    hits = {"M0": 0, "M1_c0": 0, "M1_c1": 0}
    for _ in range(REPEATS):
        prob = rng.beta(2, 2, size=N_CALIBRATION + N_TEST)
        labels = (rng.random(len(prob)) < prob).astype(int)
        calibration, test = (
            lac_frame(prob[:N_CALIBRATION], labels[:N_CALIBRATION]),
            lac_frame(prob[N_CALIBRATION:], labels[N_CALIBRATION:]),
        )
        test_labels = test["label"].to_numpy()
        sets = predict_sets(calibration, test, "M0", alpha)[0]
        low, high = reference_band(N_CALIBRATION, N_TEST, alpha)
        hits["M0"] += low <= sets[np.arange(N_TEST), test_labels].mean() <= high
        sets = predict_sets(calibration, test, "M1", alpha)[0]
        for label in [0, 1]:
            in_class = test_labels == label
            low, high = reference_band(int((calibration["label"] == label).sum()), int(in_class.sum()), alpha)
            hits[f"M1_c{label}"] += low <= sets[in_class, label].mean() <= high
    for name, count in hits.items():
        assert 0.90 <= count / REPEATS <= 0.99, (name, count / REPEATS)


def test_reference_band_is_the_beta_binomial_interval():
    n, m, alpha = 700, 715, 0.1
    ell = int((n + 1) * alpha)
    distribution = betabinom(m, n + 1 - ell, ell)
    assert reference_band(n, m, alpha) == (distribution.ppf(0.025) / m, distribution.ppf(0.975) / m)
    assert reference_band(5, 100, 0.1) == (1.0, 1.0)  # floor(6 * 0.1) = 0: infinite threshold, full coverage
