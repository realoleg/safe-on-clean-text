import numpy as np
from conftest import lac_frame

from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.conformal.metrics import triage_frame, reweight, summarize_indicators


def test_prevalence_reweighting_equals_a_class_weighted_mean(rng):
    prob = rng.beta(2, 2, size=900)
    labels = (rng.random(900) < prob).astype(int)
    calibration, test = lac_frame(prob[:500], labels[:500]), lac_frame(prob[500:], labels[500:])
    test_labels = test["label"].to_numpy()
    sets = predict_sets(calibration, test, "M0", 0.1)[0]
    indicators = triage_frame(sets, test_labels)
    summary = summarize_indicators(indicators, test_labels)
    for prevalence in [0.05, 0.1, 0.2, 0.35, 0.52]:
        weights = np.where(
            test_labels == 1, prevalence / (test_labels == 1).sum(), (1 - prevalence) / (test_labels == 0).sum()
        )
        analytic = reweight(summary, prevalence)
        assert np.isclose(analytic["coverage"], (weights * indicators["covered"]).sum(), rtol=0, atol=1e-15)
        for name in ["workload", "uncertainty_referral", "set_size"]:
            assert np.isclose(analytic[name], (weights * indicators[name]).sum(), rtol=0, atol=1e-15)
