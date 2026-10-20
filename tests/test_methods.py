import numpy as np
import pandas as pd
from conftest import lac_frame

from safe_on_clean_text.conformal.methods import temperature_refusal_sets, predict_sets
from safe_on_clean_text.conformal.metrics import INDICATORS, triage_frame
from safe_on_clean_text.conformal.scores import true_class_scores
from safe_on_clean_text.conformal.thresholds import conformal_threshold, label_shift_thresholds


def split(rng, n_calibration=500, n_test=400):
    prob = rng.beta(2, 2, size=n_calibration + n_test)
    labels = (rng.random(len(prob)) < prob).astype(int)
    return lac_frame(prob[:n_calibration], labels[:n_calibration]), lac_frame(
        prob[n_calibration:], labels[n_calibration:]
    )


def test_class_conditional_sets_use_one_threshold_per_class(rng):
    calibration, test = split(rng)
    scores = true_class_scores(calibration)
    thresholds = [conformal_threshold(scores[calibration["label"] == label], 0.1) for label in [0, 1]]
    expected = np.column_stack([test[f"score_{label}"] <= thresholds[label] for label in [0, 1]])
    assert np.array_equal(predict_sets(calibration, test, "M1", 0.1)[0], expected)


def test_risk_only_screening_flags_or_dismisses_every_case(rng):
    calibration, test = split(rng)
    m1 = predict_sets(calibration, test, "M1", 0.1)[0]
    m5 = predict_sets(calibration, test, "M5", 0.1)[0]
    assert (m5.sum(axis=1) == 1).all()
    assert np.array_equal(m5[:, 1], m1[:, 1])


def test_triage_indicators():
    sets = np.array([[True, False], [False, True], [True, True], [False, False]])
    labels = np.array([0, 0, 1, 1])
    frame = triage_frame(sets, labels)
    assert list(frame.columns) == INDICATORS
    expected = pd.DataFrame(
        {
            "covered": [1, 0, 1, 0],
            "set_size": [1, 1, 2, 0],
            "singleton": [1, 1, 0, 0],
            "empty": [0, 0, 0, 1],
            "full": [0, 0, 1, 0],
            "flag": [0, 1, 0, 0],
            "dismiss": [1, 0, 0, 0],
            "workload": [0, 1, 1, 1],
            "uncertainty_referral": [0, 0, 1, 1],
        },
        dtype=float,
    )
    pd.testing.assert_frame_equal(frame, expected)


def test_temperature_refusal_refers_exactly_the_requested_number(rng):
    margins = rng.normal(size=300)
    for n_refuse in [0, 17, 120]:
        sets = temperature_refusal_sets(margins, 1.4, n_refuse)
        assert int((sets.sum(axis=1) == 2).sum()) == n_refuse
        assert (sets.sum(axis=1) >= 1).all()


def test_label_shift_thresholds_equal_marginal_at_the_calibration_prevalence(rng):
    calibration, _ = split(rng)
    scores, labels = true_class_scores(calibration), calibration["label"].to_numpy()
    threshold = conformal_threshold(scores, 0.1)
    assert label_shift_thresholds(scores, labels, float(labels.mean()), 0.1) == (threshold, threshold)
