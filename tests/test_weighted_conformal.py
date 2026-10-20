import numpy as np
import pytest

from safe_on_clean_text.conformal.thresholds import conformal_threshold, weighted_thresholds


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_equal_weights_give_the_conformal_threshold(alpha):
    rng = np.random.default_rng(0)
    for n in [0, 1, 8, 9, 10, 19, 99, 700]:
        scores = rng.normal(size=n)
        expected = conformal_threshold(scores, alpha)
        assert np.all(weighted_thresholds(scores, np.full(n, 2.5), np.full(3, 2.5), alpha) == expected)


def test_weighted_coverage_under_covariate_shift_is_near_nominal():
    """
    Source x ~ N(0, 1), target x ~ N(1, 1) (density ratio exp(x - 1/2)); score |e| with e ~ N(0, (0.2 + |x|)^2), so the
    target has larger scores. Weighted CP with the true ratio covers about 1 - alpha on the target; unweighted CP does not.
    """

    rng, alpha, n, m, repeats = np.random.default_rng(42), 0.1, 500, 500, 200

    def draw(mean: float, size: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(mean, 1.0, size=size)
        return x, np.abs(rng.normal(0.0, 0.2 + np.abs(x)))

    weighted, unweighted = [], []
    for _ in range(repeats):
        (x_calibration, s_calibration), (x_test, s_test) = draw(0.0, n), draw(1.0, m)
        ratio = np.exp(x_calibration - 0.5), np.exp(x_test - 0.5)
        weighted.append(np.mean(s_test <= weighted_thresholds(s_calibration, *ratio, alpha)))
        unweighted.append(np.mean(s_test <= conformal_threshold(s_calibration, alpha)))
    assert abs(np.mean(weighted) - (1 - alpha)) < 0.01, np.mean(weighted)
    assert np.mean(unweighted) < 1 - alpha - 0.03, np.mean(unweighted)
