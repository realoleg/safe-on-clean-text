import math
from fractions import Fraction

import numpy as np
import pytest

from safe_on_clean_text.conformal.thresholds import conformal_rank, conformal_threshold, uncorrected_threshold


@pytest.mark.parametrize("alpha", [0.1, 0.05])
@pytest.mark.parametrize("n", [0, 1, 8, 9, 10, 18, 19, 20, 99, 700])
def test_threshold_is_the_kth_order_statistic(n, alpha):
    scores = np.random.default_rng(n).normal(size=n)
    k = math.ceil((n + 1) * (1 - Fraction(str(alpha))))
    assert conformal_rank(n, alpha) == k
    expected = math.inf if k > n else np.sort(scores)[k - 1]
    assert conformal_threshold(scores, alpha) == expected


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_threshold_is_the_smallest_with_the_finite_sample_guarantee(alpha):
    """Brute force: the smallest candidate t with #{s <= t} >= (n + 1)(1 - alpha)."""
    rng = np.random.default_rng(1)
    for n in range(1, 60):
        scores = rng.integers(0, 10, size=n).astype(float)  # many ties
        needed = (n + 1) * (1 - Fraction(str(alpha)))
        candidates = sorted(set(scores)) + [math.inf]
        expected = next(t for t in candidates if t == math.inf or (scores <= t).sum() >= needed)
        assert conformal_threshold(scores, alpha) == expected


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_infinite_threshold_boundary(alpha):
    n_finite = math.ceil(1 / Fraction(str(alpha)) - 1)
    assert math.isinf(conformal_threshold(np.linspace(0, 1, n_finite - 1), alpha))
    assert math.isfinite(conformal_threshold(np.linspace(0, 1, n_finite), alpha))


def test_uncorrected_threshold_has_no_finite_sample_correction():
    scores = np.arange(1, 101, dtype=float)
    assert uncorrected_threshold(scores, 0.1) == 90.0  # ceil(100 * 0.9) = 90
    assert conformal_threshold(scores, 0.1) == 91.0  # ceil(101 * 0.9) = 91
    assert math.isinf(uncorrected_threshold(np.array([]), 0.1))
