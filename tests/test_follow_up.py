import numpy as np
from sklearn.metrics import roc_auc_score

from safe_on_clean_text.analyses.workload_matched_sensitivity import crossing
from safe_on_clean_text.analyses.zero_shot_ranking import auroc


def test_rank_auroc_matches_sklearn_with_ties_and_infinite_scores():
    rng = np.random.default_rng(0)
    scores, labels = rng.normal(size=300), (rng.random(300) < 0.5).astype(int)
    scores[:40] = scores[40:80]  # ties
    assert abs(auroc(scores, labels) - roc_auc_score(labels, scores)) < 1e-12
    scores[labels == 1][:3] = np.inf
    finite = np.where(np.isinf(scores), 1e9, scores)
    assert abs(auroc(scores, labels) - roc_auc_score(labels, finite)) < 1e-12


def test_crossing_interpolates_and_takes_the_one_nearest_to_alpha_010():
    grid = [0.08, 0.09, 0.10, 0.11, 0.12]
    workloads = np.array([500.0, 400.0, 300.0, 400.0, 500.0])  # 450 is crossed at 0.085 and at 0.115
    i, t = crossing(workloads, 450.0, grid)
    assert (i, t) == (0, 0.5)  # equal distance: the smaller alpha
    i, t = crossing(workloads, 350.0, grid)
    assert i == 1 and abs((1 - t) * grid[i] + t * grid[i + 1] - 0.095) < 1e-12
    assert crossing(workloads, 250.0, grid) is None
