from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from conftest import REPO

from safe_on_clean_text.data.dreaddit import normalize_text_key
from safe_on_clean_text.data.sdcnl import split_by_author
from safe_on_clean_text.evaluation.resplits import make_replicates

SPLITS = ["train", "validation", "calibration", "test"]


def test_split_reports_have_no_shared_posts_or_authors():
    for name in ["dreaddit", "sdcnl"]:
        report = pd.read_csv(REPO / "data/processed" / name / "split_report.csv")
        overlap = report[report["section"] == "overlap"]
        assert len(overlap) > 0 and (overlap["value"].astype(int) == 0).all(), name


def test_dreaddit_splits_share_no_post_and_no_text():
    paths = {name: REPO / "data/processed/dreaddit" / f"{name}.csv" for name in SPLITS}
    if not all(path.exists() for path in paths.values()):
        pytest.skip("processed Dreaddit splits not generated")
    splits = {name: pd.read_csv(path) for name, path in paths.items()}
    for a, b in combinations(SPLITS, 2):
        assert not set(splits[a]["post_id"]) & set(splits[b]["post_id"])
        assert not set(normalize_text_key(splits[a]["text"])) & set(normalize_text_key(splits[b]["text"]))


def test_author_split_keeps_every_author_in_one_split(rng):
    n = 2000
    frame = pd.DataFrame(
        {
            "author_group": rng.integers(0, 900, size=n).astype(str),
            "label": rng.integers(0, 2, size=n),
            "post_id": np.arange(n),
        }
    )
    config = {
        "n_splits": 20,
        "train_folds": [0, 9],
        "validation_folds": [10, 11],
        "calibration_folds": [12, 15],
        "test_folds": [16, 19],
        "random_state": 42,
        "min_rows": 300,
        "min_rows_per_class": 100,
    }
    splits, _ = split_by_author(frame, config)
    assert sum(len(df) for df in splits.values()) == n
    for a, b in combinations(splits, 2):
        assert not set(splits[a]["author_group"]) & set(splits[b]["author_group"])


def test_resplits_and_bootstraps_keep_groups_together(rng):
    n = 1400
    pool = pd.DataFrame(
        {
            "label": rng.integers(0, 2, size=n),
            "group": rng.integers(0, 1100, size=n),
            "split": np.where(np.arange(n) < 700, "calibration", "test"),
        }
    )
    replicates = make_replicates(pool, 20, 2, 42)
    groups = pool["group"].to_numpy()
    for calibration, test in replicates["resplit"]:
        assert len(calibration) + len(test) == n and not set(calibration) & set(test)
        assert not set(groups[calibration]) & set(groups[test])
        assert abs(pool["label"].to_numpy()[calibration].mean() - pool["label"].mean()) < 0.03
    official_test = np.flatnonzero(pool["split"] == "test")
    for calibration, test in replicates["official_test"]:
        assert np.array_equal(test, official_test)
        assert set(calibration) <= set(np.flatnonzero(pool["split"] == "calibration"))
