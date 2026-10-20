import pandas as pd
import pytest
from conftest import REPO, requires_prepared_data

from safe_on_clean_text.reliability import reliability_report

PREDICTIONS = REPO / "results/runs/dreaddit/distilbert/seed_42/predictions.csv.gz"
STORED_LONG = REPO / "results/evaluation/dreaddit/long/distilbert_seed42.parquet"
N = 5
pytestmark = requires_prepared_data


@pytest.fixture(scope="module")
def report_dir(tmp_path_factory):
    output = tmp_path_factory.mktemp("reliability")
    reliability_report(PREDICTIONS, output, str(REPO / "configs/dreaddit.yaml"), n_replicates=N)
    return output


def test_report_lists_the_headline_metrics(report_dir):
    report = pd.read_csv(report_dir / "report.csv")
    m0 = report[(report["condition"] == "clean") & (report["method"] == "M0") & (report["alpha"] == 0.1)].set_index(
        "metric"
    )
    assert set(m0.index) == {
        "coverage",
        "c0",
        "c1",
        "c1_below_band",
        "workload_per_1000_pi=0.1",
        "uncertainty_referrals_per_1000_pi=0.1",
    }
    assert 0.85 <= m0.loc["coverage", "value"] <= 0.95


def test_report_rows_equal_the_stored_evaluation(report_dir):
    if not STORED_LONG.exists():
        pytest.skip("stored long tables are not in git; build them with the evaluation")
    new = pd.read_parquet(report_dir / "long.parquet")
    stored = pd.read_parquet(STORED_LONG)
    for frame in (new, stored):
        for column in ["protocol", "condition", "method"]:
            frame[column] = frame[column].astype(str)
    stored = stored[
        (stored["protocol"] == "resplit") & (stored["replicate"] < N) & stored["method"].isin(set(new["method"]))
    ]
    keys = ["replicate", "condition", "method", "alpha"]
    numeric = [c for c in new.columns if c in stored.columns and pd.api.types.is_numeric_dtype(new[c])]
    pd.testing.assert_frame_equal(
        new.sort_values(keys)[numeric].reset_index(drop=True),
        stored.sort_values(keys)[numeric].reset_index(drop=True),
        check_exact=True,
    )
