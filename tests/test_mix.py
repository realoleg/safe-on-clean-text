import pandas as pd
import pytest
from conftest import REPO

from safe_on_clean_text.config import load_config
from safe_on_clean_text.degradations import assign_mix


@pytest.mark.parametrize(
    "config_path, mix_path",
    [
        ("configs/dreaddit.yaml", "data/processed/dreaddit/mix_assignment.csv"),
        ("configs/sdcnl.yaml", "data/processed/sdcnl/mix_assignment.csv"),
    ],
)
def test_committed_mix_assignment_is_reproduced(config_path, mix_path):
    config = load_config(str(REPO / config_path))
    mix = pd.read_csv(REPO / mix_path)
    drawn = assign_mix(mix["id"], config["conditions"], int(config["seed"]))
    assert (drawn.to_numpy() == mix["condition"].to_numpy()).all()
