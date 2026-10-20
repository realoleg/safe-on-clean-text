import numpy as np
import pandas as pd
import pytest

from safe_on_clean_text.evaluation.aggregation import aggregate


def test_primary_aggregation_averages_seeds_per_replicate_then_takes_quantiles(rng):
    frame = pd.DataFrame(
        [(seed, replicate, rng.random()) for seed in ["a", "b", "c"] for replicate in range(40)],
        columns=["seed", "replicate", "c1"],
    ).assign(method="M0")
    out = aggregate(frame, ["method"], metrics=["c1"]).iloc[0]
    per_replicate = frame.groupby("replicate")["c1"].mean()
    assert out["median"] == per_replicate.median()
    # GroupBy.quantile interpolates in compiled code: fused multiply-add on arm64, not on x86-64 (last-bit differences).
    assert out["q025"] == pytest.approx(per_replicate.quantile(0.025), rel=1e-12)
    assert out["q975"] == pytest.approx(per_replicate.quantile(0.975), rel=1e-12)
    per_seed = frame.groupby("seed")["c1"].median()
    assert out["seed_min"] == per_seed.min() and out["seed_max"] == per_seed.max()


def test_single_run_models_get_plain_quantiles_and_no_seed_spread(rng):
    frame = pd.DataFrame({"seed": "P1", "replicate": range(50), "c1": rng.random(50), "method": "M1"})
    out = aggregate(frame, ["method"], metrics=["c1"]).iloc[0]
    assert out["median"] == frame["c1"].median()
    assert np.isnan(out["seed_min"]) and np.isnan(out["seed_max"])
