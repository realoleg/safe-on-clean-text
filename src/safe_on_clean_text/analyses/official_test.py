from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.methods import predict_sets
from safe_on_clean_text.evaluation.resplits import ensemble_predictions, load_pool, pool_frames

STATISTIC = {"method": "M1", "condition": "clean", "metric": "c0"}


def official_test_p_values(config: dict) -> None:
    """
    One-sided p-value of the official-test c0 under M1 (clean, alpha 0.10) against the resplits, per run ->
    official_test_permutation.csv.
    """

    dataset = config["data"]["name"]
    runs_dir = Path(config["paths"]["runs_dir"]) / dataset
    output_dir = Path(config["evaluation"]["output_dir"])
    alpha = float(config["conformal"]["alpha_values"][0])

    pool = load_pool(Path(config["data"]["processed_dir"]))
    split = pool["split"].to_numpy()
    calibration_positions, test_positions = np.flatnonzero(split == "calibration"), np.flatnonzero(split == "test")
    clean_only = pd.DataFrame({"id": pool["id"], "condition": "clean"})

    runs_by_model: dict[str, list[Path]] = {}
    for path in sorted(runs_dir.glob("*/seed_*/predictions.csv.gz")):
        if "prob_stress" in pd.read_csv(path, nrows=1).columns:
            runs_by_model.setdefault(path.parent.parent.name, []).append(path)
    runs = [
        (model, path.parent.name.removeprefix("seed_"), [path])
        for model, paths in runs_by_model.items()
        for path in paths
    ]
    runs += [(f"{model}_ensemble", "ensemble", paths) for model, paths in runs_by_model.items() if len(paths) > 1]

    rows = []
    for model, seed, paths in runs:
        predictions = ensemble_predictions(paths) if len(paths) > 1 else pd.read_csv(paths[0])
        frames, _ = pool_frames(predictions[predictions["condition"] == "clean"], pool, clean_only, ["clean"])
        test = frames["clean"].iloc[test_positions]
        sets, _ = predict_sets(frames["clean"].iloc[calibration_positions], test, STATISTIC["method"], alpha)
        labels = test["label"].to_numpy()
        official = float(sets[labels == 0, 0].mean())

        long = pd.read_parquet(
            output_dir / "long" / f"{model}_seed{seed}.parquet",
            columns=["protocol", "condition", "method", "alpha", STATISTIC["metric"]],
        )
        reference = long[
            (long["protocol"] == "resplit")
            & (long["condition"] == STATISTIC["condition"])
            & (long["method"] == STATISTIC["method"])
            & (long["alpha"] == alpha)
        ][STATISTIC["metric"]].to_numpy()
        n_at_most = int((reference <= official).sum())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "statistic": "c0_M1_clean",
                "alpha": alpha,
                "official": official,
                "resplit_median": float(np.median(reference)),
                "n_resplits": len(reference),
                "n_resplits_at_most_official": n_at_most,
                # One-sided: is the official split's c0 unusually low among exchangeable resplits?
                "p_value": (1 + n_at_most) / (1 + len(reference)),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "official_test_permutation.csv", index=False)
    print(table.round(4).to_string(index=False))
