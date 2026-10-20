from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from safe_on_clean_text.conformal.scores import RISK_CLASS, true_class_scores
from safe_on_clean_text.conformal.thresholds import conformal_threshold
from safe_on_clean_text.evaluation.aggregation import aggregate, main_alpha_rows
from safe_on_clean_text.evaluation.resplits import available_conditions, load_pool, make_replicates, pool_frames
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS, SDCNL_MODELS

# EXPLORATORY, POST HOC: decomposition of the M1 - M0 risk-class coverage gap from results already computed; no new
# runs or methods. Both methods calibrate on clean data and include the risk label iff its score s_1 <= q (M0: q0 over
# all calibration rows; M1: q1 over the risk rows), so on any test condition c1(M0) - c1(M1) = share of risk test posts
# with q1 < s_1 <= q0 minus the share with q0 < s_1 <= q1 (the signed band mass; negative when M1's cut-off is looser).
LABEL = "exploratory_post_hoc"
BAND_METRICS = ["band_mass", "band_mass_change_vs_clean"]
CUTOFF_METRICS = ["q0", "q1", "M0_risk_in_set_if_p_at_least", "M1_risk_in_set_if_p_at_least"]
# Ratio of the aggregated medians band mass(condition) / band mass(clean), only when the clean median is at least this.
RATIO_MIN_CLEAN_MEDIAN = 0.005
IDENTITY_TOLERANCE = 1e-12


def band_mass(risk_scores: np.ndarray, q0: float, q1: float) -> float:
    """
    Signed share of risk posts whose risk score lies between the M1 and M0 thresholds.
    """

    inside = (risk_scores > q1) & (risk_scores <= q0)
    reversed_inside = (risk_scores > q0) & (risk_scores <= q1)
    return float(inside.mean() - reversed_inside.mean())


def dataset_rows(config: dict, models: list[str]) -> pd.DataFrame:
    """
    Per (model, seed, resplit, condition): clean-calibration thresholds q0, q1 and their p(risk) cut-offs, the band
    mass on the test half, its paired change vs clean, and the stored c1(M0) - c1(M1) of the long table (alpha 0.10,
    resplit protocol) for the identity check.
    """

    dataset = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    pool = load_pool(data_dir, config["evaluation"]["group_column"])
    mix_assignment = pd.read_csv(data_dir / "mix_assignment.csv")
    replicates = make_replicates(
        pool, int(config["evaluation"]["n_replicates"]), int(config["evaluation"]["resplit_folds"]), int(config["seed"])
    )["resplit"]
    conditions = config["conditions"]

    tables = []
    for model in models:
        rows = []
        for path in sorted((Path(config["paths"]["runs_dir"]) / dataset / model).glob("seed_*/predictions.csv.gz")):
            seed = path.parent.name.removeprefix("seed_")
            predictions = pd.read_csv(path)
            assert available_conditions(predictions, conditions) == conditions, (model, seed)
            frames = pool_frames(predictions, pool, mix_assignment, conditions)[0]
            clean_scores, labels = true_class_scores(frames["clean"]), frames["clean"]["label"].to_numpy()
            risk_scores = {condition: frame[f"score_{RISK_CLASS}"].to_numpy() for condition, frame in frames.items()}
            for replicate, (calibration_positions, test_positions) in enumerate(replicates):
                calibration_scores, calibration_labels = (
                    clean_scores[calibration_positions],
                    labels[calibration_positions],
                )
                q0 = conformal_threshold(calibration_scores, ALPHA)
                q1 = conformal_threshold(calibration_scores[calibration_labels == RISK_CLASS], ALPHA)
                test_risk = test_positions[labels[test_positions] == RISK_CLASS]
                for condition in conditions:
                    rows.append(
                        {
                            "seed": seed,
                            "replicate": replicate,
                            "condition": condition,
                            "q0": q0,
                            "q1": q1,
                            "M0_risk_in_set_if_p_at_least": float(expit(-q0)),
                            "M1_risk_in_set_if_p_at_least": float(expit(-q1)),
                            "band_mass": band_mass(risk_scores[condition][test_risk], q0, q1),
                        }
                    )
        rows = pd.DataFrame(rows)
        clean = rows[rows["condition"] == "clean"].set_index(["seed", "replicate"])["band_mass"]
        paired = pd.MultiIndex.from_frame(rows[["seed", "replicate"]])
        rows["band_mass_change_vs_clean"] = rows["band_mass"].to_numpy() - clean.loc[paired].to_numpy()

        long = main_alpha_rows(config, model)
        c1 = long[long["method"].isin(["M0", "M1"])].pivot_table(
            index=["seed", "replicate", "condition"], columns="method", values="c1", aggfunc="first"
        )
        stored = (c1["M0"] - c1["M1"]).rename("c1_M0_minus_M1").reset_index()
        assert len(stored) == len(rows), (model, len(stored), len(rows))
        rows = rows.merge(stored, on=["seed", "replicate", "condition"], how="left", validate="one_to_one")
        assert rows["c1_M0_minus_M1"].notna().all(), model
        tables.append(rows.assign(dataset=dataset, model=model))
    return pd.concat(tables, ignore_index=True)


def summarize(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Aggregated band metrics per condition and clean-data cut-offs (thresholds come from clean calibration and do not
    depend on the test condition), with the ratio of medians and the identity check per row group.
    """

    band = aggregate(rows, [*keys, "condition"], metrics=BAND_METRICS)
    cutoffs = aggregate(rows[rows["condition"] == "clean"], [*keys, "condition"], metrics=CUTOFF_METRICS)
    table = pd.concat([band, cutoffs], ignore_index=True)

    medians = band[band["metric"] == "band_mass"].set_index([*keys, "condition"])["median"]
    clean_median = medians.xs("clean", level="condition")
    clean_median = clean_median.where(clean_median >= RATIO_MIN_CLEAN_MEDIAN)
    ratio = (medians / clean_median.reindex(medians.index.droplevel("condition")).to_numpy()).rename("ratio_of_medians")
    table = table.merge(ratio.reset_index().assign(metric="band_mass"), on=[*keys, "condition", "metric"], how="left")

    identity = (
        (rows["band_mass"] - rows["c1_M0_minus_M1"])
        .abs()
        .groupby([rows[key] for key in [*keys, "condition"]])
        .max()
        .rename("identity_max_abs_difference")
    )
    return table.merge(identity.reset_index(), on=[*keys, "condition"], how="left")


def band_decomposition_table(dreaddit: dict, sdcnl: dict) -> pd.DataFrame:
    """
    The band decomposition for every Table 3 / Table 5 run (encoders, TF-IDF, LLM S-tok P1), alpha 0.10, resplit
    protocol, 1,000 resplits, all conditions. Rows with seed "seed-averaged" use the primary aggregation (per resplit
    averaged over the model's seeds, then median and 2.5-97.5% over resplits); multi-seed models also get one row group
    per seed (median and 2.5-97.5% over that seed's resplits).
    """

    tables = []
    for config, models in [(dreaddit, DREADDIT_MODELS), (sdcnl, SDCNL_MODELS)]:
        rows = dataset_rows(config, models)
        identity = (rows["band_mass"] - rows["c1_M0_minus_M1"]).abs().max()
        assert identity <= IDENTITY_TOLERANCE, (config["data"]["name"], identity)
        primary = summarize(rows, ["dataset", "model"]).assign(seed="seed-averaged")
        multi_seed = rows.groupby("model")["seed"].transform("nunique") > 1
        per_seed = summarize(rows[multi_seed], ["dataset", "model", "seed"])
        table = pd.concat([primary, per_seed], ignore_index=True)

        # Order: models as in the study, the seed-averaged row group first, conditions in config order.
        table["model_order"] = table["model"].map({model: i for i, model in enumerate(models)})
        table["seed_order"] = (table["seed"] != "seed-averaged").astype(int)
        table["condition_order"] = table["condition"].map({c: i for i, c in enumerate(config["conditions"])})
        metrics = BAND_METRICS + CUTOFF_METRICS
        table["metric_order"] = table["metric"].map({metric: i for i, metric in enumerate(metrics)})
        order = ["model_order", "seed_order", "seed", "condition_order", "metric_order"]
        tables.append(table.sort_values(order, kind="stable").drop(columns=order[:2] + order[3:]))

    columns = ["label", "dataset", "model", "seed", "condition", "metric", "median", "mean", "q025", "q975"]
    columns += ["seed_min", "seed_max", "ratio_of_medians", "identity_max_abs_difference"]
    return pd.concat(tables, ignore_index=True).assign(label=LABEL)[columns]
