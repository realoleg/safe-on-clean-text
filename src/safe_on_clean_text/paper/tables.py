from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.scores import agreement_bucket
from safe_on_clean_text.evaluation.aggregation import aggregate, lookup
from safe_on_clean_text.evaluation.classification import classification_rows
from safe_on_clean_text.paper import TABLES
from safe_on_clean_text.study import LLM_VARIANTS, LLMS
from safe_on_clean_text.text import word_count

CONFORMAL_TABLE_METHODS = ["M0", "M1", "M2", "M3", "M3-mix", "M5", "B1", "B2"]
# Lowest per-domain coverage of a resplit among the 5 Dreaddit domains, all calibrated together (not leave-one-domain-out).
WORST_DOMAIN_SCENARIO = "worst in-distribution domain (min of 5; clean; first 200 resplits)"


def data_table(dreaddit: dict, sdcnl: dict) -> None:
    """
    Data: sizes, risk share, posts / author groups, words, Dreaddit domains and annotator-agreement buckets per split.
    SDCNL values come from its committed split report (the data files are not in the repository).
    """

    rows = []
    for split in ["train", "validation", "calibration", "test"]:
        frame = pd.read_csv(Path(dreaddit["data"]["processed_dir"]) / f"{split}.csv")
        base = {"dataset": "dreaddit", "split": split}
        rows += [
            {**base, "quantity": "rows", "value": len(frame)},
            {**base, "quantity": "posts", "value": frame["post_id"].nunique()},
            {**base, "quantity": "risk (stress) rows", "value": int(frame["label"].sum())},
            {**base, "quantity": "risk share", "value": float(frame["label"].mean())},
            {**base, "quantity": "median words", "value": float(word_count(frame["text"]).median())},
        ]
        for domain, part in frame.groupby("domain"):
            rows += [
                {**base, "quantity": f"domain {domain}: rows", "value": len(part)},
                {**base, "quantity": f"domain {domain}: risk share", "value": float(part["label"].mean())},
            ]
        for bucket, part in frame.groupby(agreement_bucket(frame["confidence"])):
            rows.append({**base, "quantity": f"agreement {bucket}: rows", "value": len(part)})
    report = pd.read_csv(Path(sdcnl["data"]["processed_dir"]) / "split_report.csv")
    report = report[report["section"] == "size"].set_index(["split", "key"])["value"].astype(float)
    for split in ["train", "validation", "calibration", "test"]:
        base = {"dataset": "sdcnl", "split": split}
        rows += [
            {**base, "quantity": "rows", "value": report[(split, "rows")]},
            {**base, "quantity": "author groups", "value": report[(split, "author_groups")]},
            {**base, "quantity": "risk (suicidal ideation) rows", "value": report[(split, "label_1")]},
            {**base, "quantity": "risk share", "value": report[(split, "label_1")] / report[(split, "rows")]},
            {**base, "quantity": "median words", "value": report[(split, "median_words")]},
        ]
    table = pd.DataFrame(rows)
    # Counts and medians as written numbers (the column also holds shares).
    table["value"] = pd.Series(
        [v if "share" in q or not float(v).is_integer() else int(v) for q, v in zip(table["quantity"], table["value"])],
        dtype=object,
    )
    table.to_csv(TABLES / "table1_data.csv", index=False)


def classification_metrics_table(dreaddit: dict, sdcnl: dict) -> None:
    """
    Classification on the fixed test split, clean texts: median and min-max over seeds (macro-F1, AUROC, Brier, NLL, ECE x 2),
    raw and temperature-scaled probabilities (Dreaddit); SDCNL computed here from the stored predictions (raw only).
    """

    dreaddit_rows = pd.read_csv(Path(dreaddit["evaluation"]["output_dir"]) / "classification.csv")
    sdcnl_rows = []
    for path in sorted((Path(sdcnl["paths"]["runs_dir"]) / sdcnl["data"]["name"]).glob("*/seed_*/predictions.csv.gz")):
        predictions = pd.read_csv(path)
        sdcnl_rows += [
            {"dataset": "sdcnl", "model": path.parent.parent.name, "seed": path.parent.name.removeprefix("seed_"), **r}
            for r in classification_rows(predictions, None)
        ]
    table = pd.concat([dreaddit_rows, pd.DataFrame(sdcnl_rows)], ignore_index=True)
    table = table[(table["condition"] == "clean") & table["split"].isin(["validation", "test"])]
    metrics = ["macro_f1", "auroc", "brier", "nll", "ece_width15", "ece_mass15"]
    grouped = table.groupby(["dataset", "model", "split", "probabilities"], sort=True)[metrics]
    out = pd.concat({"median": grouped.median(), "min": grouped.min(), "max": grouped.max()}, axis=1)
    out.columns = [f"{metric}_{statistic}" for statistic, metric in out.columns]
    out = out[[f"{metric}_{statistic}" for metric in metrics for statistic in ["median", "min", "max"]]]
    out.insert(0, "n_seeds", table.groupby(["dataset", "model", "split", "probabilities"], sort=True)["seed"].nunique())
    out.reset_index().to_csv(TABLES / "table2_classification.csv", index=False)


def conformal_table(config: dict, summary: pd.DataFrame, models: list[str], domains: bool) -> pd.DataFrame:
    """
    Tables 3 / 5: model x method x {clean, pi = 0.10, head_25, worst in-distribution domain (clean)}: coverage, c1, workload per 1,000
    (pi = 0.10) and referrals per 1,000 (full or empty sets; pi = 0.10), both alphas.
    """

    scenarios = [
        (
            "clean",
            "clean",
            ["coverage", "c0", "c1", "workload_per_1000_pi=0.1", "uncertainty_referrals_per_1000_pi=0.1"],
        ),
        ("pi=0.1", "clean", ["coverage_pi=0.1"]),
        ("head_25", "head_25", ["coverage", "c1", "workload_per_1000_pi=0.1", "uncertainty_referrals_per_1000_pi=0.1"]),
    ]
    tables = []
    for scenario, condition, metrics in scenarios:
        part = summary[
            summary["model"].isin(models)
            & (summary["condition"] == condition)
            & summary["method"].isin(CONFORMAL_TABLE_METHODS)
            & summary["metric"].isin(metrics)
        ]
        tables.append(part.assign(scenario=scenario))
    if domains:
        for model in models:
            groups = group_rows(config, model, "domain")
            groups = groups[groups["condition"] == "clean"]
            worst = (
                groups.groupby(["seed", "replicate", "method", "alpha"], observed=True)["coverage"].min().reset_index()
            )
            tables.append(
                aggregate(worst, ["method", "alpha"], metrics=["coverage"]).assign(
                    model=model,
                    scenario=WORST_DOMAIN_SCENARIO,
                    condition="clean",
                    protocol="resplit",
                )
            )
    table = pd.concat(tables, ignore_index=True)
    order = {name: i for i, name in enumerate(models)}
    table = table.sort_values(
        ["alpha", "model", "method", "scenario", "metric"], key=lambda s: s.map(order) if s.name == "model" else s
    )
    return table[["model", "alpha", "method", "scenario", "metric", "median", "q025", "q975", "seed_min", "seed_max"]]


def group_rows(config: dict, model: str, group_type: str) -> pd.DataFrame:
    directory = Path(config["evaluation"]["output_dir"]) / "groups"
    columns = [
        "seed",
        "replicate",
        "protocol",
        "condition",
        "method",
        "alpha",
        "group_type",
        "group",
        "n",
        "coverage",
        "c1",
        "workload",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in sorted(directory.glob(f"{model}_seed*.parquet"))]
    )
    for column in ["seed", "protocol", "condition", "method", "group_type", "group"]:
        frame[column] = frame[column].astype(str)
    return frame[(frame["protocol"] == "resplit") & (frame["group_type"] == group_type)]


def llm_table(dreaddit: dict, summary: pd.DataFrame) -> None:
    """
    LLMs: every confidence method (S-tok, S-verb, S-cons) and prompt (P1-P3, S-tok) with validation / test classification,
    the clean-text probability mass on the answer tokens (S-tok) and the conformal operating points (alpha 0.10, clean).
    """

    output_dir = Path(dreaddit["evaluation"]["output_dir"])
    classification = pd.read_csv(output_dir / "classification.csv")
    classification = (
        classification[(classification["condition"] == "clean") & (classification["probabilities"] == "raw")]
        .set_index(["model", "split"])
        .sort_index()
    )
    runs = pd.concat(
        [pd.read_csv(Path(dreaddit["llm"]["output_dir"]) / f"{llm}_run_summary.csv").assign(llm=llm) for llm in LLMS]
    )
    runs = runs[(runs["split"] == "test") & (runs["condition"] == "clean")].set_index(["llm", "method", "prompt"])
    rows = []
    for model in LLM_VARIANTS:
        if model not in set(summary["model"]):
            continue
        llm, confidence, prompt = model.split("_")[0], "_".join(model.split("_")[1:3]), model.split("_")[3]
        out = {
            "model": model,
            "llm": llm,
            "confidence": confidence,
            "prompt": prompt,
            "validation_macro_f1": classification.loc[(model, "validation"), "macro_f1"].item(),
            "test_macro_f1": classification.loc[(model, "test"), "macro_f1"].item(),
            "test_auroc": classification.loc[(model, "test"), "auroc"].item(),
            "test_mean_answer_mass": runs["mean_mass"].get((llm, confidence, prompt), np.nan),
        }
        for method, metric in [
            ("M0", "coverage"),
            ("M0", "c1"),
            ("M0", "coverage_pi=0.1"),
            ("M1", "c1"),
            ("M1", "coverage_pi=0.1"),
            ("M1", "workload_per_1000_pi=0.1"),
            ("M1", "uncertainty_referrals_per_1000_pi=0.1"),
        ]:
            for statistic in ["median", "q025", "q975"]:
                out[f"{method}_{metric}_{statistic}"] = lookup(
                    summary, model, "clean", method, metric, statistic=statistic
                )
        rows.append(out)
    pd.DataFrame(rows).to_csv(TABLES / "table4_llms.csv", index=False)


def jitter_table(summary: pd.DataFrame) -> None:
    """
    Tie jitter (S-verb / S-cons runs, whose scores tie): each method next to its +jitter variant.
    """

    jitter_models = sorted(summary[summary["method"].str.endswith("+jitter")]["model"].unique())
    part = summary[
        summary["model"].isin(jitter_models)
        & summary["metric"].isin(["coverage", "c0", "c1", "workload_per_1000_pi=0.1"])
    ].copy()
    part["base_method"] = part["method"].str.removesuffix("+jitter")
    part["variant"] = np.where(part["method"].str.endswith("+jitter"), "jitter", "no_jitter")
    part = part[part["base_method"].isin(part.loc[part["variant"] == "jitter", "base_method"].unique())]
    wide = part.pivot_table(
        index=["model", "condition", "alpha", "base_method", "metric"], columns="variant", values="median"
    ).reset_index()
    wide["jitter_minus_no_jitter"] = wide["jitter"] - wide["no_jitter"]
    wide.to_csv(TABLES / "appendix_jitter.csv", index=False)


def copy_fixed_split_tables(dreaddit: dict) -> None:
    """
    Appendix tables that need no resplit aggregation: official-test permutation p-values (fixed split, per seed) and
    LLM answer-token mass / unparsed rows per split and condition.
    """

    output_dir = Path(dreaddit["evaluation"]["output_dir"])
    pd.read_csv(output_dir / "official_test_permutation.csv").to_csv(
        TABLES / "appendix_official_test_permutation.csv", index=False
    )
    runs = [
        pd.read_csv(Path(dreaddit["llm"]["output_dir"]) / f"{llm}_run_summary.csv").assign(dataset="dreaddit", llm=llm)
        for llm in LLMS
    ]
    # SDCNL's only LLM run (its config's llm.output_dir; this builder receives the Dreaddit config only).
    runs.append(pd.read_csv("results/llm/sdcnl/qwen_run_summary.csv").assign(dataset="sdcnl", llm="qwen"))
    pd.concat(runs, ignore_index=True).to_csv(TABLES / "appendix_llm_mass.csv", index=False)
