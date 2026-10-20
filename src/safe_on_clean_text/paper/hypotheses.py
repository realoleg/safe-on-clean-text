from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.conformal.metrics import reference_band
from safe_on_clean_text.evaluation.aggregation import main_alpha_rows
from safe_on_clean_text.paper import TABLES, display_number
from safe_on_clean_text.study import ALPHA, DREADDIT_MODELS, LLMS, SDCNL_MODELS

DREADDIT_NAMES = {"H1": "H1", "H2": "H2", "H3a": "H3a", "H3b": "H3b", "H3c": "H3c"}
SDCNL_NAMES = {"H1": "R1", "H2": "R2", "H3a": "R3a", "H3b": "R3b", "H3c": "R3c"}
PRIMARY_COLUMNS = {
    "hypothesis": "hypothesis",
    "scope": "scope",
    "quantity": "quantity",
    "criterion": "criterion",
    "value_primary": "value",
    "margin_primary": "margin",
    "meets_primary": "meets",
    "seed_spread_min_max_of_medians": "seed_spread_min_max_of_medians",
}
SENSITIVITY_COLUMNS = [
    "hypothesis",
    "scope",
    "quantity",
    "criterion",
    "value_primary",
    "meets_primary",
    "value_sensitivity_pooled",
    "meets_sensitivity_pooled",
]


def hypothesis_table(dreaddit: dict, sdcnl: dict) -> None:
    """
    Table 6: every criterion of H1-H6 (Dreaddit) and R1-R3 (SDCNL) under the primary aggregation, and the appendix table of
    the model counts and verdicts that differ under the pooled seed x resplit aggregation.
    """

    table = pd.DataFrame(
        _coverage_criteria(dreaddit, DREADDIT_MODELS, DREADDIT_NAMES, majority=len(DREADDIT_MODELS) // 2 + 1)
        + _domain_shift_criterion(dreaddit)
        + _workload_cost_criterion(dreaddit)
        + _agreement_criterion(dreaddit)
        + _coverage_criteria(sdcnl, SDCNL_MODELS, SDCNL_NAMES, majority=2)
    )
    primary = table[list(PRIMARY_COLUMNS)].rename(columns=PRIMARY_COLUMNS)
    primary.to_csv(TABLES / "table6_hypotheses.csv", index=False)
    tallies = table[table["scope"] == "tally"]
    verdicts = table[(table["scope"] == "verdict") & table["meets_sensitivity_pooled"].notna()]
    differs = pd.concat(
        [
            tallies[tallies["value_primary"] != tallies["value_sensitivity_pooled"]],
            verdicts[verdicts["meets_primary"].astype(bool) != verdicts["meets_sensitivity_pooled"].astype(bool)],
        ]
    )
    differs[SENSITIVITY_COLUMNS].to_csv(TABLES / "appendix_table6_sensitivity_pooled.csv", index=False)
    print(primary[primary["scope"] == "verdict"][["hypothesis", "quantity", "meets"]].to_string(index=False))
    print("\nDiffers under the pooled aggregation:")
    print(
        differs[["hypothesis", "scope", "quantity", "value_primary", "value_sensitivity_pooled"]].to_string(index=False)
    )


class SeedAveraged:
    """
    One per-(seed, replicate) quantity under both aggregations: primary = mean over seeds per replicate, then quantiles
    over replicates; pooled = quantiles over all seed x replicate values (sensitivity); spread = min-max of the per-seed
    medians.
    """

    def __init__(self, series: pd.Series, replicate_level: str = "replicate", seed_level: str = "seed") -> None:
        self.primary = series.groupby(level=replicate_level).mean()
        self.pooled = series
        self.per_seed = series.groupby(level=seed_level).median()

    def median(self, pooled: bool = False) -> float:
        return float((self.pooled if pooled else self.primary).median())

    def quantile(self, q: float, pooled: bool = False) -> float:
        return float(np.quantile(self.pooled if pooled else self.primary, q))

    def spread(self) -> str:
        return (
            ""
            if len(self.per_seed) < 2
            else f"{display_number(self.per_seed.min())} – {display_number(self.per_seed.max())}"
        )


def _row(
    hypothesis: str,
    scope: str,
    quantity: str,
    criterion: str,
    primary: Any,
    margin: float | None,
    meets: Any,
    pooled: Any = None,
    meets_pooled: Any = None,
    spread: str = "",
) -> dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "scope": scope,
        "quantity": quantity,
        "criterion": criterion,
        "value_primary": primary,
        "margin_primary": margin,
        "meets_primary": meets,
        "value_sensitivity_pooled": pooled,
        "meets_sensitivity_pooled": meets_pooled,
        "seed_spread_min_max_of_medians": spread,
    }


def _tally_row(hypothesis: str, key: str, primary: list[bool], pooled: list[bool], majority: int) -> dict[str, Any]:
    return _row(
        hypothesis,
        "tally",
        f"models meeting ({hypothesis[0]}{key[1:]})",  # SDCNL rows R1-R3 replicate H1-H3
        f">= {majority}",
        f"{sum(primary)} of {len(primary)}",
        sum(primary) - majority,
        sum(primary) >= majority,
        f"{sum(pooled)} of {len(pooled)}",
        sum(pooled) >= majority,
    )


def _coverage_criteria(config: dict, models: list[str], names: dict[str, str], majority: int) -> list[dict[str, Any]]:
    """
    H1, H2, H3a, H3b and H3c (R1-R3 on SDCNL) per model, then the model counts and verdicts. Margins are positive when a
    criterion is met; values keep full precision.
    """

    rows: list[dict[str, Any]] = []
    met: dict[str, dict[str, list[bool]]] = {
        key: {"primary": [], "pooled": []} for key in ["H1", "H2a", "H2b", "H3a", "H3b", "H3c"]
    }

    def record(key: str, outcome: tuple[bool, bool]) -> None:
        met[key]["primary"].append(bool(outcome[0]))
        met[key]["pooled"].append(bool(outcome[1]))

    for model in models:
        long = main_alpha_rows(config, model)

        def part(condition: str, method: str) -> pd.DataFrame:
            return (
                long[(long["condition"] == condition) & (long["method"] == method)]
                .set_index(["seed", "replicate"])
                .sort_index()
            )

        record("H1", _marginal_valid_class_gap(rows, names["H1"], model, part("clean", "M0")))
        prevalence_m0, prevalence_m1 = _prevalence_coverage(
            rows, names["H2"], model, part("clean", "M0"), part("clean", "M1")
        )
        record("H2a", prevalence_m0)
        record("H2b", prevalence_m1)
        m0_head, m1_head = part("head_25", "M0"), part("head_25", "M1")
        record("H3a", _truncated_risk_coverage(rows, names["H3a"], model, m0_head))
        record("H3b", _class_conditional_versus_marginal(rows, names["H3b"], model, m0_head, m1_head))
        record("H3c", _mixed_calibration_in_band(rows, names["H3c"], model, config["conditions"], part))

    for key, values in met.items():
        hypothesis = names[key[:2]] if key in ["H2a", "H2b"] else names[key]
        rows.append(_tally_row(hypothesis, key, values["primary"], values["pooled"], majority))
    for key in ["H1", "H3a", "H3b", "H3c"]:
        primary, pooled = sum(met[key]["primary"]) >= majority, sum(met[key]["pooled"]) >= majority
        rows.append(
            _row(names[key], "verdict", "confirmed", f"majority >= {majority}", primary, None, primary, pooled, pooled)
        )
    primary = sum(met["H2a"]["primary"]) >= majority and sum(met["H2b"]["primary"]) >= majority
    pooled = sum(met["H2a"]["pooled"]) >= majority and sum(met["H2b"]["pooled"]) >= majority
    rows.append(
        _row(
            names["H2"],
            "verdict",
            "confirmed",
            f"M0 part and M1 part each >= {majority}",
            primary,
            None,
            primary,
            pooled,
            pooled,
        )
    )
    return rows


def _marginal_valid_class_gap(rows: list, hypothesis: str, model: str, m0_clean: pd.DataFrame) -> tuple[bool, bool]:
    """
    H1: clean marginal coverage inside its reference band and median |c1 - c0| >= 0.05.
    """

    low, high = reference_band(
        int(round(m0_clean["n_calibration"].median())), int(round(m0_clean["n"].median())), ALPHA
    )
    coverage, gap = SeedAveraged(m0_clean["coverage"]), SeedAveraged((m0_clean["c1"] - m0_clean["c0"]).abs())
    ok = {mode: low <= coverage.median(mode) <= high and gap.median(mode) >= 0.05 for mode in [False, True]}
    rows += [
        _row(
            hypothesis,
            model,
            "M0 clean median coverage",
            f"in band [{display_number(low)}, {display_number(high)}]",
            coverage.median(),
            min(coverage.median() - low, high - coverage.median()),
            low <= coverage.median() <= high,
            coverage.median(True),
            low <= coverage.median(True) <= high,
            coverage.spread(),
        ),
        _row(
            hypothesis,
            model,
            "median |c1 - c0| (M0 clean)",
            ">= 0.05",
            gap.median(),
            gap.median() - 0.05,
            gap.median() >= 0.05,
            gap.median(True),
            gap.median(True) >= 0.05,
            gap.spread(),
        ),
        _row(hypothesis, model, "meets", "both", ok[False], None, ok[False], ok[True], ok[True]),
    ]
    return ok[False], ok[True]


def _prevalence_coverage(
    rows: list, hypothesis: str, model: str, m0_clean: pd.DataFrame, m1_clean: pd.DataFrame
) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
    """
    H2: coverage reweighted to pi = 0.05 and 0.10 (pi * c1 + (1 - pi) * c0 per replicate) is below 0.87 under M0 and
    within 0.90 +/- 0.02 under M1. Returns the M0 part and the M1 part.
    """

    ok_m0, ok_m1 = {False: True, True: True}, {False: True, True: True}
    for p in [0.05, 0.1]:
        m0 = SeedAveraged(p * m0_clean["c1"] + (1 - p) * m0_clean["c0"])
        for mode in [False, True]:
            ok_m0[mode] &= m0.median(mode) < 0.87
        rows.append(
            _row(
                hypothesis,
                model,
                f"M0 median coverage at pi={p}",
                "< 0.87",
                m0.median(),
                0.87 - m0.median(),
                m0.median() < 0.87,
                m0.median(True),
                m0.median(True) < 0.87,
                m0.spread(),
            )
        )
    for p in [0.05, 0.1]:
        m1 = SeedAveraged(p * m1_clean["c1"] + (1 - p) * m1_clean["c0"])
        for mode in [False, True]:
            ok_m1[mode] &= abs(m1.median(mode) - 0.90) <= 0.02
        rows.append(
            _row(
                hypothesis,
                model,
                f"M1 median coverage at pi={p}",
                "within 0.90 +/- 0.02",
                m1.median(),
                0.02 - abs(m1.median() - 0.90),
                abs(m1.median() - 0.90) <= 0.02,
                m1.median(True),
                abs(m1.median(True) - 0.90) <= 0.02,
                m1.spread(),
            )
        )
    return (ok_m0[False], ok_m0[True]), (ok_m1[False], ok_m1[True])


def _truncated_risk_coverage(rows: list, hypothesis: str, model: str, m0_head: pd.DataFrame) -> tuple[bool, bool]:
    """
    H3a: after truncation to the first 25% of words, the risk-class coverage under M0 falls below its band.
    """

    c1_low, c1_high = reference_band(
        int(round(m0_head["n_calibration_1"].median())), int(round(m0_head["n_1"].median())), ALPHA
    )
    c1_m0 = SeedAveraged(m0_head["c1"])
    rows.append(
        _row(
            hypothesis,
            model,
            "head_25 M0 median c1",
            f"< band low (band [{display_number(c1_low)}, {display_number(c1_high)}])",
            c1_m0.median(),
            c1_low - c1_m0.median(),
            c1_m0.median() < c1_low,
            c1_m0.median(True),
            c1_m0.median(True) < c1_low,
            c1_m0.spread(),
        )
    )
    return c1_m0.median() < c1_low, c1_m0.median(True) < c1_low


def _class_conditional_versus_marginal(
    rows: list, hypothesis: str, model: str, m0_head: pd.DataFrame, m1_head: pd.DataFrame
) -> tuple[bool, bool]:
    """
    H3b: after truncation, M1's risk-class coverage is lower than M0's and the paired difference (within seed and
    resplit; averaged over seeds per resplit) has its 97.5% quantile below 0.
    """

    c1_m0, c1_m1 = SeedAveraged(m0_head["c1"]), SeedAveraged(m1_head["c1"])
    difference = SeedAveraged(m1_head["c1"] - m0_head["c1"])
    ok = {
        mode: c1_m1.median(mode) < c1_m0.median(mode) and difference.quantile(0.975, mode) < 0 for mode in [False, True]
    }
    rows += [
        _row(
            hypothesis,
            model,
            "head_25 median c1: M1 vs M0",
            "M1 < M0",
            f"{display_number(c1_m1.median())} vs {display_number(c1_m0.median())}",
            c1_m0.median() - c1_m1.median(),
            c1_m1.median() < c1_m0.median(),
            f"{display_number(c1_m1.median(True))} vs {display_number(c1_m0.median(True))}",
            c1_m1.median(True) < c1_m0.median(True),
            f"M1 {c1_m1.spread()}; M0 {c1_m0.spread()}",
        ),
        _row(
            hypothesis,
            model,
            "paired M1 - M0 c1: median [2.5%, 97.5%]",
            "upper end < 0",
            f"{display_number(difference.median())} [{display_number(difference.quantile(0.025))}, "
            f"{display_number(difference.quantile(0.975))}]",
            -difference.quantile(0.975),
            difference.quantile(0.975) < 0,
            f"{display_number(difference.median(True))} [{display_number(difference.quantile(0.025, True))}, "
            f"{display_number(difference.quantile(0.975, True))}]",
            difference.quantile(0.975, True) < 0,
            difference.spread(),
        ),
        _row(hypothesis, model, "meets", "both", ok[False], None, ok[False], ok[True], ok[True]),
    ]
    return ok[False], ok[True]


def _mixed_calibration_in_band(
    rows: list, hypothesis: str, model: str, conditions: list[str], part
) -> tuple[bool, bool]:
    """
    H3c: M3-mix keeps the risk-class coverage inside its band in every condition (plus the descriptive count of the
    conditions below and above the band).
    """

    margins, margins_pooled, below, above = [], [], 0, 0
    for condition in conditions:
        mix = part(condition, "M3-mix")
        low, high = reference_band(int(round(mix["n_calibration_1"].median())), int(round(mix["n_1"].median())), ALPHA)
        c1 = SeedAveraged(mix["c1"])
        margins.append(min(c1.median() - low, high - c1.median()))
        margins_pooled.append(min(c1.median(True) - low, high - c1.median(True)))
        below += c1.median() < low
        above += c1.median() > high
        rows.append(
            _row(
                hypothesis,
                model,
                f"{condition} M3-mix median c1",
                f"in band [{display_number(low)}, {display_number(high)}]",
                c1.median(),
                margins[-1],
                margins[-1] >= 0,
                c1.median(True),
                margins_pooled[-1] >= 0,
                c1.spread(),
            )
        )
    rows += [
        _row(
            hypothesis,
            model,
            "inside band in all conditions",
            "all 11",
            min(margins) >= 0,
            min(margins),
            min(margins) >= 0,
            min(margins_pooled) >= 0,
            min(margins_pooled) >= 0,
        ),
        _row(
            hypothesis,
            model,
            "conditions below / above the band (descriptive)",
            "",
            f"{below} below / {above} above",
            None,
            None,
        ),
    ]
    return min(margins) >= 0, min(margins_pooled) >= 0


def _domain_shift_criterion(config: dict) -> list[dict[str, Any]]:
    """
    H4: leave-one-domain-out DistilBERT (3 seeds averaged per calibration bootstrap): the worst held-out domain's median
    M0 coverage is below its band.
    """

    output_dir = Path(config["evaluation"]["output_dir"])
    rows: list[dict[str, Any]] = []
    lodo = pd.read_parquet(output_dir / "lodo_long.parquet")
    lodo = lodo[(lodo["model"] == "distilbert") & (lodo["method"] == "M0") & (lodo["alpha"] == ALPHA)]
    bands = pd.read_csv(output_dir / "lodo_summary.csv")
    bands = bands[
        (bands["model"] == "distilbert")
        & (bands["seed"].astype(str) == "all")
        & (bands["method"] == "M0")
        & (bands["alpha"] == ALPHA)
        & (bands["metric"] == "coverage")
    ].set_index("domain")["band_low"]
    medians = {}
    for domain, frame in lodo.groupby("domain"):
        values = SeedAveraged(frame.set_index(["seed", "bootstrap"])["coverage"], replicate_level="bootstrap")
        medians[domain] = (values.median(), values.median(True))
        rows.append(
            _row(
                "H4",
                f"distilbert/{domain}",
                "LODO M0 median coverage",
                f"< band low {display_number(bands[domain])}",
                values.median(),
                bands[domain] - values.median(),
                values.median() < bands[domain],
                values.median(True),
                values.median(True) < bands[domain],
                values.spread(),
            )
        )
    worst, worst_pooled = min(medians, key=lambda d: medians[d][0]), min(medians, key=lambda d: medians[d][1])
    primary, pooled = medians[worst][0] < bands[worst], medians[worst_pooled][1] < bands[worst_pooled]
    rows.append(
        _row(
            "H4",
            "verdict",
            f"worst domain (primary {worst}, pooled {worst_pooled}) below its band",
            "worst-domain median < band low",
            primary,
            bands[worst] - medians[worst][0],
            primary,
            pooled,
            pooled,
        )
    )
    return rows


def _workload_cost_criterion(config: dict) -> list[dict[str, Any]]:
    """
    H5 (alpha 0.10, M1, clean): workload per 1,000 at pi = 0.10 is higher for the best zero-shot LLM (S-tok, by
    validation macro-F1) than for RoBERTa, and higher with S-verb than with S-tok for at least 2 of the 3 LLMs.
    Referrals (full or empty sets) per 1,000 are reported as a secondary, descriptive measure.
    """

    output_dir = Path(config["evaluation"]["output_dir"])
    rows: list[dict[str, Any]] = []
    classification = pd.read_csv(output_dir / "classification.csv")
    validation_f1 = classification[
        (classification["split"] == "validation")
        & (classification["condition"] == "clean")
        & (classification["probabilities"] == "raw")
    ].set_index("model")["macro_f1"]
    best = max(LLMS, key=lambda llm: validation_f1[f"{llm}_s_tok_P1"])
    for llm in LLMS:
        rows.append(
            _row(
                "H5",
                f"{llm}_s_tok_P1",
                "validation macro-F1 (selects the best LLM)",
                "max",
                float(validation_f1[f"{llm}_s_tok_P1"]),
                None,
                llm == best,
            )
        )

    def per_1000(model: str, measure: str) -> pd.Series:
        frame = main_alpha_rows(config, model)
        frame = frame[(frame["condition"] == "clean") & (frame["method"] == "M1")].set_index(["seed", "replicate"])
        return 1000 * (0.1 * frame[f"{measure}_1"] + 0.9 * frame[f"{measure}_0"])

    verdict = {}
    for measure, label in [
        ("workload", "workload per 1,000 (verdict measure)"),
        ("uncertainty_referral", "referrals per 1,000 (secondary, descriptive)"),
    ]:
        llm, roberta = SeedAveraged(per_1000(f"{best}_s_tok_P1", measure)), SeedAveraged(per_1000("roberta", measure))
        part1 = {mode: llm.median(mode) > roberta.median(mode) for mode in [False, True]}
        rows.append(
            _row(
                "H5",
                f"part 1 / {measure}",
                f"{label}: best LLM ({best} S-tok) vs RoBERTa",
                "best LLM > RoBERTa",
                f"{display_number(llm.median())} vs {display_number(roberta.median())}",
                llm.median() - roberta.median(),
                part1[False],
                f"{display_number(llm.median(True))} vs {display_number(roberta.median(True))}",
                part1[True],
                f"RoBERTa {roberta.spread()}",
            )
        )
        paired = llm.primary - roberta.primary  # per resplit: LLM (one run) minus RoBERTa averaged over its seeds
        rows.append(
            _row(
                "H5",
                f"part 1 / {measure}",
                f"paired {best} - RoBERTa per resplit: median [2.5%, 97.5%] (descriptive)",
                "",
                f"{display_number(paired.median())} [{display_number(np.quantile(paired, 0.025))}, "
                f"{display_number(np.quantile(paired, 0.975))}]",
                None,
                None,
            )
        )
        verbal_higher = []
        for name in LLMS:
            verb, tok = (
                SeedAveraged(per_1000(f"{name}_s_verb_P1", measure)).median(),
                SeedAveraged(per_1000(f"{name}_s_tok_P1", measure)).median(),
            )
            verbal_higher.append(verb > tok)
            rows.append(
                _row(
                    "H5",
                    f"part 2 / {measure} / {name}",
                    f"{label}: S-verb vs S-tok",
                    "S-verb > S-tok",
                    f"{display_number(verb)} vs {display_number(tok)}",
                    verb - tok,
                    verb > tok,
                )
            )
        part2 = sum(verbal_higher) >= 2
        rows.append(
            _row(
                "H5",
                f"part 2 / {measure}",
                "LLMs with S-verb > S-tok",
                ">= 2 of 3",
                f"{sum(verbal_higher)} of 3",
                sum(verbal_higher) - 2,
                part2,
            )
        )
        verdict[measure] = (part1[False] and part2, part1[True] and part2)
    gpt = " / ".join(
        display_number(SeedAveraged(per_1000(f"gpt_{m}_P1", "workload")).median())
        for m in ["s_tok", "s_verb", "s_cons"]
    )
    rows.append(
        _row("H5", "GPT (descriptive)", "workload per 1,000: S-tok / S-verb / S-cons", "no criterion", gpt, None, None)
    )
    rows.append(
        _row(
            "H5",
            "verdict",
            "confirmed (workload per 1,000)",
            "part 1 and part 2",
            verdict["workload"][0],
            None,
            verdict["workload"][0],
            verdict["workload"][1],
            verdict["workload"][1],
        )
    )
    rows.append(
        _row(
            "H5",
            "verdict under the secondary measure (descriptive)",
            "confirmed (referrals)",
            "part 1 and part 2",
            verdict["uncertainty_referral"][0],
            None,
            None,
            verdict["uncertainty_referral"][1],
            None,
        )
    )
    return rows


def _agreement_criterion(config: dict) -> list[dict[str, Any]]:
    """
    H6: for DistilBERT, the AUROC of "full M0 set -> non-unanimous label" has median > 0.60 and 2.5% quantile > 0.5
    over the resplits (seeds averaged per resplit).
    """

    output_dir = Path(config["evaluation"]["output_dir"])
    auroc = pd.read_parquet(output_dir / "mechanisms" / "raw" / "auroc.parquet")
    auroc = auroc[(auroc["model"] == "distilbert") & (auroc["method"] == "M0") & (auroc["score"] == "full_set")]
    values = SeedAveraged(auroc.set_index(["seed", "replicate"])["auroc"])
    primary = values.median() > 0.60 and values.quantile(0.025) > 0.5
    pooled = values.median(True) > 0.60 and values.quantile(0.025, True) > 0.5
    return [
        _row(
            "H6",
            "distilbert",
            "AUROC full set (M0) -> non-unanimous label, median",
            "> 0.60",
            values.median(),
            values.median() - 0.60,
            values.median() > 0.60,
            values.median(True),
            values.median(True) > 0.60,
            values.spread(),
        ),
        _row(
            "H6",
            "distilbert",
            "AUROC, 2.5% quantile over resplits",
            "> 0.5",
            values.quantile(0.025),
            values.quantile(0.025) - 0.5,
            values.quantile(0.025) > 0.5,
            values.quantile(0.025, True),
            values.quantile(0.025, True) > 0.5,
        ),
        _row("H6", "verdict", "confirmed", "both", primary, None, primary, pooled, pooled),
    ]


def matched_workload_hypotheses_table(
    h7: pd.DataFrame, h8: pd.DataFrame, h9: pd.DataFrame, h10: pd.DataFrame, conditions: list[str]
) -> pd.DataFrame:
    """
    Table 7: the criteria of H7-H10 (fixed before running) per model, the model / cell counts and the verdicts, in the
    Table 6 format, from the aggregated analysis tables (primary aggregation; alpha 0.10, pi = 0.10).
    """

    degraded = [condition for condition in conditions if condition != "clean"]
    majority = 4

    def get(table: pd.DataFrame, model: str, condition: str, metric: str, **filters: str) -> pd.Series:
        mask = (table["model"] == model) & (table["condition"] == condition) & (table["metric"] == metric)
        for column, value in filters.items():
            mask &= table[column] == value
        assert mask.sum() == 1, (model, condition, metric, filters)
        return table[mask].iloc[0]

    def spread(row: pd.Series) -> str:
        return (
            "" if pd.isna(row["seed_min"]) else f"{display_number(row['seed_min'])} – {display_number(row['seed_max'])}"
        )

    def interval(row: pd.Series) -> str:
        return f"{display_number(row['median'])} [{display_number(row['q025'])}, {display_number(row['q975'])}]"

    rows: list[dict[str, Any]] = []

    # H7: losing cell = 97.5% end of the paired c1(M1 at alpha') - c1(M0) below 0.
    losing_total, losing_median = 0, 0
    for model in DREADDIT_MODELS:
        alpha_prime = get(h7, model, "clean", "alpha_prime")
        unmatched = get(h7, model, "clean", "unmatched")["mean"]
        gap = get(h7, model, "clean", "workload_gap")
        differences = [get(h7, model, condition, "c1_difference") for condition in degraded]
        losing = sum(row["q975"] < 0 for row in differences)
        losing_total += losing
        losing_median += sum(row["median"] < 0 for row in differences)
        rows += [
            _row(
                "H7",
                model,
                "alpha' median [2.5%, 97.5%]; share unmatched",
                "",
                f"{interval(alpha_prime)}; {display_number(unmatched)}",
                None,
                None,
                spread=spread(alpha_prime),
            ),
            _row(
                "H7",
                model,
                "clean workload gap M1(alpha') - M0 per 1,000",
                "",
                interval(gap),
                None,
                None,
                spread=spread(gap),
            ),
            _row(
                "H7",
                model,
                "losing degraded cells (97.5% end of paired c1 difference < 0)",
                "of 10",
                f"{losing} of 10",
                None,
                None,
            ),
        ]
    rows += [
        _row(
            "H7", "tally", "losing cells", ">= 36 of 70", f"{losing_total} of 70", losing_total - 36, losing_total >= 36
        ),
        _row(
            "H7",
            "tally (descriptive)",
            "cells with median paired c1 difference < 0",
            "",
            f"{losing_median} of 70",
            None,
            None,
        ),
        _row("H7", "verdict", "confirmed", ">= 36 of 70 losing cells", losing_total >= 36, None, losing_total >= 36),
    ]

    # H8: q_M1 < q_M0 in more than half of the (seed, resplit) pairs and median clean c1(M0) above 0.90.
    met, cells_median, cells_upper = [], 0, 0
    for model in DREADDIT_MODELS:
        below = get(h8, model, "clean", "q_M1_below_q_M0")["mean"]
        equal = get(h8, model, "clean", "q_M1_equal_q_M0")["mean"]
        above = get(h8, model, "clean", "q_M1_above_q_M0")["mean"]
        c1 = get(h8, model, "clean", "c1_M0")
        slack = get(h8, model, "clean", "calibration_slack_c1_M0")
        ok = bool(below > 0.5 and c1["median"] > 0.90)
        met.append(ok)
        differences = [get(h8, model, condition, "c1_difference_M1_minus_M0") for condition in degraded]
        cells_median += sum(row["median"] < 0 for row in differences)
        cells_upper += sum(row["q975"] < 0 for row in differences)
        rows += [
            _row("H8", model, "share of resplits with q_M1 < q_M0", "> 0.5", below, below - 0.5, below > 0.5),
            _row(
                "H8",
                model,
                "shares q_M1 = q_M0 / q_M1 > q_M0 (descriptive)",
                "",
                f"{display_number(equal)} / {display_number(above)}",
                None,
                None,
            ),
            _row(
                "H8",
                model,
                "median clean c1(M0), test half",
                "> 0.90",
                c1["median"],
                c1["median"] - 0.90,
                c1["median"] > 0.90,
                spread=spread(c1),
            ),
            _row(
                "H8",
                model,
                "calibration-half slack c1(M0) - 0.90 (descriptive)",
                "",
                interval(slack),
                None,
                None,
                spread=spread(slack),
            ),
            _row("H8", model, "meets", "both", ok, None, ok),
        ]
    rows += [
        _row(
            "H8",
            "tally",
            "models meeting",
            f">= {majority} of 7",
            f"{sum(met)} of 7",
            sum(met) - majority,
            sum(met) >= majority,
        ),
        _row(
            "H8",
            "tally (descriptive)",
            "degraded cells with M1 < M0: median / 97.5% end below 0",
            "",
            f"{cells_median} / {cells_upper} of 70",
            None,
            None,
        ),
        _row("H8", "verdict", "confirmed", f">= {majority} of 7", sum(met) >= majority, None, sum(met) >= majority),
    ]

    # H9: head_25 median per-resplit ratio M3 / oracle workload at most 1.25.
    met = []
    for model in DREADDIT_MODELS:
        ratio = get(h9, model, "head_25", "workload_ratio_M3_over_oracle")
        met.append(bool(ratio["median"] <= 1.25))
        rows.append(
            _row(
                "H9",
                model,
                "head_25 workload ratio M3 / oracle, median [2.5%, 97.5%]",
                "<= 1.25",
                interval(ratio),
                1.25 - ratio["median"],
                met[-1],
            )
        )
    rows += [
        _row(
            "H9",
            "tally",
            "models meeting",
            f">= {majority} of 7",
            f"{sum(met)} of 7",
            sum(met) - majority,
            sum(met) >= majority,
        ),
        _row("H9", "verdict", "confirmed", f">= {majority} of 7", sum(met) >= majority, None, sum(met) >= majority),
    ]

    # H10: variant a, class-conditional weighted CP; median c1 inside its band in at least 9 of 11 conditions.
    met = []
    for model in DREADDIT_MODELS:
        inside, below, above = 0, 0, 0
        for condition in conditions:
            row = get(h10, model, condition, "c1", variant="a", method="WCP-class")
            inside += row["c1_band_low"] <= row["median"] <= row["c1_band_high"]
            below += row["median"] < row["c1_band_low"]
            above += row["median"] > row["c1_band_high"]
        met.append(inside >= 9)
        rows += [
            _row(
                "H10",
                model,
                "conditions with WCP-class (a) median c1 inside its band",
                ">= 9 of 11",
                f"{inside} of 11",
                inside - 9,
                met[-1],
            ),
            _row(
                "H10",
                model,
                "conditions below / above the band (descriptive)",
                "",
                f"{below} below / {above} above",
                None,
                None,
            ),
        ]
    rows += [
        _row(
            "H10",
            "tally",
            "models meeting",
            f">= {majority} of 7",
            f"{sum(met)} of 7",
            sum(met) - majority,
            sum(met) >= majority,
        ),
        _row("H10", "verdict", "confirmed", f">= {majority} of 7", sum(met) >= majority, None, sum(met) >= majority),
    ]
    return pd.DataFrame(rows)[list(PRIMARY_COLUMNS)].rename(columns=PRIMARY_COLUMNS)


def natural_shift_hypotheses_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Table 8: the criteria of H11-H13 (fixed before running) per model, the model counts and the verdicts, in the Table 6
    format, from the aggregated natural-shift table (calibration on long posts, test on short posts; primary
    aggregation, alpha 0.10).
    """

    majority = 4

    def get(model: str, analysis: str, method: str) -> pd.Series:
        test = (
            "short clean"
            if analysis == "long calibration"
            else table.loc[table["analysis"] == analysis, "test"].iloc[0]
        )
        mask = (
            (table["model"] == model)
            & (table["analysis"] == analysis)
            & (table["test"] == test)
            & (table["method"] == method)
            & (table["metric"] == "c1")
        )
        assert mask.sum() == 1, (model, analysis, method)
        return table[mask].iloc[0]

    def interval(row: pd.Series) -> str:
        return f"{display_number(row['median'])} [{display_number(row['q025'])}, {display_number(row['q975'])}]"

    def spread(row: pd.Series) -> str:
        return (
            "" if pd.isna(row["seed_min"]) else f"{display_number(row['seed_min'])} – {display_number(row['seed_max'])}"
        )

    rows: list[dict[str, Any]] = []
    met: dict[str, list[bool]] = {"H11": [], "H12": [], "H13": []}
    for model in DREADDIT_MODELS:
        m1, m3 = get(model, "long calibration", "M1"), get(model, "long calibration", "M3-mix")
        predicted = get(model, "cross-fitted prediction", "M1")
        low, high = m1["c1_band_low"], m1["c1_band_high"]
        band = f"[{display_number(low)}, {display_number(high)}]"

        met["H11"].append(bool(m1["median"] < low))
        rows.append(
            _row(
                "H11",
                model,
                "short posts M1 median c1 [2.5%, 97.5%]",
                f"< band low (band {band})",
                interval(m1),
                low - m1["median"],
                met["H11"][-1],
                spread=spread(m1),
            )
        )
        error, naive = abs(predicted["median"] - m1["median"]), abs(0.90 - m1["median"])
        met["H12"].append(bool(error <= 0.05 and error < naive))
        rows += [
            _row(
                "H12",
                model,
                "predicted M1 c1, median [2.5%, 97.5%] (observed in H11)",
                "",
                interval(predicted),
                None,
                None,
                spread=spread(predicted),
            ),
            _row(
                "H12",
                model,
                "|predicted - observed| median c1; |0.90 - observed|",
                "<= 0.05 and < naive",
                f"{display_number(error)}; {display_number(naive)}",
                min(0.05 - error, naive - error),
                met["H12"][-1],
            ),
        ]
        met["H13"].append(bool(low <= m3["median"] <= high))
        rows.append(
            _row(
                "H13",
                model,
                "short posts M3-mix median c1 [2.5%, 97.5%]",
                f"in band {band}",
                interval(m3),
                min(m3["median"] - low, high - m3["median"]),
                met["H13"][-1],
                spread=spread(m3),
            )
        )
    for hypothesis, values in met.items():
        rows += [
            _row(
                hypothesis,
                "tally",
                "models meeting",
                f">= {majority} of 7",
                f"{sum(values)} of 7",
                sum(values) - majority,
                sum(values) >= majority,
            ),
            _row(
                hypothesis,
                "verdict",
                "confirmed",
                f">= {majority} of 7",
                sum(values) >= majority,
                None,
                sum(values) >= majority,
            ),
        ]
    order = {"H11": 0, "H12": 1, "H13": 2}
    out = pd.DataFrame(rows)
    out = out.sort_values("hypothesis", key=lambda s: s.map(order), kind="stable")
    return out[list(PRIMARY_COLUMNS)].rename(columns=PRIMARY_COLUMNS).reset_index(drop=True)
