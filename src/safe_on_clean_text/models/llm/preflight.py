from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_on_clean_text.models.llm.cache import AnswerCache
from safe_on_clean_text.models.llm.clients import APIModel, LocalModel
from safe_on_clean_text.models.llm.run import answer_examples

CHECK_PROMPT = "P1"


def summary_rows(name: str, method: str, rows: list[dict[str, Any]], labels: np.ndarray) -> list[tuple[str, str, Any]]:
    """
    Mean mass (S-tok), unparsed share (S-verb / S-cons), accuracy at p(stress) > 0.5, seconds and tokens per example.
    """

    frame = pd.DataFrame(rows)
    scope = f"{name}/{method}" + (f"/prefix={rows[0]['answer_prefix']!r}" if method != "s_cons" else "")
    p_stress = frame["p_stress"].to_numpy(dtype=float)
    out = [
        (scope, "n", len(frame)),
        (scope, "accuracy_at_0.5", round(float(np.mean((p_stress > 0.5) == labels)), 4)),
        (scope, "seconds_per_example", round(float(frame["seconds"].mean()), 4)),
    ]
    if method == "s_tok":
        out += [
            (scope, "mean_mass", round(float(frame["mass"].mean()), 4)),
            (scope, "min_mass", round(float(frame["mass"].min()), 4)),
        ]
    if method == "s_verb":
        out.append((scope, "unparsed_share", round(float(np.isnan(p_stress).mean()), 4)))
    if "n_missing_yes" in frame:
        out.append(
            (scope, "examples_any_variant_missing", int(((frame["n_missing_yes"] + frame["n_missing_no"]) > 0).sum()))
        )
        out.append(
            (scope, "examples_label_missing", int(((frame["n_missing_yes"] == 4) | (frame["n_missing_no"] == 4)).sum()))
        )
    if method == "s_cons":
        out.append((scope, "all_unrecognized_share", round(float(frame["all_unrecognized"].mean()), 4)))
        out.append((scope, "mean_recognized_of_n", round(float(frame["n_recognized"].mean()), 2)))
    if "n_prompt_tokens" in frame:
        out.append((scope, "mean_prompt_tokens", round(float(frame["n_prompt_tokens"].mean()), 1)))
    if "n_completion_tokens" in frame:
        out.append((scope, "mean_completion_tokens", round(float(frame["n_completion_tokens"].mean()), 2)))
    return out


def preflight(config: dict, models: list[str], methods: list[str] | None = None) -> None:
    """
    Answer-mass and parsing check of each LLM and confidence method on a validation sample, with the run-time
    and cost estimate of the run plan -> preflight.csv.
    """

    llm_config = config["llm"]
    data_dir = Path(config["data"]["processed_dir"])
    cache_dir = Path(llm_config["cache_dir"])
    output_dir = Path(llm_config["output_dir"])
    threshold = float(llm_config["mass_threshold"])
    price = llm_config["api_price_per_million"]

    validation = pd.read_csv(data_dir / "validation.csv")
    sample = validation.sample(n=int(llm_config["check_examples"]), random_state=int(config["seed"])).sort_values("id")
    examples = [(int(i), "validation", "clean", str(t)) for i, t in zip(sample["id"], sample["text"])]
    labels = sample["label"].to_numpy()

    # Run volume (config): clean validation + calibration and test in the listed conditions, per prompt.
    n_pool = sum(len(pd.read_csv(data_dir / f"{name}.csv")) for name in ["calibration", "test"])
    n_conditions = {"all": len(config["conditions"])}
    volume = {method: 0 for method in ["s_tok", "s_verb", "s_cons"]}
    for entry in llm_config["run_plan"]:
        conditions = entry["conditions"]
        count = n_conditions["all"] if conditions == "all" else len(conditions)
        volume[entry["method"]] += len(entry["prompts"]) * (len(validation) + count * n_pool)

    report: list[tuple[str, str, str, Any]] = [
        ("volume", "plan", f"{method}_examples", n) for method, n in volume.items()
    ]
    for name in models:
        if name == "gpt":
            model, model_name, methods = (
                APIModel(llm_config["api_model"], llm_config),
                llm_config["api_model"],
                ["s_tok", "s_verb", "s_cons"],
            )
        else:
            model_name = llm_config["local_models"][name]
            model, methods = LocalModel(model_name, llm_config), ["s_tok", "s_verb"]

        for method in [m for m in methods if methods is None or m in methods]:
            cache = AnswerCache(cache_dir / name / f"{method}.jsonl.gz")
            try:
                rows = answer_examples(model, model_name, method, CHECK_PROMPT, examples, llm_config, cache)
            except Exception as error:  # e.g. the API rejecting logprobs: reported, not hidden
                report.append(("error", f"{name}/{method}", type(error).__name__, str(error)[:300]))
                continue
            report += [("check", scope, key, value) for scope, key, value in summary_rows(name, method, rows, labels)]

            if method == "s_tok" and name != "gpt":
                mass = float(np.mean([row["mass"] for row in rows]))
                if mass < threshold:
                    prefixed = answer_examples(
                        model,
                        model_name,
                        method,
                        CHECK_PROMPT,
                        examples,
                        llm_config,
                        cache,
                        answer_prefix=llm_config["answer_prefix"],
                    )
                    report += [("check", s, k, v) for s, k, v in summary_rows(name, method, prefixed, labels)]
                    mass = float(np.mean([row["mass"] for row in prefixed]))
                    rows = prefixed
                report.append(("decision", name, "s_tok_answer_prefix", rows[0]["answer_prefix"]))
                report.append(("decision", name, "s_tok_mass_ok", mass >= threshold))
            if method == "s_tok" and name == "gpt":
                report.append(("decision", name, "logprobs_accepted", True))
                report.append(
                    ("decision", name, "s_tok_mass_ok", float(np.mean([row["mass"] for row in rows])) >= threshold)
                )

            seconds = float(np.mean([row["seconds"] for row in rows]))
            report.append(
                (
                    "estimate",
                    f"{name}/{method}",
                    "full_volume_hours_sequential",
                    round(volume[method] * seconds / 3600, 2),
                )
            )
            if name == "gpt":
                tokens_in = sum(row["n_prompt_tokens"] for row in rows)
                tokens_out = sum(row["n_completion_tokens"] for row in rows)
                cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6
                report.append(("cost", f"{name}/{method}", "check_usd", round(cost, 5)))
                report.append(
                    ("cost", f"{name}/{method}", "full_volume_usd", round(cost / len(rows) * volume[method], 2))
                )

        report.append(("calls", name, "model_calls_this_run", model.calls))

    table = pd.DataFrame(report, columns=["section", "scope", "key", "value"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "preflight.csv", index=False)
    print(table.to_string(index=False))
