from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from safe_on_clean_text.config import load_config
from safe_on_clean_text.degradations import load_condition_texts, select_condition_texts
from safe_on_clean_text.models.llm.cache import AnswerCache, cache_key, drop_rows_written_since
from safe_on_clean_text.models.llm.clients import APIModel, LocalModel
from safe_on_clean_text.models.llm.confidence import generation_settings
from safe_on_clean_text.models.llm.prompts import build_messages
from safe_on_clean_text.models.metal_guard import run_guarded
from safe_on_clean_text.provenance import timestamp

API_MODEL = "gpt"
CHUNK = 500
# After a Metal error, cache rows written this many seconds before it are removed (they may hold wrong values).
METAL_ERROR_MARGIN_SECONDS = 120


def planned_examples(config: dict, data_dir: Path, conditions: list[str] | str) -> list[tuple[Any, str, str, str]]:
    """
    Clean validation, then calibration and test in each listed condition (texts from the saved conditions file).
    """

    validation = pd.read_csv(data_dir / "validation.csv")
    examples = [(int(i), "validation", "clean", str(t)) for i, t in zip(validation["id"], validation["text"])]
    condition_texts = load_condition_texts(data_dir / "conditions.csv")
    for condition in config["conditions"] if conditions == "all" else conditions:
        for split_name in ["calibration", "test"]:
            ids = pd.read_csv(data_dir / f"{split_name}.csv")["id"]
            texts = select_condition_texts(condition_texts, split_name, condition, ids)
            examples += [(int(i), split_name, condition, str(t)) for i, t in zip(ids, texts)]
    return examples


def summarize_answers(rows: list[dict[str, Any]], method: str, prompt_id: str) -> pd.DataFrame:
    """
    Per (split, condition): n, unparsed answers, S-tok mean mass and shares of rows with mass < 0.9 and < 0.5
    (no exclusions) and, for GPT S-tok, examples whose Yes/No variants were missing from the top
    logprobs (any variant; a whole label).
    """

    frame = pd.DataFrame(rows)
    frame["unparsed"] = frame["p_stress"].isna()
    aggregations = {"n": ("id", "size"), "unparsed": ("unparsed", "sum")}
    if method == "s_tok":
        frame["mass_below_0.9"] = frame["mass"] < 0.9
        frame["mass_below_0.5"] = frame["mass"] < 0.5
        aggregations["mean_mass"] = ("mass", "mean")
        aggregations["share_mass_below_0.9"] = ("mass_below_0.9", "mean")
        aggregations["share_mass_below_0.5"] = ("mass_below_0.5", "mean")
    if "n_missing_yes" in frame:
        frame["any_variant_missing"] = (frame["n_missing_yes"] + frame["n_missing_no"]) > 0
        frame["label_missing"] = (frame["n_missing_yes"] == 4) | (frame["n_missing_no"] == 4)
        aggregations["any_variant_missing"] = ("any_variant_missing", "sum")
        aggregations["label_missing"] = ("label_missing", "sum")
    if method == "s_cons":
        aggregations["all_unrecognized"] = ("all_unrecognized", "sum")
    out = frame.groupby(["split", "condition"], sort=False).agg(**aggregations).reset_index()
    return out.assign(method=method, prompt=prompt_id)


def answer_run_plan(config: dict, model_key: str) -> None:
    llm_config = config["llm"]
    data_dir = Path(config["data"]["processed_dir"])
    cache_dir = Path(llm_config["cache_dir"]) / model_key
    if model_key == API_MODEL:
        model_name = llm_config["api_model"]
        model, workers = APIModel(model_name, llm_config), int(llm_config["api_workers"])
    else:
        model_name = llm_config["local_models"][model_key]
        model, workers = LocalModel(model_name, llm_config), 1
    prefix = llm_config["answer_prefix"] if model_key in llm_config["answer_prefix_models"] else ""

    summaries = []
    for entry in llm_config["run_plan"]:
        method = entry["method"]
        if method == "s_cons" and model_key != API_MODEL:
            continue
        examples = planned_examples(config, data_dir, entry["conditions"])
        cache = AnswerCache(cache_dir / f"{method}.jsonl.gz")
        for prompt_id in entry["prompts"]:
            rows: list[dict[str, Any]] = []
            start = time.perf_counter()
            for offset in range(0, len(examples), CHUNK):
                calls_before = model.calls
                rows += answer_examples(
                    model,
                    model_name,
                    method,
                    prompt_id,
                    examples[offset : offset + CHUNK],
                    llm_config,
                    cache,
                    answer_prefix=prefix if method == "s_tok" else "",
                    workers=workers,
                )
                print(
                    f"[{timestamp()}] {model_key} {method} {prompt_id}: {len(rows)}/{len(examples)} "
                    f"(+{model.calls - calls_before} calls, {time.perf_counter() - start:.0f} s)",
                    flush=True,
                )
            summaries.append(summarize_answers(rows, method, prompt_id))

    summary = pd.concat(summaries, ignore_index=True)
    output = Path(llm_config["output_dir"]) / f"{model_key}_run_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"[{timestamp()}] {model_key}: done, {model.calls} model calls this run; summary {output}", flush=True)
    if model_key == API_MODEL:
        price = llm_config["api_price_per_million"]
        cost = (model.prompt_tokens * price["input"] + model.completion_tokens * price["output"]) / 1e6
        print(
            f"{model_key}: {model.prompt_tokens} prompt + {model.completion_tokens} completion tokens this run, ${cost:.2f}"
        )


def run_model(config_path: str, model: str, worker: bool = False) -> None:
    """
    Answer the config's run plan with one LLM; cached answers make no call. Local models run as a child process under
    the Metal command-buffer guard.
    """

    config = load_config(config_path)
    if model not in config["llm"]["models"]:
        raise SystemExit(f"{model} is not an LLM of {config_path} (llm.models: {config['llm']['models']})")

    if worker or model == API_MODEL:
        answer_run_plan(config, model)
        return

    # Local (MLX) models run as a child under the Metal command-buffer guard.
    n_errors = run_guarded(
        [
            sys.executable,
            "-m",
            "safe_on_clean_text.cli",
            "llm",
            "run",
            "--config",
            config_path,
            "--model",
            model,
            "--worker",
        ]
    )
    if n_errors:
        since = time.time() - METAL_ERROR_MARGIN_SECONDS
        cache_dir = Path(config["llm"]["cache_dir"]) / model
        removed = {path.name: drop_rows_written_since(path, since) for path in sorted(cache_dir.glob("*.jsonl.gz"))}
        raise RuntimeError(
            f"{model}: {n_errors} Metal command-buffer error(s); removed recent cache rows {removed}; stopped."
        )


def answer_examples(
    model: LocalModel | APIModel,
    model_name: str,
    method: str,
    prompt_id: str,
    examples: list[tuple[Any, str, str, str]],
    llm_config: dict,
    cache: AnswerCache,
    answer_prefix: str = "",
    workers: int = 1,
) -> list[dict[str, Any]]:
    """
    Answers for (id, split, condition, text) examples, in order, from the cache when present (no model call).
    Rows hold no text; "seconds" is the wall time of the original call, "written_at" its Unix time.
    workers > 1 answers in parallel threads (API models only).
    """

    settings = generation_settings(llm_config, method, answer_prefix)

    def answer(example: tuple[Any, str, str, str]) -> dict[str, Any]:
        example_id, split_name, condition, text = example
        messages = build_messages(llm_config, prompt_id, method, text)
        key = cache_key(model_name, method, messages, settings)
        row = cache.get(key)
        if row is None:
            start = time.perf_counter()
            result = getattr(model, method)(messages, answer_prefix)
            row = {
                "key": key,
                "id": example_id,
                "split": split_name,
                "condition": condition,
                "prompt": prompt_id,
                "answer_prefix": answer_prefix,
                **result,
                "seconds": round(time.perf_counter() - start, 4),
                "written_at": round(time.time(), 3),
            }
            cache.put(row)
        # Identical prompts (e.g. a degradation that left the text unchanged) share one cached answer;
        # the identity fields always come from this example.
        return {**row, "id": example_id, "split": split_name, "condition": condition}

    if workers == 1:
        return [answer(example) for example in examples]
    assert isinstance(model, APIModel), "Parallel answering is for API models only."
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(answer, examples))
