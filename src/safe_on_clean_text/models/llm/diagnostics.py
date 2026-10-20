from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

LLM_PREFIXES = ("gpt_", "qwen_", "gemma_")
RUN_KEYS = [
    "model_name",
    "conditions",
    "unparsed_rows",
    "excluded_pool_ids",
    "excluded_validation_rows",
    "distinct_score_share_clean",
    "exact_0_or_1_share_clean",
]


def diagnostics(config: dict) -> None:
    """
    Per LLM run: unparsed answers and excluded ids, tie diagnostics of the scores (run_meta.json of the export) and the
    validation temperature and descriptive KS p-value of the evaluation; written to <llm.output_dir>/diagnostics.csv.
    """

    runs_dir = Path(config["paths"]["runs_dir"]) / config["data"]["name"]
    report = pd.read_csv(Path(config["evaluation"]["output_dir"]) / "diagnostics.csv")
    rows: list[tuple[str, str, str, Any]] = []
    for meta_path in sorted(runs_dir.glob("*/seed_*/run_meta.json")):
        if not meta_path.parent.parent.name.startswith(LLM_PREFIXES):
            continue
        meta = json.loads(meta_path.read_text())
        scope = meta["model"]
        rows += [("runs", scope, key, meta[key]) for key in RUN_KEYS]
        for _, row in report[report["scope"] == f"{scope}/seed={meta['seed']}"].iterrows():
            if row["key"] in ("validation_nll_temperature", "ks_pvalue"):
                rows.append(("runs", scope, row["key"], row["value"]))
    table = pd.DataFrame(rows, columns=["section", "scope", "key", "value"])
    output = Path(config["llm"]["output_dir"]) / "diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
