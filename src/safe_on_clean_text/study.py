from __future__ import annotations

# Model sets of the hypotheses: H1-H3 on Dreaddit (majority = more than half of 7), R1-R3 on SDCNL
# (majority = at least 2 of 3). LLMs with prompt P1; encoders with all 5 seeds.
# Paper aggregation: every table and figure uses the primary aggregation (safe_on_clean_text.evaluation.aggregation:
# per resplit the metric is averaged over a model's seeds, then the median and 2.5-97.5% over resplits; the between-seed
# spread is the min-max of the per-seed medians). The pooled seed x resplit version appears only in the sensitivity table.
DREADDIT_MODELS = [
    "tfidf_logreg",
    "distilbert",
    "roberta",
    "mentalroberta",
    "qwen_s_tok_P1",
    "gemma_s_tok_P1",
    "gpt_s_tok_P1",
]


SDCNL_MODELS = ["tfidf_logreg", "distilbert", "qwen_s_tok_P1"]


ENSEMBLES = ["distilbert_ensemble", "roberta_ensemble", "mentalroberta_ensemble"]


LLMS = ["gpt", "qwen", "gemma"]


LLM_VARIANTS = [
    f"{llm}_{variant}" for llm in LLMS for variant in ["s_tok_P1", "s_verb_P1", "s_cons_P1", "s_tok_P2", "s_tok_P3"]
]


ALPHA = 0.1
