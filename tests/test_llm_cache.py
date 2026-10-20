import gzip
import json
import math

from conftest import REPO

from safe_on_clean_text.models.llm.cache import cache_key
from safe_on_clean_text.models.llm.confidence import parse_verbal, parse_yes_no

CACHE_FILES = sorted((REPO / "results/llm").rglob("*.jsonl.gz"))


def test_cache_rows_hold_no_text():
    # Dreaddit: GPT 3 (s_tok, s_verb, s_cons), Qwen 2, Gemma 2; SDCNL: Qwen s_tok.
    assert len(CACHE_FILES) == 8
    for path in CACHE_FILES:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                assert len(row["key"]) == 64
                assert not {"text", "prompt", "messages"} & {
                    k for k, v in row.items() if isinstance(v, str) and len(v) > 64
                }
                assert all(not isinstance(v, str) or len(v) <= 64 for v in row.values()), path


def test_cache_key_is_deterministic_and_order_independent():
    messages = [{"role": "user", "content": "x"}]
    assert cache_key("m", "s_tok", messages, {"a": 1, "b": 2}) == cache_key("m", "s_tok", messages, {"b": 2, "a": 1})
    assert cache_key("m", "s_tok", messages, {"a": 1}) != cache_key("m", "s_verb", messages, {"a": 1})


def test_answer_parsing():
    assert parse_yes_no("Yes.") == 1 and parse_yes_no(" no, not really") == 0 and parse_yes_no("Maybe") is None
    assert parse_verbal("85") == 0.85 and parse_verbal("Probability: 7%") == 0.07
    assert math.isnan(parse_verbal("none"))
    assert math.isnan(parse_verbal("150"))  # out of range
