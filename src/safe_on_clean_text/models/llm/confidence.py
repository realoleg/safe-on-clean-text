from __future__ import annotations

import math
import re
from typing import Any

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def parse_verbal(answer: str) -> float:
    """
    p(stress) = first number in the answer / 100; NaN if there is none or it is outside [0, 100].
    """

    match = NUMBER_PATTERN.search(answer)
    if match is None:
        return math.nan
    value = float(match.group())
    return value / 100 if 0 <= value <= 100 else math.nan


def parse_yes_no(answer: str) -> int | None:
    """
    1 for an answer starting with yes, 0 for no (case and surrounding punctuation ignored), else None.
    """

    words = re.findall(r"[a-z]+", answer.lower())
    if not words:
        return None
    return {"yes": 1, "no": 0}.get(words[0])


def label_probabilities(p_yes: float, p_no: float) -> dict[str, float]:
    mass = p_yes + p_no
    return {"p_yes": p_yes, "p_no": p_no, "mass": mass, "p_stress": p_yes / mass if mass > 0 else math.nan}


def generation_settings(llm_config: dict, method: str, answer_prefix: str) -> dict[str, Any]:
    """
    Generation settings that enter the cache key.
    """

    if method == "s_tok":
        return {
            "answer_prefix": answer_prefix,
            "yes": llm_config["yes_tokens"],
            "no": llm_config["no_tokens"],
            "top_logprobs": llm_config["top_logprobs"],
        }
    if method == "s_verb":
        return {"answer_prefix": answer_prefix, "temperature": 0, "max_tokens": llm_config["verbal_max_tokens"]}
    return {
        "temperature": llm_config["consistency"]["temperature"],
        "n": llm_config["consistency"]["n"],
        "max_tokens": llm_config["consistency"]["max_tokens"],
    }
