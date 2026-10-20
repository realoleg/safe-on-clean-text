import string

import pandas as pd

from safe_on_clean_text.degradations import QWERTY_NEIGHBORS, apply_condition, keep_head, keep_tail

TEXTS = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "text": [
            "The Quick brown fox jumps over the lazy dog again and again today",
            "one",
            "Short text, with punctuation! And CAPITALS 123.",
            " ".join(f"word{i}" for i in range(97)),
        ],
    }
)


def is_subsequence(short: list[str], long: list[str]) -> bool:
    remaining = iter(long)
    return all(word in remaining for word in short)


def test_word_deletion_levels_are_nested():
    low, high = apply_condition(TEXTS, "del_15", 42), apply_condition(TEXTS, "del_30", 42)
    for clean, kept_low, kept_high in zip(TEXTS["text"], low, high):
        assert is_subsequence(kept_high.split(), kept_low.split())
        assert is_subsequence(kept_low.split(), clean.split())
        assert len(kept_high.split()) >= 1


def test_typo_levels_are_nested_and_keep_case_and_non_letters():
    low, high = apply_condition(TEXTS, "typo_05", 42), apply_condition(TEXTS, "typo_10", 42)
    for clean, typo_low, typo_high in zip(TEXTS["text"], low, high):
        assert len(clean) == len(typo_low) == len(typo_high)
        for original, a, b in zip(clean, typo_low, typo_high):
            if original not in string.ascii_letters:
                assert original == a == b
            if a != original:
                assert b == a  # every low-level typo appears at the high level with the same replacement
                assert a.lower() in QWERTY_NEIGHBORS[original.lower()] and a.isupper() == original.isupper()


def test_head_and_tail_keep_ceil_share_of_words_and_at_least_one():
    words = [f"w{i}" for i in range(10)]
    assert keep_head(" ".join(words), 0.25).split() == words[:3]
    assert keep_tail(" ".join(words), 0.25).split() == words[-3:]
    assert keep_head("single", 0.25) == "single"


def test_conditions_are_deterministic():
    assert apply_condition(TEXTS, "del_30", 42).equals(apply_condition(TEXTS, "del_30", 42))
    assert not apply_condition(TEXTS, "del_30", 42).equals(apply_condition(TEXTS, "del_30", 43))
