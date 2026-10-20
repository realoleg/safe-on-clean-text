from __future__ import annotations

import pandas as pd


def word_count(texts: pd.Series) -> pd.Series:
    """
    The single definition of text length: number of whitespace-separated words.
    """

    return texts.astype(str).str.split().str.len().astype(int)
