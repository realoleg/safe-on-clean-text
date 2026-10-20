import re

import pandas as pd
import pytest
from conftest import REPO

PAPER_CSVS = sorted((REPO / "results/paper").rglob("*.csv"))
# Columns that hold a statistic: they must be numbers, never text (e.g. a printed pandas Series).
NUMERIC_COLUMN = re.compile(r"(median|mean|q025|q975|_min|_max|macro_f1|auroc)$")


def test_paper_csvs_exist():
    assert len(PAPER_CSVS) > 20


@pytest.mark.parametrize("path", PAPER_CSVS, ids=lambda p: str(p.relative_to(REPO / "results/paper")))
def test_paper_table_cells_are_single_line_and_statistics_are_numeric(path):
    table = pd.read_csv(path)
    text = table.select_dtypes(exclude="number")
    multiline = [c for c in text.columns if text[c].astype(str).str.contains("\n").any()]
    assert not multiline, f"cells with line breaks in {multiline}"
    not_numeric = [c for c in table.columns if NUMERIC_COLUMN.search(c) and not pd.api.types.is_numeric_dtype(table[c])]
    assert not not_numeric, f"statistic columns that are not numeric: {not_numeric}"
