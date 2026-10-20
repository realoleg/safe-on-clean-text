from __future__ import annotations

from pathlib import Path

PAPER_DIR = Path("results/paper")
TABLES, FIGURES = PAPER_DIR / "tables", PAPER_DIR / "figures"


def display_number(value: float) -> str:
    """
    A number in human-readable text (digest, Table 6 criteria and value strings): 4 decimals below 10, otherwise 1
    decimal (none for whole numbers). Numeric table columns keep full precision.
    """

    if abs(value) >= 10:
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
    return f"{value:.4f}"
