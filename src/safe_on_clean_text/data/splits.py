from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_split(
    path: str | Path,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load one processed CSV split and validate a minimal schema:
    - cast text to str
    - cast label to int if present
    - add example_id if missing
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")

    df = pd.read_csv(path)

    required_columns = required_columns or ["text", "label"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    out = df.copy()

    if "text" in out.columns:
        out["text"] = out["text"].astype(str)

    if "label" in out.columns:
        out["label"] = out["label"].astype(int)

    if "example_id" not in out.columns:
        out.insert(0, "example_id", range(len(out)))

    return out


def load_splits(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all processed splits needed for transformer.
    """

    data_dir = Path(data_dir)

    return {
        "train": load_split(data_dir / "train.csv"),
        "calibration": load_split(data_dir / "calibration.csv"),
        "validation": load_split(data_dir / "validation.csv"),
        "test": load_split(data_dir / "test.csv"),
    }
