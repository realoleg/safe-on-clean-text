from __future__ import annotations

import subprocess
from itertools import combinations, count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from safe_on_clean_text.provenance import file_sha256

SPLITS = ["train", "validation", "calibration", "test"]
CHECKED_SPLITS = ["calibration", "test"]
DELETED_AUTHOR = "[deleted]"


def download(url: str, path: Path, expected_sha256: str) -> str:
    """
    Download the pinned file once (curl with retries and a time limit) and verify its SHA-256.
    """

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-sSL", "--retry", "5", "--retry-delay", "3", "--max-time", "300", "-o", str(path), url],
            check=True,
        )
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {digest} != {expected_sha256}")
    return digest


def build_frame(raw: pd.DataFrame, max_words: int) -> pd.DataFrame:
    """
    id = row number of the pinned file; post_id from the Reddit URL; label = is_suicide (1 = r/SuicideWatch);
    text = title + ". " + selftext (raw fields), cut to its first max_words words (was_capped);
    group = author, every [deleted] row its own group.
    """

    text = raw["title"].astype(str) + ". " + raw["selftext"].astype(str)
    words = text.str.split()
    was_capped = words.str.len() > max_words
    out = pd.DataFrame(
        {
            "id": np.arange(len(raw)),
            "post_id": raw["url"].str.extract(r"/comments/([a-z0-9]+)/", expand=False),
            "author_group": np.where(
                raw["author"] == DELETED_AUTHOR, [f"deleted_{i}" for i in range(len(raw))], raw["author"]
            ),
            "text": np.where(was_capped, words.str[:max_words].str.join(" "), text),
            "selftext_key": raw["selftext"].astype(str),
            "label": raw["is_suicide"].astype(int),
            "was_capped": was_capped,
        }
    )
    assert out["post_id"].notna().all()
    return out


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Duplicates by the raw selftext field: identical labels -> keep the first copy, conflicting labels -> drop
    every copy. This also drops the 65 title-only posts whose selftext is SDCNL's placeholder "emptypost".
    """

    key = frame["selftext_key"]
    group_size = key.map(key.value_counts())
    is_conflicting = frame.groupby(key)["label"].transform("nunique") > 1
    is_extra_copy = key.duplicated(keep="first") & ~is_conflicting
    stats = {
        "duplicate_groups": int(key[group_size > 1].nunique()),
        "conflicting_label_groups": int(key[is_conflicting].nunique()),
        "rows_removed_extra_copies": int(is_extra_copy.sum()),
        "rows_removed_conflicting_labels": int(is_conflicting.sum()),
    }
    return frame[~is_conflicting & ~is_extra_copy].drop(columns="selftext_key").reset_index(drop=True), stats


def split_by_author(frame: pd.DataFrame, split_config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], int]:
    """
    StratifiedGroupKFold by author group, stratified by label; folds -> train / validation / calibration / test.
    Retries random_state + 1, + 2, ... until calibration and test both meet the row and per-class minimums.
    """

    for random_state in count(split_config["random_state"]):
        splitter = StratifiedGroupKFold(n_splits=split_config["n_splits"], shuffle=True, random_state=random_state)
        folds = np.full(len(frame), -1)
        for fold, (_, test_index) in enumerate(splitter.split(frame, frame["label"], groups=frame["author_group"])):
            folds[test_index] = fold
        splits = {}
        for name in SPLITS:
            first, last = split_config[f"{name}_folds"]
            splits[name] = frame[(folds >= first) & (folds <= last)].reset_index(drop=True)
        assert sum(len(df) for df in splits.values()) == len(frame)
        if all(
            len(splits[name]) >= split_config["min_rows"]
            and splits[name]["label"].value_counts().reindex([0, 1], fill_value=0).min()
            >= split_config["min_rows_per_class"]
            for name in CHECKED_SPLITS
        ):
            return splits, random_state


def split_report(
    digest: str,
    raw_rows: int,
    dedup_stats: dict[str, int],
    frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    random_state: int,
    split_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Long report (section, split, key, value): source, deduplication, author groups, sizes and classes per split,
    capped texts, shared authors / posts between splits (must be 0), thresholds.
    """

    rows: list[tuple[str, str, str, Any]] = [
        ("source", "all", "sha256", digest),
        ("source", "all", "rows", raw_rows),
        *[("dedup", "all", key, value) for key, value in dedup_stats.items()],
        ("groups", "all", "rows_after_dedup", len(frame)),
        ("groups", "all", "author_groups", frame["author_group"].nunique()),
        ("groups", "all", "deleted_author_rows", int(frame["author_group"].str.startswith("deleted_").sum())),
        ("groups", "all", "max_rows_per_author", int(frame["author_group"].value_counts().max())),
        ("split", "all", "random_state_used", random_state),
        ("split", "all", "min_rows", split_config["min_rows"]),
        ("split", "all", "min_rows_per_class", split_config["min_rows_per_class"]),
    ]
    for name, df in splits.items():
        rows += [
            ("size", name, "rows", len(df)),
            ("size", name, "label_0", int((df["label"] == 0).sum())),
            ("size", name, "label_1", int((df["label"] == 1).sum())),
            ("size", name, "author_groups", df["author_group"].nunique()),
            ("size", name, "was_capped", int(df["was_capped"].sum())),
            ("size", name, "median_words", int(df["text"].str.split().str.len().median())),
        ]
    for a, b in combinations(SPLITS, 2):
        rows.append(
            (
                "overlap",
                f"{a}|{b}",
                "shared_authors",
                len(set(splits[a]["author_group"]) & set(splits[b]["author_group"])),
            )
        )
        rows.append(("overlap", f"{a}|{b}", "shared_posts", len(set(splits[a]["post_id"]) & set(splits[b]["post_id"]))))
    return pd.DataFrame(rows, columns=["section", "split", "key", "value"])


def prepare(config: dict) -> None:
    """
    SDCNL from its pinned source (SHA-256 checked) -> deduplicated, author-grouped splits and split_report.csv in
    data.processed_dir.
    """

    data = config["data"]
    output_dir = Path(data["processed_dir"])

    raw_path = Path(data["raw_path"])
    digest = download(data["source_url"], raw_path, data["source_sha256"])
    raw = pd.read_csv(raw_path)
    frame, dedup_stats = deduplicate(build_frame(raw, int(data["max_words"])))
    splits, random_state = split_by_author(frame, data["split"])

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    report = split_report(digest, len(raw), dedup_stats, frame, splits, random_state, data["split"])
    report_path = output_dir / "split_report.csv"
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))

    shared = report[(report["section"] == "overlap")]["value"].astype(int).sum()
    if shared:
        raise RuntimeError("Splits share authors or posts.")
