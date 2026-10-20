from __future__ import annotations

from pathlib import Path

import pandas as pd

from safe_on_clean_text.provenance import file_sha256

SPLITS = ["train", "validation", "calibration", "test"]
GENERATED_FILES = [*(f"{split}.csv" for split in SPLITS), "conditions.csv"]
CHECKSUMS = "checksums.sha256"


def write_manifest(config: dict) -> None:
    """
    Manifest of one prepared dataset: <manifests_dir>/<dataset>.csv with the id and split of every row (no text), and its
    lines of <manifests_dir>/checksums.sha256 (sha256sum format) for the generated split and condition files, whose
    paths are relative to the parent of the processed folder.
    """

    name = config["data"]["name"]
    data_dir = Path(config["data"]["processed_dir"])
    manifests_dir = Path(config["paths"]["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)

    ids = [pd.read_csv(data_dir / f"{split}.csv", usecols=["id"]).assign(split=split) for split in SPLITS]
    pd.concat(ids, ignore_index=True).to_csv(manifests_dir / f"{name}.csv", index=False)

    checksums_path = manifests_dir / CHECKSUMS
    lines = {}
    if checksums_path.exists():
        for line in checksums_path.read_text().splitlines():
            digest, path = line.split("  ", 1)
            lines[path] = digest
    lines = {path: digest for path, digest in lines.items() if not path.startswith(f"{name}/")}
    lines |= {f"{name}/{file}": file_sha256(data_dir / file) for file in GENERATED_FILES}
    checksums_path.write_text("".join(f"{lines[path]}  {path}\n" for path in sorted(lines)))
