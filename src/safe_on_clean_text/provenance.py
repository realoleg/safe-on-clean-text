from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

TRACKED_LIBRARIES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "scikit-learn",
    "numpy",
    "pandas",
    "nltk",
]


def file_sha256(path: Path) -> str:
    """
    SHA-256 of a file's bytes (hex).
    """

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commit_hash() -> str:
    """
    Hash of the current git commit (HEAD).
    """

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def library_versions() -> dict[str, str]:
    """
    Python, platform and versions of the main libraries.
    """

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for library in TRACKED_LIBRARIES:
        versions[library] = version(library)
    return versions


def timestamp() -> str:
    """
    Current local time with timezone, ISO 8601, seconds precision.
    """

    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """
    Save a dictionary to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
