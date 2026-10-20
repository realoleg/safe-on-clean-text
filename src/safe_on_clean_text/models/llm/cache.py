from __future__ import annotations

import gzip
import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def cache_key(model: str, method: str, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
    """
    SHA-256 of everything that determines the answer. Only this hash is stored, never the prompt text.
    """

    payload = json.dumps(
        {"model": model, "method": method, "messages": messages, "settings": settings},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AnswerCache:
    """
    Append-only gzip JSONL cache of one (model, method): one line per answered prompt hash (each append adds a gzip
    member; readers see one stream). Lines hold the hash, the example id and the answer fields only (no prompt or post text).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def put(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.path, "at", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.rows[row["key"]] = row


def drop_rows_written_since(path: Path, since: float) -> int:
    """
    Remove cache rows written at or after `since` (Unix time); used after a Metal error, whose
    command buffers may have produced wrong values silently. Returns the number of removed rows.
    """

    if not path.exists():
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    kept = [line for line in lines if json.loads(line)["written_at"] < since]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(kept)
    return len(lines) - len(kept)
