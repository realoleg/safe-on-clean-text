from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str = "configs/dreaddit.yaml") -> dict:
    """
    Load a YAML config file. A top-level `extends: <file>` (relative to this file's folder) loads that config first
    and deep-merges this one over it (dicts merged key by key, everything else replaced; a null value removes the
    inherited key).
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parent = config.pop("extends", None)
    if parent is None:
        return config
    return deep_merge(load_config(str(Path(config_path).parent / parent)), config)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = (
                deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
            )
    return out
