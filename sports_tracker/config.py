"""Configuration loading.

Loads `configs/default.yaml`, then overlays `configs/local.yaml` if it
exists. local.yaml is gitignored, so each user (you, your partner, the
graders) can set their own dataset paths without polluting the repo.

Usage:
    from sports_tracker.config import load_config
    cfg = load_config()
    sportsmot_root = cfg["sportsmot_root"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# The repo root is two levels up from this file:
#   sports_tracker/sports_tracker/config.py  ->  sports_tracker/
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_config() -> dict[str, Any]:
    """Load default config, then overlay local.yaml if present.

    Returns a dict. Raises FileNotFoundError if default.yaml is missing
    (which would mean the repo is broken / incomplete).
    """
    default_path = CONFIG_DIR / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Could not find {default_path}. Is the repo set up correctly?"
        )

    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)

    local_path = CONFIG_DIR / "local.yaml"
    if local_path.exists():
        with open(local_path, "r") as f:
            local_cfg = yaml.safe_load(f) or {}
        cfg.update(local_cfg)

    return cfg


def resolve_path(path_value: str) -> Path:
    """Resolve a path from config to an absolute Path.

    Relative paths are resolved against the repo root, so behavior is
    independent of where the script is invoked from.
    """
    p = Path(path_value)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p
