import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return a nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_config_with_args(cfg: dict, args: argparse.Namespace) -> dict:
    """
    Override config values with explicitly passed command-line args.
    Supports dot-notation keys in args (e.g., 'training.epochs') resolved
    into nested dict paths.
    """
    for key, value in vars(args).items():
        if value is None:
            continue
        # dot-notation support
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return cfg


def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from a config dict."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d
