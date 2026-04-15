from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)


def load_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return state
