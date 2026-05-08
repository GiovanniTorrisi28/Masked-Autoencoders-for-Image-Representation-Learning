import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_dotenv(env_path: str | Path | None = None) -> None:
    """
    Load environment variables from a .env file.
    Searches for .env in the given path, or walks up from cwd if not specified.
    """
    if env_path is None:
        # Walk up from cwd looking for .env
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break

    if env_path is None or not Path(env_path).exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def set_seed(seed: int) -> None:
    """Set seeds for torch, numpy, and random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool = True) -> torch.device:
    """Return a CUDA device if available and requested, else CPU."""
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_eta(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
