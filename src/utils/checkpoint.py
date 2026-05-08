from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    state: dict,
    checkpoint_dir: str | Path,
    filename: str,
) -> None:
    """
    Save a state dict to checkpoint_dir/filename.

    state should contain at minimum:
        {'epoch': int, 'model_state_dict': ..., 'optimizer_state_dict': ...,
         'scheduler_state_dict': ..., 'config': dict}
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: torch.device | None = None,
) -> dict:
    """
    Load a checkpoint and restore model, optimizer, and scheduler states.
    Returns the full checkpoint dict.
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"[Checkpoint] Loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def load_encoder_weights(
    mae_checkpoint_path: str | Path,
    vit_classifier: nn.Module,
    device: torch.device | None = None,
) -> None:
    """
    Load only the encoder weights from an MAE checkpoint into a ViTClassifier.
    Handles the 'encoder.' prefix that MAE state dicts use.
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(mae_checkpoint_path, map_location=device)
    mae_state = ckpt["model_state_dict"]

    # Extract encoder sub-dict (keys start with 'encoder.')
    encoder_state = {}
    for k, v in mae_state.items():
        if k.startswith("encoder."):
            new_key = k[len("encoder."):]
            encoder_state[new_key] = v

    missing, unexpected = vit_classifier.encoder.load_state_dict(
        encoder_state, strict=False
    )
    if missing:
        print(f"[Checkpoint] Encoder missing keys: {missing}")
    if unexpected:
        print(f"[Checkpoint] Encoder unexpected keys: {unexpected}")

    print(f"[Checkpoint] Encoder weights loaded from {mae_checkpoint_path}")
