import torch
import torch.nn.functional as F


def mae_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss averaged over masked patches only.

    Args:
        pred:   (B, num_patches, patch_dim)
        target: (B, num_patches, patch_dim)
        mask:   (B, num_patches) — 1 = masked, 0 = visible

    Returns:
        scalar loss
    """
    loss_per_patch = ((pred - target) ** 2).mean(dim=-1)   # (B, N)
    return (loss_per_patch * mask).sum() / mask.sum()


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy loss with optional label smoothing.

    Args:
        logits: (B, num_classes)
        labels: (B,)
        label_smoothing: float in [0, 1), default 0.0

    Returns:
        scalar loss
    """
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
