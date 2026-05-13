"""
MAE Reconstruction Visualization.

Samples images from the validation set and generates a side-by-side grid:
    Original | Masked Input (75% patches hidden) | MAE Reconstruction

Usage:
    python visualize_mae.py \
        --config experiments/configs/mae_pretrain_cluster.yaml \
        --checkpoint experiments/checkpoints/mae_pretrain/checkpoint_best.pth \
        --num-images 8
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.imagenet100 import build_dataloader, build_supervised_transform
from src.models.mae import build_mae
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.misc import get_device, load_dotenv, set_seed

load_dotenv()

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MAE reconstructions from val set")
    parser.add_argument("--config",      required=True,  help="Path to MAE config YAML")
    parser.add_argument("--checkpoint",  required=True,  help="Path to MAE checkpoint")
    parser.add_argument("--num-images",  type=int, default=8,
                        help="Number of images to visualize (default: 8)")
    parser.add_argument("--output",      default="figures/",
                        help="Output directory (default: figures/)")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Seed for reproducible masking (default: 0)")
    parser.add_argument("--split",       default="val", choices=["val", "train"],
                        help="Dataset split to sample from (default: val)")
    return parser.parse_args()


def _denorm(tensor: torch.Tensor) -> torch.Tensor:
    """Remove ImageNet normalization. Returns values clamped to [0, 1]."""
    mean = _IMAGENET_MEAN.to(tensor.device)
    std  = _IMAGENET_STD.to(tensor.device)
    return (tensor * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def get_reconstructions(
    model,
    imgs: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run MAE forward pass and build the three visualization images.

    Args:
        model: trained MAE (eval mode)
        imgs:  (B, 3, H, W) ImageNet-normalized
        seed:  fixed seed for reproducible random masking

    Returns:
        originals:   (B, 3, H, W) in [0, 1]
        masked_disp: (B, 3, H, W) in [0, 1] — visible patches + gray (0.5) where masked
        recon:       (B, 3, H, W) in [0, 1] — visible original + decoder output for masked
    """
    # Patchify original to get per-patch stats needed to invert norm_pix_loss
    patches  = model.patchify(imgs)                              # (B, N, patch_dim)
    mean_pp  = patches.mean(dim=-1, keepdim=True)               # (B, N, 1)
    var_pp   = patches.var(dim=-1, keepdim=True, unbiased=False) # (B, N, 1)

    # Forward pass with fixed seed → reproducible masking
    torch.manual_seed(seed)
    _, pred, mask = model(imgs)   # pred: (B, N, patch_dim) in per-patch normalized space
                                  # mask: (B, N)  1=masked 0=visible

    # Bring predictions back to ImageNet-normalized pixel space
    pred_pixels = pred * (var_pp + 1e-6).sqrt() + mean_pp       # (B, N, patch_dim)

    mask_3d = mask.unsqueeze(-1).expand_as(patches)             # (B, N, patch_dim)

    # --- Reconstruction: visible original patches + decoder for masked patches ---
    recon_patches = patches * (1.0 - mask_3d) + pred_pixels * mask_3d
    recon = _denorm(model.unpatchify(recon_patches))            # (B, 3, H, W)

    # --- Masked display: original pixels where visible, pure gray (0.5) where masked ---
    mask_spatial  = model.unpatchify(mask_3d)                   # (B, 3, H, W) binary 0/1
    originals     = _denorm(imgs)
    masked_disp   = originals * (1.0 - mask_spatial) + 0.5 * mask_spatial

    return originals, masked_disp, recon


def save_grid(
    originals: torch.Tensor,
    masked:    torch.Tensor,
    recon:     torch.Tensor,
    output_path: Path,
) -> None:
    """
    Save an N×3 grid PNG.  Each row = one image; columns = Original / Masked / Reconstruction.
    """
    n = originals.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Masked Input (75%)", "Reconstruction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, (o, m, r) in enumerate(zip(originals, masked, recon)):
        for col, img in enumerate([o, m, r]):
            axes[row, col].imshow(img.cpu().permute(1, 2, 0).numpy())
            axes[row, col].axis("off")

    plt.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved → {output_path}")


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(args.seed)
    device = get_device()

    # Load first batch from the chosen split (deterministic, no shuffle)
    loader = build_dataloader(
        root=cfg["data"]["root"],
        split=args.split,
        transform=build_supervised_transform(cfg["data"]["image_size"], is_train=False),
        batch_size=args.num_images,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=False,
        shuffle=False,
    )
    imgs, _ = next(iter(loader))
    imgs = imgs[:args.num_images].to(device)

    # Build and load model
    model = build_mae(cfg).to(device)
    model.eval()
    load_checkpoint(args.checkpoint, model, device=device)

    # Reconstruct and save
    originals, masked, recon = get_reconstructions(model, imgs, seed=args.seed)
    output_path = Path(args.output) / "reconstruction_grid.png"
    save_grid(originals, masked, recon, output_path)


if __name__ == "__main__":
    main()
