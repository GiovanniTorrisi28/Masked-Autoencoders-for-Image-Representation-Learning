"""
Standalone evaluation entrypoint.

Loads a trained ViTClassifier checkpoint and reports Top-1 / Top-5 accuracy
on the ImageNet100 validation set.

Usage:
    # Evaluate supervised ViT baseline
    python evaluate.py \\
        --config experiments/configs/supervised_vit.yaml \\
        --checkpoint experiments/checkpoints/supervised_vit/checkpoint_best.pth \\
        --label "Supervised ViT (scratch)"

    # Evaluate MAE + linear probe
    python evaluate.py \\
        --config experiments/configs/linear_probe.yaml \\
        --checkpoint experiments/checkpoints/linear_probe/checkpoint_best.pth \\
        --label "MAE + Linear Probe"
"""

import argparse

import torch

from src.datasets.imagenet100 import build_dataloader, build_supervised_transform
from src.evaluation.evaluator import Evaluator
from src.models.vit_classifier import build_vit_classifier
from src.utils.checkpoint import load_checkpoint
from src.utils.misc import get_device, load_dotenv, set_seed
from src.utils.config import load_config

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ViTClassifier")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--label", default="Model", help="Display name in the results table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device()

    # Validation dataloader
    val_loader = build_dataloader(
        root=cfg["data"]["root"],
        split="val",
        transform=build_supervised_transform(cfg["data"]["image_size"], is_train=False),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=False,
    )

    # Build model (no pretrained encoder loading — checkpoint has full state)
    model = build_vit_classifier(cfg, pretrained_encoder_path=None, device=device).to(device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)

    # Evaluate
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        num_classes=cfg["model"].get("num_classes", 100),
    )
    results = evaluator.evaluate()
    evaluator.print_report(results, label=args.label)


if __name__ == "__main__":
    main()
