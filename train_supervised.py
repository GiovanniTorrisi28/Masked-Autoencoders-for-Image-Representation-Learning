"""
Supervised ViT baseline training entrypoint.

Usage:
    python train_supervised.py --config experiments/configs/supervised_vit.yaml
    python train_supervised.py --config experiments/configs/supervised_vit.yaml \\
                               --resume experiments/checkpoints/supervised_vit/checkpoint_best.pth
"""

import argparse
from pathlib import Path

from src.datasets.imagenet100 import build_dataloader, build_supervised_transform
from src.models.vit_classifier import build_vit_classifier
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer_supervised import SupervisedTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import UnifiedLogger
from src.utils.misc import count_parameters, get_device, load_dotenv, set_seed
from src.utils.config import load_config

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised ViT Baseline Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.no_wandb:
        cfg["logging"]["use_wandb"] = False
    if args.no_tensorboard:
        cfg["logging"]["use_tensorboard"] = False

    set_seed(cfg["experiment"]["seed"])
    device = get_device()

    # Datasets
    train_loader = build_dataloader(
        root=cfg["data"]["root"],
        split="train",
        transform=build_supervised_transform(cfg["data"]["image_size"], is_train=True),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=True,
        fraction=cfg["data"].get("fraction", 1.0),
        fraction_seed=cfg["data"].get("fraction_seed", 42),
    )
    val_loader = build_dataloader(
        root=cfg["data"]["root"],
        split="val",
        transform=build_supervised_transform(cfg["data"]["image_size"], is_train=False),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=False,
    )

    # Model (no pretrained encoder — train from scratch)
    model = build_vit_classifier(cfg, pretrained_encoder_path=None, device=device).to(device)
    print(f"[Model] ViTClassifier parameters: {count_parameters(model):,}")

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, cfg["training"]["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["training"]["scheduler"], cfg["training"]["epochs"])

    # Logger
    logger = UnifiedLogger(
        log_dir=cfg["logging"]["log_dir"],
        experiment_name=cfg["experiment"]["name"],
        config=cfg,
        use_wandb=cfg["logging"].get("use_wandb", True),
        use_tensorboard=cfg["logging"].get("use_tensorboard", True),
    )

    # Resume from checkpoint
    start_epoch = 0
    resume_path = args.resume or cfg["checkpointing"].get("resume")
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1

    # Train
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        cfg=cfg,
        device=device,
        checkpoint_dir=Path(cfg["checkpointing"]["checkpoint_dir"]),
    )
    trainer.global_step = start_epoch * len(train_loader)
    trainer.train(start_epoch=start_epoch)
    logger.finish()


if __name__ == "__main__":
    main()
