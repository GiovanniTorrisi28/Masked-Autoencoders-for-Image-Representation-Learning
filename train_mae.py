"""
MAE pre-training entrypoint.

Usage:
    python train_mae.py --config experiments/configs/mae_pretrain.yaml
    python train_mae.py --config experiments/configs/mae_pretrain.yaml \\
                        --resume experiments/checkpoints/mae_pretrain/checkpoint_epoch_0100.pth
"""

import argparse
from pathlib import Path

from src.datasets.imagenet100 import build_dataloader, build_mae_transform
from src.models.mae import build_mae
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer_mae import MAETrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import UnifiedLogger
from src.utils.misc import count_parameters, get_device, load_dotenv, set_seed
from src.utils.config import load_config

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE Pre-training")
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

    # Dataset
    train_loader = build_dataloader(
        root=cfg["data"]["root"],
        split="train",
        transform=build_mae_transform(cfg["data"]["image_size"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=True,
    )

    # Model
    model = build_mae(cfg).to(device)
    print(f"[Model] MAE parameters: {count_parameters(model):,}")

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
    trainer = MAETrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
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
