"""
Linear probing entrypoint.

Loads encoder weights from an MAE checkpoint, freezes the encoder, and
trains only a linear classification head.

Usage:
    python train_linear_probe.py --config experiments/configs/linear_probe.yaml
    python train_linear_probe.py --config experiments/configs/linear_probe.yaml \\
                                 --mae-checkpoint experiments/checkpoints/mae_pretrain/checkpoint_best.pth
"""

import argparse
from pathlib import Path

from src.datasets.imagenet100 import build_dataloader, build_supervised_transform
from src.models.vit_classifier import build_vit_classifier
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer_linear_probe import LinearProbeTrainer
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import UnifiedLogger
from src.utils.misc import count_parameters, get_device, load_dotenv, set_seed
from src.utils.config import load_config

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE Linear Probing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--mae-checkpoint",
        default=None,
        help="Override pretrained_encoder path from config",
    )
    parser.add_argument("--resume", default=None, help="Resume linear probe training from checkpoint")
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

    # Resolve MAE checkpoint path
    mae_ckpt = args.mae_checkpoint or cfg["model"].get("pretrained_encoder")
    if mae_ckpt is None:
        raise ValueError(
            "No MAE checkpoint provided. Set model.pretrained_encoder in the config "
            "or pass --mae-checkpoint."
        )

    # Model: load MAE encoder weights, freeze encoder
    model = build_vit_classifier(cfg, pretrained_encoder_path=mae_ckpt, device=device).to(device)

    trainable = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Trainable params: {trainable:,} / {total:,} (encoder frozen)")

    # Optimizer acts only on trainable (head) parameters
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

    # Resume linear probe training
    start_epoch = 0
    resume_path = args.resume or cfg["checkpointing"].get("resume")
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1

    # Train
    trainer = LinearProbeTrainer(
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
