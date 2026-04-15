from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.datasets.imagenet100 import build_dataloaders
from src.evaluation.evaluate import evaluate_classifier
from src.models.vit_classifier import build_vit_classifier
from src.training.losses import build_classification_loss
from src.training.optimizers import build_optimizer_and_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / max(1, total_samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised ViT baseline on ImageNet100")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})
    wandb_cfg = cfg.get("wandb", {})

    device = torch.device(training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"

    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_root=data_cfg.get("data_root", "data/raw/imagenet100"),
        image_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    num_classes = len(class_to_idx)
    model = build_vit_classifier(model_name=model_cfg.get("name", "vit_b_16"), num_classes=num_classes)
    model = model.to(device)

    criterion = build_classification_loss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    epochs = int(training_cfg.get("epochs", 1))
    output_dir = Path(out_cfg.get("dir", "experiments/checkpoints/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = Path(out_cfg.get("tb_dir", str(output_dir / "tb")))
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    use_wandb = bool(wandb_cfg.get("enabled", False))
    wandb_run = None
    if use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("wandb is enabled in config but package is missing. Install with: pip install wandb") from exc

        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "mae-baseline"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", None),
            tags=wandb_cfg.get("tags", ["baseline", "supervised", "vit"]),
            config=cfg,
            dir=str(output_dir),
        )

    best_top1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_stats = evaluate_classifier(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_top1": val_stats["top1"],
            "val_top5": val_stats["top5"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        writer.add_scalar("train/loss", row["train_loss"], epoch)
        writer.add_scalar("val/loss", row["val_loss"], epoch)
        writer.add_scalar("val/top1", row["val_top1"], epoch)
        writer.add_scalar("val/top5", row["val_top5"], epoch)
        writer.add_scalar("train/lr", row["lr"], epoch)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_top1={row['val_top1']:.4f} | "
            f"val_top5={row['val_top5']:.4f}"
        )

        if row["val_top1"] > best_top1:
            best_top1 = row["val_top1"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "class_to_idx": class_to_idx,
                    "best_val_top1": best_top1,
                },
                output_dir / "best.pt",
            )

    save_checkpoint(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "class_to_idx": class_to_idx,
            "best_val_top1": best_top1,
        },
        output_dir / "last.pt",
    )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    writer.close()
    if wandb_run is not None:
        wandb_run.summary["best_val_top1"] = best_top1
        wandb_run.finish()
    print(f"Saved training history to {history_path}")
    print(f"TensorBoard logs saved to {tb_dir}")
    print(f"Best val top1: {best_top1:.4f}")


if __name__ == "__main__":
    main()
