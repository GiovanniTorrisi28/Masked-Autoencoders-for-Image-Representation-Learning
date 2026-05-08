import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.checkpoint import save_checkpoint
from src.utils.logger import UnifiedLogger
from src.utils.misc import format_eta


class MAETrainer:
    """
    Training loop for MAE self-supervised pre-training.

    Args:
        model: MAE model
        optimizer: optimizer (AdamW)
        scheduler: LambdaLR scheduler
        train_loader: training DataLoader
        logger: UnifiedLogger instance
        cfg: full experiment config dict
        device: torch.device
        checkpoint_dir: directory for saving checkpoints
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        logger: UnifiedLogger,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)

        self.epochs = cfg["training"]["epochs"]
        self.log_freq = cfg["logging"].get("log_freq", 50)
        self.save_every = cfg["checkpointing"].get("save_every", 20)
        self.grad_clip = cfg["training"].get("grad_clip", None)
        self.global_step = 0

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (imgs, _) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)

            loss, _, _ = self.model(imgs)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            self.global_step += 1

            if (step + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_left = len(self.train_loader) - step - 1
                eta = format_eta(elapsed / (step + 1) * steps_left)
                print(
                    f"  Epoch [{epoch+1}/{self.epochs}] "
                    f"Step [{step+1}/{len(self.train_loader)}] "
                    f"Loss: {loss_val:.4f}  LR: {lr:.2e}  ETA: {eta}"
                )
                self.logger.log_scalars(
                    {"train/loss_step": loss_val, "train/lr": lr},
                    step=self.global_step,
                )

        avg_loss = total_loss / len(self.train_loader)
        return {"loss": avg_loss, "lr": self.optimizer.param_groups[0]["lr"]}

    def train(self, start_epoch: int = 0) -> None:
        print(f"\n{'='*60}")
        print(f"  MAE Pre-training: {self.epochs} epochs")
        print(f"{'='*60}\n")

        best_loss = float("inf")
        prev_best_file: Path | None = None

        for epoch in range(start_epoch, self.epochs):
            metrics = self.train_one_epoch(epoch)
            self.scheduler.step()

            print(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"Avg Loss: {metrics['loss']:.4f}  "
                f"LR: {metrics['lr']:.2e}"
            )
            self.logger.log_scalars(
                {"train/loss_epoch": metrics["loss"], "train/lr_epoch": metrics["lr"]},
                step=self.global_step,
            )

            is_best = metrics["loss"] < best_loss
            if is_best:
                best_loss = metrics["loss"]
                prev_best_file = self.save_checkpoint(epoch, prev_best_file)

            self._save_latest(epoch, metrics)

        print("\nMAE pre-training complete.")

    def _save_latest(self, epoch: int, metrics: dict) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        save_checkpoint(state, self.checkpoint_dir, "checkpoint_latest.pth")

    def save_checkpoint(self, epoch: int, prev_file: Path | None = None) -> Path:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        filename = f"checkpoint_best_epoch_{epoch+1:04d}.pth"
        if prev_file is not None and prev_file.exists():
            prev_file.unlink()
        save_checkpoint(state, self.checkpoint_dir, filename)
        save_checkpoint(state, self.checkpoint_dir, "checkpoint_best.pth")
        return self.checkpoint_dir / filename
