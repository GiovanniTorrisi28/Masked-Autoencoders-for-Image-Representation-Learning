import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.losses import cross_entropy_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import UnifiedLogger
from src.utils.misc import format_eta


class SupervisedTrainer:
    """
    Training loop for end-to-end supervised ViT training.

    Args:
        model: ViTClassifier
        optimizer: optimizer
        scheduler: LR scheduler
        train_loader: training DataLoader
        val_loader: validation DataLoader
        logger: UnifiedLogger
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
        val_loader: DataLoader,
        logger: UnifiedLogger,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)

        self.epochs = cfg["training"]["epochs"]
        self.log_freq = cfg["logging"].get("log_freq", 50)
        self.grad_clip = cfg["training"].get("grad_clip", None)
        self.label_smoothing = cfg["model"].get("label_smoothing", 0.0)
        self.global_step = 0

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        t0 = time.time()

        for step, (imgs, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(imgs)
            loss = cross_entropy_loss(logits, labels, self.label_smoothing)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            self.global_step += 1

            if (step + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                eta = format_eta(elapsed / (step + 1) * (len(self.train_loader) - step - 1))
                print(
                    f"  Epoch [{epoch+1}/{self.epochs}] "
                    f"Step [{step+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}  LR: {lr:.2e}  ETA: {eta}"
                )
                self.logger.log_scalars(
                    {"train/loss_step": loss.item(), "train/lr": lr},
                    step=self.global_step,
                )

        avg_loss = total_loss / len(self.train_loader)
        top1 = 100.0 * total_correct / total_samples
        return {"loss": avg_loss, "top1_acc": top1, "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(imgs)
            loss = cross_entropy_loss(logits, labels)
            total_loss += loss.item()

            _, top5_preds = logits.topk(5, dim=1)
            top1_correct += (top5_preds[:, 0] == labels).sum().item()
            top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += labels.size(0)

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_top1": 100.0 * top1_correct / total_samples,
            "val_top5": 100.0 * top5_correct / total_samples,
        }

    def train(self, start_epoch: int = 0) -> None:
        print(f"\n{'='*60}")
        print(f"  Supervised ViT Training: {self.epochs} epochs")
        print(f"{'='*60}\n")

        best_top1 = 0.0
        prev_best_file: Path | None = None

        for epoch in range(start_epoch, self.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            self.scheduler.step()

            print(
                f"[Epoch {epoch+1}/{self.epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f}  "
                f"Train Top-1: {train_metrics['top1_acc']:.2f}%  "
                f"Val Top-1: {val_metrics['val_top1']:.2f}%  "
                f"Val Top-5: {val_metrics['val_top5']:.2f}%"
            )
            self.logger.log_scalars(
                {
                    "train/loss_epoch": train_metrics["loss"],
                    "train/top1_epoch": train_metrics["top1_acc"],
                    "train/lr_epoch": train_metrics["lr"],
                    "val/loss": val_metrics["val_loss"],
                    "val/top1": val_metrics["val_top1"],
                    "val/top5": val_metrics["val_top5"],
                },
                step=self.global_step,
            )

            is_best = val_metrics["val_top1"] > best_top1
            if is_best:
                best_top1 = val_metrics["val_top1"]
                prev_best_file = self.save_checkpoint(epoch, val_metrics, prev_best_file)

            self._save_latest(epoch, val_metrics)

        print(f"\nSupervised training complete. Best Val Top-1: {best_top1:.2f}%")

    def _save_latest(self, epoch: int, metrics: dict) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }
        save_checkpoint(state, self.checkpoint_dir, "checkpoint_latest.pth")

    def save_checkpoint(self, epoch: int, metrics: dict, prev_file: Path | None = None) -> Path:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
        }
        filename = f"checkpoint_best_epoch_{epoch+1:04d}.pth"
        if prev_file is not None and prev_file.exists():
            prev_file.unlink()
        save_checkpoint(state, self.checkpoint_dir, filename)
        save_checkpoint(state, self.checkpoint_dir, "checkpoint_best.pth")
        return self.checkpoint_dir / filename
