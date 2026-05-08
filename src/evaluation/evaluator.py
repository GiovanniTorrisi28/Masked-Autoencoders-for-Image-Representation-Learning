import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import AverageMeter


class Evaluator:
    """
    Standalone evaluator. Loads a trained model checkpoint, runs the full
    validation set, and returns a metrics dict.

    Args:
        model: ViTClassifier (encoder + head)
        val_loader: validation DataLoader
        device: torch.device
        num_classes: 100 for ImageNet100
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int = 100,
    ) -> None:
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        criterion = torch.nn.CrossEntropyLoss()

        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            B = labels.size(0)

            logits = self.model(imgs)
            loss = criterion(logits, labels)
            loss_meter.update(loss.item(), B)

            _, top5_preds = logits.topk(min(5, self.num_classes), dim=1)
            top1_correct = (top5_preds[:, 0] == labels).sum().item()
            top5_correct = (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            top1_meter.update(100.0 * top1_correct / B, B)
            top5_meter.update(100.0 * top5_correct / B, B)

        return {
            "top1": top1_meter.avg,
            "top5": top5_meter.avg,
            "loss": loss_meter.avg,
            "num_samples": top1_meter.count,
        }

    def print_report(
        self,
        results: dict[str, float],
        label: str = "Model",
    ) -> None:
        width = 60
        print("=" * width)
        print(f"  {label}")
        print("-" * width)
        print(f"  {'Metric':<20} {'Value':>10}")
        print("-" * width)
        print(f"  {'Top-1 Accuracy':<20} {results['top1']:>9.2f}%")
        print(f"  {'Top-5 Accuracy':<20} {results['top5']:>9.2f}%")
        print(f"  {'Val Loss':<20} {results['loss']:>10.4f}")
        print(f"  {'Num Samples':<20} {results['num_samples']:>10}")
        print("=" * width)


def print_comparison(results_list: list[tuple[str, dict[str, float]]]) -> None:
    """
    Print a side-by-side comparison table of multiple model evaluations.

    Args:
        results_list: list of (label, metrics_dict) tuples
    """
    width = 62
    print("\n" + "=" * width)
    print("  ImageNet100 Evaluation Results")
    print("=" * width)
    print(f"  {'Model':<35} {'Top-1 (%)':>8}  {'Top-5 (%)':>8}")
    print("-" * width)
    for label, results in results_list:
        print(f"  {label:<35} {results['top1']:>8.2f}  {results['top5']:>8.2f}")
    print("=" * width + "\n")
