from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import topk_accuracy


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        total_top5 += top5 * bs
        total_samples += bs

    if total_samples == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0}

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }
