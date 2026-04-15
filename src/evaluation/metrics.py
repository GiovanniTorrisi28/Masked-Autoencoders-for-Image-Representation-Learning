from __future__ import annotations

from typing import Iterable, Tuple

import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int] = (1, 5)) -> Tuple[float, ...]:
    if logits.ndim != 2:
        raise ValueError("Expected logits shape [batch, num_classes]")
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append((correct_k / targets.size(0)).item())
    return tuple(results)
