import torch


def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> float:
    """
    Compute Top-k accuracy over a batch.

    Args:
        logits: (B, num_classes)
        labels: (B,)
        k: top-k threshold

    Returns:
        accuracy as a float in [0.0, 100.0]
    """
    with torch.no_grad():
        _, topk_preds = logits.topk(k, dim=1)  # (B, k)
        correct = (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    return 100.0 * correct / labels.size(0)


class AverageMeter:
    """Tracks the running mean of a scalar metric across batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, val: float, n: int = 1) -> None:
        self._sum += val * n
        self._count += n

    @property
    def avg(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> int:
        return self._count
