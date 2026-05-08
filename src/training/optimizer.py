import math

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    optimizer_cfg: dict,
) -> torch.optim.Optimizer:
    """
    Build optimizer with separate param groups:
    - weight-decay on regular weights
    - no weight-decay on bias and LayerNorm parameters (standard ViT practice)

    Supported names: 'adamw' (default), 'sgd'.
    """
    no_decay_names = ["bias", "norm"]
    decay_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not any(nd in name for nd in no_decay_names)
    ]
    no_decay_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and any(nd in name for nd in no_decay_names)
    ]
    param_groups = [
        {"params": decay_params,    "weight_decay": optimizer_cfg.get("weight_decay", 0.05)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = optimizer_cfg.get("name", "adamw").lower()
    lr = optimizer_cfg["lr"]

    if name == "adamw":
        betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    elif name == "sgd":
        momentum = optimizer_cfg.get("momentum", 0.9)
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a learning rate scheduler with linear warmup + cosine decay.

    Args:
        optimizer: optimizer to wrap
        scheduler_cfg: dict with keys: name, warmup_epochs, min_lr
        num_epochs: total training epochs

    Returns:
        LambdaLR scheduler
    """
    warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
    min_lr = scheduler_cfg.get("min_lr", 0.0)
    name = scheduler_cfg.get("name", "cosine").lower()

    # Retrieve base_lr from the first param group
    base_lr = optimizer.param_groups[0]["lr"]

    if name == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / max(warmup_epochs, 1)
            # Cosine decay
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale factor relative to base_lr
            min_factor = min_lr / base_lr if base_lr > 0 else 0.0
            return max(min_factor, cosine_decay)
    elif name == "step":
        step_size = scheduler_cfg.get("step_size", 30)
        gamma = scheduler_cfg.get("gamma", 0.1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            return gamma ** ((epoch - warmup_epochs) // step_size)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
