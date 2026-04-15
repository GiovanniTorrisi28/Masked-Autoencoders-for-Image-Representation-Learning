from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 3e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))

    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    sch_cfg = cfg.get("scheduler", {})
    sch_name = sch_cfg.get("name", "cosine").lower()
    if sch_name == "none":
        scheduler = None
    elif sch_name == "cosine":
        epochs = int(cfg.get("training", {}).get("epochs", 1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif sch_name == "step":
        step_size = int(sch_cfg.get("step_size", 30))
        gamma = float(sch_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {sch_name}")

    return optimizer, scheduler
