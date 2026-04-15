from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.67, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected train/ and val/ under {root}. Found train={train_dir.exists()}, val={val_dir.exists()}"
        )

    train_t, val_t = build_transforms(image_size=image_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_t)
    val_ds = datasets.ImageFolder(val_dir, transform=val_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_ds.class_to_idx
