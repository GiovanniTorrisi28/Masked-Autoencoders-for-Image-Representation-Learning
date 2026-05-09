from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_mae_transform(image_size: int = 224) -> transforms.Compose:
    """
    Augmentation pipeline for MAE pre-training.
    No color jitter — MAE needs consistent pixel targets for reconstruction.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_supervised_transform(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Standard supervised augmentation pipeline.
    Train: RandomResizedCrop + RandomHorizontalFlip + ColorJitter + Normalize.
    Val:   Resize(256) + CenterCrop + Normalize.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _find_split_dirs(root: Path, split: str) -> list[Path]:
    """
    Locate all directories for a given split. Handles two layouts:

    Local (unified after reorganize_dataset):
        root/train/<class_id>/
        root/val/<class_id>/

    Kaggle native (as-downloaded from ambityga/imagenet100):
        root/train.X1/<class_id>/
        root/train.X2/<class_id>/
        root/train.X3/<class_id>/
        root/train.X4/<class_id>/
        root/val.X/<class_id>/

    Returns sorted list of matching Path objects.
    Raises FileNotFoundError if nothing is found.
    """
    root = Path(root)

    # 1. Unified local layout (root/train or root/val)
    direct = root / split
    if direct.is_dir():
        return [direct]

    # 2. Kaggle layout: root/train.X or root/train.X<N>
    candidates = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (
            d.name == f"{split}.X"
            or (d.name.startswith(f"{split}.X") and d.name[len(split) + 2:].isdigit())
        )
    )
    if candidates:
        return candidates

    available = [d.name for d in root.iterdir() if d.is_dir()]
    raise FileNotFoundError(
        f"Cannot find split '{split}' in {root}. "
        f"Available directories: {available}"
    )


def _build_global_class_to_idx(split_dirs: list[Path]) -> dict[str, int]:
    """
    Build a single consistent class→index mapping from one or more split
    directories. Classes are sorted alphabetically for determinism.
    """
    all_classes: set[str] = set()
    for d in split_dirs:
        all_classes.update(entry.name for entry in d.iterdir() if entry.is_dir())
    return {cls: idx for idx, cls in enumerate(sorted(all_classes))}


class _FixedClassImageFolder(datasets.ImageFolder):
    """
    ImageFolder that uses a pre-built class→index mapping.

    Guarantees consistent label indices across multiple split directories
    (e.g. train.X1 … train.X4 on Kaggle that may each contain different
    subsets of classes).
    """

    def __init__(self, root: str, class_to_idx_override: dict[str, int], **kwargs):
        self._class_to_idx_override = class_to_idx_override
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        classes = sorted(
            entry.name for entry in Path(directory).iterdir()
            if entry.is_dir() and entry.name in self._class_to_idx_override
        )
        return classes, {c: self._class_to_idx_override[c] for c in classes}


def _stratified_subset(dataset: torch.utils.data.Dataset, fraction: float, seed: int) -> Subset:
    """Returns a Subset with stratified sampling: keeps `fraction` images per class."""
    if isinstance(dataset, ConcatDataset):
        targets = [t for ds in dataset.datasets for t in ds.targets]
    else:
        targets = list(dataset.targets)

    rng = torch.Generator()
    rng.manual_seed(seed)

    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    selected: list[int] = []
    for label in sorted(class_indices):
        indices = class_indices[label]
        n_keep = max(1, int(len(indices) * fraction))
        perm = torch.randperm(len(indices), generator=rng).tolist()
        selected.extend(indices[perm[i]] for i in range(n_keep))

    return Subset(dataset, sorted(selected))


class ImageNet100Dataset(torch.utils.data.Dataset):
    """
    Dataset for ImageNet100 (or any ImageFolder-compatible subset).

    Transparently handles both layouts:
    - Local unified layout: root/train/, root/val/
    - Kaggle native layout: root/train.X1/ ... root/train.X4/, root/val.X/

    When multiple split directories are found they are concatenated with a
    globally consistent class→index mapping so labels are identical across splits.

    Args:
        root: dataset root (e.g. 'data/raw/imagenet100' or '/kaggle/input/imagenet100')
        split: 'train' or 'val'
        transform: torchvision transform pipeline
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        fraction: float = 1.0,
        fraction_seed: int = 42,
    ) -> None:
        root = Path(root)
        split_dirs = _find_split_dirs(root, split)
        class_to_idx = _build_global_class_to_idx(split_dirs)

        if len(split_dirs) == 1:
            self._dataset = _FixedClassImageFolder(
                root=str(split_dirs[0]),
                class_to_idx_override=class_to_idx,
                transform=transform,
            )
        else:
            # Multiple dirs (Kaggle train.X1…X4): concatenate with shared mapping
            sub_datasets = [
                _FixedClassImageFolder(
                    root=str(d),
                    class_to_idx_override=class_to_idx,
                    transform=transform,
                )
                for d in split_dirs
            ]
            self._dataset = ConcatDataset(sub_datasets)

        if fraction < 1.0:
            self._dataset = _stratified_subset(self._dataset, fraction, fraction_seed)

        self._class_to_idx = class_to_idx
        self._num_classes = len(class_to_idx)

        fraction_str = f" (fraction={fraction:.0%})" if fraction < 1.0 else ""
        print(
            f"[Dataset] '{split}' → {len(split_dirs)} dir(s): "
            f"{[d.name for d in split_dirs]}  |  "
            f"{len(self)} images, {self._num_classes} classes{fraction_str}"
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._dataset[idx]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._class_to_idx


def build_dataloader(
    root: str | Path,
    split: str,
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False,
    shuffle: bool | None = None,
    fraction: float = 1.0,
    fraction_seed: int = 42,
) -> DataLoader:
    """
    Factory that wraps ImageNet100Dataset in a DataLoader.
    shuffle defaults to True for 'train', False otherwise.
    fraction < 1.0 applies stratified subsampling to the training split.
    """
    dataset = ImageNet100Dataset(
        root=root, split=split, transform=transform,
        fraction=fraction, fraction_seed=fraction_seed,
    )
    if shuffle is None:
        shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    print(f"[DataLoader] {split}: {len(loader)} batches (batch_size={batch_size})")
    return loader
