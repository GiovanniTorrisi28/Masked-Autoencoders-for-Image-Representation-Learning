"""Download and prepare ImageNet100 from Kaggle.

Dataset: https://www.kaggle.com/datasets/ambityga/imagenet100

Expected final structure:
data/raw/imagenet100/
  train/<class_id>/*.jpg
  val/<class_id>/*.jpg
  Labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ImageNet100 from Kaggle")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ambityga/imagenet100",
        help="Kaggle dataset slug",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/imagenet100"),
        help="Directory where dataset files will be saved",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded archives (recommended)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/processed/imagenet100_summary.json"),
        help="Path for output summary JSON",
    )
    parser.add_argument(
        "--kaggle-json",
        type=Path,
        default=None,
        help="Optional path to kaggle.json to copy into ~/.kaggle before download",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Kaggle download progress output",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_kaggle_credentials(kaggle_json: Path | None) -> Path:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_target = kaggle_dir / "kaggle.json"

    if kaggle_json is not None:
        if not kaggle_json.exists():
            raise FileNotFoundError(f"Provided --kaggle-json does not exist: {kaggle_json}")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_target.write_text(kaggle_json.read_text(encoding="utf-8"), encoding="utf-8")

    if kaggle_target.exists():
        creds = _read_json(kaggle_target)
        if not all(k in creds for k in ("username", "key")):
            raise RuntimeError(
                f"Invalid kaggle.json at {kaggle_target}. Expected keys: 'username', 'key'."
            )
        return kaggle_target

    # Allow fallback through environment variables if user prefers not to use file.
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return kaggle_target

    raise RuntimeError(
        "Kaggle credentials not found. Provide one of these options:\n"
        "1) Place kaggle.json in ~/.kaggle/kaggle.json\n"
        "2) Pass --kaggle-json /path/to/kaggle.json\n"
        "3) Set env vars KAGGLE_USERNAME and KAGGLE_KEY"
    )


def download_kaggle_dataset(dataset: str, output_dir: Path, show_progress: bool = True) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'kaggle'. Install it with: pip install kaggle"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset,
        path=str(output_dir),
        unzip=True,
        quiet=not show_progress,
    )


def is_dataset_ready(root: Path) -> bool:
    """Check if dataset is already downloaded and extracted."""
    train_dir = root / "train"
    val_dir = root / "val"
    
    if not (train_dir.exists() and val_dir.exists()):
        return False
    
    # Check if there are image files
    img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
    train_images = sum(
        1 for f in train_dir.rglob("*") if f.is_file() and f.suffix in img_ext
    )
    val_images = sum(
        1 for f in val_dir.rglob("*") if f.is_file() and f.suffix in img_ext
    )
    
    return train_images > 0 and val_images > 0


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def extract_archives(root: Path) -> int:
    extracted = 0
    skipped = 0
    for file_path in iter_files(root):
        if zipfile.is_zipfile(file_path):
            target = file_path.with_suffix("")
            target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(target)
                extracted += 1
            except (zipfile.BadZipFile, RuntimeError) as e:
                print(f"⚠️  Warning: Corrupted ZIP archive {file_path.name}: {e}")
                skipped += 1
            continue

        if tarfile.is_tarfile(file_path):
            target = file_path.with_suffix("")
            target.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(file_path, "r:*") as tar_ref:
                    tar_ref.extractall(target)
                extracted += 1
            except (tarfile.ReadError, EOFError) as e:
                print(f"⚠️  Warning: Corrupted TAR archive {file_path.name}: {e}")
                skipped += 1

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} corrupted archives")

    return extracted


def reorganize_dataset(root: Path) -> None:
    """Merge split dataset parts and rename single split folders (train.X1, train.X2, val.X, etc.) into unified train/ and val/ folders."""
    from collections import defaultdict
    
    # Find all split folder patterns: train.X1, train.X2, train.X, val.X1, val.X, etc.
    split_folders = defaultdict(list)
    single_folders = {}  # For single folders like val.X
    
    print(f"Scanning {root} for split folders...")
    for item in root.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        print(f"  Found folder: {name}")
        
        # Match patterns like "train.X1", "train.1", "train.X", "val.X1", "val.X", etc.
        if "." in name:
            base, suffix = name.rsplit(".", 1)
            suffix_lower = suffix.lower()
            
            # Check if suffix is X+digits (X1, x1, X2, x2) or just X/x
            if suffix_lower.startswith("x"):
                suffix_clean = suffix_lower[1:]  # Remove the X/x
                
                if suffix_clean.isdigit():
                    # Split folder like train.X1, train.X2
                    split_num = int(suffix_clean)
                    split_folders[base].append((split_num, item))
                    print(f"    -> Matched as split folder for '{base}' (part {split_num})")
                elif suffix_clean == "":
                    # Single folder like train.X or val.X
                    single_folders[base] = item
                    print(f"    -> Matched as single split folder '{base}'")
    
    # First, handle single folders (rename .X to base name)
    for base_name, folder in single_folders.items():
        target_dir = root / base_name
        if not target_dir.exists() or target_dir == folder:
            print(f"Renaming {folder.name} to {base_name}...")
            folder.rename(target_dir)
            print(f"✅ Renamed to {base_name}/")
    
    # Then, handle split folders (merge multiple parts)
    for base_name, folders in split_folders.items():
        if not folders:
            continue
        
        print(f"Reorganizing {base_name} split folders...")
        folders.sort()  # Sort by number (1, 2, 3, etc.)
        
        target_dir = root / base_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge all split folders into target_dir
        for split_num, split_folder in folders:
            print(f"  Merging {split_folder.name} into {target_dir.name}...")
            for item in split_folder.iterdir():
                dest = target_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        # If it's a directory (e.g., class folder), merge contents
                        for subitem in item.iterdir():
                            subdest = dest / subitem.name
                            if subdest.exists():
                                subdest.unlink()  # Replace duplicate files
                            shutil.move(str(subitem), str(subdest))
                    else:
                        # If it's a file, replace it
                        dest.unlink()
                        shutil.move(str(item), str(dest))
                else:
                    shutil.move(str(item), str(dest))
        
        # Remove the now-empty split folders
        for _, split_folder in folders:
            shutil.rmtree(split_folder)
        
        print(f"✅ Merged into {target_dir.name}/")


def _count_images(split_dir: Path) -> Dict[str, int]:
    img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
    class_counts: Dict[str, int] = {}

    if not split_dir.exists() or not split_dir.is_dir():
        return class_counts

    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        count = sum(
            1
            for f in class_dir.rglob("*")
            if f.is_file() and f.suffix in img_ext
        )
        if count > 0:
            class_counts[class_dir.name] = count

    return class_counts


def find_dir_by_name(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists() and direct.is_dir():
        return direct

    for candidate in root.rglob(name):
        if candidate.is_dir():
            return candidate

    return None


def build_summary(data_root: Path) -> Dict[str, object]:
    train_dir = find_dir_by_name(data_root, "train")
    val_dir = find_dir_by_name(data_root, "val")

    train_counts = _count_images(train_dir) if train_dir else {}
    val_counts = _count_images(val_dir) if val_dir else {}

    train_classes = set(train_counts.keys())
    val_classes = set(val_counts.keys())

    labels_json = None
    labels_file = data_root / "Labels.json"
    if labels_file.exists():
        labels_json = str(labels_file)

    return {
        "data_root": str(data_root),
        "train_dir": str(train_dir) if train_dir else None,
        "val_dir": str(val_dir) if val_dir else None,
        "labels_json": labels_json,
        "train_classes": len(train_classes),
        "val_classes": len(val_classes),
        "train_images": sum(train_counts.values()),
        "val_images": sum(val_counts.values()),
        "missing_in_val": sorted(train_classes - val_classes),
        "missing_in_train": sorted(val_classes - train_classes),
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
    }


def main() -> None:
    args = parse_args()

    creds_path = setup_kaggle_credentials(args.kaggle_json)
    if creds_path.exists():
        print(f"Using kaggle credentials file: {creds_path}")
    else:
        print("Using kaggle credentials from environment variables.")

    # Check if dataset is already properly organized
    if is_dataset_ready(args.output_dir):
        print(f"✅ Dataset already properly organized in {args.output_dir}")
        print("Skipping download, extraction, and reorganization.")
    else:
        # Download if directory doesn't exist yet
        if not args.output_dir.exists():
            print(f"[1/3] Downloading dataset '{args.dataset}' into {args.output_dir} ...")
            print("Download progress is enabled." if not args.no_progress else "Download progress is disabled.")
            download_kaggle_dataset(args.dataset, args.output_dir, show_progress=not args.no_progress)
            print("Download completed.")

            if args.extract:
                print("[2/3] Extracting nested archives ...")
                extracted = extract_archives(args.output_dir)
                print(f"Extracted archives: {extracted}")
            else:
                print("[2/3] Extraction skipped (use --extract to enable).")
        else:
            print(f"Dataset directory exists at {args.output_dir}")
            print("Skipping download (already present).")
        
        # Always reorganize if dataset is not ready
        print("[2.5/3] Reorganizing dataset structure ...")
        reorganize_dataset(args.output_dir)

    print("[3/3] Building dataset summary ...")
    summary = build_summary(args.output_dir)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Summary:")
    print(f"  train classes: {summary['train_classes']}")
    print(f"  val classes:   {summary['val_classes']}")
    print(f"  train images:  {summary['train_images']}")
    print(f"  val images:    {summary['val_images']}")
    print(f"  summary file:  {args.summary_json}")


if __name__ == "__main__":
    main()
