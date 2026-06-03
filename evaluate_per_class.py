"""
Per-class accuracy evaluation across all 6 experiments.

Loads each model sequentially, collects per-class Top-1 accuracy on the
full validation set, and outputs:
  - Printed comparison table (stdout — captured in SLURM log)
  - figures/per_class_accuracy.csv
  - figures/per_class_accuracy.png  (bar chart, 20 hardest classes)

Usage:
    python evaluate_per_class.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.datasets.imagenet100 import build_dataloader, build_supervised_transform
from src.models.vit_classifier import build_vit_classifier
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.misc import get_device, load_dotenv, set_seed

load_dotenv()

EXPERIMENTS = [
    {
        "label": "Sup 100%",
        "config": "experiments/configs/supervised_vit_cluster.yaml",
        "checkpoint": "experiments/checkpoints/supervised_vit_baseline_200/checkpoint_best.pth",
    },
    {
        "label": "MAE+LP 100%",
        "config": "experiments/configs/linear_probe_cluster.yaml",
        "checkpoint": "experiments/checkpoints/linear_probe_200/checkpoint_best.pth",
    },
    {
        "label": "Sup 20%",
        "config": "experiments/configs/supervised_vit_small_dataset_20%_cluster.yaml",
        "checkpoint": "experiments/checkpoints/supervised_vit_small_dataset_20%/checkpoint_best.pth",
    },
    {
        "label": "MAE+LP 20%",
        "config": "experiments/configs/linear_probe_small_dataset_20%_cluster.yaml",
        "checkpoint": "experiments/checkpoints/linear_probe_small_dataset_20%/checkpoint_best.pth",
    },
    {
        "label": "Sup 10%",
        "config": "experiments/configs/supervised_vit_small_cluster.yaml",
        "checkpoint": "experiments/checkpoints/supervised_vit_small_dataset_10%/checkpoint_best.pth",
    },
    {
        "label": "MAE+LP 10%",
        "config": "experiments/configs/linear_probe_small_cluster.yaml",
        "checkpoint": "experiments/checkpoints/linear_probe_small_dataset_10%/checkpoint_best.pth",
    },
]


@torch.no_grad()
def predict_per_class(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
) -> dict[int, float]:
    """Run inference and return {class_idx: top1_accuracy_percent}."""
    model.eval()
    per_class: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]

    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs).argmax(dim=1).cpu()
        for pred, label in zip(preds, labels):
            c = label.item()
            per_class[c][1] += 1
            if pred.item() == c:
                per_class[c][0] += 1

    return {c: 100.0 * v[0] / v[1] for c, v in per_class.items()}


def load_labels(data_root: str) -> dict[str, str]:
    """Load Labels.json → {class_id: human_label}. Returns empty dict if not found."""
    labels_path = Path(data_root) / "Labels.json"
    if not labels_path.exists():
        print(f"[Warning] Labels.json not found at {labels_path}, using class IDs as names.")
        return {}
    with open(labels_path) as f:
        return json.load(f)


def build_idx_to_name(val_loader, labels_dict: dict[str, str]) -> dict[int, str]:
    """Map class index → short human-readable name (first synonym only)."""
    class_to_idx = val_loader.dataset.class_to_idx  # {n01440764: 0, ...}
    idx_to_classid = {v: k for k, v in class_to_idx.items()}
    return {
        idx: labels_dict.get(cid, cid).split(",")[0].strip()
        for idx, cid in idx_to_classid.items()
    }


def _print_table(df: pd.DataFrame, model_cols: list[str], title: str) -> None:
    col_w = max(12, max(len(c) for c in model_cols) + 2)
    name_w = 30
    header = f"  {'Class':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in model_cols)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        vals = "".join(f"{row[c]:>{col_w - 1}.0f}%" for c in model_cols)
        print(f"  {str(row['class_name']):<{name_w}}{vals}")
    print(sep)


def print_full_table(df: pd.DataFrame, model_cols: list[str]) -> None:
    _print_table(df, model_cols, "Per-Class Top-1 Accuracy — Tutti i Modelli")


def print_top_bottom(df: pd.DataFrame, model_cols: list[str], n: int = 10) -> None:
    df = df.copy()
    df["mean_acc"] = df[model_cols].mean(axis=1)
    _print_table(
        df.nsmallest(n, "mean_acc"),
        model_cols,
        f"Top-{n} Classi Peggiori (media tra tutti i modelli)",
    )
    _print_table(
        df.nlargest(n, "mean_acc"),
        model_cols,
        f"Top-{n} Classi Migliori (media tra tutti i modelli)",
    )


def save_bar_chart(
    df: pd.DataFrame,
    model_cols: list[str],
    output_path: Path,
    n: int = 20,
) -> None:
    """Bar chart showing per-model accuracy for the N hardest classes."""
    df = df.copy()
    df["mean_acc"] = df[model_cols].mean(axis=1)
    worst = df.nsmallest(n, "mean_acc").reset_index(drop=True)

    x = np.arange(n)
    bar_width = 0.13
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]

    fig, ax = plt.subplots(figsize=(18, 7))
    for i, (col, color) in enumerate(zip(model_cols, colors)):
        offset = (i - len(model_cols) / 2 + 0.5) * bar_width
        ax.bar(x + offset, worst[col], width=bar_width, label=col, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(worst["class_name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Top-{n} Classi Più Difficili — Confronto tra Modelli")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {output_path}")


def main() -> None:
    set_seed(42)
    device = get_device()

    # Build shared val loader from the first experiment's config
    first_cfg = load_config(EXPERIMENTS[0]["config"])
    val_loader = build_dataloader(
        root=first_cfg["data"]["root"],
        split="val",
        transform=build_supervised_transform(first_cfg["data"]["image_size"], is_train=False),
        batch_size=first_cfg["data"]["batch_size"],
        num_workers=first_cfg["data"]["num_workers"],
        pin_memory=first_cfg["data"].get("pin_memory", True),
        drop_last=False,
        shuffle=False,
    )

    labels_dict = load_labels(first_cfg["data"]["root"])
    idx_to_name = build_idx_to_name(val_loader, labels_dict)

    # Evaluate each model sequentially
    all_results: dict[str, dict[int, float]] = {}
    for exp in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"[Evaluating] {exp['label']}")
        print(f"{'='*50}")
        cfg = load_config(exp["config"])
        model = build_vit_classifier(cfg, pretrained_encoder_path=None, device=device).to(device)
        load_checkpoint(exp["checkpoint"], model, device=device)
        all_results[exp["label"]] = predict_per_class(model, val_loader, device)
        del model
        torch.cuda.empty_cache()

    # Build DataFrame: rows = classes, cols = models
    model_cols = [exp["label"] for exp in EXPERIMENTS]
    class_to_idx = val_loader.dataset.class_to_idx
    idx_to_classid = {v: k for k, v in class_to_idx.items()}

    rows = []
    for idx in sorted(idx_to_name):
        row = {
            "class_id": idx_to_classid[idx],
            "class_name": idx_to_name[idx],
        }
        for col in model_cols:
            row[col] = all_results[col].get(idx, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Print full table and top/bottom summaries
    print_full_table(df, model_cols)
    print_top_bottom(df, model_cols, n=10)

    # Save CSV
    csv_path = Path("experiments/results/per_class_accuracy.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Saved → {csv_path}")

    # Save bar chart (20 hardest classes)
    save_bar_chart(df, model_cols, Path("figures/per_class_accuracy.png"), n=20)

    print("\nDone.")


if __name__ == "__main__":
    main()
