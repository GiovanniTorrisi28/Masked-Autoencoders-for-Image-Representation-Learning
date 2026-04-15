# Datasets

Scripts for downloading, parsing datasets, and PyTorch `DataLoader` / `Dataset` definitions.

## ImageNet100 Download Script

Script: `src/datasets/download_imagenet100.py`

Download + extract + summary:

```bash
python -m src.datasets.download_imagenet100 --extract
```

Custom output directory inside `data/`:

```bash
python -m src.datasets.download_imagenet100 --output-dir data/raw/imagenet100 --extract
```

The script creates a summary JSON in `data/processed/imagenet100_summary.json`.
