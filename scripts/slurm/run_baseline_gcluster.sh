#!/bin/bash
#SBATCH --job-name=mae-baseline-vit
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --gres=shard:5000
#SBATCH --output=experiments/logs/job-%j.log

set -euo pipefail

REPO_DIR="$HOME/Masked-Autoencoders-for-Image-Representation-Learning"
SIF_IMAGE="/shared/sifs/latest.sif"
CONFIG_PATH="experiments/configs/baseline_supervised_cluster.yaml"

mkdir -p "$REPO_DIR/experiments/logs"
mkdir -p "$REPO_DIR/experiments/checkpoints"

cd "$REPO_DIR"

echo "Starting baseline training on gcluster..."

echo "Working directory: $(pwd)"
echo "Using config: $CONFIG_PATH"

apptainer run --nv "$SIF_IMAGE" bash -lc "
  cd '$REPO_DIR' && \
  python -m src.training.train --config '$CONFIG_PATH'
"

echo "Training completed."
