#!/bin/bash
# Download checkpoints, TensorBoard logs, and W&B offline runs from the cluster.
# Usage: bash scripts/sync_from_cluster.sh <username>
# Example: bash scripts/sync_from_cluster.sh gtorrisi

CLUSTER_USER="${1:?Usage: $0 <cluster_username>}"
CLUSTER_HOST="gcluster.dmi.unict.it"
REMOTE_DIR="~/Masked-Autoencoders-for-Image-Representation-Learning"

echo "Downloading experiments/ from cluster..."
rsync -avz --progress \
  "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/experiments/" \
  ./experiments/

echo "Downloading wandb/ offline runs from cluster..."
rsync -avz --progress \
  "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/wandb/" \
  ./wandb/

echo "Done. To sync W&B runs to the cloud, run:"
echo "  wandb sync wandb/offline-run-*/"
echo ""
echo "To view TensorBoard logs locally, run:"
echo "  tensorboard --logdir experiments/logs"
