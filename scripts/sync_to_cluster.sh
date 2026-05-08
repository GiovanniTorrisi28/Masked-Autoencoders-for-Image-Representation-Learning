#!/bin/bash
# Upload the project to the cluster.
# Usage: bash scripts/sync_to_cluster.sh <username>
# Example: bash scripts/sync_to_cluster.sh gtorrisi

CLUSTER_USER="${1:?Usage: $0 <cluster_username>}"
CLUSTER_HOST="gcluster.dmi.unict.it"
REMOTE_DIR="~/Masked-Autoencoders-for-Image-Representation-Learning"

echo "Syncing to ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/ ..."

rsync -avz --progress \
  --exclude='data/' \
  --exclude='experiments/checkpoints/' \
  --exclude='experiments/logs/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='wandb/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  . "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/"

echo "Done."
