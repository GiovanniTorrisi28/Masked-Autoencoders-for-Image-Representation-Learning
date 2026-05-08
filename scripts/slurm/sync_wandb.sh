#!/bin/bash
# Sync W&B offline runs to the cloud.
# Run this from the LOGIN NODE (has internet) after a job completes.
#
# Usage: bash scripts/slurm/sync_wandb.sh

set -e

cd ~/Masked-Autoencoders-for-Image-Representation-Learning

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "${WANDB_API_KEY}" ]; then
    echo "ERROR: WANDB_API_KEY not found in .env"
    exit 1
fi

OFFLINE_RUNS=$(find experiments/logs/ -type d -name "offline-run-*" 2>/dev/null)

if [ -z "${OFFLINE_RUNS}" ]; then
    echo "No offline W&B runs found in wandb/"
    exit 0
fi

echo "Found offline W&B runs:"
echo "${OFFLINE_RUNS}"
echo ""

for run_dir in ${OFFLINE_RUNS}; do
    echo "Syncing ${run_dir} ..."
    apptainer exec /shared/sifs/latest.sif wandb sync "${run_dir}"
    echo "Done: ${run_dir}"
done

echo ""
echo "All runs synced. View at: https://wandb.ai"
